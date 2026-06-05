# -----------------------------------------------------------------------
# CES-specific separation primitives.
#   - Standalone CES demand via conic programming
#   - Recovery of CES parameters (y, σ) from observed shares
#   - Separation subproblems (non-convex, convex surrogate, fixed-σ, dual LP,
#     inversion-multicut)
#   - Conversion (y, σ) → (c, ρ) and market expansion
# Required by separation.jl (per-class separation oracle) and used directly by
# notebooks for ad-hoc experiments.
# -----------------------------------------------------------------------

using LinearAlgebra, SparseArrays
using Optim
using LogExpFunctions: logsumexp
using ArgParse
using ExchangeMarket

# Homothetic: γ(p, w) = γ(p) — no wealth dependence in the CES share.
is_homothetic(::Val{:ces}) = true

# ---- CLI surface --------------------------------------------------------
# Defaults for the CES σ search range. CES is only defined for σ > -1
# (the Leontief limit); the LBFGS step uses a hardcoded strict-interior
# floor (-0.9) one step above that boundary because the downstream
# (y, σ) → (c, ρ) recovery (ρ = σ/(1+σ)) diverges as σ ↘ -1.
# These were previously file-level `const`s; they are now CLI knobs so
# σ-grid extents (for inversion) and the LBFGS upper bound (for the free-σ
# search) can be tuned per run without recompiling.
const _CES_SIGMA_LOWER_DEFAULT = -1.0
const _CES_SIGMA_UPPER_DEFAULT = 20.0

"""
    register_cli_ces!(s::ArgParseSettings)

Add the "Separation: CES" arg group: `--ces-sigma-lower`, `--ces-sigma-upper`.
These tune the σ search range for the free-σ LBFGS step
(`solve_separation_lbfgs_ces`) and the σ grid for per-sample inversion
(`solve_separation_inversion_ces`).
"""
function register_cli_ces!(s::ArgParseSettings)
    add_arg_group!(s, "Separation: CES")
    @add_arg_table! s begin
        "--ces-sigma-lower"
        help = "Lower bound on σ for both the LBFGS box and the inversion σ-grid. Default $(_CES_SIGMA_LOWER_DEFAULT); CES requires σ > -1, so the LBFGS box floors this at -0.98 (the (c, ρ) recovery diverges at -1). Raise (e.g. to 0) to restrict to gross substitutes."
        arg_type = Float64
        default = _CES_SIGMA_LOWER_DEFAULT
        "--ces-sigma-upper"
        help = "Upper bound on σ for both the LBFGS box and the inversion σ-grid. Default $(_CES_SIGMA_UPPER_DEFAULT); raise for near-linear (σ → ∞) regimes."
        arg_type = Float64
        default = _CES_SIGMA_UPPER_DEFAULT
    end
    return s
end

"""
    apply_cli_ces!(local_extra::Dict, cli)

Forward CES-separation σ-bound CLI values into the runner kwargs, which
solve_separation_class threads down to the CES oracles.
"""
function apply_cli_ces!(local_extra::Dict, cli)
    local_extra[:ces_sigma_lower] = cli["ces_sigma_lower"]
    local_extra[:ces_sigma_upper] = cli["ces_sigma_upper"]
    return local_extra
end

"""
    ces_config_summary(kwargs::Dict; is_multicut::Bool) -> String

Short one-line description of the CES class's separation solver, formatted
for cpm.jl's banner under the `ces` print_config entry. Owned here so the
solver knobs (σ box, separation regime) live next to the code that uses
them rather than being hardcoded in the driver.
"""
function ces_config_summary(kwargs::Dict; is_multicut::Bool)
    lo = get(kwargs, :ces_sigma_lower, _CES_SIGMA_LOWER_DEFAULT)
    hi = get(kwargs, :ces_sigma_upper, _CES_SIGMA_UPPER_DEFAULT)
    if is_multicut
        return @sprintf("Inversion (σ [%g, %g] × LBFGS refine)", lo, hi)
    else
        return @sprintf("LBFGS (σ ∈ [%g, %g])", lo, hi)
    end
end

# ---- Ground-truth CES market knobs --------------------------------------
"""
    register_cli_ces_market!(s::ArgParseSettings)

Add the "Market: CES" arg group — knobs for building the ground-truth
CES market when `--market-type ces`. Currently the per-agent ρ sampling
range, which used to be hardcoded as ρ ∈ [-3.5, 0.8) in run_test.jl.
"""
function register_cli_ces_market!(s::ArgParseSettings)
    add_arg_group!(s, "Market: CES")
    @add_arg_table! s begin
        "--ces-rho-low"
        help = "Lower bound for per-agent ρ sampling when --market-type ces. CES is defined for ρ ∈ (-∞, 1); large-negative ρ → Leontief, ρ → 1 → linear. Default -3.5 (matches prior hardcoded range)."
        arg_type = Float64
        default = -3.5
        "--ces-rho-high"
        help = "Upper bound for per-agent ρ sampling (exclusive). Default 0.8 (matches prior hardcoded range; > 0.95 may produce near-linear preferences that hurt CG convergence)."
        arg_type = Float64
        default = 0.9
    end
    return s
end

"""
    ces_rho_range_from_cli(cli) -> (lo::Float64, hi::Float64)

Bundle the parsed CES market-build flags. Asserts `lo < hi`.
"""
function ces_rho_range_from_cli(cli)
    lo, hi = cli["ces_rho_low"], cli["ces_rho_high"]
    @assert lo < hi "--ces-rho-low ($lo) must be strictly less than --ces-rho-high ($hi)"
    return (lo, hi)
end

# `y` (log-coefficient) box for the LBFGS step in
# `solve_separation_lbfgs_ces`. Kept as `const` because the CES gauge
# `max(y) = 0` is enforced by the LBFGS-projected solver itself, and
# this generous box only serves as a sanity cap.
const LOWER_Y_BOUND = -100.0
const UPPER_Y_BOUND = 100.0

# -----------------------------------------------------------------------
# Standalone CES demand via conic programming
# -----------------------------------------------------------------------
"""
    _conic_ces_primal(; p, n, cr, w, ρ, verbose=false)

Solve CES utility maximization via linear-conic programming (Mosek).
`cr` = c .^ (1/ρ). Returns optimal allocation x.
"""
function _conic_ces_primal(;
    p::Vector{T}=nothing,
    n,
    cr,
    w,
    ρ,
    verbose=false
) where {T}
    md = ExchangeMarket.__generate_empty_jump_model(; verbose=verbose, tol=1e-8)
    @variable(md, u)
    @variable(md, logu)
    log_to_expcone!(u, logu, md)

    @variable(md, x[1:n] >= 0)
    @variable(md, ξ[1:n] >= 0)
    @constraint(md, budget, p' * x <= w)
    @constraint(md, sum(ξ) == u)
    @constraint(
        md,
        ξc[j=1:n],
        [cr[j] * x[j], u, ξ[j]] in MOI.PowerCone(ρ)
    )
    @objective(md, Max, logu)

    JuMP.optimize!(md)
    x = max.(value.(x), 0.0)
    return x
end

# -----------------------------------------------------------------------
# Recover CES parameters (y, δ) from bidding vectors via LP
# -----------------------------------------------------------------------
@doc raw"""
    _linear_prog_ces_gamma_single(; pmat, gmat, δ₁=nothing, verbose=false)

Given a CES bidding vector γ, recover the CES coefficients (y, δ)
from linear-programming optimization. If there are more than one
bidding vector, the fit may not be tight.
"""
function _linear_prog_ces_gamma_single(;
    pmat::Union{SparseMatrixCSC{T},Matrix{T}}=nothing,
    gmat::Union{SparseMatrixCSC{T},Matrix{T}}=nothing,
    δ₁::Union{Float64,Nothing}=nothing,
    verbose=false
) where {T}
    md = ExchangeMarket.__generate_empty_jump_model(; verbose=verbose, tol=1e-8)

    n, K = size(pmat)
    @variable(md, y[1:n])
    @variable(md, A[1:K])
    @variable(md, r[1:n, 1:K])
    @variable(md, rmax >= 0)
    @variable(md, δ >= -1)

    # Gauge fix: y is identifiable only up to an additive constant (the
    # softmax shift), so pin one coordinate without bounding the spread.
    @constraint(md, y[1] == 0)

    if !isnothing(δ₁)
        @constraint(md, δ == δ₁)
    end
    @constraint(md,
        fitc[k=1:K],
        r[:, k] .+ y .- δ .* log.(pmat[:, k]) .- A[k] .== log.(gmat[:, k])
    )
    @constraint(md, rmax .>= r)
    @constraint(md, rmax .>= -r)

    @objective(md, Min, rmax)

    JuMP.optimize!(md)
    return value.(y), value.(δ), value.(A), md
end

"""
    produce_gamma(Ξ, y, σ)

Compute the K×n bidding matrix γ from CES parameters (y, σ) at prices in Ξ.
    γ[k, :] = softmax(y - σ log(p_k))
"""
function produce_gamma(Ξ, y::AbstractVector, σ::Real)
    K = length(Ξ)
    n = length(y)
    γ = zeros(eltype(y), K, n)
    for k in 1:K
        z_k = y .- σ .* log.(Ξ[k][1])
        γ[k, :] = exp.(z_k .- logsumexp(z_k))
    end
    return γ
end

"""
    recover_ces_params(y, σ)

Recover CES parameters (c, ρ) from the log-reparameterization (y, σ).
    y = log(c^{1+σ})  =>  c = exp(y / (1+σ))
    σ = r/(1-r)       =>  r = σ/(1+σ)
"""
function recover_ces_params(y::Vector{T}, σ::T) where T
    c = exp.(y ./ (1 + σ))
    ρ = σ / (1 + σ)
    return c, ρ
end

"""
    add_to_market!(f1::FisherMarket, c_new, ρ_new, w_new)

Add a new CES agent to an existing FisherMarket in-place.
- c_new: coefficient vector (n-dim)
- ρ_new: CES parameter ρ
- w_new: budget for the new agent

Uses expand_players! from ExchangeMarket.
"""
function add_to_market!(f1::FisherMarket, c_new::Vector{T}, ρ_new::T, w_new::T) where T
    expand_players!(f1, f1.m + 1;
        c_new=reshape(c_new, :, 1),
        ρ_new=[ρ_new],
        w_new=[w_new]
    )
    return f1
end

#=
Two formulations for the CES separation problem:

1. ORIGINAL (non-convex):
   max_{y, σ > 0} Σ_k u_k^T γ_k = Σ_k u_k^T softmax(y - σ log p_k)

2. CONVEX SURROGATE:
   max_{y, σ > 0} Σ_k u_k^T log γ_k
   = Σ_k [u_k^T (y - σ log p_k) - (1^T u_k) · lse(y - σ log p_k)]
   # @note, this will not be implemented.

The convex surrogate replaces γ with log(γ), making it concave in (y, σ).
=#

"""
    solve_separation_lbfgs_ces(Ξ, u; y_init=nothing, σ_init=0.5, verbose=false)

Solve the ORIGINAL separation problem (non-convex):
    max_{y ∈ ℝ^n, σ > -1} Σ_k u_k^T γ_k = Σ_k u_k^T softmax(y - σ log p_k)

Returns: y, σ, γ_new, obj_val
"""
function solve_separation_lbfgs_ces(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    y_init::Union{Vector{T},Nothing}=nothing,
    σ_init::T=0.5,
    # σ box for LBFGS; same values the inversion σ-grid uses.
    ces_sigma_lower::Real=_CES_SIGMA_LOWER_DEFAULT,
    ces_sigma_upper::Real=_CES_SIGMA_UPPER_DEFAULT,
    timelimit::Union{Real,Nothing}=nothing,
    verbose=false,
    kwargs...) where T

    # Lower bound for the σ box: the user's --ces-sigma-lower, floored at
    # the strict-interior value -0.98. The softmax objective itself is fine
    # at σ = -1, but the (y, σ) → (c, ρ) recovery used downstream by
    # `add_column_to_market!` (ρ = σ/(1+σ), c = exp(y/(1+σ))) diverges
    # there, so the box must stay one step above the boundary regardless
    # of the CLI value. Raising the bound (e.g. --ces-sigma-lower 0 for
    # gross substitutes only) is honored as-is.
    strict_sigma_lower = max(T(ces_sigma_lower), T(-0.98))

    K = length(Ξ)
    n = length(Ξ[1][1])

    function neg_objective(x)
        y = x[1:n]
        σ = x[n+1]
        val = zero(T)
        for k in 1:K
            p_k = Ξ[k][1]
            u_k = u[k, :]
            z_k = y .- σ .* log.(p_k)
            γ_k = exp.(z_k .- logsumexp(z_k))
            val += dot(u_k, γ_k)
        end
        return -val
    end

    function neg_gradient!(G, x)
        y = x[1:n]
        σ = x[n+1]
        G .= 0.0
        for k in 1:K
            p_k = Ξ[k][1]
            u_k = u[k, :]
            log_p_k = log.(p_k)
            z_k = y .- σ .* log_p_k
            γ_k = exp.(z_k .- logsumexp(z_k))
            coef = γ_k .* u_k .- γ_k .* dot(γ_k, u_k)
            G[1:n] .-= coef
            G[n+1] -= -dot(coef, log_p_k)
        end
    end

    lower = vcat(fill(T(LOWER_Y_BOUND), n), strict_sigma_lower)
    upper = vcat(fill(T(UPPER_Y_BOUND), n), T(ces_sigma_upper))
    # Strict-interior clamps (Fminbox warns and nudges if x0 sits on a bound).
    ϵ = T(1e-6)
    y0 = isnothing(y_init) ? zeros(T, n) :
         clamp.(y_init, T(LOWER_Y_BOUND) + ϵ, T(UPPER_Y_BOUND) - ϵ)
    σ0 = clamp(σ_init, strict_sigma_lower + ϵ, T(ces_sigma_upper) - ϵ)
    x0 = vcat(y0, σ0)

    _tlim_opts = isnothing(timelimit) || timelimit <= 0 ? NamedTuple() :
                 (time_limit=Float64(timelimit),)
    result = optimize(
        neg_objective, neg_gradient!,
        lower, upper, x0,
        Fminbox(LBFGS(; m=15)),
        Optim.Options(; show_trace=verbose, iterations=1000, g_tol=1e-8, _tlim_opts...)
    )

    x_opt = Optim.minimizer(result)
    y_opt, σ_opt = x_opt[1:n], x_opt[n+1]
    obj_val = -Optim.minimum(result)

    γ_new = produce_gamma(Ξ, y_opt, σ_opt)

    verbose && println("CES separation: σ=$σ_opt, obj=$obj_val")
    return y_opt, σ_opt, γ_new, obj_val
end

"""
    solve_separation_fix_σ_ces(Ξ, u, σ; y_init=nothing, verbose=false)

Solve the separation problem with fixed σ:
    max_{y ∈ ℝ^n} Σ_k u_k^T softmax(y - σ log p_k)

With σ fixed this is concave in y (softmax is log-concave, and u_k ≥ 0),
so any local maximum is global.

Returns: y, σ, γ_new, obj_val
"""
function solve_separation_fix_σ_ces(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T},
    σ::T;
    y_init::Union{Vector{T},Nothing}=nothing,
    timelimit::Union{Real,Nothing}=nothing,
    verbose=false) where T

    K = length(Ξ)
    n = length(Ξ[1][1])

    log_p = [log.(Ξ[k][1]) for k in 1:K]

    function neg_objective(y)
        val = zero(T)
        for k in 1:K
            z_k = y .- σ .* log_p[k]
            γ_k = exp.(z_k .- logsumexp(z_k))
            val += dot(u[k, :], γ_k)
        end
        return -val
    end

    function neg_gradient!(G, y)
        G .= 0.0
        for k in 1:K
            z_k = y .- σ .* log_p[k]
            γ_k = exp.(z_k .- logsumexp(z_k))
            u_k = u[k, :]
            G .-= γ_k .* (u_k .- dot(γ_k, u_k))
        end
    end

    y0 = isnothing(y_init) ? zeros(T, n) : copy(y_init)

    # Unconstrained LBFGS. The objective is concave in y but only up to
    # the softmax shift, so the optimum is a level set rather than a point.
    # LBFGS starting at y0 converges in objective value; the gauge is fixed
    # downstream by add_column_to_market!.
    _tlim_opts = isnothing(timelimit) || timelimit <= 0 ? NamedTuple() :
                 (time_limit=Float64(timelimit),)
    result = optimize(
        neg_objective, neg_gradient!,
        y0,
        LBFGS(),
        Optim.Options(;
            show_trace=verbose, show_every=50, iterations=1000,
            g_tol=1e-8, _tlim_opts...
        )
    )

    y_opt = Optim.minimizer(result)
    obj_val = -Optim.minimum(result)

    γ_new = produce_gamma(Ξ, y_opt, σ)

    verbose && println("Fixed-σ separation: σ=$σ, obj=$obj_val")
    return y_opt, σ, γ_new, obj_val
end

"""
    solve_separation_dual_lp_ces(Ξ, u; δ₁=nothing, verbose=false)

Separation via direct dual normalization + LP matching.
Normalizes each dual vector u_k to the simplex to obtain a bidding vector,
then calls `_linear_prog_ces_gamma_single` to recover (y, δ).
"""
function solve_separation_dual_lp_ces(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    δ₁::Union{Float64,Nothing}=nothing,
    verbose=false) where T

    K = length(Ξ)

    pmat = hcat([Ξ[k][1] for k in 1:K]...)
    gmat = similar(pmat)
    for k in 1:K
        u_k = max.(u[k, :], eps(T))
        gmat[:, k] = u_k ./ sum(u_k)
    end

    y_opt, δ_opt, _, _ = _linear_prog_ces_gamma_single(;
        pmat=pmat, gmat=gmat, δ₁=δ₁, verbose=verbose
    )

    γ_new = produce_gamma(Ξ, y_opt, δ_opt)

    obj_val = sum(dot(u[k, :], γ_new[k, :]) for k in 1:K)
    verbose && println("Dual-LP separation: δ=$δ_opt, obj=$obj_val")
    return y_opt, δ_opt, γ_new, obj_val
end

# -----------------------------------------------------------------------
# Multicut separation via inversion + 1D line search
# -----------------------------------------------------------------------
"""
    solve_separation_inversion_ces(Ξ, u; σ_grid=range(-1.0, 30.0, length=50))

Multicut separation via single-k inversion + 1D line search over σ.

For each observation k = 1,...,K:
1. Shift u_k to be positive and normalize to the simplex.
2. Invert the softmax log-ratio equations to obtain y_j(σ) = L_j + σ ℓ_j.
3. Search over σ (grid + Brent refinement) to maximize the separation objective.

Returns all K candidate columns as a vector of (y, σ, γ, obj).
"""
function solve_separation_inversion_ces(
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    # σ-grid endpoints (50 grid points). Defaults reproduce the previous
    # `const` extents.
    ces_sigma_lower::Real=_CES_SIGMA_LOWER_DEFAULT,
    ces_sigma_upper::Real=_CES_SIGMA_UPPER_DEFAULT,
    σ_grid=range(ces_sigma_lower, ces_sigma_upper, length=50),
    kwargs...
) where T

    K = length(Ξ)
    n = length(Ξ[1][1])

    log_p = [log.(Ξ[k][1]) for k in 1:K]
    results = Vector{Tuple{Vector{T},T,Matrix{T},T}}()

    for k in 1:K
        u_k = u[k, :] .- minimum(u[k, :]) .+ 1e-8
        u_k = u_k ./ sum(u_k)

        L = log.(u_k[2:end] ./ u_k[1])
        ℓ_k = log_p[k][2:end] .- log_p[k][1]

        function separation_obj(σ)
            y = zeros(T, n)
            for j in 2:n
                y[j] = L[j-1] + σ * ℓ_k[j-1]
            end
            val = zero(T)
            for k2 in 1:K
                k2 == k && continue
                z = y .- σ .* log_p[k2]
                γ = exp.(z .- logsumexp(z))
                val += dot(u[k2, :], γ)
            end
            return val
        end

        best_σ = zero(T)
        best_obj = separation_obj(best_σ)
        for σ_try in σ_grid
            obj = separation_obj(σ_try)
            if obj > best_obj
                best_obj = obj
                best_σ = σ_try
            end
        end

        idx = findfirst(s -> s == best_σ, collect(σ_grid))
        if !isnothing(idx)
            lo = idx > 1 ? σ_grid[idx-1] : σ_grid[1]
            hi = idx < length(σ_grid) ? σ_grid[idx+1] : σ_grid[end]
            res = optimize(σ -> -separation_obj(σ), lo, hi, Brent())
            best_σ = Optim.minimizer(res)
            best_obj = -Optim.minimum(res)
        end

        y_opt = zeros(T, n)
        for j in 2:n
            y_opt[j] = L[j-1] + best_σ * ℓ_k[j-1]
        end
        γ_new = produce_gamma(Ξ, y_opt, best_σ)
        push!(results, (y_opt, best_σ, γ_new, best_obj))
    end
    return results
end

# -----------------------------------------------------------------------
# Wrapper for per-class separation oracle
# -----------------------------------------------------------------------
"""
    solve_separation_ces(Ξ, u; verbose=false, kwargs...)

Run the LP-warmstart + LBFGS CES separation pipeline and return a NamedTuple
suitable for the per-class separation oracle:
    (γ_new, params=(y=y_opt, σ=σ_opt), obj=obj, class=:ces).

Same procedure as the previous cg_single path in run_method_tracked:
1. Warm-start (y, σ) via the dual-LP fit (solve_separation_dual_lp_ces).
2. Refine via the non-convex LBFGS objective (solve_separation_lbfgs_ces).
"""
function solve_separation_ces(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    verbose::Bool=false,
    timelimit::Union{Real,Nothing}=nothing,
    kwargs...) where T
    y_lp, σ_lp, _, _ = solve_separation_dual_lp_ces(
        Ξ, u; verbose=verbose
    )
    # Thread the CES σ-bound kwargs (ces_sigma_lower, ces_sigma_upper)
    # through to the LBFGS box. Other kwargs (e.g., from other classes)
    # are silently absorbed by `solve_separation_lbfgs_ces`'s `kwargs...`.
    y_opt, σ_opt, γ_new, obj = solve_separation_lbfgs_ces(
        Ξ, u;
        y_init=y_lp, σ_init=σ_lp, timelimit=timelimit, verbose=verbose,
        kwargs...
    )
    return (γ_new=γ_new, params=(y=y_opt, σ=σ_opt), obj=obj, class=:ces)
end
