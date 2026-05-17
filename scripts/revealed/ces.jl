# -----------------------------------------------------------------------
# CES-specific pricing primitives.
#   - Standalone CES demand via conic programming
#   - Recovery of CES parameters (y, σ) from observed shares
#   - Pricing subproblems (non-convex, convex surrogate, fixed-σ, dual LP,
#     inversion-multicut)
#   - Conversion (y, σ) → (c, ρ) and market expansion
# Required by pricing.jl (multi-class dispatcher) and used directly by
# notebooks for ad-hoc experiments.
# -----------------------------------------------------------------------

using LinearAlgebra, SparseArrays
using Optim
using LogExpFunctions: logsumexp
using ExchangeMarket

# Conceptual CES parameter ranges, used as the source of truth for the
# Fminbox / LBFGS bounds in `solve_pricing` and the σ grid in
# `solve_pricing_inversion`. CES is only defined for σ > -1, but `1/(1+σ)`
# explodes as σ ↘ -1, so the LBFGS step uses a strict-interior floor
# `STRICT_SIGMA_LOWER` (= σ_leontief default in leontief.jl).
const LOWER_SIGMA_BOUND  = -1.0
const UPPER_SIGMA_BOUND  = 30.0
const LOWER_Y_BOUND      = -100.0
const UPPER_Y_BOUND      = 100.0
const STRICT_SIGMA_LOWER = -0.9

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
Two formulations for the CES pricing problem:

1. ORIGINAL (non-convex):
   max_{y, σ > 0} Σ_k u_k^T γ_k = Σ_k u_k^T softmax(y - σ log p_k)

2. CONVEX SURROGATE:
   max_{y, σ > 0} Σ_k u_k^T log γ_k
   = Σ_k [u_k^T (y - σ log p_k) - (1^T u_k) · lse(y - σ log p_k)]

The convex surrogate replaces γ with log(γ), making it concave in (y, σ).
=#

"""
    solve_pricing(Ξ, u; y_init=nothing, σ_init=0.5, verbose=false)

Solve the ORIGINAL pricing problem (non-convex):
    max_{y ∈ ℝ^n, σ > -1} Σ_k u_k^T γ_k = Σ_k u_k^T softmax(y - σ log p_k)

Returns: y, σ, γ_new, obj_val
"""
function solve_pricing(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    y_init::Union{Vector{T},Nothing}=nothing,
    σ_init::T=0.5,
    timelimit::Union{Real,Nothing}=nothing,
    verbose=false) where T

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

    lower = vcat(fill(T(LOWER_Y_BOUND), n), T(STRICT_SIGMA_LOWER))
    upper = vcat(fill(T(UPPER_Y_BOUND), n), T(UPPER_SIGMA_BOUND))
    # Strict-interior clamps (Fminbox warns and nudges if x0 sits on a bound).
    ϵ = T(1e-6)
    y0 = isnothing(y_init) ? zeros(T, n) :
         clamp.(y_init, T(LOWER_Y_BOUND) + ϵ, T(UPPER_Y_BOUND) - ϵ)
    σ0 = clamp(σ_init, T(STRICT_SIGMA_LOWER) + ϵ, T(UPPER_SIGMA_BOUND) - ϵ)
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

    verbose && println("CES pricing: σ=$σ_opt, obj=$obj_val")
    return y_opt, σ_opt, γ_new, obj_val
end

"""
    solve_pricing_convex(Ξ, u; σ_init=0.5, verbose=false)

Solve the CONVEX SURROGATE pricing problem:
    max_{y ∈ ℝ^n, σ > -1} Σ_k u_k^T log γ_k

This is concave in (y, σ), guaranteeing global optimum.
Returns: y, σ, γ_new, obj_val
"""
function solve_pricing_convex(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    σ_init::T=0.5,
    verbose=false) where T

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
            lse_k = logsumexp(z_k)
            sum_u_k = sum(u_k)
            val += dot(u_k, z_k) - sum_u_k * lse_k
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
            softmax_k = exp.(z_k .- logsumexp(z_k))
            sum_u_k = sum(u_k)
            G[1:n] .-= (u_k .- sum_u_k .* softmax_k)
            G[n+1] -= -dot(u_k, log_p_k) + sum_u_k * dot(softmax_k, log_p_k)
        end
    end

    lower = vcat(fill(T(-Inf), n), T(-0.95))
    upper = vcat(fill(T(Inf),  n), T(100.0))
    # Strict-interior clamp on σ so Fminbox doesn't warn / nudge it.
    ϵ = T(1e-6)
    y_init = zeros(T, n)
    σ0 = clamp(σ_init, T(-0.95) + ϵ, T(100.0) - ϵ)
    x0 = vcat(y_init, σ0)

    result = optimize(
        neg_objective, neg_gradient!,
        lower, upper, x0,
        Fminbox(LBFGS()),
        Optim.Options(show_trace=verbose, iterations=1000, g_tol=1e-8)
    )

    x_opt = Optim.minimizer(result)
    y_opt, σ_opt = x_opt[1:n], x_opt[n+1]
    obj_val = -Optim.minimum(result)

    γ_new = produce_gamma(Ξ, y_opt, σ_opt)

    verbose && println("Convex CES pricing: σ=$σ_opt, obj=$obj_val")
    return y_opt, σ_opt, γ_new, obj_val
end

"""
    solve_pricing_fix_σ(Ξ, u, σ; y_init=nothing, verbose=false)

Solve the pricing problem with fixed σ:
    max_{y ∈ ℝ^n} Σ_k u_k^T softmax(y - σ log p_k)

With σ fixed this is concave in y (softmax is log-concave, and u_k ≥ 0),
so any local maximum is global.

Returns: y, σ, γ_new, obj_val
"""
function solve_pricing_fix_σ(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
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
        Optim.Options(; show_trace=verbose, show_every=50, iterations=1000, g_tol=1e-8, _tlim_opts...)
    )

    y_opt = Optim.minimizer(result)
    obj_val = -Optim.minimum(result)

    γ_new = produce_gamma(Ξ, y_opt, σ)

    verbose && println("Fixed-σ pricing: σ=$σ, obj=$obj_val")
    return y_opt, σ, γ_new, obj_val
end

"""
    solve_pricing_dual_lp(Ξ, u; δ₁=nothing, verbose=false)

Pricing via direct dual normalization + LP matching.
Normalizes each dual vector u_k to the simplex to obtain a bidding vector,
then calls `_linear_prog_ces_gamma_single` to recover (y, δ).
"""
function solve_pricing_dual_lp(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
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
    verbose && println("Dual-LP pricing: δ=$δ_opt, obj=$obj_val")
    return y_opt, δ_opt, γ_new, obj_val
end

# -----------------------------------------------------------------------
# Multicut pricing via inversion + 1D line search
# -----------------------------------------------------------------------
"""
    solve_pricing_inversion(Ξ, u; σ_grid=range(-1.0, 30.0, length=50))

Multicut pricing via single-k inversion + 1D line search over σ.

For each observation k = 1,...,K:
1. Shift u_k to be positive and normalize to the simplex.
2. Invert the softmax log-ratio equations to obtain y_j(σ) = L_j + σ ℓ_j.
3. Search over σ (grid + Brent refinement) to maximize the pricing objective.

Returns all K candidate columns as a vector of (y, σ, γ, obj).
"""
function solve_pricing_inversion(
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    σ_grid=range(LOWER_SIGMA_BOUND, UPPER_SIGMA_BOUND, length=50)
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

        function pricing_obj(σ)
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
        best_obj = pricing_obj(best_σ)
        for σ_try in σ_grid
            obj = pricing_obj(σ_try)
            if obj > best_obj
                best_obj = obj
                best_σ = σ_try
            end
        end

        idx = findfirst(s -> s == best_σ, collect(σ_grid))
        if !isnothing(idx)
            lo = idx > 1 ? σ_grid[idx-1] : σ_grid[1]
            hi = idx < length(σ_grid) ? σ_grid[idx+1] : σ_grid[end]
            res = optimize(σ -> -pricing_obj(σ), lo, hi, Brent())
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
# Wrapper for multi-class dispatcher
# -----------------------------------------------------------------------
"""
    solve_pricing_ces(Ξ, u; verbose=false, kwargs...)

Run the LP-warmstart + LBFGS CES pricing pipeline and return a NamedTuple
suitable for the multi-class dispatcher:
    (γ_new, params=(y=y_opt, σ=σ_opt), obj=obj, class=:ces).

Same procedure as the previous cg_single path in run_method_tracked:
1. Warm-start (y, σ) via the dual-LP fit (solve_pricing_dual_lp).
2. Refine via the non-convex LBFGS objective (solve_pricing).
"""
function solve_pricing_ces(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    verbose::Bool=false,
    timelimit::Union{Real,Nothing}=nothing,
    kwargs...) where T
    y_lp, σ_lp, _, _ = solve_pricing_dual_lp(Ξ, u; verbose=verbose)
    y_opt, σ_opt, γ_new, obj = solve_pricing(Ξ, u;
        y_init=y_lp, σ_init=σ_lp, timelimit=timelimit, verbose=verbose)
    return (γ_new=γ_new, params=(y=y_opt, σ=σ_opt), obj=obj, class=:ces)
end
