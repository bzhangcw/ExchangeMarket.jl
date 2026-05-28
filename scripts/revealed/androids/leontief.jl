# -----------------------------------------------------------------------
# Leontief market utilities for revealed-preference experiments.
#   - Random Leontief agent generation
#   - Closed-form demand
#   - Active-set diagnostics for PLC -> Leontief comparison
# -----------------------------------------------------------------------

using LinearAlgebra, Random
using Optim
using LogExpFunctions: logsumexp
using ExchangeMarket

# Homothetic: Leontief share γ_j(p) = (p_j/a_j) / Σ_k(p_k/a_k) depends on `p` alone.
is_homothetic(::Val{:leontief}) = true

"""
    leontief_config_summary(kwargs::Dict) -> String

Short one-line description of the Leontief class's separation solver, for
the banner under the `leontief` print_config entry. Owned here next to the
oracle implementation rather than hardcoded in driver code.
"""
function leontief_config_summary(kwargs::Dict)
    return "LBFGS"  # no per-class CLI knobs yet; signature parity with other *_config_summary
end

"""
    random_leontief_agent(n; seed=nothing)

Generate a random LeontiefAgent with weights `a ~ Uniform(0, 1)^n`.
"""
function random_leontief_agent(n::Int; seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    a = rand(n)
    return LeontiefAgent(a)
end

"""
    solve_leontief_demand(agent::LeontiefAgent, p, w)

Closed-form Leontief demand: at the kink aⱼ xⱼ = t for all j with budget p'x = w,
  xⱼ = w / (aⱼ Σₖ pₖ/aₖ),    u = w / Σₖ pₖ/aₖ.

Returns (x, u).
"""
function solve_leontief_demand(agent::LeontiefAgent, p::AbstractVector, w::Real)
    a = agent.a
    s = sum(p[j] / a[j] for j in eachindex(a))
    x = [w / (a[j] * s) for j in eachindex(a)]
    u = w / s
    return x, u
end

"""
    leontief_share(a, p) -> γ (length n)

Closed-form Leontief spending share γ_j(p) = (p_j/a_j) / Σ_k (p_k/a_k).
Homothetic — no wealth dependence. Equivalent to softmax(y + log p) with
y_j = -log a_j (cf. fact.demand.leontief in the paper).
"""
function leontief_share(a::AbstractVector{T}, p::AbstractVector) where T
    n = length(a)
    @assert length(p) == n "leontief_share: length(a) ≠ length(p)"
    ratios = p ./ a
    return ratios ./ sum(ratios)
end

"""
    share(agent::LeontiefAgent, p, w) -> γ (length n)

Per-class share dispatch for GenStore-backed evaluate_test_error. Wealth
is ignored (Leontief is homothetic).
"""
share(agent::LeontiefAgent, p::AbstractVector, w::Real) =
    leontief_share(agent.a, p)


# -----------------------------------------------------------------------
# Leontief separation — exact σ → -1⁺ limit (independent of the CES path)
# -----------------------------------------------------------------------
"""
    solve_separation_leontief(Ξ, u; y_init=nothing, verbose=false, kwargs...)

Separation subproblem for the Leontief function class, solved directly in
its native parameterization (no CES (c, ρ) detour). At fact.demand.leontief
the Leontief share is

    γ_j(p) = (p_j / a_j) / Σ_k (p_k / a_k) = softmax(y + log p)_j

with the gauge y_j := -log a_j. Maximizing the separation objective

    max_{y ∈ ℝ^n}  Σ_k ⟨u_k, softmax(y + log p_k)⟩

via Fminbox LBFGS gives the optimal y; we report

    params = (a = exp(-y), )

so the runner can store the column directly as a LeontiefAgent in the
GenStore (`add_column_to_market!` routes `:leontief` to `add_gen!`). No
σ is returned — the σ → -1 limit is exact, not a CES approximation.

Returns a NamedTuple compatible with the per-class separation oracle:
    (γ_new::Matrix{T}, params=(a=...,), obj::T, class=:leontief).
"""
function solve_separation_leontief(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    y_init::Union{Vector{T},Nothing}=nothing,
    verbose::Bool=false,
    timelimit::Union{Real,Nothing}=nothing,
    kwargs...) where T
    K = length(Ξ)
    n = length(Ξ[1][1])
    @assert size(u) == (K, n) "u must have shape (K, n)"

    log_p = [log.(Ξ[k][1]) for k in 1:K]

    function neg_objective(y::AbstractVector)
        val = zero(T)
        @inbounds for k in 1:K
            z_k = y .+ log_p[k]              # σ = -1 ⇒ -σ·log p = +log p
            γ_k = exp.(z_k .- logsumexp(z_k))
            for j in 1:n
                val += u[k, j] * γ_k[j]
            end
        end
        return -val
    end

    function neg_gradient!(G::AbstractVector, y::AbstractVector)
        G .= 0
        @inbounds for k in 1:K
            z_k = y .+ log_p[k]
            γ_k = exp.(z_k .- logsumexp(z_k))
            ug = dot(view(u, k, :), γ_k)
            for j in 1:n
                G[j] -= γ_k[j] * (u[k, j] - ug)
            end
        end
        return G
    end

    # Generous box — the gauge `max(y) = 0` is enforced after the solve.
    lower = fill(T(-100), n)
    upper = fill(T(100), n)
    y0 = isnothing(y_init) ? zeros(T, n) :
         clamp.(y_init, T(-99.99), T(99.99))

    _tlim_opts = isnothing(timelimit) || timelimit <= 0 ? NamedTuple() :
                 (time_limit=Float64(timelimit),)
    result = optimize(
        neg_objective, neg_gradient!,
        lower, upper, y0,
        Fminbox(LBFGS(; m=15)),
        Optim.Options(; show_trace=verbose, iterations=1000, g_tol=1e-8, _tlim_opts...)
    )

    y_opt = Optim.minimizer(result)
    obj_val = -Optim.minimum(result)

    # Gauge-fix max(y) = 0 ⇒ min(a) = 1 (canonical Leontief representative).
    y_shifted = y_opt .- maximum(y_opt)
    a_opt = exp.(-y_shifted)

    γ_new = Matrix{T}(undef, K, n)
    for k in 1:K
        γ_new[k, :] .= leontief_share(a_opt, Ξ[k][1])
    end

    verbose && println("Leontief separation (exact σ=-1): obj=$obj_val")
    return (γ_new=γ_new, params=(a=a_opt,), obj=obj_val, class=:leontief)
end

# -----------------------------------------------------------------------
# Per-sample Leontief inversion (the Leontief analogue of
# solve_separation_inversion_ces / _linear).
# -----------------------------------------------------------------------
"""
    solve_separation_inversion_leontief(Ξ, u) -> Vector{(a::Vector, γ_new::Matrix, obj::T)}

Per-sample inversion for the Leontief class. Leontief has no free σ to
search over (the σ → -1 limit is exact), so this is direct: for each
sample `k`, read the target share off the dual `u[k, :]` (after the
nonneg / simplex shift, mirroring `solve_separation_inversion_ces`),
and invert via fact.demand.leontief:

    γ_j(p_k; a) = (p_kj / a_j) / Σ_l (p_kl / a_l) = t_kj
        ⇒  a_j = p_kj / t_kj          (up to a multiplicative gauge),

normalized so `min(a) = 1` (matches the `max(y) = 0` convention in
`solve_separation_leontief`). Evaluate the resulting Leontief atom at
every sample to fill out the K × n bidding matrix and report the
separation objective Σ_{k'} ⟨u_{k'}, γ_{k'}⟩.
"""
function solve_separation_inversion_leontief(
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T}
) where T
    K = length(Ξ)
    n = length(Ξ[1][1])
    results = Vector{Tuple{Vector{T},Matrix{T},T}}()

    for k in 1:K
        # Shift u_k to be strictly positive and normalize to the simplex
        # to form the target share t_k ∈ int(Δ_n). Same convention as
        # solve_separation_inversion_ces, so the comparison across classes
        # in solve_separation_multicut is apples-to-apples.
        t_k = u[k, :] .- minimum(u[k, :]) .+ T(1e-8)
        t_k = t_k ./ sum(t_k)

        # Invert the Leontief share at p_k to recover a (gauge: min(a) = 1).
        a = Ξ[k][1] ./ t_k
        a = a ./ minimum(a)

        # Evaluate this Leontief atom at every sample.
        γ_new = Matrix{T}(undef, K, n)
        obj = zero(T)
        for k2 in 1:K
            γ_new[k2, :] .= leontief_share(a, Ξ[k2][1])
            obj += dot(view(u, k2, :), view(γ_new, k2, :))
        end
        push!(results, (a, γ_new, obj))
    end
    return results
end
