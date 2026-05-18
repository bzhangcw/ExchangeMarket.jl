# -----------------------------------------------------------------------
# Leontief market utilities for revealed-preference experiments.
#   - Random Leontief agent generation
#   - Closed-form demand
#   - Active-set diagnostics for PLC -> Leontief comparison
# -----------------------------------------------------------------------

using LinearAlgebra, Random
using ExchangeMarket

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


# -----------------------------------------------------------------------
# Leontief separation as the CES boundary σ → -1⁺
# -----------------------------------------------------------------------
"""
    solve_separation_leontief(Ξ, u; σ_leontief=-0.9, y_init=nothing, verbose=false, kwargs...)

Separation subproblem for the Leontief function class, viewed as the CES
boundary σ → -1⁺ (equivalently ρ → -∞):

    max_{y ∈ ℝ^n} Σ_k u_k^T softmax(y - σ_leontief log p_k).

The default σ_leontief = -0.9 (ρ = -9) is a numerical compromise: closer
to -1 makes the recovered c = exp(y/(1+σ)) overflow in Float64 because
1/(1+σ) grows without bound. With σ_leontief = -0.9 the storage handles
y entries up to ~35 in magnitude.

Returns a NamedTuple compatible with the per-class separation oracle:
    (γ_new::Matrix, params=(y, σ), obj, class=:leontief).
"""
function solve_separation_leontief(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    σ_leontief::Real=-0.9,
    y_init::Union{Vector{T},Nothing}=nothing,
    verbose::Bool=false,
    timelimit::Union{Real,Nothing}=nothing,
    kwargs...) where T
    σ = T(σ_leontief)
    @assert σ > -1 "σ_leontief must lie in (-1, 0); got $σ"
    y_opt, _, γ_new, obj = solve_separation_fix_σ_ces(Ξ, u, σ;
        y_init=y_init, timelimit=timelimit, verbose=verbose)
    verbose && println("Leontief separation (σ=$σ, ρ=$(σ/(1+σ))): obj=$obj")
    return (γ_new=γ_new, params=(y=y_opt, σ=σ), obj=obj, class=:leontief)
end
