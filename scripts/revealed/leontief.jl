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

"""
    leontief_from_homothetic_plc(plc::PLCAgent)

Construct the Leontief atom equivalent to a homothetic PLC with L = n and b = 0.
Computes v = A \\ 1 and returns LeontiefAgent(1 ./ v).

Throws an `ArgumentError` if the PLC is not in the homothetic theorem's hypotheses
(b ≠ 0, L ≠ n, A singular, or v has non-positive entries).
"""
function leontief_from_homothetic_plc(plc::PLCAgent)
    n = size(plc.a, 2)
    plc.L == n || throw(ArgumentError(
        "leontief_from_homothetic_plc requires L = n; got L=$(plc.L), n=$n"))
    all(iszero, plc.b) || throw(ArgumentError(
        "leontief_from_homothetic_plc requires b = 0; got b = $(plc.b)"))
    A = Matrix(plc.a)
    v = A \ ones(n)
    all(>(0), v) || throw(ArgumentError(
        "v = A \\ 1 must be strictly positive; got v = $v"))
    return LeontiefAgent(1.0 ./ v)
end

"""
    active_pieces(agent::PLCAgent, x; tol=1e-6)

Indices ℓ ∈ [L] where aℓ'x + bℓ is within `tol` of min_ℓ' (aℓ''x + bℓ').
"""
function active_pieces(agent::PLCAgent, x::AbstractVector; tol::Real=1e-6)
    vals = [dot(view(agent.a, ℓ, :), x) + agent.b[ℓ] for ℓ in 1:agent.L]
    umin = minimum(vals)
    return findall(v -> v - umin <= tol, vals)
end

"""
    plc_optimality_gap(plc::PLCAgent, x, p, w; u_star=nothing)

Quantify how far `x` is from being an optimal demand of `plc` at price `p` and budget `w`.
PLC demand is set-valued on flat utility regions, so `x` may differ from the LP solver's
output yet still be in the optimal demand set. The check is a *feasibility system*:

  1. Non-negativity        : x ≥ 0           → feas_neg   = max(0, -min(x))
  2. Budget                : p'x ≤ w         → feas_budget = max(0, p'x - w)
  3. Optimal utility value : u(x) = u_LP(p, w) → util_gap   = |u(x) - u_LP(p, w)|

`x` is in the optimal demand set iff all three are zero within tolerance.
Pass a precomputed `u_star = solve_plc_demand(plc, p, w)[2]` to avoid re-solving.
Returns a NamedTuple `(feas_neg, feas_budget, util_gap, u_cand, u_star)`.
"""
function plc_optimality_gap(plc::PLCAgent, x::AbstractVector, p::AbstractVector, w::Real;
                            u_star::Union{Nothing,Real}=nothing)
    feas_neg = max(0.0, -minimum(x))
    feas_budget = max(0.0, dot(p, x) - w)
    u_cand = minimum(dot(view(plc.a, ℓ, :), x) + plc.b[ℓ] for ℓ in 1:plc.L)
    if u_star === nothing
        _, u_star = solve_plc_demand(plc, p, w)
    end
    util_gap = abs(u_cand - u_star)
    return (feas_neg=feas_neg, feas_budget=feas_budget, util_gap=util_gap,
            u_cand=u_cand, u_star=u_star)
end

"""
    is_plc_optimal_demand(plc::PLCAgent, x, p, w; tol=1e-6, kwargs...)

Bool wrapper around [`plc_optimality_gap`](@ref): `true` iff feasibility violations and
utility gap are all within `tol`.
"""
function is_plc_optimal_demand(plc::PLCAgent, x::AbstractVector, p::AbstractVector, w::Real;
                               tol::Real=1e-6, kwargs...)
    g = plc_optimality_gap(plc, x, p, w; kwargs...)
    return max(g.feas_neg, g.feas_budget, g.util_gap) <= tol
end

"""
    leontief_from_active_subset(plc::PLCAgent, A_idx)

Construct the Leontief atom that matches the homothetic PLC restricted to the
active subset `A_idx ⊆ [L]` of size n. Requires `b[A_idx] = 0`.
"""
function leontief_from_active_subset(plc::PLCAgent, A_idx::AbstractVector{<:Integer})
    n = size(plc.a, 2)
    length(A_idx) == n || throw(ArgumentError(
        "active subset must have size n = $n; got |A_idx|=$(length(A_idx))"))
    all(iszero, view(plc.b, A_idx)) || throw(ArgumentError(
        "leontief_from_active_subset requires b[A_idx] = 0"))
    A_sub = plc.a[A_idx, :]
    v = A_sub \ ones(n)
    all(>(0), v) || throw(ArgumentError(
        "v = A_sub \\ 1 must be strictly positive; got v = $v"))
    return LeontiefAgent(1.0 ./ v)
end

# -----------------------------------------------------------------------
# Leontief pricing as the CES boundary σ → -1⁺
# -----------------------------------------------------------------------
"""
    solve_pricing_leontief(Ξ, u; σ_leontief=-0.9, y_init=nothing, verbose=false, kwargs...)

Pricing subproblem for the Leontief function class, viewed as the CES
boundary σ → -1⁺ (equivalently ρ → -∞):

    max_{y ∈ ℝ^n} Σ_k u_k^T softmax(y - σ_leontief log p_k).

The default σ_leontief = -0.9 (ρ = -9) is a numerical compromise: closer
to -1 makes the recovered c = exp(y/(1+σ)) overflow in Float64 because
1/(1+σ) grows without bound. With σ_leontief = -0.9 the storage handles
y entries up to ~35 in magnitude.

Returns a NamedTuple compatible with the multi-class dispatcher:
    (γ_new::Matrix, params=(y, σ), obj, class=:leontief).
"""
function solve_pricing_leontief(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    σ_leontief::Real=-0.9,
    y_init::Union{Vector{T},Nothing}=nothing,
    verbose::Bool=false,
    timelimit::Union{Real,Nothing}=nothing,
    kwargs...) where T
    σ = T(σ_leontief)
    @assert σ > -1 "σ_leontief must lie in (-1, 0); got $σ"
    y_opt, _, γ_new, obj = solve_pricing_fix_σ(Ξ, u, σ;
        y_init=y_init, timelimit=timelimit, verbose=verbose)
    verbose && println("Leontief pricing (σ=$σ, ρ=$(σ/(1+σ))): obj=$obj")
    return (γ_new=γ_new, params=(y=y_opt, σ=σ), obj=obj, class=:leontief)
end
