# -----------------------------------------------------------------------
# Multi-class pricing dispatcher for the column-generation framework.
#
# The pricing subproblem in CG is:
#   given dual variables (u, μ) from the master, find a column γ_new with
#   reduced cost rc = Σ_k <u_k, γ_{new,k}> - μ > 0.
#
# Different function classes (CES, linear, Leontief, ...) parameterize γ
# differently. This file:
#   - keeps the class-agnostic utilities (reduced_cost, add_to_gamma!,
#     drop_zero_columns!),
#   - includes per-class pricing files (ces.jl, linear.jl, leontief.jl),
#   - provides solve_pricing_class / solve_pricing_multiclass, which run
#     each class's pricing routine and pick the column with the largest rc.
#
# The class-specific solvers all return a NamedTuple
#   (γ_new::Matrix{T}, params::NamedTuple, obj::T, class::Symbol)
# so the dispatcher can pick the best one without knowing the class.
# -----------------------------------------------------------------------

using LinearAlgebra
using ExchangeMarket

include("./ces.jl")
include("./linear.jl")
include("./leontief.jl")

# -----------------------------------------------------------------------
# Class-agnostic CG utilities
# -----------------------------------------------------------------------
"""
    reduced_cost(γ_new, u, μ)

Compute the reduced cost for a new agent with bidding vectors γ_new:
    reduced_cost = Σ_k <u_k, γ_new_k> - μ.

If reduced_cost > 0, adding this agent can improve the master problem.
"""
function reduced_cost(γ_new::Matrix{T}, u::Matrix{T}, μ::T) where T
    K = size(γ_new, 1)
    rc = sum(dot(u[k, :], γ_new[k, :]) for k in 1:K) - μ
    return rc
end

"""
    add_to_gamma!(γ_ref::Ref{Array{T,3}}, γ_new)

Append a new agent's bidding vectors to γ in-place.
γ_ref holds γ with shape (m, K, n); γ_new has shape (K, n).
After return, γ_ref[] has shape (m+1, K, n).
"""
function add_to_gamma!(γ_ref::Ref{Array{T,3}}, γ_new::Matrix{T}) where T
    γ = γ_ref[]
    m, K, n = size(γ)
    @assert size(γ_new) == (K, n) "γ_new must be (K, n) = ($K, $n)"
    γ_expanded = zeros(T, m + 1, K, n)
    γ_expanded[1:m, :, :] .= γ
    γ_expanded[m+1, :, :] .= γ_new
    γ_ref[] = γ_expanded
    return nothing
end

"""
    drop_zero_columns!(fa::FisherMarket, γ_ref, w; tol=1e-8)

Remove agents with weight ≤ tol from both the FisherMarket and γ.
First syncs fa with w, then drops zero-weight agents. Returns the number dropped.
"""
function drop_zero_columns!(fa::FisherMarket{T}, γ_ref::Ref{Array{T,3}}, w::Vector{T}; tol=1e-8) where T
    m_γ = size(γ_ref[], 1)
    @assert length(w) == m_γ "w length ($(length(w))) must match γ_ref agent dim ($m_γ)"

    keep = findall(w .> tol)
    ndrop = m_γ - length(keep)
    ndrop == 0 && return 0

    γ_ref[] = γ_ref[][keep, :, :]

    fa.m = length(keep)
    fa.c = fa.c[:, keep]
    fa.ρ = fa.ρ[keep]
    fa.σ = fa.σ[keep]
    fa.w = w[keep]
    fa.x = fa.x[:, keep]
    fa.g = fa.g[:, keep]
    fa.s = fa.s[:, keep]
    while length(fa.val_u) < m_γ
        push!(fa.val_u, zero(T))
    end
    while length(fa.ε_br_play) < m_γ
        push!(fa.ε_br_play, fa.ε_br_play[1])
    end
    fa.val_u = fa.val_u[keep]
    fa.ε_br_play = fa.ε_br_play[keep]

    return ndrop
end

# -----------------------------------------------------------------------
# Multi-class pricing dispatcher
# -----------------------------------------------------------------------
"""
    solve_pricing_class(class::Symbol, Ξ, u; kwargs...)

Run the pricing subproblem for a single class. Returns a NamedTuple
    (γ_new::Matrix{T}, params::NamedTuple, obj::T, class::Symbol).

Supported classes:
- `:ces`      — full CES, free σ; LP warm-start + LBFGS refinement.
- `:linear`   — linear utility class H(1); big-M MIP (eq.cg.sep.linear).
- `:leontief` — CES boundary σ → -1⁺; concave fix-σ LBFGS at σ = σ_leontief.
"""
function solve_pricing_class(class::Symbol,
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    kwargs...) where T
    if class === :ces
        return solve_pricing_ces(Ξ, u; kwargs...)
    elseif class === :linear
        return solve_pricing_linear(Ξ, u; kwargs...)
    elseif class === :leontief
        return solve_pricing_leontief(Ξ, u; kwargs...)
    elseif class === :ql
        return solve_pricing_ql(Ξ, u; kwargs...)
    else
        error("Unknown pricing class: $class. Supported: :ces, :linear, :leontief, :ql.")
    end
end

"""
    solve_pricing_multiclass(Ξ, u, μ, classes; kwargs...)

Solve the pricing subproblem for each class in `classes`, compute the reduced
cost of each candidate column, and return the candidate with the largest rc.

Returns a NamedTuple
    (γ_new, params, obj, class, rc)
where `params` and `class` are passed through from the winning class's
solver and `rc = Σ_k <u_k, γ_new_k> - μ`.
"""
function solve_pricing_multiclass(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T}, μ::T, classes::Vector{Symbol};
    verbose::Bool=false,
    kwargs...) where T
    @assert !isempty(classes) "classes must be non-empty"
    best = nothing
    best_rc = T(-Inf)
    for class in classes
        cand = solve_pricing_class(class, Ξ, u; verbose=verbose, kwargs...)
        rc = reduced_cost(cand.γ_new, u, μ)
        verbose && println("  class=$class: obj=$(cand.obj), rc=$rc")
        if rc > best_rc
            best_rc = rc
            best = (γ_new=cand.γ_new, params=cand.params, obj=cand.obj, class=cand.class, rc=rc)
        end
    end
    return best
end

"""
    solve_pricing_inversion_multiclass(Ξ, u, classes; σ_grid=range(-0.9, 30.0, length=50))

Per-sample multicut inversion across the inversion-capable classes
(currently `:ces` and `:linear`). For each sample `k`, compute the
inverted candidate in every allowed class and **pick the one with the
largest pricing-oracle objective** `Σ_{k'} ⟨u_{k'}, γ(p_{k'}; y, σ)⟩`.
The objective is on the same scale across classes, so the per-`k`
argmax is a valid greedy choice for the K columns added per multicut
pass — at most K (not |classes|·K) atoms enter the surrogate.

CES uses `solve_pricing_inversion` (σ-grid + Brent refinement);
linear uses `solve_pricing_inversion_linear` (no σ — read off the
bang-per-buck winner from `argmax_j u[k, j]`).

Returns a `Vector` of K NamedTuples
    `(class::Symbol, y, σ, γ_new::Matrix, obj)`.
A sample whose value of K is `:none` indicates no allowed class
produced an inverted candidate (e.g., neither `:ces` nor `:linear`
was in `classes`); the caller should skip such entries.
"""
function solve_pricing_inversion_multiclass(
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T},
    classes::Vector{Symbol};
    σ_grid=range(-0.9, 30.0, length=50)
) where T
    K = length(Ξ)
    n = length(Ξ[1][1])
    ces_results = (:ces in classes) ?
                  solve_pricing_inversion(Ξ, u; σ_grid=σ_grid) : nothing
    lin_results = (:linear in classes) ?
                  solve_pricing_inversion_linear(Ξ, u) : nothing

    chosen = Vector{NamedTuple}(undef, K)
    for k in 1:K
        best = (class=:none, y=zeros(T, n), σ=T(NaN),
                γ_new=zeros(T, K, n), obj=T(-Inf))
        if !isnothing(ces_results)
            y, σ, γ_new, obj = ces_results[k]
            if obj > best.obj
                best = (class=:ces, y=y, σ=σ, γ_new=γ_new, obj=obj)
            end
        end
        if !isnothing(lin_results)
            y, γ_new, obj = lin_results[k]
            if obj > best.obj
                best = (class=:linear, y=y, σ=T(Inf), γ_new=γ_new, obj=obj)
            end
        end
        chosen[k] = best
    end
    return chosen
end

# -----------------------------------------------------------------------
# Class-aware market expansion
# -----------------------------------------------------------------------
"""
    add_column_to_market!(fa::FisherMarket, params::NamedTuple, class::Symbol, w_new=0.0)

Convert a class-specific pricing solution `params=(y, σ)` into a CES agent
and append it to `fa`. All classes are stored as CES: linear sits at
σ → ∞ (large σ_linear), Leontief at σ → -1⁺ (σ_leontief ≈ -0.9).

The map `(y, σ) → (c, ρ)` is `c = exp(y / (1+σ))`, `ρ = σ / (1+σ)`. Because
γ is softmax-invariant under `y → y + α·1` (equivalently `c → c · κ`), the
pricing solver only pins `y` up to an additive constant. We normalize by
shifting so `max(y) = 0`, which gives the canonical representative `c ∈
(0, 1]^n` (`max(c) = 1`). At extreme σ → -1 the small entries may underflow
to 0; `compute_gamma` handles that correctly via log-space softmax.
"""
function add_column_to_market!(fa::FisherMarket{T}, params::NamedTuple, class::Symbol, w_new::T=zero(T)) where T
    y, σ = params.y, params.σ
    y_shifted = y .- maximum(y)                              # max(y) = 0  ⇒  max(c) = 1
    if class === :linear
        # Linear utility: y = log c (raw), stored as a true LinearAgent via ρ = 1.
        # σ becomes Inf inside the FisherMarket via expand_players!; agent_type
        # dispatches on ρ == 1.0 to return LinearAgent(), and compute_gamma
        # special-cases σ = Inf to the bang-per-buck argmax.
        c_new = exp.(y_shifted)
        ρ_new = T(1)
    elseif class === :ces || class === :leontief
        c_new = exp.(y_shifted ./ (1 + σ))
        ρ_new = T(σ / (1 + σ))
    elseif class === :ql
        error(":ql atoms cannot be stored in FisherMarket — the QL share is " *
              "piecewise and budget-dependent, not CES. A parallel container " *
              "or a per-atom class tag in FisherMarket is needed before " *
              "QL atoms can participate in the master LP. The pricing oracle " *
              "(solve_pricing_ql) works standalone for off-line analysis.")
    else
        error("Unknown class for market expansion: $class.")
    end
    add_to_market!(fa, c_new, ρ_new, w_new)
    return fa
end

# -----------------------------------------------------------------------
# Pretty-print a column's class as ces(ρ).
#   :linear   → ces(1)
#   :leontief → ces(-∞)
#   :ces      → ces(0.42) with the recovered ρ
# This is the indicator the user sees in the iteration log.
# -----------------------------------------------------------------------
function format_class(class::Symbol, params::NamedTuple)
    if class === :linear
        return "ces(1)"
    elseif class === :leontief
        return "ces(-∞)"
    elseif class === :ces
        ρ = params.σ / (1 + params.σ)
        return "ces($(round(ρ; digits=2)))"
    else
        return String(class)
    end
end

format_class_from_yσ(y, σ) = "ces($(round(σ / (1 + σ); digits=2)))"
