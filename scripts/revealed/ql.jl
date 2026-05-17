# -----------------------------------------------------------------------
# Quasi-linear-log (QL) market utilities for revealed-preference experiments.
#   - Random QL agent generation
#   - Closed-form demand
#   - Revealed preference production (aggregate demand at random prices)
# -----------------------------------------------------------------------

using LinearAlgebra, Random
using ExchangeMarket

"""
    random_ql_agent(n; seed=nothing)

Generate a random QuasiLinearLogAgent in `n` goods.
  u(x) = Σ_{j<n} c_j log(x_j) + x_n, c ∈ R^{n-1}_{++}.

Each c_j is drawn from Uniform(0, 1).
"""
function random_ql_agent(n::Int; seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    c = rand(n - 1)
    return QuasiLinearLogAgent(n, c)
end

"""
    produce_revealed_preferences_ql(agents, w_vec, K, n;
        price_range=(0.5, 2.0), seed=nothing)

Generate K revealed-preference observations from a QL market.
Each observation is a random price vector and the aggregate demand g(p) = Σᵢ xᵢ(p).

Arguments:
- agents: Vector{QuasiLinearLogAgent}, one per agent
- w_vec: budgets (m,)
- K: number of observations
- n: number of goods

Returns Ξ = [(p₁, g₁), ..., (p_K, g_K)].
"""
function produce_revealed_preferences_ql(
    agents::Vector{QuasiLinearLogAgent},
    w_vec::Vector{Float64},
    K::Int,
    n::Int;
    price_range=(0.5, 2.0),
    seed=nothing
)
    !isnothing(seed) && Random.seed!(seed)
    m = length(agents)
    Ξ = Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, K)

    for k in 1:K
        p_k = price_range[1] .+ (price_range[2] - price_range[1]) .* rand(n)
        p_k = p_k ./ sum(p_k)

        g_k = zeros(n)
        for i in 1:m
            x_i, _ = solve_ql_demand(agents[i], p_k, w_vec[i])
            g_k .+= x_i
        end
        Ξ[k] = (copy(p_k), copy(g_k))
    end
    return Ξ
end

# -----------------------------------------------------------------------
# QL pricing oracle (CG separation for the QL atom subfamily)
# -----------------------------------------------------------------------
"""
    ql_share(c, p, w) -> γ (length n)

Closed-form QL spending share γ_j = p_j x_j / w at price `p`, budget `w`,
with parameter `c` of length n-1 (positive). Implements eq.ql.share:
  if p_n ≤ w / ⟨1,c⟩ : γ_j = c_j p_n / w (j < n),  γ_n = 1 - p_n ⟨1,c⟩ / w
  if p_n ≥ w / ⟨1,c⟩ : γ_j = c_j / ⟨1,c⟩ (j < n),  γ_n = 0.
"""
function ql_share(c::AbstractVector{T}, p::AbstractVector{T}, w::Real) where T
    n = length(p)
    @assert length(c) == n - 1 "QL c must have length n-1"
    γ = zeros(T, n)
    cbar = sum(c)
    pn = p[n]
    if pn <= w / cbar
        @inbounds for j in 1:(n-1)
            γ[j] = c[j] * pn / w
        end
        γ[n] = 1 - pn * cbar / w
    else
        @inbounds for j in 1:(n-1)
            γ[j] = c[j] / cbar
        end
        γ[n] = zero(T)
    end
    return γ
end

"""
    solve_pricing_ql(Ξ, u; w=1.0, verbose=false, kwargs...)

CG pricing oracle for the QL function class. Solves
    max_{c ∈ R^{n-1}_{++}}  Σ_k ⟨U e_k, γ_k(c, w)⟩
by the regime-enumeration scheme: for each m ∈ {0,...,K}, the constrained
subproblem reduces (after Charnes-Cooper β := ⟨1,c⟩, y := c/β) to a 1D max
of a convex piecewise-linear function on a closed interval, hence attained
at an endpoint. The overall optimum is the max over m and the two endpoints.

Returns a NamedTuple compatible with the multi-class dispatcher:
    (γ_new::Matrix{T}, params=(c, w), obj::T, class=:ql).

`γ_new[k, :]` holds γ_k(c*, w) evaluated at each sample's price.
"""
function solve_pricing_ql(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    w::Real=1.0,
    verbose::Bool=false,
    kwargs...) where T
    K = length(Ξ)
    n = length(Ξ[1][1])
    @assert n >= 2 "QL pricing requires n ≥ 2 goods"
    nm1 = n - 1
    w = T(w)

    # Sort samples by p_{k,n} ascending. perm[t] = original index of the t-th sorted sample.
    pn_vec = [Ξ[k][1][n] for k in 1:K]
    perm = sortperm(pn_vec)
    p_sorted_n = pn_vec[perm]

    # Endpoint β values: β_(t) = 1 / p_sorted_n[t] (for t ∈ 1:K), with sentinels
    # p_(0),n = 0 → 1/p_(0),n = +∞, p_(K+1),n = +∞ → 1/p_(K+1),n = 0.
    # For m ∈ {0,...,K}, the regime interval is β ∈ [1/p_(m+1),n, 1/p_(m),n].

    best_obj = T(-Inf)
    best_β = T(0)
    best_jstar = 1

    # Per-m coefficients: A_j = Σ_{k≤m} u_(k),j p_(k),n  (j<n),
    #                    B_j = Σ_{k>m} u_(k),j           (j<n),
    #                    C   = Σ_{k≤m} u_(k),n p_(k),n,
    #                    D   = Σ_{k≤m} u_(k),n.
    # Increment incrementally as m goes 0 → K.
    A = zeros(T, nm1)
    B = zeros(T, nm1)
    for j in 1:nm1
        for t in 1:K
            B[j] += u[perm[t], j]
        end
    end
    C = zero(T)
    D = zero(T)

    # Helper: evaluate g(β) = max_j (β A_j + B_j) - C β + D, and return (g, j*).
    function eval_g(β::T)
        best_val = T(-Inf)
        best_j = 1
        @inbounds for j in 1:nm1
            v = β * A[j] + B[j]
            if v > best_val
                best_val = v
                best_j = j
            end
        end
        # Note: u is K × n, so the constant /w from γ_j = c_j p_n / w is already
        # absorbed into the LP via the substitution. The objective expansion
        # below assumes unit budget w = 1 in the share formula; we rescale at
        # the end by 1/w when computing γ_new.
        return best_val - C * β + D, best_j
    end

    # m = 0: interval is [1/p_(1),n, +∞]; no upper endpoint to evaluate, only β = 1/p_(1),n.
    # m = K: interval is [0, 1/p_(K),n]; only β = 1/p_(K),n (lower endpoint = 0 gives degenerate atom).
    # For 0 < m < K: both endpoints.
    for m in 0:K
        # Update A, B, C, D from m-1 → m by moving the m-th sorted sample from corner to interior.
        if m >= 1
            t = m
            k_orig = perm[t]
            pkn = p_sorted_n[t]
            @inbounds for j in 1:nm1
                A[j] += u[k_orig, j] * pkn
                B[j] -= u[k_orig, j]
            end
            C += u[k_orig, n] * pkn
            D += u[k_orig, n]
        end

        # Endpoints of the interval
        β_lo = m == K ? zero(T) : T(1) / p_sorted_n[m+1]
        β_hi = m == 0 ? T(Inf) : T(1) / p_sorted_n[m]

        for β in (β_lo, β_hi)
            (β == T(Inf) || β == zero(T)) && continue
            g_val, j_star = eval_g(β)
            if g_val > best_obj
                best_obj = g_val
                best_β = β
                best_jstar = j_star
            end
        end
    end

    # Recover c* on the open simplex: vertex c = β · e_{j*} (closure limit).
    # Strict positivity is violated at the vertex; perturb slightly to avoid log singularities downstream.
    c_star = fill(T(1e-12), nm1)
    c_star[best_jstar] = best_β - (nm1 - 1) * T(1e-12)

    # Evaluate γ at every sample under c_star and the assumed budget w.
    γ_new = Matrix{T}(undef, K, n)
    for k in 1:K
        p_k = Ξ[k][1]
        γ_new[k, :] .= ql_share(c_star, p_k, w)
    end

    verbose && println("QL pricing: obj=$best_obj, β*=$best_β, j*=$best_jstar")
    return (γ_new=γ_new, params=(c=c_star, w=w), obj=best_obj, class=:ql)
end

