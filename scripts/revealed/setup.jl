# Methods and tracking utilities for revealed-preference CES surrogate fitting.
# File map (in include order):
#   - redistribute.jl : master / dual LPs (eq.cg.master, eq.cg.dual)
#   - pricing.jl      : multiclass dispatcher, drop_zero_columns!,
#                       add_column_to_market!, per-sample inversion merge
#   - frankwolfe.jl   : Frank-Wolfe runner (run_method_tracked_fw)
#   - cpm.jl          : column-generation runner (run_method_tracked)
#
# Helpers used by the FW/CG runners (produce_revealed_preferences,
# compute_gamma_from_market, compute_gamma, compute_gamma_matrix,
# evaluate_test_error) live in this file so they're available to all
# downstream includes.

using Printf
using Random
using LinearAlgebra
using ExchangeMarket

# -----------------------------------------------------------------------
# Master / dual LP solvers (define before runners include them).
# -----------------------------------------------------------------------
include("./redistribute.jl")

# -----------------------------------------------------------------------
# Revealed-preference data preparation
# -----------------------------------------------------------------------
"""
    produce_revealed_preferences(alg, f1::FisherMarket, K; price_range=(0.5, 2.0), seed=nothing)

Generate K random price vectors and compute aggregate demands from a FisherMarket.
Returns Ξ = [(p_1, g_1), ..., (p_K, g_K)] where g_k is the aggregate demand at price p_k.

Arguments:
- alg: Algorithm object (e.g., HessianBar)
- f1: FisherMarket object containing the market structure
- K: Number of price observations to generate
- price_range: (min, max) range for random prices
- seed: Random seed (optional)

After calling play!(alg, f1), the demand is computed and stored in f1.x.
"""
function produce_revealed_preferences(alg, f1::FisherMarket, K::Int;
    price_range=(0.5, 2.0), seed=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    n = f1.n
    Ξ = Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, K)

    for k in 1:K
        # Random price vector (normalized to sum to 1)
        p_k = price_range[1] .+ (price_range[2] - price_range[1]) .* rand(n)
        p_k = p_k ./ sum(p_k)  # normalize prices

        # Set price in the algorithm
        alg.p .= p_k

        # Compute demand via play!
        play!(alg, f1)

        # Aggregate demand: sum over all agents
        g_k = sum(f1.x, dims=2)[:]

        Ξ[k] = (copy(p_k), copy(g_k))
    end

    return Ξ
end

"""
    compute_gamma(p, c, σ)

Compute the CES bidding vector γ for given price p, coefficients c, and elasticity parameter σ.
    γ_j = (c_j^{1+σ} * p_j^{-σ}) / sum_ℓ(c_ℓ^{1+σ} * p_ℓ^{-σ})

Uses log-space computation (softmax) to avoid overflow for large |σ|.

Special case: when `σ` is `+Inf` (the linear regime, ρ = 1), γ is the
bang-per-buck vertex indicator `e_{argmax_j c_j / p_j}` as in
`fact.demand.linear`. This matches the storage convention used by
`add_column_to_market!` for the `:linear` class.
"""
function compute_gamma(p::AbstractVector, c::AbstractVector, σ::Real)
    # Linear regime: ρ = 1, σ = +∞; γ is the bang-per-buck vertex.
    if isinf(σ) && σ > 0
        γ = zeros(eltype(p), length(c))
        j_star = argmax(c ./ p)
        γ[j_star] = one(eltype(p))
        return γ
    end
    # log(numerator_j) = (1+σ) log(c_j) - σ log(p_j)
    z = (1 + σ) .* log.(c) .- σ .* log.(p)
    z_max = maximum(z)
    ez = exp.(z .- z_max)
    γ = ez ./ sum(ez)
    return γ
end

"""
    compute_gamma_from_market(f1::FisherMarket, Ξ)

Compute the bidding matrix γ[i,k,:] for a FisherMarket given revealed preferences Ξ.
Uses the market's CES parameters (c, σ) to compute bidding vectors.

Returns γ as a 3D array of size (m, K, n).
"""
function compute_gamma_from_market(f1::FisherMarket, Ξ::Vector{Tuple{Vector{T},Vector{T}}}) where T
    m, n = f1.m, f1.n
    K = length(Ξ)

    γ = zeros(T, m, K, n)
    for i in 1:m
        c_i = Vector(f1.c[:, i])  # ensure it's a dense vector
        σ_i = f1.σ[i]
        for k in 1:K
            p_k, _ = Ξ[k]
            γ[i, k, :] = compute_gamma(p_k, c_i, σ_i)
        end
    end

    return γ
end

"""
    compute_gamma_matrix(Ξ, C, σ_vec)

Compute the bidding matrix γ[i,k,:] for all agents i and observations k.
- Ξ: Vector of (p_k, g_k) tuples
- C: Matrix of coefficients, C[i,:] = c_i
- σ_vec: Vector of elasticity parameters σ_i

Returns γ as a 3D array of size (m, K, n).
"""
function compute_gamma_matrix(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    C::Matrix{T},
    σ_vec::Vector{T}) where T
    m, n = size(C)
    K = length(Ξ)

    γ = zeros(T, m, K, n)
    for i in 1:m
        for k in 1:K
            p_k, _ = Ξ[k]
            γ[i, k, :] = compute_gamma(p_k, C[i, :], σ_vec[i])
        end
    end

    return γ
end


# -----------------------------------------------------------------------
# evaluation on test set: mean L∞ error of Σ w_i γ_i(p) vs target P g
# -----------------------------------------------------------------------
function evaluate_test_error(fa, Ξ_test)
    isempty(fa.w) && return NaN
    K = length(Ξ_test)
    n = length(Ξ_test[1][1])
    errs = Float64[]
    for (p, g) in Ξ_test
        target = p .* g
        fitted = zeros(n)
        for i in 1:fa.m
            c_i = Vector(fa.c[:, i])
            fitted .+= fa.w[i] .* compute_gamma(p, c_i, fa.σ[i])
        end
        push!(errs, norm(fitted .- target, Inf))
    end
    return sum(errs) / K
end

# -----------------------------------------------------------------------
# Pricing dispatcher + FW / CG runners.
# Order matters: pricing.jl defines `drop_zero_columns!`,
# `add_column_to_market!`, etc., which both runners use.
# -----------------------------------------------------------------------
include("./pricing.jl")
include("./frankwolfe.jl")
include("./cpm.jl")



# -----------------------------------------------------------------------
# methods: (name, pricing_kind, kwargs)
# pricing_kind ∈ {:cg_single, :cg_multicut} for CG, :fw for Frank-Wolfe.
# The kwargs key :classes selects which function classes the pricing
# dispatcher tries each iteration (defaults to [:ces] when omitted).
# Supported classes: :ces, :linear, :leontief, :ql (pricing only — storage TBD).
# -----------------------------------------------------------------------
method_kwargs = [
    [:CG, :cg_single,
        Dict(
            :max_iters => 200,
            :tol_obj => 1e-3,
            :tol_rc => 1e-5,
            :tol_delta => 1e-5,
            :drop => true,
            :classes => [:ces, :linear],
        )
    ],
    [:MultiCut, :cg_multicut,
        Dict(
            :max_iters => 200,
            :tol_obj => 1e-3,
            :tol_rc => 1e-3,
            :tol_delta => 1e-5,
            :drop => true,
            :classes => [:ces],
        )
    ],
    [:FW, :fw,
        Dict(
            :max_iters => 200,
            :batch_size => 0,           # 0 → full batch; set e.g. 32 for stochastic
            :tol_obj => 1e-3,
            :tol_delta => 1e-5,
            :step_rule => :diminishing,
            :seed => 0,
        )
    ],
    [:SFW, :fw,
        Dict(
            :max_iters => 200,
            :batch_size => 32,          # mini-batch stochastic FW
            :tol_obj => 1e-3,
            :tol_delta => 1e-5,
            :step_rule => :diminishing,
            :seed => 0,
        )
    ],
    [:FWjl, :fwjl,
        Dict(
            :max_iters => 200,
            :tol_obj => 1e-3,
            :seed => 0,
        )
    ],
]

colors = Dict(
    :CG => 1,
    :MultiCut => 2,
    :FW => 4,
    :SFW => 5,
    :FWjl => 6,
)

marker_style = Dict(
    :CG => :circle,
    :MultiCut => :rect,
    :FW => :diamond,
    :SFW => :star5,
    :FWjl => :utriangle,
)

# Pretty display names for legends and summary output. The CLI / symbol
# table key remains the Julia-friendly identifier; this dict lets us
# render dots or whitespace in labels (e.g., `FWjl` → "FW.jl"). Falls
# back to `String(name)` for unlisted methods.
display_name = Dict(
    :CG => "CG",
    :MultiCut => "MultiCut",
    :FW => "FW",
    :SFW => "SFW",
    :FWjl => "FW.jl",
)