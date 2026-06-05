# -----------------------------------------------------------------------
# structs for Arrow–Debreu Exchange Market
# Endowments b (same size as allocation x); budgets w = b' * p
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2025/10/11
# -----------------------------------------------------------------------
__precompile__(true)

using LinearAlgebra, SparseArrays
using JuMP, Random
import MathOptInterface as MOI
using Printf, DataFrames

Base.@kwdef mutable struct ArrowDebreuMarket{T} <: AbstractMarket
    m::Int # number of agents
    n::Int # number of goods
    x::Union{Matrix{T},SparseMatrixCSC{T}} # allocation (n × m)
    p::Vector{T} # prices (n)
    w::Vector{T} # budgets (m) = b' * p
    q::Vector{T} # total supply of goods (n) = sum(b; dims=2)

    # endowments (n × m), same size as x
    b::Union{Matrix{T},SparseMatrixCSC{T}}

    # -------------------------------------------------------------------
    # utility and indirect utility
    # -------------------------------------------------------------------
    f::Union{Function,Nothing} = nothing
    f∇f::Union{Function,Nothing} = nothing

    # current values
    val_u::Vector{T}
    val_∇u::Union{Matrix{T},SparseMatrixCSC{T}}
    val_f::Vector{T}
    val_∇f::Union{Matrix{T},SparseMatrixCSC{T}}
    val_Hf::Union{Matrix{T},SparseMatrixCSC{T}}

    # parameters
    c::Union{Matrix{T},SparseMatrixCSC{T}} = nothing
    sumx::Vector{T}
    ρ::Vector{T}
    σ::Vector{T}

    # sparsity of cost matrix c (fraction of nonzeros)
    sparsity::Float64 = 1.0

    # dataframe for logging
    df::Union{DataFrame,Nothing} = nothing

    # constraints (reuse LinearConstr)
    constr_p::Union{LinearConstr,Nothing} = nothing

    """
    ArrowDebreuMarket(m, n; ρ=1.0, c=nothing, b=nothing, seed=1,
                      scale=1.0, sparsity=(2.0/n), bool_force_dense=true)

    Create an Arrow–Debreu exchange market. Endowments `b` have size n×m and
    budgets are computed as `w = b' * p`.
    """
    function ArrowDebreuMarket(m, n; ρ=1.0,
        constr_p=nothing,
        c=nothing,
        b=nothing,
        seed=1,
        scale=1.0,
        sparsity=(2.0 / n),
        bool_force_dense=true,
        # Normalize endowments so each good's total endowment is 1
        # (b ./= sum(b; dims=2) ⇒ q = 1). Matches the unit-supply convention
        # of the revealed-preference scripts and the AD wealth-redistribution
        # master (Σ_t b_t = 1).
        bool_unit_supply=false,
        verbose=true,
    )
        ts = time()
        verbose && println("ArrowDebreuMarket initialization started...")
        Random.seed!(seed)
        this = new{Float64}()
        this.m = m
        this.n = n
        this.sparsity = sparsity
        # allow scalar or vector ρ via helper
        _ρ, _σ = normalize_rho_sigma(ρ, m)
        this.ρ = copy(_ρ)
        this.σ = copy(_σ)

        # utilities parameter c (n × m)
        c = isnothing(c) ? scale * sprand(Float64, n, m, sparsity) : c
        if bool_force_dense
            c = Matrix(c)
        end
        this.c = copy(c)

        # endowments b (n × m), total supply q = sum(b, dims=2)
        _b = isnothing(b) ? sprand(Float64, n, m, sparsity) : b
        _b = bool_force_dense ? Matrix(_b) : _b
        if bool_unit_supply
            # Per-good normalization: each row of b sums to 1 ⇒ q = 1.
            row_sums = sum(_b; dims=2)
            @assert all(row_sums .> 0) "bool_unit_supply: every good needs at least one positive endowment"
            _b = _b ./ row_sums
        end
        this.b = copy(_b)
        this.q = sum(_b; dims=2)[:]

        # prices, allocations, budgets
        this.p = zeros(n)
        this.x = similar(c)
        this.sumx = zeros(n)
        this.w = zeros(m) # will be set via update_budget!(this)

        # aux storages
        this.df = DataFrame()
        this.constr_p = constr_p

        # indirect utility (price-space) closures
        this.f = (p, i) -> begin
            ρᵢ = this.ρ[i]
            if ρᵢ == 1.0
                return maximum(c[:, i] ./ p)
            else
                σᵢ = this.σ[i]
                _cs = spow.(c[:, i], (1 + σᵢ))
                _ps = spow.(p, (-σᵢ))
                return _cs' * _ps
            end
        end
        this.f∇f = (p, i) -> begin
            ρᵢ = this.ρ[i]
            if ρᵢ == 1.0
                f = maximum(c[:, i] ./ p)
                n = length(p)
                return f, zeros(n), zeros(n)
            else
                σᵢ = this.σ[i]
                _ci = c[:, i]
                _cs = spow.(_ci, 1 + σᵢ)
                _ps = spow.(p, -σᵢ)
                _cp = _cs' * _ps
                f = _cp
                ∇f = -σᵢ .* _cs .* spow.(p, (-σᵢ - 1))
                Hf = σᵢ * (σᵢ + 1) .* _cs .* (spow.(p, -σᵢ - 2))
                return f, ∇f, Hf
            end
        end

        this.val_u = zeros(m)
        this.val_∇u = similar(c)
        this.val_f = zeros(m)
        this.val_∇f = similar(c)
        this.val_Hf = similar(c)

        verbose && println("ArrowDebreuMarket initialized in $(time() - ts) seconds")
        return this
    end
end

"""
    ces_share(ad::ArrowDebreuMarket, i::Int, p::AbstractVector)

Closed-form CES expenditure share of agent `i` at price `p` (independent of
the budget by homotheticity):

    γ_{ij} = c_{ij}^{1+σ_i} p_j^{-σ_i} / Σ_l c_{il}^{1+σ_l} p_l^{-σ_i},

computed in log space for numerical stability. The linear boundary ρ_i = 1
(σ_i = ∞) is the bang-per-buck vertex: γ puts mass 1 on argmax_j c_{ij}/p_j.
Zero entries of `c` (sparse instances) carry zero share.
"""
function ces_share(ad::ArrowDebreuMarket{T}, i::Int, p::AbstractVector) where {T}
    n = ad.n
    c_i = @view ad.c[:, i]
    σ_i = ad.σ[i]
    γ = zeros(T, n)
    if isinf(σ_i)
        # linear utility: spend everything on the best bang-per-buck good
        j_star = argmax([c_i[j] > 0 ? c_i[j] / p[j] : -Inf for j in 1:n])
        γ[j_star] = one(T)
        return γ
    end
    # log-space softmax over goods with positive c
    z = fill(T(-Inf), n)
    for j in 1:n
        c_i[j] > 0 && (z[j] = (1 + σ_i) * log(c_i[j]) - σ_i * log(p[j]))
    end
    z_max = maximum(z)
    @assert isfinite(z_max) "ces_share: agent $i has no positive c entries"
    s = zero(T)
    for j in 1:n
        if isfinite(z[j])
            γ[j] = exp(z[j] - z_max)
            s += γ[j]
        end
    end
    γ ./= s
    return γ
end

"""
    aggregate_demand(ad::ArrowDebreuMarket, p::AbstractVector)

Aggregate Arrow–Debreu demand at price `p`: each agent's budget is the value
of its endowment, w_i(p) = ⟨p, b_i⟩, and its demand is the CES closed form

    x_{ij} = w_i(p) γ_{ij}(p) / p_j.

Returns g(p) = Σ_i x_i ∈ ℝⁿ. Side effect: `ad.w` is updated to the budgets at
`p` (via `update_budget!`).
"""
function aggregate_demand(ad::ArrowDebreuMarket{T}, p::AbstractVector) where {T}
    update_budget!(ad, Vector{T}(p))
    g = zeros(T, ad.n)
    for i in 1:ad.m
        γ_i = ces_share(ad, i, p)
        g .+= ad.w[i] .* γ_i ./ p
    end
    return g
end

"""
    update_budget!(ad::ArrowDebreuMarket)
    Update budgets w from current prices p and endowments b: w = b' * p.
"""
function update_budget!(ad::ArrowDebreuMarket, p::Vector{T}) where {T}
    ad.w .= ad.b' * p
    return ad
end

"""
    update_supply!(ad::ArrowDebreuMarket)
    Update total supply q from endowments b.
"""
function update_supply!(ad::ArrowDebreuMarket)
    ad.q .= sum(ad.b; dims=2)[:]
    return ad
end


Base.copy(z::ArrowDebreuMarket{T}) where {T} = begin
    this = ArrowDebreuMarket(z.m, z.n)
    copy_fields(this, z)
    return this
end