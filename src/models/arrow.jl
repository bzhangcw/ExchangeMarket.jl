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

spow(x, y) = x == 0.0 ? 0.0 : x^y

Base.@kwdef mutable struct ArrowDebreuMarket{T}
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
    uₛ::Union{Function,Nothing} = nothing
    u::Union{Function,Nothing} = nothing
    ∇u::Union{Function,Nothing} = nothing

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
    ρ::T
    σ::T

    # dataframe for logging
    df::Union{DataFrame,Nothing} = nothing

    # constraints (reuse LinearConstr)
    constr_x::Union{Vector{LinearConstr},Nothing} = nothing
    constr_p::Union{LinearConstr,Nothing} = nothing

    """
    ArrowDebreuMarket(m, n; ρ=1.0, c=nothing, b=nothing, seed=1,
                      scale=1.0, sparsity=(2.0/n), bool_force_dense=true)

    Create an Arrow–Debreu exchange market. Endowments `b` have size n×m and
    budgets are computed as `w = b' * p`.
    """
    function ArrowDebreuMarket(m, n; ρ=1.0,
        constr_x=nothing,
        constr_p=nothing,
        c=nothing,
        b=nothing,
        seed=1,
        scale=1.0,
        sparsity=(2.0 / n),
        bool_force_dense=true,
    )
        ts = time()
        println("ArrowDebreuMarket initialization started...")
        Random.seed!(seed)
        this = new{Float64}()
        this.m = m
        this.n = n
        this.ρ = ρ
        this.σ = σ = ρ == 1.0 ? Inf : ρ / (1.0 - ρ)

        # utilities parameter c (n × m)
        c = isnothing(c) ? scale * sprand(Float64, n, m, sparsity) : c
        if bool_force_dense
            c = Matrix(c)
        end
        this.c = copy(c)
        this.uₛ = (x, i) -> c[:, i]' * x

        # endowments b (n × m), total supply q = sum(b, dims=2)
        _b = isnothing(b) ? sprand(Float64, n, m, sparsity) : b
        _b = bool_force_dense ? Matrix(_b) : _b
        this.b = copy(_b)
        this.q = sum(_b; dims=2)[:]

        # prices, allocations, budgets
        this.p = zeros(n)
        this.x = similar(c)
        this.sumx = zeros(n)
        this.w = zeros(m) # will be set via update_budget!(this)

        # aux storages
        this.df = DataFrame()
        this.constr_x = constr_x
        this.constr_p = constr_p

        if ρ == 1.0
            # linear utility
            this.u = this.uₛ
            this.∇u = (x, i) -> c[:, i]
            this.f = (p, i) -> begin
                maximum(c[:, i] ./ p)
            end
            this.f∇f = (p, i) -> begin
                f = maximum(c[:, i] ./ p)
                ∇f = nothing # nonsmooth
                return f, ∇f
            end
        else
            # CES utility, ρ < 1
            this.u = (x, i) -> sum(c[:, i] .* spow.(x, ρ))^(1 / ρ)
            this.∇u = (x, i) -> sum(c[:, i] .* spow.(x, ρ))^(1 / ρ - 1) .*
                                spow.(x, ρ - 1) .* c[:, i]

            # indirect utility pieces
            this.f∇f = (p, _ci) -> begin
                _cs = spow.(_ci, 1 + σ)
                _ps = spow.(p, -σ)
                _cp = _cs' * _ps
                f = _cp
                ∇f = -σ .* _cs .* spow.(p, (-σ - 1))
                Hf = σ * (σ + 1) .* _cs .* (spow.(p, -σ - 2))
                return f, ∇f, Hf
            end
            this.f = (p, i) -> begin
                _cs = spow.(c[:, i], (1 + σ))
                _ps = spow.(p, (-σ))
                _cp = _cs' * _ps
                f = _cp
                return f
            end
        end

        this.val_u = zeros(m)
        this.val_∇u = similar(c)
        this.val_f = zeros(m)
        this.val_∇f = similar(c)
        this.val_Hf = similar(c)

        println("ArrowDebreuMarket initialized in $(time() - ts) seconds")
        return this
    end
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