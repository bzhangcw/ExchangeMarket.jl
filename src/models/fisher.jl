# -----------------------------------------------------------------------
# structs for Fisher Market
# @author:Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
__precompile__(false)

using LinearAlgebra, SparseArrays
using JuMP, Random
import MathOptInterface as MOI
using Printf, DataFrames

function toy_fisher(ρ)
    m = 2
    n = 3
    fisher = FisherMarket(m, n; ρ=ρ)
    c = [
        0.7337 6.9883 9.1493
        3.4924 6.2826 1.9281
    ]
    fisher.uₛ = (x, i) -> c[i, :]' * x
    # initialize as linear utility
    fisher.u = fisher.uₛ
    fisher.∇u = (x, i) -> c[i, :]
    fisher.w = [0.1677; 0.5711]
    fisher.ρ = ρ
    return fisher
end


Base.@kwdef mutable struct FisherMarket{T}
    m::Int # number of agents
    n::Int # number of goods
    x::Matrix{T} # equilibrium allocation
    p::Vector{T} # equilibrium price
    w::Vector{T} # wealth
    q::Vector{T} # goods
    # -----------------------------------------------------------------------
    # utility function
    # default utility function: linear
    uₛ::Union{Function,Nothing} = nothing
    # override utility function
    # - utility function and gradient function of utility 
    u::Union{Function,Nothing} = nothing
    ∇u::Union{Function,Nothing} = nothing
    # - current value and gradient of utility
    val_u::Vector{T}
    val_∇u::Matrix{T}
    # - indirect utility function and gradient function of indirect utility
    ν::Union{Function,Nothing} = nothing
    ν∇ν::Union{Function,Nothing} = nothing

    # - current value and gradient of indirect utility
    val_ν::Vector{T}
    val_∇ν::Matrix{T}
    val_Hν::Matrix{T}

    # - the vector c to parameterize the utility function
    c::Matrix{T}
    # CES parameter
    ρ::T
    σ::T

    # -----------------------------------------------------------------------
    # dataframe for logging and display
    df::Union{DataFrame,Nothing} = nothing

    # -----------------------------------------------------------------------
    FisherMarket(m, n; ρ=1.0, seed=1) = (
        Random.seed!(seed);
        this = new{Float64}();
        this.m = m;
        this.n = n;
        this.ρ = ρ;
        this.σ = σ = ρ == 1.0 ? Inf : ρ / (1.0 - ρ);
        this.c = c = 10 * rand(m, n);
        this.uₛ = (x, i) -> c[i, :]' * x;

        this.q = m * rand(n); # in O(m)
        this.w = w = rand(m);
        this.p = zeros(n);
        this.x = zeros(m, n);
        this.df = DataFrame();

        if ρ == 1.0
            # linear utility
            this.u = this.uₛ
            this.∇u = (x, i) -> c[i, :]
            this.ν∇ν = (p, i) -> begin
                ν = w[i] * maximum(c[i, :] ./ p)
                ∇ν = nothing # nonsmooth
                return ν, ∇ν
            end
        else
            # CES utility, ρ < 1
            this.u = (x, i) -> sum(c[i, :] .* (x .^ ρ))^(1 / ρ)
            this.∇u = (x, i) -> sum(c[i, :] .* (x .^ ρ))^(1 / ρ - 1) .*
                                (x .^ (ρ - 1)) .*
                                c[i, :]

            # ignore outer power and coeff
            # only compute
            # c^(1+σ)'p^(-σ)
            this.ν∇ν = (p, i) -> begin
                _cs = c[i, :] .^ (1 + σ)
                _ps = p .^ (-σ)
                _cp = _cs' * _ps
                ν = _cp
                ∇ν = -σ .* _cs .* (p .^ (-σ - 1))
                Hν = σ * (σ + 1) .* _cs .* (p .^ (-σ - 2))
                return ν, ∇ν, Hν
            end
            this.ν = (p, i) -> begin
                _cs = c[i, :] .^ (1 + σ)
                _ps = p .^ (-σ)
                _cp = _cs' * _ps
                ν = _cp
                return ν
            end

        end;

        this.val_u = zeros(m);
        this.val_∇u = zeros(m, n);
        this.val_ν = zeros(m);
        this.val_∇ν = zeros(m, n);
        this.val_Hν = zeros(m, n);
        return this
    )
end

@doc """
    __validate(fisher::FisherMarket)
    -----------------------------------------------------------------------
    validate the equilibrium of the Fisher Market.
    use the price attached in the FisherMarket if no alg is provided.
    inner use only.
"""
function __validate(fisher::FisherMarket)
    validate(fisher, nothing)
end

function validate(fisher::FisherMarket, alg)
    m = fisher.m
    n = fisher.n
    u = fisher.u
    x = fisher.x
    p = isnothing(alg) ? fisher.p : alg.p
    μ = isnothing(alg) ? 0.0 : alg.μ
    w = fisher.w

    fisher.df = df = DataFrame(
        :utility => fisher.val_u,
        :left_budget => w - x * p,
        :left_budget_μ => w - x * p .+ μ * n,
    )
    println(__default_sep)
    @printf("\t\tequilibrium information\n")
    println(__default_sep)
    @show first(df, 10)
    println(__default_sep)
    _excess = sum(fisher.x; dims=1)[:] - fisher.q
    @printf(" :market excess: [%.4e, %.4e]\n", maximum(_excess), minimum(_excess))
    println(__default_sep)
end


Base.copy(z::FisherMarket{T}) where {T} = begin
    this = FisherMarket(z.m, z.n)
    copy_fields(this, z)
    return this
end