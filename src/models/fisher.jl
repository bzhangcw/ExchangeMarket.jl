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

function toy_fisher()
    m = 2
    n = 3
    fisher = FisherMarket(m, n)
    c = [
        0.7337 6.9883 9.1493
        3.4924 6.2826 1.9281
    ]
    fisher.uₛ = (x, i) -> c[i, :]' * x
    # initialize as linear utility
    fisher.u = fisher.uₛ
    fisher.∇u = (x, i) -> c[i, :]
    fisher.w = [0.1677; 0.5711]
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
    # - utility function
    u::Union{Function,Nothing} = nothing
    # - gradient function of utility 
    ∇u::Union{Function,Nothing} = nothing
    # - current value of utility function
    val_u::Vector{T}
    # - current value of gradient function
    val_∇u::Matrix{T}
    # - the vector c to parameterize the utility function
    c::Matrix{T}
    # CES parameter
    ρ::T

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
        this.c = c = 10 * rand(m, n);
        this.uₛ = (x, i) -> c[i, :]' * x;
        # initialize as linear utility
        this.u = this.uₛ;
        this.∇u = (x, i) -> c[i, :];
        this.val_u = zeros(m);
        this.val_∇u = c;
        this.q = m * rand(n); # in O(m)
        this.w = rand(m);
        this.p = zeros(n);
        this.x = zeros(m, n);
        this.df = DataFrame();
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