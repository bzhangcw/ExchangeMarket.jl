# -----------------------------------------------------------------------
# structs for Fisher Market
# @author:Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
__precompile__(false)

using LinearAlgebra, SparseArrays
using JuMP, COPT, MosekTools, Random
import MathOptInterface as MOI
using Printf, DataFrames

# translate log utility to exponential cone
# v <= log(x) => [v, 1, x] in MOI.ExponentialCone()
log_to_expcone!(x, v, model) = @constraint(
    model, [v, 1, x] in MOI.ExponentialCone()
)

Base.@kwdef mutable struct FisherMarket{T}
    model::Model
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

    # -----------------------------------------------------------------------
    # dataframe for logging and display
    df::Union{DataFrame,Nothing} = nothing

    # -----------------------------------------------------------------------
    FisherMarket(m, n; seed=1) = (
        Random.seed!(seed);
        this = new{Float64}();
        this.m = m;
        this.n = n;
        c = rand(m, n);
        this.uₛ = (x, i) -> c[i, :]' * x;
        # initialize as linear utility
        this.u = this.uₛ;
        this.∇u = (x, i) -> c[i, :];
        this.val_∇u = c;
        this.q = m * rand(n); # in O(m)
        this.w = rand(m);
        this.model = Model(
            optimizer_with_attributes(
                () -> MosekTools.Optimizer(),
                # "LogToConsole" => true
            )
        );
        this.df = DataFrame();
        return this
    )
end

function create_jump_model(fisher::FisherMarket)
    model = fisher.model
    m = fisher.m
    n = fisher.n
    u = fisher.u
    q = fisher.q
    w = fisher.w
    @variable(model, x[1:m, 1:n] >= 0)
    @variable(model, v[1:m])
    @variable(model, ℓ[1:m])
    @constraint(model, limit, x' * ones(m) .<= q)
    for i in 1:m
        @constraint(model, ℓ[i] == u(x[i, :], i))
        log_to_expcone!(ℓ[i], v[i], model)
    end
    @objective(model, Min, -sum([w[i] * v[i] for i in 1:m]))
    return
end

function solve_jump_model(fisher::FisherMarket)
    model = fisher.model
    optimize!(model)
    fisher.x = value.(model[:x])
    fisher.p = -dual.(model[:limit])
    fisher.val_u = value.(model[:ℓ])
    return
end

function validate(fisher::FisherMarket)
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
    @printf("agent information\n")
    println(__default_sep)
    @show first(df, 10)
    println(__default_sep)
    _excess = sum(fisher.x; dims=1)[:] - fisher.q
    @printf("goods: [%.4e, %.4e]\n", maximum(_excess), minimum(_excess))
    println(__default_sep)
end


Base.copy(z::FisherMarket{T}) where {T} = begin
    this = FisherMarket(z.m, z.n)
    copy_fields(this, z)
    return this
end