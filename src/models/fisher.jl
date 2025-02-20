# -----------------------------------------------------------------------
# structs for Fisher Market
# @author:Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
__precompile__(true)

using LinearAlgebra, SparseArrays
using JuMP, Random
import MathOptInterface as MOI
using Printf, DataFrames

function toy_fisher(ρ)
    m = 2
    n = 3
    c = [
        0.7337 6.9883 9.1493
        3.4924 6.2826 1.9281
    ]
    w = [0.1677; 0.5711]
    fisher = FisherMarket(m, n; ρ=ρ, c=c, w=w)
    return fisher
end

spow(x, y) = x == 0.0 ? 0.0 : x^y

Base.@kwdef mutable struct FisherMarket{T}
    m::Int # number of agents
    n::Int # number of goods
    x::Matrix{T} # allocation
    p::Vector{T} # price
    w::Vector{T} # wealth
    q::Vector{T} # goods
    # -------------------------------------------------------------------
    # for propotional dynamics
    # -------------------------------------------------------------------
    b::Matrix{T} # bids

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
    f::Union{Function,Nothing} = nothing
    f∇f::Union{Function,Nothing} = nothing

    # - current value and gradient of indirect utility
    val_f::Vector{T}
    val_∇f::Matrix{T}
    val_Hf::Matrix{T}

    # - the vector c to parameterize the utility function
    c::Matrix{T}
    # CES parameter
    ρ::T
    σ::T

    # -----------------------------------------------------------------------
    # dataframe for logging and display
    df::Union{DataFrame,Nothing} = nothing

    # -----------------------------------------------------------------------
    FisherMarket(m, n; ρ=1.0,
        c=nothing, w=nothing, seed=1,
        scale=1.0, sparsity=(2.0 / n),
        bool_unit=true, bool_sparse=true,
        bool_unit_wealth=true,
    ) = (
        Random.seed!(seed);
        this = new{Float64}();
        this.m = m;
        this.n = n;
        this.ρ = ρ;
        this.σ = σ = ρ == 1.0 ? Inf : ρ / (1.0 - ρ);
        c = isnothing(c) ? scale * sprand(Float64, m, n, sparsity) : c;

        # ensure each row has at least one nonzero entry
        for i in 1:m
            if all(c[i, :] .== 0.0)
                c[i, rand(1:size(c, 2))] = scale
            end
        end;
        # ensure each column has at least one nonzero entry
        for j in 1:n
            if all(c[:, j] .== 0.0)
                c[rand(1:size(c, 1)), j] = scale
            end
        end;
        (!bool_sparse) && (c .+= scale);
        this.c = c = sparse(c);
        this.uₛ = (x, i) -> c[i, :]' * x;

        this.q = bool_unit ? ones(n) : m * rand(n); # in O(m)
        this.w = w = (isnothing(w) ? rand(m) : w) * scale;
        this.w .= bool_unit_wealth ? w ./ sum(w) : w;
        this.p = zeros(n);
        this.x = zeros(m, n);
        # sometimes use bids instead of allocation
        this.b = zeros(m, n);
        this.df = DataFrame();

        if ρ == 1.0
            # linear utility
            this.u = this.uₛ
            this.∇u = (x, i) -> c[i, :]
            this.f = (p, i) -> begin
                w[i] * maximum(c[i, :] ./ p)
            end
            this.f∇f = (p, i) -> begin
                f = w[i] * maximum(c[i, :] ./ p)
                ∇f = nothing # nonsmooth
                return f, ∇f
            end
        else
            # CES utility, ρ < 1
            this.u = (x, i) -> sum(c[i, :] .* spow.(x, ρ))^(1 / ρ)
            this.∇u = (x, i) -> sum(c[i, :] .* spow.(x, ρ))^(1 / ρ - 1) .*
                                spow.(x, ρ - 1) .* c[i, :]

            # the derivatives respect to p
            # ignore outer power and coeff
            # only compute
            # r := c^(1+σ)'p^(-σ)
            this.f∇f = (p, i) -> begin
                _cs = c[i, :] .^ (1 + σ)
                _ps = p .^ (-σ)
                _cp = _cs' * _ps
                f = _cp
                ∇f = -σ .* _cs .* (p .^ (-σ - 1))
                Hf = σ * (σ + 1) .* _cs .* (p .^ (-σ - 2))
                return f, ∇f, Hf
            end
            this.f = (p, i) -> begin
                _cs = c[i, :] .^ (1 + σ)
                _ps = p .^ (-σ)
                _cp = _cs' * _ps
                f = _cp
                return f
            end
        end;

        this.val_u = zeros(m);
        this.val_∇u = zeros(m, n);
        this.val_f = zeros(m);
        this.val_∇f = zeros(m, n);
        this.val_Hf = zeros(m, n);
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
    @printf(" :problem size\n")
    @printf(" :    number of agents: %d\n", fisher.m)
    @printf(" :    number of goods: %d\n", fisher.n)
    @printf(" :    avg number of nonzero entries in c: %.4f\n",
        length(sparse(fisher.c).nzval) / (fisher.m * fisher.n)
    )
    @printf(" :equilibrium information\n")
    @printf(" :method: %s\n", alg.name)
    println(__default_sep)
    println(first(df, 10))
    println(__default_sep)
    _excess = (sum(fisher.x; dims=1)[:] - fisher.q) ./ maximum(fisher.q)
    @printf(" :(normalized) market excess: [%.4e, %.4e]\n", minimum(_excess), maximum(_excess))
    @printf(" :            social welfare:  %.4e\n", (log.(fisher.val_u))' * fisher.w)
    println(__default_sep)
end


Base.copy(z::FisherMarket{T}) where {T} = begin
    this = FisherMarket(z.m, z.n)
    copy_fields(this, z)
    return this
end