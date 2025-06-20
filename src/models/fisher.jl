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

function add_nonzero_entries!(c, m, n, scale)
    rows = 1:m
    cols = rand(1:n, m)
    vals = fill(scale, m)
    c += sparse(rows, cols, vals, m, n)
    cols = 1:n
    rows = rand(1:m, n)
    vals = fill(scale, n)
    c += sparse(rows, cols, vals, m, n)
    return c
end


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
    x::Union{Matrix{T},SparseMatrixCSC{T}} # allocation
    p::Vector{T} # price
    w::Vector{T} # wealth
    q::Vector{T} # goods
    # -------------------------------------------------------------------
    # for propotional dynamics and IPMs
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
    val_∇u::Union{Matrix{T},SparseMatrixCSC{T}}
    # - indirect utility function and gradient function of indirect utility
    f::Union{Function,Nothing} = nothing
    f∇f::Union{Function,Nothing} = nothing

    # - current value and gradient of indirect utility
    val_f::Vector{T}
    val_∇f::Union{Matrix{T},SparseMatrixCSC{T}}
    val_Hf::Union{Matrix{T},SparseMatrixCSC{T}}

    # - the vector c to parameterize the utility function
    c::Union{Matrix{T},SparseMatrixCSC{T}} = nothing
    # sum of x (for a limited samples)
    sumx::Vector{T}
    # CES parameter
    ρ::T
    σ::T

    # -----------------------------------------------------------------------
    # dataframe for logging and display
    df::Union{DataFrame,Nothing} = nothing

    # -----------------------------------------------------------------------
    # constraints
    # -----------------------------------------------------------------------
    # - linear constraints on allocation
    constr_x::Union{Vector{LinearConstr},Nothing} = nothing
    # - linear constraints on price
    constr_p::Union{LinearConstr,Nothing} = nothing

    # -----------------------------------------------------------------------
    """
    FisherMarket(m, n; ρ=1.0, c=nothing, w=nothing, seed=1, scale=1.0, sparsity=(2.0/n), bool_unit=true, bool_unit_wealth=true, bool_ensure_nz=true, bool_force_dense=false)

    Create a Fisher market model with m agents and n goods.

    # Arguments
    - `m::Int`: Number of agents
    - `n::Int`: Number of goods
    - `ρ::Float64`: CES parameter (default: 1.0 for linear utility)
    - `c`: Utility function parameterization matrix (default: random sparse matrix)
    - `w`: Wealth vector (default: random vector)
    - `seed::Int`: Random seed (default: 1)
    - `scale::Float64`: Scale factor for costs and wealth (default: 1.0)
    - `sparsity::Float64`: Sparsity of cost matrix (default: 2.0/n)
    - `bool_unit::Bool`: Whether to use unit goods (default: true)
    - `bool_unit_wealth::Bool`: Whether to normalize wealth (default: true)
    - `bool_ensure_nz::Bool`: Whether to ensure non-zero entries (default: true)
    - `bool_force_dense::Bool`: Whether to force dense matrix (default: true)
    """
    function FisherMarket(m, n; ρ=1.0,
        constr_x=nothing,
        constr_p=nothing,
        c=nothing, w=nothing, seed=1,
        scale=1.0, sparsity=(2.0 / n),
        bool_unit=true,
        bool_unit_wealth=true,
        bool_ensure_nz=false,
        bool_force_dense=true,
    )
        ts = time()
        println("FisherMarket initialization started...")
        Random.seed!(seed)
        this = new{Float64}()
        this.m = m
        this.n = n
        this.ρ = ρ
        this.σ = σ = ρ == 1.0 ? Inf : ρ / (1.0 - ρ)
        c = isnothing(c) ? scale * sprand(Float64, n, m, sparsity) : c
        if bool_ensure_nz
            # ensure each row and column has at least one non-zero entry
            c = add_nonzero_entries!(c, n, m, scale)
        end
        if bool_force_dense
            c = Matrix(c)
        end
        this.c = copy(c)
        println("FisherMarket cost matrix initialized in $(time() - ts) seconds")
        this.uₛ = (x, i) -> c[:, i]' * x

        this.q = bool_unit ? ones(n) : m * rand(n) # in O(m)
        this.w = w = (isnothing(w) ? rand(m) : w) * scale
        this.w .= bool_unit_wealth ? w ./ sum(w) : w
        this.p = zeros(n)
        this.x = similar(c)
        this.sumx = zeros(n)
        # sometimes use bids instead of allocation
        this.b = similar(c)
        this.df = DataFrame()
        this.constr_x = constr_x
        this.constr_p = constr_p

        if ρ == 1.0
            # linear utility
            this.u = this.uₛ
            this.∇u = (x, i) -> c[:, i]
            this.f = (p, i) -> begin
                w[i] * maximum(c[:, i] ./ p)
            end
            this.f∇f = (p, i) -> begin
                f = w[i] * maximum(c[:, i] ./ p)
                ∇f = nothing # nonsmooth
                return f, ∇f
            end
        else
            # CES utility, ρ < 1
            this.u = (x, i) -> sum(c[:, i] .* spow.(x, ρ))^(1 / ρ)
            this.∇u = (x, i) -> sum(c[:, i] .* spow.(x, ρ))^(1 / ρ - 1) .*
                                spow.(x, ρ - 1) .* c[:, i]

            # the derivatives respect to p
            # ignore outer power and coeff
            # only compute
            # r := c^(1+σ)'p^(-σ)
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
        println("FisherMarket initialized in $(time() - ts) seconds")
        return this
    end
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
        :left_budget => w - x' * p,
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
    _excess = (sum(fisher.x; dims=2)[:] - fisher.q) ./ maximum(fisher.q)
    @printf(" :(normalized) market excess: [%.4e, %.4e]\n", minimum(_excess), maximum(_excess))
    @printf(" :            social welfare:  %.8e\n", (log.(fisher.val_u))' * fisher.w)
    println(__default_sep)
end


Base.copy(z::FisherMarket{T}) where {T} = begin
    this = FisherMarket(z.m, z.n)
    copy_fields(this, z)
    return this
end