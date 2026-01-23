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
    g::Matrix{T} # bids (rename from b to avoid endowment name clash)

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
    # CES parameters (per-agent)
    ρ::Vector{T}
    σ::Vector{T}

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
        # handle scalar or vector ρ with helper
        _ρ, _σ = normalize_rho_sigma(ρ, m)
        this.ρ = copy(_ρ)
        this.σ = copy(_σ)
        c = isnothing(c) ? scale * sprand(Float64, n, m, sparsity) : c
        if bool_ensure_nz
            # ensure each row and column has at least one non-zero entry
            c = add_nonzero_entries!(c, n, m, scale)
        end
        if bool_force_dense
            c = Matrix(c)
        end
        this.c = copy(c)
        @printf("FisherMarket cost matrix initialized in %.4f seconds\n", time() - ts)
        this.uₛ = (x, i) -> c[:, i]' * x

        this.q = bool_unit ? ones(n) : m * rand(n) # in O(m)
        this.w = w = (isnothing(w) ? rand(m) : w) * scale
        this.w .= bool_unit_wealth ? w ./ sum(w) : w
        this.p = zeros(n)
        this.x = similar(c)
        this.sumx = zeros(n)
        # sometimes use bids instead of allocation
        this.g = similar(c)
        this.df = DataFrame()
        this.constr_x = constr_x
        this.constr_p = constr_p

        # utility and indirect utility definitions supporting per-agent ρ/σ
        # NOTE: use this.c, this.w, this.ρ, this.σ to support expand_players!
        this.u = (x, i) -> begin
            ρᵢ = this.ρ[i]
            if ρᵢ == 1.0
                return this.uₛ(x, i)
            else
                return sum(this.c[:, i] .* spow.(x, ρᵢ))^(1 / ρᵢ)
            end
        end
        this.∇u = (x, i) -> begin
            ρᵢ = this.ρ[i]
            if ρᵢ == 1.0
                return this.c[:, i]
            else
                s = sum(this.c[:, i] .* spow.(x, ρᵢ))^(1 / ρᵢ - 1)
                return s .* spow.(x, ρᵢ - 1) .* this.c[:, i]
            end
        end
        # f and f∇f with per-agent σ
        this.f = (p, i) -> begin
            ρᵢ = this.ρ[i]
            if ρᵢ == 1.0
                return this.w[i] * maximum(this.c[:, i] ./ p)
            else
                σᵢ = this.σ[i]
                _cs = spow.(this.c[:, i], (1 + σᵢ))
                _ps = spow.(p, (-σᵢ))
                return _cs' * _ps
            end
        end
        this.f∇f = (p, i) -> begin
            ρᵢ = this.ρ[i]
            if ρᵢ == 1.0
                # return dummy smooth pieces for linear (not used in linear branches)
                f = this.w[i] * maximum(this.c[:, i] ./ p)
                n = length(p)
                return f, zeros(n), zeros(n)
            else
                σᵢ = this.σ[i]
                _ci = this.c[:, i]
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
        @printf("FisherMarket initialized in %.4f seconds\n", time() - ts)
        return this
    end
end

Base.copy(z::FisherMarket{T}) where {T} = begin
    this = FisherMarket(z.m, z.n)
    copy_fields(this, z)
    return this
end

"""
    expand_players!(this::FisherMarket, m_new; c_new=nothing, ρ_new=nothing, w_new=nothing)

Expand a FisherMarket from m to m_new players in-place.
Reallocates arrays to accommodate new players.

Arguments:
- this: FisherMarket to expand (modified in-place)
- m_new: New number of players (must be >= this.m)
- c_new: Coefficients for new players (n × (m_new - m)), default random
- ρ_new: CES parameters for new players ((m_new - m)-vector), default same as this.ρ[1]
- w_new: Budgets for new players ((m_new - m)-vector), default equal share

Returns this (modified).
"""
function expand_players!(this::FisherMarket{T}, m_new::Int;
    c_new::Union{Matrix{T},Nothing}=nothing,
    ρ_new::Union{Vector{T},Nothing}=nothing,
    w_new::Union{Vector{T},Nothing}=nothing
) where T
    m_old = this.m
    n = this.n
    m_add = m_new - m_old

    @assert m_new >= m_old "m_new ($m_new) must be >= m ($m_old)"

    if m_add == 0
        return this
    end

    # Default new coefficients: zero
    if isnothing(c_new)
        c_new = zeros(T, n, m_add)
    end
    @assert size(c_new) == (n, m_add) "c_new must be (n, m_add) = ($n, $m_add)"

    # Default new ρ: same as first player
    if isnothing(ρ_new)
        ρ_new = fill(this.ρ[1], m_add)
    end
    @assert length(ρ_new) == m_add "ρ_new must have length $m_add"

    # Default new budgets: zero
    if isnothing(w_new)
        w_new = zeros(T, m_add)
    end
    @assert length(w_new) == m_add "w_new must have length $m_add"

    # Compute σ_new from ρ_new
    σ_new = ρ_new ./ (1 .- ρ_new)

    # Update m
    this.m = m_new

    # Expand per-agent parameter arrays
    this.c = hcat(this.c, c_new)
    this.ρ = vcat(this.ρ, ρ_new)
    this.σ = vcat(this.σ, σ_new)
    this.w = vcat(this.w, w_new)

    # Expand allocation arrays (n × m)
    this.x = hcat(this.x, zeros(T, n, m_add))
    this.g = hcat(this.g, zeros(T, n, m_add))
    this.val_∇u = hcat(this.val_∇u, zeros(T, n, m_add))
    this.val_∇f = hcat(this.val_∇f, zeros(T, n, m_add))
    this.val_Hf = hcat(this.val_Hf, zeros(T, n, m_add))

    # Expand per-agent value arrays (m)
    this.val_u = vcat(this.val_u, zeros(T, m_add))
    this.val_f = vcat(this.val_f, zeros(T, m_add))

    return this
end