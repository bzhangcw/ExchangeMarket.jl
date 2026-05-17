# -----------------------------------------------------------------------
# FisherMarket: thin coordinator over a HeterogeneousWorkspace.
#
# Per-agent data (c, ρ, σ, w, x, g, s, sumx, val_u, ε_br_play) lives in
# the workspace's CES substore (`fa.storage.ces`). Market-wide supply
# `q` lives in `fa.storage.q`. The struct exposes per-agent fields via
# `Base.getproperty` / `Base.setproperty!` shims so out-of-tree callers
# that read `fa.c[:, i]` or `fa.σ[i]` keep working.
#
# Initialization is workspace-first: callers build the workspace via
# `cpu_workspace(n, m; ...)` (or `gpu_workspace`) and then construct the
# market with `FisherMarket(ws)`. There is no convenience `(m, n; ...)`
# constructor — see the migration note in toy_fisher below.
#
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date:   2024/11/22  (workspace refactor: 2026/05)
# -----------------------------------------------------------------------
__precompile__(true)

using LinearAlgebra, SparseArrays
using JuMP, Random
import MathOptInterface as MOI
using Printf, DataFrames

# `toy_fisher` is a tiny fixed test instance kept for the test suite.
function toy_fisher(ρ)
    m = 2
    n = 3
    c = [
        0.7337 6.9883 9.1493
        3.4924 6.2826 1.9281
    ]
    w = [0.1677; 0.5711]
    ws = cpu_workspace(n)
    add_ces!(ws, m; ρ=ρ, c=c, w=w)
    return FisherMarket(ws)
end

# -----------------------------------------------------------------------
# Struct
# -----------------------------------------------------------------------
mutable struct FisherMarket{T} <: AbstractMarket
    m::Int
    n::Int
    p::Vector{T}                                              # price (market-wide)
    storage::HeterogeneousWorkspace                           # owns per-agent data + q
    sparsity::Float64
    df::Union{DataFrame, Nothing}
    constr_x::Union{Vector{LinearConstr}, Nothing}            # allocation constraints
    constr_p::Union{LinearConstr, Nothing}                    # price constraints
    agents::Vector                                            # AgentView registry (lazy)
    workspace::Any                                            # batched compute handle (nothing = per-agent path)
    gpu_workspace_cache::Any                                  # cached GPU MarketWorkspace
end

# -----------------------------------------------------------------------
# Constructors
# -----------------------------------------------------------------------
"""
    FisherMarket(storage::HeterogeneousWorkspace; kwargs...) -> FisherMarket

Explicit constructor: caller has built the workspace via
`cpu_workspace(n, m; ...)` or `gpu_workspace(...)`.

Kwargs:
- `ε_br_play`  override per-agent tolerance (scalar or m-vector); when
                provided, copied into `storage.ces.ε_br_play`.
- `constr_x`, `constr_p`  optional allocation / price constraints.
- `sparsity`    informational; default 1.0.
- `df`          dataframe for logging; default `DataFrame()`.
"""
function FisherMarket(storage::HeterogeneousWorkspace;
    ε_br_play=nothing,
    constr_x::Union{Vector{LinearConstr}, Nothing}=nothing,
    constr_p::Union{LinearConstr, Nothing}=nothing,
    sparsity::Float64=1.0,
    df::Union{DataFrame, Nothing}=DataFrame(),
)
    T = Float64
    m = storage.m
    n = storage.n
    if !isnothing(ε_br_play)
        ε_vec = isa(ε_br_play, Number) ? fill(T(ε_br_play), m) : Vector{T}(ε_br_play)
        @assert length(ε_vec) == m "ε_br_play length must equal m=$m"
        storage.ces.ε_br_play .= ε_vec
    end
    return FisherMarket{T}(
        m, n,
        zeros(T, n),
        storage,
        sparsity, df, constr_x, constr_p,
        Vector{Any}(),
        nothing, nothing,
    )
end

# -----------------------------------------------------------------------
# Backward-compat property shims: route per-agent reads/writes into the
# CES substore (or `q` into the workspace top level).
# -----------------------------------------------------------------------
@inline _is_ces_field(name::Symbol) =
    name === :c || name === :ρ || name === :σ || name === :w ||
    name === :x || name === :g || name === :s || name === :sumx ||
    name === :val_u || name === :ε_br_play

function Base.getproperty(fa::FisherMarket, name::Symbol)
    if _is_ces_field(name)
        return getproperty(getfield(fa, :storage).ces, name)
    elseif name === :q
        return getfield(fa, :storage).q
    else
        return getfield(fa, name)
    end
end

function Base.setproperty!(fa::FisherMarket, name::Symbol, val)
    if _is_ces_field(name)
        setproperty!(getfield(fa, :storage).ces, name, val)
    elseif name === :q
        setproperty!(getfield(fa, :storage), :q, val)
    elseif name === :m
        setfield!(fa, :m, val)
        setproperty!(getfield(fa, :storage), :m, val)
        setproperty!(getfield(fa, :storage).ces, :m, val)
    elseif name === :n
        setfield!(fa, :n, val)
        setproperty!(getfield(fa, :storage), :n, val)
        setproperty!(getfield(fa, :storage).ces, :n, val)
    else
        setfield!(fa, name, val)
    end
    return val
end

# `propertynames` advertises both the struct fields and the shimmed
# per-agent fields, so `dump(fa)` and tab-completion still surface the
# familiar names.
function Base.propertynames(fa::FisherMarket, private::Bool=false)
    base = fieldnames(typeof(fa))
    return (base..., :c, :ρ, :σ, :w, :x, :g, :s, :sumx, :val_u, :ε_br_play, :q)
end

# -----------------------------------------------------------------------
# Copy
# -----------------------------------------------------------------------
"""
    Base.copy(fa::FisherMarket) -> FisherMarket

Deep-copy the storage workspace and clone the FisherMarket coordinator.
`agents` is reset (views must be rebuilt against the new workspace); the
GPU cache is dropped.
"""
function Base.copy(z::FisherMarket{T}) where {T}
    new_ws = deepcopy(z.storage)
    new_constr_x = isnothing(z.constr_x) ? nothing : deepcopy(z.constr_x)
    new_constr_p = isnothing(z.constr_p) ? nothing : deepcopy(z.constr_p)
    this = FisherMarket(new_ws;
        constr_x=new_constr_x, constr_p=new_constr_p,
        sparsity=z.sparsity, df=isnothing(z.df) ? nothing : copy(z.df),
    )
    this.p .= z.p
    return this
end

# -----------------------------------------------------------------------
# expand_players! — grow the market by appending CES agents.
# Delegates to expand_ces! on the substore.
# -----------------------------------------------------------------------
"""
    expand_players!(this::FisherMarket, m_new; c_new=nothing, ρ_new=nothing, w_new=nothing)

Append `m_new - this.m` CES agents in-place. Per-agent arrays are grown
inside `this.storage.ces` via `expand_ces!`; `this.m` is updated.
"""
function expand_players!(this::FisherMarket{T}, m_new::Int;
    c_new::Union{Matrix{T},Nothing}=nothing,
    ρ_new::Union{Vector{T},Nothing}=nothing,
    w_new::Union{Vector{T},Nothing}=nothing,
) where {T}
    m_old = this.m
    @assert m_new >= m_old "m_new ($m_new) must be >= m ($m_old)"
    m_add = m_new - m_old
    m_add == 0 && return this
    expand_ces!(this.storage.ces, m_new;
        c_new=c_new, ρ_new=ρ_new, w_new=w_new,
    )
    setproperty!(this.storage, :m, m_new)
    setfield!(this, :m, m_new)
    # Clear AgentView registry: views may have stale array pointers if
    # arrays were reallocated by hcat/vcat above.
    setfield!(this, :agents, Vector{Any}())
    return this
end
