# -----------------------------------------------------------------------
# FisherMarket: thin coordinator over an AgentWorkspace.
#
# Storage layout (universal per-agent arrays live at the workspace
# level; substores hold class-specific extras + views back into ws):
#   - `fa.w, fa.x, fa.g, fa.val_u, fa.ε_br_play` → `fa.storage.*`
#     (covers BOTH CES and Gen agents, in global insertion order)
#   - `fa.c, fa.ρ, fa.σ, fa.s, fa.sumx` → `fa.storage.ces.*`
#     (CES-specific parameters / iterates only)
#   - `fa.q` → `fa.storage.q` (market-wide supply)
#
# Initialization is workspace-first: callers build the workspace via
# `cpu_workspace(n, m; ...)` (or the staged `AgentWorkspace(n)` +
# `cpu_workspace!(ws, m; ...)`) and then construct the market with
# `FisherMarket(ws)`.
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
    storage::AgentWorkspace                                   # owns per-agent data + q
    sparsity::Float64
    df::Union{DataFrame,Nothing}
    constr_p::Union{LinearConstr,Nothing}                    # price constraints
    agents::Vector                                            # AgentView registry (lazy)
end

# -----------------------------------------------------------------------
# Constructors
# -----------------------------------------------------------------------
"""
    FisherMarket(storage::AgentWorkspace; kwargs...) -> FisherMarket

Explicit constructor: caller has built the workspace via
`cpu_workspace(n, m; ...)` or `gpu_workspace(...)`.

Kwargs:
- `ε_br_play`  override per-agent tolerance (scalar or m-vector); when
                provided, copied into `storage.ε_br_play`.
- `constr_p`    optional price constraint.
- `sparsity`    informational; default 1.0.
- `df`          dataframe for logging; default `DataFrame()`.
"""
function FisherMarket(storage::AgentWorkspace;
    ε_br_play=nothing,
    constr_p::Union{LinearConstr,Nothing}=nothing,
    sparsity::Float64=1.0,
    df::Union{DataFrame,Nothing}=DataFrame(),
)
    T = Float64
    m = storage.m
    n = storage.n
    if !isnothing(ε_br_play)
        ε_vec = isa(ε_br_play, Number) ? fill(T(ε_br_play), m) : Vector{T}(ε_br_play)
        @assert length(ε_vec) == m "ε_br_play length must equal m=$m"
        storage.ε_br_play .= ε_vec
    end
    return FisherMarket{T}(
        m, n,
        zeros(T, n),
        storage,
        sparsity, df, constr_p,
        Vector{Any}(),
    )
end

# -----------------------------------------------------------------------
# Property shims:
#   - Universal per-agent fields (cover both CES and Gen) → ws.*
#   - CES-only parameters/iterates                        → ws.ces.*
#   - q (supply, market-wide)                             → ws.q
# -----------------------------------------------------------------------
@inline _is_universal_field(name::Symbol) =
    name === :w || name === :x || name === :g ||
    name === :val_u || name === :ε_br_play
@inline _is_ces_only_field(name::Symbol) =
    name === :c || name === :ρ || name === :σ ||
    name === :s || name === :sumx

function Base.getproperty(fa::FisherMarket, name::Symbol)
    if _is_universal_field(name)
        return getproperty(getfield(fa, :storage), name)
    elseif _is_ces_only_field(name)
        return getproperty(getfield(fa, :storage).ces, name)
    elseif name === :q
        return getfield(fa, :storage).q
    else
        return getfield(fa, name)
    end
end

function Base.setproperty!(fa::FisherMarket, name::Symbol, val)
    if _is_universal_field(name)
        setproperty!(getfield(fa, :storage), name, val)
    elseif _is_ces_only_field(name)
        setproperty!(getfield(fa, :storage).ces, name, val)
    elseif name === :q
        setproperty!(getfield(fa, :storage), :q, val)
    elseif name === :m
        setfield!(fa, :m, val)
        setproperty!(getfield(fa, :storage), :m, val)
        # Do NOT cascade into ws.ces.m / ws.gen.m: those are
        # maintained by add_ces!/add_gen!/prune_workspace! and writing
        # fa.m would desync them from the routing length.
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
`agents` is reset (AgentViews must be rebuilt against the new
workspace's arrays). The substore views are rebuilt via `_reslice!`
on the cloned workspace to guarantee they point at the cloned parent
arrays (deepcopy preserves the relationship via IdDict, but reslicing
is cheap and removes any subtle aliasing concerns).
"""
function Base.copy(z::FisherMarket{T}) where {T}
    new_ws = deepcopy(z.storage)
    _reslice!(new_ws)
    new_constr_p = isnothing(z.constr_p) ? nothing : deepcopy(z.constr_p)
    this = FisherMarket(new_ws;
        constr_p=new_constr_p,
        sparsity=z.sparsity, df=isnothing(z.df) ? nothing : copy(z.df),
    )
    this.p .= z.p
    return this
end

# -----------------------------------------------------------------------
# expand_players! — grow the market by appending CES agents.
# -----------------------------------------------------------------------
"""
    expand_players!(this::FisherMarket, m_new; c_new=nothing, ρ_new=nothing, w_new=nothing)

Append `m_new - this.m` CES agents in-place via `expand_ces!` on the
underlying workspace. `this.m` is updated and the AgentView registry
is cleared (views may have stale pointers after array reallocations).
"""
function expand_players!(this::FisherMarket{T}, m_new::Int;
    c_new::Union{Matrix{T},Nothing}=nothing,
    ρ_new::Union{Vector{T},Nothing}=nothing,
    w_new::Union{Vector{T},Nothing}=nothing,
) where {T}
    ws = this.storage
    m_add = m_new - this.m
    @assert m_add >= 0 "m_new ($m_new) must be >= m ($(this.m))"
    m_add == 0 && return this
    expand_ces!(ws, ws.ces.m + m_add;
        c_new=c_new, ρ_new=ρ_new, w_new=w_new,
    )
    setfield!(this, :m, ws.m)
    # Clear AgentView registry: views may have stale array pointers if
    # ws.x / ws.w were reallocated by hcat/vcat above.
    setfield!(this, :agents, Vector{Any}())
    return this
end
