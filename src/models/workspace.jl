# -----------------------------------------------------------------------
# AgentWorkspace: canonical per-agent storage for a Fisher market.
#
# Storage layout (insertion-order, unified):
#   - Universal per-agent arrays (`x, g, w, val_u, ε_br_play`) live at
#     the AgentWorkspace level, sized n×m / m, indexed by global
#     insertion order (`routing[i]` says which substore agent i came
#     from and at what substore-local index `j`).
#   - `ces::CESStore` keeps only CES-specific extras (`c, ρ, σ, s,
#     sumx`); its `x, g, w, val_u, ε_br_play` are SubArray VIEWS into
#     the parent ws arrays gathered at `ces_idx` (the global indices
#     of CES-tagged agents). Likewise for `gen::GenStore`.
#   - `q::V` is the market-wide supply (n).
#
# Substores can be interleaved in global order (e.g. routing =
# [(:gen,1), (:ces,1), (:gen,2)]), so `ces.x` / `gen.x` are non-
# contiguous views. They support per-column access, broadcast
# assignment, and indexed writes that propagate back to ws.
#
# The `FisherMarket` struct (src/models/fisher.jl) is a thin coordinator
# over an `AgentWorkspace`: per-agent fields (x, w, val_u, …) are
# exposed via `getproperty` shims that route to the workspace parent,
# while CES-only parameters (c, ρ, σ, s, sumx) still route to `ces`.
# -----------------------------------------------------------------------

using LinearAlgebra, SparseArrays, Random

# -----------------------------------------------------------------------
# Helper formerly in fisher.jl; cpu_workspace also needs it during
# initialization, so it lives next to the workspace constructors.
# -----------------------------------------------------------------------
"""
    add_nonzero_entries!(c, m, n, scale)

Ensure each row and each column of `c` has at least one nonzero entry.
"""
function add_nonzero_entries!(c, m, n, scale)
    for i in 1:m
        j = rand(1:n)
        c[i, j] += scale
    end
    for j in 1:n
        i = rand(1:m)
        c[i, j] += scale
    end
    return c
end

# -----------------------------------------------------------------------
# CESStore: CES-specific extras + views into the parent ws arrays
# -----------------------------------------------------------------------
"""
    CESStore{T, Tc, V}

CES-family substore. Holds CES-specific parameters (`c, ρ, σ`) and
CES-only iterates (`s, sumx`) directly. The universal per-agent arrays
(`x, g, w, val_u, ε_br_play`) are SubArray views into the parent
AgentWorkspace arrays, gathered at the global indices of CES-tagged
agents. The view fields are typed loosely (`AbstractArray{T,N}`) so
`_reslice!` can swap UnitRange-indexed views for Vector{Int}-indexed
views (and vice versa) without a type mismatch.

Type parameters:
- `T`:  scalar type (`Float64`).
- `Tc`: cost matrix type (`Matrix{T}` or `SparseMatrixCSC{T}`).
- `V`:  per-agent vector type (`Vector{T}` on CPU, `CuVector{T}` on GPU).
"""
mutable struct CESStore{T, Tc<:AbstractMatrix{T}, V<:AbstractVector{T}}
    n::Int
    m::Int
    # CES-specific parameters (owned)
    c::Tc           # n × m, cost matrix (may be sparse)
    ρ::V            # m
    σ::V            # m, σᵢ = ρᵢ/(1-ρᵢ) (+Inf when ρᵢ = 1)
    # CES-only iterates (owned)
    s::Matrix{T}    # n × m, dual slacks
    sumx::V         # n, aggregate demand
    # Views into parent AgentWorkspace arrays (gathered at ws.ces_idx)
    w::AbstractVector{T}            # m, view into ws.w
    x::AbstractMatrix{T}            # n × m, view into ws.x
    g::AbstractMatrix{T}            # n × m, view into ws.g
    val_u::AbstractVector{T}        # m, view into ws.val_u
    ε_br_play::AbstractVector{T}    # m, view into ws.ε_br_play
end

# -----------------------------------------------------------------------
# GenStore: typed heterogeneous agents + views into parent ws arrays
# -----------------------------------------------------------------------
"""
    GenStore{T, Tc, V}

Generic substore for non-CES agent families (QL, NN, future classes).
Holds a `Vector{AgentType}` of concrete agent instances. Per-agent
class-specific params live inside the typed `AgentType` instance (e.g.
`QuasiLinearLogAgent.c`). Universal per-agent arrays are SubArray views
into the parent AgentWorkspace, gathered at the global indices of
Gen-tagged agents.

Type parameters `Tc, V` are present for signature parity with
`CESStore`; `Tc` is unused inside GenStore. GenStore is CPU-only.
"""
mutable struct GenStore{T, Tc<:AbstractMatrix{T}, V<:AbstractVector{T}}
    n::Int
    m::Int
    # Typed agent instances (owned).
    agents::Vector{AgentType}
    # Views into parent AgentWorkspace arrays (gathered at ws.gen_idx)
    w::AbstractVector{T}            # m, view into ws.w
    x::AbstractMatrix{T}            # n × m, view into ws.x
    g::AbstractMatrix{T}            # n × m, view into ws.g
    val_u::AbstractVector{T}        # m, view into ws.val_u
    ε_br_play::AbstractVector{T}    # m, view into ws.ε_br_play
end

# -----------------------------------------------------------------------
# AgentWorkspace: umbrella container
# -----------------------------------------------------------------------
"""
    AgentWorkspace{T, Tc, V}

Top-level per-agent storage. Owns the universal per-agent arrays
(`x, g, w, val_u, ε_br_play`) and the market-wide supply `q`. Holds
`ces::CESStore` and `gen::GenStore` whose universal arrays are views
back into the umbrella here.

Insertion order is global: `routing[i] = (sub, j)` says agent i lives
at local index `j` of substore `sub`. `ces_idx[j]` and `gen_idx[j]`
are the inverse maps (substore-local index → global agent index) used
to gather the substore SubArrays.

`m == ces.m + gen.m == length(routing)`.
"""
mutable struct AgentWorkspace{T, Tc<:AbstractMatrix{T}, V<:AbstractVector{T}}
    n::Int
    m::Int
    # Universal per-agent storage (in global insertion order)
    w::V                # m
    x::Matrix{T}        # n × m
    g::Matrix{T}        # n × m
    val_u::V            # m
    ε_br_play::V        # m
    q::V                # n, total supply per good
    # Substores (universal arrays are views back into the fields above)
    ces::CESStore
    gen::GenStore
    # Per-agent routing in CG-insertion order. `routing[i] = (sub, j)`
    # says agent i lives at local index `j` of substore `sub`.
    routing::Vector{Tuple{Symbol,Int}}
    # Inverse maps: ces_idx[j] = global i of jᵗʰ CES agent. Rebuilt by
    # `_reslice!` whenever routing changes; consumed when re-viewing
    # substore universal arrays.
    ces_idx::Vector{Int}
    gen_idx::Vector{Int}
end

# -----------------------------------------------------------------------
# Reslice: rebuild substore views from routing
# -----------------------------------------------------------------------
"""
    _reslice!(ws::AgentWorkspace) -> ws

Rebuild `ws.ces_idx`, `ws.gen_idx`, and the SubArray view fields on
`ws.ces` / `ws.gen` from `ws.routing`. Must be called after any
structural mutation (insertion, prune, routing rewrite) so the
substore-local views point at the right columns/elements of the parent
arrays. Also re-sets `ws.ces.m`, `ws.gen.m`, `ws.m`.
"""
function _reslice!(ws::AgentWorkspace)
    m = length(ws.routing)
    @assert size(ws.x, 2) == m "ws.x columns ($(size(ws.x, 2))) ≠ length(routing) ($m)"
    # Walk routing to (a) count substore sizes, (b) gather the
    # inverse-index arrays. Each substore-local j must appear exactly
    # once in routing for its substore; we assume callers maintain
    # this invariant via add_ces!/add_gen!/_prune_*.
    ces_count = 0
    gen_count = 0
    for (_, (sub, _)) in pairs(ws.routing)
        sub === :ces ? (ces_count += 1) : (gen_count += 1)
    end
    ces_idx = Vector{Int}(undef, ces_count)
    gen_idx = Vector{Int}(undef, gen_count)
    for (i, (sub, j)) in pairs(ws.routing)
        sub === :ces ? (ces_idx[j] = i) : (gen_idx[j] = i)
    end
    ws.ces_idx = ces_idx
    ws.gen_idx = gen_idx
    ws.m = m
    # Rebuild substore views.
    ces = ws.ces
    ces.m = ces_count
    ces.w         = view(ws.w,         ces_idx)
    ces.x         = view(ws.x,         :, ces_idx)
    ces.g         = view(ws.g,         :, ces_idx)
    ces.val_u     = view(ws.val_u,     ces_idx)
    ces.ε_br_play = view(ws.ε_br_play, ces_idx)
    gen = ws.gen
    gen.m = gen_count
    gen.w         = view(ws.w,         gen_idx)
    gen.x         = view(ws.x,         :, gen_idx)
    gen.g         = view(ws.g,         :, gen_idx)
    gen.val_u     = view(ws.val_u,     gen_idx)
    gen.ε_br_play = view(ws.ε_br_play, gen_idx)
    return ws
end

# -----------------------------------------------------------------------
# Empty substore constructors
# -----------------------------------------------------------------------
# Build a zero-agent CES substore. Universal-array view fields are
# placeholders (empty views) until `_reslice!` is called.
function _empty_cesstore(::Type{T}, ::Type{Tc}, ::Type{V}, n::Int) where {T, Tc, V}
    empty_v = V(T[])
    empty_m = zeros(T, n, 0)
    empty_view_v = view(empty_v, 1:0)
    empty_view_m = view(empty_m, :, 1:0)
    return CESStore{T, Tc, V}(
        n, 0,
        # CES-specific (owned)
        Tc <: SparseMatrixCSC ? spzeros(T, n, 0) : zeros(T, n, 0),
        V(T[]),                 # ρ
        V(T[]),                 # σ
        zeros(T, n, 0),         # s
        V(zeros(T, n)),         # sumx
        # View placeholders (overwritten by _reslice!)
        empty_view_v, empty_view_m, empty_view_m,
        empty_view_v, empty_view_v,
    )
end

function _empty_genstore(::Type{T}, ::Type{Tc}, ::Type{V}, n::Int) where {T, Tc, V}
    empty_v = V(T[])
    empty_m = zeros(T, n, 0)
    empty_view_v = view(empty_v, 1:0)
    empty_view_m = view(empty_m, :, 1:0)
    return GenStore{T, Tc, V}(
        n, 0,
        AgentType[],
        # View placeholders (overwritten by _reslice!)
        empty_view_v, empty_view_m, empty_view_m,
        empty_view_v, empty_view_v,
    )
end

# -----------------------------------------------------------------------
# Empty AgentWorkspace constructor — building block for staged init
# -----------------------------------------------------------------------
"""
    AgentWorkspace(n::Int; T=Float64, Tc=SparseMatrixCSC{T,Int}, q=nothing) -> AgentWorkspace

Construct an empty workspace over `n` goods with no agents in either
substore. Allocates `q` (defaults to `ones(T, n)`) and zero-length
per-agent storage. Use `cpu_workspace!`, `add_ces!`, or `add_gen!` to
populate.
"""
function AgentWorkspace(n::Int;
    T::Type=Float64,
    Tc::Type=SparseMatrixCSC{T,Int},
    q::Union{AbstractVector,Nothing}=nothing,
)
    V = Vector{T}
    q_vec = isnothing(q) ? ones(T, n) : Vector{T}(q)
    @assert length(q_vec) == n "q length must equal n=$n"
    ces = _empty_cesstore(T, Tc, V, n)
    gen = _empty_genstore(T, Tc, V, n)
    ws = AgentWorkspace{T, Tc, V}(
        n, 0,
        V(T[]),             # w
        zeros(T, n, 0),     # x
        zeros(T, n, 0),     # g
        V(T[]),             # val_u
        V(T[]),             # ε_br_play
        q_vec,
        ces, gen,
        Tuple{Symbol,Int}[],
        Int[], Int[],
    )
    _reslice!(ws)
    return ws
end

# -----------------------------------------------------------------------
# CPU constructor — back-compat shim over AgentWorkspace + add_ces!
# -----------------------------------------------------------------------
"""
    cpu_workspace(n::Int, m::Int=0; kwargs...) -> AgentWorkspace

Build a CPU workspace with `m` CES agents populated. Thin convenience
wrapper that calls `AgentWorkspace(n)` followed by `cpu_workspace!`
(which dispatches to `add_ces!`). Kwargs match `add_ces!` and
`cpu_workspace!`; see those for details.
"""
function cpu_workspace(n::Int, m::Int=0;
    ρ=1.0,
    c=nothing,
    w=nothing,
    q=nothing,
    seed=1,
    scale=1.0,
    sparsity=(2.0 / n),
    bool_unit=true,
    bool_unit_wealth=true,
    bool_ensure_nz=false,
    bool_force_dense=false,
    ε_br_play=1e-8,
)
    T = Float64
    Random.seed!(seed)
    q_vec = isnothing(q) ? (bool_unit ? ones(T, n) : Float64(m) * rand(T, n)) : Vector{T}(q)
    ws = AgentWorkspace(n; T=T, Tc=SparseMatrixCSC{T,Int}, q=q_vec)
    cpu_workspace!(ws, m;
        ρ=ρ, c=c, w=w, scale=scale, sparsity=sparsity,
        bool_unit_wealth=bool_unit_wealth,
        bool_ensure_nz=bool_ensure_nz,
        bool_force_dense=bool_force_dense,
        ε_br_play=ε_br_play,
    )
    return ws
end

"""
    cpu_workspace!(ws::AgentWorkspace, m_ces::Int; kwargs...) -> ws

Populate the CES substore of an existing (empty) workspace with
`m_ces` CES agents. Thin wrapper over `add_ces!` that applies the
unit-wealth normalization convention used by `cpu_workspace`.

Kwargs (forwarded to `add_ces!` except `bool_unit_wealth`):
- `ρ, c, w, scale, sparsity, bool_ensure_nz, bool_force_dense, ε_br_play`
- `bool_unit_wealth`  normalize CES wealth to sum 1 (default `true`).
"""
function cpu_workspace!(ws::AgentWorkspace, m_ces::Int;
    ρ=1.0,
    c=nothing,
    w=nothing,
    scale=1.0,
    sparsity=(2.0 / ws.n),
    bool_unit_wealth=true,
    bool_ensure_nz=false,
    bool_force_dense=false,
    ε_br_play=1e-8,
)
    m_ces == 0 && return ws
    add_ces!(ws, m_ces;
        ρ=ρ, c=c, w=w, scale=scale, sparsity=sparsity,
        bool_ensure_nz=bool_ensure_nz,
        bool_force_dense=bool_force_dense,
        ε_br_play=ε_br_play,
    )
    if bool_unit_wealth
        ws.ces.w ./= sum(ws.ces.w)
    end
    return ws
end

"""
    gpu_workspace(n::Int, m::Int; device, kwargs...) -> AgentWorkspace

Build a GPU-backed workspace. `device` should be `cu` from CUDA.jl.
Constructs CPU-side then copies the universal per-agent arrays and the
CES parameters/iterates to the device. The CES cost matrix is
densified (sparse GPU formats are not supported by this storage). `gen`
remains CPU-only and is empty here (no path inserts Gen on GPU).
"""
function gpu_workspace(n::Int, m::Int; device, kwargs...)
    ws_cpu = cpu_workspace(n, m; kwargs...)
    T = Float64
    ces_cpu = ws_cpu.ces
    c_dense = Matrix{T}(ces_cpu.c)
    c_gpu = device(c_dense)
    V_gpu = typeof(device(ws_cpu.w))
    Tc_gpu = typeof(c_gpu)
    # Build empty GPU substores first (universal arrays are views back
    # into the umbrella, which we'll allocate on-device below).
    gen_gpu = _empty_genstore(T, Tc_gpu, V_gpu, n)
    # CESStore on-device with universal view placeholders; _reslice!
    # will populate them.
    empty_v = V_gpu(T[])
    empty_view_v = view(empty_v, 1:0)
    ces_gpu_empty = CESStore{T, Tc_gpu, V_gpu}(
        n, 0,
        c_gpu,
        device(ces_cpu.ρ), device(ces_cpu.σ),
        device(ces_cpu.s), device(ces_cpu.sumx),
        empty_view_v, view(c_gpu, :, 1:0), view(c_gpu, :, 1:0),
        empty_view_v, empty_view_v,
    )
    ws_gpu = AgentWorkspace{T, Tc_gpu, V_gpu}(
        n, m,
        device(ws_cpu.w),
        device(ws_cpu.x),
        device(ws_cpu.g),
        device(ws_cpu.val_u),
        device(ws_cpu.ε_br_play),
        device(ws_cpu.q),
        ces_gpu_empty, gen_gpu,
        copy(ws_cpu.routing),
        Int[], Int[],
    )
    # Set ces.m to the true count before reslicing so the views span
    # the populated columns.
    ces_gpu_empty.m = m
    _reslice!(ws_gpu)
    return ws_gpu
end

# -----------------------------------------------------------------------
# Incremental CES agent insertion
# -----------------------------------------------------------------------
"""
    add_ces!(ws::AgentWorkspace, m_add::Int; kwargs...) -> ws

Append `m_add` CES agents to `ws`. The CES-specific parameters
(`c, ρ, σ`) extend `ws.ces`; the universal per-agent fields
(`w, x, g, val_u, ε_br_play`) extend `ws.*` and the substore views are
rebuilt via `_reslice!`.

Kwargs (same conventions as `cpu_workspace`):
- `ρ`         CES parameter, scalar or m_add-vector (default 1.0).
- `c`         (n × m_add) cost matrix; default random sparse of `sparsity`.
- `w`         m_add wealth vector; default `rand(m_add) * scale`.
- `seed`      RNG seed; `nothing` (default) means do not re-seed.
- `scale`     scale on c and w (default 1.0).
- `sparsity`  fraction of nonzeros in default-random c (default `2/n`).
- `bool_ensure_nz`    ensure each row/col of c has ≥ 1 nonzero (default false).
- `bool_force_dense`  densify c (default false).
- `ε_br_play` per-agent best-response tolerance (default 1e-8).

Note: wealth is **not** auto-renormalized. After all desired agents are
added, the caller may do `ws.ces.w ./= sum(ws.ces.w)` if a unit-budget
convention is needed.
"""
function add_ces!(ws::AgentWorkspace{T, Tc, V}, m_add::Int;
    ρ=1.0,
    c=nothing,
    w=nothing,
    seed=nothing,
    scale=1.0,
    sparsity=(2.0 / ws.n),
    bool_ensure_nz=false,
    bool_force_dense=false,
    ε_br_play=1e-8,
) where {T, Tc, V}
    m_add == 0 && return ws
    !isnothing(seed) && Random.seed!(seed)
    n = ws.n

    _ρ, _ = normalize_rho_sigma(ρ, m_add)

    if isnothing(c)
        c = scale * sprand(T, n, m_add, sparsity)
    end
    if bool_ensure_nz
        c = add_nonzero_entries!(c, n, m_add, scale)
    end
    if bool_force_dense
        c = Matrix(c)
    end
    c_new = copy(c)
    @assert size(c_new) == (n, m_add) "c must be (n, m_add) = ($n, $m_add)"

    w_raw = isnothing(w) ? rand(T, m_add) : copy(w)
    w_raw = w_raw .* scale
    @assert length(w_raw) == m_add "w length must equal m_add=$m_add"

    ε_vec = isa(ε_br_play, Number) ? fill(T(ε_br_play), m_add) : Vector{T}(ε_br_play)
    @assert length(ε_vec) == m_add "ε_br_play length must equal m_add=$m_add"

    ces_m_before = ws.ces.m
    # Extend CES-specific parameters in ws.ces.
    _expand_ces_params!(ws.ces, ces_m_before + m_add;
        c_new=c_new, ρ_new=Vector{T}(_ρ),
    )
    # Extend universal per-agent arrays at the workspace level.
    _grow_ws_per_agent!(ws, m_add; w_new=w_raw, ε_new=ε_vec)
    # Update routing for the freshly inserted agents.
    for k in 1:m_add
        push!(ws.routing, (:ces, ces_m_before + k))
    end
    _reslice!(ws)
    return ws
end

# -----------------------------------------------------------------------
# Generic (non-CES) agent insertion
# -----------------------------------------------------------------------
"""
    add_gen!(ws::AgentWorkspace{T}, agent::AgentType, w_new::Real=0;
             ε_br_play::Real=1e-8) -> ws

Append one typed `AgentType` instance (e.g. `QuasiLinearLogAgent`) to
`ws.gen.agents`, growing the universal per-agent arrays at the
workspace level by one column / one element. Pushes a
`(:gen, ws.gen.m + 1)` entry to `ws.routing` so iterators that walk
all agents in CG-insertion order pick up the new agent.
"""
function add_gen!(ws::AgentWorkspace{T, Tc, V}, agent::AgentType,
    w_new::Real=zero(T);
    ε_br_play::Real=1e-8,
) where {T, Tc, V}
    push!(ws.gen.agents, agent)
    gen_local = ws.gen.m + 1
    _grow_ws_per_agent!(ws, 1;
        w_new=T[T(w_new)], ε_new=T[T(ε_br_play)],
    )
    push!(ws.routing, (:gen, gen_local))
    _reslice!(ws)
    return ws
end

# -----------------------------------------------------------------------
# Internal: grow universal per-agent arrays at the workspace level
# -----------------------------------------------------------------------
function _grow_ws_per_agent!(ws::AgentWorkspace{T}, m_add::Int;
    w_new::AbstractVector,
    ε_new::AbstractVector,
) where {T}
    @assert length(w_new) == m_add
    @assert length(ε_new) == m_add
    n = ws.n
    ws.w         = vcat(ws.w,         w_new)
    ws.x         = hcat(ws.x,         zeros(T, n, m_add))
    ws.g         = hcat(ws.g,         zeros(T, n, m_add))
    ws.val_u     = vcat(ws.val_u,     zeros(T, m_add))
    ws.ε_br_play = vcat(ws.ε_br_play, ε_new)
    return ws
end

# -----------------------------------------------------------------------
# Internal: extend CES-specific parameter arrays (c, ρ, σ, s, sumx)
# -----------------------------------------------------------------------
function _expand_ces_params!(store::CESStore{T}, m_new::Int;
    c_new::Union{AbstractMatrix{T},Nothing}=nothing,
    ρ_new::Union{AbstractVector{T},Nothing}=nothing,
) where {T}
    m_old = store.m
    n = store.n
    m_add = m_new - m_old
    @assert m_new >= m_old "m_new ($m_new) must be >= m ($m_old)"
    m_add == 0 && return store

    if isnothing(c_new)
        c_new = zeros(T, n, m_add)
    end
    @assert size(c_new) == (n, m_add) "c_new must be (n, m_add) = ($n, $m_add)"

    if isnothing(ρ_new)
        ρ_default = isempty(store.ρ) ? T(1.0) : store.ρ[end]
        ρ_new = fill(ρ_default, m_add)
    end
    @assert length(ρ_new) == m_add "ρ_new must have length $m_add"

    σ_new = ρ_new ./ (1 .- ρ_new)

    store.m = m_new
    store.c = hcat(store.c, c_new)
    store.ρ = vcat(store.ρ, ρ_new)
    store.σ = vcat(store.σ, σ_new)
    store.s = hcat(store.s, zeros(T, n, m_add))
    # sumx is market-wide (n-shaped) and not touched.
    return store
end

# -----------------------------------------------------------------------
# expand_ces!: public name kept for back-compat with expand_players!
# (now also grows the ws universal arrays and reslices).
# -----------------------------------------------------------------------
"""
    expand_ces!(ws::AgentWorkspace, m_new; c_new, ρ_new, w_new, ε_new) -> ws

Grow the CES substore from its current `m` to `m_new`, appending
agents. Mirrors `add_ces!` but takes pre-built `c_new, ρ_new, w_new,
ε_new` rather than building them from kwarg conventions.

Kept for compatibility with `expand_players!`.
"""
function expand_ces!(ws::AgentWorkspace{T}, m_new::Int;
    c_new::Union{AbstractMatrix{T},Nothing}=nothing,
    ρ_new::Union{AbstractVector{T},Nothing}=nothing,
    w_new::Union{AbstractVector{T},Nothing}=nothing,
    ε_new::Union{AbstractVector{T},Nothing}=nothing,
) where {T}
    m_old = ws.ces.m
    m_add = m_new - m_old
    @assert m_add >= 0 "m_new ($m_new) must be >= m ($m_old)"
    m_add == 0 && return ws
    if isnothing(w_new)
        w_new = zeros(T, m_add)
    end
    if isnothing(ε_new)
        ε_default = T(1e-8)
        if !isempty(ws.ε_br_play)
            ε_default = ws.ε_br_play[end]
        end
        ε_new = fill(ε_default, m_add)
    end
    _expand_ces_params!(ws.ces, m_new; c_new=c_new, ρ_new=ρ_new)
    _grow_ws_per_agent!(ws, m_add; w_new=w_new, ε_new=ε_new)
    for k in 1:m_add
        push!(ws.routing, (:ces, m_old + k))
    end
    _reslice!(ws)
    return ws
end

# Back-compat: `expand_ces!(store::CESStore, ...)` no longer makes
# sense since CES-only fields live separately from the universal
# arrays. Callers should pass the workspace.
function expand_ces!(::CESStore, args...; kwargs...)
    error("expand_ces! now takes an AgentWorkspace, not a CESStore (universal per-agent arrays live at the workspace level).")
end

# -----------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------
"""
    prune_workspace!(ws::AgentWorkspace; ces_keep, gen_keep) -> ws

Prune `ws` in one shot, retaining the CES agents at substore-local
indices `ces_keep` (in order) and the Gen agents at substore-local
indices `gen_keep` (in order). Rewrites CES-specific parameters, gen
agent list, workspace universal arrays (in their original global
order), and `routing`. Re-slices the substore views at the end.

`ces_keep` defaults to `1:ws.ces.m` (keep all CES); `gen_keep` defaults
to `1:ws.gen.m` (keep all gen).
"""
function prune_workspace!(ws::AgentWorkspace;
    ces_keep::AbstractVector{Int}=1:ws.ces.m,
    gen_keep::AbstractVector{Int}=1:ws.gen.m,
)
    ces_store = ws.ces
    gen_store = ws.gen
    @assert all(1 .<= ces_keep .<= ces_store.m) "ces_keep has out-of-range local indices"
    @assert all(1 .<= gen_keep .<= gen_store.m) "gen_keep has out-of-range local indices"

    # Build per-substore "old global index" arrays for survivors.
    old_ces_global = [ws.ces_idx[j] for j in ces_keep]
    old_gen_global = [ws.gen_idx[j] for j in gen_keep]
    # Global survivors in original insertion order.
    keep_global = sort!(vcat(old_ces_global, old_gen_global))

    # 1. Prune CES-specific (owned) arrays at local CES indices.
    ces_store.m = length(ces_keep)
    ces_store.c = ces_store.c[:, ces_keep]
    ces_store.ρ = ces_store.ρ[ces_keep]
    ces_store.σ = ces_store.σ[ces_keep]
    ces_store.s = ces_store.s[:, ces_keep]
    # sumx is n-shaped (market-wide); untouched.

    # 2. Prune gen agent list at local gen indices.
    gen_store.m = length(gen_keep)
    gen_store.agents = gen_store.agents[gen_keep]

    # 3. Prune workspace universal arrays by the global keep set.
    ws.w         = ws.w[keep_global]
    ws.x         = ws.x[:, keep_global]
    ws.g         = ws.g[:, keep_global]
    ws.val_u     = ws.val_u[keep_global]
    ws.ε_br_play = ws.ε_br_play[keep_global]

    # 4. Rebuild routing: each new global agent index points to its
    # NEW substore-local position. Build inverse maps from "old global
    # index" → new local position for each substore.
    ces_global_to_new_local = Dict{Int,Int}(
        old_global => new_local
        for (new_local, old_global) in enumerate(old_ces_global)
    )
    gen_global_to_new_local = Dict{Int,Int}(
        old_global => new_local
        for (new_local, old_global) in enumerate(old_gen_global)
    )
    new_routing = Vector{Tuple{Symbol,Int}}(undef, length(keep_global))
    for (new_i, old_global) in enumerate(keep_global)
        if haskey(ces_global_to_new_local, old_global)
            new_routing[new_i] = (:ces, ces_global_to_new_local[old_global])
        else
            new_routing[new_i] = (:gen, gen_global_to_new_local[old_global])
        end
    end
    ws.routing = new_routing

    _reslice!(ws)
    return ws
end

# Substore-typed back-compat shims. Callers that did
# `_prune_ces!(ws.ces, keep)` should switch to `prune_workspace!(ws;
# ces_keep=keep)`, but these wrappers preserve the original signature.
"""
    _prune_ces!(ws::AgentWorkspace, keep::Vector{Int}) -> ws.ces

Keep CES agents at substore-local indices `keep`; all gen agents are
retained. Wrapper around `prune_workspace!`.
"""
_prune_ces!(ws::AgentWorkspace, keep::Vector{Int}) =
    (prune_workspace!(ws; ces_keep=keep); ws.ces)

"""
    _prune_gen!(ws::AgentWorkspace, keep::Vector{Int}) -> ws.gen

Keep gen agents at substore-local indices `keep`; all CES agents are
retained. Wrapper around `prune_workspace!`.
"""
_prune_gen!(ws::AgentWorkspace, keep::Vector{Int}) =
    (prune_workspace!(ws; gen_keep=keep); ws.gen)

# Old substore-typed signatures are no longer meaningful (substores
# can't be pruned independently of the workspace's universal arrays and
# routing). Raise rather than silently corrupt state.
_prune_ces!(::CESStore, ::Vector{Int}) =
    error("_prune_ces! now takes an AgentWorkspace; call _prune_ces!(ws, keep) or prune_workspace!(ws; ces_keep=keep)")
_prune_gen!(::GenStore, ::Vector{Int}) =
    error("_prune_gen! now takes an AgentWorkspace; call _prune_gen!(ws, keep) or prune_workspace!(ws; gen_keep=keep)")

# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------
nagents(ws::AgentWorkspace) = ws.m
nagents(ws::AgentWorkspace, family::Symbol) =
    family === :ces ? ws.ces.m :
    family === :gen ? ws.gen.m : 0
