# -----------------------------------------------------------------------
# HeterogeneousWorkspace: canonical per-agent storage for a Fisher market.
#
# This file defines a storage layout that owns per-agent data, grouped
# by agent family. Currently only a CES substore is implemented. Other
# families (Leontief, PLC, QL) will be added as Union{Nothing,...}
# substore fields here in follow-up passes.
#
# The `FisherMarket` struct (src/models/fisher.jl) becomes a thin
# coordinator over a `HeterogeneousWorkspace`: per-agent fields
# (c, ρ, σ, x, ...) are exposed through `getproperty` shims that route
# into the appropriate substore, so out-of-tree callers keep working.
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
# CESStore: contiguous storage for all CES-family agents
# -----------------------------------------------------------------------
"""
    CESStore{T, Tc, V}

CES-family substore. Holds per-agent CES data laid out contiguously so
batched ops (CPU dense / GPU) can broadcast across all CES agents at
once and the AgentView-per-agent path can take zero-copy column views.

Type parameters:
- `T`:  scalar type (`Float64`).
- `Tc`: cost matrix type (`Matrix{T}` or `SparseMatrixCSC{T}`).
- `V`:  per-agent vector type (`Vector{T}` on CPU, `CuVector{T}` on GPU).
"""
mutable struct CESStore{T, Tc<:AbstractMatrix{T}, V<:AbstractVector{T}}
    n::Int
    m::Int
    # CES parameters
    c::Tc           # n × m, cost matrix (may be sparse)
    ρ::V            # m
    σ::V            # m, σᵢ = ρᵢ/(1-ρᵢ) (+Inf when ρᵢ = 1)
    w::V            # m, wealth
    # Per-agent iterates
    x::Matrix{T}    # n × m, allocation
    g::Matrix{T}    # n × m, bids
    s::Matrix{T}    # n × m, dual slacks
    sumx::V         # n, aggregate demand
    val_u::V        # m, utility values
    # Per-agent numerical tolerance
    ε_br_play::V    # m
end

# -----------------------------------------------------------------------
# HeterogeneousWorkspace: umbrella container
# -----------------------------------------------------------------------
"""
    HeterogeneousWorkspace{T, Tc, V}

Top-level per-agent storage. Holds one substore per agent family. In
this pass only `ces::CESStore` is populated; reserved slots for other
families will be added (initially as `Union{Nothing, ...}`) in follow-up
passes. The market-wide supply `q` lives here too.
"""
mutable struct HeterogeneousWorkspace{T, Tc<:AbstractMatrix{T}, V<:AbstractVector{T}}
    n::Int
    m::Int
    ces::CESStore{T, Tc, V}
    q::V                # n, total supply per good
end

# -----------------------------------------------------------------------
# CPU constructor
# -----------------------------------------------------------------------
"""
    cpu_workspace(n::Int, m::Int; kwargs...) -> HeterogeneousWorkspace

Build a CPU workspace from scratch with all data owned (not aliased).
Kwargs shape the cost matrix / wealth / supply built inside the substore.

- `ρ`            CES parameter, scalar or m-vector (default 1.0).
- `c`            cost matrix; default random sparse of given sparsity.
- `w`            wealth vector; default random.
- `q`            supply vector; default `ones(n)` (when `bool_unit=true`).
- `seed`         RNG seed (default 1).
- `scale`        scale factor on c and w (default 1.0).
- `sparsity`     fraction of nonzeros in default-random c (default `2/n`).
- `bool_unit`    use unit supplies q = `ones(n)` (default true).
- `bool_unit_wealth`  normalize wealth to sum 1 (default true).
- `bool_ensure_nz`    ensure each row/col of c has ≥ 1 nonzero (default false).
- `bool_force_dense`  densify c (default false).
- `ε_br_play`    per-agent best-response tolerance (scalar or m-vector; default 1e-8).
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
    Random.seed!(seed)
    T = Float64

    _ρ, _σ = normalize_rho_sigma(ρ, m)

    if isnothing(c)
        c = scale * sprand(T, n, m, sparsity)
    end
    if bool_ensure_nz
        c = add_nonzero_entries!(c, n, m, scale)
    end
    if bool_force_dense
        c = Matrix(c)
    end
    c_owned = copy(c)

    q_vec = isnothing(q) ? (bool_unit ? ones(T, n) : m * rand(T, n)) : Vector{T}(q)

    w_raw = isnothing(w) ? rand(T, m) : copy(w)
    w_raw = w_raw .* scale
    w_vec = bool_unit_wealth ? (w_raw ./ sum(w_raw)) : w_raw

    ε_vec = isa(ε_br_play, Number) ? fill(T(ε_br_play), m) : Vector{T}(ε_br_play)
    @assert length(ε_vec) == m "ε_br_play length must equal m=$m"

    ces = CESStore{T, typeof(c_owned), Vector{T}}(
        n, m,
        c_owned, Vector{T}(_ρ), Vector{T}(_σ), w_vec,
        zeros(T, n, m), zeros(T, n, m), zeros(T, n, m),
        zeros(T, n), zeros(T, m),
        ε_vec,
    )
    return HeterogeneousWorkspace{T, typeof(c_owned), Vector{T}}(n, m, ces, q_vec)
end

"""
    gpu_workspace(n::Int, m::Int; device, kwargs...) -> HeterogeneousWorkspace

Build a GPU workspace. `device` should be `cu` from CUDA.jl. The cost
matrix and per-agent arrays are constructed CPU-side then copied to the
device. The GPU substore stores `c` densified — sparse GPU formats
aren't supported by `_play_batched!`.
"""
function gpu_workspace(n::Int, m::Int; device, kwargs...)
    ws_cpu = cpu_workspace(n, m; kwargs...)
    ces_cpu = ws_cpu.ces
    c_dense = Matrix{Float64}(ces_cpu.c)
    c_gpu = device(c_dense)
    V_gpu = typeof(device(ces_cpu.w))
    ces_gpu = CESStore{Float64, typeof(c_gpu), V_gpu}(
        n, m,
        c_gpu,
        device(ces_cpu.ρ), device(ces_cpu.σ), device(ces_cpu.w),
        device(ces_cpu.x), device(ces_cpu.g), device(ces_cpu.s),
        device(ces_cpu.sumx), device(ces_cpu.val_u),
        device(ces_cpu.ε_br_play),
    )
    q_gpu = device(ws_cpu.q)
    return HeterogeneousWorkspace{Float64, typeof(c_gpu), V_gpu}(n, m, ces_gpu, q_gpu)
end

# -----------------------------------------------------------------------
# Incremental agent insertion
# -----------------------------------------------------------------------
"""
    add_ces!(ws::HeterogeneousWorkspace, m_add::Int; kwargs...) -> ws

Append `m_add` CES agents to `ws.ces`. Cost columns, wealth, and
tolerance are built from the kwargs (same conventions as
`cpu_workspace`) and forwarded to `expand_ces!`. `ws.m` is incremented.

Kwargs:
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
function add_ces!(ws::HeterogeneousWorkspace{T, Tc, V}, m_add::Int;
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

    m_new = ws.ces.m + m_add
    expand_ces!(ws.ces, m_new;
        c_new=c_new, ρ_new=Vector{T}(_ρ), w_new=w_raw, ε_new=ε_vec,
    )
    setproperty!(ws, :m, m_new)
    return ws
end

# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------
nagents(ws::HeterogeneousWorkspace) = ws.m
nagents(ws::HeterogeneousWorkspace, family::Symbol) =
    family === :ces ? ws.ces.m : 0

"""
    expand_ces!(store::CESStore{T}, m_new; c_new, ρ_new, w_new, ε_new)

Grow a CESStore from its current `m` to `m_new`, appending agents.
Used by `expand_players!` on FisherMarket; kept here so substore growth
logic lives alongside the substore.
"""
function expand_ces!(store::CESStore{T}, m_new::Int;
    c_new::Union{AbstractMatrix{T},Nothing}=nothing,
    ρ_new::Union{AbstractVector{T},Nothing}=nothing,
    w_new::Union{AbstractVector{T},Nothing}=nothing,
    ε_new::Union{AbstractVector{T},Nothing}=nothing,
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
        ρ_new = fill(store.ρ[1], m_add)
    end
    @assert length(ρ_new) == m_add "ρ_new must have length $m_add"

    if isnothing(w_new)
        w_new = zeros(T, m_add)
    end
    @assert length(w_new) == m_add "w_new must have length $m_add"

    if isnothing(ε_new)
        ε_default = isempty(store.ε_br_play) ? T(1e-8) : store.ε_br_play[end]
        ε_new = fill(ε_default, m_add)
    end
    @assert length(ε_new) == m_add "ε_new must have length $m_add"

    σ_new = ρ_new ./ (1 .- ρ_new)

    store.m = m_new
    store.c = hcat(store.c, c_new)
    store.ρ = vcat(store.ρ, ρ_new)
    store.σ = vcat(store.σ, σ_new)
    store.w = vcat(store.w, w_new)
    store.x = hcat(store.x, zeros(T, n, m_add))
    store.g = hcat(store.g, zeros(T, n, m_add))
    store.s = hcat(store.s, zeros(T, n, m_add))
    store.val_u = vcat(store.val_u, zeros(T, m_add))
    store.ε_br_play = vcat(store.ε_br_play, ε_new)
    return store
end
