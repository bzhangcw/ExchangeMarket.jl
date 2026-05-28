# -----------------------------------------------------------------------
# AgentView: zero-copy per-agent data slice with type dispatch
#   Agent type definitions live in models/agent_types.jl.
#   Utility evaluation methods live in the corresponding response files.
# -----------------------------------------------------------------------
"""
    AgentView{T, Vc, AT, Vs}

Zero-copy view of agent `i`'s data from a Market. `i` is the GLOBAL
agent index (covers both CES and Gen substores in insertion order).

- `AT <: AgentType` — utility / response dispatch type (LinearAgent,
  CESAgent, PLCAgent, QuasiLinearLogAgent, …).
- `Vc` — cost vector type. For CES: `SparseColRef` (sparse-aware) or a
  `SubArray` (dense). For Gen agents whose typed instance owns its own
  coefficient vector (e.g. `QuasiLinearLogAgent.c`), this points at
  the agent-owned vector. The convention is "whatever the response /
  utility code wants for this agent class."
- `x` — writable n-long view into `ws.x[:, i]`.
- `s` — CES-only dual-slack view into `ws.ces.s[:, j_ces]` where
  `j_ces` is the substore-local index. `Nothing` for Gen agents (the
  CES-only response paths that read `s` never run on Gen agents).

Mutable scalars (`w`, `ε`) are NOT cached — access from market at
use-time. The agent's job: given prices `p`, compute allocation `x`;
then `utility(agent)` gives u(x).
"""
struct AgentView{T, Vc<:AbstractVector{T}, AT<:AgentType, Vs}
    i::Int             # global agent index
    n::Int             # number of goods
    atype::AT          # agent type
    c::Vc              # cost vector (CES column ref or agent-owned)
    x::SubArray{T,1}   # allocation column view into ws.x
    s::Vs              # SubArray{T,1} for CES; Nothing for Gen
end

"""
    AgentView(market::FisherMarket, i::Int) -> AgentView

Construct a zero-copy AgentView for agent `i` (global index). Routes
through `market.storage.routing[i]` to determine which substore owns
the agent and pulls the class-specific data accordingly:

- `(:ces, j)` → CES agent: `atype = CESAgent/LinearAgent` from
  `ces.ρ[j], ces.σ[j]`, `c = sparse_col_ref(ces.c, j)`,
  `s = view(ces.s, :, j)`.
- `(:gen, j)` → Gen agent: `atype = ws.gen.agents[j]` directly,
  `c = atype.c` (or empty if the agent class has no `.c`),
  `s = nothing`.

The `x` view always points into the unified `ws.x[:, i]` regardless
of substore.
"""
function AgentView(market::FisherMarket{T}, i::Int) where {T}
    ws = getfield(market, :storage)
    @assert 1 <= i <= ws.m "agent index $i out of range 1:$(ws.m)"
    sub, j = ws.routing[i]
    x_view = view(ws.x, :, i)
    if sub === :ces
        ces = ws.ces
        at = agent_type(ces.ρ[j], ces.σ[j])
        c_ref = sparse_col_ref(ces.c, j)
        s_view = view(ces.s, :, j)
        return AgentView{T, typeof(c_ref), typeof(at), typeof(s_view)}(
            i, market.n, at, c_ref, x_view, s_view,
        )
    else  # :gen
        at = ws.gen.agents[j]
        # Gen agents' cost vector is owned by the typed agent (when it
        # has one). QuasiLinearLogAgent has `.c::Vector{Float64}`;
        # other Gen classes that don't carry a per-good `c` (e.g. PLC,
        # which uses an `a` matrix) can pass an empty Vector{T} — the
        # corresponding response code reads `av.atype.*` directly.
        c_ref = hasproperty(at, :c) ? getproperty(at, :c) : T[]
        return AgentView{T, typeof(c_ref), typeof(at), Nothing}(
            i, market.n, at, c_ref, x_view, nothing,
        )
    end
end

"""Compute utility u(xᵢ) for this agent."""
@inline utility(a::AgentView) = utility(a.atype, a.c, a.x)

# -----------------------------------------------------------------------
# Initialize agents vector in market
# -----------------------------------------------------------------------
"""
    init_agents!(market::FisherMarket)

Pre-construct AgentView objects for ALL agents (CES + Gen, in global
insertion order) and store in `market.agents`. Called once at market
creation; views remain valid as long as the workspace's universal
arrays (`ws.x`, etc.) and the CES `s` matrix are not reallocated.

`market.agents` is `Vector{Any}` because individual AgentViews are
parameterized differently across CES/Gen (different `AT`, `Vc`, `Vs`).
"""
function init_agents!(market::FisherMarket{T}) where {T}
    market.agents = [AgentView(market, i) for i in 1:market.m]
    return market
end
