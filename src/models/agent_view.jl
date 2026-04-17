# -----------------------------------------------------------------------
# AgentView: zero-copy per-agent data slice with type dispatch
#   Agent type definitions live in models/agent_types.jl.
#   Utility evaluation methods live in the corresponding response files.
# -----------------------------------------------------------------------
"""
    AgentView{T, Vc, AT}

Zero-copy view of agent `i`'s data from a Market.

- `AT <: AgentType`: determines utility type (LinearAgent or CESAgent)
- `Vc`: cost vector type (SparseColRef for sparse c, SubArray for dense c)
- `x`, `s`: writable SubArray views into market arrays

Mutable scalars (`w`, `ε`) are NOT cached — access from market at use-time.
The agent's job: given prices `p`, compute allocation `x`. Then `utility(agent)` gives u(x).
"""
struct AgentView{T, Vc<:AbstractVector{T}, AT<:AgentType}
    i::Int             # agent index
    n::Int             # number of goods
    atype::AT          # agent type
    c::Vc              # cost column view (sparse-aware, zero-copy)
    x::SubArray{T,1}   # allocation column view (writable)
    s::SubArray{T,1}   # dual slack column view (writable)
end

"""
    AgentView(market::FisherMarket, i::Int) -> AgentView

Construct a zero-copy AgentView for agent `i`.
Agent type is derived from `market.ρ[i]`.
"""
function AgentView(market::FisherMarket{T}, i::Int) where {T}
    at = agent_type(market.ρ[i], market.σ[i])
    AgentView(
        i, market.n, at,
        sparse_col_ref(market.c, i),
        view(market.x, :, i),
        view(market.s, :, i),
    )
end

"""Compute utility u(xᵢ) for this agent."""
@inline utility(a::AgentView) = utility(a.atype, a.c, a.x)

# -----------------------------------------------------------------------
# Initialize agents vector in market
# -----------------------------------------------------------------------
"""
    init_agents!(market::FisherMarket)

Pre-construct AgentView objects for all agents and store in `market.agents`.
Called once at market creation; views remain valid as long as market arrays
are not reallocated.
"""
function init_agents!(market::FisherMarket{T}) where {T}
    market.agents = [AgentView(market, i) for i in 1:market.m]
    return market
end
