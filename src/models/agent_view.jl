# -----------------------------------------------------------------------
# Agent type system and AgentView
#   Each agent has a type (LinearAgent, CESAgent) that determines
#   how to compute allocation x from prices p.
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Agent type hierarchy
# -----------------------------------------------------------------------
abstract type AgentType end

"""
    LinearAgent <: AgentType

Linear utility: u(x) = ⟨c, x⟩.
"""
struct LinearAgent <: AgentType end

"""
    CESAgent <: AgentType

CES utility: u(x) = (Σⱼ cⱼ xⱼ^ρ)^(1/ρ), with ρ < 1, σ = ρ/(1-ρ).
"""
struct CESAgent <: AgentType
    ρ::Float64
    σ::Float64
end

"""
    agent_type(ρ, σ) -> AgentType

Derive the agent type from CES parameter ρ.
"""
@inline function agent_type(ρ::Float64, σ::Float64)
    ρ == 1.0 ? LinearAgent() : CESAgent(ρ, σ)
end

# -----------------------------------------------------------------------
# Utility: the only function agents need to report
#   Given x (already computed by the response function), return u(x).
# -----------------------------------------------------------------------
@inline utility(::LinearAgent, c, x) = sparse_dot(c, x)

@inline function utility(at::CESAgent, c, x)
    s = 0.0
    foreach_nz(c) do j, cj
        s += cj * spow(x[j], at.ρ)
    end
    return spow(s, 1.0 / at.ρ)
end

# -----------------------------------------------------------------------
# AgentView: zero-copy per-agent data slice with type dispatch
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
