# -----------------------------------------------------------------------
# run subproblems as best-response-type mappings
#   using proportional response (bids) updates
# -----------------------------------------------------------------------

@doc raw"""
    Proportional response update using bids.

    Recovers allocation from bids ``g`` and updates bids
    proportionally based on the CES utility structure.
    Handles both ρ ≥ 0 and ρ < 0 cases.
"""
function __bids_response(;
    i::Int=1,
    p::Vector{T}=nothing,
    market::Market=nothing,
    agent::Union{AgentView,Nothing}=nothing,
    kwargs...
) where {T}
    av = isnothing(agent) ? market.agents[i] : agent
    w = market.w[av.i]
    gᵢ = view(market.g, :, av.i)

    av.x .= gᵢ ./ p
    market.val_u[av.i] = utility(av)

    at = av.atype
    if at isa LinearAgent || (at isa CESAgent && at.ρ >= 0)
        ρᵢ = at isa LinearAgent ? 1.0 : at.ρ
        cs = av.c .* spow.(av.x, ρᵢ)
        sumcs = sum(cs)
        gᵢ .= w * cs ./ sumcs
    else
        ρᵢ = at.ρ
        cs = spow.(av.c ./ spow.(p, ρᵢ), 1 / (1 - ρᵢ))
        sumcs = sum(cs)
        gᵢ .= w * cs ./ sumcs
    end
    return nothing
end

Bids = ResponseOptimizer(
    __bids_response,
    :bids,
    "Bids"
)
