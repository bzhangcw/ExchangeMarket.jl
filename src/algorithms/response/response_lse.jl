# -----------------------------------------------------------------------
# LSE (logsumexp-smoothed) response for linear Fisher markets
#   softmax allocation, no inner optimization
#   temperature t from market.ε_br_play[1]
# -----------------------------------------------------------------------

@doc raw"""
    LSE response: entropy-regularized (softmax) allocation.

    Given linear utility u = ⟨c, x⟩, the LSE response at temperature t is:
        γᵢ = softmax(cᵢ / (t·p))
        xᵢ = wᵢ γᵢ / p

    The temperature parameter t is read from `market.ε_br_play[1]`.
"""
function __lse_response(;
    i::Int=1,
    p::Vector{T}=nothing,
    market::Market=nothing,
    agent::Union{AgentView,Nothing}=nothing,
    kwargs...
) where {T}
    av = isnothing(agent) ? market.agents[i] : agent
    t = market.ε_br_play[1]
    w = market.w[av.i]
    c = av.c
    z = c ./ p
    γ = __lse_softmax(z ./ t)
    av.x .= w .* γ ./ p
    market.val_u[av.i] = sparse_dot(c, av.x)
    return nothing
end

LSEResponse = ResponseOptimizer(
    __lse_response,
    :lse,
    "LSEResponse"
)
