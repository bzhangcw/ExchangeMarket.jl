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
    kwargs...
) where {T}
    t = market.ε_br_play[1]
    c_i = market.c[:, i]
    z = c_i ./ p
    γ = __lse_softmax(z ./ t)
    market.x[:, i] .= market.w[i] .* γ ./ p
    market.val_u[i] = dot(c_i, market.x[:, i])
    return nothing
end

LSEResponse = ResponseOptimizer(
    __lse_response,
    :lse,
    "LSEResponse"
)
