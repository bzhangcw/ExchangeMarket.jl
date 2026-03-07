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
    kwargs...
) where {T}
    if all(market.ρ .>= 0)
        market.x[:, i] = market.g[:, i] ./ p
        market.val_u[i] = market.u(market.x[:, i], i)
        market.val_f[i], market.val_∇f[:, i], market.val_Hf[:, i] = market.f∇f(p, i)
        cs = market.c[:, i] .* spow.(market.x[:, i], market.ρ[i])
        sumcs = sum(cs)
        # update bids
        market.g[:, i] .= market.w[i] * cs ./ sumcs
    else
        market.x[:, i] = market.g[:, i] ./ p
        market.val_u[i] = market.u(market.x[:, i], i)
        market.val_f[i], market.val_∇f[:, i], market.val_Hf[:, i] = market.f∇f(p, i)
        cs = spow.(market.c[:, i] ./ spow.(p, market.ρ[i]), 1 / (1 - market.ρ[i]))
        sumcs = sum(cs)
        market.g[:, i] .= market.w[i] * cs ./ sumcs
    end
    return nothing
end

Bids = ResponseOptimizer(
    __bids_response,
    :bids,
    "Bids"
)
