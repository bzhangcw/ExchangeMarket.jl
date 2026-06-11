# -----------------------------------------------------------------------
# run subproblems as best-response-type mappings
#   using induced utility function from Eigenberg-Gale-type potentials
#   the response mapping is captured by linear-conic programming
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
using JuMP
import MathOptInterface as MOI

# --------------------------------------------------------------------------
# Utility evaluation for Linear and CES agents
# --------------------------------------------------------------------------
@inline utility(::LinearAgent, c, x) = sparse_dot(c, x)

@inline function utility(at::CESAgent, c, x)
    s = 0.0
    foreach_nz(c) do j, cj
        s += cj * spow(x[j], at.ρ)
    end
    return spow(s, 1.0 / at.ρ)
end

# --------------------------------------------------------------------------
# CES closed-form demand: x_j = w c_j^(1+σ) p_j^(-σ-1) / Σ_k c_k^(1+σ) p_k^(-σ)
# --------------------------------------------------------------------------
@inline function _ces_demand!(x, c::SparseColRef, p, w, σ)
    denom = sparse_reduce(c) do j, cj
        cj^(1.0 + σ) * p[j]^(-σ)
    end
    coeff = w / denom
    x .= 0.0
    sparse_scatter!(x, c) do j, cj
        coeff * cj^(1.0 + σ) * p[j]^(-σ - 1.0)
    end
end

@inline function _ces_demand!(x, c::AbstractVector, p, w, σ)
    cs = c .^ (1.0 + σ)
    denom = dot(cs, p .^ (-σ))
    x .= (w / denom) .* cs .* p .^ (-σ - 1.0)
end

# --------------------------------------------------------------------------
# solve the CES utility maximization problem analytically
# --------------------------------------------------------------------------
@doc raw"""
    Analytic best-response for linear and CES markets.

    For linear markets, the optimal bundle concentrates on the good
    with the highest bang-per-buck ratio ``c_{ji}/p_j``.

    For CES markets (ρ < 1), uses the closed-form via the
    induced convex potential ``f`` and its gradient.
"""
function __analytic_response(;
    i::Int=1,
    p::Vector{T}=nothing,
    market::Market=nothing,
    agent::Union{AgentView,Nothing}=nothing,
    kwargs...
) where {T}
    av = isnothing(agent) ? market.agents[i] : agent
    # Budget: Fisher agents have a fixed budget in `market.w`; Arrow–Debreu
    # agents carry an endowment view, so the budget is its value `⟨p, b⟩`.
    # Write it back so downstream grad!/eval!/hess! see the fresh budget
    # (removes the need for a separate update_budget! before play!).
    w = av.b === nothing ? market.w[av.i] : dot(p, av.b)
    av.b === nothing || (market.w[av.i] = w)
    if av.atype isa LinearAgent
        # linear: bang-per-buck — concentrate on best c_j/p_j
        j₊ = sparse_argmax(av.c) do j, cj
            cj / p[j]
        end
        av.x .= 0
        av.x[j₊] = w / p[j₊]
    else
        # CES closed-form demand:
        #   x_j = w · c_j^(1+σ) · p_j^(-σ-1) / Σ_k c_k^(1+σ) · p_k^(-σ)
        _ces_demand!(av.x, av.c, p, w, av.atype.σ)
    end
    market.val_u[av.i] = utility(av)
    return nothing
end

CESAnalytic = ResponseOptimizer(
    __analytic_response,
    :analytic,
    "CESAnalytic"
)
