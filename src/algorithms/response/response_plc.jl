# -----------------------------------------------------------------------
# PLC (Piecewise-Linear Concave) utility response
#   u(x) = min_{ℓ∈[L]} { aℓ'x + bℓ },  aℓ ≥ 0, bℓ ≥ 0
#   UMP solved as LP via Mosek
# -----------------------------------------------------------------------
using JuMP
import MathOptInterface as MOI

# --------------------------------------------------------------------------
# Utility evaluation for PLC agents
# --------------------------------------------------------------------------
"""
    utility(at::PLCAgent, c, x)

Evaluate PLC utility: u(x) = min_ℓ { aℓ'x + bℓ }.
The `c` argument is unused (kept for dispatch consistency with AgentView).
"""
@inline function utility(at::PLCAgent, c, x)
    val = Inf
    for ℓ in 1:at.L
        val = min(val, dot(view(at.a, ℓ, :), x) + at.b[ℓ])
    end
    return val
end

# --------------------------------------------------------------------------
# LP response: solve UMP as an LP via epigraph
# --------------------------------------------------------------------------
"""
    __lp_response_plc(; plc_agent, p, w, n, verbose=false, μ=1e-4, kwargs...)

Solve the PLC utility maximization problem as an LP:
  max t
  s.t. t ≤ aℓ'x + bℓ,  ∀ℓ∈[L]
       ⟨p, x⟩ ≤ w
       x ≥ 0

Returns (x, u) where x is the optimal allocation and u is the utility value.
"""
function __lp_response_plc(;
    plc_agent::PLCAgent,
    p::AbstractVector{T},
    w::Real,
    n::Int,
    verbose=false,
    μ=1e-4,
    kwargs...
) where {T}
    L = plc_agent.L
    md = __generate_empty_jump_model(; verbose=verbose, tol=μ * 1e-5)

    @variable(md, x[1:n] >= 0)
    @variable(md, t)  # epigraph variable for min_ℓ

    # t ≤ aℓ'x + bℓ for each hyperplane ℓ
    for ℓ in 1:L
        aℓ = view(plc_agent.a, ℓ, :)
        @constraint(md, t <= dot(aℓ, x) + plc_agent.b[ℓ])
    end

    # budget constraint
    @constraint(md, dot(p, x) <= w)

    @objective(md, Max, t)
    JuMP.optimize!(md)

    x_val = max.(value.(x), 0.0)
    u_val = utility(plc_agent, nothing, x_val)
    return x_val, u_val
end

PLCResponse = ResponseOptimizer(
    __lp_response_plc,
    :linconic,
    "PLCResponse"
)
