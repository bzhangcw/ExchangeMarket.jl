# -----------------------------------------------------------------------
# Quasi-linear-log utility response
#   u(x) = Σ_{j<n} c_j log(x_j) + x_n
# Closed-form UMP via the regime split p_n vs w / ⟨1,c⟩.
# -----------------------------------------------------------------------

# --------------------------------------------------------------------------
# Utility evaluation
# --------------------------------------------------------------------------
"""
    utility(at::QuasiLinearLogAgent, c, x)

Evaluate the quasi-linear-log utility u(x) = Σ_{j<n} c_j log(x_j) + x_n.
The `c` argument is unused (kept for dispatch consistency with AgentView).
Returns -Inf if any x[j] ≤ 0 for j < n.
"""
@inline function utility(at::QuasiLinearLogAgent, c, x)
    val = 0.0
    @inbounds for j in 1:(at.n-1)
        x[j] <= 0.0 && return -Inf
        val += at.c[j] * log(x[j])
    end
    val += x[at.n]
    return val
end

# --------------------------------------------------------------------------
# Closed-form demand: two-regime piecewise solution
# --------------------------------------------------------------------------
"""
    solve_ql_demand(agent::QuasiLinearLogAgent, p, w)

Closed-form UMP for quasi-linear-log utility at price `p` (length `agent.n`)
and budget `w > 0`. Returns (x, u).

Let c̄ := Σⱼ c_j and threshold T := w / c̄. With pn := p[n]:
  Interior (pn ≤ T): x_j = c_j pn / p_j for j<n; x_n = w/pn - c̄.
  Corner  (pn ≥ T): x_j = c_j w / (c̄ p_j) for j<n; x_n = 0.
The two formulas agree at the boundary pn = T (x_n = 0).
"""
function solve_ql_demand(agent::QuasiLinearLogAgent, p::AbstractVector, w::Real)
    n = agent.n
    c = agent.c
    cbar = sum(c)
    pn = p[n]
    T = w / cbar
    x = zeros(n)
    if pn <= T
        @inbounds for j in 1:(n-1)
            x[j] = c[j] * pn / p[j]
        end
        x[n] = w / pn - cbar
    else
        @inbounds for j in 1:(n-1)
            x[j] = c[j] * w / (cbar * p[j])
        end
        x[n] = 0.0
    end
    return x, utility(agent, nothing, x)
end
