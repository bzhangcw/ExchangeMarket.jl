# -----------------------------------------------------------------------
# Legacy CES conic best-response (primal + dual via JuMP). Kept for
# the constrained / equality-form experiments under scripts/fisher/.
# The duplicate utility / _ces_demand! / __analytic_response / CESAnalytic
# definitions that used to live here were dropped — they are owned by
# src/algorithms/response/response_ces.jl, which is loaded first.
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
using JuMP
import MathOptInterface as MOI

# --------------------------------------------------------------------------
# Primal form of CES economy in linear-conic form.
#   max log(u_i(x_i)) s.t. p'x ≤ w, x ≥ 0, ξ_j ≥ (c_j^(1/ρ) x_j) ⊗ u via PowerCone(ρ).
# --------------------------------------------------------------------------
@doc raw"""
  solve the logarithmic utility maximization problem by JuMP + optimizer
  for CES utility function of ρ ≤ 1
    use max log(uᵢ(xᵢ))
"""
function __conic_log_response_ces(;
    i::Int=1,
    p::Vector{T}=nothing,
    market::Market=nothing,
    agent::Union{AgentView,Nothing}=nothing,
    μ=1e-4,
    verbose=false,
    kwargs...
) where {T}
    av = isnothing(agent) ? market.agents[i] : agent
    n = av.n
    w = market.w[av.i]
    ρ = av.atype isa CESAgent ? av.atype.ρ : 1.0
    c = Vector(av.c)  # JuMP needs dense Vector
    ϵᵢ = μ * 1e-5
    md = __generate_empty_jump_model(; verbose=verbose, tol=ϵᵢ)

    @variable(md, u)
    @variable(md, logu)
    log_to_expcone!(u, logu, md)

    @variable(md, x[1:n] >= 0)
    @variable(md, ξ[1:n] >= 0)
    @constraint(md, budget, p' * x <= w)
    _c = c .^ (1 / ρ)
    @constraint(md, sum(ξ) == u)
    @constraint(
        md,
        ξc[j=1:n],
        [_c[j] * x[j], u, ξ[j]] in MOI.PowerCone(ρ)
    )
    @objective(md, Max, logu)

    JuMP.optimize!(md)
    av.x .= max.(value.(x), 0.0)
    market.val_u[av.i] = utility(av)
    return nothing
end

CESConic = CESConicResponse = ResponseOptimizer(
    __conic_log_response_ces,
    :linconic,
    "CESConicResponse"
)

# --------------------------------------------------------------------------
# Dual form of CES economy in linear-conic form.
# --------------------------------------------------------------------------
@doc raw"""
solve the logarithmic utility maximization problem by JuMP + optimizer
  for linear utility function in the `dual form`
"""
function __conic_log_response_ces_dual(;
    i::Int=1,
    p::Vector{T}=nothing,
    market::Market=nothing,
    agent::Union{AgentView,Nothing}=nothing,
    μ=1e-4,
    verbose=false,
    kwargs...
) where {T}
    av = isnothing(agent) ? market.agents[i] : agent
    n = av.n
    w = market.w[av.i]
    c = Vector(av.c)  # JuMP needs dense Vector
    ϵᵢ = μ * 1e-5
    md = __generate_empty_jump_model(; verbose=verbose, tol=ϵᵢ)
    @variable(md, s[1:n] .>= 0)
    @variable(md, logs[1:n])
    @variable(md, v .>= 0)
    @variable(md, logv)
    log_to_expcone!.(s, logs, md)
    log_to_expcone!(v, logv, md)
    @objective(md, Min, -w * logv - μ * sum(logs))
    @constraint(md, xc, s + v .* c - p .== 0)
    JuMP.optimize!(md)
    av.x .= abs.(dual.(xc))
    return nothing
end

DualCESConic = DualCESConicResponse = ResponseOptimizer(
    __conic_log_response_ces_dual,
    :linconic,
    "DualCESConicResponse"
)
