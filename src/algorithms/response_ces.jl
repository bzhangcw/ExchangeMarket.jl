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
# primal form of CES economy in linear-conic form
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
    μ=1e-4,
    verbose=false,
    kwargs...
) where {T}
    ρ = market.ρ
    ϵᵢ = μ * 1e-5
    md = __generate_empty_jump_model(; verbose=verbose, tol=ϵᵢ)

    @variable(md, u)
    @variable(md, logu)
    log_to_expcone!(u, logu, md)

    @variable(md, x[1:market.n] >= 0)
    @variable(md, ξ[1:market.n] >= 0)
    # budget constraint
    @constraint(md, budget, p' * x <= market.w[i])
    # utility constraint
    # Δ^{ρ} ξ^{1-ρ}≥ r 
    # ⇒ [Δ,ξ,r] ∈ P₃(ρ) [power cone]
    _c = market.c[:, i] .^ (1 / ρ)
    @constraint(md, sum(ξ) == u)
    @constraint(
        md,
        ξc[j=1:market.n],
        [_c[j] * x[j], u, ξ[j]] in MOI.PowerCone(ρ)
    )
    @objective(md, Max, logu)

    JuMP.optimize!(md)
    # ensure non-negativity
    market.x[:, i] .= max.(value.(x), 0.0)
    return ResponseInfo(
        objective_value(md),
        # the rest is dummy
        ϵᵢ,
        1,
        md
    )
end

CESConic = CESConicResponse = ResponseOptimizer(
    __conic_log_response_ces,
    :linconic,
    "CESConicResponse"
)

# --------------------------------------------------------------------------
# dual form of CES economy in linear-conic form
# --------------------------------------------------------------------------
@doc raw"""
solve the logarithmic utility maximization problem by JuMP + optimizer
  for linear utility function in the `dual form`
"""
function __conic_log_response_ces_dual(;
    i::Int=1,
    p::Vector{T}=nothing,
    market::Market=nothing,
    μ=1e-4,
    verbose=false
) where {T}
    ϵᵢ = μ * 1e-5
    md = __generate_empty_jump_model(; verbose=verbose, tol=ϵᵢ)
    @variable(md, s[1:market.n] .>= 0)
    @variable(md, logs[1:market.n])
    @variable(md, v .>= 0)
    @variable(md, logv)
    log_to_expcone!.(s, logs, md)
    log_to_expcone!(v, logv, md)
    @objective(md, Min, -market.w[i] * logv - μ * sum(logs))
    @constraint(md, xc, s + v .* market.c[:, i] - p .== 0)
    JuMP.optimize!(md)
    market.x[:, i] .= abs.(dual.(xc))
    return ResponseInfo(
        objective_value(md),
        # the rest is dummy
        ϵᵢ,
        1,
        md
    )
end
DualCESConic = DualCESConicResponse = ResponseOptimizer(
    __conic_log_response_ces_dual,
    :linconic,
    "DualCESConicResponse"
)

# --------------------------------------------------------------------------
# solve the CES utility maximization problem analytically
# --------------------------------------------------------------------------
@doc raw"""
solve the utility maximization problem analytically
  induced from convex potentials
  for CES utility function of ρ < 1
"""
function __analytic_log_response_ac(;
    kwargs...
)
    # keep skeleton only
end

CESAnalytic = ResponseOptimizer(
    __analytic_log_response_ac,
    :analytic,
    "CESAnalytic"
)

