# -----------------------------------------------------------------------
# run subproblems as best-response-type mappings
#   using induced utility function from Eigenberg-Gale-type potentials
#   the response mapping is induced from linear-conic programming
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
using JuMP
import MathOptInterface as MOI

# --------------------------------------------------------------------------
# solve the utility maximization problem by JuMP + optimizer
#   for linear utility function
# --------------------------------------------------------------------------
@doc raw"""
solve the utility maximization problem by JuMP + optimizer
  for linear utility function
"""
function __conic_log_response(;
    i::Int=1,
    p::Vector{T}=nothing,
    fisher::FisherMarket=nothing,
    μ=1e-4,
    verbose=false
) where {T}
    ϵᵢ = μ * 1e-5
    md = __generate_empty_jump_model(; verbose=verbose, tol=ϵᵢ)
    @variable(md, s[1:fisher.n] .>= 0)
    @variable(md, logs[1:fisher.n])
    @variable(md, v .>= 0)
    @variable(md, logv)
    log_to_expcone!.(s, logs, md)
    log_to_expcone!(v, logv, md)
    @objective(md, Min, -fisher.w[i] * logv - μ * sum(logs))
    @constraint(md, xc, s + v .* fisher.c[:, i] - p .== 0)
    JuMP.optimize!(md)
    val_x = abs.(dual.(xc))
    return ResponseInfo(
        objective_value(md),
        # the rest is dummy
        ϵᵢ,
        1,
        md
    )
end

EGConic = EigenbergGaleConicResponse = ResponseOptimizer(
    __conic_log_response,
    :linconic,
    "EigenbergGaleConicResponse"
)

@doc raw"""
  solve the utility maximization problem by JuMP + optimizer
  for CES utility function of ρ < 1
    use max log(uᵢ(xᵢ))
"""
function __conic_log_response_ces(;
    i::Int=1,
    p::Vector{T}=nothing,
    fisher::FisherMarket=nothing,
    μ=1e-4,
    verbose=false
) where {T}
    ρ = fisher.ρ
    ϵᵢ = μ * 1e-5
    md = __generate_empty_jump_model(; verbose=verbose, tol=ϵᵢ)

    @variable(md, u)
    @variable(md, logu)
    log_to_expcone!(u, logu, md)

    @variable(md, x[1:fisher.n] >= 0)
    @variable(md, ξ[1:fisher.n] >= 0)
    # budget constraint
    @constraint(md, budget, p' * x <= fisher.w[i])
    # utility constraint
    # Δ^{ρ} ξ^{1-ρ}≥ r 
    # ⇒ [Δ,ξ,r] ∈ P₃(ρ) [power cone]
    _c = fisher.c[:, i] .^ (1 / ρ)
    @constraint(md, sum(ξ) == u)
    @constraint(
        md,
        ξc[j=1:fisher.n],
        [_c[j] * x[j], u, ξ[j]] in MOI.PowerCone(ρ)
    )
    @objective(md, Max, logu)

    JuMP.optimize!(md)
    fisher.x[:, i] .= value.(x)
    return ResponseInfo(
        objective_value(md),
        # the rest is dummy
        ϵᵢ,
        1,
        md
    )
end

EGConicCES = EigenbergGaleConicCESResponse = ResponseOptimizer(
    __conic_log_response_ces,
    :linconic,
    "EigenbergGaleConicCESResponse"
)


# --------------------------------------------------------------------------
# solve the utility maximization problem analytically
#   induced from Eigenberg-Gale-type potentials
#   for CES utility function of ρ < 1
# --------------------------------------------------------------------------
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

