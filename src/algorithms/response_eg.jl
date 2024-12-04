# -----------------------------------------------------------------------
# run subproblems as best-response-type mappings
#   using induced utility function from Eigenberg-Gale-type potentials
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
using JuMP, COPT
import MathOptInterface as MOI

# --------------------------------------------------------------------------
# solve the utility maximization problem by JuMP + optimizer
#   induced from Eigenberg-Gale-type potentials
#   for linear utility function
# --------------------------------------------------------------------------
@doc raw"""
    solve the dual problem of the following form:
    min - wᵢ log (vᵢ) + μ ⋅ logbar(sᵢ)
"""
function __conic_eigenberg_gale_response(;
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
    @constraint(md, xc, s + v .* fisher.c[i, :] - p .== 0)
    JuMP.optimize!(md)
    val_x = abs.(dual.(xc))
    return ResponseInfo(
        val_x,
        objective_value(md),
        # the rest is dummy
        val_x,
        ϵᵢ,
        1,
        md
    )
end

EGConic = EigenbergGaleConicResponse = ResponseOptimizer(
    __conic_eigenberg_gale_response,
    :structured,
    "EigenbergGaleConicResponse"
)

# --------------------------------------------------------------------------
# solve the utility maximization problem by JuMP + optimizer
#   induced from Eigenberg-Gale-type potentials
#   for CES utility function of ρ < 1
# --------------------------------------------------------------------------
function __conic_eigenberg_gale_response_ces_type_i(;
    i::Int=1,
    p::Vector{T}=nothing,
    fisher::FisherMarket=nothing,
    μ=1e-4,
    verbose=false
) where {T}
    ρ = fisher.ρ
    ϵᵢ = μ * 1e-5
    md = __generate_empty_jump_model(; verbose=verbose, tol=ϵᵢ)

    @variable(md, λ)
    @variable(md, logλ)
    log_to_expcone!(λ, logλ, md)

    # Δ^{ρ} ξ^{1-ρ}≥ r 
    # ⇒ [Δ,ξ,r] ∈ P₃(ρ) [power cone]
    @variable(md, ξ[1:fisher.n])
    @constraint(md, sum(ξ) <= 1)

    @constraint(
        md,
        ξc[j=1:fisher.n],
        [p[j], ξ[j], fisher.c[i, j] * λ] in MOI.PowerCone(ρ)
    )
    @objective(md, Min,
        -1 / ρ * fisher.w[i] * logλ
    )

    JuMP.optimize!(md)
    val_x = first.(dual.(md[:ξc]))
    return ResponseInfo(
        val_x,
        objective_value(md),
        # the rest is dummy
        val_x,
        ϵᵢ,
        1,
        md
    )
end

EGConicCESTypeI = EigenbergGaleConicCESResponseTypeI = ResponseOptimizer(
    __conic_eigenberg_gale_response_ces_type_i,
    :structured,
    "EigenbergGaleConicCESResponseTypeI"
)


# --------------------------------------------------------------------------
# solve the utility maximization problem analytically
#   induced from Eigenberg-Gale-type potentials
#   for CES utility function of ρ < 1
# --------------------------------------------------------------------------
function __analytic_eigenberg_gale_response_ac(;
    kwargs...
)
    # keep skeleton only
end

EGConicAC = EigenbergGaleAnalyticalCESResponse = ResponseOptimizer(
    __analytic_eigenberg_gale_response_ac,
    :analytic,
    "EigenbergGaleAnalyticalCESResponse"
)

