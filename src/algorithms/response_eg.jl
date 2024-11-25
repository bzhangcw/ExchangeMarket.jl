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
#   only support linear utility function for now.
# --------------------------------------------------------------------------
@doc raw"""
    solve the problem of the following form:
    min [-w⋅log(u(x)) + p'x (-μ logbar(x))]
"""
function __conic_eigenberg_gale_response(;
    i::Int=1,
    p::Vector{T}=nothing,
    fisher::FisherMarket=nothing,
    μ=1e-4,
    verbose=false
) where {T}
    md = __generate_empty_jump_model(; verbose=verbose)
    @variable(md, x[1:fisher.n] .>= 0)
    @variable(md, v[1:fisher.n] .>= 0)
    @variable(md, ul)
    @variable(md, uv)
    log_to_expcone!.(x, v, md)
    log_to_expcone!(uv, ul, md)
    @objective(md, Min, -fisher.w[i] * ul + p' * x - μ * sum(v))
    JuMP.optimize!(md)
    val_x = value.(x)
    return ResponseInfo(
        val_x,
        objective_value(md),
        # the rest is dummy
        fisher.∇u(val_x, i),
        1e-6,
        1
    )
end

EGConic = EigenbergGaleConicResponse = ResponseOptimizer(__conic_eigenberg_gale_response, :structured, "EigenbergGaleConicResponse")
