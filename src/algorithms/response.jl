# -----------------------------------------------------------------------
# run subproblems as best-response-type mappings 
#   using original utility function
# also define the ResponseInfo and ResponseOptimizer structs
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
using JuMP
import MathOptInterface as MOI
mutable struct ResponseInfo
    f_val::Float64
    ϵ::Float64 # subproblem optimality, distance to best-response
    k::Int
    md::Any
end

Base.@kwdef mutable struct ResponseOptimizer
    optfunc::Union{Function,Nothing}
    style::Symbol
    name::String = ""
end

# call to the optimizer of response mappings
solve!(optimizer::ResponseOptimizer; kwargs...) = optimizer.optfunc(; kwargs...)

# --------------------------------------------------------------------------
# solve the original utility maximization problem by JuMP + optimizer
#   only support linear utility function for now.
# --------------------------------------------------------------------------
@doc raw"""
    solve the problem of the following form:
    min f(x) with gradient g(x)
    s.t. x >= 0
"""
function __original_utility_response(;
    i::Int=1,
    p::Vector{T}=nothing,
    fisher::FisherMarket=nothing,
    μ=1e-4,
    verbose=false
) where {T}
    @warn "This function is deprecated. Use `EGConic types or General NLP types` instead."
    ϵᵢ = μ * 1e-5
    md = __generate_empty_jump_model(; verbose=verbose, tol=ϵᵢ)
    @variable(md, x[1:fisher.n] .>= 0)
    @objective(md, Min, -fisher.u(x, i))
    @constraint(md, xc, p' * x <= fisher.w[i])
    JuMP.optimize!(md)
    val_x = abs.(value.(x))
    return ResponseInfo(
        objective_value(md),
        # the rest is dummy
        ϵᵢ,
        1,
        md
    )
end

BR = OriginalBestResponse = ResponseOptimizer(__original_utility_response, :linconic, "BestResponse")