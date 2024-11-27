# -----------------------------------------------------------------------
# run subproblems as best-response-type mappings 
#   using original utility function
# also define the ResponseInfo and ResponseOptimizer structs
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
using JuMP, COPT
import MathOptInterface as MOI
mutable struct ResponseInfo
    x::Vector{Float64}
    f_val::Float64
    g_val::Vector{Float64}
    ϵ::Float64
    k::Int
    md::Union{JuMP.Model,Nothing}
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
    ϵᵢ = μ * 1e-5
    md = __generate_empty_jump_model(; verbose=verbose, tol=ϵᵢ)
    @variable(md, x[1:fisher.n] .>= 0)
    @objective(md, Min, -fisher.val_∇u[i, :]' * x)
    @constraint(md, xc, p' * x <= fisher.w[i])
    JuMP.optimize!(md)
    val_x = abs.(value.(x))
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

BR = OriginalBestResponse = ResponseOptimizer(__original_utility_response, :structured, "BestResponse")