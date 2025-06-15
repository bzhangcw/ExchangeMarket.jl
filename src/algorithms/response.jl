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