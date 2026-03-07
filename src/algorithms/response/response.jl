# -----------------------------------------------------------------------
# run subproblems as best-response-type mappings
#   using original utility function
# also define the ResponseOptimizer struct
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
using JuMP
import MathOptInterface as MOI

Base.@kwdef mutable struct ResponseOptimizer
    optfunc::Union{Function,Nothing}
    style::Symbol
    name::String = ""
end

# call to the optimizer of response mappings
solve!(optimizer::ResponseOptimizer; kwargs...) = optimizer.optfunc(; kwargs...)