# -----------------------------------------------------------------------
# run subproblems as best-response-type mappings 
#   using original utility function
# also define the ResponseInfo and ResponseOptimizer structs
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------

mutable struct ResponseInfo
    x::Vector{Float64}
    f_val::Float64
    g_val::Vector{Float64}
    ϵ::Float64
    k::Int
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
function __original_utility_response(u, ∇u;
    x₀=nothing, n=10, maxiter=1000, tol=1e-4,
    verbose=false
)
end

BR = OriginalBestResponse = ResponseOptimizer(__original_utility_response, :structured, "BestResponse")