# -----------------------------------------------------------------------
# run subproblems as best-response-type mappings
# @author:Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------


# using Optim
mutable struct ResponseInfo
    x::Vector{Float64}
    f_val::Float64
    g_val::Vector{Float64}
    ϵ::Float64
end

# solve the problem of the following form:
# min f(x) with gradient g(x)
#   x >= 0
function default_newton_response(f, g;
    H=nothing, x₀=nothing, n=10, maxiter=1000, tol=1e-4
)
    x = isnothing(x₀) ? rand(n) : copy(x₀)
    f_val = f(x)
    g_val = g(x)
    gₙ = 1e5
    for _ in 1:maxiter
        gₙ = norm(g_val)
        if gₙ <= tol
            break
        end
        H_val = isnothing(H) ? ForwardDiff.hessian(f, x) : H(x)
        dp = -H_val \ g_val
        αₘ = minimum(proj.(-g_val ./ dp))
        α = αₘ * 0.99
        x .= x .+ α * dp
        @debug """progress: f_val: $(f_val), |g|: $(gₙ), α: $(α)
        """
        f_val = f(x)
        g_val = g(x)
    end
    return ResponseInfo(x, f_val, g_val, gₙ)
end

@doc raw"""
    mosek_response(f, g; H=nothing, x₀=nothing, n=10, maxiter=1000, tol=1e-4)
    can only deal with linear utility function
"""
function mosek_response(f, g; H=nothing, x₀=nothing, n=10, maxiter=1000, tol=1e-4)
end
