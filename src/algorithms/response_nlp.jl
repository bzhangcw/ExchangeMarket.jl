# -----------------------------------------------------------------------
# run subproblems as best-response-type mappings
#   using primal-dual nonlinear programming methods
# these methods will treat the subproblems 
#   as general block-box nonlinear programming problems 
#   regardless of the specific functional forms of utilities.
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
using Optim, LineSearches, ForwardDiff

# --------------------------------------------------------------------------
# vanilla pure primal affine-scaling Newton method
# --------------------------------------------------------------------------
@doc raw"""
    a vanilla primal Newton affine-scaling method for the problem:
    min f(x) with gradient g(x)
    s.t. x >= 0
"""
function __vanilla_newton_response(f, g;
    H=nothing, x₀=nothing, n=10, maxiter=1000, tol=1e-4,
    verbose=false
)
    @warn "This function has bad performance! Use other ResponseOptimizer instead."
    x = isnothing(x₀) ? rand(n) : copy(x₀)
    H = isnothing(H) ? ForwardDiff.hessian(f, x) : H
    f_val = f(x)
    g_val = g(x)
    gₙ = 1e5
    for k in 1:maxiter
        gₙ = norm(g_val)
        if gₙ <= tol
            break
        end
        H_val = H(x)
        dp = -H_val \ g_val
        αₘ = minimum(proj.(-g_val ./ dp))
        α = αₘ * 0.99
        x .= x .+ α * dp
        verbose && println("$(k): f_val: $(f_val), |g|: $(gₙ), α: $(α)")
        f_val = f(x)
        g_val = g(x)
    end
    return ResponseInfo(x, f_val, g_val, gₙ, 1)
end

NR = NewtonResponse = ResponseOptimizer(__vanilla_newton_response, :structured, "NewtonResponse")


# --------------------------------------------------------------------------
# third-party method from Optim.jl
# --------------------------------------------------------------------------
__optim_ipnewton = IPNewton(;
    linesearch=Optim.backtrack_constrained_grad,
    μ0=:auto,
    show_linesearch=false
)

__get_options(tol=1e-8) = Optim.Options(
    g_abstol=tol,
    iterations=1_000,
    store_trace=false,
    show_trace=false,
    show_every=1,
    time_limit=100
)

"""
This module provides optimization functions for the subproblem.

The `optim_newton` function implements a Newton optimization method that can handle
twice differentiable functions with constraints. It utilizes the `Optim` package for
the optimization process.

## Functions

### optim_newton(f, g; H=nothing, x₀=nothing, n=10, maxiter=1000, tol=1e-4, verbose=false)

- `f`: The objective function to minimize.
- `g`: The gradient of the objective function.
- `H`: The Hessian of the objective function (optional).
- `x₀`: Initial guess for the optimization (optional).
- `n`: The dimension of the problem (default is 10).
- `maxiter`: Maximum number of iterations (default is 1000).
- `tol`: Tolerance for convergence (default is 1e-4).
- `verbose`: If true, prints detailed optimization progress (default is false).

Returns the result of the optimization process.
"""
function __optim_newton(;
    f=nothing, g=nothing,
    H=nothing, x₀=nothing,
    n=10, maxiter=1000,
    tol=1e-4,
    verbose=false
)
    x = isnothing(x₀) ? rand(n) : copy(x₀)
    H = isnothing(H) ? ForwardDiff.hessian(f, x) : H
    lx = fill(0, length(x))
    ux = fill(Inf, length(x))
    _g!(g_val, x) = (g_val .= g(x))
    _H!(H_val, x) = (H_val .= H(x))
    dfc = TwiceDifferentiableConstraints(lx, ux)
    obj = TwiceDifferentiable(f, _g!, _H!, x)
    res = optimize(
        obj, dfc, x₀, __optim_ipnewton, __get_options(tol)
    )
    return ResponseInfo(
        res.minimizer,
        res.minimum,
        g(res.minimizer),
        res.g_residual,
        res.iterations
    )
end

ONR = OptimjlNewtonResponse = ResponseOptimizer(__optim_newton, :nlp, "Optim.jl-Newton")
