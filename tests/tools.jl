using Optim, LineSearches
using ForwardDiff
using ExchangeMarket

__optim_ipnewton = IPNewton(;
    linesearch=Optim.backtrack_constrained_grad,
    μ0=:auto,
    show_linesearch=false
)

options = Optim.Options(
    g_tol=1e-8,
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
function optim_newton(f, g;
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
        obj, dfc, x₀, __optim_ipnewton, options
    )
    return ResponseInfo(
        res.minimizer,
        res.minimum,
        g(res.minimizer),
        res.g_residual,
        res.iterations
    )
end