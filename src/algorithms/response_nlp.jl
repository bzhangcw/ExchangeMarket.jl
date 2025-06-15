# -----------------------------------------------------------------------
# run subproblems as best-response-type mappings
#   using primal-dual nonlinear programming methods
# these methods will treat the subproblems 
#   as general block-box nonlinear programming problems 
#   regardless of the specific functional forms of utilities.
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# @todo: 
#   - not good, planning to change to MadNLP.jl
# -----------------------------------------------------------------------
using Optim, LineSearches, ForwardDiff


# --------------------------------------------------------------------------
# third-party method from Optim.jl
# --------------------------------------------------------------------------
__optim_ipnewton = IPNewton(;
    linesearch=Optim.backtrack_constrained_grad,
    μ0=:auto,
    show_linesearch=false
)

__get_options(tol=1e-8, verbose=false, store_trace=false) = Optim.Options(
    f_abstol=tol,
    g_abstol=tol,
    iterations=1_000,
    store_trace=store_trace,
    show_trace=verbose,
    show_every=1,
    time_limit=100
)

"""
This module provides optimization functions for the subproblem.

The `optim_newton` function implements a Newton optimization method that can handle
twice differentiable functions with box constraints [ℓ, u] only. 

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
    n=10,
    ub=nothing,
    lb=nothing,
    tol=1e-8,
    store_trace=false,
    verbose=false,
)
    x = isnothing(x₀) ? rand(n) : copy(x₀)
    H = isnothing(H) ? ForwardDiff.hessian(f, x) : H
    lx = isnothing(lb) ? fill(0.0, length(x)) : lb
    ux = isnothing(ub) ? fill(1.0, length(x)) : ub
    _g!(g_val, x) = (g_val .= g(x))
    _H!(H_val, x) = (H_val .= H(x))
    dfc = TwiceDifferentiableConstraints(lx, ux)
    obj = TwiceDifferentiable(f, _g!, _H!, x)
    res = optimize(
        obj, dfc, x₀, __optim_ipnewton, __get_options(tol, verbose, store_trace)
    )
    return ResponseInfo(
        res.minimizer,
        res.minimum,
        g(res.minimizer),
        res.g_residual,
        res.iterations,
        res
    )
end

ONR = OptimjlNewtonResponse = ResponseOptimizer(__optim_newton, :nlp, "Optim.jl-Newton")
