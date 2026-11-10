# Linear proportional response for the shared-target CPU/GPU comparison.
# Include linear-perturbed.jl first for linear_equilibrium_metrics.

"""
    pr_step!(x, p, c, w, weighted_utility)

Apply one linear proportional-response update, reusing all storage. Each
column of `x` temporarily holds bids w[i] * c[:,i] .* x[:,i] / dot(c[:,i], x[:,i]).
After summing bids into `p`, recover allocations at those updated prices.
`weighted_utility` is a length-n scratch vector shared by all buyers.
"""
function pr_step!(x, p, c, w, weighted_utility)
    n, m = size(x)
    @inbounds for i in 1:m
        for j in 1:n
            weighted_utility[j] = c[j, i] * x[j, i]
        end
        utility = sum(weighted_utility)
        for j in 1:n
            x[j, i] = w[i] * weighted_utility[j] / utility
        end
    end
    sum!(p, x)
    x ./= p
    return nothing
end

"""
    run_pr(f; stop, maxiter)
    run_pr(c, w; stop, maxiter)

Linear proportional response with uniform initial bids and prices, evaluated
against the shared equilibrium metrics after every update. Input valuations
and budgets are read-only. Allocate one allocation/bid matrix and reusable
price, utility, and demand vectors; only scalar trajectory records grow with
the number of iterations. The update is the linear specialization of Bids.
"""
run_pr(f; stop, maxiter) = run_pr(f.c, f.w; stop, maxiter)

function run_pr(c, w; stop::F, maxiter) where {F}
    t0 = time_ns()
    n, m = size(c)
    linear_device(c) == linear_device(w) || throw(ArgumentError("mixed CPU/GPU market arrays"))
    length(w) == m || throw(DimensionMismatch("one budget is required per buyer"))
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative"))
    T = promote_type(Float64, eltype(c), eltype(w))
    p = linear_prices(c, w, nothing)
    x = similar(c, T, n, m)
    x .= (w' ./ n) ./ p
    weighted_utility = linear_pr_workspace(c, p)
    metrics_workspace = linear_metrics_workspace(c)
    demand = similar(p)
    z = linear_equilibrium_metrics(c, w, p, x; demand, workspace=metrics_workspace)
    traj = typeof(merge((k=0, t=0.0), z))[]
    status = :max_iter
    for k in 0:maxiter
        push!(traj, merge((k=k, t=(time_ns() - t0) / 1e9), z))
        if stop(z)
            status = :converged
            break
        end
        k == maxiter && break
        pr_step!(x, p, c, w, weighted_utility)
        z = linear_equilibrium_metrics(c, w, p, x; demand, workspace=metrics_workspace)
    end
    return (; st=(p=p, x=x), traj, status, factorizations=0)
end
