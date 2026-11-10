# Array ownership and backend operations shared by PR and perturbed IPM.
# GPU methods are loaded explicitly from linear-gpu.jl; CPU needs no CUDA.
linear_device(::AbstractArray) = :cpu
linear_sync(::AbstractArray) = nothing
linear_zeros(a, dims...) = fill!(similar(a, Float64, dims...), 0.0)
linear_arrays(c, w) = (Matrix{Float64}(c), Vector{Float64}(w))
linear_default_solver(::AbstractArray) = :cholesky
linear_supports(c) = [findall(>(0), view(c, :, i)) for i in axes(c, 2)]
linear_response_workspace(c, w) = nothing
linear_metrics_workspace(c) = nothing
linear_solver_workspace(c, w, method) = nothing
linear_validate_solver(c, method) = method == :cudss ?
    throw(ArgumentError(":cudss requires GPU arrays")) : nothing
linear_prices(c, w, ::Nothing) = fill!(similar(w, size(c, 1)), sum(w) / size(c, 1))
linear_prices(c, w, p) = copyto!(similar(w, size(c, 1)), p)
linear_pr_workspace(c, p) = similar(p)
linear_allocations(c, T, n, m) = Matrix{T}(undef, n, m)

function linear_set_diagonal!(H, diagonal)
    for j in axes(H, 1)
        H[j, j] = diagonal[j]
    end
    H
end
function linear_shift_diagonal!(H, shift)
    for j in axes(H, 1)
        H[j, j] += shift
    end
    H
end
linear_max_step(d) = min(1.0, -0.99 / min(minimum(d), -eps(Float64)))

# The generic array kernels below can be tested on CPU arrays as well as CUDA.
# Workspaces are allocated once, never transferred between devices in a solve.
function linear_array_metrics_workspace(c)
    n, m = size(c)
    (; tmp=similar(c), spend=similar(c, 1, m), utility=similar(c, 1, m),
       best=similar(c, 1, m), demand=similar(c, n), budget=similar(c, 1, m),
       regret=similar(c, 1, m))
end
function linear_array_metrics(c, w, p, x, ws; demand=ws.demand)
    total_budget = sum(w)
    if !all(isfinite, p) || !all(>(0), p) || !all(isfinite, x) || any(<(0), x)
        return (; mc=Inf, budget=Inf, regret=Inf, error=Inf,
            total_spending=Inf, total_budget, total_budget_error=Inf)
    end
    sum!(reshape(demand, :, 1), x)
    ws.tmp .= p .* x
    sum!(ws.spend, ws.tmp)
    ws.tmp .= c .* x
    sum!(ws.utility, ws.tmp)
    ws.tmp .= c ./ p
    maximum!(ws.best, ws.tmp)
    ws.budget .= abs.(ws.spend ./ w' .- 1)
    ws.regret .= max.(0.0, 1 .- ws.utility ./ (w' .* ws.best))
    mc = maximum(v -> abs(v - 1), demand)
    budget, regret, total_spending = maximum(ws.budget), maximum(ws.regret), sum(ws.spend)
    (; mc, budget, regret, error=max(mc, budget, regret), total_spending,
       total_budget, total_budget_error=abs(total_spending / total_budget - 1))
end

function linear_array_response_workspace(c, w)
    n, m = size(c)
    mat() = similar(c)
    row() = similar(c, 1, m)
    (; ratios=mat(), tmp=mat(), tmp2=mat(), gamma=mat(), mask=similar(c, Bool),
       best=row(), lo=row(), hi=row(), z=row(), tau=row(), mass=row(), deriv=row(),
       active=similar(c, Bool, 1, m), counts=similar(c, Int, 1, m),
       largest=row(), rest=row(), rest_mass=row(), denom=row(), utility=row(),
       rows=reshape(collect(1:n), n, 1))
end

function linear_array_response!(x, s, lam, delta, c, w, p, sigma, ws;
    rtol=1e-12, maxiter=100)
    isfinite(sigma) && sigma > 0 || throw(ArgumentError("sigma must be positive"))
    all(isfinite, p) && all(>(0), p) || throw(ArgumentError("prices must be positive"))
    n, m = size(c)
    ws.ratios .= c ./ p
    maximum!(ws.best, ws.ratios)
    all(>(0), ws.best) || throw(ArgumentError("a buyer has no positive valuation"))
    ws.ratios ./= ws.best
    ws.tau .= sigma ./ w'
    ws.lo .= ws.tau ./ (1 .+ n .* ws.tau)
    ws.hi .= min.(1.0, n .* ws.lo)
    ws.z .= ifelse.(isfinite.(delta') .& (delta' .> 0),
        clamp.(delta', ws.lo, ws.hi), (ws.lo .+ ws.hi) ./ 2)
    fill!(ws.active, true)
    fill!(ws.counts, 0)
    for it in 1:maxiter
        ws.counts .+= ws.active
        ws.tmp .= ws.tau .* (1 .- ws.z) .* ws.ratios ./ ((1 .- ws.ratios) .+ ws.z .* ws.ratios)
        sum!(ws.mass, ws.tmp)
        ws.mass .-= 1
        ws.tmp .= .-ws.tau .* ws.ratios ./ ((1 .- ws.ratios) .+ ws.z .* ws.ratios).^2
        sum!(ws.deriv, ws.tmp)
        ws.active .&= abs.(ws.mass) .> rtol
        !any(ws.active) && break
        ws.lo .= ifelse.(ws.active .& (ws.mass .> 0), ws.z, ws.lo)
        ws.hi .= ifelse.(ws.active .& (ws.mass .<= 0), ws.z, ws.hi)
        ws.best .= ws.z .- ws.mass ./ ws.deriv
        ws.z .= ifelse.(ws.active,
            ifelse.((ws.lo .< ws.best) .& (ws.best .< ws.hi), ws.best,
                ifelse.(ws.mass .< 0, ws.lo, (ws.lo .+ ws.hi) ./ 2)), ws.z)
    end
    any(ws.active) && error("perturbed response failed at sigma=$sigma")
    # Recompute each buyer's max ratio: ws.best was reused for Newton proposals.
    ws.tmp .= c ./ p
    maximum!(ws.best, ws.tmp)
    delta .= vec(ws.z)
    lam .= vec((1 .- ws.z) ./ ws.best)
    s .= p .* ((1 .- ws.ratios) .+ ws.z .* ws.ratios)
    x .= sigma ./ s
    sum(ws.counts)
end

function linear_array_hessian!(H, V, diagonal, c, w, p, x, sigma, ws; sensitivity=nothing)
    ws.tmp .= c .* x
    sum!(ws.utility, ws.tmp)
    ws.gamma .= ws.tmp ./ ws.utility
    # Exclude precisely one largest coordinate, including when there are ties.
    indices = argmax(ws.gamma; dims=1)
    ws.mask .= ws.rows .== getindex.(indices, 1)
    maximum!(ws.largest, ws.gamma)
    ws.tmp .= ifelse.(ws.mask, 0.0, ws.gamma.^2)
    sum!(ws.rest, ws.tmp)
    ws.tmp .= ifelse.(ws.mask, 0.0, ws.gamma)
    sum!(ws.rest_mass, ws.tmp)
    ws.tau .= sigma ./ w'
    ws.denom .= ws.tau .+ ws.largest.^2 .+ ws.rest
    ws.tmp .= ifelse.(ws.mask, ws.rest, ws.largest.^2 .+ max.(0.0, ws.rest .- ws.gamma.^2))
    ws.tmp2 .= (p .* x).^2 .* (ws.tau .+ ws.tmp) ./ (sigma .* ws.denom)
    sum!(reshape(diagonal, :, 1), ws.tmp2)
    V .= p .* x .* ws.gamma ./ sqrt.(sigma .* ws.denom)
    if sensitivity !== nothing
        ws.tmp .= ifelse.(ws.mask, ws.tau .+ ws.rest .- ws.largest .* ws.rest_mass,
            ws.denom .- ws.gamma)
        ws.tmp2 .= x .* ws.tmp ./ ws.denom
        sum!(reshape(sensitivity, :, 1), ws.tmp2)
    end
    mul!(H, V, V', -1.0, 0.0)
    linear_set_diagonal!(H, diagonal)
end

function linear_array_pr_step!(x, p, c, w, ws)
    ws.tmp .= c .* x
    sum!(ws.utility, ws.tmp)
    x .= w' .* ws.tmp ./ ws.utility
    sum!(reshape(p, :, 1), x)
    x ./= p
    nothing
end

# Storage hooks let sparse markets share the same continuation and line search.
linear_demand(x) = vec(sum(x; dims=2))
linear_copy!(target, source) = copyto!(target, source)
function linear_validate_market(c, w)
    all(isfinite, c) && all(>=(0), c) || throw(ArgumentError("valuations must be nonnegative and finite"))
    all(isfinite, w) && all(>(0), w) || throw(ArgumentError("budgets must be positive and finite"))
    all(vec(maximum(c; dims=1)) .> 0) || throw(ArgumentError("every buyer must value a good"))
    all(vec(maximum(c; dims=2)) .> 0) || throw(ArgumentError("every good must be valued for strict market clearing"))
end
function linear_ipm_buffers(c, w, solver_workspace)
    n,m = size(c)
    (; x=linear_zeros(c,n,m), s=linear_zeros(c,n,m), H=linear_zeros(c,n,n),
       V=linear_zeros(c,n,m), diagonal=linear_zeros(w,n), sensitivity=linear_zeros(w,n))
end
function linear_pr_initial_allocation(c, w, p)
    n,m = size(c)
    x = linear_allocations(c, promote_type(Float64,eltype(c),eltype(w)), n,m)
    x .= (w' ./ n) ./ p
    x
end
