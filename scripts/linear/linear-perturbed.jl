using LinearAlgebra, Printf
include("linear-backend.jl")
include("linear-perturbed-linsys.jl")

"""
    linear_equilibrium_metrics(c, w, p, x; demand=nothing)

Common accuracy measures for a unit-supply linear Fisher market: maximum
market-clearing error, relative budget error, and relative utility loss against
an exact linear best response at the current prices. `error` is their maximum.
Also returns total_spending, total_budget, and total_budget_error for reporting.
These aggregate fields do not change the individual-budget stopping metric.
No allocation or price normalization is applied when measuring these errors.
Pass a reusable length-n `demand` vector to avoid allocating row-sum storage.
"""
function linear_equilibrium_metrics(c, w, p, x; demand=nothing, workspace=nothing)
    n, m = size(c)
    total_budget = sum(w)
    if !all(isfinite, p) || !all(>(0), p) || !all(isfinite, x) || any(<(0), x)
        return (mc=Inf, budget=Inf, regret=Inf, error=Inf,
            total_spending=Inf, total_budget=total_budget, total_budget_error=Inf)
    end
    if demand === nothing
        demand = vec(sum(x; dims=2))
    else
        length(demand) == n || throw(DimensionMismatch("demand must have length n"))
        sum!(demand, x)
    end
    mc = maximum(v -> abs(v - 1), demand)
    budget = 0.0
    regret = 0.0
    total_spending = 0.0
    @inbounds for i in 1:m
        spend, utility, best = 0.0, 0.0, 0.0
        for j in 1:n
            spend += p[j] * x[j, i]
            utility += c[j, i] * x[j, i]
            best = max(best, c[j, i] / p[j])
        end
        total_spending += spend
        budget = max(budget, abs(spend / w[i] - 1))
        regret = max(regret, best > 0 ? 1 - utility / (w[i] * best) : 0.0)
    end
    return (mc=mc, budget=budget, regret=regret, error=max(mc, budget, regret),
        total_spending=total_spending, total_budget=total_budget,
        total_budget_error=abs(total_spending / total_budget - 1))
end

mutable struct PerturbedLinearState{V<:AbstractVector{Float64},M<:AbstractMatrix{Float64}}
    p::V
    x::M
    s::M
    λ::V
    δ::V
    σ::Float64
    μ::Float64
    name::Symbol
end

"""
    linear_perturbed_response!(x, s, λ, δ, c, w, p, σ; rtol=1e-12, maxiter=100)

Evaluate the unconstrained, barrier-perturbed conjugate response:

    fᵢ,σ(p) = max_{xᵢ>0} wᵢ log(cᵢ'xᵢ) + σ sum(log, xᵢ) - p'xᵢ.

Its scalar dual is min_{0<λᵢ<min(p./cᵢ)} -wᵢ log(λᵢ) - σ sum(log, p-λᵢcᵢ),
up to a price-independent constant. Recover sᵢ = p-λᵢcᵢ and xᵢ = σ ./ sᵢ.
The scalar stationarity equation gives λᵢ(cᵢ'xᵢ)=wᵢ, hence spending is
wᵢ+nσ. There is no budget constraint and no rescaling of the response.

Use δᵢ = 1-λᵢ max(cᵢ./p) as the scalar unknown. Solving in δ avoids
subtracting nearly equal prices and valuations when σ is small. Safeguarded
Newton enforces the scalar dual stationarity equation; `δ` supplies warm starts.
Optional `supports` lists the positive valuation indices in each column of c.
Zero-valued coordinates are recovered analytically as x_ji=σ/p_j.
Returns the total number of scalar iterations; throws if an oracle fails.
"""
function linear_perturbed_response!(x, s, λ, δ, c, w, p, σ;
    rtol=1e-12, maxiter=100, supports=nothing, workspace=nothing)
    n, m = size(c)
    isfinite(σ) && σ > 0 || throw(ArgumentError("σ must be positive and finite"))
    all(isfinite, p) && all(>(0), p) || throw(ArgumentError("prices must be positive and finite"))
    ratios = Vector{Float64}(undef, n)
    iterations = 0
    zero_value_x = σ ./ p
    @inbounds for i in 1:m
        J = supports === nothing ? axes(c, 1) : supports[i]
        # For c_ji=0 the conjugate equations give s_ji=p_j, x_ji=σ/p_j.
        copyto!(view(s, :, i), p)
        copyto!(view(x, :, i), zero_value_x)
        M = 0.0
        for j in J
            ratios[j] = c[j, i] / p[j]
            M = max(M, ratios[j])
        end
        M > 0 || throw(ArgumentError("buyer $i has no positive valuation"))
        for j in J
            ratios[j] /= M
        end
        τ = σ / w[i]
        t = τ / (1 + n * τ)
        lo, hi = t, min(1.0, n * t)
        z = isfinite(δ[i]) && δ[i] > 0 ? clamp(δ[i], lo, hi) : (lo + hi) / 2
        converged = false
        for it in 1:maxiter
            iterations += 1
            mass, derivative = 0.0, 0.0
            for j in J
                r = ratios[j]
                den = (1 - r) + z * r
                mass += τ * (1 - z) * r / den
                derivative -= τ * r / den^2
            end
            residual = mass - 1
            if abs(residual) <= rtol
                converged = true
                break
            end
            if residual > 0
                lo = z
            else
                hi = z
            end
            proposal = z - residual / derivative
            # This scalar residual is convex and decreasing. A Newton step
            # from the lower bracket stays below the root. If a warm start
            # overshoots that bracket, restart at lo instead of bisecting the
            # much wider full interval (important when σ is small).
            z = lo < proposal < hi ? proposal : (residual < 0 ? lo : (lo + hi) / 2)
        end
        converged || error("perturbed response failed for buyer $i at σ=$σ")
        δ[i] = z
        λ[i] = (1 - z) / M
        for j in J
            s[j, i] = p[j] * ((1 - ratios[j]) + z * ratios[j])
            x[j, i] = σ / s[j, i]
        end
    end
    return iterations
end

"""
    linear_perturbed_potential(c, w, p, x, σ)

Evaluate Φσ(p) = sum(p) + Σᵢ fᵢ,σ(p), given the exact conjugate responses `x`.
Its gradient is 1-Σᵢxᵢ and its Hessian is -ΣᵢDₚxᵢ. `x` is used directly;
no budget normalization or logarithmic-homogeneity factor is introduced.
"""
function linear_perturbed_potential(c, w, p, x, σ)
    value = sum(p)
    for i in axes(c, 2)
        xi = view(x, :, i)
        value += w[i] * log(dot(view(c, :, i), xi)) +
                 σ * sum(log, xi) - dot(p, xi)
    end
    return value
end

"""
    linear_perturbed_hessian!(H, V, diagonal, c, w, p, x, σ; sensitivity=nothing, supports=nothing, workspace=nothing)

Build the exact scaled demand Hessian H = -P (Σᵢ Dₚxᵢ) P using a
diagonal-minus-rank-m representation and BLAS multiplication. The diagonal is
evaluated separately to avoid cancellation as a buyer concentrates on one good.
Here -Dₚxᵢ = (σ Xᵢ⁻² + wᵢ cᵢcᵢ'/(cᵢ'xᵢ)²)⁻¹, so this is exactly the
PD Schur complement when r₂ = r₃ = 0 and xᵢ∘sᵢ = σ1.
If supplied, `sensitivity` receives σΣᵢ∂σxᵢ. `supports` lists the positive
valuation indices per buyer; zero-valued coordinates contribute analytically.
"""
function linear_perturbed_hessian!(H, V, diagonal, c, w, p, x, σ;
    sensitivity=nothing, supports=nothing, workspace=nothing)
    n, m = size(c)
    # A zero valuation contributes σ to the scaled Hessian diagonal and σ/p
    # to σ∂σx. Start with this contribution for every buyer, then replace the
    # entries on each buyer's valuation support.
    fill!(diagonal, m * σ)
    fill!(V, 0.0)
    sensitivity === nothing || (sensitivity .= (m * σ) ./ p)
    γ = Vector{Float64}(undef, n)
    @inbounds for i in 1:m
        J = supports === nothing ? axes(c, 1) : supports[i]
        u = 0.0
        for j in J
            u += c[j, i] * x[j, i]
        end
        jmax = first(J)
        for j in J
            γ[j] = c[j, i] * x[j, i] / u
            γ[j] > γ[jmax] && (jmax = j)
        end
        rest, rest_mass = 0.0, 0.0
        for j in J
            if j != jmax
                rest += γ[j]^2
                rest_mass += γ[j]
            end
        end
        largest = γ[jmax]^2
        τ = σ / w[i]
        denom = τ + largest + rest
        scale = 1 / σ
        vscale = sqrt(scale / denom)
        for j in J
            spend = p[j] * x[j, i]
            others = j == jmax ? rest : largest + max(0.0, rest - γ[j]^2)
            diagonal[j] += scale * spend^2 * (τ + others) / denom - σ
            V[j, i] = vscale * spend * γ[j]
            if sensitivity !== nothing
                numerator = j == jmax ? τ + rest - γ[jmax] * rest_mass : denom - γ[j]
                sensitivity[j] += x[j, i] * numerator / denom - σ / p[j]
            end
        end
    end
    mul!(H, V, V', -1.0, 0.0)
    @inbounds for j in 1:n
        H[j, j] = diagonal[j]
    end
    return H
end

"""
    linear_perturbed_ipm(f; μ₀=sum(f.w)/(n*(m+1)), σ₀=μ₀, σ_min=1e-3 * minimum(f.w) / size(f.c, 1), tol=1e-6, ...)

Price-only Newton method with exact unconstrained conjugate responses.
Array types select the backend; CUDA arrays require linear-gpu.jl.
The continuation and line search are shared by CPU/GPU and direct/PCG solves.
`f` supplies `c` (goods × buyers) and `w`; supplies are one. Every buyer and
good must have a positive valuation. The default is CPU Cholesky or GPU cuDSS;
`linear_solver` also accepts :cg, :pcg_jacobi, :pcg_rowsum, :pcg_dr1,
:pcg_optimal (requires `optimal_solver`), :pcg_ic (incomplete Cholesky), and
the approximate :dr1 direction. IC uses `ic_drop=0` (IC(0)) and
`ic_normalize=true` by default; its setup is included in the reported timings.

At the current exact responses, form r=P(1-Σxᵢ)-μ1, B=H+μI and
v=σΣᵢ∂σxᵢ. A joint Newton predictor solves

    B d = -r - θσ*Pv - θμ*μ1,  Δp=Pd, Δσ=-θσ*σ, Δμ=-θμ*μ.

`reduction` (default 0.4) is the attempted retained price-barrier fraction.
`σ_min=1e-3 * minimum(f.w) / size(f.c, 1)` sets the absolute buyer floor,
corresponding to 0.1% maximum relative budget excess (up to rounding).
Require 0 ≤ σ_min ≤ σ₀. Set σ_min=σ₀ to fix σ from the start.
Require σ₀ ≥ μ₀ and maintain σ ≥ μ at every accepted iterate and trial.
Set a nonnegative absolute `σ_min` explicitly to choose a different floor;
`σ_min=0` reproduces the historical trajectory. The initial σ/μ ratio is
preserved until σ reaches its floor; thereafter only μ continues decreasing.
Each trial recomputes exact responses at the trial prices and parameters.
The continuation merit is max(||1-Σxᵢ-μ/p||∞, nκμ/min(w)), κ=σ₀/μ₀;
when μ is fixed, the corrector uses only the residual. Actual budget error
nσ/min(w) is always reported separately, including once σ is fixed.
Every outer step solves one price system, reused through backtracking.
`linear_solver=:cholesky` uses one Cholesky factorization per step.
`linear_initial_guess=:dr1` initializes each iterative solve with the DR1
approximation; `:zero` starts from zero (default).
Iterative methods use Krylov's preconditioned stopping tolerance `linear_rtol`
and report the final original-system residual separately; failures
are reported without a hidden direct-solver fallback. `record_systems=true`
retains each matrix and right-hand side for comparisons on identical systems.
`average_linear_iterations` records the mean Krylov iterations per attempted
linear solve, including failed solves; trajectory entries use all solves so far.
It is zero before any solve and for direct methods. `linear_solves` records the
denominator, and `linear_stats.iterations` retains each solve's iteration count.
σ is an absolute buyer complementarity target. Its floor imposes the
irreducible relative budget error n*σ_min/minimum(w). A run that reaches
this floor and solves the perturbed market, but misses the original tolerance,
returns :sigma_floor_reached rather than :converged.
This is a safeguarded continuation method, not a short-step complexity claim.

By default, reports :converged when actual market, relative budget, and
relative utility errors are all ≤ `tol`. An optional `stop(metrics)` predicate
replaces this convergence check and is evaluated at each accepted state,
including initialization. This lets comparisons share exactly one stopping
rule. Failure and :sigma_floor_reached statuses remain distinct.
At the floor, :sigma_floor_reached requires
both ||F||∞ and max(μ./p) ≤ tol/4, so market clearing is still checked.
Initialization, oracle work, line searches, assembly and solves are included in `elapsed`; the trajectory records accepted iterates including
the initial and final states. The input market is not modified.
"""
function linear_perturbed_ipm(f;
    μ₀=sum(f.w) / (size(f.c, 1) * (size(f.c, 2) + 1)), σ₀=μ₀, σ_min=1e-3 * minimum(f.w) / size(f.c, 1),
    p₀=nothing, tol=1e-6, maxiter=200, reduction=0.4,
    oracle_tol=min(1e-12, tol / 100), max_backtrack=40,
    show_trace=true, show_every=1, linear_solver=:auto,
    linear_rtol=1e-9, linear_maxiter=max(20, 5size(f.c, 1)),
    optimal_solver=nothing, record_systems=false, linear_initial_guess=:zero, stop=nothing,
    ic_drop=0.0, ic_normalize::Bool=true, time_limit=Inf, progress=nothing
)
    time_limit>0 || throw(ArgumentError("time_limit must be positive"))
    t0 = time_ns()
    c, w = linear_arrays(f.c, f.w)
    linear_solver = linear_solver == :auto ? linear_default_solver(c) : linear_solver
    linear_validate_solver(c, linear_solver)
    n, m = size(c)
    n > 0 && m > 0 && length(w) == m || throw(ArgumentError("incompatible market dimensions"))
    linear_validate_market(c, w)
    isfinite(σ₀) && σ₀ > 0 && isfinite(μ₀) && μ₀ > 0 || throw(ArgumentError("σ₀ and μ₀ must be positive and finite"))
    σ₀ >= μ₀ || throw(ArgumentError("σ₀ must be at least μ₀ to maintain σ ≥ μ"))
    0 < tol < 1 && 0 < reduction < 1 && 0 < oracle_tol < tol || throw(ArgumentError("invalid tolerance or reduction"))
    maxiter >= 0 && max_backtrack > 0 && show_every > 0 || throw(ArgumentError("invalid iteration limits"))
    supports = linear_supports(c)
    response_workspace = linear_response_workspace(c, w)
    metrics_workspace = linear_metrics_workspace(c)
    solver_workspace = linear_solver_workspace(c, w, linear_solver)
    min_budget = minimum(w)
    sigma_floor = Float64(σ_min)
    isfinite(sigma_floor) && 0 <= sigma_floor <= σ₀ ||
        throw(ArgumentError("σ_min must be finite and between zero and σ₀"))
    budget_error_floor = n * sigma_floor / min_budget
    p = linear_prices(c, w, p₀)
    length(p) == n || throw(ArgumentError("p₀ has wrong length"))
    buffers = linear_ipm_buffers(c, w, solver_workspace)
    st = PerturbedLinearState(p, buffers.x, buffers.s, linear_zeros(w, m), linear_zeros(w, m),
        Float64(σ₀), Float64(μ₀), :PerturbedLinear)
    H, V, diagonal = buffers.H, buffers.V, buffers.diagonal
    sensitivity = buffers.sensitivity
    trial_p, trial_x, trial_s = similar(p), similar(st.x), similar(st.s)
    trial_λ, trial_δ = similar(st.λ), similar(st.δ)
    factor_count, oracle_count, scalar_iterations = 0, 0, 0
    total_linear_iterations, linear_solves = 0, 0
    preconditioner_factorizations, sdp_iterations = 0, 0
    systems, linear_stats = NamedTuple[], NamedTuple[]
    t_preconditioner = 0.0
    t_oracle, t_build, t_solve = 0.0, 0.0, 0.0
    t_oracle += @elapsed scalar_iterations += linear_perturbed_response!(
        st.x, st.s, st.λ, st.δ, c, w, st.p, st.σ; rtol=oracle_tol, supports, workspace=response_workspace)
    oracle_count += 1
    traj = NamedTuple[]
    status = :max_iter
    last_α = 0.0
    last_parameter_ratio = 1.0
    last_sigma_ratio = 1.0
    backtracks = 0
    buyer_price_ratio = st.σ / st.μ
    show_trace && @printf("%4s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %7s | %8s\n",
        "k", "σ", "μ", "nσ", "mc", "spent", "budget", "regret", "α", "time(s)")

    for k in 0:maxiter
        metrics = linear_equilibrium_metrics(c, w, st.p, st.x; workspace=metrics_workspace)
        g = 1 .- linear_demand(st.x)
        F = g .- st.μ ./ st.p
        elapsed = (time_ns() - t0) / 1e9
        push!(traj, merge((k=k, σ=st.σ, μ=st.μ, nσ=n * st.σ, α=last_α, t=elapsed,
                factorizations=factor_count, oracle_calls=oracle_count,
                average_linear_iterations=linear_solves == 0 ? 0.0 : total_linear_iterations / linear_solves, linear_solves=linear_solves,
                merit=max(norm(F, Inf), n * (sigma_floor == 0 ? st.σ : buyer_price_ratio * st.μ) / min_budget),
                sigma_floor=sigma_floor, sigma_floor_active=sigma_floor > 0 && st.σ == sigma_floor,
                sigma_parameter_ratio=last_sigma_ratio,
                parameter_ratio=last_parameter_ratio, backtracks=backtracks,
                t_oracle=t_oracle, t_build=t_build, t_solve=t_solve), metrics))
        if show_trace && k % show_every == 0
            @printf("%4d | %9.2e | %9.2e | %9.2e | %9.2e | %9.6f | %9.6f | %9.2e | %7.4f | %8.3f\n",
                k, st.σ, st.μ, n * st.σ, metrics.mc, metrics.total_spending,
                metrics.total_budget, metrics.regret, last_α, elapsed)
            flush(stdout)
        end
        progress === nothing || progress(traj[end])
        if (time_ns()-t0)/1e9 >= time_limit
            status = :time_limit
            break
        end
        if stop === nothing ? metrics.error <= tol : stop(metrics)
            status = :converged
            break
        end
        if sigma_floor > 0 && st.σ == sigma_floor && norm(F, Inf) <= tol / 4 &&
           maximum(st.μ ./ st.p) <= tol / 4
            status = :sigma_floor_reached
            break
        end
        k == maxiter && break

        # Joint predictor: change prices and both barriers at the same time.
        # The derivative is evaluated at the current exact conjugate response.
        μ_floor = min(tol * minimum(st.p) / 8,
            tol * min_budget / (8n * buyer_price_ratio))
        target_μ = min(st.μ, max(μ_floor, reduction * st.μ))
        θ = 1 - target_μ / st.μ
        θσ = θ
        target_σ = buyer_price_ratio * target_μ
        hits_sigma_floor = false
        if sigma_floor > 0
            if st.σ == sigma_floor
                θσ = 0.0
                target_σ = sigma_floor
            elseif θ >= 1 - sigma_floor / st.σ
                # Land on the buyer floor before continuing μ independently.
                θ = θσ = 1 - sigma_floor / st.σ
                target_σ = sigma_floor
                hits_sigma_floor = true
            else
                target_σ = max(sigma_floor, (1 - θσ) * st.σ)
            end
        end
        t_build += @elapsed begin
            linear_perturbed_hessian!(H, V, diagonal, c, w, st.p, st.x, st.σ; sensitivity, supports, workspace=response_workspace)
            linear_shift_diagonal!(H, st.μ)
            linear_sync(H)
        end
        if (time_ns()-t0)/1e9 >= time_limit
            status = :time_limit
            break
        end
        # Once σ is fixed, its continuation term disappears from the rhs.
        d = θσ == θ ? -(st.p .* g .- st.μ) .- θ .* (st.p .* sensitivity .+ st.μ) :
            -(st.p .* g .- st.μ) .- θσ .* st.p .* sensitivity .- θ * st.μ
        row_sum = st.p .* linear_demand(st.x) .+ st.μ
        if record_systems
            push!(systems, (k=k, B=copy(H), b=copy(d), V=copy(V),
                row_sum=copy(row_sum), p=copy(st.p), σ=st.σ, μ=st.μ,
                θμ=θ, θσ=θσ, sigma_floor=sigma_floor))
        end
        solve_result = nothing
        t_solve += @elapsed solve_result = linear_perturbed_solve(H, d;
            method=linear_solver, row_sum, V, optimal_solver, initial_guess=linear_initial_guess,
            rtol=linear_rtol, maxiter=linear_maxiter, ic_drop, ic_normalize, workspace=solver_workspace)
        factor_count += solve_result.factorizations
        preconditioner_factorizations += solve_result.preconditioner_factorizations
        sdp_iterations += solve_result.sdp_iterations
        total_linear_iterations += solve_result.iterations
        linear_solves += 1
        t_preconditioner += solve_result.setup_seconds
        push!(linear_stats, (k=k, σ=st.σ, μ=st.μ, nσ=n * st.σ, method=linear_solver,
            initial_guess=solve_result.initial_guess,
            initial_residual=solve_result.initial_relative_residual,
            initializer_applications=solve_result.initializer_applications,
            iterations=solve_result.iterations, residual=solve_result.relative_residual,
            setup_seconds=solve_result.setup_seconds, solve_seconds=solve_result.solve_seconds,
            success=solve_result.success))
        if (time_ns()-t0)/1e9 >= time_limit
            status = :time_limit
            break
        end
        if !solve_result.success
            status = linear_solver in (:cholesky, :cudss) ? :factorization_failed : :linear_solve_failed
            break
        end
        d = solve_result.direction
        α = linear_max_step(d)
        accepted = false
        base_norm = norm(F, Inf)
        driver_sigma = sigma_floor == 0 ? st.σ : buyer_price_ratio * st.μ
        base_merit = θ > 0 ? max(base_norm, n * driver_sigma / min_budget) : base_norm
        trial_μ, trial_σ = st.μ, st.σ
        for bt in 1:max_backtrack
            if (time_ns()-t0)/1e9 >= time_limit
                status = :time_limit
                break
            end
            trial_p .= st.p .* (1 .+ α .* d)
            trial_μ = (1 - α * θ) * st.μ
            trial_σ = sigma_floor == 0 ? buyer_price_ratio * trial_μ :
                      max(sigma_floor, α == 1 && hits_sigma_floor ? sigma_floor :
                                       (1 - α) * st.σ + α * target_σ)
            # Preserve σ ≥ μ also under floating-point trial interpolation.
            trial_σ = max(trial_μ, trial_σ)
            # The distance from the active price boundary is O(σ).
            trial_δ .= (trial_σ / st.σ) .* st.δ
            t_oracle += @elapsed scalar_iterations += linear_perturbed_response!(
                trial_x, trial_s, trial_λ, trial_δ, c, w, trial_p, trial_σ; rtol=oracle_tol, supports, workspace=response_workspace)
            oracle_count += 1
            trial_F = 1 .- linear_demand(trial_x) .- trial_μ ./ trial_p
            trial_driver = sigma_floor == 0 ? trial_σ : buyer_price_ratio * trial_μ
            trial_merit = θ > 0 ? max(norm(trial_F, Inf), n * trial_driver / min_budget) : norm(trial_F, Inf)
            if trial_merit <= (1 - 1e-4 * α * (θ > 0 ? θ : 1)) * base_merit
                accepted = true
                break
            end
            backtracks += 1
            α *= 0.5
        end
        if !accepted
            status == :time_limit || (status = :line_search_failed)
            break
        end
        last_parameter_ratio = 1 - α * θ
        last_sigma_ratio = trial_σ / st.σ
        st.μ = trial_μ
        st.σ = trial_σ
        st.p .= trial_p
        linear_copy!(st.x, trial_x)
        linear_copy!(st.s, trial_s)
        st.λ .= trial_λ
        st.δ .= trial_δ
        last_α = α
    end
    # Record the final work counters on failure as well as the returned state.
    # Rejected trials never change the current prices or barrier parameters.
    if !(status in (:converged, :sigma_floor_reached))
        metrics = linear_equilibrium_metrics(c, w, st.p, st.x; workspace=metrics_workspace)
        F = 1 .- linear_demand(st.x) .- st.μ ./ st.p
        push!(traj, merge((k=traj[end].k, σ=st.σ, μ=st.μ, nσ=n * st.σ, α=0.0,
                t=(time_ns() - t0) / 1e9, factorizations=factor_count,
                oracle_calls=oracle_count, t_oracle=t_oracle,
                average_linear_iterations=linear_solves == 0 ? 0.0 : total_linear_iterations / linear_solves, linear_solves=linear_solves,
                merit=max(norm(F, Inf), n * (sigma_floor == 0 ? st.σ : buyer_price_ratio * st.μ) / min_budget),
                sigma_floor=sigma_floor, sigma_floor_active=sigma_floor > 0 && st.σ == sigma_floor,
                sigma_parameter_ratio=1.0,
                parameter_ratio=1.0, backtracks=backtracks,
                t_build=t_build, t_solve=t_solve), metrics))
    end
    return (st=st, traj=traj, status=status, sigma_floor=sigma_floor,
        budget_error_floor=budget_error_floor, factorizations=factor_count,
        oracle_calls=oracle_count, scalar_iterations=scalar_iterations, backtracks=backtracks,
        elapsed=(time_ns() - t0) / 1e9,
        t_oracle=t_oracle, t_build=t_build, t_solve=t_solve,
        t_preconditioner=t_preconditioner, average_linear_iterations=linear_solves == 0 ? 0.0 : total_linear_iterations / linear_solves,
        linear_solves=linear_solves, linear_stats=linear_stats, systems=systems,
        preconditioner_factorizations=preconditioner_factorizations, sdp_iterations=sdp_iterations)
end

"""Copy the perturbed allocation, prices and utilities into an existing market."""
function linear_perturbed_allocation!(f, st::PerturbedLinearState)
    f.p .= st.p
    f.x .= st.x
    f.val_u .= vec(sum(f.c .* st.x; dims=1))
    f.sumx .= linear_demand(st.x)
    return f
end

include("linear-sparse.jl")

"""Shared PR / perturbed IPM / PCG entry point for every linear utility source."""
function run_linear(method,c,w;tol=1e-4,maxiter_pr=40000,maxiter_ipm=200,
    time_limit=300.0,direct_solver=:auto,linear_rtol=1e-9,
    linear_maxiter=max(20,5size(c,1)),show_trace=false,show_every=1,
    stop=z->linear_meets_target(z,tol),progress=nothing)
    method==:pr && return run_pr(c,w;stop,maxiter=maxiter_pr,time_limit,progress)
    method in (:perturbed,:pcg) || throw(ArgumentError("unknown linear comparison method: $method"))
    linear_perturbed_ipm((;c,w);tol,stop,progress,time_limit,
        σ_min=tol*minimum(w)/size(c,1),maxiter=maxiter_ipm,
        linear_solver=method==:pcg ? :pcg_ic : direct_solver,
        linear_rtol,linear_maxiter,show_trace,show_every,ic_drop=0.0,ic_normalize=true)
end
