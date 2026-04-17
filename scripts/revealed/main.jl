# Column generation: iteratively add new CES androids to fit revealed preferences
include("./master.jl")
include("./pricing.jl")

"""
    cg(Ξ; max_iters=50, tol=1e-3, verbose=true, drop=true)

Standard column generation with LBFGS pricing.
Each iteration solves a nonconvex pricing problem over (y, σ) jointly
to find the single best improving column.

If `drop=true`, zero-weight agents are removed after each master solve.

Returns: (fa, γ_ref, history)
"""
function cg(Ξ; max_iters=50, tol=1e-3, verbose=true, drop=true)
    n = length(Ξ[1][1])

    # Initialize surrogate market with one random CES agent
    fa = FisherMarket(1, n; ρ=rand(1), scale=30.0, sparsity=0.99)
    γ_ref = Ref(compute_gamma_from_market(fa, Ξ))

    history = Dict(
        :primal_obj => Float64[],

        :reduced_cost => Float64[],
        :num_agents => Int[]
    )

    verbose && println("=== Column Generation (standard) ===\n")

    for iter in 1:max_iters
        verbose && println("--- Iteration $iter ($(fa.m) agents) ---")

        # Solve master problem and extract duals
        w, s, model_primal, balance, budget = solve_master_problem(Ξ, γ_ref[]; verbose=false)
        primal_obj = objective_value(model_primal)
        K = length(Ξ); n_obs = length(Ξ[1][1])
        u, μ = extract_duals(model_primal, balance, budget, K, n_obs)

        if verbose
            println("  Primal obj: $(round(primal_obj, digits=6))")
        end

        push!(history[:primal_obj], primal_obj)
        push!(history[:num_agents], fa.m)

        # Update weights and drop zero-weight columns
        if drop
            ndrop = drop_zero_columns!(fa, γ_ref, w)
            ndrop > 0 && verbose && println("  Dropped $ndrop zero-weight agents ($(fa.m) remain)")
        else
            fa.w .= w
        end

        # Solve pricing: dual LP warm-starts the nonconvex pricing
        y_lp, σ_lp, _, _ = solve_pricing_dual_lp(Ξ, u)
        y_opt, σ_opt, γ_new, pricing_obj = solve_pricing(Ξ, u; y_init=y_lp, σ_init=σ_lp)

        rc = reduced_cost(γ_new, u, μ)
        push!(history[:reduced_cost], rc)

        verbose && println("  Pricing σ:  $(round(σ_opt, digits=4)),  rc: $(round(rc, digits=6))")

        if rc <= tol
            verbose && println("\n✓ Converged! (reduced cost ≤ $tol)")
            break
        end

        # Add new column
        add_to_gamma!(γ_ref, γ_new)
        c_new, ρ_new = recover_ces_params(y_opt, σ_opt)
        add_to_market!(fa, c_new, ρ_new, 0.0)

        verbose && println("  → Added agent $(fa.m)")

        if iter == max_iters
            verbose && println("\n⚠ Maximum iterations reached")
        end
    end

    # Final weights
    w_final, _, _, _, _ = solve_master_problem(Ξ, γ_ref[]; verbose=false)
    if drop
        drop_zero_columns!(fa, γ_ref, w_final)
    else
        fa.w .= w_final
    end

    if verbose
        println("\n=== Results ===")
        println("Agents: $(fa.m)")
        println("Final obj: $(round(history[:primal_obj][end], digits=6))")
    end

    return fa, γ_ref, history
end

"""
    solve_pricing_inversion(Ξ, u; σ_grid=range(-0.9, 30.0, length=50))

Multicut pricing via single-k inversion + 1D line search over σ.

For each observation k = 1,...,K:
1. Shift u_k to be positive and normalize to the simplex: ũ_k = (u_k - min(u_k) + ε) / ||·||₁.
   This is valid because softmax outputs lie on the simplex, so shifting u_k
   by a constant does not change the pricing objective (Σ_k u_k^T γ_k shifts by a constant).
2. Set the target expenditure share γ_k* = ũ_k (the CES consumer whose demand at
   prices p_k matches the normalized dual). Invert the softmax log-ratio equations:
       log(ũ_{k,j}/ũ_{k,1}) = y_j - σ (log p_{k,j} - log p_{k,1}),   j = 2,...,n
   with y_1 = 0. For any σ, this gives y_j(σ) = L_j + σ ℓ_j in closed form,
   where L_j = log(ũ_{k,j}/ũ_{k,1}) and ℓ_j = log(p_{k,j}/p_{k,1}).
3. Search over σ (grid + Brent refinement) to maximize the full pricing objective
   F(σ) = Σ_{k'} u_{k'}^T softmax(y(σ) - σ log p_{k'}).

This replaces the nonconvex (n+1)-dimensional pricing problem with K independent
1D line searches. For each σ, y is given in closed form by the inversion —
no iterative solve is needed.

Returns all K candidate columns as a vector of (y, σ, γ, obj).
"""
function solve_pricing_inversion(
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    σ_grid=range(-0.9, 30.0, length=50)
) where T

    K = length(Ξ)
    n = length(Ξ[1][1])

    # Precompute log-ratio data for each k
    log_p = [log.(Ξ[k][1]) for k in 1:K]

    results = Vector{Tuple{Vector{T},T,Matrix{T},T}}()

    for k in 1:K
        # Shift u_k so all entries are positive (softmax is shift-invariant on the simplex)
        u_k = u[k, :] .- minimum(u[k, :]) .+ 1e-8
        u_k = u_k ./ sum(u_k)  # normalize to simplex

        # Log-ratios: L_j = log(u_{k,j} / u_{k,1}), ℓ_j = log(p_{k,j} / p_{k,1})
        L = log.(u_k[2:end] ./ u_k[1])
        ℓ_k = log_p[k][2:end] .- log_p[k][1]

        # For each σ: y_j = L_j + σ ℓ_j (j≥2), y_1 = 0
        # Evaluate full pricing objective and pick best σ
        function pricing_obj(σ)
            y = zeros(T, n)
            for j in 2:n
                y[j] = L[j-1] + σ * ℓ_k[j-1]
            end
            val = zero(T)
            for k2 in 1:K
                k2 == k && continue  # anchor term is constant in σ
                z = y .- σ .* log_p[k2]
                γ = exp.(z .- logsumexp(z))
                val += dot(u[k2, :], γ)
            end
            return val
        end

        # 1D search: grid + refinement
        best_σ = zero(T)
        best_obj = pricing_obj(best_σ)
        for σ_try in σ_grid
            obj = pricing_obj(σ_try)
            if obj > best_obj
                best_obj = obj
                best_σ = σ_try
            end
        end

        # Refine with golden section on bracket around best_σ
        idx = findfirst(s -> s == best_σ, collect(σ_grid))
        if !isnothing(idx)
            lo = idx > 1 ? σ_grid[idx-1] : σ_grid[1]
            hi = idx < length(σ_grid) ? σ_grid[idx+1] : σ_grid[end]
            res = optimize(σ -> -pricing_obj(σ), lo, hi, Brent())
            best_σ = Optim.minimizer(res)
            best_obj = -Optim.minimum(res)
        end

        # Build the column
        y_opt = zeros(T, n)
        for j in 2:n
            y_opt[j] = L[j-1] + best_σ * ℓ_k[j-1]
        end
        γ_new = produce_gamma(Ξ, y_opt, best_σ)
        push!(results, (y_opt, best_σ, γ_new, best_obj))
    end
    return results
end

"""
    solve_pricing_leastsq(Ξ, u; verbose=false)

Closed-form least-squares pricing: find (y, σ) that best fits
γ_k = u_k / ||u_k||_1 for all k simultaneously.

Solves: min_{y₂,...,yₙ,σ} Σ_k Σ_{j≥2} (L_{k,j} - y_j + σ ℓ_{k,j})²
where L_{k,j} = log(u_{k,j}/u_{k,1}), ℓ_{k,j} = log(p_{k,j}/p_{k,1}).

Returns: (y, σ, γ, obj)
"""
function solve_pricing_leastsq(
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    verbose=false
) where T

    K = length(Ξ)
    n = length(Ξ[1][1])

    log_p = [log.(Ξ[k][1]) for k in 1:K]

    # Build the linear system: A x = b
    # x = (y_2, ..., y_n, σ)  ∈ R^n
    # For each (k, j≥2): row is (e_j, -ℓ_{k,j}), rhs is L_{k,j}
    nrows = K * (n - 1)
    ncols = n  # (n-1) for y_{2:n} + 1 for σ
    A = zeros(T, nrows, ncols)
    b = zeros(T, nrows)

    row = 0
    for k in 1:K
        u_k = u[k, :]
        if any(u_k .<= 1e-15)
            # Fill with zeros (skip this k)
            row += n - 1
            continue
        end
        for j in 2:n
            row += 1
            A[row, j-1] = 1.0                                  # y_j
            A[row, n] = -(log_p[k][j] - log_p[k][1])          # -ℓ_{k,j}
            b[row] = log(u_k[j] / u_k[1])                     # L_{k,j}
        end
    end

    # Solve least squares
    x = A \ b
    y = zeros(T, n)
    y[2:end] = x[1:n-1]
    σ = x[n]

    γ = produce_gamma(Ξ, y, σ)
    obj = sum(dot(u[k, :], γ[k, :]) for k in 1:K)

    if verbose
        residual = norm(A * x - b)
        println("Least-squares pricing: σ=$(round(σ, digits=4)), obj=$(round(obj, digits=6)), residual=$(round(residual, digits=6))")
    end

    return y, σ, γ, obj
end

"""
    cg_multicut(Ξ; max_iters=50, tol=1e-3, verbose=true, drop=true)

Column generation with multicut pricing via single-k inversion.
Each iteration generates up to K candidate columns (one per observation)
via closed-form inversion + 1D line search, and adds all improving ones.

If `drop=true`, zero-weight agents are removed after each master solve.

Falls back to least-squares consensus when inversions fail.

Returns: (fa, γ_ref, history)
"""
function cg_multicut(Ξ; max_iters=50, tol=1e-3, verbose=true, drop=true)
    n = length(Ξ[1][1])

    # Initialize surrogate market with one random CES agent
    fa = FisherMarket(1, n; ρ=rand(1), scale=30.0, sparsity=0.99)
    γ_ref = Ref(compute_gamma_from_market(fa, Ξ))

    history = Dict(
        :primal_obj => Float64[],

        :num_agents => Int[],
        :cols_added => Int[]
    )

    verbose && println("=== Column Generation (multicut) ===\n")

    for iter in 1:max_iters
        verbose && println("--- Iteration $iter ($(fa.m) agents) ---")

        # Solve master and extract duals
        w, s, model_primal, balance, budget = solve_master_problem(Ξ, γ_ref[]; verbose=false)
        primal_obj = objective_value(model_primal)
        K = length(Ξ); n_obs = length(Ξ[1][1])
        u, μ = extract_duals(model_primal, balance, budget, K, n_obs)

        if verbose
            println("  Primal obj: $(round(primal_obj, digits=6))")
        end

        push!(history[:primal_obj], primal_obj)
        push!(history[:num_agents], fa.m)

        # Update weights and drop zero-weight columns
        if drop
            ndrop = drop_zero_columns!(fa, γ_ref, w)
            ndrop > 0 && verbose && println("  Dropped $ndrop zero-weight agents ($(fa.m) remain)")
        else
            fa.w .= w
        end

        # Multicut pricing: inversion + 1D search per k
        candidates = solve_pricing_inversion(Ξ, u)

        verbose && println("  Candidates: $(length(candidates))")

        # Add all improving columns
        added = 0
        for (y_opt, σ_opt, γ_new, rc) in candidates
            add_to_gamma!(γ_ref, γ_new)
            c_new, ρ_new = recover_ces_params(y_opt, σ_opt)
            add_to_market!(fa, c_new, ρ_new, 0.0)
            added += 1
        end

        push!(history[:cols_added], added)
        verbose && println("  → Added $added agents (total: $(fa.m))")

        # Stop if primal objective not improving over last 2 iterations
        if length(history[:primal_obj]) >= 3
            prev = history[:primal_obj][end-1]
            prev2 = history[:primal_obj][end-2]
            improvement = max(prev2 - prev, prev - primal_obj)
            if improvement < tol
                verbose && println("\n✓ Converged! (primal obj stalled, Δ < $tol)")
                break
            end
        end

        if iter == max_iters
            verbose && println("\n⚠ Maximum iterations reached")
        end
    end

    # Final weights
    w_final, _, _, _, _ = solve_master_problem(Ξ, γ_ref[]; verbose=false)
    if drop
        drop_zero_columns!(fa, γ_ref, w_final)
    else
        fa.w .= w_final
    end

    if verbose
        println("\n=== Results ===")
        println("Agents: $(fa.m),  nonzero: $(sum(w_final .> 1e-6))")
        println("Final obj: $(round(history[:primal_obj][end], digits=6))")
    end

    return fa, γ_ref, history
end

"""
    cg_lsq(Ξ; max_iters=50, tol=1e-3, verbose=true, drop=true)

Column generation with least-squares consensus pricing.
Each iteration solves a closed-form linear least-squares problem to find
(y, σ) that best fits the dual variables across all K observations,
producing one column per iteration.

If `drop=true`, zero-weight agents are removed after each master solve.

Returns: (fa, γ_ref, history)
"""
function cg_lsq(Ξ; max_iters=50, tol=1e-3, verbose=true, drop=true)
    n = length(Ξ[1][1])

    # Initialize surrogate market with one random CES agent
    fa = FisherMarket(1, n; ρ=rand(1), scale=30.0, sparsity=0.99)
    γ_ref = Ref(compute_gamma_from_market(fa, Ξ))

    history = Dict(
        :primal_obj => Float64[],

        :num_agents => Int[]
    )

    verbose && println("=== Column Generation (least-squares) ===\n")

    for iter in 1:max_iters
        verbose && println("--- Iteration $iter ($(fa.m) agents) ---")

        # Solve master and extract duals
        w, s, model_primal, balance, budget = solve_master_problem(Ξ, γ_ref[]; verbose=false)
        primal_obj = objective_value(model_primal)
        K = length(Ξ); n_obs = length(Ξ[1][1])
        u, μ = extract_duals(model_primal, balance, budget, K, n_obs)

        if verbose
            println("  Primal obj: $(round(primal_obj, digits=6))")
        end

        push!(history[:primal_obj], primal_obj)
        push!(history[:num_agents], fa.m)

        # Update weights and drop zero-weight columns
        if drop
            ndrop = drop_zero_columns!(fa, γ_ref, w)
            ndrop > 0 && verbose && println("  Dropped $ndrop zero-weight agents ($(fa.m) remain)")
        else
            fa.w .= w
        end

        # Least-squares pricing
        y_opt, σ_opt, γ_new, pricing_obj = solve_pricing_leastsq(Ξ, u; verbose=false)

        verbose && println("  Pricing σ:  $(round(σ_opt, digits=4))")

        if length(history[:primal_obj]) >= 3
            prev = history[:primal_obj][end-1]
            prev2 = history[:primal_obj][end-2]
            improvement = max(prev2 - prev, prev - primal_obj)
            if improvement < tol
                verbose && println("\n✓ Converged! (primal obj stalled, Δ < $tol)")
                break
            end
        end

        # Add new column
        add_to_gamma!(γ_ref, γ_new)
        c_new, ρ_new = recover_ces_params(y_opt, σ_opt)
        add_to_market!(fa, c_new, ρ_new, 0.0)

        verbose && println("  → Added agent $(fa.m)")

        if iter == max_iters
            verbose && println("\n⚠ Maximum iterations reached")
        end
    end

    # Final weights
    w_final, _, _, _, _ = solve_master_problem(Ξ, γ_ref[]; verbose=false)
    if drop
        drop_zero_columns!(fa, γ_ref, w_final)
    else
        fa.w .= w_final
    end

    if verbose
        println("\n=== Results ===")
        println("Agents: $(fa.m)")
        println("Final obj: $(round(history[:primal_obj][end], digits=6))")
    end

    return fa, γ_ref, history
end
