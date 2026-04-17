using LinearAlgebra, AppleAccelerate
using Random
using MadNLP
using ADNLPModels
using LogExpFunctions: logsumexp

"""
    solve_nls(Ξ, t; verbose=false, seed=nothing)

Solve the nonlinear least squares problem for t CES agents jointly:
    min_{y_i, σ_i, w} Σ_k ||Σ_i w_i γ_i(p_k) - P_k g_k||²
where γ_i(p_k) = softmax(y_i - σ_i log(p_k)) and w ∈ Δ^t.

Unlike column generation (which alternates LP master + nonconvex pricing),
this determines all parameters (y_i, σ_i, w_i) simultaneously via a single
nonconvex optimization.

Arguments:
- Ξ: Vector of (p_k, g_k) tuples, K observations
- t: Number of CES agents

Keyword arguments:
- Y_init: Initial y matrix (t × n), optional
- σ_init: Initial σ vector (t,), optional
- w_init: Initial weight vector (t,), optional
- verbose: Print optimization trace
- seed: Random seed

Returns:
- Y: Optimal y matrix (t × n)
- σ_vec: Optimal σ vector (t,)
- w: Optimal weight vector (t,)
- γ_fitted: Fitted bidding matrix (t, K, n)
- result: MadNLP result object
"""
function solve_nls(Ξ::Vector{Tuple{Vector{T},Vector{T}}}, t::Int;
    Y_init::Union{Matrix{T},Nothing}=nothing,
    σ_init::Union{Vector{T},Nothing}=nothing,
    w_init::Union{Vector{T},Nothing}=nothing,
    verbose::Bool=false,
    seed::Union{Int,Nothing}=nothing) where T

    K = length(Ξ)
    n = length(Ξ[1][1])

    # Precompute targets P_k g_k and log-prices
    targets = zeros(T, K, n)
    log_prices = zeros(T, K, n)
    for k in 1:K
        p_k, g_k = Ξ[k]
        targets[k, :] = p_k .* g_k
        log_prices[k, :] = log.(p_k)
    end

    # Parameter layout:
    #   θ = [y_1(n); σ_1; y_2(n); σ_2; ...; y_t(n); σ_t; α_1; ...; α_t]
    # Weights via softmax: w = softmax(α) to enforce Δ^t
    dim_agent = n + 1
    dim_total = t * dim_agent + t

    function unpack(θ)
        S = eltype(θ)
        Y = zeros(S, t, n)
        σ_vec = zeros(S, t)
        for i in 1:t
            off = (i - 1) * dim_agent
            Y[i, :] = θ[off+1:off+n]
            σ_vec[i] = θ[off+n+1]
        end
        α = θ[t*dim_agent+1:end]
        w = exp.(α .- logsumexp(α))
        return Y, σ_vec, w
    end

    function objective(θ)
        S = eltype(θ)
        Y, σ_vec, w = unpack(θ)
        val = zero(S)
        for k in 1:K
            res_k = S.(-targets[k, :])
            for i in 1:t
                z = Y[i, :] .- σ_vec[i] .* log_prices[k, :]
                γ_ik = exp.(z .- logsumexp(z))
                res_k .+= w[i] .* γ_ik
            end
            val += dot(res_k, res_k)
        end
        return val
    end

    function gradient!(G, θ)
        Y, σ_vec, w = unpack(θ)
        G .= 0

        # Forward pass: compute all γ and residuals
        γ_all = zeros(T, t, K, n)
        residuals = zeros(T, K, n)
        for k in 1:K
            residuals[k, :] = -targets[k, :]
            for i in 1:t
                z = Y[i, :] .- σ_vec[i] .* log_prices[k, :]
                γ_all[i, k, :] = exp.(z .- logsumexp(z))
                residuals[k, :] .+= w[i] .* γ_all[i, k, :]
            end
        end

        # ∂/∂(y_i, σ_i): chain through softmax Jacobian
        for i in 1:t
            off = (i - 1) * dim_agent
            for k in 1:K
                γ_ik = @view γ_all[i, k, :]
                r_k = @view residuals[k, :]
                # (diag(γ) - γγ^T) r = γ ⊙ r - γ (γ^T r)
                γr = dot(γ_ik, r_k)
                @inbounds for j in 1:n
                    dγ_r_j = γ_ik[j] * r_k[j] - γ_ik[j] * γr
                    G[off+j] += 2 * w[i] * dγ_r_j
                    G[off+n+1] += -2 * w[i] * dγ_r_j * log_prices[k, j]
                end
            end
        end

        # ∂/∂α: chain through softmax reparameterization of w
        # ∂obj/∂w_i = 2 Σ_k γ_ik^T r_k
        dobj_dw = zeros(T, t)
        for i in 1:t
            for k in 1:K
                dobj_dw[i] += 2 * dot(γ_all[i, k, :], residuals[k, :])
            end
        end
        # ∂w_i/∂α_j = w_i(δ_{ij} - w_j)
        # ⇒ ∂obj/∂α_j = w_j(dobj_dw_j - Σ_i w_i dobj_dw_i)
        mean_dw = dot(w, dobj_dw)
        off_α = t * dim_agent
        for j in 1:t
            G[off_α+j] = w[j] * (dobj_dw[j] - mean_dw)
        end
    end

    !isnothing(seed) && Random.seed!(seed)

    θ0 = zeros(T, dim_total)
    if !isnothing(Y_init) && !isnothing(σ_init)
        for i in 1:min(t, size(Y_init, 1))
            off = (i - 1) * dim_agent
            θ0[off+1:off+n] = Y_init[i, :]
            θ0[off+n+1] = σ_init[i]
        end
        if !isnothing(w_init)
            θ0[t*dim_agent+1:end] = log.(max.(w_init, eps(T)))
        end
    else
        for i in 1:t
            off = (i - 1) * dim_agent
            θ0[off+1:off+n] = randn(T, n) * 0.5
            θ0[off+n+1] = rand(T) * 2.0
        end
    end

    lower = fill(T(-Inf), dim_total)
    upper = fill(T(Inf), dim_total)
    for i in 1:t
        off = (i - 1) * dim_agent
        lower[off+n+1] = T(-0.98)  # σ > -1 for ρ < 1
        upper[off+n+1] = T(400.0)
    end

    nlp = ADNLPModel(objective, θ0, lower, upper;
        hessian_backend=ADNLPModels.EmptyADbackend,
    )
    best_result = madnlp(nlp;
        max_iter=100, tol=1e-6,
        hessian_approximation=MadNLP.CompactLBFGS,
        linear_solver=LapackCPUSolver,
        print_level=verbose ? MadNLP.INFO : MadNLP.ERROR,
    )

    θ_opt = best_result.solution
    Y_opt, σ_opt, w_opt = unpack(θ_opt)

    γ_fitted = zeros(T, t, K, n)
    for i in 1:t
        for k in 1:K
            z = Y_opt[i, :] .- σ_opt[i] .* log_prices[k, :]
            γ_fitted[i, k, :] = exp.(z .- logsumexp(z))
        end
    end

    return Y_opt, σ_opt, w_opt, γ_fitted, best_result
end

"""
    evaluate_nls(Ξ, Y, σ_vec, w)

Evaluate the NLS fit: compute per-observation residuals and summary statistics.

Returns:
- obj: Total objective Σ_k ||residual_k||²
- residuals: Matrix (K, n) of residual vectors
- max_err: L∞ error max_k ||residual_k||_∞
"""
function evaluate_nls(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    Y::Matrix{T}, σ_vec::Vector{T}, w::Vector{T}) where T

    K = length(Ξ)
    n = length(Ξ[1][1])
    t = length(σ_vec)

    residuals = zeros(T, K, n)
    for k in 1:K
        p_k, g_k = Ξ[k]
        log_p = log.(p_k)
        target_k = p_k .* g_k
        fitted_k = zeros(T, n)
        for i in 1:t
            z = Y[i, :] .- σ_vec[i] .* log_p
            γ_ik = exp.(z .- logsumexp(z))
            fitted_k .+= w[i] .* γ_ik
        end
        residuals[k, :] = fitted_k .- target_k
    end

    obj = sum(abs2, residuals)
    max_err = maximum(abs, residuals)
    return obj, residuals, max_err
end

"""
    nls_to_gamma(Ξ, Y, σ_vec, w)

Convert NLS solution to the γ array format used by solve_master_problem.
Returns γ of size (t, K, n) and weight vector w.
"""
function nls_to_gamma(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    Y::Matrix{T}, σ_vec::Vector{T}) where T

    t = size(Y, 1)
    K = length(Ξ)
    n = length(Ξ[1][1])

    γ = zeros(T, t, K, n)
    for i in 1:t
        for k in 1:K
            p_k = Ξ[k][1]
            z = Y[i, :] .- σ_vec[i] .* log.(p_k)
            γ[i, k, :] = exp.(z .- logsumexp(z))
        end
    end
    return γ
end
