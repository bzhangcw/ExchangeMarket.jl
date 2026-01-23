using LinearAlgebra
using Optim
using LogExpFunctions: logsumexp
using ExchangeMarket

#=
Two formulations for the pricing problem:

1. ORIGINAL (non-convex):
   max_{y, σ > 0} Σ_k u_k^T γ_k = Σ_k u_k^T softmax(y - σ log p_k)

2. CONVEX SURROGATE:
   max_{y, σ > 0} Σ_k u_k^T log γ_k 
   = Σ_k [u_k^T (y - σ log p_k) - (1^T u_k) · lse(y - σ log p_k)]

The convex surrogate replaces γ with log(γ), making it concave in (y, σ).
=#

"""
    solve_pricing_original(Ξ, u; σ_init=0.5, verbose=false)

Solve the ORIGINAL pricing problem (non-convex):
    max_{y ∈ ℝ^n, σ > 0} Σ_k u_k^T γ_k = Σ_k u_k^T softmax(y - σ log p_k)

Arguments:
- Ξ: Vector of (p_k, g_k) tuples, K observations
- u: Dual variables matrix of size (K, n) from the master dual problem

Returns:
- y, σ, γ_new, obj_val
"""
function solve_pricing(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    σ_init::T=0.5,
    verbose=false) where T

    K = length(Ξ)
    n = length(Ξ[1][1])

    # Objective: max Σ_k u_k^T softmax(y - σ log p_k)
    function neg_objective(x)
        y = x[1:n]
        σ = x[n+1]
        val = zero(T)
        for k in 1:K
            p_k = Ξ[k][1]
            u_k = u[k, :]
            z_k = y .- σ .* log.(p_k)
            γ_k = exp.(z_k .- logsumexp(z_k))  # softmax
            val += dot(u_k, γ_k)
        end
        return -val
    end

    # Gradient of original objective
    function neg_gradient!(G, x)
        y = x[1:n]
        σ = x[n+1]
        G .= 0.0
        for k in 1:K
            p_k = Ξ[k][1]
            u_k = u[k, :]
            log_p_k = log.(p_k)
            z_k = y .- σ .* log_p_k
            γ_k = exp.(z_k .- logsumexp(z_k))
            # ∂γ/∂z = diag(γ) - γγ^T
            # ∂(u^T γ)/∂y = (diag(γ) - γγ^T) u = γ ⊙ u - γ (γ^T u)
            coef = γ_k .* u_k .- γ_k .* dot(γ_k, u_k)
            G[1:n] .-= coef
            # ∂(u^T γ)/∂σ = -coef^T log_p
            G[n+1] -= -dot(coef, log_p_k)
        end
    end

    y_init = zeros(T, n)
    x0 = vcat(y_init, σ_init)
    lower = vcat(fill(-Inf, n), T(-0.99))  # σ > -1 required for ρ < 1
    upper = vcat(fill(Inf, n), T(100.0))

    result = optimize(
        neg_objective, neg_gradient!,
        lower, upper, x0,
        Fminbox(LBFGS()),
        Optim.Options(show_trace=verbose, iterations=1000, g_tol=1e-8)
    )

    x_opt = Optim.minimizer(result)
    y_opt, σ_opt = x_opt[1:n], x_opt[n+1]
    obj_val = -Optim.minimum(result)

    γ_new = zeros(T, K, n)
    for k in 1:K
        z_k = y_opt .- σ_opt .* log.(Ξ[k][1])
        γ_new[k, :] = exp.(z_k .- logsumexp(z_k))
    end

    verbose && println("Original pricing: σ=$σ_opt, obj=$obj_val")
    return y_opt, σ_opt, γ_new, obj_val
end

"""
    add_to_gamma!(γ_ref::Ref{Array{T,3}}, γ_new)

Add a new android's bidding vectors to the existing γ matrix in-place.
γ_ref is a Ref containing γ with shape (m, K, n), γ_new has shape (K, n).
Mutates γ_ref to contain the expanded γ with shape (m+1, K, n).
"""
function add_to_gamma!(γ_ref::Ref{Array{T,3}}, γ_new::Matrix{T}) where T
    γ = γ_ref[]
    m, K, n = size(γ)
    @assert size(γ_new) == (K, n) "γ_new must be (K, n) = ($K, $n)"
    γ_expanded = zeros(T, m + 1, K, n)
    γ_expanded[1:m, :, :] .= γ
    γ_expanded[m+1, :, :] .= γ_new
    γ_ref[] = γ_expanded
    return nothing
end

"""
    add_to_market!(f1::FisherMarket, c_new, ρ_new, w_new)

Add a new CES agent to an existing FisherMarket in-place.
- c_new: coefficient vector (n-dim)
- ρ_new: CES parameter ρ
- w_new: budget for the new agent

Uses expand_players! from ExchangeMarket.
"""
function add_to_market!(f1::FisherMarket, c_new::Vector{T}, ρ_new::T, w_new::T) where T
    expand_players!(f1, f1.m + 1;
        c_new=reshape(c_new, :, 1),
        ρ_new=[ρ_new],
        w_new=[w_new]
    )
    return f1
end

"""
    solve_pricing_convex(Ξ, u; σ_init=0.5, verbose=false)

Solve the CONVEX SURROGATE pricing problem:
    max_{y ∈ ℝ^n, σ > 0} Σ_k u_k^T log γ_k 
    = Σ_k [u_k^T (y - σ log p_k) - (1^T u_k) · lse(y - σ log p_k)]

This is concave in (y, σ), guaranteeing global optimum.

Arguments:
- Ξ: Vector of (p_k, g_k) tuples, K observations
- u: Dual variables matrix of size (K, n) from the master dual problem

Returns:
- y, σ, γ_new, obj_val
"""
function solve_pricing_convex(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    σ_init::T=0.5,
    verbose=false) where T

    K = length(Ξ)
    n = length(Ξ[1][1])

    # Objective: max Σ_k [u_k^T (y - σ log p_k) - (1^T u_k) · lse(y - σ log p_k)]
    function neg_objective(x)
        y = x[1:n]
        σ = x[n+1]
        val = zero(T)
        for k in 1:K
            p_k = Ξ[k][1]
            u_k = u[k, :]
            z_k = y .- σ .* log.(p_k)
            lse_k = logsumexp(z_k)
            sum_u_k = sum(u_k)
            val += dot(u_k, z_k) - sum_u_k * lse_k
        end
        return -val
    end

    # Gradient
    function neg_gradient!(G, x)
        y = x[1:n]
        σ = x[n+1]
        G .= 0.0
        for k in 1:K
            p_k = Ξ[k][1]
            u_k = u[k, :]
            log_p_k = log.(p_k)
            z_k = y .- σ .* log_p_k
            softmax_k = exp.(z_k .- logsumexp(z_k))
            sum_u_k = sum(u_k)
            # ∂/∂y = u_k - sum(u_k) * softmax(z_k)
            G[1:n] .-= (u_k .- sum_u_k .* softmax_k)
            # ∂/∂σ = -u_k^T log(p_k) + sum(u_k) * softmax(z_k)^T log(p_k)
            G[n+1] -= -dot(u_k, log_p_k) + sum_u_k * dot(softmax_k, log_p_k)
        end
    end

    y_init = zeros(T, n)
    x0 = vcat(y_init, σ_init)
    lower = vcat(fill(-Inf, n), T(-0.99))  # σ > -1 required for ρ < 1
    upper = vcat(fill(Inf, n), T(100.0))

    result = optimize(
        neg_objective, neg_gradient!,
        lower, upper, x0,
        Fminbox(LBFGS()),
        Optim.Options(show_trace=verbose, iterations=1000, g_tol=1e-8)
    )

    x_opt = Optim.minimizer(result)
    y_opt, σ_opt = x_opt[1:n], x_opt[n+1]
    obj_val = -Optim.minimum(result)

    γ_new = zeros(T, K, n)
    for k in 1:K
        z_k = y_opt .- σ_opt .* log.(Ξ[k][1])
        γ_new[k, :] = exp.(z_k .- logsumexp(z_k))
    end

    verbose && println("Convex pricing: σ=$σ_opt, obj=$obj_val")
    return y_opt, σ_opt, γ_new, obj_val
end

"""
    reduced_cost(γ_new, u, μ)

Compute the reduced cost for a new android with bidding vectors γ_new.
    reduced_cost = Σ_k <u_k, γ_new_k> - μ

If reduced_cost > 0, adding this android can improve the master problem.
"""
function reduced_cost(γ_new::Matrix{T}, u::Matrix{T}, μ::T) where T
    K, n = size(γ_new)
    rc = sum(dot(u[k, :], γ_new[k, :]) for k in 1:K) - μ
    return rc
end

"""
    recover_ces_params(y, σ)

Recover CES parameters (c, ρ) from the log-reparameterization (y, σ).
    y = log(c^{1+σ})  =>  c = exp(y / (1+σ))
    σ = r/(1-r)       =>  r = σ/(1+σ)
"""
function recover_ces_params(y::Vector{T}, σ::T) where T
    c = exp.(y ./ (1 + σ))
    ρ = σ / (1 + σ)
    return c, ρ
end
