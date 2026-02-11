using LinearAlgebra
using Optim
using LogExpFunctions: logsumexp
using ExchangeMarket
using JuMP
import MathOptInterface as MOI
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
    lower = vcat(fill(-20.0, n), T(-0.99))  # σ > -1 required for ρ < 1
    upper = vcat(fill(20.0, n), T(30.0))

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

# -----------------------------------------------------------------------
# General piecewise-linear pricing
# -----------------------------------------------------------------------

"""
    compute_general_pwl_demand(p, A_planes, b_planes, w, i)

Compute demand for general piecewise linear concave utility:
    u_i(x) = min_l { a_{i,l}^T x + b_{i,l} }
by solving the LP with an auxiliary variable z.
"""
function compute_general_pwl_demand(
    p::Vector{T},
    A_planes::Array{T,3},
    b_planes::Matrix{T},
    w::T,
    i::Int
) where T
    n = length(p)
    L = size(A_planes, 2)

    md = try
        ExchangeMarket.__generate_empty_jump_model(; verbose=false, tol=1e-8)
    catch
        Model()
    end

    @variable(md, x[1:n] >= 0)
    @variable(md, z)
    for l in 1:L
        @constraint(md, z <= dot(@view(A_planes[:, l, i]), x) + b_planes[l, i])
    end
    @constraint(md, dot(p, x) <= w)
    @objective(md, Max, z)

    JuMP.optimize!(md)

    if termination_status(md) != MOI.OPTIMAL && termination_status(md) != MOI.SLOW_PROGRESS
        @warn "General PWL demand LP failed with status: $(termination_status(md))"
        return zeros(T, n)
    end

    return [max(0.0, JuMP.value(x[j])) for j in 1:n]
end
function compute_linear_demand(p::AbstractVector{T}, A_planes::Array{T,3}, w::T, i::Int) where T
    n = length(p)
    x = zeros(T, n)
    # argmax of 1/p (i.e., min price)
    ratio = A_planes[:, 1, i] ./ p
    maxv = maximum(ratio)
    idxs = findall(ratio .== maxv)
    x[idxs] .= (w / length(idxs)) ./ p[idxs]
    return x
end
"""
    solve_pricing_piecewise(Ξ, u; M=20, verbose=false)

Heuristic pricing for general PWL agents: sample M candidate agents
and keep the one maximizing sum_k <u_k, x_candidate(p^k)>.
"""
function solve_pricing_piecewise(
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    M::Int=20,
    L_max::Int=4,
    b_min::T=-1.0,
    b_max::T=1.0,
    verbose=false
) where T
    K = length(Ξ)
    n = length(Ξ[1][1])

    best_obj = -Inf
    best_params = nothing
    best_x = zeros(T, K, n)

    for _ in 1:M
        A_planes_candidate, b_planes_candidate = generate_random_piecewise_params(n; L_max=L_max, b_min=b_min, b_max=b_max)

        x_candidate = zeros(T, K, n)
        for k in 1:K
            p_k = Ξ[k][1]
            if L_max == 1
                x_candidate[k, :] = compute_linear_demand(p_k, A_planes_candidate, 1.0, 1)
            else
                x_candidate[k, :] = compute_general_pwl_demand(
                    p_k, A_planes_candidate, b_planes_candidate, 1.0, 1
                )
            end
        end

        obj = sum(dot(u[k, :], x_candidate[k, :]) for k in 1:K)
        if obj > best_obj
            best_obj = obj
            best_params = (A_planes_candidate, b_planes_candidate)
            best_x = x_candidate
        end
    end

    verbose && println("Piecewise pricing: best_obj=$best_obj")
    if best_params === nothing
        error("No valid candidate found")
    end

    A_best, b_best = best_params
    return A_best, b_best, best_x, best_obj
end
"""
    add_piecewise_to_market!(f1, A_planes_new, b_planes_new, w_new)

Add a new general PWL agent to an existing FisherMarket.
"""
function add_piecewise_to_market!(
    f1::FisherMarket,
    A_planes_new::Array{T,3},
    b_planes_new::Matrix{T},
    w_new::T
) where T
    expand_players!(f1, f1.m + 1;
        A_planes_new=A_planes_new,
        b_planes_new=b_planes_new,
        w_new=[w_new]
    )
    return f1
end

"""
    generate_random_piecewise_params(n; L_max=4, a_max=10.0, b_min=-1.0, b_max=1.0)

Generate random general PWL parameters:
- A_planes: (n, L_max, 1)
- b_planes: (L_max, 1)
"""
function generate_random_piecewise_params(
    n::Int;
    L_max=4,
    a_max=10.0,
    b_min=-1.0,
    b_max=1.0
)
    A_planes = zeros(Float64, n, L_max, 1)
    b_planes = zeros(Float64, L_max, 1)

    for l in 1:L_max
        for j in 1:n
            A_planes[j, l, 1] = rand() * a_max
        end
        b_planes[l, 1] = rand() * (b_max - b_min) + b_min
    end

    return A_planes, b_planes
end

"""
    reduced_cost_piecewise(x_new, u, μ)

Compute reduced cost for general PWL agent using demand vectors.
    rc = sum_k <u_k, x_new(p^k)> - μ
"""
function reduced_cost_piecewise(x_new::Matrix{T}, u::Matrix{T}, μ::T) where T
    K, n = size(x_new)
    rc = sum(dot(u[k, :], x_new[k, :]) for k in 1:K) - μ
    return rc
end

"""
    add_to_demand!(x_ref::Ref{Array{T,3}}, x_new)

Add a new agent's demand vectors to the existing demand matrix.
Similar to add_to_gamma! but for demand vectors.
"""
function add_to_demand!(x_ref::Ref{Array{T,3}}, x_new::Matrix{T}) where T
    x = x_ref[]
    N, K, n = size(x)
    @assert size(x_new) == (K, n) "x_new must be (K, n) = ($K, $n)"
    x_expanded = zeros(T, N + 1, K, n)
    x_expanded[1:N, :, :] .= x
    x_expanded[N+1, :, :] .= x_new
    x_ref[] = x_expanded
    return nothing
end

"""
    update_pwl_optimizer_from_market!(alg, market)

Update the PiecewiseLPResponse optimizer in the algorithm to include all agents in the market.
This is needed after column generation adds new agents.
"""
function update_pwl_optimizer_from_market!(alg, market::FisherMarket)
    if market.A_planes === nothing
        error("Market does not have general piecewise linear parameters (A_planes, b_planes)")
    end

    alg.optimizer = PiecewiseLPResponse()
    return alg
end
