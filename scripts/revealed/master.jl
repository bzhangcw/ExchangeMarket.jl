using JuMP
using LinearAlgebra
using Random
using ExchangeMarket
using Gurobi


"""
    produce_revealed_preferences(alg, f1::FisherMarket, K; price_range=(0.5, 2.0), seed=nothing)

Generate K random price vectors and compute aggregate demands from a FisherMarket.
Returns Ξ = [(p_1, g_1), ..., (p_K, g_K)] where g_k is the aggregate demand at price p_k.

Arguments:
- alg: Algorithm object (e.g., HessianBar)
- f1: FisherMarket object containing the market structure
- K: Number of price observations to generate
- price_range: (min, max) range for random prices
- seed: Random seed (optional)

After calling play!(alg, f1), the demand is computed and stored in f1.x.
"""
function produce_revealed_preferences(alg, f1::FisherMarket, K::Int;
    price_range=(0.5, 2.0), seed=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    n = f1.n
    Ξ = Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, K)

    for k in 1:K
        # Random price vector (normalized to sum to 1)
        p_k = price_range[1] .+ (price_range[2] - price_range[1]) .* rand(n)
        p_k = p_k ./ sum(p_k)  # normalize prices

        # Set price in the algorithm
        alg.p .= p_k

        # Compute demand via play!
        println("Computing demand for observation $k at prices: ", p_k)
        play!(alg, f1)
        
        # Aggregate demand: sum over all agents
        g_k = sum(f1.x, dims=2)[:]

        Ξ[k] = (copy(p_k), copy(g_k))
    end

    return Ξ
end
  
"""
    compute_gamma_from_market(f1::FisherMarket, Ξ)

Compute the bidding matrix γ[i,k,:] for a FisherMarket given revealed preferences Ξ.
Uses the market's CES parameters (c, σ) to compute bidding vectors.

Returns γ as a 3D array of size (m, K, n).
"""
function compute_gamma_from_market(f1::FisherMarket, Ξ::Vector{Tuple{Vector{T},Vector{T}}}) where T
    m, n = f1.m, f1.n
    K = length(Ξ)

    γ = zeros(T, m, K, n)
    for i in 1:m
        c_i = Vector(f1.c[:, i])  # ensure it's a dense vector
        σ_i = f1.σ[i]
        for k in 1:K
            p_k, _ = Ξ[k]
            γ[i, k, :] = compute_gamma(p_k, c_i, σ_i)
        end
    end

    return γ
end

"""
    compute_gamma(p, c, σ)

Compute the CES bidding vector γ for given price p, coefficients c, and elasticity parameter σ.
    γ_j = (c_j^{1+σ} * p_j^{-σ}) / sum_ℓ(c_ℓ^{1+σ} * p_ℓ^{-σ})
"""
function compute_gamma(p::AbstractVector, c::AbstractVector, σ::Real)
    numerator = (c .^ (1 + σ)) .* (p .^ (-σ))
    γ = numerator ./ sum(numerator)
    return γ
end

"""
    compute_gamma_matrix(Ξ, C, σ_vec)

Compute the bidding matrix γ[i,k,:] for all agents i and observations k.
- Ξ: Vector of (p_k, g_k) tuples
- C: Matrix of coefficients, C[i,:] = c_i
- σ_vec: Vector of elasticity parameters σ_i

Returns γ as a 3D array of size (m, K, n).
"""
function compute_gamma_matrix(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    C::Matrix{T},
    σ_vec::Vector{T}) where T
    m, n = size(C)
    K = length(Ξ)

    γ = zeros(T, m, K, n)
    for i in 1:m
        for k in 1:K
            p_k, _ = Ξ[k]
            γ[i, k, :] = compute_gamma(p_k, C[i, :], σ_vec[i])
        end
    end

    return γ
end

"""
    solve_master_problem(Ξ, γ; verbose=false)

Solve the master problem (primal):
    Q = min_{w, s_k} sum_{k∈[K]} ||s_k||_∞
    s.t.  s_k ∈ ℝ^n_+, s_k + sum_{i∈[m]} γ_{ik} w_i ≥ P_k g_k, ∀k∈[K]
          sum_{i∈[m]} w_i = 1, w ∈ ℝ^m_+

Arguments:
- Ξ: Vector of (p_k, g_k) tuples, K observations
- γ: Bidding matrix of size (m, K, n)

Returns:
- w: Optimal weights
- s: Slack matrix of size (K, n)
- Q: Optimal objective value
"""
function solve_master_problem(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    γ::Array{T,3};
    verbose=false) where T
    m, K, n = size(γ)

    # Create model
    model = Model(() -> Gurobi.Optimizer())
    set_silent(model)
    if verbose
        unset_silent(model)
    end

    # Variables
    @variable(model, w[1:m] >= 0)
    @variable(model, s[1:K, 1:n] >= 0)
    @variable(model, t[1:K] >= 0)  # t_k = ||s_k||_∞

    # Constraints
    # s_k + sum_i γ_{ik} w_i ≥ P_k g_k
    for k in 1:K
        p_k, g_k = Ξ[k]
        P_k_g_k = p_k .* g_k  # P_k g_k (element-wise)
        for j in 1:n
            @constraint(model, s[k, j] + sum(γ[i, k, j] * w[i] for i in 1:m) >= P_k_g_k[j])
        end
    end

    # t_k >= s_k (infinity norm)
    for k in 1:K
        for j in 1:n
            @constraint(model, t[k] >= s[k, j])
        end
    end

    # sum_i w_i = 1
    @constraint(model, sum(w) == 1)

    # Objective: min sum_k t_k
    @objective(model, Min, sum(t))

    optimize!(model)

    return value.(w), value.(s), model
end

"""
    solve_dual_problem(Ξ, γ; verbose=false)

Solve the dual problem:
    Q_* = max_{u_k, μ} sum_{k∈[K]} <u_k, P_k g_k> - μ
    s.t.  sum_{k∈[K]} <u_k, γ_{ik}> ≤ μ, ∀i∈[m]
          ||u_k||_1 ≤ 1, u_k ∈ ℝ^n_+, ∀k∈[K]

Arguments:
- Ξ: Vector of (p_k, g_k) tuples, K observations
- γ: Bidding matrix of size (m, K, n)

Returns:
- u: Dual variables matrix of size (K, n)
- μ: Dual variable (scalar)
- Q_star: Optimal objective value
"""
function solve_dual_problem(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    γ::Array{T,3};
    verbose=false) where T
    m, K, n = size(γ)

    # Create model
    model = Model(() -> Gurobi.Optimizer())
    set_silent(model)
    if verbose
        unset_silent(model)
    end

    # Variables
    @variable(model, u[1:K, 1:n] >= 0)
    @variable(model, μ)

    # Constraints
    # sum_k <u_k, γ_{ik}> ≤ μ, ∀i∈[m]
    for i in 1:m
        @constraint(model, sum(sum(u[k, j] * γ[i, k, j] for j in 1:n) for k in 1:K) <= μ)
    end

    # ||u_k||_1 ≤ 1, ∀k∈[K]
    for k in 1:K
        @constraint(model, sum(u[k, :]) <= 1)
    end

    # Objective: max sum_k <u_k, P_k g_k> - μ
    obj = sum(
        u[k, j] * Ξ[k][1][j] * Ξ[k][2][j]
        for j in 1:n
        for k in 1:K
    ) - μ
    @objective(model, Max, obj)

    optimize!(model)

    return value.(u), value(μ), model
end

"""
    validate_strong_duality(Ξ, γ; verbose=false)

Validate that primal and dual objectives match (strong duality).
"""
function validate_strong_duality(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    γ::Array{T,3};
    verbose=false) where T
    w, s, Q_primal, _ = solve_master_problem(Ξ, γ; verbose=verbose)
    u, μ, Q_dual, _ = solve_dual_problem(Ξ, γ; verbose=verbose)

    println("=== Strong Duality Validation ===")
    println("Primal objective (Q):     ", Q_primal)
    println("Dual objective (Q_*):     ", Q_dual)
    println("Gap:                      ", abs(Q_primal - Q_dual))
    println()
    println("Primal solution w:        ", w)
    println("Dual solution μ:          ", μ)

    return Q_primal, Q_dual, w, u, μ
end

# ------------------------------------------------
# Piecewise-linear pricing
# ------------------------------------------------

"""
    compute_demand_from_market(f1::FisherMarket, Ξ, alg)

Compute the demand matrix x[i,k,:] for a FisherMarket given revealed preferences Ξ.
For piecewise linear agents, uses greedy algorithm to compute demand.
For CES agents, can use analytic formula or compute via play!.

Returns x as a 3D array of size (m, K, n).
"""
function compute_demand_from_market(f1::FisherMarket, Ξ::Vector{Tuple{Vector{T},Vector{T}}}, alg) where T
    m, n = f1.m, f1.n
    K = length(Ξ)
    
    x = zeros(T, m, K, n)
    
    for i in 1:m
        for k in 1:K
            p_k, _ = Ξ[k]
            
            # Check if agent is general piecewise linear
            if f1.A_planes !== nothing && f1.b_planes !== nothing && size(f1.A_planes, 3) >= i
                # General piecewise linear: solve LP with agent's actual budget
                x[i, k, :] = compute_general_pwl_demand(p_k, f1.A_planes, f1.b_planes, f1.w[i], i)
            else
                # CES: use existing method or compute via play!
                alg.p .= p_k
                temp_market = FisherMarket(1, n; ρ=[f1.ρ[i]], c=f1.c[:, i:i], w=[1.0])
                play!(alg, temp_market)
                x[i, k, :] = temp_market.x[:, 1]
            end
        end
    end
    
    return x
end
"""
    solve_master_problem_piecewise(Ξ, x; verbose=false)

Solve the master problem using demand vectors directly (for piecewise linear):
    Q = min_{λ, s} sum_{k∈[K]} ||s_k||_∞
    s.t.  s_j^k + sum_{r∈[N]} λ_r x_{rj}(p^k) ≥ g_j^k, ∀j, k
          s_j^k - sum_{r∈[N]} λ_r x_{rj}(p^k) ≥ -g_j^k, ∀j, k
          sum_{r∈[N]} λ_r = 1, λ ≥ 0

Arguments:
- Ξ: Vector of (p_k, g_k) tuples, K observations
- x: Demand matrix of size (N, K, n) where N is number of agents

Returns:
- λ: Optimal weights
- s: Slack matrix of size (K, n)
- model: JuMP model
"""
function solve_master_problem_piecewise(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    x::Array{T,3};
    verbose=false) where T
    N, K, n = size(x)
    
    model = Model(() -> Gurobi.Optimizer())
    set_silent(model)
    if verbose
        unset_silent(model)
    end
    
    # Variables
    @variable(model, λ[1:N] >= 0)
    @variable(model, s[1:K, 1:n] >= 0)
    @variable(model, t[1:K] >= 0)  # t_k = ||s_k||_∞
    
    # Constraints: s_j^k + sum_r λ_r x_{rj}(p^k) ≥ g_j^k
    # and: s_j^k - sum_r λ_r x_{rj}(p^k) ≥ -g_j^k
    for k in 1:K
        _, g_k = Ξ[k]
        for j in 1:n
            @constraint(model, s[k, j] + sum(λ[r] * x[r, k, j] for r in 1:N) >= g_k[j])
            @constraint(model, s[k, j] - sum(λ[r] * x[r, k, j] for r in 1:N) >= -g_k[j])
        end
    end
    
    # t_k >= s_k (infinity norm)
    for k in 1:K
        for j in 1:n
            @constraint(model, t[k] >= s[k, j])
        end
    end
    
    # sum_r λ_r = 1
    @constraint(model, sum(λ) == 1)
    
    # Objective: min sum_k t_k
    @objective(model, Min, sum(t))
    
    optimize!(model)
    
    return value.(λ), value.(s), model
end

"""
    solve_dual_problem_piecewise(Ξ, x; verbose=false)

Solve the dual problem using demand vectors:
    Q_* = max_{u_k, μ} sum_{k∈[K]} <u_k, g_k> - μ
    s.t.  sum_{k∈[K]} <u_k, x_r(p^k)> ≤ μ, ∀r∈[N]
          ||u_k||_1 ≤ 1, u_k ∈ ℝ^n_+, ∀k∈[K]

Arguments:
- Ξ: Vector of (p_k, g_k) tuples
- x: Demand matrix of size (N, K, n)

Returns:
- u: Dual variables matrix of size (K, n)
- μ: Dual variable (scalar)
- model: JuMP model
"""
function solve_dual_problem_piecewise(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    x::Array{T,3};
    verbose=false) where T
    N, K, n = size(x)
    
    model = Model(() -> Gurobi.Optimizer())
    set_silent(model)
    if verbose
        unset_silent(model)
    end
    
    # Variables
    @variable(model, u_pos[1:K, 1:n] >= 0)
    @variable(model, u_neg[1:K, 1:n] >= 0)
    @variable(model, μ)

    # sum_k <u_k, x_r(p^k)> ≤ μ, with u = u_pos - u_neg
    for r in 1:N
        @constraint(model,
            sum(sum((u_pos[k,j] - u_neg[k,j]) * x[r,k,j] for j in 1:n) for k in 1:K) <= μ
        )
    end

    # ||u_k||_1 ≤ 1
    for k in 1:K
        @constraint(model, sum(u_pos[k, :] + u_neg[k, :]) <= 1)
    end

    obj = sum(
        (u_pos[k,j] - u_neg[k,j]) * Ξ[k][2][j]
        for k in 1:K, j in 1:n
    ) - μ
    @objective(model, Max, obj)

    optimize!(model)

    u = value.(u_pos) .- value.(u_neg)
    return u, value(μ), model
end
