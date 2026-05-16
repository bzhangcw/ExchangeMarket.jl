# Wealth redistribution problem — master (primal) and dual LP solvers.
# Mirrors equations (eq.cg.master) / (eq.cg.dual) in the paper. Data-prep
# helpers (produce_revealed_preferences, compute_gamma, …) live in setup.jl;
# the CG iteration runner (run_method_tracked) lives in cpm.jl.

using JuMP
using LinearAlgebra
using ExchangeMarket
using MosekTools

"""
    solve_master_problem(Ξ, γ; verbose=false, style=:inf)

Solve the master problem (primal, eq.cg.master):
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
    verbose=false,
    style=:inf
) where T
    m, K, n = size(γ)

    # Create model
    model = Model(Mosek.Optimizer)
    set_silent(model)
    if verbose
        unset_silent(model)
    end

    # Variables
    @variable(model, w[1:m] >= 0)
    if style == :inf
        @variable(model, s[1:K, 1:n])
    else
        @variable(model, s[1:K, 1:n] >= 0)
    end
    @variable(model, t[1:K] >= 0)  # t_k = ||s_k||_∞

    # Constraints
    # s_k + sum_i γ_{ik} w_i ≥ P_k g_k
    balance = Matrix{ConstraintRef}(undef, K, n)
    for k in 1:K
        p_k, g_k = Ξ[k]
        P_k_g_k = p_k .* g_k
        for j in 1:n
            if style == :inf
                balance[k, j] = @constraint(model, s[k, j] + sum(γ[i, k, j] * w[i] for i in 1:m) == P_k_g_k[j])
            else
                balance[k, j] = @constraint(model, s[k, j] + sum(γ[i, k, j] * w[i] for i in 1:m) >= P_k_g_k[j])
            end
        end
    end

    # t_k >= |s_{k,j}| (infinity norm)
    for k in 1:K
        for j in 1:n
            @constraint(model, t[k] >= s[k, j])
            if style == :inf
                @constraint(model, t[k] >= -s[k, j])
            end
        end
    end

    # sum_i w_i = 1
    @constraint(model, budget, sum(w) == 1)

    # Objective: min sum_k t_k
    @objective(model, Min, sum(t))

    optimize!(model)

    return value.(w), value.(s), model, balance, budget
end

"""
    extract_duals(model, balance, budget, K, n)

Extract dual variables (u, μ) from a solved primal master model.
- u[k,j] = dual of balance constraint (k,j)
- μ     = dual of budget constraint (sum w = 1)
"""
function extract_duals(model, balance::Matrix{ConstraintRef}, budget::ConstraintRef, K::Int, n::Int)
    u = [dual(balance[k, j]) for k in 1:K, j in 1:n]
    # MOI convention: dual of (Σw == 1) in a Min problem enters as μ_MOI*(1 - Σw),
    # so μ_MOI = -μ_explicit where the explicit dual uses max ... - μ, Σ<u,γ> ≤ μ.
    μ = -dual(budget)
    return u, μ
end

"""
    solve_dual_problem(Ξ, γ; verbose=false, style=:inf)

Solve the dual problem (eq.cg.dual):
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
    verbose=false,
    style=:inf
) where T
    m, K, n = size(γ)

    # Create model
    model = Model(Mosek.Optimizer)
    set_silent(model)
    if verbose
        unset_silent(model)
    end

    # Variables
    if style == :inf
        @variable(model, u[1:K, 1:n])
    else
        @variable(model, u[1:K, 1:n] >= 0)
    end
    @variable(model, μ)

    # Constraints
    # sum_k <u_k, γ_{ik}> ≤ μ, ∀i∈[m]
    for i in 1:m
        @constraint(model, sum(sum(u[k, j] * γ[i, k, j] for j in 1:n) for k in 1:K) <= μ)
    end

    # ||u_k||_1 ≤ 1, ∀k∈[K]
    if style == :inf
        # |u_{kj}| ≤ a_{kj}, Σ_j a_{kj} ≤ 1
        @variable(model, a[1:K, 1:n] >= 0)
        for k in 1:K
            for j in 1:n
                @constraint(model, a[k, j] >= u[k, j])
                @constraint(model, a[k, j] >= -u[k, j])
            end
            @constraint(model, sum(a[k, :]) <= 1)
        end
    else
        for k in 1:K
            @constraint(model, sum(u[k, :]) <= 1)
        end
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
    w, s, model_p, _, _ = solve_master_problem(Ξ, γ; verbose=verbose)
    Q_primal = objective_value(model_p)
    u, μ, model_d = solve_dual_problem(Ξ, γ; verbose=verbose)
    Q_dual = objective_value(model_d)

    println("=== Strong Duality Validation ===")
    println("Primal objective (Q):     ", Q_primal)
    println("Dual objective (Q_*):     ", Q_dual)
    println("Gap:                      ", abs(Q_primal - Q_dual))
    println()
    println("Primal solution w:        ", w)
    println("Dual solution μ:          ", μ)

    return Q_primal, Q_dual, w, u, μ
end
