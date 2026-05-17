# Wealth redistribution problem — master (primal) and dual LP solvers.
# Mirrors equations (eq.cg.master) / (eq.cg.dual) in the paper. Data-prep
# helpers (produce_revealed_preferences, compute_gamma, …) live in setup.jl;
# the CG iteration runner (run_method_tracked) lives in cpm.jl.

using JuMP
using LinearAlgebra
using ExchangeMarket
using Gurobi
# The Gurobi env is owned by `linear.jl::_gurobi_env`; we reuse it here so
# the master LP doesn't reprint the license banner on every call.
# `MosekTools` is no longer needed for the master / dual — both run on the
# shared Gurobi env. Other scripts in the repo still load Mosek directly.

"""
    solve_wealth_redistribution_primal(Ξ, γ; verbose=false, style=:inf)

Solve the wealth-redistribution primal (eq.cg.master). With the default
`style=:inf`, balance is an equality and `s_k` is free (signed deficit):

    Q = min_{w, s_k} sum_{k∈[K]} ‖s_k‖_∞
    s.t.  s_k + sum_{i∈[m]} γ_{ik} w_i = P_k g_k,  ∀k∈[K]
          sum_{i∈[m]} w_i = 1,  w ∈ ℝ^m_+,  s_k ∈ ℝ^n

(Implementation lifts ‖s_k‖_∞ to auxiliary `t_k ≥ ±s_{k,j}`, so the LP
minimizes `sum_k t_k`.)

With `style=:nonneg`, the balance becomes inequality and `s_k ≥ 0`:

    s_k + sum_{i∈[m]} γ_{ik} w_i ≥ P_k g_k,  s_k ∈ ℝ^n_+

which penalizes only under-supply (Pg − γw)_+, not over-supply.

Arguments:
- Ξ: Vector of (p_k, g_k) tuples, K observations
- γ: Bidding matrix of size (m, K, n)

Returns:
- w: Optimal weights
- s: Slack matrix of size (K, n)
- model, balance, budget: JuMP handles for downstream dual extraction
"""
function solve_wealth_redistribution_primal(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    γ::Array{T,3};
    verbose=false,
    style=:inf,
    timelimit::Union{Real,Nothing}=nothing,
    # When non-`nothing`, the previous model (with its `s`, `t`, `balance`,
    # `budget`, and the w-vars for atoms 1..last_m) is reused; only the new
    # atoms `last_m+1..m` are added via `set_normalized_coefficient` on the
    # existing constraints. Caller wipes the Ref to force a rebuild (e.g.,
    # after `drop_zero_columns!` reshuffles the γ rows).
    cache::Union{Ref,Nothing}=nothing,
) where T
    m, K, n = size(γ)

    # ----------------------------------------------------------------
    # Cache hit: same (K, n, style), and γ grew by appending. Add the new
    # w variables, extend balance / budget coefficients, leave everything
    # else (constraints, s, t, objective) untouched.
    # ----------------------------------------------------------------
    cache_hit = false
    model = nothing
    w_vars = nothing
    s_var = nothing
    balance = nothing
    budget = nothing
    if !isnothing(cache) && !isnothing(cache[])
        c = cache[]
        if c.K == K && c.n == n && c.style == style && c.last_m <= m
            model   = c.model
            w_vars  = c.w_vars
            s_var   = c.s
            balance = c.balance
            budget  = c.budget
            # Append variables for new atoms.
            # NOTE: the `lower_bound = 0.0` kwarg on an anonymous `@variable`
            # call is silently dropped by some JuMP versions (the bug that
            # caused master objective to go negative on iter 2). Apply the
            # bound explicitly with `set_lower_bound` to be safe.
            for i in (c.last_m + 1):m
                w_new = @variable(model, base_name = "w_$(i)")
                set_lower_bound(w_new, 0.0)
                push!(w_vars, w_new)
                for k in 1:K, j in 1:n
                    set_normalized_coefficient(balance[k, j], w_new, γ[i, k, j])
                end
                set_normalized_coefficient(budget, w_new, 1.0)
            end
            cache_hit = true
            if !isnothing(timelimit) && timelimit > 0
                set_time_limit_sec(model, Float64(timelimit))
            end
        end
    end

    if !cache_hit
        model = Model(() -> Gurobi.Optimizer(_gurobi_env()))
        if !verbose
            set_attribute(model, "LogToConsole", 0)
        end
        # Pure LP — let Gurobi pick the algorithm; barrier is usually fastest
        # for the dense balance system, simplex for warm-starts after small
        # incremental changes (Gurobi handles the switch internally).
        if !isnothing(timelimit) && timelimit > 0
            set_time_limit_sec(model, Float64(timelimit))
        end

        # Variables — use the named `>= 0` form because the anonymous
        # `@variable(model, [1:K], lower_bound = 0.0)` syntax has silently
        # dropped the lower bound in some JuMP versions, which let the LP
        # report negative objective values (the original bug that this
        # rewrite hit). Named form is the conservative choice.
        @variable(model, w[1:m] >= 0)
        w_vars = collect(w)
        if style == :inf
            @variable(model, s[1:K, 1:n])
        else
            @variable(model, s[1:K, 1:n] >= 0)
        end
        s_var = s
        @variable(model, t[1:K] >= 0)
        t_var = t

        # Per-sample balance s_k + Σ_i γ_{ik} w_i  (=|≥)  P_k g_k
        balance = Matrix{ConstraintRef}(undef, K, n)
        for k in 1:K
            p_k, g_k = Ξ[k]
            P_k_g_k = p_k .* g_k
            for j in 1:n
                lhs = s_var[k, j] + sum(γ[i, k, j] * w_vars[i] for i in 1:m)
                if style == :inf
                    balance[k, j] = @constraint(model, lhs == P_k_g_k[j])
                else
                    balance[k, j] = @constraint(model, lhs >= P_k_g_k[j])
                end
            end
        end

        # t_k ≥ |s_{k,j}| (infinity norm).
        for k in 1:K, j in 1:n
            @constraint(model, t_var[k] >= s_var[k, j])
            if style == :inf
                @constraint(model, t_var[k] >= -s_var[k, j])
            end
        end

        # sum_i w_i = 1
        budget = @constraint(model, sum(w_vars) == 1)

        @objective(model, Min, sum(t_var))
    end

    optimize!(model)

    if !isnothing(cache)
        cache[] = (model=model, w_vars=w_vars, s=s_var, balance=balance,
                   budget=budget, last_m=m, K=K, n=n, style=style)
    end

    return value.(w_vars), value.(s_var), model, balance, budget
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
    solve_wealth_redistribution_dual(Ξ, γ; verbose=false, style=:inf)

Solve the wealth-redistribution dual (eq.cg.dual). With the default
`style=:inf`, `u_k` is signed and the L1 ball is enforced via auxiliary
nonneg `a_{k,j} ≥ ±u_{k,j}` with `sum_j a_{k,j} ≤ 1`:

    Q_* = max_{u_k, μ} sum_{k∈[K]} ⟨u_k, P_k g_k⟩ − μ
    s.t.  sum_{k∈[K]} ⟨u_k, γ_{ik}⟩ ≤ μ,  ∀i∈[m]
          ‖u_k‖_1 ≤ 1,  u_k ∈ ℝ^n,  ∀k∈[K]

With `style=:nonneg`, `u_k ∈ ℝ^n_+` and the L1 constraint collapses to
`sum_j u_{k,j} ≤ 1`. Should match the primal's `:nonneg` form by LP duality.

Arguments:
- Ξ: Vector of (p_k, g_k) tuples, K observations
- γ: Bidding matrix of size (m, K, n)

Returns:
- u: Dual variables matrix of size (K, n)
- μ: Dual variable (scalar)
- model: JuMP handle (so callers can read `objective_value`)
"""
function solve_wealth_redistribution_dual(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    γ::Array{T,3};
    verbose=false,
    style=:inf
) where T
    m, K, n = size(γ)

    # Create model — same Gurobi env as the primal master.
    model = Model(() -> Gurobi.Optimizer(_gurobi_env()))
    if !verbose
        set_attribute(model, "LogToConsole", 0)
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
    w, s, model_p, _, _ = solve_wealth_redistribution_primal(Ξ, γ; verbose=verbose)
    Q_primal = objective_value(model_p)
    u, μ, model_d = solve_wealth_redistribution_dual(Ξ, γ; verbose=verbose)
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
