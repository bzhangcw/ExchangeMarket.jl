# Wealth redistribution problem — master (primal) and dual LP solvers.
# Mirrors the hybrid homothetic / non-homothetic LP pair
# `eq.wealth.hybrid.lp` in overleaf/read-econ/wealth-dist.tex. The
# non-homothetic block enters with each weight pinned at a fixed wealth
# `w_t = pinned_w` (the column-generation driver supplies the pinning
# indices and shared `w_0`). Data-prep helpers (produce_revealed_preferences,
# compute_gamma, …) live in setup.jl; the CG iteration runner
# (run_method_tracked) lives in cpm.jl.

using JuMP
using LinearAlgebra
using ArgParse
using ExchangeMarket
# Master LP models are built via `new_model()` from engine.jl, which picks
# Gurobi (when available) or Mosek; `set_lp_barrier!` applies the Gurobi-only
# barrier attribute and no-ops on Mosek.

"""
    solve_wealth_redist_primal(Ξ, γ; pinned_idx=Int[], pinned_w=0.0, verbose=false)

Solve the wealth-redistribution primal (`eq.wealth.hybrid.lp`). All
constraints are equalities and `s_k` is the signed deficit:

    Q = min_{w, s_k}  sum_{k∈[K]} ‖s_k‖_∞
    s.t.  s_k + sum_{i∈[m]} γ_{ik} w_i = P_k g_k,  ∀k∈[K]
          sum_{i∈[m]} w_i = 1,
          w_i = pinned_w,  ∀i ∈ pinned_idx     (non-homothetic atoms)
          w ∈ ℝ^m_+,  s_k ∈ ℝ^n

The ‖·‖_∞ in the objective is lifted to auxiliary `t_k ≥ ±s_{k,j}` so the
program is a Gurobi LP. Non-homothetic atoms whose γ row was evaluated
at the same `w_0 = pinned_w` are listed in `pinned_idx`; the dual ν_t of
each pinning constraint can be read post-solve via `dual(pinned[i])`
and matches the multiplier in `eq.wealth.hybrid.lp` up to MOI's sign
convention (see `extract_duals` for the simplex μ).

Arguments:
- Ξ: Vector of (p_k, g_k) tuples, K observations
- γ: Bidding matrix of size (m, K, n)

Keyword arguments:
- `pinned_idx::Vector{Int}` — indices i ∈ 1:m of non-homothetic atoms;
  empty (default) ⇒ pure homothetic master, identical to the old behavior.
- `pinned_w::T` — common pinning wealth w_0 for every i ∈ pinned_idx.
- `cache::Union{Ref,Nothing}` — persistent JuMP model. On hit, only the
  new w variables (and their pinning constraints, if applicable) are
  appended; everything else is reused.

Returns:
- w: Optimal weights
- s: Slack matrix of size (K, n)
- model, balance, budget: JuMP handles for downstream dual extraction
"""
function solve_wealth_redist_primal(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    γ::Array{T,3};
    verbose=false,
    timelimit::Union{Real,Nothing}=nothing,
    # Indices i ∈ 1:m of non-homothetic atoms whose weight is pinned at
    # w_t = pinned_w. Empty vector (default) ⇒ pure homothetic master.
    pinned_idx::Vector{Int}=Int[],
    pinned_w::T=zero(T),
    # When non-`nothing`, the previous model (with its `s`, `t`, `balance`,
    # `budget`, `pinned`, and the w-vars for androids 1..last_m) is reused;
    # only the new androids `last_m+1..m` are added via
    # `set_normalized_coefficient` on the existing constraints, and any
    # new pinning constraints are appended. Caller wipes the Ref to force a
    # rebuild (e.g., after `drop_zero_columns!` reshuffles the γ rows).
    cache::Union{Ref,Nothing}=nothing,
) where T
    m, K, n = size(γ)

    # ----------------------------------------------------------------
    # Cache hit: same (K, n, pinned_w), and γ grew by appending. Add the
    # new w variables, extend balance / budget coefficients, append any
    # new pinning constraints, leave everything else (constraints, s, t,
    # objective) untouched.
    # ----------------------------------------------------------------
    cache_hit = false
    model = nothing
    w_vars = nothing
    s_var = nothing
    balance = nothing
    budget = nothing
    pinned = nothing
    if !isnothing(cache) && !isnothing(cache[])
        c = cache[]
        if c.K == K && c.n == n && c.pinned_w == pinned_w && c.last_m <= m
            model = c.model
            w_vars = c.w_vars
            s_var = c.s
            balance = c.balance
            budget = c.budget
            pinned = c.pinned
            # Append variables for new androids.
            # NOTE: the `lower_bound = 0.0` kwarg on an anonymous `@variable`
            # call is silently dropped by some JuMP versions (the bug that
            # caused master objective to go negative on iter 2). Apply the
            # bound explicitly with `set_lower_bound` to be safe.
            for i in (c.last_m+1):m
                w_new = @variable(model, base_name = "w_$(i)")
                set_lower_bound(w_new, 0.0)
                push!(w_vars, w_new)
                for k in 1:K, j in 1:n
                    set_normalized_coefficient(balance[k, j], w_new, γ[i, k, j])
                end
                set_normalized_coefficient(budget, w_new, 1.0)
            end
            # Pin any newly-added non-homothetic atoms.
            for i in pinned_idx
                if !haskey(pinned, i)
                    pinned[i] = @constraint(model, w_vars[i] == pinned_w)
                end
            end
            cache_hit = true
            if !isnothing(timelimit) && timelimit > 0
                set_time_limit_sec(model, Float64(timelimit))
            end
        end
    end

    if !cache_hit
        model = new_model()
        # Interior-point (barrier) without simplex crossover. Method=2
        # forces the barrier solver (Gurobi's default would auto-pick
        # between primal simplex, dual simplex, and barrier); Crossover=0
        # skips the simplex-vertex cleanup that normally follows barrier.
        # Two reasons this is the right choice for CG:
        #   (i)  A near-optimal interior (w, u, μ) is enough for the dual
        #        extraction the separation oracles consume — and CG
        #        re-solves the master every iteration anyway, so any
        #        slight suboptimality in w is corrected on the next round.
        #   (ii) IPM yields a SMOOTHER (u, μ) trajectory across CG iters
        #        than simplex's vertex-to-vertex jumps; smoother duals
        #        produce more diverse separation columns (less risk of
        #        the cycling-on-duplicate-atoms degeneracy we hit with
        #        the Mosek-AC + simplex path).
        # Crossover is also the dominant cost on warm-started LPs at
        # scale, so skipping it shortens each master solve.
        set_lp_barrier!(model)
        # set_attribute(model, "Crossover", 0)
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
        @variable(model, s[1:K, 1:n])
        s_var = s
        # Per-(sample, good) absolute-value lift so the objective is the
        # per-sample ℓ_1 residual summed across samples
        # (Σ_{k,j} |s_{k,j}|). Dual of an ℓ_1 norm is ℓ_∞, so the recovered
        # u_{k,j} live in [-1, 1] under this dualization.
        @variable(model, t[1:K, 1:n] >= 0)
        t_var = t

        # Per-sample balance s_k + Σ_i γ_{ik} w_i = P_k g_k
        balance = Matrix{ConstraintRef}(undef, K, n)
        for k in 1:K
            p_k, g_k = Ξ[k]
            P_k_g_k = p_k .* g_k
            for j in 1:n
                lhs = s_var[k, j] + sum(γ[i, k, j] * w_vars[i] for i in 1:m)
                balance[k, j] = @constraint(model, lhs == P_k_g_k[j])
            end
        end

        # t_{k,j} ≥ |s_{k,j}| (per-(sample, good) absolute value; ℓ_1).
        for k in 1:K, j in 1:n
            @constraint(model, t_var[k, j] >= s_var[k, j])
            @constraint(model, t_var[k, j] >= -s_var[k, j])
        end

        # sum_i w_i = 1
        budget = @constraint(model, sum(w_vars) == 1)

        # Pin non-homothetic atoms at w_t = pinned_w. The dual ν_t is
        # recoverable via `dual(pinned[i])` post-solve; with the simplex μ
        # and the per-sample u, that completes the hybrid-LP dual triple
        # (u, μ, ν) of eq.wealth.hybrid.lp.
        pinned = Dict{Int,ConstraintRef}()
        for i in pinned_idx
            pinned[i] = @constraint(model, w_vars[i] == pinned_w)
        end

        @objective(model, Min, sum(t_var))
    end

    # Apply the per-call verbose toggle on every solve (rebuild OR cache hit),
    # via the solver-portable `set_silent` / `unset_silent` (Gurobi-specific
    # "OutputFlag"/"LogToConsole" would not exist on Mosek). Without this, a
    # cached model retains the last call's flags, so iter-1's `verbose=true`
    # would leak into later iterations under the cache.
    verbose ? unset_silent(model) : set_silent(model)
    optimize!(model)

    if !isnothing(cache)
        cache[] = (model=model, w_vars=w_vars, s=s_var, balance=balance,
            budget=budget, pinned=pinned, last_m=m, K=K, n=n,
            pinned_w=pinned_w)
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
    solve_wealth_redist_dual(Ξ, γ; verbose=false)

Solve the wealth-redistribution dual (eq.wealth.hybrid.lp, homothetic
block). `u_k` is signed and the L1 ball is enforced via auxiliary
nonneg `a_{k,j} ≥ ±u_{k,j}` with `sum_j a_{k,j} ≤ 1`:

    Q_* = max_{u_k, μ} sum_{k∈[K]} ⟨u_k, P_k g_k⟩ − μ
    s.t.  sum_{k∈[K]} ⟨u_k, γ_{ik}⟩ ≤ μ,  ∀i∈[m]
          ‖u_k‖_1 ≤ 1,  u_k ∈ ℝ^n,  ∀k∈[K]

Should match `solve_wealth_redist_primal` by LP duality on a purely
homothetic instance. The non-homothetic pinning multipliers ν_t are
not modeled here; recover them from the primal model's `pinned`
constraint duals if needed.

Arguments:
- Ξ: Vector of (p_k, g_k) tuples, K observations
- γ: Bidding matrix of size (m, K, n)

Returns:
- u: Dual variables matrix of size (K, n)
- μ: Dual variable (scalar)
- model: JuMP handle (so callers can read `objective_value`)
"""
function solve_wealth_redist_dual(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    γ::Array{T,3};
    verbose=false,
) where T
    m, K, n = size(γ)

    # Create model on the selected engine (Gurobi/Mosek). `set_silent` is the
    # solver-portable silencer (Gurobi's "LogToConsole" would not exist on Mosek).
    model = new_model()
    verbose || set_silent(model)

    # Variables
    @variable(model, u[1:K, 1:n])
    @variable(model, μ)

    # sum_k <u_k, γ_{ik}> ≤ μ, ∀i∈[m]
    for i in 1:m
        @constraint(model, sum(sum(u[k, j] * γ[i, k, j] for j in 1:n) for k in 1:K) <= μ)
    end

    # ||u_k||_1 ≤ 1 via |u_{kj}| ≤ a_{kj}, Σ_j a_{kj} ≤ 1.
    @variable(model, a[1:K, 1:n] >= 0)
    for k in 1:K
        for j in 1:n
            @constraint(model, a[k, j] >= u[k, j])
            @constraint(model, a[k, j] >= -u[k, j])
        end
        @constraint(model, sum(a[k, :]) <= 1)
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

Validate that primal and dual objectives match (strong duality) on a
purely homothetic instance.
"""
function validate_strong_duality(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    γ::Array{T,3};
    verbose=false) where T
    w, s, model_p, _, _ = solve_wealth_redist_primal(Ξ, γ; verbose=verbose)
    Q_primal = objective_value(model_p)
    u, μ, model_d = solve_wealth_redist_dual(Ξ, γ; verbose=verbose)
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

# ---- CLI surface --------------------------------------------------------
"""
    register_cli_redist!(s::ArgParseSettings)

Add the "Master: redistribute" arg group: `--redist-use-nlp`,
`--redist-nonh-w`. Both flow through `local_extra` into the
runner kwargs; `cpm.jl` resolves the default for `--redist-nonh-w`
(sentinel `-1`) to `1/max_iters` per method.
"""
function register_cli_redist!(s::ArgParseSettings)
    add_arg_group!(s, "Master: redistribute")
    @add_arg_table! s begin
        "--redist-use-nlp"
        help = "Use the NLP master (currently NOT IMPLEMENTED; reserved for the upcoming MadNLP integration). When false (default), the LP master with pinned non-homothetic weights is used."
        action = :store_true
        "--redist-nonh-w"
        help = "Pinned wealth w_0 for non-homothetic atoms in the LP master. -1 (default) resolves to 1/max_iters per method, capping the number of non-homothetic columns at max_iters (simplex feasibility: |N| ≤ ⌊1/w_0⌋)."
        arg_type = Float64
        default = -1.0
    end
    return s
end

"""
    apply_cli_redist!(local_extra::Dict, cli)

Forward `--redist-use-nlp` / `--redist-nonh-w` into the runner kwargs.
A negative `--redist-nonh-w` is left unset so `cpm.jl` can resolve it
from `max_iters`.
"""
function apply_cli_redist!(local_extra::Dict, cli)
    if cli["redist_use_nlp"]
        local_extra[:redist_use_nlp] = true
    end
    if cli["redist_nonh_w"] > 0
        local_extra[:redist_nonh_w] = cli["redist_nonh_w"]
    end
    return local_extra
end
