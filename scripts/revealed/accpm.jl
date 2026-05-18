# Analytic Center Cutting-Plane Method (ACCPM) runner.
#
# Drop-in alternative to cpm.jl's `run_method_tracked`: same kwargs Dict
# and same return shape `(fa, γ_ref, history)`, but the separation oracle is
# fed the ANALYTIC CENTER of the current dual polytope instead of an
# LP-optimal dual vertex.
#
# Motivation:
#   LP-optimal duals jump between polytope vertices iter to iter, often
#   producing redundant or zig-zagging cuts. The analytic center sits in
#   the strict interior of the polytope (maximizer of Σ log-slack), so
#   the separation direction varies smoothly across iterations.
#   Goffin-Vial '93 / Atkinson-Vaidya '95 prove O(n²/ε²) iteration
#   complexity for convex feasibility; empirically ACCPM often needs
#   fewer cuts than vertex-based CPM at the cost of a heavier per-iter
#   subproblem.
#
# Trade-off:
#   Each iteration now solves a small nonlinear convex program
#   (analytic-center subproblem, exp-cone constraints) via Mosek, vs. an
#   LP via Gurobi in cpm.jl. The bet is that fewer outer iterations
#   amortize the per-iter cost. For tiny K · n this is the wrong bet;
#   for larger problems where the LP-CG cut count grows fast, it pays.
#
# Scope of this initial implementation:
#   - Single-cut separation only. No stage-2 multicut / stage-demotion logic
#     (the analytic-center direction replaces that heuristic).
#   - No sample-size subsampling on the dual (the analytic center wants
#     the full polytope to be well-defined).
#   - No persistent linear-MILP warm-start cache (could add later;
#     analytic-center separation directions change shape iter-to-iter, so
#     the warm-start benefit is smaller).
#   - test_err / excess / drop / banner: same as cpm.jl.
#
# Depends on:
#   - solve_wealth_redistribution_primal       (redistribute.jl; final + μ_ub refresh)
#   - drop_zero_columns!, add_to_gamma!,
#     add_column_to_market!, format_class,
#     solve_separation                 (separation.jl)
#   - CPM_TABLE / IterTable / print_banner / print_config / print_row /
#     print_continuation / BANNER_TITLE        (logging.jl)

using JuMP, MosekTools
using SparseArrays, LinearAlgebra
using ArgParse
import MathOptInterface as MOI

# ---- CLI surface --------------------------------------------------------
# `register_cli_accpm!(s)` adds the "Method: ACCPM" arg group to an
# ArgParseSettings; `apply_cli_accpm!(local_extra, cli)` forwards the
# parsed values into the kwargs Dict the runner consumes. Called from
# run_test.jl so the ACCPM-specific CLI lives next to the runner.
"""
    register_cli_accpm!(s::ArgParseSettings)

Add ACCPM's CLI flags (`--ac-solver`, `--interval-primal`, `--mu-ub-init`,
`--ac-verbose`, `--multicut-accpm`) to the given ArgParseSettings as a
"Method: ACCPM" arg group.
"""
function register_cli_accpm!(s::ArgParseSettings)
    add_arg_group!(s, "Method: ACCPM")
    @add_arg_table! s begin
        "--ac-solver"
        help = "Analytic-center subproblem solver: 'newton' (hand-rolled Newton on the log-barrier per choice-accpm.tex, with warm-start chaining; default) or 'mosek' (exp-cone JuMP reference, no warm-start)."
        arg_type = String
        default = "newton"
        range_tester = x -> x in ("newton", "mosek")
        "--interval-primal"
        help = "ACCPM only: solve the LP master every N iterations to refresh μ_ub and record primal_obj. 1 (default) = every iter; 0 disables (μ_ub stays at its initial loose problem-data bound until the final solve)."
        arg_type = Int
        default = 1
        "--mu-ub-init"
        help = "ACCPM only: initial upper bound on μ that keeps the dual polytope bounded. ≤ 0 (default) uses the loose problem-data bound Σ_k ‖P_k g_k‖_∞ · 1.1 + 1."
        arg_type = Float64
        default = -1.0
        "--ac-verbose"
        help = "ACCPM only: pass the inner AC solver log through to stdout (Mosek interior-point trace, or Newton iter-by-iter when wired). Default off."
        action = :store_true
        "--multicut-accpm"
        help = "ACCPM only: opt in to per-sample multicut (K cuts/iter) instead of the default single-cut. Empirically worse on dense AC duals (the K cuts are collectively redundant), but useful for small K or after warm-up."
        action = :store_true
    end
    return s
end

"""
    apply_cli_accpm!(local_extra::Dict, cli)

Read ACCPM-specific CLI values from the parsed `cli` and forward them to
the runner's `kwargs` Dict. No-op keys for non-ACCPM methods (the other
runners ignore unknown kwargs).
"""
function apply_cli_accpm!(local_extra::Dict, cli)
    local_extra[:ac_solver] = Symbol(cli["ac_solver"])
    if cli["interval_primal"] >= 0
        local_extra[:interval_primal] = cli["interval_primal"]
    end
    if cli["mu_ub_init"] > 0
        local_extra[:mu_ub_init] = cli["mu_ub_init"]
    end
    if cli["ac_verbose"]
        local_extra[:ac_verbose] = true
    end
    if cli["multicut_accpm"]
        local_extra[:multicut] = true
    end
    return local_extra
end

# ---- Hand-rolled Newton analytic-center solver --------------------------
# Follows the derivation in overleaf/read-econ/choice-accpm.tex.
#
# Variable layout  z = [vec(u); μ; vec(a)]  ∈ ℝ^{2nK + 1}, where
#   u ∈ ℝ^{K×n}   (dual matrix, column-stacked to length K·n)
#   μ ∈ ℝ
#   a ∈ ℝ^{K×n}   (lift auxiliaries for the ℓ₁ ball)
#
# Stacked constraint set  A z ≤ b  with slack s(z) = b − A z > 0, with
# four row groups in order:
#   cut       (m rows):    γ_t-weighted u − μ ≤ 0
#   μ-bound   (1 row):     μ ≤ μ_ub
#   simplex   (K rows):    Σ_j a_{kj} ≤ 1
#   lift+     (Kn rows):   u_{kj} − a_{kj} ≤ 0
#   lift−     (Kn rows):  −u_{kj} − a_{kj} ≤ 0
# Total M = m + 1 + K + 2Kn rows.

@inline _u_idx(k, j, K)       = k + K * (j - 1)
@inline _μ_idx(K, n)          = K * n + 1
@inline _a_idx(k, j, K, n)    = K * n + 1 + k + K * (j - 1)

# Build sparse A and vector b for the stacked polytope of accpm.tex.
function _ac_build_constraints(γ::Array{T,3}, μ_ub::Real) where T
    m, K, n = size(γ)
    d = 2 * K * n + 1
    M = m + 1 + K + 2 * K * n
    Is = Int[]; Js = Int[]; Vs = Float64[]
    b = zeros(Float64, M)
    row = 0
    # Cut block: A_i z = ⟨γ_i, u⟩ − μ ≤ 0  (b = 0)
    for i in 1:m
        row += 1
        for k in 1:K, j in 1:n
            γijk = Float64(γ[i, k, j])
            if γijk != 0.0
                push!(Is, row); push!(Js, _u_idx(k, j, K)); push!(Vs, γijk)
            end
        end
        push!(Is, row); push!(Js, _μ_idx(K, n)); push!(Vs, -1.0)
    end
    # μ-bound: μ ≤ μ_ub
    row += 1
    push!(Is, row); push!(Js, _μ_idx(K, n)); push!(Vs, 1.0)
    b[row] = Float64(μ_ub)
    # Simplex on a (per k): Σ_j a_{kj} ≤ 1
    for k in 1:K
        row += 1
        for j in 1:n
            push!(Is, row); push!(Js, _a_idx(k, j, K, n)); push!(Vs, 1.0)
        end
        b[row] = 1.0
    end
    # Lift+:  u_{kj} − a_{kj} ≤ 0
    for k in 1:K, j in 1:n
        row += 1
        push!(Is, row); push!(Js, _u_idx(k, j, K),  );    push!(Vs,  1.0)
        push!(Is, row); push!(Js, _a_idx(k, j, K, n));    push!(Vs, -1.0)
    end
    # Lift−: −u_{kj} − a_{kj} ≤ 0
    for k in 1:K, j in 1:n
        row += 1
        push!(Is, row); push!(Js, _u_idx(k, j, K),  );    push!(Vs, -1.0)
        push!(Is, row); push!(Js, _a_idx(k, j, K, n));    push!(Vs, -1.0)
    end
    A = sparse(Is, Js, Vs, M, d)
    return A, b
end

# Phase-I starting point per accpm.tex §sec.accpm.impl: u = 0,
# μ = μ_ub/2, a = 1/(2n) · 1. Strictly feasible whenever μ_ub > 0
# and the cut constraints aren't already binding at μ = μ_ub/2.
function _ac_phase1(K::Int, n::Int, μ_ub::Real)
    d = 2 * K * n + 1
    z = zeros(Float64, d)
    z[_μ_idx(K, n)] = μ_ub / 2
    for k in 1:K, j in 1:n
        z[_a_idx(k, j, K, n)] = 1.0 / (2 * n)
    end
    return z
end

# Damped Newton iteration on Φ(z) = −Σ log s_i(z).
# See accpm.tex eq.accpm.{grad,hessian,newton.system,damped.step,backtrack}.
function _ac_newton!(A::AbstractMatrix, b::AbstractVector, z::AbstractVector;
    ε_ac::Float64=1e-6, max_iters::Int=100, η::Float64=0.25, c1::Float64=1e-4,
    verbose::Bool=false)

    verbose && @printf("  AC Newton: %4s | %12s | %12s | %12s | %10s\n",
                       "it", "Φ(z)", "λ_N", "‖∇Φ‖∞", "α")
    for iter in 1:max_iters
        s = b .- A * z
        if any(s .<= 0)
            error("_ac_newton!: z is infeasible (min slack = $(minimum(s)))")
        end
        sinv = 1.0 ./ s
        grad = A' * sinv                           # eq.accpm.grad
        Dvec = sinv .^ 2
        H    = Symmetric(Matrix(A' * Diagonal(Dvec) * A))  # eq.accpm.hessian
        F    = cholesky(H; check=false)
        if !issuccess(F)
            # Add a tiny diagonal regularizer; under self-concordance H
            # is theoretically PD, but on the first few iters numerical
            # rounding can shave it. λ ~ 1e-10 scaled by trace keeps it
            # essentially the same problem.
            ε_reg = 1e-12 * tr(H) / size(H, 1)
            F = cholesky(Symmetric(Matrix(H) + ε_reg * I); check=false)
            issuccess(F) || error("_ac_newton!: Hessian not PD even after regularization")
        end
        Δz   = F \ (-grad)                         # eq.accpm.newton.system
        slope = dot(grad, Δz)                      # < 0 since Δz = -H⁻¹ grad
        λN_sq = -slope
        λN_sq < 0 && (λN_sq = 0.0)                 # rounding guard
        λN = sqrt(λN_sq)                           # eq.accpm.decrement
        Φ_z = -sum(log, s)
        if λN <= ε_ac
            verbose && @printf("            %4d | %12.5e | %12.3e | %12.3e | %10s  ✓ converged\n",
                               iter, Φ_z, λN, maximum(abs.(grad)), "-")
            return (z=z, niter=iter - 1, λN=λN)
        end
        # Self-concordant step size: damped if λN > η, full otherwise.
        α = λN > η ? 1.0 / (1.0 + λN) : 1.0
        # Backtrack for strict feasibility + Armijo (eq.accpm.backtrack).
        for _ in 1:60
            z_new = z .+ α .* Δz
            s_new = b .- A * z_new
            if all(s_new .> 0)
                Φ_new = -sum(log, s_new)
                if Φ_new <= Φ_z + c1 * α * slope
                    break
                end
            end
            α *= 0.5
            α < 1e-12 && error("_ac_newton!: backtracking shrunk α below 1e-12")
        end
        verbose && @printf("            %4d | %12.5e | %12.3e | %12.3e | %10.3e\n",
                           iter, Φ_z, λN, maximum(abs.(grad)), α)
        z .= z .+ α .* Δz
    end
    return (z=z, niter=max_iters, λN=NaN)
end

"""
    solve_analytic_center_newton(Ξ, γ, μ_ub; z_warm=nothing, kwargs...)

Hand-rolled Newton solver for the analytic-center subproblem described
in `overleaf/read-econ/choice-accpm.tex`. Drop-in alternative to
`solve_analytic_center` (the Mosek version) with the same `(u, μ, info)`
return shape, but no JuMP / Mosek call: builds the stacked sparse `A`
once, then runs damped Newton with self-concordant step size and
Armijo + strict-feasibility backtracking.

Keyword arguments:
- `z_warm::Union{AbstractVector,Nothing} = nothing` — warm start (the
  previous outer iteration's z is the natural choice; see warm-start
  paragraph in §sec.accpm.alg of the doc).
- `ε_ac::Float64 = 1e-6` — Newton-decrement stopping threshold.
- `max_iters::Int = 100` — outer Newton cap.
- `verbose::Bool = false` — ignored (no inner solver to log).

Returns `(u, μ, info)` where `info = (niter, λN, z)`; `info.z` can be
fed back as `z_warm` next outer iteration.
"""
function solve_analytic_center_newton(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    γ::Array{T,3}, μ_ub::Real;
    z_warm::Union{AbstractVector,Nothing}=nothing,
    ε_ac::Float64=1e-6,
    max_iters::Int=100,
    verbose::Bool=false) where T
    # `Ξ` is kept in the signature for drop-in compatibility with
    # solve_analytic_center (Mosek version); the polytope is determined
    # entirely by γ and μ_ub here.
    _ = Ξ
    _, K, n = size(γ)
    A, b = _ac_build_constraints(γ, μ_ub)
    z = isnothing(z_warm) ? _ac_phase1(K, n, μ_ub) : Vector{Float64}(z_warm)
    # If the warm start went out of the (possibly tightened μ_ub) box,
    # fall back to Phase-I rather than failing the strict-feasibility check.
    if any((b .- A * z) .<= 0)
        z = _ac_phase1(K, n, μ_ub)
    end
    out = _ac_newton!(A, b, z; ε_ac=ε_ac, max_iters=max_iters, verbose=verbose)
    u = reshape(out.z[1:K*n], K, n)
    μ = out.z[_μ_idx(K, n)]
    return u, μ, out
end

# Iteration-table layout for run_method_tracked_accpm. Same columns as
# CPM_TABLE so the two methods are visually comparable; "T(ac)" /
# "T(lp)" replaces the "class" cell with the analytic-center solver
# status when interesting.
const ACCPM_TABLE = IterTable(
    ["k", "primal", "test", "rc", "Δ(μ_ub)", "T", "class", "t(s)"],
    ["%5d", "%10.3e", "%10.3e", "%10.3e", "%10.3e", "%5d", "%14s", "%10.4f"],
    Any[1, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1, "ces×1", 1.234],
)

# Loose safety-net plateau detector. ACCPM is O(n²/ε²) by Goffin-Vial, so
# slow legitimate progress is normal; the detector exists only to escape
# a TRULY stuck loop when the user has disabled all tolerance-based stops.
# Window of 50 with relative tol 1e-12 means: the run must be flat to
# 12 sig figs for 50 consecutive iters before we declare stall, and we
# only check after MIN_ITERS_BEFORE_PLATEAU iters so early ramp-up isn't
# misclassified.
const ACCPM_PLATEAU_WINDOW            = 50
const ACCPM_PLATEAU_RTOL              = 1e-12
const ACCPM_MIN_ITERS_BEFORE_PLATEAU  = 100

# ---- Analytic-center subproblem -----------------------------------------
"""
    solve_analytic_center(Ξ, γ, μ_ub; verbose=false)

Find the analytic center of the wealth-redistribution dual polytope:

    max  Σ_i log(μ - ⟨u, γ_i⟩) + log(μ_ub - μ) + Σ_k log(1 - ‖u_k‖_1)
    s.t. ⟨u, γ_i⟩ < μ  ∀i,   μ < μ_ub,   ‖u_k‖_1 < 1  ∀k

over (u ∈ ℝ^{K×n}, μ ∈ ℝ) — matches `style=:inf` in
solve_wealth_redistribution_dual (signed u, L1 ball lifted via auxiliaries
a_{kj} ≥ ±u_{kj} with Σ_j a_{kj} ≤ 1).

Solved as a smooth convex program via Mosek using the exponential-cone
reformulation `t ≤ log(s)  ⇔  (t, 1, s) ∈ MOI.ExponentialCone`.

`μ_ub` is the loose upper bound on μ that keeps the polytope bounded.
Returns `(u, μ, model)`.
"""
function solve_analytic_center(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    γ::Array{T,3}, μ_ub::Real;
    verbose::Bool=false) where T

    m, K, n = size(γ)
    model = Model(Mosek.Optimizer)
    verbose || set_silent(model)

    @variable(model, u[1:K, 1:n])
    @variable(model, μ)

    # L1 ball lift: a_{kj} ≥ ±u_{kj},  Σ_j a_{kj} ≤ 1.
    @variable(model, a[1:K, 1:n] >= 0)
    @constraint(model, [k=1:K, j=1:n], a[k, j] >= u[k, j])
    @constraint(model, [k=1:K, j=1:n], a[k, j] >= -u[k, j])

    # Strict-interior slacks. Mosek handles `s ≥ 0` plus the exp-cone
    # constraint implicitly (the cone forces s > 0 when t is finite).
    @variable(model, s_cut[1:m] >= 0)
    @constraint(model, [i=1:m],
        s_cut[i] == μ - sum(γ[i, k, j] * u[k, j] for k in 1:K, j in 1:n))
    @variable(model, s_μ >= 0)
    @constraint(model, s_μ == μ_ub - μ)
    @variable(model, s_l1[1:K] >= 0)
    @constraint(model, [k=1:K], s_l1[k] == 1 - sum(a[k, j] for j in 1:n))

    # Log-barriers via exp-cone.
    # MOI.ExponentialCone is {(x1, x2, x3) : x2·exp(x1/x2) ≤ x3, x2 > 0}.
    # With x2 = 1:  exp(x1) ≤ x3  ⇔  x1 ≤ log(x3),  i.e. (t, 1, s) encodes t ≤ log(s).
    @variable(model, t_cut[1:m])
    @variable(model, t_μ)
    @variable(model, t_l1[1:K])
    @constraint(model, [i=1:m],
        [t_cut[i], 1.0, s_cut[i]] in MOI.ExponentialCone())
    @constraint(model, [t_μ, 1.0, s_μ] in MOI.ExponentialCone())
    @constraint(model, [k=1:K],
        [t_l1[k], 1.0, s_l1[k]] in MOI.ExponentialCone())

    @objective(model, Max, sum(t_cut) + t_μ + sum(t_l1))
    optimize!(model)

    return value.(u), value(μ), model
end

# Loose initial μ_ub from problem data. The dual objective at any
# feasible (u, μ) satisfies Σ ⟨u_k, P_k g_k⟩ ≤ Σ ‖u_k‖_∞ · ‖P_k g_k‖_1
# but with ‖u_k‖_1 ≤ 1 it's bounded by Σ ‖P_k g_k‖_∞, so μ at optimum is
# at most this. Adding a small safety factor.
function _initial_mu_ub(Ξ_train)
    s = 0.0
    @inbounds for k in eachindex(Ξ_train)
        p_k, g_k = Ξ_train[k]
        s += maximum(abs.(p_k .* g_k))
    end
    return s * 1.1 + 1.0
end

"""
    run_method_tracked_accpm(name, separation_kind, kwargs, Ξ_train, Ξ_test=nothing;
                              verbosity=1, sanity=false)

ACCPM iteration runner. Same return shape as `run_method_tracked` so it
plugs into the same downstream plotting / aggregation.

ACCPM-specific `kwargs` entries (everything else mirrors cpm.jl):
- `:interval_primal` (1)   solve the LP master every N iters to refresh
                            μ_ub and record `primal_obj`. 0 disables
                            (μ_ub stays at its initial value; primal_obj
                            never recorded until the final solve).
- `:mu_ub_init`  (auto)    initial μ upper bound; default is a loose
                            problem-data bound. Override if you need
                            tighter from the start.
- `:ac_verbose`  (false)   pass the Mosek log through.
"""
function run_method_tracked_accpm(name::Symbol, separation_kind::Symbol,
    kwargs::Dict, Ξ_train, Ξ_test=nothing;
    verbosity::Int=1, sanity=false)

    verbose = verbosity >= 1
    verbose_separation = verbosity >= 2

    max_iters         = get(kwargs, :max_iters, 50)
    tol_obj           = get(kwargs, :tol_obj, 5e-3)
    tol_rc            = get(kwargs, :tol_rc, 1e-3)
    tol_delta         = get(kwargs, :tol_delta, 1e-3)
    drop              = get(kwargs, :drop, true)
    use_indicators_kw = get(kwargs, :use_indicators, false)
    interval_dropping = get(kwargs, :interval_dropping, 1)
    classes           = get(kwargs, :classes, Symbol[:ces])
    timelimit         = get(kwargs, :timelimit, Inf)
    interval_eval_test = get(kwargs, :interval_eval_test, 1)
    f_real            = get(kwargs, :f_real, nothing)
    interval_eval_excess = get(kwargs, :interval_eval_excess, 0)
    # ACCPM-specific
    interval_primal   = get(kwargs, :interval_primal, 1)
    ac_verbose        = get(kwargs, :ac_verbose, false)
    # Multicut: at each iter call find_cuts_multi for K per-sample
    # inversion cuts instead of one find_cut_single column. Default OFF
    # based on empirical evidence: at the analytic-center u, the K per-
    # sample inversions tend to be collectively redundant (many describe
    # the same boundary direction), inflating the polytope's row count
    # and slowing each iter without improving the LP master primal.
    # Opt in with :multicut => true when the AC dual is well-spread
    # (e.g. small K or after some warm-up).
    multicut          = get(kwargs, :multicut, false)
    # AC solver: :newton (hand-rolled per choice-accpm.tex, default) or
    # :mosek (the exp-cone reference; uses a slightly different barrier
    # set — see test_accpm_newton.jl). Newton supports z_warm chaining
    # across outer iterations; Mosek does not.
    ac_solver         = get(kwargs, :ac_solver, :newton)
    @assert ac_solver in (:newton, :mosek) ":ac_solver must be :newton or :mosek"
    @assert !isempty(classes) ":classes must be non-empty"

    n = length(Ξ_train[1][1])
    K_train = length(Ξ_train)
    has_test = !isnothing(Ξ_test)

    # Initial surrogate market — same seeding as cpm.jl.
    ws = cpu_workspace(n)
    add_ces!(ws, 1; ρ=rand(1), scale=30.0, sparsity=0.99)
    ws.ces.w ./= sum(ws.ces.w)
    fa = FisherMarket(ws)
    γ_ref = Ref(compute_gamma_from_market(fa, Ξ_train))

    μ_ub = get(kwargs, :mu_ub_init, _initial_mu_ub(Ξ_train))
    master_cache = Ref{Any}(nothing)

    history = Dict(
        :primal_obj => Float64[],
        :test_err   => Float64[],
        :excess     => Float64[],
        :num_agents => Int[],
    )
    last_primal_obj = Ref(NaN)
    last_test_err   = Ref(NaN)
    last_excess     = Ref(NaN)
    last_μ_ub       = Ref(μ_ub)
    has_excess      = !isnothing(f_real) && interval_eval_excess > 0
    # Newton AC warm-start state. The previous iter's z is strictly
    # feasible for the new polytope (a cut only shrinks the set), so
    # we carry it forward; Phase-I init only fires the first iter.
    # `nothing` ⇒ Phase-I will be used on next call.
    z_warm_ac = Ref{Union{Nothing,Vector{Float64}}}(nothing)

    _t0 = time()
    if verbose
        print_banner(ACCPM_TABLE, BANNER_TITLE)
        print_config("method",            String(name))
        print_config("alias",             "ACCPM (analytic-center CG)")
        print_config("classes",           join(String.(classes), ", "))
        :ces      in classes && print_config("ces",      "free σ via LBFGS, warm-started by dual-LP"; indent=true)
        :linear   in classes && print_config("linear",
            "Gurobi MILP, " *
            (use_indicators_kw ? "indicator constraints" : "big-M (2·max|log p|)") *
            ", warm-start + model cache"; indent=true)
        :leontief in classes && print_config("leontief", "fixed σ = -1 via LBFGS"; indent=true)
        :ql       in classes && print_config("ql",       "piecewise-linear-concave QL (w = 1)"; indent=true)
        print_config("K (training samples)", K_train)
        print_config("n (goods)",            n)
        print_config("max_iters",            max_iters)
        print_config("timelimit (s)",        @sprintf("%g", Float64(timelimit)))
        print_config("tol_obj",              isnothing(tol_obj)   ? "off" : @sprintf("%g", tol_obj))
        print_config("tol_delta",            isnothing(tol_delta) ? "off" : @sprintf("%g", tol_delta))
        print_config("tol_rc",               isnothing(tol_rc)    ? "off" : @sprintf("%g", tol_rc))
        print_config("ac_solver",            String(ac_solver))
        print_config("multicut",             multicut)
        print_config("interval_primal",      interval_primal)
        print_config("interval_dropping",    interval_dropping)
        print_config("μ_ub (init)",          @sprintf("%g", μ_ub))
        println("-"^table_width(ACCPM_TABLE))
        print_header(ACCPM_TABLE)
    end

    rc_val      = NaN
    improvement = NaN

    for iter in 1:max_iters
        if time() - _t0 > timelimit
            verbose && print_continuation(ACCPM_TABLE,
                @sprintf("time limit reached (%.1fs > %.1fs)", time() - _t0, timelimit))
            break
        end

        # 1. Analytic-center solve of current dual polytope.
        # Force-enable the inner AC log on iter 1 so the user sees the
        # Newton trace for the first centering (Phase-I → AC), regardless
        # of --ac-verbose. Subsequent iters use the user's `ac_verbose`.
        ac_verbose_iter = (iter == 1) || ac_verbose
        if iter == 1 && verbose
            print_continuation(ACCPM_TABLE, "iter 1 AC Newton trace:")
        end
        u, μ, ac_info = if ac_solver === :newton
            solve_analytic_center_newton(Ξ_train, γ_ref[], μ_ub;
                z_warm=z_warm_ac[], verbose=ac_verbose_iter)
        else
            solve_analytic_center(Ξ_train, γ_ref[], μ_ub; verbose=ac_verbose_iter)
        end
        # Carry the AC point forward for the next outer iteration (Newton only).
        if ac_solver === :newton && ac_info isa NamedTuple && haskey(ac_info, :z)
            z_warm_ac[] = copy(ac_info.z)
        end

        # 2. Optional LP-master solve to refresh μ_ub and record primal_obj.
        if interval_primal > 0 && (iter % interval_primal == 0)
            _remaining = isfinite(timelimit) ? max(1.0, timelimit - (time() - _t0)) : nothing
            w_lp, _, model_p, _, _ = solve_wealth_redistribution_primal(Ξ_train, γ_ref[];
                verbose=false, timelimit=_remaining, cache=master_cache)
            last_primal_obj[] = objective_value(model_p) / K_train
            # Tighten μ_ub from incumbent (LP primal == LP dual at optimum,
            # so primal_obj * K_train is an upper bound on μ at the dual
            # optimum and hence on μ at every feasible dual point worth
            # centering around).
            μ_new_ub = last_primal_obj[] * K_train + 1e-6
            if μ_new_ub < μ_ub
                last_μ_ub[] = μ_ub
                μ_ub = μ_new_ub
            end
            if drop && interval_dropping > 0 && (iter % interval_dropping == 0)
                drop_zero_columns!(fa, γ_ref, w_lp)
                master_cache[] = nothing
            else
                fa.w .= w_lp
            end
        end

        # 3. Record history slot.
        if has_test && interval_eval_test > 0 && (iter % interval_eval_test == 0)
            last_test_err[] = evaluate_test_error(fa, Ξ_test)
        end
        te = last_test_err[]
        if has_excess && fa.m > 0 && (iter % interval_eval_excess == 0)
            try
                v = validate_surrogate(fa, f_real; verbose=false)
                last_excess[] = v.excess_surrogate_linf
            catch err
                @warn "[$name iter $iter] validate_surrogate failed" err
            end
        end
        push!(history[:primal_obj], last_primal_obj[])
        push!(history[:test_err],   te)
        push!(history[:excess],     last_excess[])
        push!(history[:num_agents], fa.m)

        # Improvement on primal_obj (NaN-safe).
        improvement = length(history[:primal_obj]) >= 2 ?
            (history[:primal_obj][end-1] - last_primal_obj[]) : NaN

        _log_row(; rc=NaN, class_str="-") = (verbose && print_row(ACCPM_TABLE,
            Any[iter, last_primal_obj[], te, rc,
                isfinite(last_μ_ub[]) ? (last_μ_ub[] - μ_ub) : 0.0,
                fa.m, class_str, time() - _t0]))

        # Convergence: primal_obj < tol_obj.
        if !isnothing(tol_obj) && isfinite(last_primal_obj[]) && last_primal_obj[] < tol_obj
            _log_row()
            verbose && print_continuation(ACCPM_TABLE,
                @sprintf("converged (obj/K = %.2e < tol_obj=%g)", last_primal_obj[], tol_obj))
            break
        end
        # Convergence: stalled primal_obj.
        if length(history[:primal_obj]) >= 3 && !isnothing(tol_delta)
            recent = history[:primal_obj][end-2:end]
            if all(isfinite, recent)
                imp2 = max(recent[1] - recent[2], recent[2] - recent[3])
                if imp2 < tol_delta
                    _log_row()
                    verbose && print_continuation(ACCPM_TABLE,
                        @sprintf("converged (Δ = %.2e < tol_delta=%g)", imp2, tol_delta))
                    break
                end
            end
        end

        # 4. Separation at the analytic-center (u, μ). sample_size=0 because
        # the AC subproblem needs the full Ξ. Multicut (default) calls
        # find_cuts_multi for K per-sample inversions per iter; single
        # falls back to one find_cut_single column.
        _separation_remaining = isfinite(timelimit) ?
            max(1.0, timelimit - (time() - _t0)) : nothing
        cands_multi = NamedTuple[]   # populated only in multicut path
        if multicut
            cands_multi = find_cuts_multi(Ξ_train, u, classes; sample_size=0)
            # Best rc across the batch — used both for the log "rc" column
            # and for the tol_rc convergence check below.
            rc_val = isempty(cands_multi) ? NaN :
                maximum(reduced_cost(c.γ_new, u, μ) for c in cands_multi)
            class_str = format_cuts_tag(cands_multi)
        else
            cand = find_cut_single(Ξ_train, u, μ, classes;
                sample_size=0, verbose=verbose_separation,
                timelimit=_separation_remaining, kwargs...)
            rc_val    = cand.rc
            class_str = format_class(cand.class, cand.params)
        end
        # Convergence: reduced cost ≤ tol_rc with rc > 0 (LP-CG semantics).
        # IMPORTANT: rc against the analytic-center (u_AC, μ_AC) is NOT a
        # certificate of LP optimality. The AC's μ sits in the interior of
        # the polytope, generically ABOVE the LP-optimal μ; a negative rc
        # only means the pricer can't beat the (loose) interior μ, but many
        # cuts that would help the LP master may still exist. So we keep
        # the cpm.jl-style `rc > 0` guard here and rely on tol_obj /
        # tol_delta for outer termination.
        if !isnothing(tol_rc) && (rc_val > 0.0) && (rc_val <= tol_rc)
            _log_row(rc=rc_val, class_str=class_str)
            verbose && print_continuation(ACCPM_TABLE,
                @sprintf("converged (rc = %.2e ≤ tol_rc=%g, %s)",
                         rc_val, tol_rc, class_str))
            break
        end
        # Stall safety net (loose): if the user disables tol_rc AND
        # tol_delta AND `primal_obj` has been completely flat for the last
        # ACCPM_PLATEAU_WINDOW iters AND we've taken at least
        # ACCPM_MIN_ITERS_BEFORE_PLATEAU, terminate to avoid an
        # uninterrupted million-iter loop. The window is long enough that
        # slow legitimate progress isn't caught: ACCPM has O(n²/ε²)
        # iteration complexity (Goffin-Vial), so on hard instances it can
        # take hundreds of iters to refine.
        if length(history[:primal_obj]) >= ACCPM_MIN_ITERS_BEFORE_PLATEAU
            window = history[:primal_obj][end-ACCPM_PLATEAU_WINDOW+1:end]
            if all(isfinite, window)
                spread = maximum(window) - minimum(window)
                scale  = max(1.0, abs(window[end]))
                if spread < ACCPM_PLATEAU_RTOL * scale
                    _log_row(rc=rc_val, class_str=class_str)
                    verbose && print_continuation(ACCPM_TABLE,
                        @sprintf("plateau-stop (obj flat for %d iters, no progress; AC dual may be undercutting useful directions)",
                                 ACCPM_PLATEAU_WINDOW))
                    break
                end
            end
        end
        # Append the cut(s) to γ and to the surrogate market.
        if multicut
            for c in cands_multi
                add_to_gamma!(γ_ref, c.γ_new)
                add_column_to_market!(fa, (y=c.y, σ=c.σ), c.class, 0.0)
            end
        else
            add_to_gamma!(γ_ref, cand.γ_new)
            add_column_to_market!(fa, cand.params, cand.class, 0.0)
        end

        _log_row(rc=rc_val, class_str=class_str)
    end

    # Final LP solve — recover w for the surrogate market and the closing
    # primal_obj value (analytic center alone doesn't give a primal).
    w_final, _, model_p_final, _, _ = solve_wealth_redistribution_primal(Ξ_train, γ_ref[];
        verbose=false, cache=master_cache)
    if drop
        drop_zero_columns!(fa, γ_ref, w_final)
        master_cache[] = nothing
    else
        fa.w .= w_final
    end
    last_primal_obj[] = objective_value(model_p_final) / K_train

    _elapsed = time() - _t0
    if has_test && !isempty(history[:test_err])
        history[:test_err][end] = evaluate_test_error(fa, Ξ_test)
    end
    if has_excess && fa.m > 0 && !isempty(history[:excess])
        try
            v = validate_surrogate(fa, f_real; verbose=false)
            history[:excess][end] = v.excess_surrogate_linf
        catch err
            @warn "[$name final] validate_surrogate failed" err
        end
    end
    if !isempty(history[:primal_obj])
        history[:primal_obj][end] = last_primal_obj[]
    end
    if verbose
        @printf("--- done: %d agents, obj/K=%.3e, t=%.4fs ---\n",
                fa.m, last_primal_obj[], _elapsed)
    end

    return fa, γ_ref, history
end
