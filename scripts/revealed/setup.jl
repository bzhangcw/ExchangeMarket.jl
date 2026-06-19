# Methods and tracking utilities for revealed-preference CES surrogate fitting.
# File map (in include order):
#   - engine.jl                    : solver-engine wrapper (new_model, set_engine!,
#                                    gurobi_available, _gurobi_env); loaded first so every
#                                    later solver site can pick Gurobi/Mosek/MadNLP
#   - redistribute.jl              : wealth-redistribution primal / dual LPs
#                                    (solve_wealth_redist_{primal,dual},
#                                     eq.cg.master / eq.cg.dual)
#   - separation.jl                : per-class separation oracle, drop_zero_columns!,
#                                    add_column_to_market!, multicut merge, and
#                                    the runner-facing find_cut_single / find_cuts_multi
#                                    wrappers shared between cpm.jl and accpm.jl
#                                    (includes the per-android files under androids/)
#   - androids/{ces,linear,leontief,ql,plc}.jl
#                                  : per-android-class generators, demand routines,
#                                    and class-specific separation solvers
#                                    (solve_separation_ces, _linear, _leontief, _ql).
#                                    plc.jl is the ground-truth PLC market generator
#                                    used by run_test.jl --market-type plc (no
#                                    separation solver; PLC isn't a surrogate class).
#   - logging.jl                   : IterTable + banner/config helpers (both runners)
#   - frankwolfe/frankwolfe.jl     : manual Frank-Wolfe runner (run_method_tracked_fw)
#   - cpm.jl                       : column-generation runner (run_method_tracked)
#   - accpm.jl                     : analytic-center CG variant (run_method_tracked_accpm)
#   - validate.jl                  : surrogate / real-market validation
#   - frankwolfe/wrapper_frankwolfe.jl
#       : FrankWolfe.jl-package wrapper (run_method_tracked_fwjl); loaded
#         last because it depends on validate_surrogate.
#
# Helpers used by the FW/CG runners (produce_revealed_preferences,
# compute_gamma_from_market, compute_gamma, compute_gamma_matrix,
# evaluate_test_error) live in this file so they're available to all
# downstream includes.

using Printf
using Random
using LinearAlgebra
using Serialization
using YAML
using ExchangeMarket

# -----------------------------------------------------------------------
# Shared Gurobi env (used by master LP and the linear MILP pricer).
# Loaded first so the license banner fires once at script load.
# -----------------------------------------------------------------------
include("./engine.jl")

# -----------------------------------------------------------------------
# SPLC ground-truth market (separable PLC; greedy closed-form demand).
# Included here (not in the drivers) because build_rep_data below needs
# random_splc_agent / produce_revealed_preferences_splc. The general-PLC
# generator (androids/plc.jl) stays driver-included since it needs Mosek.
# -----------------------------------------------------------------------
include("./androids/splc.jl")

# -----------------------------------------------------------------------
# NGES ground-truth market (non-additive GES; real-market only, no surrogate
# / separation). Included here because build_rep_data needs random_nges_agent
# / produce_revealed_preferences_nges. See eq.gnae.utility.
# -----------------------------------------------------------------------
include("./androids/nges.jl")

# -----------------------------------------------------------------------
# Master / dual LP solvers (define before runners include them).
# -----------------------------------------------------------------------
include("./redistribute.jl")
# NLP master (JuMP + MadNLP), used by cpm.jl whenever :ql is in classes
# (the QL contribution w_i · γ(p, w_i) is piecewise-linear-concave in w_i
# and not LP-expressible). See redistribute_nlp.jl for the column protocol.
include("./redistribute_nlp.jl")

# -----------------------------------------------------------------------
# Revealed-preference data preparation
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Wealth functions  w(p) → ℝ^m : the per-agent budgets at price p. Setting
# `f1.w .= wealth_fn(p)` before `play!` lets a single FisherMarket producer
# reproduce both fixed-budget (Fisher) and endowment (Arrow–Debreu) demand,
# since the CES best-response is x_i = w_i γ_i(p)/p. Used uniformly by
# `produce_revealed_preferences` and selectable from the CLI via
# `--wealth-function` (0 = constant, 1 = first-order).
# -----------------------------------------------------------------------
wealth_constant(w0::AbstractVector) = (p -> w0)              # 0: degree-0, w_i = w0_i
wealth_firstorder(B::AbstractMatrix) = (p -> B' * p)         # 1: degree-1, w_i = ⟨p, b_i⟩
# 2: degree-2 quadratic shares. `Q` is an n×n×m tensor (Q[:,:,i] = agent i's
# PSD form). The raw quadratic v_i(p) = pᵀQ[:,:,i]p is normalized into budget
# shares w_i(p) = v_i / Σ_j v_j, so total spending Σ_i w_i(p) = 1 for every p
# (matching the constant/first-order convention) while the per-agent split still
# varies quadratically (income effects). The contraction is one matvec:
# pᵀQ[:,:,i]p = ⟨vec(Q[:,:,i]), vec(ppᵀ)⟩.
function wealth_quadratic(Q::AbstractArray{<:Real,3})
    n, _, m = size(Q)
    Qm = reshape(Q, n * n, m)                 # column i = vec(Q[:,:,i])
    return p -> (v = Qm' * vec(p * p'); v ./ sum(v))
end

"""
    random_wealth_quad_tensor(n, m, sparsity) -> Q  (n×n×m)

Random PSD forms `Q[:,:,i] = AᵀA`, with `A` an n×n Gaussian masked to density
`sparsity` (controls how concentrated the resulting budget shares are). Each
agent is guaranteed at least one nonzero entry, so `pᵀQ_ip = ‖Ap‖² > 0` at any
strictly positive price and the normalized share `w_i(p)` is never zero (the
`< 0.01` sparsity regime would otherwise leave most `A_i` all-zero, giving a
zero-wealth agent that breaks the per-agent UMP solvers).
"""
function random_wealth_quad_tensor(n::Int, m::Int, sparsity::Real)
    Q = Array{Float64,3}(undef, n, n, m)
    for i in 1:m
        mask = rand(n, n) .< sparsity
        any(mask) || (mask[rand(1:n), rand(1:n)] = true)   # ≥1 nonzero ⇒ w_i(p)>0
        A = randn(n, n) .* mask
        Q[:, :, i] = A' * A
    end
    return Q
end

"""
    make_wealth_function(code::Int; w0=nothing, B=nothing, Q=nothing)

Build the wealth function selected by `--wealth-function`: `0` → constant
`wealth_constant(w0)`, `1` → first-order `wealth_firstorder(B)`, `2` →
second-order `wealth_quadratic(Q)` (normalized quadratic shares; `Q` is an
n×n×m tensor).
"""
function make_wealth_function(code::Int; w0=nothing, B=nothing, Q=nothing)
    if code == 0
        @assert w0 !== nothing "wealth-function 0 (constant) needs w0"
        return wealth_constant(w0)
    elseif code == 1
        @assert B !== nothing "wealth-function 1 (first-order) needs endowment matrix B"
        return wealth_firstorder(B)
    elseif code == 2
        @assert Q !== nothing "wealth-function 2 (second-order) needs quadratic forms Q"
        return wealth_quadratic(Q)
    else
        error("unknown wealth-function code $code (expected 0, 1, or 2)")
    end
end

"""
    produce_revealed_preferences(alg, f1::FisherMarket, K, wealth_fn; seed=nothing, lift=false, M=0.0)

Generate K revealed-preference samples `(p, g)` from a FisherMarket. For each
sampled price the agents' wealth is reset via `wealth_fn(p)` and the aggregate
demand `g = Σ_i x_i` is computed by `play!` (CES best-response `x_i = w_i γ_i(p)/p`).
This single producer covers both constant (Fisher) and first-order (Arrow–Debreu)
wealth — pick the family by passing `wealth_constant`/`wealth_firstorder`.

With `lift=true`, prices are sampled on the lifted simplex Δ̄_{n+1}; each sample is
`(p̄, d̄)` with `d̄ = (d(q), M + ⟨q,1⟩ − W(q))`, `q = p/π`, `W(q)=⟨q,d(q)⟩` (the
money-lift; `lem.lift`). `M` is the money supply.
"""
function produce_revealed_preferences(alg, f1::FisherMarket, K::Int, wealth_fn;
    seed=nothing, lift=false, M=0.0)
    !isnothing(seed) && Random.seed!(seed)
    n = f1.n
    # CES demand oracle: reset the agents' wealth via `wealth_fn(q)` at the
    # (possibly unnormalized) price q, best-respond, and aggregate. γ is
    # degree-0, so an unnormalized q is fine. This is the only CES-specific
    # piece; the lift itself is the shared `produce_revealed_preferences_lifted`.
    d_oracle = q -> begin
        f1.w .= wealth_fn(q)
        alg.p .= q
        play!(alg, f1)
        sum(f1.x, dims=2)[:]
    end
    lift && return produce_revealed_preferences_lifted(d_oracle, K, n; M=M)
    # Unlifted: sample uniformly on Δ_n (Dirichlet(1,…,1)).
    Ξ = Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, K)
    for k in 1:K
        ē = -log.(rand(n))
        p_k = ē ./ sum(ē)
        Ξ[k] = (copy(p_k), d_oracle(p_k))
    end
    return Ξ
end

# -----------------------------------------------------------------------
# Generic real-market demand (any non-CES ground-truth family).
#
# `agent_demand(agent, p, w)` returns the demand bundle x_i(p, w_i) for one
# agent at price p and budget w. Per-class: GES/NGES via the spending share
# (x = w γ / p), SPLC via the greedy `solve_splc_demand`. This lets the lift
# and the validation evaluate the aggregate demand of ANY such market at an
# arbitrary (possibly unnormalized) price, without a market-specific path.
# -----------------------------------------------------------------------
agent_demand(a::GESAgent, p::AbstractVector, w::Real) = w .* share(a, p, w) ./ p
agent_demand(a::NGESAgent, p::AbstractVector, w::Real) = w .* share(a, p, w) ./ p
agent_demand(a::SPLCAgent, p::AbstractVector, w::Real) = first(solve_splc_demand(a, p, w))

"""
    wealth_at(budgets, p) -> w

Resolve a ground-truth budget object into the per-agent wealth vector at price
`p`. Three representations, one per `--wealth-function` code:

* `Vector`   → fixed Fisher budgets `w_i` (code 0).
* `Matrix`   → n×m endowments, `w_i(p) = ⟨p, b_i⟩` (code 1, Arrow–Debreu).
* callable   → price-dependent wealth function `p ↦ w` (code 2, e.g. the
  normalized quadratic shares `w_i(p) = pᵀQ_ip / Σ_j pᵀQ_jp`).
"""
wealth_at(budgets::AbstractVector, p::AbstractVector) = budgets
wealth_at(budgets::AbstractMatrix, p::AbstractVector) = budgets' * p
wealth_at(budgets, p::AbstractVector) = budgets(p)

"""
    aggregate_real_demand(agents, budgets, p) -> d(p)

Aggregate demand `d(p) = Σ_i x_i(p, w_i)` of a real market given as `agents`
plus `budgets` (resolved per agent by `wealth_at`: a Vector of fixed Fisher
budgets, an n×m endowment Matrix, or a price-dependent wealth function), via
`agent_demand`. Valid at any positive price.
"""
function aggregate_real_demand(agents, budgets, p::AbstractVector)
    w = wealth_at(budgets, p)
    g = zeros(length(p))
    for (i, ag) in enumerate(agents)
        g .+= agent_demand(ag, p, w[i])
    end
    return g
end

"""
    produce_revealed_preferences_lifted(d_oracle, K, n; M, seed=nothing)

Money-lift (lem.lift) as a single, market-agnostic transformation on the
aggregate demand. Given ANY unlifted demand oracle `d_oracle: q ↦ ℝⁿ`, sample
`(p,π)` on the lifted simplex Δ̄_{n+1}, project `q = p/π`, and emit `(p̄, d̄)`
with `d̄ = (d(q), M + ⟨q,1⟩ − W(q))`, `W(q) = ⟨q, d(q)⟩`. The only per-family
thing is the oracle (CES: `play!`; GES/SPLC/NGES: `aggregate_real_demand`);
PLC has no single-valued oracle and is unsupported. `M` is the money supply.
"""
function produce_revealed_preferences_lifted(d_oracle, K::Int, n::Int;
    M::Real, seed=nothing, q_lo::Real=1e-4, q_hi::Real=1e4, max_tries::Int=1000)
    !isnothing(seed) && Random.seed!(seed)
    Ξ = Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, K)
    for k in 1:K
        tries = 0
        while true
            tries += 1
            ē = -log.(rand(n + 1))
            p̄ = ē ./ sum(ē)
            q = p̄[1:n] ./ p̄[n+1]
            # Reject pathological price ratios. A tiny π = p̄[n+1] sends q → ∞;
            # the GES/SPLC demand solvers (pⱼ^{-σⱼ} etc.) then overflow to Inf
            # and the money demand M+⟨q,1⟩−⟨q,d⟩ collapses to Inf−Inf = NaN, which
            # poisons the downstream fitting LP. Keep q within [q_lo, q_hi] and
            # require the oracle to return finite values (CES is log-space safe;
            # this guard matters for the share/demand families).
            if all(q_lo .≤ q .≤ q_hi)
                d = d_oracle(q)
                money = M + sum(q) - dot(q, d)
                if all(isfinite, d) && isfinite(money)
                    Ξ[k] = (copy(p̄), vcat(d, money))
                    break
                end
            end
            tries ≥ max_tries && error(
                "produce_revealed_preferences_lifted: no finite in-range sample " *
                "in $max_tries tries (q∈[$q_lo, $q_hi]); the market/wealth model " *
                "may be degenerate at extreme prices.")
        end
    end
    return Ξ
end

# Shared one-line announcement of the money-lift parameters (all families).
_announce_lift(n::Int, M::Real) = @printf(
    "[lift] money-lifted: %d goods → %d (good %d = money), money supply M = %.4g; each sample lives on the (n+1)-simplex, money demand = M + ⟨q,1⟩ − W(q) with W(q) = ⟨q,d(q)⟩.\n",
    n, n + 1, n + 1, M)

"""
    compute_gamma(p, c, σ)

Compute the CES bidding vector γ for given price p, coefficients c, and elasticity parameter σ.
    γ_j = (c_j^{1+σ} * p_j^{-σ}) / sum_ℓ(c_ℓ^{1+σ} * p_ℓ^{-σ})

Uses log-space computation (softmax) to avoid overflow for large |σ|.

Special case: when `σ` is `+Inf` (the linear regime, ρ = 1), γ is the
bang-per-buck vertex indicator `e_{argmax_j c_j / p_j}` as in
`fact.demand.linear`. This matches the storage convention used by
`add_column_to_market!` for the `:linear` class.
"""
function compute_gamma(p::AbstractVector, c::AbstractVector, σ::Real)
    # Linear regime: ρ = 1, σ = +∞; γ is the bang-per-buck vertex.
    if isinf(σ) && σ > 0
        γ = zeros(eltype(p), length(c))
        j_star = argmax(c ./ p)
        γ[j_star] = one(eltype(p))
        return γ
    end
    # log(numerator_j) = (1+σ) log(c_j) - σ log(p_j)
    z = (1 + σ) .* log.(c) .- σ .* log.(p)
    z_max = maximum(z)
    ez = exp.(z .- z_max)
    γ = ez ./ sum(ez)
    return γ
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


# -----------------------------------------------------------------------
# evaluation on test set: mean ℓ_1 error of Σ w_i γ_i(p) vs target P g
# (matches the master LP's per-sample ℓ_1 residual; see redistribute.jl).
# -----------------------------------------------------------------------
function evaluate_test_error(fa, Ξ_test)
    ws = fa.storage
    (ws.ces.m + ws.gen.m == 0) && return NaN
    K = length(Ξ_test)
    n = length(Ξ_test[1][1])
    errs = Float64[]
    for (p, g) in Ξ_test
        target = p .* g
        fitted = zeros(n)
        # Walk routing so :ces rows dispatch via compute_gamma (CES share)
        # and :gen rows dispatch via `share(agent, p, w)` per agent class
        # (e.g. share(::QuasiLinearLogAgent, p, w) -> ql_share).
        for (sub, j) in ws.routing
            if sub === :ces
                c_j = Vector(ws.ces.c[:, j])
                fitted .+= ws.ces.w[j] .* compute_gamma(p, c_j, ws.ces.σ[j])
            else
                agent = ws.gen.agents[j]
                w_j = ws.gen.w[j]
                fitted .+= w_j .* share(agent, p, w_j)
            end
        end
        push!(errs, norm(fitted .- target, 1))
    end
    return sum(errs) / K
end

# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
function parse_args_for_test_real(argv=ARGS)
    s = ArgParseSettings(
        prog="run_test.jl",
        description="Benchmark CG / Multicut / FW (Fisher & Arrow–Debreu) on a CES or PLC market.",
        autofix_names=true,
    )

    # ===================================================================
    # `--help` group order is workflow-natural:
    #   (1) Problem  →  what you're solving (size, sparsity, seed)
    #   (2) Market   →  ground-truth families: CES (always), PLC (opt-in)
    #   (3) Method   →  which algorithms + their per-method knobs
    #   (4) Separation → shared + per-android-class oracle knobs
    #   (5) Stopping →  when each method halts
    #   (6) Evaluation → how we measure
    #   (7) IO       →  where output goes
    # ===================================================================

    # ---- (1) Problem instance ----------------------------------------
    add_arg_group!(s, "Problem instance")
    @add_arg_table! s begin
        "--n", "-n"
        help = "Number of goods"
        arg_type = Int
        default = 5
        "--m", "-m"
        help = "Number of agents in the real market"
        arg_type = Int
        default = 50
        "--k", "-k"
        help = "Number of TRAINING observations (was both train and test until this split; test size now via --k-test)."
        arg_type = Int
        default = 100
        "--k-test"
        help = "Number of TEST observations. Independent of --k."
        arg_type = Int
        default = 20
        "--seed", "-s"
        help = "Master random seed"
        arg_type = Int
        default = 42
        "--rep"
        help = "Number of repetitions (different seeds). When > 1, the plot shows mean ± 1σ ribbon across reps."
        arg_type = Int
        default = 1
        "--sparsity"
        help = "Per-agent coefficient density (CES `c` / PLC `A`): fraction of NONZERO entries when generating the ground-truth market (passed straight to `sprand(n, m, p)`; high = dense, low = sparse). 0.99 (default) matches the pre-CLI hardcoded value; 0.0 is all-zero (degenerate) and 1.0 is fully dense (also rejected by ArgParse below)."
        arg_type = Float64
        default = 0.99
        range_tester = x -> 0.0 <= x < 1.0
    end

    # ---- (2) Market: ground-truth family + wealth model + per-family knobs
    add_arg_group!(s, "Market")
    @add_arg_table! s begin
        "--market-type", "-t"
        help = "Ground-truth market family: ces, plc, ges, splc, nges. (nges = non-additive GES u(x)=Σ_j c_j (A x)_j^{r_j}, strictly positive A; real-market only, no surrogate.)"
        arg_type = String
        default = "ces"
        range_tester = x -> x in ("ces", "plc", "ges", "splc", "nges")
        "--wealth-function"
        help = "Per-agent wealth model w(p): 0 = constant (Fisher, fixed budgets w_i, default); 1 = first-order (Arrow–Debreu, endowments b_i with w_i(p)=⟨p,b_i⟩); 2 = second-order (normalized quadratic shares w_i(p)=pᵀQ_ip / Σ_j pᵀQ_jp). Replaces the old --budget-type. All three codes work for every market family (ces, plc, ges, splc, nges)."
        arg_type = Int
        default = 0
        range_tester = x -> x in (0, 1, 2)
        "--wealth-quad-sparsity"
        help = "Density of the second-order quadratic forms Q[:,:,i] = AᵀA (--wealth-function 2): fraction of NONZERO entries in A. The forms are normalized into budget shares (Σ_i w_i(p)=1), so this controls how concentrated/heterogeneous the per-agent shares are (low = sparser A), not the magnitude. Default 0.3."
        arg_type = Float64
        default = 0.3
        range_tester = x -> 0.0 < x <= 1.0
        "--lift"
        help = "Money-lift the revealed-preference data: adjoin money as good n+1 and emit lifted (p̄,d̄) pairs on the (n+1)-simplex (CES only; ignored for other families)."
        action = :store_true
    end
    # One register_cli_*_market! call per market family. These only
    # bite when `--market-type` selects the matching family.
    register_cli_ces_market!(s)   # --ces-rho-low, --ces-rho-high
    register_cli_plc!(s)          # --plc-L, --plc-no-intercept
    register_cli_splc!(s)         # --splc-L

    # ---- (3) Method selection + per-method knobs ---------------------
    add_arg_group!(s, "Method selection")
    @add_arg_table! s begin
        "--preset"
        help = "Path to a YAML method-catalog preset (default: revealed/presets.yaml). Defines each method's separation_kind + base kwargs + plot style and the ablation variant groups; CLI flags below still override individual kwargs."
        arg_type = String
        default = ""
        "--methods"
        help = """Comma-separated method names to run (any of cg,cgma,fw,sfw,fwjl), fw and sfw are implemented by myself. By default we don't include them yet, instead we compare to FrankWolfe.jl"""
        arg_type = String
        default = "cg,cgma,fwjl"
        "--method"
        help = "Comma-separated preset VARIANT syms for run_test.jl to run (the presets.yaml `variant:` names, e.g. cg,adfw,adcg_hard_vardelta). Empty (default) runs every variant. Distinct from --methods, which picks base method names."
        arg_type = String
        default = ""
        "--classes"
        help = "Comma-separated function classes for android classes in separation (any of ces,linear,leontief,ql,nn)"
        arg_type = String
        default = "ces,linear,leontief"
        "--engine"
        help = "LP/MIP master backend: gurobi (default when available), mosek, or auto. NLP separations always use MadNLP. If Gurobi is unavailable the backend falls back to Mosek and the linear android is disabled."
        arg_type = String
        default = "auto"
    end
    # Each method owns its CLI surface in its own runner file
    # (cpm.jl::register_cli_cpm!, accpm.jl::register_cli_accpm!).
    register_cli_cpm!(s)
    register_cli_accpm!(s)
    # Master LP (redistribute.jl) owns --redist-use-nlp and --redist-nonh-w.
    register_cli_redist!(s)
    # AD master (redistribute_ad.jl) owns --ad-endow-mode and --ad-mask-size.
    register_cli_ad!(s)
    # Frank–Wolfe (frankwolfe.jl) owns --step-rule, --no-away-steps,
    # --interval-logging (shared by fw and adfw).
    register_cli_fw!(s)

    # ---- (4) Separation oracle knobs ---------------------------------
    # Separation CLI surface lives in the per-android files so each
    # class owns its own knobs (mirrors register_cli_cpm! / _accpm!).
    # Shared knobs first, then per-class.
    register_cli_separation!(s)   # --sample-size
    register_cli_ces!(s)          # --sep-ces-sigma-lower, --sep-ces-sigma-upper
    register_cli_ges!(s)          # --sep-ges-sigma-{lower,upper}, --sep-ges-y-{lower,upper}
    register_cli_linear!(s)       # --sep-linear-separation-indicator
    register_cli_nn!(s)           # --sep-nn-hidden, --sep-nn-iters

    # ---- (5) Stopping ------------------------------------------------
    add_arg_group!(s, "Stopping")
    @add_arg_table! s begin
        "--timelimit", "-T"
        help = "Wall-clock cap per method, in seconds"
        arg_type = Float64
        default = 60.0
        "--iterlimit", "-I"
        help = "Iteration cap per method; if > 0, overrides each method's :max_iters (default: use setup.jl value)"
        arg_type = Int
        default = -1
        "--tol-obj"
        help = "Objective tolerance; if > 0, overrides each method's :tol_obj; 0 disables the check; < 0 uses setup.jl default"
        arg_type = Float64
        default = -1.0
        "--tol-delta"
        help = "Fixed-point/improvement tolerance; if > 0, overrides each method's :tol_delta; 0 disables the stall stop (non-stop); < 0 uses setup.jl default"
        arg_type = Float64
        default = -1.0
        "--tol-rc"
        help = "Threshold on the per-iteration |∇| (reduced cost of the winning candidate column) for the stage-1 convergence check. The flag stays `--tol-rc` for backward compatibility, but the iteration table and log messages refer to this quantity as `|∇|`. > 0 overrides each method's :tol_rc; 0 disables the |∇|-based stop entirely; < 0 uses setup.jl default."
        arg_type = Float64
        default = 1e-3
    end

    # ---- (6) Evaluation ----------------------------------------------
    add_arg_group!(s, "Evaluation")
    @add_arg_table! s begin
        "--interval-eval-test"
        help = "Evaluate test error every N iterations (per method); intervening iters carry forward the last value. Default 1 (every iter); -1 evaluates only once at the end."
        arg_type = Int
        default = 1
        "--interval-eval-excess"
        help = "Evaluate per-iteration market-excess ‖p(q-g)‖∞ every N iterations (CES, constant wealth only). Default 0 = OFF — per-iter tracking is opt-in even when --validate is on (it only controls the post-run table). Set N>0 to enable, or -1 to inherit --interval-eval-test."
        arg_type = Int
        default = 0
        "--interval-logging"
        help = "Print the per-iteration table row every N iterations. Shared across ALL methods (cg/cgma/adcg and fw/adfw). Default -1 keeps each method's default (every iter, or the preset's `log_interval` for fw/adfw)."
        arg_type = Int
        default = -1
        "--no-validate"
        help = "Skip the CES surrogate equilibrium validation (default ON for --market-type ces; PLC is default OFF)."
        action = :store_true
    end

    # ---- (7) IO -------------------------------------------------------
    add_arg_group!(s, "IO")
    @add_arg_table! s begin
        "--out-dir"
        help = "Output directory for the data file (and the .csv if --csv is set without an absolute path). Defaults to the package timestamped results dir, re-evaluated each script invocation via `ExchangeMarket.current_results_dir()` so the hourly bucket reflects run time, not Julia start time."
        arg_type = String
        default = ExchangeMarket.current_results_dir()
        "--data-file", "-d"
        help = "Serialize the per-method aggregation context to this path (consumable by run_plot.jl -f). Empty default ⇒ <out-dir>/real_<market>.jls."
        arg_type = String
        default = ""
        "--no-data-file"
        help = "Suppress the per-run context dump (overrides --data-file)."
        action = :store_true
        "--csv"
        help = "If non-empty, append per-method-per-rep rows to this CSV (path; relative to revealed/)"
        arg_type = String
        default = ""
        "--verbosity", "-v"
        help = "0 = silent; 1 = per-iteration table; 2 = + per-separation detail"
        arg_type = Int
        default = 0
        range_tester = x -> x in (0, 1, 2)
    end

    return parse_args(argv, s)
end

# -----------------------------------------------------------------------
# Separation oracle + FW / CG runners.
# Order matters: separation.jl defines `drop_zero_columns!`,
# `add_column_to_market!`, etc., which both runners use. logging.jl
# defines the shared IterTable / print_banner used by both runners.
# -----------------------------------------------------------------------
include("./timing.jl")        # @time_sep / @time_redist — must precede separation.jl
include("./separation.jl")
include("./logging.jl")
include("./cpm.jl")
include("./accpm.jl")
# Arrow–Debreu stack: redistribute_ad.jl holds the master/dual/pricing
# primitives + ad_market_from_atoms; cpm_ad.jl holds the CG loop run_ad_tracked
# (sibling of redistribute.jl / cpm.jl). Loaded after separation.jl
# (find_cut_single, _gamma_over_full_from_cand) and cpm.jl/logging.jl
# (CPM_TABLE, print_*) since they reuse all of them.
include("./redistribute_ad.jl")
include("./cpm_ad.jl")
include("./validate.jl")
include("./validate_ad.jl")     # AD surrogate validation (pulls scripts/arrow potred solver)
include("./frankwolfe/frankwolfe.jl")
include("./frankwolfe/wrapper_frankwolfe.jl")
# Arrow–Debreu FrankWolfe.jl wrapper (run_ad_tracked_fwjl); loaded last since it
# reuses FWJL_TABLE (wrapper_frankwolfe.jl) plus ad_market_from_atoms /
# evaluate_test_error_ad (redistribute_ad.jl) and find_cut_single (separation.jl).
include("./frankwolfe/wrapper_frankwolfe_ad.jl")
# Hand-rolled away-step FW for Arrow–Debreu (run_ad_tracked_fw); reuses
# fw_line_search (frankwolfe.jl) + ad_market_from_atoms / evaluate_test_error_ad
# (redistribute_ad.jl) + find_cut_single / _gamma_over_full_from_cand (separation.jl).
include("./frankwolfe/frankwolfe_ad.jl")

# -----------------------------------------------------------------------
# methods: (name, separation_kind, kwargs)
# separation_kind ∈ {:cg_single, :cg_multicut} for CG, :accpm for
# analytic-center CG, :fw for manual Frank-Wolfe, :fwjl for the
# FrankWolfe.jl wrapper.
# The kwargs key :classes selects which function classes the separation
# the per-class separation oracle tries each iteration (defaults to [:ces] when omitted).
# Supported classes: :ces, :linear, :leontief, :ql (separation only — storage TBD).
# -----------------------------------------------------------------------
# The method catalog and per-method plot style are no longer hardcoded here;
# they are loaded from a YAML preset (default revealed/presets.yaml, override
# with --preset PATH). `load_presets!` populates the same globals the rest of
# the scripts consume, so downstream code is unchanged:
#   method_kwargs    Vector of (name, separation_kind, kwargs::Dict) specs
#   colors           Dict name => palette index or named color
#   marker_style     Dict name => marker Symbol
#   display_name     Dict name => pretty label
#   method_variants  Vector{NamedTuple} — the run set (variants run_test iterates)
# CLI flags still override individual kwargs at run time via run_one_method, so
# the preset only sets per-method defaults.

# Convert a YAML-parsed kwarg value to the type the runners expect: strings →
# Symbols (e.g. "diminishing" → :diminishing), arrays of strings →
# Vector{Symbol} (["ces","linear"] → [:ces,:linear]); numbers / bools pass.
_preset_kwval(x::AbstractString) = Symbol(x)
_preset_kwval(x::AbstractVector) = Symbol[Symbol(e) for e in x]
_preset_kwval(x) = x

# A style "color" is either a palette index (Int) or a named color
# ("crimson" → :crimson); markers are always Symbols.
_preset_color(x::AbstractString) = Symbol(x)
_preset_color(x) = x

# Default bundled preset; --preset PATH overrides it.
_default_preset_path() = joinpath(@__DIR__, "presets.yaml")

"""
    load_presets!(path=_default_preset_path()) -> nothing

(Re)populate the method-catalog globals (`method_kwargs`, `colors`,
`marker_style`, `display_name`, `method_variants`) from a YAML preset file.
Called at include time with the default preset; `build_run_config` re-invokes
it with `--preset PATH` when one is supplied.
"""
function load_presets!(path::AbstractString=_default_preset_path())
    isfile(path) || error("preset file not found: $path")
    data = YAML.load_file(path)

    global method_kwargs = Any[]
    global colors = Dict{Symbol,Any}()
    global marker_style = Dict{Symbol,Any}()
    global display_name = Dict{Symbol,String}()
    for entry in get(data, "method", Any[])
        name = Symbol(entry["name"])
        sep = Symbol(entry["separation_kind"])
        kw = Dict{Symbol,Any}(Symbol(k) => _preset_kwval(v)
                              for (k, v) in get(entry, "kwargs", Dict{String,Any}()))
        push!(method_kwargs, (name, sep, kw))
        haskey(entry, "color") && (colors[name] = _preset_color(entry["color"]))
        haskey(entry, "marker") && (marker_style[name] = Symbol(entry["marker"]))
        haskey(entry, "label") && (display_name[name] = String(entry["label"]))
    end
    isempty(method_kwargs) && error("preset $path defines no [[method]] entries")

    # The run set: an ordered list of variants. Each reuses a catalog `method`
    # (inheriting its separation_kind + kwargs) and layers `cli` overrides on
    # top, plus its own plot style. run_test.jl iterates over this list.
    global method_variants = NamedTuple[]
    for v in get(data, "variant", Any[])
        push!(method_variants, (
            label=String(v["label"]),
            method=Symbol(v["method"]),
            sym=Symbol(v["sym"]),
            color=_preset_color(v["color"]),
            marker=Symbol(v["marker"]),
            plotlabel=String(v["plotlabel"]),
            cli=Dict{String,Any}(get(v, "cli", Dict{String,Any}())),
        ))
    end
    return nothing
end

# Load the default catalog at include time so the style tables exist for
# run_plot.jl, which renders from the globals without going through
# build_run_config (and thus without a --preset reload).
load_presets!()

# -----------------------------------------------------------------------
# Shared experiment driver
#
# `build_run_config`, `build_rep_data`, and `run_one_method` were factored
# out of the driver so the auxiliary scripts (qn_tat.jl, run_plc_phased.jl) can
# reuse the exact same CLI plumbing and per-method runner without
# duplicating the kwargs-assembly logic. They take an explicit `cfg`
# (the NamedTuple returned by `build_run_config`) instead of closing over
# script globals.
# -----------------------------------------------------------------------

# Case-insensitive method lookup against the canonical names in the active
# `method_kwargs`. Accepts e.g. `accpm`, `ACCPM`, `AcCpM` → :ACCPM. Computed
# from the current catalog each call so a --preset reload is reflected.
# Unknown tokens raise with the full known list.
_canonical_methods() = Dict(lowercase(String(spec[1])) => spec[1] for spec in method_kwargs)
function _resolve_method(token::AbstractString)
    d = _canonical_methods()
    key = lowercase(strip(token))
    haskey(d, key) ||
        error("unknown method: '$token' (known: $(join(sort(string.(values(d))), ", ")))")
    return d[key]
end

"""
    build_run_config(cli) -> NamedTuple

Derive the experiment configuration from the parsed CLI dict
(`parse_args_for_test_real`). Centralizes the scalar/option unpacking
shared by run_test.jl and the auxiliary scripts (qn_tat.jl, run_plc_phased.jl).
Side effect: `mkpath(out_dir)`.
"""
function build_run_config(cli)
    # Reload the method catalog from the requested preset (default bundled
    # file) BEFORE resolving --methods, so a --preset swap is reflected in the
    # name lookup and the per-method specs the scripts read afterward.
    load_presets!(isempty(cli["preset"]) ? _default_preset_path() : abspath(cli["preset"]))
    # `autofix_names=true` maps hyphens to underscores in the returned dict.
    # Lowercase the market-type token so case-insensitive CLI input (`CES`,
    # `Ces`, `ces`) all resolve to the same dispatch symbol `:ces` AND yield
    # the same lowercase output filename.
    market_type = Symbol(lowercase(strip(cli["market_type"])))
    # Wealth model of the ground truth: 0 = constant (fixed w_i, Fisher),
    # 1 = first-order (endowments, w_i(p) = ⟨p, b_i⟩, Arrow–Debreu). Drives
    # all market families. `lift` emits money-lifted (n+1)-dim data (CES only).
    wealth_function = cli["wealth_function"]
    lift = cli["lift"]
    n = cli["n"]
    m = cli["m"]
    K = cli["k"]
    K_test = cli["k_test"]
    seed = cli["seed"]
    rep = cli["rep"]
    timelimit = cli["timelimit"]
    iterlimit_override = cli["iterlimit"]
    tol_obj_override = cli["tol_obj"]
    tol_delta_override = cli["tol_delta"]
    interval_eval_test = cli["interval_eval_test"]
    # Per-iter excess tracking is OFF by default (0), independent of --validate;
    # it is opt-in via a positive N, or -1 to inherit the test cadence.
    interval_eval_excess = cli["interval_eval_excess"] == -1 ?
                           max(interval_eval_test, 0) : cli["interval_eval_excess"]
    do_validate = !cli["no_validate"]   # default ON; --no-validate disables
    csv_path = cli["csv"]
    verbosity = cli["verbosity"]
    out_dir = abspath(cli["out_dir"])
    mkpath(out_dir)

    # Data-file path mirrors the PDF naming (`real_<market>.jls` in out_dir)
    # unless overridden. `--no-data-file` suppresses the dump entirely.
    data_file_path = if cli["no_data_file"]
        ""
    elseif !isempty(cli["data_file"])
        abspath(cli["data_file"])
    else
        joinpath(out_dir, "real_$(String(market_type)).jls")
    end

    # Select the LP/MIP backend (engine.jl). :auto picks Gurobi when usable,
    # else Mosek; the linear android needs Gurobi's big-M MIP, so it is dropped
    # when Gurobi is unavailable.
    engine = cli["engine"]
    set_engine!(engine)

    method_names = _resolve_method.(split(cli["methods"], ","))
    allowed_classes = Symbol.(lowercase.(strip.(split(cli["classes"], ","))))
    if :linear in allowed_classes && !gurobi_available()
        @warn "Gurobi unavailable: dropping the :linear android (its big-M MIP separation needs Gurobi)."
        allowed_classes = filter(!=(:linear), allowed_classes)
    end
    opt_plc = plc_opt_from_cli(cli)
    opt_splc = splc_opt_from_cli(cli)
    ces_rho_range = ces_rho_range_from_cli(cli)
    sparsity = cli["sparsity"]
    wealth_quad_sparsity = cli["wealth_quad_sparsity"]

    return (; market_type, wealth_function, lift, n, m, K, K_test, seed, rep, timelimit,
        iterlimit_override, tol_obj_override, tol_delta_override,
        interval_eval_test, interval_eval_excess, do_validate, csv_path,
        verbosity, out_dir, data_file_path, method_names, allowed_classes,
        opt_plc, opt_splc, ces_rho_range, sparsity, wealth_quad_sparsity, engine)
end

# -----------------------------------------------------------------------
# Ground-truth budget object for the share/demand families (GES/SPLC/NGES/PLC),
# the analogue of the CES wealth-function switch. Returns the object that
# `wealth_at` / the per-class producer consume, plus bookkeeping:
#   0 → fixed Fisher budget Vector w (Σw = 1)
#   1 → n×m endowment Matrix B (w_i(p) = ⟨p, b_i⟩, Arrow–Debreu), unit supply
#   2 → normalized quadratic wealth fn p ↦ pᵀQ_ip / Σ_j pᵀQ_jp (Σw = 1), Q_i⪰0
# `fkey` is the f_real NamedTuple key (:b for endowments, :w otherwise);
# `M_lift` the lift money supply; `tag` the print-tree label suffix.
# -----------------------------------------------------------------------
function real_budget(wealth_function::Int, n::Int, m::Int, quad_sparsity::Real)
    if wealth_function == 1
        B = rand(n, m)
        B ./= sum(B; dims=2)                 # unit supply per good
        return (B, :b, 1.0, "Arrow–Debreu")
    elseif wealth_function == 2
        Q = random_wealth_quad_tensor(n, m, quad_sparsity)
        return (make_wealth_function(2; Q=Q), :w, 1.0, "second-order quadratic")
    else
        w = rand(m)
        w ./= sum(w)
        return (w, :w, sum(w), "Fisher")
    end
end

# -----------------------------------------------------------------------
# Per-rep data builder (sequential, fast). Returns (Ξ_train, Ξ_test).
# Each rep gets a different seed so reps see independent train/test data.
# -----------------------------------------------------------------------
function build_rep_data(cfg, rep_idx::Int, rep_seed::Int)
    (; market_type, wealth_function, lift, n, m, K, K_test, ces_rho_range, opt_plc, opt_splc, sparsity, wealth_quad_sparsity) = cfg
    Random.seed!(rep_seed)
    # CES: f_real is always a FisherMarket; the per-price wealth is set by a
    #   wealth function (`--wealth-function`: 0 = constant/Fisher, 1 =
    #   first-order/Arrow–Debreu w_i(p)=⟨p,b_i⟩). `--lift` makes the producer
    #   emit (n+1)-dim money-lifted data.
    # For PLC/GES/SPLC: f_real is the NamedTuple `(agents=..., w=...)` /
    #   `(agents=..., b=...)` that the joint-LP equilibrium check in validate.jl
    #   dispatches on; the budget object per --wealth-function is built by
    #   `real_budget` (fixed Vector / endowment Matrix / quadratic wealth fn).
    if lift && market_type === :plc
        @warn "--lift is not implemented for --market-type plc (no single-valued demand); ignoring"
    end
    # Generic Ξ generator for the share/demand-based families (GES/SPLC/NGES):
    # `--lift` routes through the shared money-lifted producer with the
    # `aggregate_real_demand` oracle (emits (n+1)-dim data, money supply
    # M = total Fisher budget, or 1 for the AD endowment case); otherwise the
    # per-class producer. CES has its own oracle; PLC has no single-valued one.
    _lift_announced = Ref(false)
    function _gen(producer, agents, budgets, Kn, seed)
        if lift && market_type !== :plc
            # Total money supply = total ground-truth spending: Σw for fixed
            # Fisher budgets, 1 for endowments / normalized quadratic shares.
            M = budgets isa AbstractVector ? sum(budgets) : 1.0
            _lift_announced[] || (_announce_lift(n, M); _lift_announced[] = true)
            return produce_revealed_preferences_lifted(
                q -> aggregate_real_demand(agents, budgets, q), Kn, n; M=M, seed=seed)
        end
        return producer(agents, budgets, Kn, n; seed=seed)
    end
    f_real = nothing
    wealth_fn = nothing      # ground-truth wealth model w(p); set for CES (used by validation)
    if market_type === :ces
        ρ_lo, ρ_hi = ces_rho_range
        ρ_vec = ρ_lo .+ (ρ_hi - ρ_lo) .* rand(m)
        ws = cpu_workspace(n)
        add_ces!(ws, m; ρ=ρ_vec, scale=30.0, sparsity=sparsity)
        ws.ces.w ./= sum(ws.ces.w)
        f0 = FisherMarket(ws)
        linconstr = LinearConstr(1, n, ones(1, n), [1.0])
        f1 = copy(f0)
        p₀ = ones(n) ./ n
        f1.x .= ones(n, m) ./ m
        alg = HessianBar(n, m, p₀; linconstr=linconstr)
        alg.linsys = :direct
        # Select the wealth function and lift money supply M.
        if wealth_function == 1                  # first-order: w_i(p) = ⟨p, b_i⟩
            B = rand(n, m)
            B ./= sum(B; dims=2)              # unit supply per good
            wealth_fn = make_wealth_function(1; B=B)
            M_lift = 1.0
        elseif wealth_function == 2              # second-order: normalized quadratic shares
            # n×n×m PSD tensor masked to `--wealth-quad-sparsity` density (how
            # concentrated the per-agent shares are; magnitude is irrelevant since
            # wealth_quadratic normalizes Σ_i w_i(p) = 1). Total spending is 1, so
            # the lift's money demand is M+⟨q,1⟩−1 = ⟨q,1⟩ ≥ 0 with M=1 (no blow-up).
            Q = random_wealth_quad_tensor(n, m, wealth_quad_sparsity)
            wealth_fn = make_wealth_function(2; Q=Q)
            M_lift = 1.0
        else                                     # constant (Fisher): fixed budgets
            w0 = collect(f1.w)
            wealth_fn = make_wealth_function(0; w0=w0)
            M_lift = sum(w0)
        end
        wealth_label = wealth_function == 1 ? "first-order (AD)" :
                       wealth_function == 2 ? "second-order (quadratic)" : "constant (Fisher)"
        print_tree("ground-truth: CES (Fisher)", [
            "dimensions" => ["n" => n, "m" => m, "K" => K, "K_test" => K_test],
            "wealth_function" => wealth_label,
            "lift" => lift,
            "ces_ρ_range" => ces_rho_range,
            "ρ_range" => extrema(f1.ρ),
            "σ_range" => extrema(f1.σ),
            "seed" => rep_seed,
        ])
        lift && _announce_lift(n, M_lift)
        Ξ_train = produce_revealed_preferences(alg, f1, K, wealth_fn; seed=rep_seed, lift=lift, M=M_lift)
        Ξ_test = produce_revealed_preferences(alg, f1, K_test, wealth_fn; seed=rep_seed + 1, lift=lift, M=M_lift)
        f_real = f1
    elseif market_type === :plc
        L = opt_plc.L
        plc_agents = [random_plc_agent(n, L; sparsity=sparsity, intercept=opt_plc.intercept) for _ in 1:m]
        # Budget per `--wealth-function`: fixed Fisher budgets, Arrow–Debreu
        # endowments (w_i(p_k)=⟨p_k,b_i⟩), or a price-dependent wealth function;
        # the producer resolves it per sample via `wealth_at`.
        budgets, fkey, _M, tag = real_budget(wealth_function, n, m, wealth_quad_sparsity)
        print_tree("ground-truth: PLC ($tag)", [
            "dimensions" => ["n" => n, "m" => m, "L" => L, "K" => K, "K_test" => K_test],
            "intercept" => opt_plc.intercept,
            "seed" => rep_seed,
        ])
        Ξ_train = produce_revealed_preferences_plc(plc_agents, budgets, K, n; seed=rep_seed)
        Ξ_test = produce_revealed_preferences_plc(plc_agents, budgets, K_test, n; seed=rep_seed + 1)
        f_real = merge((agents=plc_agents,), NamedTuple{(fkey,)}((budgets,)))
    elseif market_type === :ges
        # Non-homothetic polynomial-utility market (cf. sec.ges). Like PLC,
        # f_real is the NamedTuple `(agents=..., w=...)` (Fisher) or
        # `(agents=..., b=...)` (Arrow–Debreu endowments); demand depends on
        # the budget via `share(::GESAgent, p, w)`.
        ges_agents = [random_ges_agent(n) for _ in 1:m]
        budgets, fkey, _M, tag = real_budget(wealth_function, n, m, wealth_quad_sparsity)
        print_tree("ground-truth: GES ($tag)", [
            "dimensions" => ["n" => n, "m" => m, "K" => K, "K_test" => K_test],
            "seed" => rep_seed,
        ])
        Ξ_train = _gen(produce_revealed_preferences_ges, ges_agents, budgets, K, rep_seed)
        Ξ_test = _gen(produce_revealed_preferences_ges, ges_agents, budgets, K_test, rep_seed + 1)
        f_real = merge((agents=ges_agents,), NamedTuple{(fkey,)}((budgets,)))
    elseif market_type === :splc
        # Separable PLC market (cf. sec.splc in choice-ump-utility.tex):
        # u_i(x) = Σ_j f_ij(x_j), each f_ij concave piecewise-linear
        # increasing. The Vazirani–Yannakakis setting — Fisher equilibrium
        # is PPAD-complete, but individual demand is a closed-form greedy
        # (no LP), so sampling is much cheaper than general PLC. Like
        # PLC/GES, f_real is a NamedTuple with fixed budgets `w` (Fisher)
        # or endowments `b` (Arrow–Debreu).
        L_splc = opt_splc.L
        # intercept=false ⇒ homogeneous SPLC: zero intercepts collapse each
        # f_ij to a single linear piece, so every agent is LINEAR
        # (rem.splc.homogeneous) — the homothetic regime of SPLC.
        if !opt_splc.intercept
            @warn "--splc-no-intercept: homogeneous SPLC collapses to ONE linear piece per good " *
                  "(min_ℓ a_ℓ x = (min_ℓ a_ℓ)x, cf. rem.splc.homogeneous) — every ground-truth agent " *
                  "is a linear agent and --splc-L=$(L_splc) is ignored. Linear androids alone should " *
                  "fit this market to ≈0 error."
        end
        splc_agents = [random_splc_agent(n, L_splc; intercept=opt_splc.intercept) for _ in 1:m]
        budgets, fkey, _M, tag = real_budget(wealth_function, n, m, wealth_quad_sparsity)
        print_tree("ground-truth: SPLC ($tag)", [
            "dimensions" => ["n" => n, "m" => m, "L" => L_splc, "K" => K, "K_test" => K_test],
            "intercept" => opt_splc.intercept,
            "seed" => rep_seed,
        ])
        Ξ_train = _gen(produce_revealed_preferences_splc, splc_agents, budgets, K, rep_seed)
        Ξ_test = _gen(produce_revealed_preferences_splc, splc_agents, budgets, K_test, rep_seed + 1)
        f_real = merge((agents=splc_agents,), NamedTuple{(fkey,)}((budgets,)))
    elseif market_type === :nges
        # Non-additive GES real market (REAL-MARKET ONLY; eq.gnae.utility).
        # u(x) = Σ_j c_j (A_i x)_j^{r_j} with strictly positive A_i. Like GES,
        # f_real is the NamedTuple (agents=..., w=...) / (agents=..., b=...);
        # demand depends on the budget via share(::NGESAgent, p, w). There is
        # no NGES surrogate or separation oracle.
        nges_agents = [random_nges_agent(n) for _ in 1:m]
        budgets, fkey, _M, tag = real_budget(wealth_function, n, m, wealth_quad_sparsity)
        print_tree("ground-truth: NGES ($tag)", [
            "dimensions" => ["n" => n, "m" => m, "K" => K, "K_test" => K_test],
            "seed" => rep_seed,
        ])
        Ξ_train = _gen(produce_revealed_preferences_nges, nges_agents, budgets, K, rep_seed)
        Ξ_test = _gen(produce_revealed_preferences_nges, nges_agents, budgets, K_test, rep_seed + 1)
        f_real = merge((agents=nges_agents,), NamedTuple{(fkey,)}((budgets,)))
    else
        error("Unknown market_type: $market_type")
    end
    return (Ξ_train=Ξ_train, Ξ_test=Ξ_test, rep_seed=rep_seed, f_real=f_real, wealth_fn=wealth_fn)
end

# -----------------------------------------------------------------------
# Per-method runner: takes the rep's data + the method spec.
# -----------------------------------------------------------------------
function run_one_method(cfg, cli, rep_idx::Int, rep_seed::Int,
    Ξ_train, Ξ_test, f_real,
    name::Symbol, separation_kind::Symbol, kwargs::Dict; wealth_fn=nothing)
    (; timelimit, interval_eval_test, interval_eval_excess, allowed_classes,
        tol_obj_override, tol_delta_override, iterlimit_override, verbosity,
        do_validate, wealth_function, lift) = cfg
    local_extra = Dict{Symbol,Any}(
        :timelimit => timelimit,
        :interval_eval_test => interval_eval_test,
    )
    # `--no-validate` (do_validate = false) disables BOTH the post-run
    # `validate_surrogate` call in run_test.jl AND the per-iter excess
    # tracking inside the method loop. Per-iter tracking is opt-in
    # (--interval-eval-excess N>0, OFF by default) and now lift-/AD-/wealth-
    # aware via `validate_surrogate` + `wealth_fn`. A CES (FisherMarket)
    # ground truth supports all cases; a PLC NamedTuple ground truth only
    # the fixed-budget joint-LP path (constant wealth, no lift).
    pass_excess = interval_eval_excess > 0 && !isnothing(f_real) &&
                  (f_real isa FisherMarket || (wealth_function == 0 && !lift))
    if do_validate && pass_excess
        local_extra[:f_real] = f_real
        local_extra[:wealth_fn] = wealth_fn
        local_extra[:interval_eval_excess] = interval_eval_excess
    end
    if separation_kind !== :fw
        local_extra[:classes] = allowed_classes
    end
    # Override semantics: > 0 sets the value; == 0 disables the corresponding
    # stop check (stored as `nothing` so runners skip it); < 0 leaves the
    # method's setup.jl default in place.
    if tol_obj_override >= 0
        local_extra[:tol_obj] = tol_obj_override == 0 ? nothing : tol_obj_override
    end
    if tol_delta_override >= 0
        local_extra[:tol_delta] = tol_delta_override == 0 ? nothing : tol_delta_override
    end
    if cli["tol_rc"] >= 0
        local_extra[:tol_rc] = cli["tol_rc"] == 0 ? nothing : cli["tol_rc"]
    end
    if iterlimit_override > 0
        local_extra[:max_iters] = iterlimit_override
    end
    # --interval-logging: shared per-iteration log cadence (:log_interval), honored
    # by every runner. > 0 overrides each method's default / preset; -1 keeps it.
    if get(cli, "interval_logging", -1) > 0
        local_extra[:log_interval] = cli["interval_logging"]
    end
    # Per-class and per-method CLI forwarding lives next to each owner;
    # the keys each apply_*! writes are no-ops for unrelated methods.
    # Add more apply_cli_*! calls here when registering a new arg group.
    apply_cli_separation!(local_extra, cli)
    apply_cli_ces!(local_extra, cli)
    apply_cli_ges!(local_extra, cli)
    apply_cli_linear!(local_extra, cli)
    apply_cli_nn!(local_extra, cli)
    apply_cli_cpm!(local_extra, cli)
    apply_cli_accpm!(local_extra, cli)
    apply_cli_redist!(local_extra, cli)
    apply_cli_ad!(local_extra, cli)
    apply_cli_fw!(local_extra, cli)
    if haskey(kwargs, :seed)
        local_extra[:seed] = rep_seed
    end
    local_kwargs = merge(kwargs, local_extra)
    print_tree("spawned $name", [
        "classes" => get(local_kwargs, :classes, "n/a"),
        "timelimit" => @sprintf("%g s", timelimit),
    ])
    t_elapsed = @elapsed begin
        if separation_kind === :fw
            fa, γ_ref, hist = run_method_tracked_fw(
                name, local_kwargs, Ξ_train, Ξ_test; verbosity=verbosity
            )
        elseif separation_kind === :fwjl
            fa, γ_ref, hist = run_method_tracked_fwjl(
                name, local_kwargs, Ξ_train, Ξ_test; verbosity=verbosity
            )
        elseif separation_kind === :accpm
            fa, γ_ref, hist = run_method_tracked_accpm(
                name, separation_kind, local_kwargs, Ξ_train, Ξ_test; verbosity=verbosity
            )
        elseif separation_kind === :cg_ad
            # Arrow–Debreu CG: returns (fa::ArrowDebreuMarket, γ_ref, hist).
            # run_ad_tracked enforces homothetic-only classes.
            fa, γ_ref, hist = run_ad_tracked(
                local_kwargs, Ξ_train, Ξ_test; verbosity=verbosity
            )
        elseif separation_kind === :ad_fwjl
            # Arrow–Debreu via FrankWolfe.jl (δ = 1): same return triple as the
            # AD CG runner (fa::ArrowDebreuMarket, γ_ref, hist).
            fa, γ_ref, hist = run_ad_tracked_fwjl(
                name, local_kwargs, Ξ_train, Ξ_test; verbosity=verbosity
            )
        elseif separation_kind === :fw_ad
            # Arrow–Debreu via our hand-rolled away-step FW (δ = 1): same triple.
            fa, γ_ref, hist = run_ad_tracked_fw(
                name, local_kwargs, Ξ_train, Ξ_test; verbosity=verbosity
            )
        else
            fa, γ_ref, hist = run_method_tracked(
                name, separation_kind, local_kwargs, Ξ_train, Ξ_test; verbosity=verbosity
            )
        end
    end
    print_tree("$name done", [
        "iters" => length(hist[:primal_obj]),
        "atoms (T)" => fa.m,
        "final train" => @sprintf("%.3e", hist[:primal_obj][end]),
        "final test" => @sprintf("%.3e", hist[:test_err][end]),
        "time" => @sprintf("%.2f s", t_elapsed),
    ])
    return (rep_idx=rep_idx, name=name, fa=fa, hist=hist, t=t_elapsed)
end

# -----------------------------------------------------------------------
# Generic run serialization (mirrors save_plot_ctx / load_plot_ctx in
# run_plot.jl, but kept here so run_one_method.jl needn't pull in the
# plotting stack). Uses Julia's built-in binary `Serialization`.
# -----------------------------------------------------------------------
function save_run(path::AbstractString, payload)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        serialize(io, payload)
    end
    @info "saved run" path
    return path
end

load_run(path::AbstractString) = open(deserialize, path)