# Methods and tracking utilities for revealed-preference CES surrogate fitting.
# File map (in include order):
#   - gurobi_env.jl                : shared Gurobi env singleton (_gurobi_env);
#                                    loaded first so redistribute.jl + androids/linear.jl share it
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
using ExchangeMarket

# -----------------------------------------------------------------------
# Shared Gurobi env (used by master LP and the linear MILP pricer).
# Loaded first so the license banner fires once at script load.
# -----------------------------------------------------------------------
include("./gurobi_env.jl")

# -----------------------------------------------------------------------
# SPLC ground-truth market (separable PLC; greedy closed-form demand).
# Included here (not in the drivers) because build_rep_data below needs
# random_splc_agent / produce_revealed_preferences_splc. The general-PLC
# generator (androids/plc.jl) stays driver-included since it needs Mosek.
# -----------------------------------------------------------------------
include("./androids/splc.jl")

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

    # `price_range` is kept on the signature for backward compatibility
    # but is unused: drawing uniform on `[lo, hi]^n` and then normalizing
    # is NOT uniform on the unit simplex (the normalization Jacobian
    # biases away from corners). Sampling exponential RVs and normalizing
    # is equivalent to Dirichlet(1, …, 1), which IS uniform on Δ_{n-1}.
    for k in 1:K
        e_k = -log.(rand(n))                 # n iid Exp(1)
        p_k = e_k ./ sum(e_k)                # uniform on the unit simplex

        # Set price in the algorithm
        alg.p .= p_k

        # Compute demand via play!
        play!(alg, f1)

        # Aggregate demand: sum over all agents
        g_k = sum(f1.x, dims=2)[:]

        Ξ[k] = (copy(p_k), copy(g_k))
    end

    return Ξ
end

"""
    produce_revealed_preferences_ad(ad::ArrowDebreuMarket, K; seed=nothing)

Arrow–Debreu sibling of `produce_revealed_preferences`: prices are sampled
uniformly on the simplex (same Exp(1)-normalize recipe), and the aggregate
demand at each price is the closed-form `aggregate_demand(ad, p_k)` from
src/models/arrow.jl — each agent's budget is its endowment value
w_i(p_k) = ⟨p_k, b_i⟩, so no equilibrium solver (play!/HessianBar) is needed.
"""
function produce_revealed_preferences_ad(ad::ArrowDebreuMarket, K::Int; seed=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    n = ad.n
    Ξ = Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, K)
    for k in 1:K
        e_k = -log.(rand(n))                 # n iid Exp(1)
        p_k = e_k ./ sum(e_k)                # uniform on the unit simplex
        g_k = aggregate_demand(ad, p_k)      # price-dependent budgets ⟨p_k, b_i⟩
        Ξ[k] = (copy(p_k), copy(g_k))
    end
    return Ξ
end

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
        description="Benchmark CG / Multicut / FW / SFW on a CES or PLC market.",
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

    # ---- (2) Market: ground-truth family + budget model + per-family knobs
    add_arg_group!(s, "Market")
    @add_arg_table! s begin
        "--market-type", "-t"
        help = "Ground-truth market family"
        arg_type = String
        default = "ces"
        range_tester = x -> x in ("ces", "plc", "ges", "splc")
        "--budget-type"
        help = "Budget model of the ground-truth market: fisher (fixed budgets w_i, default) or ad (Arrow–Debreu: endowments b_i with price-dependent budgets w_i(p)=⟨p,b_i⟩). Supported for all market types (ces/plc/ges)."
        arg_type = String
        default = "fisher"
        range_tester = x -> x in ("fisher", "ad")
    end
    # One register_cli_*_market! call per market family. These only
    # bite when `--market-type` selects the matching family.
    register_cli_ces_market!(s)   # --ces-rho-low, --ces-rho-high
    register_cli_plc!(s)          # --plc-L, --plc-no-intercept
    register_cli_splc!(s)         # --splc-L

    # ---- (3) Method selection + per-method knobs ---------------------
    add_arg_group!(s, "Method selection")
    @add_arg_table! s begin
        "--methods"
        help = """Comma-separated method names to run (any of cg,cgma,fw,sfw,fwjl), fw and sfw are implemented by myself. By default we don't include them yet, instead we compare to FrankWolfe.jl"""
        arg_type = String
        default = "cg,cgma,fwjl"
        "--classes"
        help = "Comma-separated function classes for android classes in separation (any of ces,linear,leontief,ql,nn)"
        arg_type = String
        default = "ces,linear,leontief"
    end
    # Each method owns its CLI surface in its own runner file
    # (cpm.jl::register_cli_cpm!, accpm.jl::register_cli_accpm!).
    register_cli_cpm!(s)
    register_cli_accpm!(s)
    # Master LP (redistribute.jl) owns --redist-use-nlp and --redist-nonh-w.
    register_cli_redist!(s)
    # AD master (redistribute_ad.jl) owns --ad-endow-mode and --ad-mask-size.
    register_cli_ad!(s)

    # ---- (4) Separation oracle knobs ---------------------------------
    # Separation CLI surface lives in the per-android files so each
    # class owns its own knobs (mirrors register_cli_cpm! / _accpm!).
    # Shared knobs first, then per-class.
    register_cli_separation!(s)   # --sample-size
    register_cli_ces!(s)          # --ces-sigma-lower, --ces-sigma-upper
    register_cli_ges!(s)          # --ges-sigma-{lower,upper}, --ges-y-{lower,upper}
    register_cli_linear!(s)       # --linear-separation-indicator
    register_cli_nn!(s)           # --nn-hidden, --nn-iters

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
        help = "Evaluate market-excess ‖p(q-g)‖∞ every N iterations (CES only; needs validation). Default -1 inherits --interval-eval-test; 0 disables per-iter tracking."
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
include("./frankwolfe/frankwolfe.jl")
include("./frankwolfe/wrapper_frankwolfe.jl")

# -----------------------------------------------------------------------
# methods: (name, separation_kind, kwargs)
# separation_kind ∈ {:cg_single, :cg_multicut} for CG, :accpm for
# analytic-center CG, :fw for manual Frank-Wolfe, :fwjl for the
# FrankWolfe.jl wrapper.
# The kwargs key :classes selects which function classes the separation
# the per-class separation oracle tries each iteration (defaults to [:ces] when omitted).
# Supported classes: :ces, :linear, :leontief, :ql (separation only — storage TBD).
# -----------------------------------------------------------------------
method_kwargs = [
    [:CG, :cg_single,
        Dict(
            :max_iters => 500,
            :tol_obj => 1e-3,
            :tol_rc => 1e-5,
            :tol_delta => 1e-5,
            :drop => true,
            :classes => [:ces, :linear],
        )
    ],
    # Arrow–Debreu column generation: same CG loop with the AD master
    # (endowments b_t ∈ ℝⁿ₊, Σ_t b_t = 1, price-dependent budget ⟨p,b_t⟩).
    # separation_kind :cg_ad routes run_one_method to run_ad_tracked.
    # Homothetic classes only (ces,linear,leontief); --classes overrides.
    [:adcg, :cg_ad,
        Dict(
            :max_iters => 500,
            :tol_obj => 1e-3,
            :tol_rc => 1e-5,
            :tol_delta => 1e-5,
            :drop => true,
            :classes => [:ces],
        )
    ],
    [:cgma, :cg_multicut,
        Dict(
            :max_iters => 500,
            :tol_obj => 1e-3,
            :tol_rc => 1e-3,
            :tol_delta => 1e-5,
            :tol_stage_2 => 5e-4,   # demote stage 2 → 1 on this looser stall
            :drop => true,
            :classes => [:ces],
        )
    ],
    [:FW, :fw,
        Dict(
            :max_iters => 10000,
            :batch_size => 0,           # 0 → full batch; set e.g. 32 for stochastic
            :tol_obj => 1e-3,
            :tol_delta => 1e-5,
            :step_rule => :diminishing,
            :seed => 0,
        )
    ],
    [:SFW, :fw,
        Dict(
            :max_iters => 10000,
            :batch_size => 32,          # mini-batch stochastic FW
            :tol_obj => 1e-3,
            :tol_delta => 1e-5,
            :step_rule => :diminishing,
            :seed => 0,
        )
    ],
    [:FWjl, :fwjl,
        Dict(
            :max_iters => 50000,
            :tol_obj => 1e-3,
            :seed => 0,
        )
    ],
    [:ACCPM, :accpm,
        Dict(
            :max_iters => 500,
            :tol_obj => 1e-3,
            :tol_rc => 1e-3,
            :tol_delta => 1e-5,
            # Cuts must NOT be dropped in ACCPM: each γ_t is a polytope
            # constraint ⟨u, γ_t⟩ ≤ μ, and dropping it expands the
            # polytope (the AC then moves backward, undoing prior cuts).
            # Vertex-CG can drop zero-weight columns safely because the
            # LP optimum is at a polytope vertex; ACCPM's AC interior is
            # a different object.
            :drop => false,
            :classes => [:ces, :linear],
            # `:multicut` defaults to false in run_method_tracked_accpm
            # (K per-sample inversions at the interior AC u are collectively
            # redundant and inflate the polytope without improving primal).
            # Override here to opt in for specific instances.
            :interval_primal => 1,   # refresh μ_ub + record primal_obj every iter
        )
    ],
]

colors = Dict(
    :CG => 1,
    :cgma => 2,
    :FW => 4,
    :SFW => 5,
    :FWjl => 3,
    :ACCPM => 6,
    :adcg => 7,
)

marker_style = Dict(
    :CG => :circle,
    :cgma => :rect,
    :FW => :diamond,
    :SFW => :star5,
    :FWjl => :rect,
    :ACCPM => :utriangle,
    :adcg => :pentagon,
)

# Pretty display names for legends and summary output. The CLI / symbol
# table key remains the Julia-friendly identifier; this dict lets us
# render dots or whitespace in labels (e.g., `FWjl` → "FW.jl"). Falls
# back to `String(name)` for unlisted methods.
display_name = Dict(
    :CG => "CG",
    :cgma => "CG(MA)",
    :FW => "FW",
    :SFW => "SFW",
    :FWjl => "FW.jl",
    :adcg => "AD-CG",
)

# -----------------------------------------------------------------------
# Shared experiment driver
#
# `build_run_config`, `build_rep_data`, and `run_one_method` were factored
# out of run_test.jl so the single-method script (run_one_method.jl) can
# reuse the exact same CLI plumbing and per-method runner without
# duplicating the kwargs-assembly logic. They take an explicit `cfg`
# (the NamedTuple returned by `build_run_config`) instead of closing over
# script globals.
# -----------------------------------------------------------------------

# Case-insensitive method lookup against the canonical names in
# `method_kwargs`. Accepts e.g. `accpm`, `ACCPM`, `AcCpM` → :ACCPM.
# Unknown tokens raise with the full known list.
const _CANONICAL_METHOD = Dict(lowercase(String(spec[1])) => spec[1]
                               for spec in method_kwargs)
const _METHOD_LIST_STR = join(sort(collect(values(_CANONICAL_METHOD))), ", ")
function _resolve_method(token::AbstractString)
    key = lowercase(strip(token))
    haskey(_CANONICAL_METHOD, key) ||
        error("unknown method: '$token' (known: $_METHOD_LIST_STR)")
    return _CANONICAL_METHOD[key]
end

"""
    build_run_config(cli) -> NamedTuple

Derive the experiment configuration from the parsed CLI dict
(`parse_args_for_test_real`). Centralizes the scalar/option unpacking
shared by run_test.jl and run_one_method.jl. Side effect: `mkpath(out_dir)`.
"""
function build_run_config(cli)
    # `autofix_names=true` maps hyphens to underscores in the returned dict.
    # Lowercase the market-type token so case-insensitive CLI input (`CES`,
    # `Ces`, `ces`) all resolve to the same dispatch symbol `:ces` AND yield
    # the same lowercase output filename.
    market_type = Symbol(lowercase(strip(cli["market_type"])))
    # Budget model of the ground truth: :fisher (fixed w_i) or :ad
    # (endowments, w_i(p) = ⟨p, b_i⟩). Supported for all market types:
    # CES rides ArrowDebreuMarket; PLC/GES carry an n×m endowment matrix
    # in their f_real NamedTuple and evaluate budgets per sample.
    budget_type = Symbol(lowercase(strip(cli["budget_type"])))
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
    # Default: per-iter excess shares the test cadence; -1 sentinel inherits.
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

    method_names = _resolve_method.(split(cli["methods"], ","))
    allowed_classes = Symbol.(lowercase.(strip.(split(cli["classes"], ","))))
    opt_plc = plc_opt_from_cli(cli)
    opt_splc = splc_opt_from_cli(cli)
    ces_rho_range = ces_rho_range_from_cli(cli)
    sparsity = cli["sparsity"]

    return (; market_type, budget_type, n, m, K, K_test, seed, rep, timelimit,
        iterlimit_override, tol_obj_override, tol_delta_override,
        interval_eval_test, interval_eval_excess, do_validate, csv_path,
        verbosity, out_dir, data_file_path, method_names, allowed_classes,
        opt_plc, opt_splc, ces_rho_range, sparsity)
end

# -----------------------------------------------------------------------
# Per-rep data builder (sequential, fast). Returns (Ξ_train, Ξ_test).
# Each rep gets a different seed so reps see independent train/test data.
# -----------------------------------------------------------------------
function build_rep_data(cfg, rep_idx::Int, rep_seed::Int)
    (; market_type, budget_type, n, m, K, K_test, ces_rho_range, opt_plc, opt_splc, sparsity) = cfg
    Random.seed!(rep_seed)
    # For CES + fisher budgets: f_real is a FisherMarket.
    # For CES + ad budgets: f_real is an ArrowDebreuMarket (endowments b,
    #   budgets w_i(p) = ⟨p, b_i⟩).
    # For PLC/GES: f_real is the NamedTuple `(agents=..., w=...)` that the
    # joint-LP equilibrium check in validate.jl dispatches on.
    f_real = nothing
    if market_type === :ces && budget_type === :ad
        # Arrow–Debreu ground truth: same per-agent CES preferences as the
        # Fisher branch (ρ sampled from ces_rho_range, scale-30 coefficients),
        # but each agent owns an endowment column b_i; budgets are the
        # endowment values at the sampled price. Demand is closed-form
        # (aggregate_demand in src/models/arrow.jl), so no play!/HessianBar.
        ρ_lo, ρ_hi = ces_rho_range
        ρ_vec = ρ_lo .+ (ρ_hi - ρ_lo) .* rand(m)
        b_mat = rand(n, m)
        # c is generated inside the constructor (scale * sprand(n, m, sparsity)),
        # same coefficient policy as the Fisher branch's add_ces!.
        f_real = ArrowDebreuMarket(m, n; ρ=ρ_vec, b=b_mat,
            scale=30.0, sparsity=sparsity,
            bool_unit_supply=true, verbose=false, seed=rep_seed)
        print_tree("[rep $rep_idx] ground-truth: CES Arrow–Debreu", [
            "dimensions" => ["n" => n, "m" => m, "K" => K, "K_test" => K_test],
            "ces_ρ_range" => ces_rho_range,
            "ρ_range" => extrema(f_real.ρ),
            "σ_range" => extrema(f_real.σ),
            "seed" => rep_seed,
        ])
        Ξ_train = produce_revealed_preferences_ad(f_real, K; seed=rep_seed)
        Ξ_test = produce_revealed_preferences_ad(f_real, K_test; seed=rep_seed + 1)
    elseif market_type === :ces
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
        print_tree("[rep $rep_idx] ground-truth: CES (Fisher)", [
            "dimensions" => ["n" => n, "m" => m, "K" => K, "K_test" => K_test],
            "ces_ρ_range" => ces_rho_range,
            "ρ_range" => extrema(f1.ρ),
            "σ_range" => extrema(f1.σ),
            "seed" => rep_seed,
        ])
        Ξ_train = produce_revealed_preferences(alg, f1, K; seed=rep_seed)
        Ξ_test = produce_revealed_preferences(alg, f1, K_test; seed=rep_seed + 1)
        f_real = f1
    elseif market_type === :plc
        L = opt_plc.L
        plc_agents = [random_plc_agent(n, L; sparsity=sparsity, intercept=opt_plc.intercept) for _ in 1:m]
        if budget_type === :ad
            # Arrow–Debreu: each agent owns an endowment column b_i (n×m),
            # normalized so each good's total endowment is 1 (unit supply).
            # The producer evaluates budgets per sample as w_i(p_k) = ⟨p_k, b_i⟩.
            B = rand(n, m)
            B ./= sum(B; dims=2)
            print_tree("[rep $rep_idx] ground-truth: PLC Arrow–Debreu", [
                "dimensions" => ["n" => n, "m" => m, "L" => L, "K" => K, "K_test" => K_test],
                "intercept" => opt_plc.intercept,
                "seed" => rep_seed,
            ])
            Ξ_train = produce_revealed_preferences_plc(plc_agents, B, K, n; seed=rep_seed)
            Ξ_test = produce_revealed_preferences_plc(plc_agents, B, K_test, n; seed=rep_seed + 1)
            f_real = (agents=plc_agents, b=B)
        else
            w_vec = rand(m)
            w_vec ./= sum(w_vec)
            print_tree("[rep $rep_idx] ground-truth: PLC (Fisher)", [
                "dimensions" => ["n" => n, "m" => m, "L" => L, "K" => K, "K_test" => K_test],
                "intercept" => opt_plc.intercept,
                "seed" => rep_seed,
            ])
            Ξ_train = produce_revealed_preferences_plc(plc_agents, w_vec, K, n; seed=rep_seed)
            Ξ_test = produce_revealed_preferences_plc(plc_agents, w_vec, K_test, n; seed=rep_seed + 1)
            f_real = (agents=plc_agents, w=w_vec)
        end
    elseif market_type === :ges
        # Non-homothetic polynomial-utility market (cf. sec.ges). Like PLC,
        # f_real is the NamedTuple `(agents=..., w=...)` (Fisher) or
        # `(agents=..., b=...)` (Arrow–Debreu endowments); demand depends on
        # the budget via `share(::GESAgent, p, w)`.
        ges_agents = [random_ges_agent(n) for _ in 1:m]
        if budget_type === :ad
            B = rand(n, m)
            B ./= sum(B; dims=2)
            print_tree("[rep $rep_idx] ground-truth: GES Arrow–Debreu", [
                "dimensions" => ["n" => n, "m" => m, "K" => K, "K_test" => K_test],
                "seed" => rep_seed,
            ])
            Ξ_train = produce_revealed_preferences_ges(ges_agents, B, K, n; seed=rep_seed)
            Ξ_test = produce_revealed_preferences_ges(ges_agents, B, K_test, n; seed=rep_seed + 1)
            f_real = (agents=ges_agents, b=B)
        else
            w_vec = rand(m)
            w_vec ./= sum(w_vec)
            print_tree("[rep $rep_idx] ground-truth: GES (Fisher)", [
                "dimensions" => ["n" => n, "m" => m, "K" => K, "K_test" => K_test],
                "seed" => rep_seed,
            ])
            Ξ_train = produce_revealed_preferences_ges(ges_agents, w_vec, K, n; seed=rep_seed)
            Ξ_test = produce_revealed_preferences_ges(ges_agents, w_vec, K_test, n; seed=rep_seed + 1)
            f_real = (agents=ges_agents, w=w_vec)
        end
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
        if budget_type === :ad
            B = rand(n, m)
            B ./= sum(B; dims=2)
            print_tree("[rep $rep_idx] ground-truth: SPLC Arrow–Debreu", [
                "dimensions" => ["n" => n, "m" => m, "L" => L_splc, "K" => K, "K_test" => K_test],
                "intercept" => opt_splc.intercept,
                "seed" => rep_seed,
            ])
            Ξ_train = produce_revealed_preferences_splc(splc_agents, B, K, n; seed=rep_seed)
            Ξ_test = produce_revealed_preferences_splc(splc_agents, B, K_test, n; seed=rep_seed + 1)
            f_real = (agents=splc_agents, b=B)
        else
            w_vec = rand(m)
            w_vec ./= sum(w_vec)
            print_tree("[rep $rep_idx] ground-truth: SPLC (Fisher)", [
                "dimensions" => ["n" => n, "m" => m, "L" => L_splc, "K" => K, "K_test" => K_test],
                "intercept" => opt_splc.intercept,
                "seed" => rep_seed,
            ])
            Ξ_train = produce_revealed_preferences_splc(splc_agents, w_vec, K, n; seed=rep_seed)
            Ξ_test = produce_revealed_preferences_splc(splc_agents, w_vec, K_test, n; seed=rep_seed + 1)
            f_real = (agents=splc_agents, w=w_vec)
        end
    else
        error("Unknown market_type: $market_type")
    end
    return (Ξ_train=Ξ_train, Ξ_test=Ξ_test, rep_seed=rep_seed, f_real=f_real)
end

# -----------------------------------------------------------------------
# Per-method runner: takes the rep's data + the method spec.
# -----------------------------------------------------------------------
function run_one_method(cfg, cli, rep_idx::Int, rep_seed::Int,
    Ξ_train, Ξ_test, f_real,
    name::Symbol, separation_kind::Symbol, kwargs::Dict)
    (; timelimit, interval_eval_test, interval_eval_excess, allowed_classes,
        tol_obj_override, tol_delta_override, iterlimit_override, verbosity,
        do_validate, budget_type) = cfg
    local_extra = Dict{Symbol,Any}(
        :timelimit => timelimit,
        :interval_eval_test => interval_eval_test,
    )
    # `--no-validate` (do_validate = false) disables BOTH the post-run
    # `validate_surrogate` call in run_one_method.jl AND the per-iter
    # excess tracking inside the CG loop. The per-iter call goes
    # through the same `validate_surrogate` machinery (MirrorDec /
    # CESAnalytic) and crashes on mixed surrogates (any non-CES atom in
    # `fa.storage.gen`: QL, GES, Leontief), so users running with those
    # classes should pass `--no-validate` to skip both.
    # An AD ground truth (--budget-type ad, any market type) is also excluded:
    # validate_surrogate has no method for price-dependent budgets (AD
    # validation is deferred).
    if do_validate && !isnothing(f_real) && interval_eval_excess > 0 &&
       budget_type !== :ad
        local_extra[:f_real] = f_real
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
    if haskey(kwargs, :seed)
        local_extra[:seed] = rep_seed
    end
    local_kwargs = merge(kwargs, local_extra)
    print_tree("[rep $rep_idx] spawned $name", [
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
        else
            fa, γ_ref, hist = run_method_tracked(
                name, separation_kind, local_kwargs, Ξ_train, Ξ_test; verbosity=verbosity
            )
        end
    end
    print_tree("[rep $rep_idx] $name done", [
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