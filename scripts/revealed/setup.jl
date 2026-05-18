# Methods and tracking utilities for revealed-preference CES surrogate fitting.
# File map (in include order):
#   - gurobi_env.jl                : shared Gurobi env singleton (_gurobi_env);
#                                    loaded first so redistribute.jl + androids/linear.jl share it
#   - redistribute.jl              : wealth-redistribution primal / dual LPs
#                                    (solve_wealth_redistribution_{primal,dual},
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
using ExchangeMarket

# -----------------------------------------------------------------------
# Shared Gurobi env (used by master LP and the linear MILP pricer).
# Loaded first so the license banner fires once at script load.
# -----------------------------------------------------------------------
include("./gurobi_env.jl")

# -----------------------------------------------------------------------
# Master / dual LP solvers (define before runners include them).
# -----------------------------------------------------------------------
include("./redistribute.jl")

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

    for k in 1:K
        # Random price vector (normalized to sum to 1)
        p_k = price_range[1] .+ (price_range[2] - price_range[1]) .* rand(n)
        p_k = p_k ./ sum(p_k)  # normalize prices

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
# evaluation on test set: mean L∞ error of Σ w_i γ_i(p) vs target P g
# -----------------------------------------------------------------------
function evaluate_test_error(fa, Ξ_test)
    isempty(fa.w) && return NaN
    K = length(Ξ_test)
    n = length(Ξ_test[1][1])
    errs = Float64[]
    for (p, g) in Ξ_test
        target = p .* g
        fitted = zeros(n)
        for i in 1:fa.m
            c_i = Vector(fa.c[:, i])
            fitted .+= fa.w[i] .* compute_gamma(p, c_i, fa.σ[i])
        end
        push!(errs, norm(fitted .- target, Inf))
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

    # ---- Problem instance ------------------------------------------------
    add_arg_group!(s, "Problem instance")
    @add_arg_table! s begin
        "--market-type", "-t"
        help = "Ground-truth market family"
        arg_type = String
        default = "ces"
        range_tester = x -> x in ("ces", "plc", "ql")
        "--n", "-n"
        help = "Number of goods"
        arg_type = Int
        default = 5
        "--m", "-m"
        help = "Number of agents in the real market"
        arg_type = Int
        default = 50
        "--k", "-k"
        help = "Number of training (and test) observations"
        arg_type = Int
        default = 100
        "--seed", "-s"
        help = "Master random seed"
        arg_type = Int
        default = 42
        "--rep"
        help = "Number of repetitions (different seeds). When > 1, the plot shows mean ± 1σ ribbon across reps."
        arg_type = Int
        default = 1
        "--sparsity"
        help = "Per-agent coefficient sparsity (CES `c` / PLC `A`): fraction of entries set to 0 when generating the ground-truth market. 0 = fully dense, 1 = all-zero (degenerate). 0.2 (default) matches prior behavior."
        arg_type = Float64
        default = 0.2
        range_tester = x -> 0.0 <= x < 1.0
    end

    # ---- IO --------------------------------------------------------------
    add_arg_group!(s, "IO")
    @add_arg_table! s begin
        "--out-dir"
        help = "Output directory for the data file (and the .csv if --csv is set without an absolute path)."
        arg_type = String
        default = @__DIR__
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

    # ---- Stopping --------------------------------------------------------
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
        help = "Reduced-cost tolerance for the stage-1 convergence check; > 0 overrides each method's :tol_rc; 0 disables the rc-based stop entirely; < 0 uses setup.jl default."
        arg_type = Float64
        default = 1e-3
    end

    # ---- Method selection ------------------------------------------------
    add_arg_group!(s, "Method selection")
    @add_arg_table! s begin
        "--methods"
        help = """Comma-separated method names to run (any of cg,cgma,fw,sfw,fwjl), fw and sfw are implemented by myself. By default we don't include them yet, instead we compare to FrankWolfe.jl"""
        arg_type = String
        default = "cg,cgma,fwjl"
        "--classes"
        help = "Comma-separated function classes for android classes in separation (any of ces,linear,leontief,ql)"
        arg_type = String
        default = "ces,linear,leontief"
    end

    # ---- Evaluation ------------------------------------------------------
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
        help = "Skip the CES surrogate equilibrium validation (default ON for --market-type ces; PLC / QL are always skipped)."
        action = :store_true
    end

    # ---- Per-method arg groups ----------------------------------------
    # Each method's CLI surface lives in its own runner file
    # (cpm.jl::register_cli_cpm!, accpm.jl::register_cli_accpm!). Add
    # more method-specific blocks here when you add a runner — the goal
    # is for run_test.jl to stay focused on cross-cutting infrastructure.
    register_cli_cpm!(s)
    register_cli_accpm!(s)

    # ---- Per-class separation arg groups ----------------------------------
    # Separation CLI surface lives in the separation files themselves so each
    # class owns its own knobs (mirrors register_cli_cpm! / _accpm!
    # above). Add a new separation class? Add register_cli_<class>! to its
    # file and one line here.
    register_cli_separation!(s)
    register_cli_ces!(s)
    register_cli_linear!(s)
    register_cli_plc!(s)   # ground-truth PLC market flags (--plc-L, --plc-no-intercept)

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
    [:cgma, :cg_multicut,
        Dict(
            :max_iters => 500,
            :tol_obj => 1e-3,
            :tol_rc => 1e-3,
            :tol_delta => 1e-5,
            :tol_stage_2 => 5e-4,   # demote stage 2 → 1 on this looser stall
            :stage1_ces_rho => 0.97, # post-demotion: near-linear CES (σ ≈ 32)
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
)

marker_style = Dict(
    :CG => :circle,
    :cgma => :rect,
    :FW => :diamond,
    :SFW => :star5,
    :FWjl => :rect,
    :ACCPM => :utriangle,
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
)