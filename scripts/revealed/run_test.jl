# Compare CG / Multicut / FW / SFW on a real market (CES or PLC).
# Separate training and test revealed-preference sets of the same size K,
# track train primal objective and test error per iteration, then plot.
#
# When --rep > 1, the whole experiment (data resampling + method runs) is
# repeated with different seeds, and the plot shows the mean trajectory
# per method with a ±1σ ribbon over the rep dimension.
#
# CLI: see `julia run_test.jl -h` for all options.

using Revise
using Random, SparseArrays, LinearAlgebra
using DelimitedFiles
using ArgParse
using JuMP, MosekTools
using Plots, LaTeXStrings, Printf
using Statistics
import MathOptInterface as MOI

using ExchangeMarket

include("../tools.jl")
include("../plots.jl")
include("./plc.jl")
include("./ql.jl")
include("./setup.jl")

switch_to_pdf(; bool_use_html=false)

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
        "--plc-L"
        help = "Number of PLC pieces (PLC market only)"
        arg_type = Int
        default = 5
        "--plc-no-intercept"
        help = "Disable PLC intercept (b=0); only meaningful for --market-type plc"
        action = :store_true
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
        help = "0 = silent; 1 = per-iteration table; 2 = + per-pricing detail"
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
        help = """Comma-separated method names to run (any of CG,MultiCut,FW,SFW,FWjl), FW and SFW are implemented by myself, so by default we compare to FrankWolfe.jl"""
        arg_type = String
        default = "CG,MultiCut,FWjl"
        "--classes"
        help = "Comma-separated function classes for CG dispatch (any of ces,linear,leontief,ql)"
        arg_type = String
        default = "ces,linear,leontief"
    end

    # ---- Method: CG/MultiCut -----------------------------------------------------
    add_arg_group!(s, "Method: CG/MultiCut")
    @add_arg_table! s begin
        "--stage-2-tol"
        help = "MultiCut only: improvement threshold below which stage 2 auto-demotes to stage 1 (single-cut cleanup). > 0 overrides the per-method :tol_stage_2; ≤ 0 (default) keeps the in-code default (typically equals :tol_delta)."
        arg_type = Float64
        default = -1.0
        "--android-dropping-interval"
        help = "Drop zero-weight androids (columns) from the surrogate market every N iterations (working like regularization). 5 (default) is every iter; larger keeps dormant atoms longer (cheaper LP, but more clutter in the surrogate)."
        arg_type = Int
        default = 5
    end

    # ---- Pricing (shared across all classes) -----------------------------
    add_arg_group!(s, "Pricing (shared)")
    @add_arg_table! s begin
        "--sample-size"
        help = "Mini-batch size for the pricing oracle (Higle-Sen / Joachims style). If 0 (default) or >= K, uses the full training set; otherwise each pricing call sees a random subset of this size, master uses full K."
        arg_type = Int
        default = 0
    end

    # ---- Pricing: CES ----------------------------------------------------
    add_arg_group!(s, "Pricing: CES")
    @add_arg_table! s begin
        "--stage1-ces-rho"
        help = "MultiCut only: after the stage-2 → stage-1 demotion, restrict the CES pricer to a fixed ρ (corresponding σ = ρ/(1-ρ)) instead of the free-σ search. ρ near 1 (e.g. 0.97) yields a near-linear CES boundary cleanup. ≤ 0 (default) keeps the unrestricted free-σ behavior."
        arg_type = Float64
        default = -1.0
    end

    # ---- Pricing: Linear -------------------------------------------------
    add_arg_group!(s, "Pricing: Linear")
    @add_arg_table! s begin
        "--linear-pricing-indicator"
        help = "Linear pricing MILP: use Gurobi indicator constraints instead of the default big-M formulation. Slower on dense u (more per-node overhead), but tighter LP relaxation on sparse data."
        action = :store_true
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

    return parse_args(argv, s)
end

const cli = parse_args_for_test_real()

# `autofix_names=true` maps hyphens to underscores in the returned dict.
market_type = Symbol(cli["market_type"])
n = cli["n"]
m = cli["m"]
K = cli["k"]
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

method_names = Symbol.(split(cli["methods"], ","))
allowed_classes = Symbol.(split(cli["classes"], ","))
opt_plc = (L=cli["plc_L"], intercept=!cli["plc_no_intercept"])

method_filter(name) = name in method_names

@info "configuration" market_type n m K seed rep timelimit methods = method_names classes = allowed_classes

# -----------------------------------------------------------------------
# Per-rep data builder (sequential, fast). Returns (Ξ_train, Ξ_test).
# Each rep gets a different seed so reps see independent train/test data.
# -----------------------------------------------------------------------
function build_rep_data(rep_idx::Int, rep_seed::Int)
    Random.seed!(rep_seed)
    # For CES: f_real is a FisherMarket.
    # For PLC: f_real is the NamedTuple `(agents=..., w=...)` that the
    # joint-LP equilibrium check in validate.jl dispatches on.
    f_real = nothing
    if market_type === :ces
        ρ_vec = -3.5 .+ 4.3 .* rand(m)
        ws = cpu_workspace(n)
        add_ces!(ws, m; ρ=ρ_vec, scale=30.0, sparsity=0.99)
        ws.ces.w ./= sum(ws.ces.w)
        f0 = FisherMarket(ws)
        linconstr = LinearConstr(1, n, ones(1, n), [1.0])
        f1 = copy(f0)
        p₀ = ones(n) ./ n
        f1.x .= ones(n, m) ./ m
        alg = HessianBar(n, m, p₀; linconstr=linconstr)
        alg.linsys = :direct
        @info "[rep $rep_idx] Ground-truth CES market" n m K seed = rep_seed σ_range = extrema(f1.σ)
        Ξ_train = produce_revealed_preferences(alg, f1, K; seed=rep_seed)
        Ξ_test = produce_revealed_preferences(alg, f1, K; seed=rep_seed + 1)
        f_real = f1
    elseif market_type === :plc
        L = opt_plc.L
        plc_agents = [random_plc_agent(n, L; intercept=opt_plc.intercept) for _ in 1:m]
        w_vec = rand(m)
        w_vec ./= sum(w_vec)
        @info "[rep $rep_idx] Ground-truth PLC market" n m L K seed = rep_seed
        Ξ_train = produce_revealed_preferences_plc(plc_agents, w_vec, K, n; seed=rep_seed)
        Ξ_test = produce_revealed_preferences_plc(plc_agents, w_vec, K, n; seed=rep_seed + 1)
        f_real = (agents=plc_agents, w=w_vec)
    elseif market_type === :ql
        ql_agents = [random_ql_agent(n) for _ in 1:m]
        w_vec = rand(m)
        w_vec ./= sum(w_vec)
        @info "[rep $rep_idx] Ground-truth QL market" n m K seed = rep_seed
        Ξ_train = produce_revealed_preferences_ql(ql_agents, w_vec, K, n; seed=rep_seed)
        Ξ_test = produce_revealed_preferences_ql(ql_agents, w_vec, K, n; seed=rep_seed + 1)
    else
        error("Unknown market_type: $market_type")
    end
    return (Ξ_train=Ξ_train, Ξ_test=Ξ_test, rep_seed=rep_seed, f_real=f_real)
end

# -----------------------------------------------------------------------
# Per-method runner: takes the rep's data + the method spec.
# -----------------------------------------------------------------------
function run_one_method(rep_idx::Int, rep_seed::Int,
    Ξ_train, Ξ_test, f_real,
    name::Symbol, pricing_kind::Symbol, kwargs::Dict)
    local_extra = Dict{Symbol,Any}(
        :timelimit => timelimit,
        :interval_eval_test => interval_eval_test,
    )
    if !isnothing(f_real) && interval_eval_excess > 0
        local_extra[:f_real] = f_real
        local_extra[:interval_eval_excess] = interval_eval_excess
    end
    if pricing_kind !== :fw
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
    if cli["sample_size"] > 0
        local_extra[:sample_size] = cli["sample_size"]
    end
    if cli["android_dropping_interval"] > 0
        local_extra[:interval_dropping] = cli["android_dropping_interval"]
    end
    if cli["linear_pricing_indicator"]
        local_extra[:use_indicators] = true
    end
    if cli["stage1_ces_rho"] > 0
        local_extra[:stage1_ces_rho] = cli["stage1_ces_rho"]
    end
    if cli["stage_2_tol"] > 0
        local_extra[:tol_stage_2] = cli["stage_2_tol"]
    end
    if haskey(kwargs, :seed)
        local_extra[:seed] = rep_seed
    end
    local_kwargs = merge(kwargs, local_extra)
    @info "[rep $rep_idx] spawned $name" classes = get(local_kwargs, :classes, "n/a") timelimit = timelimit
    t_elapsed = @elapsed begin
        if pricing_kind === :fw
            fa, γ_ref, hist = run_method_tracked_fw(
                name, local_kwargs, Ξ_train, Ξ_test; verbosity=verbosity
            )
        elseif pricing_kind === :fwjl
            fa, γ_ref, hist = run_method_tracked_fwjl(
                name, local_kwargs, Ξ_train, Ξ_test; verbosity=verbosity
            )
        else
            fa, γ_ref, hist = run_method_tracked(
                name, pricing_kind, local_kwargs, Ξ_train, Ξ_test; verbosity=verbosity
            )
        end
    end
    @info "[rep $rep_idx] $name done" iters = length(hist[:primal_obj]) atoms_T = fa.m final_train = hist[:primal_obj][end] final_test = hist[:test_err][end] time_s = t_elapsed
    return (rep_idx=rep_idx, name=name, fa=fa, hist=hist, t=t_elapsed)
end

# -----------------------------------------------------------------------
# Drive the reps × methods grid concurrently.
# Reps' data are built sequentially (cheap setup); the (rep, method)
# pairs are then `Threads.@spawn`-ed all at once so the scheduler can
# utilize every available thread across the whole grid.
# -----------------------------------------------------------------------
@info "task scheduler" nthreads = Threads.nthreads() rep methods = method_names

selected_methods = [(name, pk, kw) for (name, pk, kw) in method_kwargs if method_filter(name)]

rep_data = Vector{NamedTuple}(undef, rep)
for r in 1:rep
    rep_data[r] = build_rep_data(r, seed + (r - 1) * 1000)
end

grid_tasks = Task[]
for r in 1:rep, (name, pk, kw) in selected_methods
    rd = rep_data[r]
    push!(grid_tasks, Threads.@spawn run_one_method(r, rd.rep_seed, rd.Ξ_train, rd.Ξ_test, rd.f_real, name, pk, kw))
end
grid_results = [fetch(t) for t in grid_tasks]

per_rep_results = [Dict{Symbol,NamedTuple}() for _ in 1:rep]
for res in grid_results
    per_rep_results[res.rep_idx][res.name] = (fa=res.fa, hist=res.hist, t=res.t)
end

# -----------------------------------------------------------------------
# Optional validation: for each fitted surrogate, solve its CES
# equilibrium and judge it by the real market via the price-scaled
# excess demand z(p_s) = p_s · (q − g_real(p_s)). Detailed per-rep
# results are printed below; the per-method mean ± std is then folded
# into the summary table.
# -----------------------------------------------------------------------
validate_per_rep = [Dict{Symbol,NamedTuple}() for _ in 1:rep]
do_validate_effective = do_validate && (market_type === :ces || market_type === :plc)
if do_validate && !(market_type === :ces || market_type === :plc)
    @warn "--validate only supported for --market-type ces|plc; skipping" market_type
end
if do_validate_effective
    println()
    println("=== Validation: real-market clearing at surrogate equilibrium price ===")
    @printf("%4s | %-10s | %12s | %12s\n",
        "rep", "method", "‖p(q-g)‖∞", "‖p(q-g)‖₁")
    @printf("%4s-+-%-10s-+-%12s-+-%12s\n",
        "----", "----------", "------------", "------------")
    for r in 1:rep
        f_real = rep_data[r].f_real
        f_real === nothing && continue
        for name in method_names
            haskey(per_rep_results[r], name) || continue
            fa = per_rep_results[r][name].fa
            v = validate_surrogate(fa, f_real; verbose=false)
            validate_per_rep[r][name] = v
            pretty = get(display_name, name, String(name))
            @printf("%4d | %-10s | %12.3e | %12.3e\n",
                r, pretty, v.excess_surrogate_linf, v.excess_surrogate_l1)
        end
    end
end

# -----------------------------------------------------------------------
# Per-method aggregation across reps:
#   - pad trajectories with their last value to the longest length
#   - compute mean & std at each iteration
# Final-iter scalars (atoms, train, test, t) are reported as mean.
# -----------------------------------------------------------------------
function pad_forward(v::Vector{T}, L::Int) where {T}
    length(v) >= L && return v[1:L]
    last = v[end]
    return vcat(v, fill(last, L - length(v)))
end

agg = Dict{Symbol,NamedTuple}()
for name in method_names
    runs = [pr[name] for pr in per_rep_results if haskey(pr, name)]
    isempty(runs) && continue
    Ls = [length(r.hist[:primal_obj]) for r in runs]
    Lmax = maximum(Ls)
    train_mat = hcat([pad_forward(max.(r.hist[:primal_obj], 1e-8), Lmax) for r in runs]...)
    test_mat = hcat([pad_forward(max.(r.hist[:test_err], 1e-8), Lmax) for r in runs]...)
    # Per-iter market excess history (NaN-filled when not tracked).
    excess_runs = [haskey(r.hist, :excess) && !isempty(r.hist[:excess]) ?
                   pad_forward(max.(r.hist[:excess], 1e-8), Lmax) :
                   fill(NaN, Lmax) for r in runs]
    excess_mat = hcat(excess_runs...)
    has_excess_curve = any(!isnan, excess_mat)
    nag = [r.hist[:num_agents][end] for r in runs]
    # Std of final values across reps — NaN when only one rep so the
    # summary table can render "-" instead of a fake number.
    _std_or_nan(v) = length(v) > 1 ? std(v) : NaN
    # Validation metric (market excess at surrogate equilibrium).
    val_vals = Float64[get(validate_per_rep[r], name, (excess_surrogate_linf=NaN,)).excess_surrogate_linf
                       for r in 1:rep if haskey(per_rep_results[r], name)]
    val_clean = filter(!isnan, val_vals)
    finals = (
        train_mean=mean(train_mat[end, :]),
        train_std=_std_or_nan(train_mat[end, :]),
        test_mean=mean(test_mat[end, :]),
        test_std=_std_or_nan(test_mat[end, :]),
        t_mean=mean(r.t for r in runs),
        t_std=_std_or_nan([r.t for r in runs]),
        atoms_mean=mean(r.fa.m for r in runs),
        iters_mean=mean(Ls),
        nag_mean=mean(nag),
        val_mean=isempty(val_clean) ? NaN : mean(val_clean),
        val_std=_std_or_nan(val_clean),
    )
    train_mean = vec(mean(train_mat; dims=2))
    train_std = vec(std(train_mat; dims=2))
    train_min = vec(minimum(train_mat; dims=2))
    train_max = vec(maximum(train_mat; dims=2))
    test_mean = vec(mean(test_mat; dims=2))
    test_std = vec(std(test_mat; dims=2))
    test_min = vec(minimum(test_mat; dims=2))
    test_max = vec(maximum(test_mat; dims=2))
    # Excess summary stats — skipnans across reps; per-iter slots with all
    # NaN reps stay NaN so the plotter draws gaps cleanly.
    _safe_mean(v) = (vc = filter(!isnan, v); isempty(vc) ? NaN : mean(vc))
    _safe_std(v) = (vc = filter(!isnan, v); length(vc) > 1 ? std(vc) : NaN)
    _safe_min(v) = (vc = filter(!isnan, v); isempty(vc) ? NaN : minimum(vc))
    _safe_max(v) = (vc = filter(!isnan, v); isempty(vc) ? NaN : maximum(vc))
    excess_mean = [_safe_mean(excess_mat[i, :]) for i in 1:Lmax]
    excess_std = [_safe_std(excess_mat[i, :]) for i in 1:Lmax]
    excess_min = [_safe_min(excess_mat[i, :]) for i in 1:Lmax]
    excess_max = [_safe_max(excess_mat[i, :]) for i in 1:Lmax]
    agg[name] = (
        runs=runs,
        Lmax=Lmax,
        train_mean=train_mean, train_std=train_std, train_min=train_min, train_max=train_max,
        test_mean=test_mean, test_std=test_std, test_min=test_min, test_max=test_max,
        excess_mean=excess_mean, excess_std=excess_std,
        excess_min=excess_min, excess_max=excess_max,
        has_excess=has_excess_curve,
        finals=finals,
    )
end

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
println("\n=== Summary (mean over $rep rep$(rep == 1 ? "" : "s")) ===")
fmt_std(s) = isnan(s) ? "       -" : @sprintf("%8.1e", s)
fmt_mean(v) = @sprintf("%9.3e", v)
if do_validate_effective
    @printf("%-12s %8s %8s %20s %20s %16s %20s\n",
        "method", "iters", "atoms", "train_obj (std)", "test_err (std)",
        "time (s) (std)", "‖p(q-g)‖∞ (std)")
else
    @printf("%-12s %8s %8s %20s %20s %16s\n",
        "method", "iters", "atoms", "train_obj (std)", "test_err (std)",
        "time (s) (std)")
end
for name in method_names
    haskey(agg, name) || continue
    f = agg[name].finals
    if do_validate_effective
        val_str = isnan(f.val_mean) ? "        -" : fmt_mean(f.val_mean)
        @printf("%-12s %8.1f %8.1f %s (%s) %s (%s) %6.3f (%s) %s (%s)\n",
            get(display_name, name, String(name)),
            f.iters_mean, f.atoms_mean,
            fmt_mean(f.train_mean), fmt_std(f.train_std),
            fmt_mean(f.test_mean), fmt_std(f.test_std),
            f.t_mean, fmt_std(f.t_std),
            val_str, fmt_std(f.val_std))
    else
        @printf("%-12s %8.1f %8.1f %s (%s) %s (%s) %6.3f (%s)\n",
            get(display_name, name, String(name)),
            f.iters_mean, f.atoms_mean,
            fmt_mean(f.train_mean), fmt_std(f.train_std),
            fmt_mean(f.test_mean), fmt_std(f.test_std),
            f.t_mean, fmt_std(f.t_std))
    end
end

# Optional CSV log: one row per (rep, method).
if !isempty(csv_path)
    csv_path = isabspath(csv_path) ? csv_path : joinpath(@__DIR__, csv_path)
    new_file = !isfile(csv_path)
    open(csv_path, "a") do io
        if new_file
            println(io, "market_type,n,m,K,rep,method,iters,atoms_T,train_obj,test_err,time_s")
        end
        for r_idx in 1:rep
            for name in method_names
                haskey(per_rep_results[r_idx], name) || continue
                run = per_rep_results[r_idx][name]
                println(io, join((
                        String(market_type), n, m, K, r_idx, String(name),
                        length(run.hist[:primal_obj]), run.fa.m,
                        run.hist[:primal_obj][end], run.hist[:test_err][end],
                        round(run.t; digits=3),
                    ), ","))
            end
        end
    end
    @info "appended results" csv_path
end

# -----------------------------------------------------------------------
# Persist the per-method aggregation context. Plotting is delegated to
# run_plot.jl so this script stays focused on the benchmark itself; see
# the hint printed at the end for the exact replay command.
# -----------------------------------------------------------------------
include("./run_plot.jl")
plot_ctx = build_plot_ctx(;
    agg=agg, rep=rep, market_type=market_type,
    n=n, m=m, K=K, opt_plc=opt_plc,
    method_names=method_names, interval_marker=10,
)
if !isempty(data_file_path)
    save_plot_ctx(data_file_path, plot_ctx)
    println()
    println("─"^60)
    println("To render the risk curves:")
    println("  julia --project=. revealed/run_plot.jl -f $(data_file_path)")
    println("  (add --smooth N for a moving-average window, --no-tex to skip pgfplots .tex)")
    println("─"^60)
end
