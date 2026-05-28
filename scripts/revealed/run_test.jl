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
include("./androids/plc.jl")
# androids/ql.jl is included transitively via separation.jl (loaded by
# setup.jl below); it's the source of `solve_separation_ql` for the
# :ql android class — the ground-truth :ql MARKET branch was removed.
include("./setup.jl")

switch_to_pdf(; bool_use_html=false)

const cli = parse_args_for_test_real()

# All CLI unpacking / derivation lives in `build_run_config` (setup.jl) so
# run_one_method.jl shares the exact same plumbing. Destructure the fields
# the rest of this script references by bare name.
cfg = build_run_config(cli)
(; market_type, n, m, K, K_test, seed, rep, timelimit, iterlimit_override,
    tol_obj_override, tol_delta_override, interval_eval_test,
    interval_eval_excess, do_validate, csv_path, verbosity, out_dir,
    data_file_path, method_names, allowed_classes, opt_plc, ces_rho_range,
    sparsity) = cfg

method_filter(name) = name in method_names

@info "configuration" market_type n m K seed rep timelimit methods = method_names classes = allowed_classes ces_rho_range sparsity

# `build_rep_data` and `run_one_method` now live in setup.jl (shared with
# run_one_method.jl); both take the `cfg` NamedTuple built above.

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
    rep_data[r] = build_rep_data(cfg, r, seed + (r - 1) * 1000)
end

grid_tasks = Task[]
for r in 1:rep, (name, pk, kw) in selected_methods
    rd = rep_data[r]
    push!(grid_tasks, Threads.@spawn run_one_method(cfg, cli, r, rd.rep_seed, rd.Ξ_train, rd.Ξ_test, rd.f_real, name, pk, kw))
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
# Final-iter scalars (androids, train, test, t) are reported as mean.
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
        "method", "iters", "androids", "train_obj (std)", "test_err (std)",
        "time (s) (std)", "‖p(q-g)‖∞ (std)")
else
    @printf("%-12s %8s %8s %20s %20s %16s\n",
        "method", "iters", "androids", "train_obj (std)", "test_err (std)",
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
