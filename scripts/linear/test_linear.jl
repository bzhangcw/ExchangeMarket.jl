#!/usr/bin/env julia
# CPU comparison: shared-target PR, perturbed Newton, and PCG (the same IPM
# using IC(0)-preconditioned CG). Native ExchangeMarket PR is opt-in via
# --methods pr_native and uses MirrorDec/opt! with its native stopping rule.
# From any directory:
#   julia --threads=1 /path/to/test_linear.jl
#   julia --threads=1 /path/to/test_linear.jl --tol 1e-3
#   julia --threads=1 /path/to/test_linear.jl --sizes_n 100 500 1000 2000 --sizes_m 1000 2000 --repeats 3
#   julia --threads=1 /path/to/test_linear.jl --methods perturbed pcg
#   julia --threads=1 /path/to/test_linear.jl --densities 0.01 0.02 0.05
# Sweep all combinations of sizes_n, sizes_m, and densities; sizes_m defaults
# to [2000], and densities default to [0.01, 0.02, 0.05].
# Outputs: runs.csv, summary.csv, and scaling.png, grouped by n, m, and density.
# All methods use the same market and uniform initial prices. Native PR stops
# when norm(Delta p) / maximum(p) < tol^1.5; all returned allocations are also
# scored against the shared clearing, individual-budget, and utility target.
# Timings include solver initialization and diagnostics, but exclude compilation,
# market generation, final validation, plotting, and file export.
# Native PR retains its package diagnostics. The other methods evaluate
# the shared metrics every iterate; native PR is scored on that target only
# after its timed opt! call. Thus stop rules and monitoring costs both differ.

module LinearCPUComparison

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
using Random, SparseArrays, LinearAlgebra, Statistics, Printf
using ExchangeMarket, ArgParse, CSV, DataFrames
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots
include(joinpath(@__DIR__, "linear-perturbed.jl"))
include(joinpath(@__DIR__, "propres.jl"))

const METHODS = (:pr, :pr_native, :perturbed, :pcg)
const DEFAULT_METHODS = (:pr, :perturbed, :pcg)
const LABELS = Dict(:pr => "PR (shared target)", :pr_native => "PR (ExchangeMarket)",
    :perturbed => "Perturbed", :pcg => "PCG")
is_ipm(method) = method in (:perturbed, :pcg)
method_label(method) = LABELS[method]
meets_target(z, tol) = max(z.mc, z.budget, z.regret) <= tol * (1 + 1e-8)

function configuration(args)
    # Preserve the requested underscore spellings of --sizes_n and --sizes_m.
    settings = ArgParseSettings(autofix_names=false,
        description="Compare PR, perturbed Newton, and IC-preconditioned PCG on CPU; native ExchangeMarket PR is opt-in; sweep goods, buyers, and densities.")
    @add_arg_table! settings begin
        "--tol"
            help = "Shared accuracy tolerance and native PR tol (native price-step threshold = tol^1.5)"
            arg_type = Float64
            default = 1e-4
        "--sizes_n"
            help = "Numbers of goods n to benchmark"
            arg_type = Int
            nargs = '+'
            default = [100, 200, 500, 1000, 1500, 2000]
        "--sizes_m"
            help = "Numbers of buyers m to benchmark"
            arg_type = Int
            nargs = '+'
            default = [2000]
        "--methods"
            help = "Methods to run: pr, pr_native, perturbed, pcg (default: pr perturbed pcg)"
            nargs = '+'
            default = string.(collect(DEFAULT_METHODS))
        "--repeats"
            help = "Timed runs per method and (n, m, density) case, after warm-up"
            arg_type = Int
            default = 1
        "--seed"
            arg_type = Int
            default = 1
        "--densities", "--density"
            help = "Valuation densities before repairing empty rows and columns (default: 0.01 0.02 0.05)"
            arg_type = Float64
            nargs = '+'
            default = [0.01, 0.02, 0.05]
        "--maxiter-pr"
            dest_name = "maxiter_pr"
            arg_type = Int
            default = 20_000
        "--maxiter-newton"
            dest_name = "maxiter_newton"
            arg_type = Int
            default = 200
        "--blas-threads"
            dest_name = "blas_threads"
            arg_type = Int
            default = 1
        "--output-dir"
            dest_name = "output_dir"
            help = "Directory for CSV tables and the scaling plot"
            default = joinpath(@__DIR__, "linear_cpu_results")
        "--reference-summary"
            dest_name = "reference_summary"
            help = "Optional earlier summary.csv: verify PR/perturbed numerical results and report timing ratios"
            default = ""
    end
    options = parse_args(args, settings; as_symbols=true)
    cfg = (; options...)
    0 < cfg.tol < 1 || throw(ArgumentError("tol must lie in (0, 1)"))
    all(d -> 0 < d <= 1, cfg.densities) || throw(ArgumentError("densities must lie in (0, 1]"))
    all(>(0), cfg.sizes_n) || throw(ArgumentError("sizes_n must be positive"))
    all(>(0), cfg.sizes_m) || throw(ArgumentError("sizes_m must be positive"))
    cfg.repeats > 0 && cfg.blas_threads > 0 || throw(ArgumentError("repeats and blas-threads must be positive"))
    cfg.maxiter_pr >= 0 && cfg.maxiter_newton >= 0 || throw(ArgumentError("iteration limits must be nonnegative"))
    selected_methods = unique(Symbol.(cfg.methods))
    all(in(METHODS), selected_methods) || throw(ArgumentError("Unknown comparison method"))
    cfg = merge(cfg, (sizes_n=sort(unique(cfg.sizes_n)), sizes_m=sort(unique(cfg.sizes_m)),
        densities=sort(unique(cfg.densities)),
        methods=selected_methods, output_dir=abspath(cfg.output_dir),
        reference_summary=isempty(cfg.reference_summary) ? "" : abspath(cfg.reference_summary)))
    if !isempty(cfg.reference_summary)
        isfile(cfg.reference_summary) || throw(ArgumentError("Reference summary does not exist"))
        cfg.reference_summary != joinpath(cfg.output_dir, "summary.csv") ||
            throw(ArgumentError("Choose a different output-dir so the reference summary is preserved"))
    end
    return cfg
end

function comparison_market(n, m; seed=1, density=0.02)
    rng = MersenneTwister(seed)
    c = Matrix(sprand(rng, n, m, density))
    for i in 1:m
        maximum(view(c, :, i)) == 0 && (c[rand(rng, 1:n), i] = 0.5 + rand(rng))
    end
    for j in 1:n
        maximum(view(c, j, :)) == 0 && (c[j, rand(rng, 1:m)] = 0.5 + rand(rng))
    end
    w = 0.5 .+ rand(rng, m)
    w ./= sum(w)
    ws = cpu_workspace(n)
    add_ces!(ws, m; ρ=1.0, c=c, w=w)
    return FisherMarket(ws)
end

"""Run the package's actual MirrorDec/Bids/opt! driver with its native stop."""
function run_native_pr(f; tol, maxiter)
    market = copy(f)
    p = fill(sum(market.w) / market.n, market.n)
    # Uniform x gives native iterate!'s uniform initial bids. Its initial scale
    # cancels in the first Bids update, matching run_pr after that update.
    market.x .= 1 / market.m
    market.g .= market.w' ./ market.n
    alg = MirrorDec(market.n, market.m, p; optimizer=Bids,
        option_step=:shmyrev, tol, maxiter, maxtime=Inf)
    alg.μ = 0.0
    alg.t = alg.tₗ = 0.0
    alg.dₙ = Inf
    # opt! prints unconditionally. Suppress terminal output, while retaining
    # all of the package's computation, diagnostics, and final play!.
    redirect_stdout(devnull) do
        opt!(alg, market; tol, maxiter, maxtime=Inf, keep_traj=false,
            loginterval=max(1, maxiter))
    end
    native_price_step = alg.dₙ / maximum(alg.p)
    native_price_threshold = tol^1.5
    native_stop_met = native_price_step < native_price_threshold
    status = native_stop_met ? :native_converged : :max_iter
    # Final play! recovers x from the current bids/prices, then advances bids.
    # Score its returned x; recomputing g ./ p here would use the extra update.
    market.p .= alg.p
    return (; st=(p=alg.p, x=market.x), traj=[(k=alg.k,)], status, factorizations=0,
        native_price_step, native_price_threshold, native_stop_met)
end

function run_method(method, f, cfg)
    stop = z -> meets_target(z, cfg.tol)
    method == :pr && return run_pr(f; stop, maxiter=cfg.maxiter_pr)
    method == :pr_native && return run_native_pr(f; tol=cfg.tol, maxiter=cfg.maxiter_pr)
    is_ipm(method) || throw(ArgumentError("Unknown method $method"))
    n, m = size(f.c)
    μ₀ = sum(f.w) / (n * (m + 1))
    σ_min = cfg.tol * minimum(f.w) / n
    return linear_perturbed_ipm(f; p₀=fill(sum(f.w) / n, n), μ₀, σ₀=μ₀, σ_min,
        tol=cfg.tol, stop, maxiter=cfg.maxiter_newton, show_trace=false,
        linear_solver=method == :pcg ? :pcg_ic : :cholesky,
        linear_rtol=1e-9, ic_drop=0.0, ic_normalize=true)
end

function check_pr_agreement(cfg)
    rows = NamedTuple[]
    for m in cfg.sizes_m, density in cfg.densities,
        n in unique([first(cfg.sizes_n), last(cfg.sizes_n)])
        market = comparison_market(n, m; seed=cfg.seed, density)
        for cap in (1, 5, 20)
            # A stringent native tolerance makes these fixed-step comparisons;
            # an exactly stationary market may legitimately stop earlier.
            native = run_native_pr(market; tol=1e-14, maxiter=cap)
            k = native.traj[end].k
            shared = run_pr(market; stop=z -> false, maxiter=k)
            price_error = norm(native.st.p - shared.st.p, Inf) / norm(shared.st.p, Inf)
            allocation_error = norm(native.st.x - shared.st.x, Inf) /
                max(1.0, norm(shared.st.x, Inf))
            agrees = price_error < 1e-12 && allocation_error < 1e-12
            push!(rows, (n=n, m=m, density=density, requested_updates=cap, updates=k,
                price_error, allocation_error, agrees))
        end
    end
    CSV.write(joinpath(cfg.output_dir, "pr_agreement.csv"), DataFrame(rows))
    @assert all(r.agrees for r in rows) "Native and shared-target PR iterates differ; see pr_agreement.csv"
    println("Native/shared PR agreement verified at ", length(rows), " checkpoints.")
    flush(stdout)
    return rows
end

function measured_run(method, f, cfg, repetition, actual_density)
    GC.gc()
    timed = @timed run_method(method, f, cfg)
    result = timed.value
    z = linear_equilibrium_metrics(f.c, f.w, result.st.p, result.st.x)
    target_met = meets_target(z, cfg.tol)
    @assert all(isfinite, (z.mc, z.budget, z.regret, z.total_spending))
    @assert result.status != :converged || target_met
    @assert isapprox(z.total_spending, dot(result.st.p, vec(sum(result.st.x; dims=2))); rtol=1e-12)
    if is_ipm(method)
        @assert all(t.σ >= max(t.μ, result.sigma_floor) for t in result.traj)
        @assert isapprox(z.total_spending, sum(f.w) + f.n * f.m * result.st.σ; rtol=1e-10)
    end
    # Work-count convention: PR updates or total inner PCG iterations;
    # direct factorization has no comparable matrix-vector-product count.
    matrix_vector_products = method == :perturbed ? "-" :
        method == :pcg ? sum((s.iterations for s in result.linear_stats); init=0) :
        result.traj[end].k
    # Report aggregate spending/budget only; keep the notebook's individual-
    # budget check inside target_met and error_ratio.
    return (n=f.n, m=f.m, tol=cfg.tol, seed=cfg.seed, density=cfg.density,
        actual_density=actual_density, julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(), maxiter_pr=cfg.maxiter_pr,
        maxiter_newton=cfg.maxiter_newton, method=method_label(method),
        repetition=repetition, status=string(result.status), target_met=target_met,
        stop_rule=method == :pr_native ? "native_price_step" : "shared_accuracy",
        native_stop_met=method == :pr_native ? result.native_stop_met : missing,
        native_price_step=method == :pr_native ? result.native_price_step : missing,
        native_price_threshold=method == :pr_native ? result.native_price_threshold : missing,
        updates=result.traj[end].k, factorizations=result.factorizations,
        matrix_vector_products=matrix_vector_products,
        linear_solver=is_ipm(method) ? (method == :pcg ? "pcg_ic" : "cholesky") : missing,
        linear_rtol=is_ipm(method) ? 1e-9 : missing,
        average_linear_iterations=is_ipm(method) ? result.average_linear_iterations : missing,
        preconditioner_factorizations=is_ipm(method) ? result.preconditioner_factorizations : 0,
        linear_setup_seconds=is_ipm(method) ? result.t_preconditioner : 0.0,
        ic_drop=method == :pcg ? 0.0 : missing,
        ic_normalized=method == :pcg ? true : missing,
        wall_seconds=timed.time, gc_seconds=timed.gctime, allocated_MiB=timed.bytes / 2.0^20,
        mc=z.mc, regret=z.regret, error_ratio=max(z.mc, z.budget, z.regret) / cfg.tol,
        total_spending=z.total_spending, total_budget=z.total_budget,
        total_budget_error=z.total_budget_error,
        sigma=is_ipm(method) ? result.st.σ : missing,
        mu=is_ipm(method) ? result.st.μ : missing,
        sigma_min=cfg.tol * minimum(f.w) / f.n)
end

function summarize(rows)
    times = [r.wall_seconds for r in rows]
    mid = median(times)
    representative = rows[argmin(abs.(times .- mid))]
    return merge(representative, (repeats=length(rows), median_seconds=mid,
        min_seconds=minimum(times), max_seconds=maximum(times),
        median_allocated_MiB=median([r.allocated_MiB for r in rows]),
        all_targets_met=all(r.target_met for r in rows)))
end

function save_plot(rows, cfg)
    gr()
    panels = []
    for m in cfg.sizes_m, density in cfg.densities
        case_rows = filter(r -> r.m == m && r.density == density, rows)
        timing = plot(xlabel="Goods (n)", ylabel="Median wall time (s)", yscale=:log10,
            title="CPU time (m = $m, density = $density)", legend=:bottomright)
        updates = plot(xlabel="Goods (n)", ylabel="Outer updates",
            title="Iterations (m = $m, density = $density)", legend=false)
        for (method, color) in zip(METHODS, (:steelblue, :darkorange, :seagreen, :purple))
            rs = sort(filter(r -> r.method == method_label(method), case_rows); by=r -> r.n)
            isempty(rs) && continue
            ns = [r.n for r in rs]
            for (panel, values) in ((timing, [r.median_seconds for r in rs]),
                                    (updates, [r.updates for r in rs]))
                # Only successful runs form the scaling curve. A cross shows a
                # stopped run that did not reach the shared target.
                good = [r.all_targets_met for r in rs]
                plot!(panel, ns, [ok ? v : NaN for (ok, v) in zip(good, values)];
                    label=method_label(method), color, marker=:circle, lw=2)
                if any(.!good)
                    scatter!(panel, ns[.!good], values[.!good]; color, marker=:x,
                        label="", markersize=7)
                end
            end
        end
        if any(!r.all_targets_met for r in case_rows)
            scatter!(timing, [NaN], [NaN]; color=:black, marker=:x, markersize=7,
                label="Shared target not met")
        end
        append!(panels, (timing, updates))
    end
    panel_rows = length(cfg.sizes_m) * length(cfg.densities)
    fig = plot(panels...; layout=(panel_rows, 2),
        size=(1100, 420 * panel_rows), margin=6Plots.mm,
        plot_title="tol = $(cfg.tol)")
    savefig(fig, joinpath(cfg.output_dir, "scaling.png"))
end

function compare_reference(summary, cfg)
    isempty(cfg.reference_summary) && return nothing
    reference = CSV.read(cfg.reference_summary, DataFrame)
    comparison = NamedTuple[]
    for row in summary
        row.method in (method_label(:pr), method_label(:perturbed)) || continue
        old_label = row.method == method_label(:pr) ? "PR" : row.method
        matches = filter(r -> r.n == row.n && r.m == row.m && r.density == row.density &&
            r.method in (old_label, row.method), reference)
        nrow(matches) == 1 || error("Expected one reference row for n=$(row.n), m=$(row.m), density=$(row.density), $(row.method)")
        ref = only(eachrow(matches))
        for key in (:m, :tol, :seed, :density, :maxiter_pr, :maxiter_newton)
            getproperty(ref, key) == getproperty(row, key) || error("Reference setting differs: $key")
        end
        same_metrics = all(isapprox(getproperty(ref, key), getproperty(row, key);
            atol=1e-10, rtol=1e-8) for key in
            (:mc, :regret, :error_ratio, :total_spending, :total_budget, :total_budget_error))
        reproduced = ref.status == row.status && ref.updates == row.updates &&
            ref.target_met == row.target_met && same_metrics
        push!(comparison, (n=row.n, m=row.m, density=row.density, method=row.method, reproduced,
            reference_updates=ref.updates, updates=row.updates,
            reference_status=ref.status, status=row.status,
            reference_seconds=ref.median_seconds, seconds=row.median_seconds,
            time_ratio=row.median_seconds / ref.median_seconds))
    end
    isempty(comparison) && return nothing
    CSV.write(joinpath(cfg.output_dir, "reproduction.csv"), DataFrame(comparison))
    @assert all(r.reproduced for r in comparison) "Numerical reproduction differs; see reproduction.csv"
    println("Reproduced numerical results in all ", length(comparison),
        " reference rows. Timings are measured afresh, not required to match exactly.")
    return comparison
end

function main(args=ARGS)
    cfg = configuration(args)
    BLAS.set_num_threads(cfg.blas_threads)
    mkpath(cfg.output_dir)
    println("Julia ", VERSION, " | CPU | Julia threads = ", Threads.nthreads(),
        " | BLAS threads = ", BLAS.get_num_threads())
    println("n = ", cfg.sizes_n, " | m = ", cfg.sizes_m, " | tol = ", cfg.tol,
        " | densities = ", cfg.densities, " | repeats = ", cfg.repeats)
    println("Methods: ", join(method_label.(cfg.methods), ", "))
    println("PR (shared target), Perturbed, and PCG use the common clearing, budget, and utility target.")
    println("PCG uses IC(0), scalar-normalized inverse, zero initial direction, linear rtol = 1e-9; setup time is included.")
    if :pr_native in cfg.methods
        println("PR (ExchangeMarket) calls native opt!: relative price step < tol^1.5 = ", cfg.tol^1.5)
    end
    println("target_met always scores the same shared target for every selected method.")
    println("Budget reporting compares total spending with total budget.\n")
    flush(stdout)

    # When native PR is selected, verify its agreement with buffered PR outside
    # all benchmark timers. Default runs do not invoke the native driver.
    :pr_native in cfg.methods && check_pr_agreement(cfg)

    # The same dense Float64 code paths apply at every size. Compile complete
    # solves once on a small market, rather than timing compilation at each n.
    warmup_cfg = merge(cfg, (density=first(cfg.densities),))
    warmup = comparison_market(min(100, minimum(cfg.sizes_n)), min(2000, minimum(cfg.sizes_m));
        seed=cfg.seed, density=warmup_cfg.density)
    for method in cfg.methods
        result = run_method(method, warmup, warmup_cfg)
        println("Warm-up ", method_label(method), ": ", result.status)
        flush(stdout)
    end
    warmup = nothing
    runs, summary = NamedTuple[], NamedTuple[]
    for m in cfg.sizes_m, n in cfg.sizes_n, density in cfg.densities
        case_cfg = merge(cfg, (; density))
        market = comparison_market(n, m; seed=cfg.seed, density)
        actual_density = count(!iszero, market.c) / length(market.c)
        println("\nn = ", n, ", m = ", m, ", density = ", density)
        for repetition in 1:cfg.repeats
            for method in circshift(cfg.methods, repetition - 1)
                @printf("  %-20s run %d/%d ... ", method_label(method), repetition, cfg.repeats)
                flush(stdout)
                row = measured_run(method, market, case_cfg, repetition, actual_density)
                push!(runs, row)
                @printf("%8.3f s | %5d updates | %-20s | shared target=%s | spent %.8f / budget %.8f\n",
                    row.wall_seconds, row.updates, row.status, string(row.target_met), row.total_spending, row.total_budget)
                flush(stdout)
                CSV.write(joinpath(cfg.output_dir, "runs.csv"), DataFrame(runs))
            end
        end
        for method in cfg.methods
            push!(summary, summarize(filter(r -> r.n == n && r.m == m && r.density == density &&
                r.method == method_label(method), runs)))
        end
        CSV.write(joinpath(cfg.output_dir, "summary.csv"), DataFrame(summary))
        compare_reference(summary, cfg)
    end
    save_plot(summary, cfg)
    println("\nMedian-time comparison (remaining fields use the run closest to the median):")
    show(stdout, MIME("text/plain"), select(DataFrame(summary), :n, :m, :density, :method,
        :all_targets_met, :status, :updates, :factorizations, :matrix_vector_products,
        :average_linear_iterations,
        :median_seconds, :regret, :total_spending, :total_budget);
        allrows=true, allcols=true)
    println("\n\nResults saved in ", cfg.output_dir)
    return (; runs, summary)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    LinearCPUComparison.main()
end
