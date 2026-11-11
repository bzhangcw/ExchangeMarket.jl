#!/usr/bin/env julia
# Standalone PR / Perturbed / PCG comparison with a shared CPU/GPU runner.
# julia test-linear.jl --device cpu --sizes_n 500 --sizes_m 1000 --tol 1e-4
# julia test-gpu.jl                         # same case on CUDA, using cuDSS / IC(0)
# The generated market is copied to the selected device once per case. Transfer
# time is reported separately; wall_seconds includes solver initialization,
# sparse analysis, numerical factorization, diagnostics and synchronization.
# Compilation, market generation, plotting and export are excluded. allocated_MiB
# is Julia host allocation accounting, not GPU allocation accounting.

module LinearComparison

using Pkg
requested_gpu = get(ENV, "LINEAR_DEVICE", "cpu") == "gpu" ||
    any(i -> ARGS[i] == "--device" && i < length(ARGS) && ARGS[i+1] == "gpu", eachindex(ARGS)) ||
    "--device=gpu" in ARGS
Pkg.activate(get(ENV, "LINEAR_PROJECT", requested_gpu ? joinpath(@__DIR__, "gpu") : normpath(joinpath(@__DIR__, ".."))))
using Random, SparseArrays, LinearAlgebra, Statistics, Printf
using ArgParse, CSV, DataFrames
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots
include(joinpath(@__DIR__, "linear-perturbed.jl"))
include(joinpath(@__DIR__, "propres.jl"))
requested_gpu && include(joinpath(@__DIR__, "linear-gpu.jl"))

const METHODS = (:pr, :perturbed, :pcg)
const DEFAULT_METHODS = (:pr, :perturbed, :pcg)
const LABELS = Dict(:pr => "PR",
    :perturbed => "Perturbed", :pcg => "PCG")
is_ipm(method) = method in (:perturbed, :pcg)
method_label(method) = LABELS[method]
meets_target(z, tol) = max(z.mc, z.budget, z.regret) <= tol * (1 + 1e-8)

function configuration(args)
    # Preserve the requested underscore spellings of --sizes_n and --sizes_m.
    settings = ArgParseSettings(autofix_names=false,
        description="Compare standalone PR, perturbed IPM, and IC-preconditioned PCG; select CPU or GPU.")
    @add_arg_table! settings begin
        "--device"
            help = "Array backend: cpu or gpu (GPU uses CUDA)"
            default = get(ENV, "LINEAR_DEVICE", "cpu")
        "--direct-solver"
            dest_name = "direct_solver"
            help = "Direct backend: auto, cholesky (CPU), or cudss (GPU)"
            default = "auto"
        "--tol"
            help = "Shared accuracy tolerance; sigma_min = tol * minimum(w) / n"
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
            help = "Methods to run: pr, perturbed, pcg (default: all three)"
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
            default = ""
        "--reference-summary"
            dest_name = "reference_summary"
            help = "Optional earlier summary.csv: verify PR/perturbed numerical results and report timing ratios"
            default = ""
    end
    options = parse_args(args, settings; as_symbols=true)
    cfg = (; options...)
    cfg.device in ("cpu", "gpu") || throw(ArgumentError("device must be cpu or gpu"))
    cfg.direct_solver in ("auto", "cholesky", "cudss") || throw(ArgumentError("unknown direct solver"))
    direct = cfg.direct_solver == "auto" ? (cfg.device == "gpu" ? :cudss : :cholesky) : Symbol(cfg.direct_solver)
    direct == (cfg.device == "gpu" ? :cudss : :cholesky) || throw(ArgumentError("direct solver is incompatible with device"))
    cfg = merge(cfg, (device=Symbol(cfg.device), direct_solver=direct,
        output_dir=isempty(cfg.output_dir) ? joinpath(@__DIR__, "linear-$(cfg.device)-results") : cfg.output_dir))
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
    return (; n, m, c, w)
end

function run_method(method, f, cfg)
    stop = z -> meets_target(z, cfg.tol)
    method == :pr && return run_pr(f; stop, maxiter=cfg.maxiter_pr)
    is_ipm(method) || throw(ArgumentError("Unknown method $method"))
    n, m = size(f.c)
    μ₀ = sum(f.w) / (n * (m + 1))
    σ_min = cfg.tol * minimum(f.w) / n
    return linear_perturbed_ipm(f; μ₀, σ₀=μ₀, σ_min,
        tol=cfg.tol, stop, maxiter=cfg.maxiter_newton, show_trace=false,
        linear_solver=method == :pcg ? :pcg_ic : cfg.direct_solver,
        linear_rtol=1e-9, ic_drop=0.0, ic_normalize=true)
end

function device_market(f, device)
    device == :cpu && return f
    isdefined(@__MODULE__, :CUDA) || error("Start with test-gpu.jl or --device gpu to load the GPU environment")
    CUDA.functional() || error("A functional NVIDIA CUDA device is required")
    (; n=f.n, m=f.m, c=CUDA.CuArray(f.c), w=CUDA.CuArray(f.w))
end

function measured_run(method, f, cfg, repetition, actual_density)
    GC.gc()
    linear_sync(f.c)
    timed = @timed begin
        result = run_method(method, f, cfg)
        linear_sync(f.c)
        result
    end
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
        stop_rule="shared_accuracy", device=string(cfg.device),
        gpu_name=cfg.device == :gpu ? CUDA.name(CUDA.device()) : missing,
        transfer_seconds=cfg.transfer_seconds,
        end_to_end_seconds=timed.time + cfg.transfer_seconds,
        updates=result.traj[end].k, factorizations=result.factorizations,
        matrix_vector_products=matrix_vector_products,
        linear_solver=is_ipm(method) ? (method == :pcg ? "pcg_ic" : string(cfg.direct_solver)) : missing,
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
    vary_m = length(cfg.sizes_n) == 1 && length(cfg.sizes_m) > 1
    axis_key, fixed_key = vary_m ? (:m, :n) : (:n, :m)
    fixed_sizes = vary_m ? cfg.sizes_n : cfg.sizes_m
    axis_label = vary_m ? "Buyers (m)" : "Goods (n)"
    for fixed_size in fixed_sizes, density in cfg.densities
        case_rows = filter(r -> getproperty(r, fixed_key) == fixed_size && r.density == density, rows)
        case_title = "$fixed_key = $fixed_size, density = $density"
        timing = plot(xlabel=axis_label, ylabel="Median wall time (s)", yscale=:log10,
            title="$(uppercase(string(cfg.device))) time ($case_title)", legend=:bottomright)
        updates = plot(xlabel=axis_label, ylabel="Outer updates",
            title="Iterations ($case_title)", legend=false)
        for (method, color) in zip(METHODS, (:steelblue, :seagreen, :purple))
            rs = sort(filter(r -> r.method == method_label(method), case_rows); by=r -> getproperty(r, axis_key))
            isempty(rs) && continue
            sizes = [getproperty(r, axis_key) for r in rs]
            for (panel, values) in ((timing, [r.median_seconds for r in rs]),
                                    (updates, [r.updates for r in rs]))
                # Only successful runs form the scaling curve. A cross shows a
                # stopped run that did not reach the shared target.
                good = [r.all_targets_met for r in rs]
                plot!(panel, sizes, [ok ? v : NaN for (ok, v) in zip(good, values)];
                    label=method_label(method), color, marker=:circle, lw=2)
                if any(.!good)
                    scatter!(panel, sizes[.!good], values[.!good]; color, marker=:x,
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
    panel_rows = length(fixed_sizes) * length(cfg.densities)
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
        old_label = row.method == method_label(:pr) ? "PR (shared target)" : row.method
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
    println("Julia ", VERSION, " | ", uppercase(string(cfg.device)), " | Julia threads = ", Threads.nthreads(),
        " | BLAS threads = ", BLAS.get_num_threads())
    println("n = ", cfg.sizes_n, " | m = ", cfg.sizes_m, " | tol = ", cfg.tol,
        " | densities = ", cfg.densities, " | repeats = ", cfg.repeats)
    println("Methods: ", join(method_label.(cfg.methods), ", "))
    println("PR, Perturbed, and PCG use the common clearing, budget, and utility target.")
    println("PCG uses IC(0), scalar-normalized inverse, zero initial direction, linear rtol = 1e-9; setup time is included.")
    println("target_met always scores the same shared target for every selected method.")
    println("Budget reporting compares total spending with total budget.\n")
    flush(stdout)

    # Warm up the selected case (including GPU sparse solvers) before timing.
    warmup_cfg = merge(cfg, (density=first(cfg.densities),))
    warmup = comparison_market(minimum(cfg.sizes_n), minimum(cfg.sizes_m);
        seed=cfg.seed, density=warmup_cfg.density)
    warmup = device_market(warmup, cfg.device)
    for method in cfg.methods
        result = run_method(method, warmup, warmup_cfg)
        linear_sync(warmup.c)
        println("Warm-up ", method_label(method), ": ", result.status)
        flush(stdout)
    end
    warmup = nothing
    runs, summary = NamedTuple[], NamedTuple[]
    for m in cfg.sizes_m, n in cfg.sizes_n, density in cfg.densities
        case_cfg = merge(cfg, (; density))
        market = comparison_market(n, m; seed=cfg.seed, density)
        actual_density = count(!iszero, market.c) / length(market.c)
        transfer_seconds = @elapsed begin
            market = device_market(market, cfg.device)
            linear_sync(market.c)
        end
        case_cfg = merge(case_cfg, (; transfer_seconds))
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
    LinearComparison.main()
end
