#!/usr/bin/env julia
# Regenerate synthetic comparison reports from saved measurements; no solves.
module LinearPlots
using Pkg
Pkg.activate(get(ENV, "LINEAR_PROJECT", normpath(joinpath(@__DIR__, ".."))))
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using ArgParse, CSV, DataFrames, Statistics, Printf, Plots
include(joinpath(@__DIR__, "linear-labels.jl"))

const METHODS = ("PR", "Factorization", "PCG")
const COLORS = Dict("PR" => "#2878B5", "Factorization" => "#16836B", "PCG" => "#B1518C")
casekey(r) = (r.sweep, r.n, r.m, r.density, r.repetition)
pairkey(r) = (casekey(r)..., r.method)
good(r) = r.target_met && r.status == "converged"
safe_mean(xs) = isempty(xs) ? NaN : mean(xs)
safe_median(xs) = isempty(xs) ? NaN : median(xs)
safe_max(xs) = isempty(xs) ? NaN : maximum(xs)

function configuration(args)
    s = ArgParseSettings(description="Plot saved synthetic CPU/GPU runs without rerunning benchmarks.")
    @add_arg_table! s begin
        "--cpu-dir"
            default = joinpath(@__DIR__, "linear-cpu-sparse-results")
        "--gpu-dir"
            default = joinpath(@__DIR__, "linear-gpu-sparse-results")
        "--output-dir"
            default = ""
    end
    a = parse_args(args, s)
    cpu, gpu = abspath(a["cpu-dir"]), abspath(a["gpu-dir"])
    output = isempty(a["output-dir"]) ? joinpath(gpu, "comparison") : abspath(a["output-dir"])
    (; cpu, gpu, output)
end

function validate_pairs(cpu, gpu)
    a = Dict(pairkey(r) => r for r in eachrow(cpu))
    b = Dict(pairkey(r) => r for r in eachrow(gpu))
    length(a) == nrow(cpu) && length(b) == nrow(gpu) || error("Duplicate run identifiers")
    keys(a) == keys(b) || error("CPU/GPU cases differ")
    for k in keys(a)
        for field in (:market_sha256, :tol, :storage, :seed, :time_limit, :maxiter_pr, :maxiter_newton)
            a[k][field] == b[k][field] || error("CPU/GPU mismatch: $field in $k")
        end
    end
    a, b
end

function export_statistics(data, a, b, output)
    stats, winners, paired = NamedTuple[], NamedTuple[], NamedTuple[]
    for device in ("cpu", "gpu")
        df = data[device]
        fastest = Dict(m => 0 for m in METHODS)
        for group in groupby(df, [:sweep, :n, :m, :density, :repetition])
            eligible = filter(good, group)
            isempty(eligible) && continue
            fastest[eligible.method[argmin(eligible.wall_seconds)]] += 1
        end
        for method in METHODS
            push!(winners, (; device, method=linear_method_label(method, device), fastest_converged_cases=fastest[method],
                cases=length(groupby(df, [:sweep, :n, :m, :density, :repetition]))))
        end
        for scope in ("all", "vary-n", "vary-m"), method in METHODS
            rows = filter(r -> r.method == method && (scope == "all" || r.sweep == scope), df)
            isempty(rows) && continue
            solved = filter(good, rows)
            push!(stats, (; device, scope, method=linear_method_label(method, device), runs=nrow(rows), converged=nrow(solved),
                success_rate=nrow(solved)/nrow(rows),
                time_limits=count(==("time_limit"), rows.status),
                iteration_limits=count(==("max_iter"), rows.status),
                other_failures=count(r -> !good(r) && !(r.status in ("time_limit", "max_iter")), eachrow(rows)),
                mean_seconds_all=mean(rows.wall_seconds), median_seconds_all=median(rows.wall_seconds),
                mean_seconds_converged=safe_mean(solved.wall_seconds),
                median_seconds_converged=safe_median(solved.wall_seconds),
                median_updates_converged=safe_median(solved.updates),
                min_updates_converged=isempty(solved) ? NaN : minimum(solved.updates),
                max_updates_converged=safe_max(solved.updates),
                max_market_clearing_error_converged=safe_max(solved.mc),
                max_regret_converged=safe_max(solved.regret),
                max_total_budget_error_converged=safe_max(solved.total_budget_error),
                max_error_ratio_converged=safe_max(solved.error_ratio)))
        end
    end
    for method in METHODS
        kk = [k for k in keys(a) if a[k].method == method && good(a[k]) && good(b[k])]
        ct, gt = [a[k].wall_seconds for k in kk], [b[k].wall_seconds for k in kk]
        ratios = ct ./ gt
        push!(paired, (; method, cpu_method=linear_method_label(method, :cpu), gpu_method=linear_method_label(method, :gpu), pairs=length(kk), cpu_mean_seconds=safe_mean(ct),
            gpu_mean_seconds=safe_mean(gt), ratio_of_mean_times=safe_mean(ct)/safe_mean(gt),
            geometric_mean_speedup=isempty(ratios) ? NaN : exp(mean(log.(ratios))),
            median_speedup=safe_median(ratios), gpu_faster_cases=count(>(1), ratios),
            min_speedup=isempty(ratios) ? NaN : minimum(ratios), max_speedup=safe_max(ratios)))
    end
    CSV.write(joinpath(output, "statistics.csv"), DataFrame(stats))
    CSV.write(joinpath(output, "fastest-counts.csv"), DataFrame(winners))
    CSV.write(joinpath(output, "paired-speedups.csv"), DataFrame(paired))
    failures = vcat([filter(r -> !good(r), data[d]) for d in ("cpu", "gpu")]...)
    failures.method = [linear_method_label(r.method, r.device) for r in eachrow(failures)]
    CSV.write(joinpath(output, "failures.csv"), failures)
    println(DataFrame(stats)[:, [:device, :scope, :method, :converged, :runs, :median_updates_converged]])
    println(DataFrame(paired))
end

function save_figure(fig, output, name)
    for suffix in ("png", "pdf")
        savefig(fig, joinpath(output, "$name.$suffix"))
    end
end

function scaling(data, output, sweep, quantity)
    variable = sweep == "vary-n" ? :n : :m
    allrows = vcat([filter(r -> r.sweep == sweep, data[d]) for d in ("cpu", "gpu")]...)
    fixed = only(unique(allrows[!, variable == :n ? :m : :n]))
    xlabel = variable == :n ? "Goods (n); buyers m = $fixed" : "Buyers (m); goods n = $fixed"
    tol = only(unique(allrows.tol))
    storage = only(unique(allrows.storage))
    densities = sort(unique(allrows.density))
    times = quantity == :wall_seconds
    values = Float64.(allrows[!, quantity])
    finite_values = filter(x -> isfinite(x) && x > 0, values)
    lower = 10.0^floor(log10(minimum(finite_values)))
    upper = times ? max(maximum(allrows.time_limit)*1.35, maximum(finite_values)*1.4) : maximum(finite_values)*2
    panels = []
    for (i, density) in enumerate(densities), (j, device) in enumerate(("cpu", "gpu"))
        rows = filter(r -> r.sweep == sweep && r.density == density, data[device])
        sizes = sort(unique(rows[!, variable]))
        padding = (maximum(sizes)/minimum(sizes))^0.04
        ticklabels = [x >= 1000 ? @sprintf("%gk", x/1000) : string(x) for x in sizes]
        p = plot(; xlabel=i == length(densities) ? xlabel : "",
            ylabel=times ? "Solve time (s)" : "Outer updates",
            title="$(uppercase(device)) | density = $(round(Int,100density))%",
            yscale=:log10, xscale=:log10, ylims=(lower, upper),
            xlims=(minimum(sizes)/padding, maximum(sizes)*padding),
            xticks=(sizes, ticklabels), legend=i == 1 ? :outertop : false, legend_columns=2,
            gridalpha=0.15, foreground_color_legend=:transparent,
            background_color_legend=:white, legendfontsize=9)
        cap = only(unique(rows.time_limit))
        times && hline!(p, [cap]; color=:gray, linestyle=:dot, label="$(round(Int,cap)) s cap", linewidth=1)
        for method in METHODS
            rr = sort(filter(r -> r.method == method, rows), variable)
            ok = good.(eachrow(rr))
            yy = Float64.(rr[!, quantity])
            plot!(p, rr[!, variable], ifelse.(ok, yy, NaN); label=linear_method_label(method, device),
                color=COLORS[method], marker=:circle, markersize=4, linewidth=2)
            bad = .!ok .& isfinite.(yy) .& (yy .> 0)
            any(bad) && scatter!(p, rr[bad, variable], yy[bad]; label=false,
                color=COLORS[method], marker=:xcross, markersize=7, markerstrokewidth=2)
        end
        push!(panels, p)
    end
    title = times ? "CPU / GPU runtime | $storage | tol = $tol | x = target missed" :
        "CPU / GPU outer updates | $storage | tol = $tol | x = target missed"
    fig = plot(panels...; layout=(length(densities), 2), size=(1250, 1100),
        plot_title=title, margin=6Plots.mm, bottom_margin=8Plots.mm, dpi=180)
    save_figure(fig, output, "$(times ? "runtime" : "iterations")-$sweep")
end

function performance_profiles(data, output)
    profiles = Dict{String,Dict{String,Vector{Float64}}}()
    for device in ("cpu", "gpu")
        ratios = Dict(m => Float64[] for m in METHODS)
        for group in groupby(data[device], [:sweep, :n, :m, :density, :repetition])
            solved = filter(good, group)
            best = isempty(solved) ? Inf : minimum(solved.wall_seconds)
            for method in METHODS
                rows = filter(r -> r.method == method, group)
                length(unique(rows.repetition)) == 1 && nrow(rows) == 1 || error("Expected one row per case and method")
                r = first(eachrow(rows))
                push!(ratios[method], good(r) ? r.wall_seconds/best : Inf)
            end
        end
        profiles[device] = ratios
    end
    finite_ratios = [r for ds in values(profiles) for rs in values(ds) for r in rs if isfinite(r)]
    xmax = 2.0^ceil(log2(maximum(finite_ratios)))
    panels = []
    for device in ("cpu", "gpu")
        p = plot(; xlabel="Time / fastest converged method on the same case",
            ylabel="Cases solved within this time factor (%)", title=uppercase(device),
            xscale=:log2, xlims=(1, xmax), ylims=(0, 104), yticks=0:20:100,
            legend=:bottomright, gridalpha=0.15)
        for method in METHODS
            rs = profiles[device][method]
            xs = sort(unique(vcat(1.0, filter(isfinite, rs), xmax)))
            ys = [100count(<=(x), rs)/length(rs) for x in xs]
            plot!(p, xs, ys; seriestype=:steppost, color=COLORS[method], linewidth=2.3, label=linear_method_label(method, device))
        end
        push!(panels, p)
    end
    cases = length(profiles["cpu"][first(METHODS)])
    fig = plot(panels...; layout=(1, 2), size=(1300, 520), margin=6Plots.mm,
        bottom_margin=10Plots.mm, dpi=180, plot_titlevspan=0.12, plot_titlefontsize=15,
        plot_title="Performance profiles | $cases sweep cases per device | failed runs remain unsolved")
    save_figure(fig, output, "performance-profile")
end

function main(args=ARGS)
    cfg = configuration(args)
    data = Dict("cpu" => CSV.read(joinpath(cfg.cpu, "runs.csv"), DataFrame),
        "gpu" => CSV.read(joinpath(cfg.gpu, "runs.csv"), DataFrame))
    for (device, rows) in data
        all(rows.device .== device) || error("Result device differs from input directory")
        rows.method = linear_method_family.(rows.method)
    end
    a, b = validate_pairs(data["cpu"], data["gpu"])
    mkpath(cfg.output)
    export_statistics(data, a, b, cfg.output)
    for sweep in ("vary-n", "vary-m"), quantity in (:wall_seconds, :updates)
        scaling(data, cfg.output, sweep, quantity)
    end
    performance_profiles(data, cfg.output)
    # Preserve the earlier v2/v3 names beside the new per-sweep output.
    for directory in (cfg.cpu, cfg.gpu), (sweep, version) in (("vary-n", "v2"), ("vary-m", "v3"))
        cp(joinpath(directory, sweep, "scaling.png"), joinpath(directory, "scaling-$version.png"); force=true)
        cp(joinpath(directory, sweep, "summary.csv"), joinpath(directory, "summary-$version.csv"); force=true)
    end
    shellquote(s) = "'" * replace(string(s), "'" => "'\\''") * "'"
    open(joinpath(cfg.output, "cmd.sh"), "w") do io
        println(io, "#!/usr/bin/env bash\nset -euo pipefail")
        println(io, "cd ", shellquote(normpath(joinpath(@__DIR__, "../.."))))
        println(io, "LINEAR_PROJECT=", shellquote(dirname(Base.active_project())),
            " julia --startup-file=no ", shellquote(@__FILE__),
            " --cpu-dir ", shellquote(cfg.cpu), " --gpu-dir ", shellquote(cfg.gpu),
            " --output-dir ", shellquote(cfg.output))
    end
    println("Saved comparison CSVs and PNG/PDF figures in ", cfg.output)
    println("Paired inputs verified: ", length(a), "; no benchmarks rerun.")
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
end
