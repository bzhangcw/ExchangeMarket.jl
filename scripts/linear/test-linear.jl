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
using Random, SparseArrays, LinearAlgebra, Statistics, Printf, SHA, TOML, Dates
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
meets_target(z, tol) = linear_meets_target(z, tol)

function configuration(args)
    # Preserve the requested underscore spellings of --sizes_n and --sizes_m.
    settings = ArgParseSettings(autofix_names=false,
        description="Compare standalone PR, perturbed IPM, and IC-preconditioned PCG; select CPU or GPU.")
    @add_arg_table! settings begin
        "--device"
            help = "Array backend: cpu or gpu (GPU uses CUDA)"
            default = get(ENV, "LINEAR_DEVICE", "cpu")
        "--storage"
            help = "Utility/allocation storage: sparse (default) or dense"
            default = "sparse"
        "--time-limit"
            dest_name = "time_limit"
            help = "Maximum timed runtime per method in seconds; enforced by a supervisor"
            arg_type = Float64
            default = 300.0
        "--verbose"
            help = "Stream worker logs, including IPM iterations"
            action = :store_true
        "--resume"
            help = "Continue this exact configuration from completed jobs"
            action = :store_true
        "--worker"
            help = "Internal benchmark worker"
            action = :store_true
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
            default = 40_000
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
    cfg.storage in ("dense","sparse") || throw(ArgumentError("storage must be dense or sparse"))
    isfinite(cfg.time_limit) && cfg.time_limit>0 || throw(ArgumentError("time-limit must be positive and finite"))
    cfg.device in ("cpu", "gpu") || throw(ArgumentError("device must be cpu or gpu"))
    cfg.direct_solver in ("auto", "cholesky", "cudss") || throw(ArgumentError("unknown direct solver"))
    direct = cfg.direct_solver == "auto" ? (cfg.device == "gpu" ? :cudss : :cholesky) : Symbol(cfg.direct_solver)
    direct == (cfg.device == "gpu" ? :cudss : :cholesky) || throw(ArgumentError("direct solver is incompatible with device"))
    cfg = merge(cfg, (device=Symbol(cfg.device), storage=Symbol(cfg.storage), direct_solver=direct,
        output_dir=isempty(cfg.output_dir) ? joinpath(@__DIR__, "linear-$(cfg.device)-$(cfg.storage)-results") : cfg.output_dir))
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
    rng=MersenneTwister(seed)
    c=sprand(rng,n,m,density)
    for i in 1:m
        isempty(nzrange(c,i)) && (c[rand(rng,1:n),i]=0.5+rand(rng))
    end
    occupied=falses(n)
    for row in rowvals(c);occupied[row]=true;end
    for j in 1:n
        !occupied[j] && (c[j,rand(rng,1:m)]=0.5+rand(rng))
    end
    w=0.5 .+ rand(rng,m);w./=sum(w)
    (;n,m,c,w)
end
function market_digest(f)
    io=IOBuffer()
    write(io,Int64(f.n),Int64(f.m),Int64.(f.c.colptr),Int64.(f.c.rowval),f.c.nzval,f.w)
    bytes2hex(sha256(take!(io)))
end
function run_method(method,f,cfg;progress=nothing)
    run_linear(method,f.c,f.w;tol=cfg.tol,maxiter_pr=cfg.maxiter_pr,
        maxiter_ipm=cfg.maxiter_newton,time_limit=cfg.time_limit,
        direct_solver=cfg.direct_solver,show_trace=cfg.verbose,progress)
end
function device_market(f,device,storage)
    prepared=linear_prepare_market(f.c,f.w;device,storage)
    (;n=f.n,m=f.m,prepared...)
end

function measured_run(method, f, cfg, repetition, actual_density; progress=nothing)
    linear_sync(f.c)
    timed = @timed begin
        result = run_method(method, f, cfg; progress)
        linear_sync(f.c)
        result
    end
    result = timed.value
    z = linear_equilibrium_metrics(f.c, f.w, result.st.p, result.st.x)
    target_met = meets_target(z, cfg.tol) && timed.time <= cfg.time_limit
    status = timed.time > cfg.time_limit ? :time_limit : result.status
    @assert all(isfinite, (z.mc, z.budget, z.regret, z.total_spending))
    @assert status != :converged || target_met
    @assert isapprox(z.total_spending, dot(result.st.p, linear_demand(result.st.x)); rtol=1e-12)
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
        repetition=repetition, status=string(status), target_met=target_met,
        stop_rule="shared_accuracy", device=string(cfg.device), storage=string(cfg.storage),
        time_limit=cfg.time_limit, warmup_seconds=cfg.warmup_seconds,
        market_sha256=cfg.market_sha256, message="",
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
        case_title = "$fixed_key = $fixed_size, density = $density, $(cfg.storage)"
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
        old_storage=hasproperty(ref,:storage) ? ref.storage : "dense"
        old_storage==row.storage || error("Reference storage differs")
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

# Each job is checkpointed; a persistent worker avoids repeated compilation.
# The supervisor restarts it after a timeout/error and skips completed jobs.
atomic_csv(path,row)=begin
    CSV.write(path*".tmp",[row]);mv(path*".tmp",path;force=true)
end
function atomic_toml(path,data)
    open(path*".tmp","w") do io;TOML.print(io,data);end
    mv(path*".tmp",path;force=true)
end
read_row(path)=only(Tables.rowtable(CSV.File(path;stringtype=String)))
const Tables=CSV.Tables
function jobs(cfg)
    [(;n,m,density,repetition,method,
        path=joinpath(cfg.output_dir,"jobs","n-$(n)-m-$(m)-density-$(density)-$(method)-$(repetition)"))
        for m in cfg.sizes_m for n in cfg.sizes_n for density in cfg.densities
        for repetition in 1:cfg.repeats for method in circshift(cfg.methods,repetition-1)]
end
function pending_row(job,cfg,f,digest,actual_density,transfer_seconds)
    (;n=f.n,m=f.m,tol=cfg.tol,seed=cfg.seed,density=job.density,actual_density,
      julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),
      maxiter_pr=cfg.maxiter_pr,maxiter_newton=cfg.maxiter_newton,method=method_label(job.method),
      repetition=job.repetition,status="pending",target_met=false,stop_rule="shared_accuracy",
      device=string(cfg.device),storage=string(cfg.storage),time_limit=cfg.time_limit,
      warmup_seconds=NaN,market_sha256=digest,message="",
      gpu_name=cfg.device==:gpu ? CUDA.name(CUDA.device()) : missing,
      transfer_seconds,end_to_end_seconds=NaN,updates=0,factorizations=0,matrix_vector_products=job.method==:perturbed ? "-" : "0",
      linear_solver=job.method==:pr ? missing : job.method==:pcg ? "pcg_ic" : string(cfg.direct_solver),
      linear_rtol=job.method==:pr ? missing : 1e-9,average_linear_iterations=job.method==:pr ? missing : 0.0,
      preconditioner_factorizations=0,linear_setup_seconds=0.0,
      ic_drop=job.method==:pcg ? 0.0 : missing,ic_normalized=job.method==:pcg ? true : missing,
      wall_seconds=0.0,gc_seconds=0.0,allocated_MiB=0.0,mc=Inf,regret=Inf,error_ratio=Inf,
      total_spending=NaN,total_budget=sum(f.w),total_budget_error=Inf,
      sigma=job.method==:pr ? missing : NaN,mu=job.method==:pr ? missing : NaN,
      sigma_min=cfg.tol*minimum(f.w)/f.n)
end
function phase!(cfg,job,phase,limit)
    atomic_toml(joinpath(cfg.output_dir,"current.toml"),
        Dict("job"=>job.path,"phase"=>phase,"started"=>time(),"limit"=>limit))
end
mutable struct BenchmarkProgress{B}
    base::B
    path::String
    last_saved::Float64
end
function (monitor::BenchmarkProgress)(z)
    if z.t-monitor.last_saved>=1.0 || z.k==0
        row=merge(monitor.base,(updates=z.k,wall_seconds=z.t,mc=z.mc,regret=z.regret,
            error_ratio=z.error/monitor.base.tol,total_spending=z.total_spending,
            total_budget_error=z.total_budget_error,
            factorizations=hasproperty(z,:factorizations) ? z.factorizations : 0,
            sigma=hasproperty(z,:σ) ? z.σ : missing,mu=hasproperty(z,:μ) ? z.μ : missing))
        atomic_csv(monitor.path*".progress.csv",row);monitor.last_saved=z.t
    end
    nothing
end
function worker_main(cfg)
    BLAS.set_num_threads(cfg.blas_threads)
    previous=nothing;market=nothing;digest="";actual_density=0.0;transfer_seconds=0.0
    for job in jobs(cfg)
        isfile(job.path*".csv") && continue
        key=(job.n,job.m,job.density)
        if key!=previous
            raw=comparison_market(job.n,job.m;seed=cfg.seed,density=job.density)
            digest=market_digest(raw);actual_density=nnz(raw.c)/length(raw.c)
            transfer_seconds=@elapsed begin
                market=device_market(raw,cfg.device,cfg.storage);linear_sync(market.c)
            end
            previous=key
        end
        base=pending_row(job,cfg,market,digest,actual_density,transfer_seconds)
        atomic_csv(job.path*".progress.csv",base)
        try
            println("Running ",job.method," n=",job.n," m=",job.m," density=",job.density);flush(stdout)
            phase!(cfg,job,"warmup",300.0)
            # Two untimed updates at the actual shape; never warm up to convergence.
            casecfg=merge(cfg,(density=job.density,transfer_seconds,warmup_seconds=0.0,market_sha256=digest))
            warmcfg=merge(casecfg,(maxiter_pr=2,maxiter_newton=2,verbose=false,time_limit=300.0))
            monitor=BenchmarkProgress(base,job.path,-Inf)
            warmup_seconds=@elapsed begin
                # Warm the exact timed wrapper, callback, diagnostics, and row construction.
                measured_run(job.method,market,warmcfg,job.repetition,actual_density;progress=monitor)
            end
            base=merge(base,(;warmup_seconds));atomic_csv(job.path*".progress.csv",base)
            casecfg=merge(casecfg,(;warmup_seconds))
            monitor.base=base;monitor.last_saved=-Inf
            GC.gc();linear_sync(market.c)
            phase!(cfg,job,"solve",cfg.time_limit)
            row=measured_run(job.method,market,casecfg,job.repetition,actual_density;progress=monitor)
            atomic_csv(job.path*".csv",row)
        catch err
            showerror(stderr,err,catch_backtrace());println(stderr);flush(stderr)
            row=merge(read_row(job.path*".progress.csv"),(status="error",target_met=false,message=sprint(showerror,err)))
            atomic_csv(job.path*".csv",row)
            # A failed native GPU call may invalidate the CUDA context.
            cfg.device==:gpu && return
        end
    end
end
function command_args(cfg)
    args=String["--device",string(cfg.device),"--storage",string(cfg.storage),"--direct-solver",string(cfg.direct_solver),
        "--tol",string(cfg.tol),"--time-limit",string(cfg.time_limit),"--seed",string(cfg.seed),
        "--repeats",string(cfg.repeats),"--blas-threads",string(cfg.blas_threads),
        "--maxiter-pr",string(cfg.maxiter_pr),"--maxiter-newton",string(cfg.maxiter_newton),"--output-dir",cfg.output_dir]
    for (flag,values) in (("--sizes_n",cfg.sizes_n),("--sizes_m",cfg.sizes_m),("--densities",cfg.densities),("--methods",cfg.methods))
        push!(args,flag);append!(args,string.(values))
    end
    cfg.verbose && push!(args,"--verbose")
    isempty(cfg.reference_summary) || append!(args,["--reference-summary",cfg.reference_summary])
    args
end
function stop_worker(process)
    process_running(process) || return
    kill(process,Base.SIGTERM)
    deadline=time()+1.0
    while process_running(process) && time()<deadline;sleep(0.05);end
    process_running(process) && kill(process,Base.SIGKILL)
    wait(process)
end
shellquote(s)="'"*replace(string(s),"'"=>"'\\''")*"'"
function save_metadata(cfg)
    signature=Dict(string(k)=>v isa Symbol ? string(v) : v isa Vector{Symbol} ? string.(v) : v
        for (k,v) in pairs(cfg) if !(k in (:resume,:worker,:verbose)))
    signature["julia_version"]=string(VERSION)
    source=IOBuffer()
    for name in ("test-linear.jl","linear-backend.jl","linear-perturbed.jl","linear-perturbed-linsys.jl",
                 "propres.jl","linear-sparse.jl","linear-gpu.jl","linear-gpu-sparse.jl")
        write(source,read(joinpath(@__DIR__,name)))
    end
    signature["source_sha256"]=bytes2hex(sha256(take!(source)))
    configpath=joinpath(cfg.output_dir,"config.toml")
    if isfile(configpath)
        cfg.resume || error("results directory already initialized; use --resume or a new --output-dir")
        TOML.parsefile(configpath)==signature || error("resume settings differ from saved run")
    else
        atomic_toml(configpath,signature)
    end
    open(joinpath(cfg.output_dir,"cmd.sh"),"w") do io
        println(io,"#!/usr/bin/env bash\nset -euo pipefail")
        println(io,"cd ",shellquote(normpath(joinpath(@__DIR__,"../.."))))
        println(io,"LINEAR_PROJECT=",shellquote(dirname(Base.active_project()))," julia --startup-file=no --threads=",Threads.nthreads()," ",
            join(shellquote.(vcat([@__FILE__],command_args(cfg)))," "))
    end
    open(joinpath(cfg.output_dir,"environment.txt"),"w") do io
        println(io,"UTC: ",Dates.now(Dates.UTC),"\nJulia: ",VERSION,"\nHost: ",gethostname(),"\nStorage: ",cfg.storage)
        println(io,"CPU: ",Sys.CPU_NAME,"; Julia threads: ",Threads.nthreads(),"; BLAS threads: ",cfg.blas_threads)
        try println(io,"Commit: ",readchomp(`git -C $(@__DIR__) rev-parse HEAD`)) catch;end
        Pkg.status(;io)
    end
end
function main(args=ARGS)
    cfg=configuration(args)
    cfg.worker && return worker_main(cfg)
    BLAS.set_num_threads(cfg.blas_threads)
    mkpath(joinpath(cfg.output_dir,"jobs"));save_metadata(cfg)
    alljobs=jobs(cfg);runs=NamedTuple[]
    current=joinpath(cfg.output_dir,"current.toml")
    logfile=joinpath(cfg.output_dir,"run.log");logoffset=isfile(logfile) ? filesize(logfile) : 0
    command=`$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) $(@__FILE__) $(command_args(cfg)) --worker`
    println("Julia ",VERSION," | ",cfg.device," | ",cfg.storage," | tol=",cfg.tol,
        " | PR maxiter=",cfg.maxiter_pr," | runtime limit=",cfg.time_limit,"s per method")
    flush(stdout)
    index=1
    while index<=length(alljobs)
        if isfile(alljobs[index].path*".csv")
            row=read_row(alljobs[index].path*".csv");push!(runs,row);index+=1
            @printf("n=%d m=%d density=%.3f %-10s %8.3fs %-20s target=%s spent=%.8f / %.8f\n",
                row.n,row.m,row.density,row.method,row.wall_seconds,row.status,string(row.target_met),row.total_spending,row.total_budget)
            flush(stdout)
            CSV.write(joinpath(cfg.output_dir,"runs.csv"),DataFrame(runs))
            continue
        end
        rm(current;force=true)
        launched=time()
        open(logfile,"a") do log
            process=run(pipeline(addenv(command,"LINEAR_PROJECT"=>dirname(Base.active_project()));stdout=log,stderr=log);wait=false)
            try
                while process_running(process)
                    if cfg.verbose && isfile(logfile)
                        open(logfile) do io;seek(io,logoffset);print(stdout,read(io,String));logoffset=position(io);end
                        flush(stdout)
                    end
                    while index<=length(alljobs) && isfile(alljobs[index].path*".csv")
                        row=read_row(alljobs[index].path*".csv");push!(runs,row);index+=1
                        @printf("n=%d m=%d density=%.3f %-10s %8.3fs %-20s target=%s spent=%.8f / %.8f\n",
                            row.n,row.m,row.density,row.method,row.wall_seconds,row.status,string(row.target_met),row.total_spending,row.total_budget)
                        flush(stdout);CSV.write(joinpath(cfg.output_dir,"runs.csv"),DataFrame(runs))
                    end
                    if isfile(current)
                        state=TOML.parsefile(current);jobpath=state["job"]
                        if !isfile(jobpath*".csv") && time()-state["started"]>=state["limit"]
                            elapsed=time()-state["started"]
                            stop_worker(process)
                            # Keep a completed result if it arrived while termination was requested.
                            if !isfile(jobpath*".csv")
                                row=read_row(jobpath*".progress.csv")
                                status=state["phase"]=="solve" ? "time_limit" : "warmup_time_limit"
                                row=merge(row,(;status,target_met=false,wall_seconds=state["phase"]=="solve" ? elapsed : 0.0,
                                    end_to_end_seconds=state["phase"]=="solve" ? elapsed+row.transfer_seconds : NaN,
                                    message="Supervisor stopped "*state["phase"]*" at its runtime limit"))
                                atomic_csv(jobpath*".csv",row)
                            end
                            break
                        end
                    elseif time()-launched>600
                        stop_worker(process);error("worker startup exceeded 600 seconds; see $logfile")
                    end
                    sleep(0.1)
                end
            finally
                process_running(process) && stop_worker(process)
            end
            wait(process)
            if index<=length(alljobs) && !isfile(alljobs[index].path*".csv")
                if isfile(current)
                    state=TOML.parsefile(current);jobpath=state["job"]
                    if !isfile(jobpath*".csv")
                        row=merge(read_row(jobpath*".progress.csv"),(status="error",target_met=false,message="Worker exited; see run.log"))
                        atomic_csv(jobpath*".csv",row)
                    end
                end
                isfile(alljobs[index].path*".csv") || error("worker exited before initializing a job; see $logfile")
            end
        end
    end
    rm(current;force=true)
    summary=NamedTuple[]
    for m in cfg.sizes_m,n in cfg.sizes_n,density in cfg.densities,method in cfg.methods
        push!(summary,summarize(filter(r->r.n==n && r.m==m && r.density==density && r.method==method_label(method),runs)))
    end
    CSV.write(joinpath(cfg.output_dir,"summary.csv"),DataFrame(summary))
    compare_reference(summary,cfg);save_plot(summary,cfg)
    println("Saved ",length(runs)," runs to ",cfg.output_dir)
    (;runs,summary)
end
end # module
if abspath(PROGRAM_FILE)==@__FILE__
    LinearComparison.main()
end
