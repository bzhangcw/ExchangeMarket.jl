#!/usr/bin/env julia
# JD Table 3 scales, raw click utilities, budgets summing to one per market.
module JDComparison
using ArgParse, CSV, SparseArrays, LinearAlgebra, Random, SHA, TOML, Dates
const Tables=CSV.Tables
include(joinpath(@__DIR__, "../../dataset/jd-data.jl"))

function options(args)
    settings=ArgParseSettings(description="JD GPU comparison: PR, perturbed IPM/cuDSS, IPM/PCG-IC(0)")
    @add_arg_table! settings begin
        "--data"
            default=normpath(joinpath(@__DIR__,"../../dataset/jd/utilities.bin"))
        "--outdir"
            default=joinpath(@__DIR__,"jd_gpu_results")
        "--users"
            help="Comma-separated prefixes; all means every identified user"
            default="2000,5000,20000,50000,200000,500000,1000000,all"
        "--methods"
            default="pr,perturbed,pcg"
        "--tol"
            arg_type=Float64
            default=1e-4
        "--criterion"
            range_tester=x->x in ("paper","shared")
            default="shared"
        "--time-limit"
            arg_type=Float64
            default=7200.0
        "--maxiter-pr"
            arg_type=Int
            default=1000000
        "--maxiter-ipm"
            arg_type=Int
            default=100000
        "--linear-rtol"
            arg_type=Float64
            default=1e-9
        "--linear-maxiter"
            help="0 uses max(20,5*goods)"
            arg_type=Int
            default=0
        "--repeats"
            arg_type=Int
            default=1
        "--resume"
            action=:store_true
        "--worker"
            help="Internal: isolate a single method in a fresh CUDA process"
            action=:store_true
        "--job"
            default=""
        "--job-repeat"
            help="Internal repetition index"
            arg_type=Int
            default=1
    end
    cfg=parse_args(args,settings)
    0<cfg["tol"]<1 || error("tol must be in (0,1)")
    cfg["time-limit"]>0 && cfg["repeats"]>0 || error("invalid run limits")
    cfg["maxiter-pr"]>=0 && cfg["maxiter-ipm"]>=0 || error("negative update limit")
    0<cfg["linear-rtol"]<1 && cfg["linear-maxiter"]>=0 || error("invalid linear solve limits")
    methods=split(cfg["methods"],',')
    all(x->x in ("pr","perturbed","pcg"),methods) || error("unknown method")
    length(unique(methods))==length(methods) || error("duplicate methods")
    cfg["data"]=abspath(cfg["data"]);cfg["outdir"]=abspath(cfg["outdir"])
    cfg
end

function atomic_toml(path,values)
    open(path*".tmp","w") do io;TOML.print(io,values);end
    mv(path*".tmp",path;force=true)
end
function write_rows(path,rows)
    CSV.write(path*".tmp",rows)
    mv(path*".tmp",path;force=true)
end
function base_row(f,method,repeat,cfg,digest)
    idx=f.m==f.full_users ? 8 : findfirst(==(f.m),JDData.PAPER_USERS)
    expected(a)=idx===nothing ? 0 : a[idx]
    (;users=f.m,goods=f.n,nnz=nnz(f.c),paper_users=expected(JDData.PAPER_USERS),
      paper_goods=expected(JDData.PAPER_GOODS),paper_nnz=expected(JDData.PAPER_NNZ),
      method=method=="pr" ? "PR" : method=="perturbed" ? "Perturbed" : "PCG",repeat,
      status="pending",target_met=false,criterion=cfg["criterion"],tol=cfg["tol"],
      wall_seconds=NaN,transfer_seconds=NaN,warmup_seconds=NaN,updates=0,
      paper_error=NaN,paper_primal=NaN,paper_dual=NaN,paper_gap=NaN,
      mc=NaN,regret=NaN,shared_error=NaN,total_spending=NaN,total_budget=sum(f.w),
      total_budget_error=NaN,sigma=NaN,mu=NaN,sigma_min=cfg["tol"]*minimum(f.w)/f.n,
      linear_iterations=0,average_linear_iterations=0.0,linear_solves=0,
      factorizations=0,preconditioner_factorizations=0,preconditioner_seconds=0.0,
      oracle_seconds=0.0,hessian_seconds=0.0,solve_seconds=0.0,allocated_MiB=NaN,
      time_limit=cfg["time-limit"],maxiter_pr=cfg["maxiter-pr"],maxiter_ipm=cfg["maxiter-ipm"],
      linear_rtol=cfg["linear-rtol"],linear_maxiter=cfg["linear-maxiter"],
      budget_seed=f.seed,utility_normalization="none",budget_normalization="sum_one",
      matrix_sha256=digest,gpu="",julia_version=string(VERSION),message="")
end
function metric_fields(z)
    (;paper_error=z.paper_error,paper_primal=z.paper_primal,paper_dual=z.paper_dual,
      paper_gap=z.paper_gap,mc=z.mc,regret=z.regret,shared_error=z.error,
      total_spending=z.total_spending,total_budget=z.total_budget,total_budget_error=z.total_budget_error)
end
stopping_error(z,cfg)=cfg["criterion"]=="paper" ? z.paper_error : z.error
target(z,cfg)=stopping_error(z,cfg)<=cfg["tol"]*(1+1e-8)
mutable struct JDStop
    cfg::Dict{String,Any}
    job::String
    started::Float64
    last_saved::Float64
    calls::Int
    last_metrics::Any
    warming::Bool
    gpu::String
end
function (monitor::JDStop)(z)
    monitor.warming && return false
    elapsed=time()-monitor.started;monitor.calls+=1;monitor.last_metrics=z
    if elapsed-monitor.last_saved>=30 || target(z,monitor.cfg) || elapsed>=monitor.cfg["time-limit"]
        atomic_toml(monitor.job*".progress",Dict(string(k)=>v for (k,v) in pairs(
            merge(metric_fields(z),(wall_seconds=elapsed,updates=monitor.calls-1,gpu=monitor.gpu)))))
        println("updates=",monitor.calls-1," elapsed=",round(elapsed;digits=2),
                " ",monitor.cfg["criterion"],"_error=",stopping_error(z,monitor.cfg)," total_spending=",z.total_spending)
        flush(stdout);monitor.last_saved=elapsed
    end
    target(z,monitor.cfg) || elapsed>=monitor.cfg["time-limit"]
end
function solve(f,method,cfg,stop)
    method=="pr" && return run_pr(f.c,f.w;stop,maxiter=cfg["maxiter-pr"])
    linear_perturbed_ipm(f;tol=cfg["tol"],stop,σ_min=cfg["tol"]*minimum(f.w)/size(f.c,1),
        linear_solver=method=="perturbed" ? :cudss : :pcg_ic,show_trace=false,
        maxiter=cfg["maxiter-ipm"],linear_rtol=cfg["linear-rtol"],
        linear_maxiter=cfg["linear-maxiter"]==0 ? max(20,5size(f.c,1)) : cfg["linear-maxiter"])
end
function warmup(f,method,cfg)
    small=merge(cfg,Dict("maxiter-pr"=>2,"maxiter-ipm"=>2))
    result=solve(f,method,small,JDStop(small,"",time(),0.0,0,nothing,true,""))
    # Exercise timed diagnostics too, including their first-use compilation.
    mktempdir() do directory
        forced=merge(cfg,Dict("time-limit"=>0.0))
        monitor=JDStop(forced,joinpath(directory,"warmup"),time(),0.0,0,nothing,false,CUDA.name(CUDA.device()))
        monitor(result.traj[end])
    end
    CUDA.synchronize();nothing
end
function worker(cfg)
    job=cfg["job"];method=cfg["methods"]
    f=JDData.load_market(cfg["data"];users=cfg["users"]=="all" ? nothing : parse(Int,cfg["users"]))
    configpath=joinpath(cfg["outdir"],"config.toml")
    digest=isfile(configpath) ? TOML.parsefile(configpath)["matrix_sha256"] : open(sha256,cfg["data"]) |> bytes2hex
    row=base_row(f,method,cfg["job-repeat"],cfg,digest)
    try
        # Keep the supervisor free of CUDA contexts and permit CPU-only --help.
        @eval begin
            include(joinpath(@__DIR__,"linear-perturbed.jl"))
            include(joinpath(@__DIR__,"propres.jl"))
            include(joinpath(@__DIR__,"linear-gpu.jl"))
            include(joinpath(@__DIR__,"linear-jd-gpu.jl"))
        end
        # Includes add methods in a newer world than this function's compilation.
        return Base.invokelatest(worker_gpu,cfg,f,row)
    catch err
        showerror(stderr,err,catch_backtrace());println(stderr)
        row=merge(row,(status="error",message=sprint(showerror,err)))
        write_rows(job*".csv",[row]);return
    end
end
function worker_gpu(cfg,f,row)
    CUDA.functional() || error("CUDA device required")
    BLAS.set_num_threads(1)
    method=cfg["methods"];job=cfg["job"]
    row=merge(row,(gpu=CUDA.name(CUDA.device()),))
    transferred=@timed begin
        device=(c=jd_gpu_array(f.c),w=CuArray(f.w));CUDA.synchronize();device
    end
    row=merge(row,(transfer_seconds=transferred.time,))
    # CUDA reductions specialize on problem shape; a tiny market alone leaves
    # compilation in the first timed iteration. Use two untimed actual-size
    # updates, then initialize a fresh solver state for the measured run.
    warmup_seconds=@elapsed warmup(transferred.value,method,cfg)
    GC.gc();CUDA.reclaim()
    row=merge(row,(warmup_seconds=warmup_seconds,))
    started=time()
    monitor=JDStop(cfg,job,started,0.0,0,nothing,false,row.gpu)
    atomic_toml(job*".started",Dict("time"=>started))
    try
        measurement=@timed begin
            result=solve(transferred.value,method,cfg,monitor);CUDA.synchronize();result
        end
        result=measurement.value;z=result.traj[end]
        met=target(z,cfg) && measurement.time<=cfg["time-limit"]
        status=measurement.time>cfg["time-limit"] ? "time_limit" : met ? "converged" : string(result.status)
        row=merge(row,metric_fields(z),(status=status,target_met=met,updates=z.k,
            wall_seconds=measurement.time,allocated_MiB=measurement.bytes/2.0^20))
        if method!="pr"
            row=merge(row,(sigma=result.st.σ,mu=result.st.μ,
                linear_iterations=sum(s.iterations for s in result.linear_stats;init=0),
                average_linear_iterations=result.average_linear_iterations,linear_solves=result.linear_solves,
                factorizations=result.factorizations,preconditioner_factorizations=result.preconditioner_factorizations,
                preconditioner_seconds=result.t_preconditioner,oracle_seconds=result.t_oracle,
                hessian_seconds=result.t_build,solve_seconds=result.t_solve))
        end
        CSV.write(job*".trajectory.csv",result.traj)
    catch err
        showerror(stderr,err,catch_backtrace());println(stderr)
        row=merge(row,(status="error",wall_seconds=time()-started,message=sprint(showerror,err)))
        monitor.last_metrics===nothing || (row=merge(row,metric_fields(monitor.last_metrics),(updates=monitor.calls-1,)))
    end
    write_rows(job*".csv",[row])
end

shellquote(s)="'"*replace(string(s),"'"=>"'\\''")*"'"
function run_case(cfg,users,method,repeat,digest,job)
    f=JDData.load_market(cfg["data"];users=users=="all" ? nothing : parse(Int,users))
    row=base_row(f,method,repeat,cfg,digest);f=nothing;GC.gc()
    child=String["--worker","--job",job,"--job-repeat",string(repeat),"--users",users,"--methods",method]
    for key in ("data","outdir","tol","criterion","time-limit","maxiter-pr","maxiter-ipm","linear-rtol","linear-maxiter")
        append!(child,["--"*key,string(cfg[key])])
    end
    project=dirname(Base.active_project())
    command=`$(Base.julia_cmd()) --startup-file=no --project=$project $(@__FILE__) $child`
    launch=time();failure=""
    open(job*".log","w") do log
        process=run(pipeline(addenv(command,"JULIA_NUM_THREADS"=>"1","OPENBLAS_NUM_THREADS"=>"1");stdout=log,stderr=log);wait=false)
        while process_running(process)
            if isfile(job*".started")
                start=TOML.parsefile(job*".started")["time"]
                time()-start>cfg["time-limit"]+5 && (failure="time_limit")
            elseif time()-launch>600
                failure="startup_time_limit"
            end
            if !isempty(failure)
                kill(process);deadline=time()+5
                while process_running(process) && time()<deadline;sleep(.1);end
                process_running(process) && kill(process,Base.SIGKILL)
                break
            end
            sleep(.25)
        end
        wait(process)
        isempty(failure) && !success(process) && (failure="process_error")
    end
    if isfile(job*".csv")
        result=only(Tables.rowtable(CSV.File(job*".csv";stringtype=String)))
        return merge(row,result,(repeat=repeat,matrix_sha256=digest))
    end
    if isfile(job*".progress")
        progress=TOML.parsefile(job*".progress")
        row=merge(row,(; (Symbol(k)=>v for (k,v) in progress)...))
    end
    row=merge(row,(status=isempty(failure) ? "missing_result" : failure,
        target_met=false,message="See "*basename(job)*".log; metrics, if present, are the last saved checkpoint"))
    isfile(job*".started") && (row=merge(row,(wall_seconds=time()-TOML.parsefile(job*".started")["time"],)))
    write_rows(job*".csv",[row]);row
end
function main(args=ARGS)
    cfg=options(args)
    cfg["worker"] && return worker(cfg)
    digest=open(sha256,cfg["data"]) |> bytes2hex
    out=cfg["outdir"];mkpath(out);mkpath(joinpath(out,"jobs"))
    signature=filter(p->!(p.first in ("resume","worker","job","job-repeat")),cfg)
    signature["matrix_sha256"]=digest
    configpath=joinpath(out,"config.toml")
    if isfile(configpath)
        cfg["resume"] || error("results directory already initialized; use --resume or a new --outdir")
        TOML.parsefile(configpath)==signature || error("resume settings/source differ from saved run")
    else
        atomic_toml(configpath,signature)
    end
    open(joinpath(out,"cmd.sh"),"w") do io
        println(io,"#!/usr/bin/env bash\nset -euo pipefail")
        println(io,"cd ",shellquote(normpath(joinpath(@__DIR__,"../.."))))
        println(io,join(shellquote.(vcat(Base.julia_cmd().exec,["--startup-file=no","--project="*dirname(Base.active_project()),@__FILE__],args))," "))
    end
    open(joinpath(out,"environment.txt"),"w") do io
        println(io,"UTC: ",Dates.now(Dates.UTC),"\nJulia: ",VERSION,"\nHost: ",gethostname())
        try println(io,"Commit: ",readchomp(`git -C $(@__DIR__) rev-parse HEAD`)) catch;end
        println(io,"Utilities: raw repeated click counts; selected budgets sum to 1; unit supplies.")
    end
    rows=NamedTuple[]
    for users in split(cfg["users"],','),repeat in 1:cfg["repeats"],method in split(cfg["methods"],',')
        job=joinpath(out,"jobs","users_$(users)_$(method)_$(repeat)")
        println("Running users=",users," method=",method," repeat=",repeat);flush(stdout)
        if cfg["resume"] && isfile(job*".csv")
            row=only(Tables.rowtable(CSV.File(job*".csv";stringtype=String)))
            row=merge(row,(repeat=repeat,matrix_sha256=digest))
        else
            # Only our incomplete per-job markers are replaced on resume.
            for suffix in (".started",".progress");rm(job*suffix;force=true);end
            row=run_case(cfg,users,method,repeat,digest,job)
        end
        push!(rows,row);write_rows(joinpath(out,"runs.csv"),rows)
        println(row.method," ",row.status," time=",row.wall_seconds,"s, ",row.criterion,"_error=",
            row.criterion=="paper" ? row.paper_error : row.shared_error);flush(stdout)
    end
    println("Saved ",length(rows)," method runs to ",out)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    JDComparison.main()
end
