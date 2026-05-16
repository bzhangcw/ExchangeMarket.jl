# Compare CG / Multicut / FW / SFW on a real market (CES or PLC).
# Separate training and test revealed-preference sets of the same size K,
# track train primal objective and test error per iteration, then plot.
#
# CLI: see `julia test_real.jl -h` for all options.

using Revise
using Random, SparseArrays, LinearAlgebra
using DelimitedFiles
using ArgParse
using JuMP, MosekTools
using Plots, LaTeXStrings, Printf
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
        prog="test_real.jl",
        description="Benchmark CG / Multicut / FW / SFW on a CES or PLC market.",
        autofix_names=true,
    )
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
        "--timelimit", "-T"
        help = "Wall-clock cap per method, in seconds"
        arg_type = Float64
        default = 60.0
        "--methods"
        help = "Comma-separated method names to run (any of CG,MultiCut,FW,SFW,FWjl)"
        arg_type = String
        default = "CG,MultiCut,FW,SFW,FWjl"
        "--classes"
        help = "Comma-separated function classes for CG dispatch (any of ces,linear,leontief,ql)"
        arg_type = String
        default = "ces,linear,leontief"
        "--csv"
        help = "If non-empty, append per-method rows to this CSV (path; relative to revealed/)"
        arg_type = String
        default = ""
        "--no-plot"
        help = "Skip plotting (still writes the CSV)"
        action = :store_true
        "--verbosity", "-v"
        help = "0 = silent; 1 = per-iteration table; 2 = + per-pricing detail"
        arg_type = Int
        default = 0
        range_tester = x -> x in (0, 1, 2)
        "--pdf-dir"
        help = "Directory for PDF output (per-panel and combined)"
        arg_type = String
        default = @__DIR__
        "--tex-dir"
        help = "Directory for pgfplots .tex output; empty to skip"
        arg_type = String
        default = @__DIR__
        "--plc-L"
        help = "Number of PLC pieces (PLC market only)"
        arg_type = Int
        default = 5
        "--plc-no-intercept"
        help = "Disable PLC intercept (b=0); only meaningful for --market-type plc"
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
timelimit = cli["timelimit"]
csv_path = cli["csv"]
do_plot = !cli["no_plot"]
verbosity = cli["verbosity"]
pdf_dir = abspath(cli["pdf_dir"])
tex_dir = isempty(cli["tex_dir"]) ? "" : abspath(cli["tex_dir"])
do_plot && mkpath(pdf_dir)
do_plot && !isempty(tex_dir) && mkpath(tex_dir)

method_names = Symbol.(split(cli["methods"], ","))
allowed_classes = Symbol.(split(cli["classes"], ","))
opt_plc = (L=cli["plc_L"], intercept=!cli["plc_no_intercept"])

method_filter(name) = name in method_names

@info "configuration" market_type n m K seed timelimit methods = method_names classes = allowed_classes

# -----------------------------------------------------------------------
# Build the underlying (real) market
# -----------------------------------------------------------------------
Random.seed!(seed)
if market_type === :ces
    ρ_vec = -3.5 .+ 4.3 .* rand(m)
    f0 = FisherMarket(m, n; ρ=ρ_vec, scale=30.0, sparsity=0.99)
    linconstr = LinearConstr(1, n, ones(1, n), [1.0])
    f1 = copy(f0)
    p₀ = ones(n) ./ n
    f1.x .= ones(n, m) ./ m
    alg = HessianBar(n, m, p₀; linconstr=linconstr)
    alg.linsys = :direct
    @info "Ground-truth CES market" n m K σ_range = extrema(f1.σ)
    Ξ_train = produce_revealed_preferences(alg, f1, K; seed=seed)
    Ξ_test = produce_revealed_preferences(alg, f1, K; seed=seed + 1)
elseif market_type === :plc
    L = opt_plc.L
    plc_agents = [random_plc_agent(n, L; intercept=opt_plc.intercept) for _ in 1:m]
    w_vec = rand(m)
    w_vec ./= sum(w_vec)
    @info "Ground-truth PLC market" n m L K intercept = opt_plc.intercept
    @info "producing training set..."
    Ξ_train = produce_revealed_preferences_plc(plc_agents, w_vec, K, n; seed=seed)
    @info "producing test set..."
    Ξ_test = produce_revealed_preferences_plc(plc_agents, w_vec, K, n; seed=seed + 1)
elseif market_type === :ql
    ql_agents = [random_ql_agent(n) for _ in 1:m]
    w_vec = rand(m)
    w_vec ./= sum(w_vec)
    @info "Ground-truth QL market" n m K
    @info "producing training set..."
    Ξ_train = produce_revealed_preferences_ql(ql_agents, w_vec, K, n; seed=seed)
    @info "producing test set..."
    Ξ_test = produce_revealed_preferences_ql(ql_agents, w_vec, K, n; seed=seed + 1)
else
    error("Unknown market_type: $market_type")
end
tag = String(market_type)

# -----------------------------------------------------------------------
# Run methods in parallel via task-based concurrency.
# Each method runs in its own Julia task; results are gathered with `fetch`.
# Launch Julia with `--threads=N` (N >= number of methods) for true
# parallelism; otherwise tasks are interleaved by the scheduler.
# -----------------------------------------------------------------------
@info "task scheduler" nthreads = Threads.nthreads()
tasks = Task[]
task_names = Symbol[]
for (name, pricing_kind, kwargs) in method_kwargs
    !method_filter(name) && continue
    push!(task_names, name)
    push!(tasks, Threads.@spawn begin
        local_extra = Dict{Symbol,Any}(:timelimit => timelimit)
        if pricing_kind !== :fw
            local_extra[:classes] = allowed_classes
        end
        local_kwargs = merge(kwargs, local_extra)
        @info "spawned $name" classes = get(local_kwargs, :classes, "n/a") timelimit = timelimit
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
        @info "$name done" iters = length(hist[:primal_obj]) atoms_T = fa.m final_train = hist[:primal_obj][end] final_test = hist[:test_err][end] time_s = t_elapsed
        (name, fa, hist, t_elapsed)
    end)
end
results = [fetch(t) for t in tasks]

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
println("\n=== Summary ===")
@printf("%-12s %8s %8s %12s %12s %10s\n", "method", "iters", "atoms", "train_obj", "test_err", "time (s)")
for (name, fa, hist, t) in results
    @printf("%-12s %8d %8d %12.6f %12.6f %10.3f\n",
        get(display_name, name, String(name)),
        length(hist[:primal_obj]), fa.m,
        hist[:primal_obj][end], hist[:test_err][end], t)
end

# Optional CSV log for cross-run aggregation.
if !isempty(csv_path)
    csv_path = isabspath(csv_path) ? csv_path : joinpath(@__DIR__, csv_path)
    new_file = !isfile(csv_path)
    open(csv_path, "a") do io
        if new_file
            println(io, "market_type,n,m,K,method,iters,atoms_T,train_obj,test_err,time_s")
        end
        for (name, fa, hist, t) in results
            println(io, join((
                    String(market_type), n, m, K, String(name),
                    length(hist[:primal_obj]), fa.m,
                    hist[:primal_obj][end], hist[:test_err][end],
                    round(t; digits=3)
                ), ","))
        end
    end
    @info "appended results" csv_path
end

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
function make_figure(ylabel_str, title_str; ylims=(5e-4, 1e0), xlims=(1, 200))
    fig = generate_empty(; shape=:wide)
    plot!(fig,
        ylabel=ylabel_str,
        xlabel=L"\textrm{iteration}",
        title=title_str,
        legendbackgroundcolor=RGBA(1.0, 1.0, 1.0, 0.8),
        yticks=10.0 .^ (-10:1:4),
        yscale=:log10,
        # ylims=ylims,
        # xlims=xlims,
        xtickfont=font(18),
        ytickfont=font(18),
        xscale=:identity,
        size=(600, 500),
        left_margin=12Plots.mm,
        bottom_margin=8Plots.mm,
        legendfontsize=18,
    )
    return fig
end

if do_plot
    title_train = market_type === :ces ?
                  L"Training error ($n=%$n,~m=%$m,~K=%$K$)" :
                  L"Training error ($n=%$n,~m=%$m,~L=%$(opt_plc.L),~K=%$K$)"
    title_test = market_type === :ces ?
                 L"Test error ($n=%$n,~m=%$m,~K=%$K$)" :
                 L"Test error ($n=%$n,~m=%$m,~L=%$(opt_plc.L),~K=%$K$)"

    ylims_common = (5e-4, 1e0)
    xlims_common = (1, 200)
    fig_train = make_figure(
        # L"\textrm{value}", 
        # title_train;
        L"",
        L"";
        ylims=ylims_common,
        xlims=xlims_common
    )
    fig_test = make_figure(
        # L"\textrm{value}", 
        # title_test;
        L"",
        L"";
        ylims=ylims_common,
        xlims=xlims_common
    )

    for (name, fa, hist, _) in results
        iters = 1:length(hist[:primal_obj])
        c = get(colors, name, 1)
        mk = get(marker_style, name, :circle)
        m_end = hist[:num_agents][end]
        pretty = get(display_name, name, String(name))
        lbl = L"($T=%$m_end$)~\texttt{%$pretty}"
        # Two-call pattern: pgfplotsx ignores `markevery`, so draw the
        # line first (full resolution) and then drop markers at every
        # 5th index via a scatter overlay sharing the legend entry.
        train_ser = max.(hist[:primal_obj], 1e-12)
        test_ser  = max.(hist[:test_err],   1e-12)
        mk_idx = 1:5:length(iters)
        plot!(fig_train, iters, train_ser;
            label=lbl, linewidth=3, color=c)
        scatter!(fig_train, collect(mk_idx), train_ser[mk_idx];
            label="", color=c, markershape=mk, markersize=6, markerstrokewidth=0)
        plot!(fig_test, iters, test_ser;
            label=lbl, linewidth=3, color=c)
        scatter!(fig_test, collect(mk_idx), test_ser[mk_idx];
            label="", color=c, markershape=mk, markersize=6, markerstrokewidth=0)
    end

    savefig(fig_train, joinpath(pdf_dir, "real_$(tag)_train.pdf"))
    savefig(fig_test, joinpath(pdf_dir, "real_$(tag)_test.pdf"))
    @info "saved PDFs" pdf_dir

    # Emit pgfplots .tex via the pgfplotsx backend (activated by switch_to_pdf).
    if !isempty(tex_dir)
        savefig(fig_train, joinpath(tex_dir, "real_$(tag)_train.tex"))
        savefig(fig_test, joinpath(tex_dir, "real_$(tag)_test.tex"))
        @info "saved pgfplots .tex" tex_dir
    end
end
