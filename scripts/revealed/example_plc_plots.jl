# Generate the two-commodity PLC illustration figures for arxiv/numeric.tex.
# For each PLC type (homogeneous, non-homogeneous), fit two CG surrogates:
#   - boundary classes only: allowed_classes = [:linear, :leontief]
#   - full class:            allowed_classes = [:ces, :linear, :leontief]
# Plot solid ground truth + two dashed surrogates per panel.
#
# Outputs (controlled by CLI flags):
#   --pdf-dir       PDF panels and the combined panel
#   --tex-dir       pgfplots .tex panels (for \input{} in numeric.tex)
#
# CLI: see `julia example_plc_plots.jl -h` for all options. The Julia
# project must be the scripts project that owns ExchangeMarket.jl, e.g.
#   julia --project=$(dirname $(dirname $(realpath example_plc_plots.jl))) ...

using ArgParse

function parse_args_for_example_plc(argv=ARGS)
    s = ArgParseSettings(
        prog="example_plc_plots.jl",
        description="Generate the two-commodity PLC illustration panels.",
        autofix_names=true,
    )
    @add_arg_table! s begin
        "--pdf-dir"
        help = "Directory for PDF output (per-panel and combined)"
        arg_type = String
        default = @__DIR__
        "--tex-dir"
        help = "Directory for pgfplots .tex output (e.g., overleaf/arxiv/figs)"
        arg_type = String
        default = ""
    end
    return parse_args(argv, s)
end

const cli = parse_args_for_example_plc()
const PDF_DIR = abspath(cli["pdf_dir"])
const TEX_DIR = isempty(cli["tex_dir"]) ? "" : abspath(cli["tex_dir"])
mkpath(PDF_DIR)
isempty(TEX_DIR) || mkpath(TEX_DIR)

using Revise
using Random, SparseArrays, LinearAlgebra
using JuMP, MosekTools
using Plots, LaTeXStrings, Printf
import MathOptInterface as MOI

using ExchangeMarket

# Sibling/parent script files: tools.jl and plots.jl live in scripts/
# (one level above this file); plc.jl and setup.jl are sibling to this file.
const _HERE = @__DIR__
include(joinpath(_HERE, "..", "tools.jl"))
include(joinpath(_HERE, "..", "plots.jl"))
include(joinpath(_HERE, "plc.jl"))
include(joinpath(_HERE, "setup.jl"))

switch_to_pdf(; bool_use_html=false)

# -----------------------------------------------------------------------
# Shared geometry
# -----------------------------------------------------------------------
const n = 2
const m = 5
const L = 4
const seed = 42

const p1_train = collect(0.02:0.02:0.98)
const p1_test  = collect(range(0.03, 0.97, length=100))
const p1_sweep = collect(range(0.02, 0.98, length=200))

function make_market(intercept::Bool; seed=seed)
    Random.seed!(seed)
    agents = [random_plc_agent(n, L; intercept=intercept) for _ in 1:m]
    w_vec = rand(m); w_vec ./= sum(w_vec)
    return agents, w_vec
end

function ground_truth_demand(agents, w_vec, p)
    g = zeros(n)
    for i in 1:m
        x_i, _ = solve_plc_demand(agents[i], p, w_vec[i])
        g .+= x_i
    end
    return g
end

function make_grid(agents, w_vec, p1_grid)
    Ξ = Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, length(p1_grid))
    for (k, p1) in enumerate(p1_grid)
        p = [p1, 1.0 - p1]
        Ξ[k] = (p, ground_truth_demand(agents, w_vec, p))
    end
    return Ξ
end

# -----------------------------------------------------------------------
# Run CG for a chosen allowed_classes config; return sweep of g_truth, g_ces
# -----------------------------------------------------------------------
function fit_and_sweep(agents, w_vec, allowed_classes::Vector{Symbol})
    Ξ_train = make_grid(agents, w_vec, p1_train)
    Ξ_test  = make_grid(agents, w_vec, p1_test)
    name, pricing_kind, raw_kwargs = method_kwargs[1]   # :CG, :cg_single
    kwargs = merge(Dict(raw_kwargs), Dict(:classes => allowed_classes))
    Random.seed!(seed)
    fa, _, _ = run_method_tracked(name, pricing_kind, kwargs, Ξ_train, Ξ_test; verbosity=1)

    g_truth = zeros(n, length(p1_sweep))
    g_ces   = zeros(n, length(p1_sweep))
    for (k, p1) in enumerate(p1_sweep)
        p = [p1, 1.0 - p1]
        g_truth[:, k] = p .* ground_truth_demand(agents, w_vec, p)
        fitted = zeros(n)
        for i in 1:fa.m
            c_i = Vector(fa.c[:, i])
            fitted .+= fa.w[i] .* compute_gamma(p, c_i, fa.σ[i])
        end
        g_ces[:, k] = fitted
    end
    return (; fa, g_truth, g_ces)
end

# -----------------------------------------------------------------------
# Plot: solid ground truth + two dashed surrogates
# -----------------------------------------------------------------------
function make_panel(res_linear, res_boundary, res_full, title_str)
    fig = generate_empty(; shape=:wide, settick=false)
    plot!(fig,
        ylabel=L"\hat g_1(p)",
        xlabel=L"p_1",
        title=title_str,
        legendbackgroundcolor=RGBA(1.0, 1.0, 1.0, 0.8),
        xtickfont=font(18),
        ytickfont=font(18),
        legendfontsize=14,
        xscale=:identity,
        yscale=:identity,
        size=(600, 500),
        legend=:topright,
    )
    # Solid ground truth (shared between configs).
    plot!(fig, p1_sweep, res_linear.g_truth[1, :];
        label=L"\textrm{ground~truth}", linewidth=3, color=1)
    # Dotted: linear-only surrogate (H(1)).
    plot!(fig, p1_sweep, res_linear.g_ces[1, :];
        label=L"\texttt{CG}:~\textrm{linear~only}",
        linewidth=3, linestyle=:dot, color=4)
    # Dashed: boundary-only surrogate (linear + Leontief).
    plot!(fig, p1_sweep, res_boundary.g_ces[1, :];
        label=L"\texttt{CG}:~\textrm{linear}+\textrm{Leontief}",
        linewidth=3, linestyle=:dash, color=2)
    # Dash-dot: interior CES class only (H(R^\circ)).
    plot!(fig, p1_sweep, res_full.g_ces[1, :];
        label=L"\texttt{CG}:~\textrm{interior~CES}",
        linewidth=3, linestyle=:dashdot, color=3)
    return fig
end

# -----------------------------------------------------------------------
# Two PLC types × two configs
# -----------------------------------------------------------------------
agents_homo, w_homo = make_market(false)
agents_nh,   w_nh   = make_market(true)

@info "homogeneous PLC, linear only"
res_homo_l = fit_and_sweep(agents_homo, w_homo, [:linear])
@info "homogeneous PLC, boundary classes only"
res_homo_b = fit_and_sweep(agents_homo, w_homo, [:linear, :leontief])
@info "homogeneous PLC, interior CES only"
res_homo_f = fit_and_sweep(agents_homo, w_homo, [:ces])
@info "non-homogeneous PLC, linear only"
res_nh_l   = fit_and_sweep(agents_nh,   w_nh,   [:linear])
@info "non-homogeneous PLC, boundary classes only"
res_nh_b   = fit_and_sweep(agents_nh,   w_nh,   [:linear, :leontief])
@info "non-homogeneous PLC, interior CES only"
res_nh_f   = fit_and_sweep(agents_nh,   w_nh,   [:ces])

title_homo = L"\textrm{Homogeneous~PLC}~(n=%$n,~m=%$m,~L=%$L)"
title_nh   = L"\textrm{Non\textendash homogeneous~PLC}~(n=%$n,~m=%$m,~L=%$L)"
fig_homo   = make_panel(res_homo_l, res_homo_b, res_homo_f, title_homo)
fig_nh     = make_panel(res_nh_l,   res_nh_b,   res_nh_f,   title_nh)
fig_combo  = plot(fig_homo, fig_nh; layout=(1, 2), size=(1500, 500))

savefig(fig_homo,  joinpath(PDF_DIR, "example_plc_homo.pdf"))
savefig(fig_nh,    joinpath(PDF_DIR, "example_plc_nonhomo.pdf"))
savefig(fig_combo, joinpath(PDF_DIR, "example_plc.pdf"))
@info "saved PDFs" PDF_DIR

# Emit pgfplots .tex via the pgfplotsx backend (activated by switch_to_pdf).
if !isempty(TEX_DIR)
    savefig(fig_homo, joinpath(TEX_DIR, "example_plc_homo.tex"))
    savefig(fig_nh,   joinpath(TEX_DIR, "example_plc_nonhomo.tex"))
    @info "saved pgfplots .tex" TEX_DIR
end

@info "homo linear T=$(res_homo_l.fa.m)  boundary T=$(res_homo_b.fa.m)  full T=$(res_homo_f.fa.m)"
@info "nh   linear T=$(res_nh_l.fa.m)    boundary T=$(res_nh_b.fa.m)    full T=$(res_nh_f.fa.m)"
