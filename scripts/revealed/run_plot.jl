# Plotting helper for revealed-preference benchmarks.
#
# Two roles:
#  (1) library — `run_test.jl` includes this file and calls
#      `plot_risk_curves(ctx; pdf_dir, tex_dir, ...)` directly with the
#      aggregation context it just built;
#  (2) standalone script — `julia run_plot.jl -f <datafile> [...]`
#      reads a serialized context dumped by an earlier `run_test.jl`
#      run and reproduces the figures without re-running the benchmark.
#
# The context object is a NamedTuple holding only the slim per-method
# summaries the plotter needs (per-iter mean / std / min / max of train,
# test, and excess curves, plus the `finals` for legend annotations);
# heavy state like `runs` is dropped before serialization.

using Plots, LaTeXStrings, Printf
using Serialization, ArgParse

# `setup.jl` exposes `colors`, `marker_style`, `display_name`; the plot
# library leans on those tables but doesn't redefine them.
isdefined(Main, :colors) || include(joinpath(@__DIR__, "setup.jl"))
isdefined(Main, :switch_to_pdf) || include(joinpath(@__DIR__, "..", "plots.jl"))

# Force the PGFPlotsX backend at top level. The backend switch must
# happen outside any function — calling `pgfplotsx()` from inside a
# function leaves savefig writing empty `.tex` / `.pdf` files.
switch_to_pdf(; bool_use_html=false)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
# NaN-safe centered moving average. `window <= 1` returns the input
# unchanged. Useful for de-noising per-iteration curves before plotting;
# NaN entries (e.g., excess slots where the equilibrium wasn't evaluated
# yet) are ignored — both as inputs and as outputs (output stays NaN if
# every entry in the window is NaN).
function _moving_average(v::AbstractVector{<:Real}, window::Integer)
    window <= 1 && return collect(v)
    L = length(v)
    out = similar(v, Float64)
    half = window ÷ 2
    @inbounds for i in 1:L
        lo = max(1, i - half)
        hi = min(L, i + half)
        s = 0.0
        n = 0
        for j in lo:hi
            x = v[j]
            isnan(x) && continue
            s += x
            n += 1
        end
        out[i] = n == 0 ? NaN : s / n
    end
    return out
end

function _make_figure(ylabel_str, title_str; ylims=(5e-4, 1e0), xlims=(1, 200))
    fig = generate_empty(; shape=:wide)
    plot!(fig,
        ylabel=ylabel_str,
        xlabel=L"\textrm{iteration}",
        title=title_str,
        legendbackgroundcolor=RGBA(1.0, 1.0, 1.0, 0.8),
        yticks=10.0 .^ (-8:1:4),
        yscale=:log10,
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

# -----------------------------------------------------------------------
# Slim per-method summary used by the plotter. `runs` and any heavy
# objects in the original `agg` entries are dropped, so the serialized
# context is portable and stable across Julia sessions.
# -----------------------------------------------------------------------
function build_plot_ctx(;
    agg::Dict,
    rep::Int,
    market_type::Symbol,
    n::Int, m::Int, K::Int,
    opt_plc,
    method_names::Vector{Symbol},
    interval_marker::Int,
)
    slim = Dict{Symbol,NamedTuple}()
    for (name, a) in agg
        slim[name] = (
            Lmax=a.Lmax,
            train_mean=a.train_mean, train_std=a.train_std,
            train_min=a.train_min, train_max=a.train_max,
            test_mean=a.test_mean, test_std=a.test_std,
            test_min=a.test_min, test_max=a.test_max,
            excess_mean=a.excess_mean, excess_std=a.excess_std,
            excess_min=a.excess_min, excess_max=a.excess_max,
            has_excess=a.has_excess,
            finals=a.finals,
        )
    end
    return (
        agg=slim,
        rep=rep,
        market_type=market_type,
        n=n, m=m, K=K,
        opt_plc=opt_plc,
        method_names=method_names,
        interval_marker=interval_marker,
    )
end

function save_plot_ctx(path::AbstractString, ctx)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        serialize(io, ctx)
    end
    @info "saved plot context" path
    return path
end

function load_plot_ctx(path::AbstractString)
    open(path, "r") do io
        deserialize(io)
    end
end

# -----------------------------------------------------------------------
# Main plotter. Renders three figures (train / test / excess) per call,
# writes PDFs to `pdf_dir`, optionally also pgfplots `.tex` to `tex_dir`.
# Returns the figure tuple for callers that want to compose further.
# -----------------------------------------------------------------------
function plot_risk_curves(ctx;
    pdf_dir::AbstractString,
    tex_dir::AbstractString="",
    ylims_common::Tuple=(5e-4, 1e0),
    xlims_common::Tuple=(1, 200),
    smooth::Real=1.0,
    interval_marker::Union{Nothing,Int}=nothing,
)
    rep = ctx.rep
    market_type = ctx.market_type
    method_names = ctx.method_names
    interval_marker = something(interval_marker, ctx.interval_marker)
    agg = ctx.agg

    mkpath(pdf_dir)
    !isempty(tex_dir) && mkpath(tex_dir)

    fig_train = _make_figure(L"", L""; ylims=ylims_common, xlims=xlims_common)
    fig_test = _make_figure(L"", L""; ylims=ylims_common, xlims=xlims_common)
    fig_excess = _make_figure(L"", L""; ylims=ylims_common, xlims=xlims_common)
    any_excess = any(name -> haskey(agg, name) && agg[name].has_excess, method_names)

    # Smoothing: window size for the centered NaN-safe moving average.
    # `smooth == 1.0` is identity (no smoothing); fractional values round.
    window = max(1, round(Int, smooth))
    sm(v) = _moving_average(v, window)

    for name in method_names
        haskey(agg, name) || continue
        a = agg[name]
        iters = 1:a.Lmax
        c = get(colors, name, 1)
        mk = get(marker_style, name, :circle)
        m_end = round(Int, a.finals.nag_mean)
        pretty = get(display_name, name, String(name))
        lbl = L"($T=%$m_end$)~\texttt{%$pretty}"
        mk_idx = 1:max(interval_marker, 1):length(iters)

        # Pre-smooth all curves; ribbon bounds get smoothed too so the
        # band stays consistent with the mean.
        tr_mean = sm(a.train_mean)
        te_mean = sm(a.test_mean)
        ex_mean = a.has_excess ? sm(a.excess_mean) : a.excess_mean
        tr_std = sm(a.train_std)
        te_std = sm(a.test_std)
        ex_std = a.has_excess ? sm(a.excess_std) : a.excess_std
        tr_mn = sm(a.train_min)
        tr_mx = sm(a.train_max)
        te_mn = sm(a.test_min)
        te_mx = sm(a.test_max)
        ex_mn = a.has_excess ? sm(a.excess_min) : a.excess_min
        ex_mx = a.has_excess ? sm(a.excess_max) : a.excess_max

        if rep > 1
            tr_lo = max.(max.(tr_mean .- tr_std, tr_mn), 1e-8)
            tr_hi = min.(tr_mean .+ tr_std, tr_mx)
            te_lo = max.(max.(te_mean .- te_std, te_mn), 1e-8)
            te_hi = min.(te_mean .+ te_std, te_mx)
            plot!(fig_train, iters, tr_mean;
                ribbon=(tr_mean .- tr_lo, tr_hi .- tr_mean),
                fillalpha=0.2, label=lbl, linewidth=3, color=c)
            plot!(fig_test, iters, te_mean;
                ribbon=(te_mean .- te_lo, te_hi .- te_mean),
                fillalpha=0.2, label=lbl, linewidth=3, color=c)
            if a.has_excess
                ex_lo = max.(max.(ex_mean .- ex_std, ex_mn), 1e-8)
                ex_hi = min.(ex_mean .+ ex_std, ex_mx)
                plot!(fig_excess, iters, ex_mean;
                    ribbon=(ex_mean .- ex_lo, ex_hi .- ex_mean),
                    fillalpha=0.2, label=lbl, linewidth=3, color=c)
            end
        else
            plot!(fig_train, iters, tr_mean;
                label=lbl, linewidth=3, color=c)
            scatter!(fig_train, collect(mk_idx), tr_mean[mk_idx];
                label="", color=c, markershape=mk, markersize=3, markerstrokewidth=0)
            plot!(fig_test, iters, te_mean;
                label=lbl, linewidth=3, color=c)
            scatter!(fig_test, collect(mk_idx), te_mean[mk_idx];
                label="", color=c, markershape=mk, markersize=3, markerstrokewidth=0)
            if a.has_excess
                plot!(fig_excess, iters, ex_mean;
                    label=lbl, linewidth=3, color=c)
                mk_valid = [i for i in mk_idx if !isnan(ex_mean[i])]
                if !isempty(mk_valid)
                    scatter!(fig_excess, collect(mk_valid), ex_mean[mk_valid];
                        label="", color=c, markershape=mk, markersize=3, markerstrokewidth=0)
                end
            end
        end
    end

    tag = String(market_type)
    savefig(fig_train, joinpath(pdf_dir, "real_$(tag)_train.pdf"))
    savefig(fig_test, joinpath(pdf_dir, "real_$(tag)_test.pdf"))
    any_excess && savefig(fig_excess, joinpath(pdf_dir, "real_$(tag)_excess.pdf"))
    @info "saved PDFs" pdf_dir excess = any_excess

    if !isempty(tex_dir)
        savefig(fig_train, joinpath(tex_dir, "real_$(tag)_train.tex"))
        savefig(fig_test, joinpath(tex_dir, "real_$(tag)_test.tex"))
        any_excess && savefig(fig_excess, joinpath(tex_dir, "real_$(tag)_excess.tex"))
        @info "saved pgfplots .tex" tex_dir
    end
    return (train=fig_train, test=fig_test, excess=fig_excess)
end

# -----------------------------------------------------------------------
# CLI: `julia run_plot.jl -f <ctx.jls> [--pdf-dir ...] [--tex-dir ...]`
# -----------------------------------------------------------------------
function _main_test_plot(argv)
    s = ArgParseSettings(prog="run_plot.jl",
        description="Replay risk-curve plots from a saved benchmark context.")
    @add_arg_table! s begin
        "--file", "-f"
        help = "Path to the context file produced by run_test.jl (--data-file)"
        arg_type = String
        required = true
        "--pdf-dir"
        help = "Directory for PDF output (default: same dir as the context file)"
        arg_type = String
        default = ""
        "--tex-dir"
        help = "Directory for pgfplots .tex output (default: same as --pdf-dir; pass empty string \"\" to skip)"
        arg_type = String
        default = ""
        "--no-tex"
        help = "Skip writing pgfplots .tex output."
        action = :store_true
        "--smooth"
        help = "Centered moving-average window (factor). 1.0 ⇒ no smoothing; larger ⇒ smoother."
        arg_type = Float64
        default = 1.0
        "--interval-marker"
        help = "Place a plot marker every N iterations (single-rep mode only). Overrides the value stored in the ctx file."
        arg_type = Int
        default = -1
    end
    cli = parse_args(argv, s)
    ctx_path = cli["file"]
    ctx = load_plot_ctx(ctx_path)
    pdf_dir = isempty(cli["pdf-dir"]) ? dirname(abspath(ctx_path)) : abspath(cli["pdf-dir"])
    tex_dir = if cli["no-tex"]
        ""
    elseif isempty(cli["tex-dir"])
        pdf_dir
    else
        abspath(cli["tex-dir"])
    end
    im_arg = cli["interval-marker"]
    plot_risk_curves(ctx;
        pdf_dir=pdf_dir, tex_dir=tex_dir, smooth=cli["smooth"],
        interval_marker=(im_arg > 0 ? im_arg : nothing))
end

if abspath(PROGRAM_FILE) == @__FILE__
    _main_test_plot(ARGS)
end
