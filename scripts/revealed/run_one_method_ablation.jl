# Ablation of the two Arrow–Debreu adcg knobs — boosting-style mini-batch
# (--sample-hard) and the free supply scale δ (--ad-delta-free) — on ONE
# dataset. Fits adcg four times and prints a side-by-side comparison:
#
#   1. default                 (uniform mini-batch, δ = 1 fixed)
#   2. sample-hard             (residual-weighted mini-batch, δ = 1 fixed)
#   3. var-δ                   (uniform mini-batch, δ free; eq.ad.master.scaled)
#   4. sample-hard + var-δ     (both)
#
# Same CLI as run_one_method.jl (`parse_args_for_test_real`); the four
# variants OVERRIDE whatever --sample-hard / --ad-delta-free the CLI sets, so
# you don't pass those yourself. The method is forced to `adcg` regardless of
# --methods. Pass --sample-size > 0 for the mini-batch knob to bite (with
# --sample-size 0 the two mini-batch variants are identical to their uniform
# siblings — the script warns).
#
# The dataset is built ONCE and the global RNG is reseeded to cfg.seed before
# each fit, so the bootstrap atom is identical across variants and the only
# differences are the two knobs.
#
# Run / plot split (mirrors run_test.jl ↔ run_plot.jl): this script does the
# fitting and serializes a run_plot.jl-compatible plot context (the four
# variants as the "method" axis, each curve = its per-iteration history). It
# does NOT plot — render the train/test figures separately with
#   julia --project=. revealed/run_plot.jl -f <ctx.jls> --xmax … --ymax …
# (the exact command, with axis limits fitted to the run, is printed at the
# end). Keeping the fit plotting-free means this step needs no Plots /
# PGFPlotsX / LaTeX stack. The ctx carries its own per-variant `style`, so
# run_plot.jl needs no entries in setup.jl's global style tables.
#
# CLI: see `julia run_one_method_ablation.jl -h` for all options.

using Revise
using Random, SparseArrays, LinearAlgebra
using DelimitedFiles
using ArgParse
using JuMP, MosekTools
using Printf
using Statistics
using Serialization
import MathOptInterface as MOI

using ExchangeMarket

include("../tools.jl")
include("./androids/plc.jl")   # ground-truth PLC generator (see run_test.jl note)
include("./setup.jl")

const cli = parse_args_for_test_real()
cfg = build_run_config(cli)

# This script is adcg-specific: force the method regardless of --methods.
const NAME = :adcg
spec = first(s for s in method_kwargs if s[1] == NAME)
(_, separation_kind, base_kwargs) = spec
if cfg.method_names != [NAME]
    @warn "run_one_method_ablation always runs adcg; ignoring --methods=$(cfg.method_names)"
end

# The four variants. Each entry sets the two CLI flags the adcg knobs read
# (`apply_cli_separation!` / `apply_cli_ad!` consume cli["sample_hard"] and
# cli["ad_delta_free"]); the rest describes the variant for the comparison
# table and the plot context:
#   - `sym`      : Symbol key in the serialized plot ctx (method-like axis).
#   - `color`/`marker`/`plotlabel` : per-series style carried by the ctx so
#     run_plot.jl renders the variants without entries in setup.jl's global
#     `colors` / `marker_style` / `display_name` tables. `plotlabel` is ASCII
#     ("var-delta", not "var-δ") because it flows into a pgfplots \texttt{…}.
const VARIANTS = [
    (label="default", sample_hard=false, delta_free=false,
        sym=:adcg_default, color=:steelblue, marker=:circle, plotlabel="default"),
    (label="sample-hard", sample_hard=true, delta_free=false,
        sym=:adcg_hard, color=:darkorange, marker=:rect, plotlabel="sample-hard"),
    (label="var-δ", sample_hard=false, delta_free=true,
        sym=:adcg_vardelta, color=:seagreen, marker=:diamond, plotlabel="var-delta"),
    (label="sample-hard + var-δ", sample_hard=true, delta_free=true,
        sym=:adcg_hard_vardelta, color=:crimson, marker=:utriangle,
        plotlabel="sample-hard + var-delta"),
]

print_tree("ablation configuration", [
    "market" => cfg.market_type,
    "dimensions" => [
        "n" => cfg.n,
        "m" => cfg.m,
        "K" => cfg.K,
        "K_test" => cfg.K_test,
    ],
    "method" => NAME,
    "classes" => cfg.allowed_classes,
    "sample_size" => cli["sample_size"],
    "variants" => [string(i) => v.label for (i, v) in enumerate(VARIANTS)],
    "seed" => cfg.seed,
    "timelimit" => @sprintf("%g s", cfg.timelimit),
])
if cli["sample_size"] <= 0
    @warn "--sample-size is 0 (full batch): the --sample-hard variants are " *
          "identical to their uniform siblings. Pass --sample-size > 0 to " *
          "exercise the boosting mini-batch."
end

# Build the dataset ONCE so all four variants fit the same (Ξ_train, Ξ_test).
rd = build_rep_data(cfg, 1, cfg.seed)

results = Vector{NamedTuple}(undef, length(VARIANTS))
for (i, v) in enumerate(VARIANTS)
    println()
    println("="^100)
    @printf("VARIANT %d/%d: %s  (sample-hard=%s, δ=%s)\n",
        i, length(VARIANTS), v.label, v.sample_hard, v.delta_free ? "free" : "1 fixed")
    println("="^100)

    # Per-variant CLI: copy the parsed flags and override the two knobs. A
    # shallow copy suffices — we only reassign top-level scalar keys.
    cli_v = copy(cli)
    cli_v["sample_hard"] = v.sample_hard
    cli_v["ad_delta_free"] = v.delta_free

    # Reseed so the bootstrap CES atom (and any sampling randomness) starts
    # identically across variants — the knobs become the only difference.
    Random.seed!(cfg.seed)
    res = run_one_method(cfg, cli_v, 1, rd.rep_seed, rd.Ξ_train, rd.Ξ_test, rd.f_real,
        NAME, separation_kind, base_kwargs)

    δ_final = haskey(res.hist, :delta) && !isempty(res.hist[:delta]) ?
              res.hist[:delta][end] : NaN
    results[i] = (
        variant=v,
        iters=length(res.hist[:primal_obj]),
        atoms=res.fa.m,
        train=res.hist[:primal_obj][end],
        test=res.hist[:test_err][end],
        delta=δ_final,
        t=res.t,
        hist=res.hist,
    )
end

# -----------------------------------------------------------------------
# Comparison table.
# -----------------------------------------------------------------------
println()
println("="^100)
println("ABLATION SUMMARY  (market=$(cfg.market_type), n=$(cfg.n), m=$(cfg.m), K=$(cfg.K), sample_size=$(cli["sample_size"]))")
println("="^100)
@printf("%-22s | %5s | %5s | %11s | %11s | %8s | %8s\n",
    "variant", "iters", "atoms", "train_obj", "test_err", "δ*", "time(s)")
@printf("%-22s-+-%5s-+-%5s-+-%11s-+-%11s-+-%8s-+-%8s\n",
    "-"^22, "-"^5, "-"^5, "-"^11, "-"^11, "-"^8, "-"^8)
for r in results
    @printf("%-22s | %5d | %5d | %11.3e | %11.3e | %8.4f | %8.2f\n",
        r.variant.label, r.iters, r.atoms, r.train, r.test, r.delta, r.t)
end

# Best train / test, for a quick read.
best_train = argmin(r -> r.train, results)
best_test = argmin(r -> r.test, results)
println()
@printf("best train_obj: %s (%.3e)\n", best_train.variant.label, best_train.train)
@printf("best test_err : %s (%.3e)\n", best_test.variant.label, best_test.test)

# -----------------------------------------------------------------------
# Serialize a run_plot.jl-compatible plot context (same shape as
# build_plot_ctx) so the heavy fitting and the figure rendering are split,
# exactly as run_test.jl ↔ run_plot.jl. The four variants become the
# "method" axis; each per-iteration curve is the raw history (rep = 1, so
# the std/min/max fields are unused by the plotter — set to the mean). The
# ctx carries its own `style` (color/marker/label per variant) so run_plot.jl
# renders them without entries in setup.jl's global style tables.
#
# Built inline (not via build_plot_ctx) to keep this run step free of the
# Plots / PGFPlotsX / LaTeX stack — only run_plot.jl needs it.
# -----------------------------------------------------------------------
function _ctx_entry(r)
    tr = Float64.(r.hist[:primal_obj])
    te = Float64.(r.hist[:test_err])
    L = length(tr)
    nan = fill(NaN, L)
    return (
        Lmax=L,
        train_mean=tr, train_std=tr, train_min=tr, train_max=tr,
        test_mean=te, test_std=te, test_min=te, test_max=te,
        excess_mean=nan, excess_std=nan, excess_min=nan, excess_max=nan,
        has_excess=false,
        finals=(nag_mean=Float64(r.atoms),),
    )
end

agg = Dict{Symbol,NamedTuple}(r.variant.sym => _ctx_entry(r) for r in results)
style = Dict{Symbol,Any}(
    r.variant.sym => (color=r.variant.color, marker=r.variant.marker, label=r.variant.plotlabel)
    for r in results)
plot_ctx = (
    agg=agg,
    rep=1,
    market_type=Symbol(string(cfg.market_type), "_adcg_ablation"),
    n=cfg.n, m=cfg.m, K=cfg.K,
    opt_plc=nothing,
    method_names=Symbol[r.variant.sym for r in results],
    interval_marker=1,
    style=style,
)

out_path = if cli["no_data_file"]
    ""
elseif !isempty(cli["data_file"])
    abspath(cli["data_file"])
else
    joinpath(cfg.out_dir, "ablation_$(String(cfg.market_type))_adcg.jls")
end

if !isempty(out_path)
    mkpath(dirname(abspath(out_path)))
    open(io -> serialize(io, plot_ctx), out_path, "w")
    # Suggest axis limits fitted to this run: x to the longest history, y to
    # the observed train/test range (padded), so the replayed figure isn't
    # squished by run_plot.jl's benchmark-scale defaults.
    xmax = maximum(r.iters for r in results)
    vals = Float64[]
    for r in results, v in vcat(r.hist[:primal_obj], r.hist[:test_err])
        (isfinite(v) && v > 0) && push!(vals, v)
    end
    ymin = isempty(vals) ? 5e-4 : minimum(vals) / 2
    ymax = isempty(vals) ? 1e0 : maximum(vals) * 2
    tag = string(cfg.market_type) * "_adcg_ablation"
    println()
    println("─"^60)
    println("Saved plot context (4 variants). To render train/test figures:")
    @printf("  julia --project=. revealed/run_plot.jl -f %s --xmax %d --ymin %.1e --ymax %.1e\n",
        out_path, xmax, ymin, ymax)
    println("  → writes real_$(tag)_{train,test}.pdf in the same dir")
    println("  (add --smooth N to denoise, --no-tex to skip the pgfplots .tex)")
    println("─"^60)
end
