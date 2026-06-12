# Unified experiment driver for revealed-preference CES surrogate fitting.
#
# Fits every variant in the preset's run set (presets.yaml `variant:`) on ONE
# real-market dataset, prints a side-by-side comparison table (+ optional
# real-market validation), and serializes a run_plot.jl-compatible plot context
# (each variant a styled series). Replaces the former run_test / run_one_method /
# run_one_method_ablation trio: the run set IS the variant list, and a plain
# method is just a variant with an empty `cli:`.
#
# Selection is the variant list minus `SKIP_VARIANTS` below — edit that to drop
# a few without touching the YAML. Use `--preset PATH` for a different catalog.
#
# The dataset is built ONCE and the global RNG is reseeded to --seed before each
# fit, so the bootstrap atom is identical across variants and only the
# method/knobs differ. Plotting is delegated to run_plot.jl (see the hint at the
# end), so this step needs no Plots / PGFPlotsX / LaTeX stack.
#
# NOTE: the shared CLI parser still accepts --methods / --rep, but this driver
# ignores them; the run set comes from the preset and each fit uses one dataset.
#
# CLI: see `julia run_test.jl -h` for all options.

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
include("./androids/plc.jl")   # ground-truth PLC generator
include("./setup.jl")

# Variants to skip, by `sym` (the preset's variant names; presets.yaml
# `variant:`). Plain strings — edit to drop a few from the run without touching
# the YAML. Current syms: 
ALL_VARIANTS = [
    "cg", "cgma",
    "adcg_hard_vardelta",
    "fwjl", "adfwjl", # FW.jl
    "fw", "adfw"
]
# E.g. to run only our own AD-FW:
#   const SKIP_VARIANTS = ["cg", "cgma", "adcg_hard_vardelta", "fw", "fwjl", "adfwjl"]
# (set to String[] to run every variant).
# SKIP_VARIANTS = String[]
# SKIP_VARIANTS = filter(x -> x !== "cg", ALL_VARIANTS)
# SKIP_VARIANTS = filter(x -> x !== "adcg_hard_vardelta", ALL_VARIANTS)
SKIP_VARIANTS = filter(x -> x ∈ ["cgma", "fwjl", "adfwjl"], ALL_VARIANTS)
# SKIP_VARIANTS = filter(x -> x ∈ ["fwjl", "adfwjl"], ALL_VARIANTS)
# SKIP_VARIANTS = filter(x -> !(x in ["adcg_hard_vardelta", "adfw"]), ALL_VARIANTS)

const cli = parse_args_for_test_real()
cfg = build_run_config(cli)

# Available variant syms (for error / warning messages and skip validation).
const _ALL_SYMS = [String(v.sym) for v in method_variants]
for s in SKIP_VARIANTS
    s in _ALL_SYMS || @warn "SKIP_VARIANTS entry \"$s\" matches no variant sym" available = _ALL_SYMS
end
variants = NamedTuple[v for v in method_variants if !(String(v.sym) in SKIP_VARIANTS)]
isempty(variants) &&
    error("no variants to run — all skipped? SKIP_VARIANTS = $(SKIP_VARIANTS), available = $(_ALL_SYMS)")

print_tree("configuration", [
    "market" => cfg.market_type,
    "wealth_function" => cfg.wealth_function == 1 ? "first-order (AD)" :
                         cfg.wealth_function == 2 ? "second-order (quadratic)" : "constant (Fisher)",
    "lift" => cfg.lift,
    "dimensions" => ["n" => cfg.n, "m" => cfg.m, "K" => cfg.K, "K_test" => cfg.K_test],
    "classes" => cfg.allowed_classes,
    "engine" => "$(cfg.engine) (LP→$(lp_engine()), gurobi=$(gurobi_available()))",
    "sample_size" => cli["sample_size"],
    "variants" => [string(i) => v.label for (i, v) in enumerate(variants)],
    "skipped" => isempty(SKIP_VARIANTS) ? "(none)" : join(string.(SKIP_VARIANTS), ", "),
    "seed" => cfg.seed,
    "timelimit" => @sprintf("%g s", cfg.timelimit),
])
if cli["sample_size"] <= 0 && any(get(v.cli, "sample_hard", false) === true for v in variants)
    @warn "--sample-size is 0 (full batch): the sample_hard variants are " *
          "identical to their uniform siblings. Pass --sample-size > 0 to " *
          "exercise the boosting mini-batch."
end

# One dataset at the master seed.
rd = build_rep_data(cfg, 1, cfg.seed)

results = Vector{NamedTuple}(undef, length(variants))
for (i, v) in enumerate(variants)
    println()
    println("="^100)
    ov = isempty(v.cli) ? "" :
         "  [" * join(["$k=$val" for (k, val) in v.cli], ", ") * "]"
    @printf("VARIANT %d/%d: %s  (method=%s)%s\n",
        i, length(variants), v.label, v.method, ov)
    println("="^100)

    # Look up this variant's base method spec (separation_kind + base kwargs).
    (_, separation_kind, base_kwargs) = first(s for s in method_kwargs if s[1] == v.method)

    # Per-variant CLI: copy the parsed flags, then apply this variant's
    # overrides (e.g. "sample_hard", "ad_delta_free"). No-ops for methods that
    # don't read them. A shallow copy suffices — only top-level scalar keys change.
    cli_v = copy(cli)
    for (k, val) in v.cli
        cli_v[k] = val
    end

    # Reseed so the bootstrap atom (and any sampling randomness) starts
    # identically across variants — the method/knobs become the only difference.
    Random.seed!(cfg.seed)
    res = run_one_method(cfg, cli_v, 1, rd.rep_seed, rd.Ξ_train, rd.Ξ_test, rd.f_real,
        v.method, separation_kind, base_kwargs; wealth_fn=rd.wealth_fn)

    δ_final = haskey(res.hist, :delta) && !isempty(res.hist[:delta]) ?
              res.hist[:delta][end] : NaN
    results[i] = (
        variant=v,
        fa=res.fa,
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
# Optional validation: solve each fitted surrogate's equilibrium and judge it
# by the real market via the price-scaled excess demand. AD (adcg surrogate or
# AD ground truth) is skipped — validate_surrogate has no price-dependent-budget
# method.
# -----------------------------------------------------------------------
validation = Dict{Symbol,Any}()
# Validation works for ANY real-market family: solve the surrogate's
# equilibrium (Fisher via Mirror Descent, AD via potred), project the price
# back to the real goods if the surrogate is money-lifted, and evaluate the
# real market's aggregate demand there. CES ground truth uses rd.wealth_fn
# (constant/first-order); GES/SPLC/NGES use their per-class demand; PLC uses
# its fixed-budget joint-LP (constant wealth, no lift).
do_validate_effective = cfg.do_validate
if cfg.do_validate && cfg.market_type === :plc && (cfg.wealth_function != 0 || cfg.lift)
    @warn "PLC ground-truth validation supports only --wealth-function 0 without --lift; skipping" wealth_function = cfg.wealth_function lift = cfg.lift
    global do_validate_effective = false
end
if do_validate_effective && rd.f_real !== nothing
    println()
    println("=== Validation: real-market clearing at surrogate equilibrium price ===")
    println("    orig = ‖q(1−d(q))‖∞ on the n real goods; norm = same at the simplex-normalized price q̃=q/⟨1,q⟩ (scale-invariant, comparable across lift/no-lift); lift = ‖p̄(supply−d̄)‖∞ on the n+1 lifted goods (− if not lifted); surr = surrogate's OWN clearing residual at p* (≈0 ⇒ equilibrium reached ⇒ real excess is model misspecification)")
    @printf("%-22s | %14s | %14s | %14s | %14s\n", "variant", "orig ‖p(q-g)‖∞", "norm ‖q̃(1-d)‖∞", "lift ‖p̄(q̄-d̄)‖∞", "surr ‖p(1-d̂)‖∞")
    @printf("%-22s-+-%14s-+-%14s-+-%14s-+-%14s\n", "-"^22, "-"^14, "-"^14, "-"^14, "-"^14)
    for r in results
        println("--- solving $(r.variant.label) surrogate equilibrium ---")
        vres = validate_surrogate(r.fa, rd.f_real; wealth_fn=rd.wealth_fn, verbose=true)
        validation[r.variant.sym] = vres
        liftstr = isnan(vres.excess_lift_linf) ? "-" : @sprintf("%.3e", vres.excess_lift_linf)
        surrstr = isnan(vres.excess_surr_self_linf) ? "-" : @sprintf("%.3e", vres.excess_surr_self_linf)
        @printf("%-22s | %14.3e | %14.3e | %14s | %14s\n",
            r.variant.label, vres.excess_surrogate_linf, vres.excess_norm_linf, liftstr, surrstr)
    end
end

# -----------------------------------------------------------------------
# Comparison table.
# -----------------------------------------------------------------------
println()
println("="^100)
println("SUMMARY  (market=$(cfg.market_type), wealth=$(cfg.wealth_function), lift=$(cfg.lift), n=$(cfg.n), m=$(cfg.m), K=$(cfg.K), sample_size=$(cli["sample_size"]))")
println("="^100)
@printf("%-22s | %5s | %5s | %11s | %11s | %8s | %8s | %9s | %11s | %11s | %11s | %11s\n",
    "variant", "iters", "atoms", "train_obj", "test_err", "δ*", "time(s)", "t/it(ms)", "‖p(q-g)‖∞", "norm‖∞", "lift‖∞", "surr‖∞")
@printf("%-22s-+-%5s-+-%5s-+-%11s-+-%11s-+-%8s-+-%8s-+-%9s-+-%11s-+-%11s-+-%11s-+-%11s\n",
    "-"^22, "-"^5, "-"^5, "-"^11, "-"^11, "-"^8, "-"^8, "-"^9, "-"^11, "-"^11, "-"^11, "-"^11)
for r in results
    vres = get(validation, r.variant.sym, nothing)
    valstr = isnothing(vres) ? "          -" : @sprintf("%11.3e", vres.excess_surrogate_linf)
    normstr = isnothing(vres) ? "          -" : @sprintf("%11.3e", vres.excess_norm_linf)
    liftstr = (isnothing(vres) || isnan(vres.excess_lift_linf)) ? "          -" :
              @sprintf("%11.3e", vres.excess_lift_linf)
    surrstr = (isnothing(vres) || isnan(vres.excess_surr_self_linf)) ? "          -" :
              @sprintf("%11.3e", vres.excess_surr_self_linf)
    t_per_it = r.iters > 0 ? 1000 * r.t / r.iters : NaN   # mean wall-clock per iteration (ms)
    @printf("%-22s | %5d | %5d | %11.3e | %11.3e | %8.4f | %8.2f | %9.3f | %s | %s | %s | %s\n",
        r.variant.label, r.iters, r.atoms, r.train, r.test, r.delta, r.t, t_per_it, valstr, normstr, liftstr, surrstr)
end

best_train = argmin(r -> r.train, results)
best_test = argmin(r -> r.test, results)
println()
@printf("best train_obj: %s (%.3e)\n", best_train.variant.label, best_train.train)
@printf("best test_err : %s (%.3e)\n", best_test.variant.label, best_test.test)

# Optional CSV log: one row per variant.
if !isempty(cfg.csv_path)
    csv_path = isabspath(cfg.csv_path) ? cfg.csv_path : joinpath(@__DIR__, cfg.csv_path)
    new_file = !isfile(csv_path)
    open(csv_path, "a") do io
        new_file && println(io,
            "market_type,n,m,K,variant,method,iters,atoms_T,train_obj,test_err,delta,time_s")
        for r in results
            println(io, join((
                    String(cfg.market_type), cfg.n, cfg.m, cfg.K,
                    String(r.variant.sym), String(r.variant.method),
                    r.iters, r.atoms, r.train, r.test, r.delta, round(r.t; digits=3),
                ), ","))
        end
    end
    @info "appended results" csv_path
end

# -----------------------------------------------------------------------
# Serialize a run_plot.jl-compatible plot context (each variant a styled
# series). Built inline so this run step needs no Plots / PGFPlotsX / LaTeX
# stack — only run_plot.jl does. rep = 1, so the std/min/max fields equal the
# single trajectory (the plotter draws no ribbon).
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
    market_type=cfg.market_type,
    n=cfg.n, m=cfg.m, K=cfg.K,
    opt_plc=cfg.opt_plc,
    method_names=Symbol[r.variant.sym for r in results],
    interval_marker=10,
    style=style,
)

if !isempty(cfg.data_file_path)
    mkpath(dirname(abspath(cfg.data_file_path)))
    open(io -> serialize(io, plot_ctx), cfg.data_file_path, "w")
    # Render hint to stderr (unbuffered; stdout is block-buffered when piped).
    println(stderr)
    println(stderr, "─"^60)
    println(stderr, "To render the risk curves:")
    println(stderr, "  julia --project=. revealed/run_plot.jl -f $(cfg.data_file_path)")
    println(stderr, "  (add --smooth N for a moving-average window, --no-tex to skip pgfplots .tex)")
    println(stderr, "─"^60)
    flush(stdout)
end

# -----------------------------------------------------------------------
# Serialize the ground-truth real market and every fitted surrogate, so a run
# can be re-validated / inspected without refitting. The saved NamedTuple holds
# the real market (`f_real`) and its wealth model (`wealth_fn`), the train/test
# revealed-preference datasets, the validation results, and one entry per
# variant carrying the fitted surrogate market (`fa`) plus its labels. Reload
# with `Serialization.deserialize` (ExchangeMarket must be loaded for the market
# types). Suppressed by `--no-data-file`.
# -----------------------------------------------------------------------
if !cli["no_data_file"]
    surr_path = joinpath(cfg.out_dir, "surrogates_$(String(cfg.market_type)).jls")
    saved = (
        market_type=cfg.market_type,
        wealth_function=cfg.wealth_function,
        lift=cfg.lift,
        n=cfg.n, m=cfg.m, K=cfg.K, seed=cfg.seed,
        f_real=rd.f_real,
        wealth_fn=rd.wealth_fn,
        Xi_train=rd.Ξ_train,
        Xi_test=rd.Ξ_test,
        validation=validation,
        surrogates=[(sym=r.variant.sym, label=r.variant.label,
            method=r.variant.method, fa=r.fa) for r in results],
    )
    mkpath(cfg.out_dir)
    open(io -> serialize(io, saved), surr_path, "w")
    @info "saved real market + fitted surrogates" surr_path n_variants = length(results)
end
