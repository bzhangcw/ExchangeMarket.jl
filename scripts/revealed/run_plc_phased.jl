# Phased column generation on a PLC ground-truth market.
#
# Idea: start with a thin class menu (`:ces` only). When the CG run stalls
# or exhausts its phase budget, expand the menu and warm-restart with the
# accumulated `fa` (column-generation columns are reused; only the
# separation oracle's allowed classes change). Repeat.
#
# Phase schedule (`PHASES`):
#   1. `:ces`
#   2. `:ces, :leontief, :linear`
#   3. `:ces, :leontief, :linear, :ges`
#
# Output: a single `.jls` payload with the concatenated trajectory and a
# `phase_boundaries` vector marking iter counts at phase transitions, plus
# a train/test error PDF. The data file is consumable by `run_plot.jl`-style
# tooling that knows the `hist` schema.
#
# CLI: inherits `parse_args_for_test_real` (run_test.jl's parser). Requires
# `--market-type plc`. Per-phase budgets and stall tolerance are script
# constants (`PHASE_ITERS`, `PHASE_TOL_DELTA`) — edit at the top of the file.

using Revise
using Random, SparseArrays, LinearAlgebra
using ArgParse
using JuMP, MosekTools
using Printf, Serialization
using Plots, LaTeXStrings
import MathOptInterface as MOI

using ExchangeMarket

include("../tools.jl")
include("../plots.jl")        # generate_empty, switch_to_pdf — house plot style
include("./androids/plc.jl")
include("./setup.jl")

# Backend selection must happen at top level (calling pgfplotsx() from
# inside a function leaves savefig writing empty .tex / .pdf files —
# same constraint as run_plot.jl).
switch_to_pdf(; bool_use_html=false)

# -----------------------------------------------------------------------
# Phase schedule. Each phase fires a fresh `run_method_tracked` call with
# the listed classes; the surrogate market `fa` is threaded through, so
# columns from earlier phases stay in play.
# -----------------------------------------------------------------------
const PHASES = (
    (label="CES", classes=[:ces]),
    # (label="+Leontief", classes=[:ces, :leontief]),
    (label="+Linear", classes=[:ces, :linear]),
)
# const PHASE_ITERS = (500, 500, 200)   # max iters per phase
const PHASE_ITERS = (500, 100)   # max iters per phase
const PHASE_TOL_DELTA = 1e-9         # per-iter movement below this ⇒ phase considered stuck (overridden by --tol-delta if set)

# -----------------------------------------------------------------------
# CLI + ground truth
# -----------------------------------------------------------------------
const cli = parse_args_for_test_real()
cfg = build_run_config(cli)
cfg.market_type === :plc ||
    error("run_plc_phased: requires --market-type plc, got $(cfg.market_type)")

# Arrow–Debreu mode: `--methods adcg` runs each phase through the AD master
# (run_ad_tracked, endowments b_t ∈ ℝⁿ₊) instead of the Fisher master.
# Columns carry across phases via run_ad_tracked's :initial_cands warm
# restart. Default (`cg`/`cgma`/…) keeps the Fisher path. AD requires
# homothetic phase classes (run_ad_tracked errors otherwise).
const use_ad = :adcg in cfg.method_names

@info "configuration" market_type = cfg.market_type cfg.n cfg.m cfg.K cfg.K_test cfg.seed cfg.timelimit method = (use_ad ? :adcg : :cg) phases = [ph.label for ph in PHASES] phase_iters = PHASE_ITERS

rd = build_rep_data(cfg, 1, cfg.seed)

# -----------------------------------------------------------------------
# Base kwargs assembly. Mirrors `run_one_method` but is inlined here so we
# can interpose between phases and avoid being routed through a single
# method spec.
# -----------------------------------------------------------------------
function _base_kwargs()
    kw = Dict{Symbol,Any}(
        # No wall-clock cap per phase: each phase exits on `tol_delta`
        # (stuck) or `max_iters`. The CLI `--timelimit` is ignored on purpose.
        :timelimit => Inf,
        :interval_eval_test => cfg.interval_eval_test,
    )
    # First-order wealth (--wealth-function 1): per-iter excess validation not wired.
    if cfg.do_validate && !isnothing(rd.f_real) && cfg.interval_eval_excess > 0 &&
       cfg.wealth_function == 0 && !cfg.lift
        kw[:f_real] = rd.f_real
        kw[:interval_eval_excess] = cfg.interval_eval_excess
    end
    # tol_delta: per-iter movement below this ⇒ phase considered stuck.
    # Use CLI override if positive, 0 ⇒ disabled (run full PHASE_ITERS),
    # negative (default) ⇒ script default `PHASE_TOL_DELTA`.
    kw[:tol_delta] = if cfg.tol_delta_override > 0
        cfg.tol_delta_override
    elseif cfg.tol_delta_override == 0
        nothing
    else
        PHASE_TOL_DELTA
    end
    # tol_obj (absolute primal-objective threshold): pass CLI value if set;
    # don't impose a script default — phase boundaries should be driven by
    # stagnation (tol_delta), not by hitting an absolute target.
    if cfg.tol_obj_override >= 0
        kw[:tol_obj] = cfg.tol_obj_override == 0 ? nothing : cfg.tol_obj_override
    end
    if cli["tol_rc"] >= 0
        kw[:tol_rc] = cli["tol_rc"] == 0 ? nothing : cli["tol_rc"]
    end
    apply_cli_separation!(kw, cli)
    apply_cli_ces!(kw, cli)
    apply_cli_ges!(kw, cli)
    apply_cli_linear!(kw, cli)
    apply_cli_nn!(kw, cli)
    apply_cli_cpm!(kw, cli)
    apply_cli_redist!(kw, cli)
    return kw
end

# -----------------------------------------------------------------------
# Phased loop. `fa === nothing` on phase 1 ⇒ run_method_tracked bootstraps
# from a single random CES atom; subsequent phases pass `initial_fa = fa`
# to resume CG with the existing column set.
# -----------------------------------------------------------------------
fa = nothing
γ_ref = nothing
# AD mode only: the prior phase's atom list (NamedTuples with oracle params)
# and endowment masks, threaded into the next phase via :initial_cands /
# :initial_masks. The returned ArrowDebreuMarket stores converted (c, ρ)
# columns, not the oracle params or masks, so the warm restart needs this
# side channel (run_ad_tracked puts them in hist[:cands] / hist[:masks]).
prev_cands = nothing
prev_masks = nothing
combined = Dict(
    :primal_obj => Float64[],
    :test_err => Float64[],
    :excess => Float64[],
    :num_agents => Int[],
)
phase_boundaries = Int[0]        # cumulative iter count at the end of each phase
t_total = 0.0

for (ph_idx, ph) in enumerate(PHASES)
    kw = _base_kwargs()
    kw[:classes] = ph.classes
    kw[:max_iters] = PHASE_ITERS[ph_idx]
    if !isnothing(fa)
        if use_ad
            # AD warm restart: thread the prior phase's atoms + masks + tensor.
            kw[:initial_cands] = prev_cands
            kw[:initial_masks] = prev_masks
        else
            kw[:initial_fa] = fa
        end
        kw[:initial_γ_ref] = γ_ref   # row-aligned after the previous phase's tail drop
    end

    @info "=== Phase $ph_idx/$(length(PHASES)): $(ph.label) ==="
    t_ph = @elapsed begin
        global fa, γ_ref, prev_cands, prev_masks
        if use_ad
            fa, γ_ref, hist = run_ad_tracked(kw, rd.Ξ_train, rd.Ξ_test;
                verbosity=cfg.verbosity)
            prev_cands = hist[:cands]
            prev_masks = hist[:masks]
        else
            fa, γ_ref, hist = run_method_tracked(Symbol("phase_", ph_idx), :cg, kw,
                rd.Ξ_train, rd.Ξ_test; verbosity=cfg.verbosity)
        end
    end
    global t_total += t_ph
    append!(combined[:primal_obj], hist[:primal_obj])
    append!(combined[:test_err], hist[:test_err])
    append!(combined[:excess], hist[:excess])
    append!(combined[:num_agents], hist[:num_agents])
    push!(phase_boundaries, length(combined[:primal_obj]))
    @info "Phase $ph_idx done" androids = fa.m phase_iters = length(hist[:primal_obj]) phase_time = t_ph
end

@printf("\n=== run_plc_phased done ===\n")
@printf("total iters=%d  androids=%d  train_obj=%.3e  test_err=%.3e  time=%.3fs\n",
    length(combined[:primal_obj]), fa.m,
    combined[:primal_obj][end], combined[:test_err][end], t_total)

# -----------------------------------------------------------------------
# Snapshot + serialize. Same `deepcopy` discipline as run_one_method.jl.
# -----------------------------------------------------------------------
fa_fit = deepcopy(fa)
f_real_ser = deepcopy(rd.f_real)
payload = (
    fa=fa_fit,
    hist=combined,
    phase_boundaries=phase_boundaries,
    phases=[(label=ph.label, classes=ph.classes) for ph in PHASES],
    phase_iters=collect(PHASE_ITERS),
    t=t_total,
    name=:phased,
    market_type=:plc,
    n=cfg.n, m=cfg.m, K=cfg.K, K_test=cfg.K_test, seed=cfg.seed,
    cli=cli,
    f_real=f_real_ser,
    validation=nothing,
)

out_path = if cli["no_data_file"]
    ""
elseif !isempty(cli["data_file"])
    abspath(cli["data_file"])
else
    joinpath(cfg.out_dir, "run_plc_phased.jls")
end
isempty(out_path) || save_run(out_path, payload)

# -----------------------------------------------------------------------
# Train / test error trajectories. Style mirrors `_make_figure` in
# revealed/run_plot.jl (pgfplotsx, log-y, dense ticks, large fonts) so
# these PDFs slot into the same paper-figure pipeline. Phase transitions
# are drawn as dashed gray verticals annotated with the upcoming class.
# -----------------------------------------------------------------------
let
    ts = collect(1:length(combined[:primal_obj]))
    train = max.(combined[:primal_obj], 1e-8)
    test = max.(combined[:test_err], 1e-8)
    xlims = (1, length(ts))
    ylims = (
        min(minimum(train), minimum(test)) / 2,
        max(maximum(train), maximum(test)) * 2
    )

    _phased_fig(ylabel_str) = begin
        f = generate_empty(; shape=:wide)
        plot!(f,
            ylabel=ylabel_str,
            xlabel=L"\textrm{iteration}",
            legendbackgroundcolor=RGBA(1.0, 1.0, 1.0, 0.8),
            yscale=:log10,
            xtickfont=font(18),
            ytickfont=font(18),
            xscale=:identity,
            size=(600, 500),
            left_margin=12Plots.mm,
            bottom_margin=8Plots.mm,
            legendfontsize=18,
            xlims=xlims,
            ylims=ylims,
        )
        return f
    end

    fig = _phased_fig(L"\textrm{error}")
    plot!(fig, ts, train; label=L"\textrm{train}", linewidth=3, color=1)
    plot!(fig, ts, test; label=L"\textrm{test}", linewidth=3, color=2)

    # Label every phase at the centre of its iteration range. Transitions
    # between phases are drawn as dashed gray verticals at the boundary.
    y_offset = ylims[1] * 10
    for k in 1:length(PHASES)
        lo = phase_boundaries[k]            # 0 for phase 1, prior boundary otherwise
        hi = phase_boundaries[k+1]
        mid = (lo + hi) / 2
        annotate!(fig, mid, y_offset,
            text(L"\textrm{%$(PHASES[k].label)}", 12, :center))
    end
    for b in phase_boundaries[2:end-1]
        vline!(fig, [b]; ls=:dash, lc=:gray, label="")
    end

    base_path = isempty(out_path) ?
                joinpath(cfg.out_dir, "run_plc_phased") :
                first(splitext(out_path))
    plot_tex = base_path * ".tex"
    plot_pdf = base_path * ".pdf"
    savefig(fig, plot_tex)
    savefig(fig, plot_pdf)
    @info "saved train/test plot" plot_tex
end
