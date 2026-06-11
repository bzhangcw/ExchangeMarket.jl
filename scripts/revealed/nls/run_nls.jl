# Fit a fixed-agent NLS surrogate to ONE real-market dataset and serialize the
# result — the NLS sibling of ../run_one_method.jl.
#
# Instead of growing the surrogate by column generation, this fixes the number
# of CES agents t and solves for all (yᵢ, σᵢ) and the wealth split w ∈ Δᵗ
# jointly (nls.jl::solve_nls, via MadNLP). Pass a comma-separated `--nls-t`
# list to sweep several agent counts in one run (e.g. `--nls-t 5,10,20,40`);
# the serialized history then carries a risk-vs-agents curve directly
# comparable to CG's risk-vs-androids curve.
#
# The ground-truth market, the train/test revealed-preference data, and the
# equilibrium validation all reuse ../setup.jl unchanged — only the fitting
# step differs. The surrogate is a genuine CES FisherMarket, so
# validate_surrogate judges it exactly as it judges a CG surrogate.
#
# CLI: identical to run_one_method.jl (parse_args_for_test_real), PLUS the
# NLS-only flags consumed here before the shared parser sees ARGS:
#   --nls-t LIST          comma-separated agent counts to fit (default "10")
#   --nls-budget-type B   surrogate budget model: fisher|ad. Independent of the
#                         ground-truth --wealth-function; defaults to inheriting it.
#                         (e.g. --wealth-function 1 --nls-budget-type fisher fits
#                         a Fisher surrogate to an Arrow–Debreu market.)
#   --nls-max-iter N      MadNLP iteration cap per fit (default 100)
#   --nls-ad-delta D      AD surrogate supply scale δ ≥ 0 (default 1.0; the
#                         unit-supply predictor). AD only.
#   --nls-ad-delta-free   fit δ ≥ 0 jointly as a variable budget scale
#                         (eq.ad.master.scaled); overrides --nls-ad-delta. AD only.
#   --nls-no-warmstart    fit each t from scratch (default: warm-start from the
#                         previous, smaller fit)
# `--methods` is ignored (there is only one method here).

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

include("../../tools.jl")
include("../androids/plc.jl")   # ground-truth PLC generator (see run_one_method.jl)
include("../setup.jl")
include("./nls.jl")             # solve_nls / evaluate_nls (MadNLP, Fisher)
include("./nls_ad.jl")          # solve_nls_ad / evaluate_nls_ad (MadNLP, Arrow–Debreu)
include("./nls_runner.jl")      # run_method_tracked_nls / build_fa_from_nls

# -----------------------------------------------------------------------
# Pull the NLS-only flags out of ARGS, then hand the remainder to the
# shared parser (which would otherwise reject the unknown flags). This
# keeps the experimental NLS module fully decoupled from setup.jl's CLI.
# -----------------------------------------------------------------------
function _extract_nls_args(argv)
    rest = String[]
    t_list = [10]
    warmstart = true
    max_iter = 100
    # `nothing` ⇒ inherit the surrogate budget from the ground-truth
    # --wealth-function. An explicit --nls-budget-type decouples them, so you can
    # e.g. fit a Fisher surrogate to an Arrow–Debreu market or vice versa.
    nls_budget = nothing
    # AD supply scale δ: a fixed number, or :free (--nls-ad-delta-free).
    ad_delta = 1.0
    ad_delta_free = false
    _parse_budget(v) = (s = Symbol(lowercase(strip(v)));
        s in (:fisher, :ad) ? s : error("--nls-budget-type must be fisher|ad, got '$v'"))
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--nls-t"
            i += 1
            i <= length(argv) || error("--nls-t requires a value (e.g. --nls-t 5,10,20)")
            t_list = parse.(Int, split(strip(argv[i]), ","))
        elseif startswith(a, "--nls-t=")
            t_list = parse.(Int, split(strip(split(a, "=", limit=2)[2]), ","))
        elseif a == "--nls-budget-type"
            i += 1
            i <= length(argv) || error("--nls-budget-type requires a value (fisher|ad)")
            nls_budget = _parse_budget(argv[i])
        elseif startswith(a, "--nls-budget-type=")
            nls_budget = _parse_budget(split(a, "=", limit=2)[2])
        elseif a == "--nls-max-iter"
            i += 1
            i <= length(argv) || error("--nls-max-iter requires a value")
            max_iter = parse(Int, strip(argv[i]))
        elseif startswith(a, "--nls-max-iter=")
            max_iter = parse(Int, strip(split(a, "=", limit=2)[2]))
        elseif a == "--nls-ad-delta"
            i += 1
            i <= length(argv) || error("--nls-ad-delta requires a value (δ ≥ 0)")
            ad_delta = parse(Float64, strip(argv[i]))
        elseif startswith(a, "--nls-ad-delta=")
            ad_delta = parse(Float64, strip(split(a, "=", limit=2)[2]))
        elseif a == "--nls-ad-delta-free"
            ad_delta_free = true
        elseif a == "--nls-no-warmstart"
            warmstart = false
        else
            push!(rest, a)
        end
        i += 1
    end
    all(>(0), t_list) || error("--nls-t values must be positive, got $t_list")
    max_iter > 0 || error("--nls-max-iter must be positive, got $max_iter")
    ad_delta >= 0 || error("--nls-ad-delta must be ≥ 0, got $ad_delta")
    # :free overrides any fixed --nls-ad-delta (mirrors adcg's --ad-delta-free).
    delta_spec = ad_delta_free ? :free : ad_delta
    return rest, sort(unique(t_list)), warmstart, nls_budget, max_iter, delta_spec
end

const _rest, _t_list, _warmstart, _nls_budget_opt, _nls_max_iter, _nls_delta =
    _extract_nls_args(ARGS)
const cli = parse_args_for_test_real(_rest)
cfg = build_run_config(cli)
const name = :NLS
# Ground-truth budget symbol derived from --wealth-function (0→fisher, 1→ad).
const _gt_budget = cfg.wealth_function == 1 ? :ad : :fisher
# Surrogate budget: explicit --nls-budget-type wins; else inherit the
# ground-truth wealth model (preserving prior single-flag behavior).
const _nls_budget = isnothing(_nls_budget_opt) ? _gt_budget : _nls_budget_opt

print_tree("configuration", [
    "market" => cfg.market_type,
    "dimensions" => [
        "n" => cfg.n,
        "m" => cfg.m,
        "K" => cfg.K,
        "K_test" => cfg.K_test,
    ],
    "method" => name,
    "budget (ground truth)" => _gt_budget,
    "budget (nls surrogate)" => _nls_budget,
    "nls_t" => _t_list,
    "nls_max_iter" => _nls_max_iter,
    "ad supply δ" => _nls_budget === :ad ? string(_nls_delta) : "n/a (fisher)",
    "warmstart" => _warmstart,
    "ces_ρ_range" => cfg.ces_rho_range,
    "sparsity" => cfg.sparsity,
    "seed" => cfg.seed,
])

# One dataset at the master seed; one NLS fit per t.
rd = build_rep_data(cfg, 1, cfg.seed)

nls_kwargs = Dict{Symbol,Any}(
    :t_list => _t_list,
    :seed => cfg.seed,
    :warmstart => _warmstart,
    :budget => _nls_budget,        # surrogate budget: --nls-budget-type or inherited
    :nls_max_iter => _nls_max_iter,
    :ad_delta => _nls_delta,       # AD supply scale δ (fixed number or :free)
)

t_elapsed = @elapsed begin
    fa, γ_ref, hist = run_method_tracked_nls(
        nls_kwargs, rd.Ξ_train, rd.Ξ_test; verbosity=max(cfg.verbosity, 1)
    )
end

# Snapshot the as-fit surrogate BEFORE validation (validate_surrogate solves
# the surrogate equilibrium in-place, mutating fa.p / fa.x). deepcopy clones
# the AgentView registry faithfully; see run_one_method.jl for the rationale.
fa_fit = deepcopy(fa)

# -----------------------------------------------------------------------
# Optional validation: solve the surrogate equilibrium and judge it against
# the real market via the price-scaled excess demand. Requires (1) a CES
# Fisher surrogate — validate_surrogate solves a CES equilibrium, undefined
# for an AD surrogate — and (2) a Fisher CES/PLC ground truth, the only
# f_real validate_surrogate has a method for (AD ground truth deferred).
# -----------------------------------------------------------------------
do_validate_effective = cfg.do_validate &&
                        (cfg.market_type === :ces || cfg.market_type === :plc) &&
                        cfg.wealth_function == 0 && !cfg.lift &&  # ground-truth f_real validate-able
                        _nls_budget !== :ad            # surrogate is a FisherMarket
if cfg.do_validate && !(cfg.market_type === :ces || cfg.market_type === :plc)
    @warn "--validate only supported for --market-type ces|plc; skipping" market_type = cfg.market_type
end
validation = nothing
if do_validate_effective && rd.f_real !== nothing
    validation = validate_surrogate(fa, rd.f_real; verbose=false)
    if !isempty(hist[:excess])
        hist[:excess][end] = validation.excess_surrogate_linf
    end
    println()
    println("=== Validation: real-market clearing at surrogate equilibrium price ===")
    @printf("%-10s | %12s | %12s\n", "method", "‖p(q-g)‖∞", "‖p(q-g)‖₁")
    @printf("%-10s-+-%12s-+-%12s\n", "----------", "------------", "------------")
    @printf("%-10s | %12.3e | %12.3e\n",
        String(name), validation.excess_surrogate_linf, validation.excess_surrogate_l1)
end

@printf("\n=== %s done ===\n", String(name))
@printf("t_list=%s  androids(final)=%d  train=%.3e  test=%.3e  time=%.3fs\n",
    string(_t_list), fa.m,
    hist[:primal_obj][end], hist[:test_err][end], t_elapsed)

# -----------------------------------------------------------------------
# Serialize. Mirror run_one_method.jl: deepcopy + strip closures from any
# ArrowDebreuMarket ground truth so the payload round-trips across sessions.
# -----------------------------------------------------------------------
_strip_closures!(x) = x
function _strip_closures!(ad::ArrowDebreuMarket)
    ad.f = nothing
    ad.f∇f = nothing
    return ad
end
_strip_closures!(fa_fit)
f_real_ser = _strip_closures!(deepcopy(rd.f_real))
payload = (
    fa=fa_fit,
    hist=hist,
    t=t_elapsed,
    name=name,
    market_type=cfg.market_type,
    n=cfg.n, m=cfg.m, K=cfg.K, K_test=cfg.K_test, seed=cfg.seed,
    nls_t=_t_list,
    nls_budget=_nls_budget,
    nls_max_iter=_nls_max_iter,
    nls_delta=_nls_delta,
    warmstart=_warmstart,
    cli=cli,
    f_real=f_real_ser,
    validation=validation,
)

out_path = if cli["no_data_file"]
    ""
elseif !isempty(cli["data_file"])
    abspath(cli["data_file"])
else
    joinpath(cfg.out_dir, "run_$(String(cfg.market_type))_nls.jls")
end

if !isempty(out_path)
    save_run(out_path, payload)
    println()
    println("─"^60)
    println("Saved fitted NLS surrogate + history. To reload:")
    println("  julia> using Serialization; include(\"revealed/setup.jl\")")
    println("  julia> r = load_run(\"$(out_path)\")")
    println("  julia> r.fa            # ::$(_nls_budget === :ad ? "ArrowDebreuMarket" : "FisherMarket")")
    do_validate_effective &&
        println("  julia> validate_surrogate(r.fa, r.f_real)")
    println("─"^60)
end
