# Run a SINGLE method on ONE real-market dataset and serialize the fitted
# surrogate (`fa`) along with its training/test history, the parsed CLI
# config, and the ground-truth market.
#
# This is the leaner sibling of run_test.jl: instead of the reps × methods
# grid + aggregation + plotting, it fits one method once and dumps a
# self-contained `.jls` payload that downstream tools (validation, custom
# plots, warm starts) can `load_run` without re-running the fit.
#
# The CLI is identical to run_test.jl (`parse_args_for_test_real`); pass
# exactly one token to `--methods`.
#
# CLI: see `julia run_one_method.jl -h` for all options.

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

# Single-method contract: reuse the grid plumbing but reject ambiguity.
length(cfg.method_names) == 1 ||
    error("run_one_method expects exactly one --methods token, got $(cfg.method_names)")
name = cfg.method_names[1]
spec = first(s for s in method_kwargs if s[1] == name)
(_, separation_kind, kwargs) = spec

print_tree("configuration", [
    "market" => cfg.market_type,
    "dimensions" => [
        "n" => cfg.n,
        "m" => cfg.m,
        "K" => cfg.K,
        "K_test" => cfg.K_test,
    ],
    "method" => name,
    "classes" => cfg.allowed_classes,
    "ces_ρ_range" => cfg.ces_rho_range,
    "sparsity" => cfg.sparsity,
    "seed" => cfg.seed,
    "timelimit" => @sprintf("%g s", cfg.timelimit),
])

# One dataset at the master seed; one fit.
rd = build_rep_data(cfg, 1, cfg.seed)
res = run_one_method(cfg, cli, 1, rd.rep_seed, rd.Ξ_train, rd.Ξ_test, rd.f_real,
    name, separation_kind, kwargs)

# Snapshot the as-fit surrogate BEFORE validation. `validate_surrogate`
# solves the surrogate equilibrium in-place (mutating fa.p / fa.x), so we
# snapshot first to persist the fitted state.
#
# Use `deepcopy`, NOT `Base.copy`: copy resets the AgentView registry
# (`fa.agents`), which `validate_surrogate` reads — a copied fa validates to
# the wrong number. `deepcopy` faithfully clones the registry (its views
# rebind to the cloned storage) and round-trips through Serialization. The
# CPU revealed-preference path leaves `fa.workspace`/`gpu_workspace_cache`
# at `nothing`, so there is no GPU state to strip.
fa_fit = deepcopy(res.fa)

# -----------------------------------------------------------------------
# Optional validation: solve the surrogate equilibrium and judge it by the
# real market via the price-scaled excess demand. Mirrors run_test.jl.
# -----------------------------------------------------------------------
do_validate_effective = cfg.do_validate &&
                        (cfg.market_type === :ces || cfg.market_type === :plc) &&
                        # AD validation (surrogate or ground truth) not wired yet
                        cfg.budget_type !== :ad &&
                        !(res.fa isa ArrowDebreuMarket)
if cfg.do_validate && !(cfg.market_type === :ces || cfg.market_type === :plc)
    @warn "--validate only supported for --market-type ces|plc; skipping" market_type = cfg.market_type
end
validation = nothing
if do_validate_effective && rd.f_real !== nothing
    validation = validate_surrogate(res.fa, rd.f_real; verbose=false)
    pretty = get(display_name, name, String(name))
    println()
    println("=== Validation: real-market clearing at surrogate equilibrium price ===")
    @printf("%-10s | %12s | %12s\n", "method", "‖p(q-g)‖∞", "‖p(q-g)‖₁")
    @printf("%-10s-+-%12s-+-%12s\n", "----------", "------------", "------------")
    @printf("%-10s | %12.3e | %12.3e\n",
        pretty, validation.excess_surrogate_linf, validation.excess_surrogate_l1)
end

@printf("\n=== %s done ===\n", String(name))
@printf("iters=%d  androids=%d  train_obj=%.3e  test_err=%.3e  time=%.3fs\n",
    length(res.hist[:primal_obj]), res.fa.m,
    res.hist[:primal_obj][end], res.hist[:test_err][end], res.t)

# -----------------------------------------------------------------------
# Serialize the run. `deepcopy` the ground-truth market for the same reason
# (CES `f_real` is a FisherMarket whose AgentView registry must survive;
# the PLC `f_real` is a plain NamedTuple of numeric agents). deepcopy
# detaches it from the live objects mutated during validation.
#
# ArrowDebreuMarket (AD ground truth and/or adcg surrogate) carries closure
# fields (f, f∇f) whose anonymous types don't round-trip across Julia
# sessions — strip them on the serialized copies; all numeric state
# (c, ρ, σ, b, w, q) survives.
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
    hist=res.hist,
    t=res.t,
    name=name,
    market_type=cfg.market_type,
    n=cfg.n, m=cfg.m, K=cfg.K, K_test=cfg.K_test, seed=cfg.seed,
    cli=cli,
    f_real=f_real_ser,
    validation=validation,
)

# Default path is distinct from run_test.jl's plot ctx (`real_<market>.jls`).
# Honor --data-file / --no-data-file exactly like run_test.jl.
out_path = if cli["no_data_file"]
    ""
elseif !isempty(cli["data_file"])
    abspath(cli["data_file"])
else
    joinpath(cfg.out_dir, "run_$(String(cfg.market_type))_$(lowercase(String(name))).jls")
end

if !isempty(out_path)
    save_run(out_path, payload)
    println()
    println("─"^60)
    println("Saved fitted surrogate + history. To reload:")
    println("  julia> using Serialization; include(\"revealed/setup.jl\")")
    println("  julia> r = load_run(\"$(out_path)\")")
    println("  julia> r.fa            # ::FisherMarket")
    println("  julia> validate_surrogate(r.fa, r.f_real)")
    println("─"^60)
end
