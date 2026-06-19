# Solver-engine wrapper for /revealed/.
#
# One factory (`new_model`) picks the optimization backend so the scripts run
# whether or not Gurobi is installed/licensed:
#   * LP / MIP / conic masters → Gurobi when available and selected, else Mosek.
#   * NLP separations          → always MadNLP (the only NLP engine).
#
# Gurobi is loaded CONDITIONALLY: a missing package or an unusable license must
# not crash the whole `run_test.jl` include chain at load time (several files
# used to `using Gurobi` unconditionally). After this file, `gurobi_available()`
# reports whether Gurobi is usable, callers route model creation through
# `new_model`, and Gurobi-only attributes go through `set_lp_barrier!`.
#
# Included FIRST from setup.jl so every later solver site sees these helpers.

using JuMP, MosekTools, MadNLP

# --- Conditional Gurobi load ------------------------------------------------
# `@eval import Gurobi` so a missing package throws here (caught) instead of at
# parse time. Env creation is a SEPARATE top-level statement: that advances the
# world age, so `Gurobi.Env()` resolves the freshly-imported binding.
const _HAS_GUROBI = Ref(false)
try
    @eval import Gurobi
    _HAS_GUROBI[] = true
catch
    _HAS_GUROBI[] = false
end

const _GRB_ENV = Ref{Any}()
if _HAS_GUROBI[]
    try
        # One shared env across all Gurobi models: the academic-license banner
        # prints once at load (not per CG iteration) and OutputFlag=0 silences
        # the per-model "Set parameter X" echoes.
        _GRB_ENV[] = Gurobi.Env()
        Gurobi.GRBsetintparam(_GRB_ENV[], "OutputFlag", 0)
    catch e
        # Installed but unusable (expired / missing / no token). Fall back.
        _HAS_GUROBI[] = false
    end
end

if !_HAS_GUROBI[]
    @warn "Gurobi unavailable: using Mosek for LP/MIP masters and disabling the linear android." maxlog = 1
end

"""Whether Gurobi is importable and its environment usable."""
gurobi_available() = _HAS_GUROBI[]

"""Shared Gurobi environment (or an unassigned ref if Gurobi is unavailable)."""
_gurobi_env() = _GRB_ENV[]

# --- Engine selection -------------------------------------------------------
const _ENGINE = Ref(:auto)   # :auto | :gurobi | :mosek (set from --engine)

"""Set the LP/MIP backend from `--engine` (`auto` / `gurobi` / `mosek`)."""
set_engine!(name) = (_ENGINE[] = Symbol(lowercase(String(name))))

"""
    lp_engine() -> :gurobi | :mosek

Resolve the effective LP/MIP backend: `:auto` picks Gurobi when available and
Mosek otherwise; `:gurobi` errors if Gurobi is unavailable; `:mosek` is always
honored.
"""
function lp_engine()
    e = _ENGINE[]
    if e === :gurobi
        gurobi_available() || error("--engine gurobi requested but Gurobi is unavailable")
        return :gurobi
    elseif e === :mosek
        return :mosek
    else  # :auto
        return gurobi_available() ? :gurobi : :mosek
    end
end

# --- Model factory ----------------------------------------------------------
"""
    new_model(; nlp=false) -> JuMP.Model

Create a JuMP model on the selected engine. `nlp=true` always uses MadNLP (the
only NLP engine). Otherwise the LP / conic / MIP master uses Gurobi when the
resolved `lp_engine()` is `:gurobi`, else Mosek.
"""
function new_model(; nlp::Bool=false)
    nlp && return Model(MadNLP.Optimizer)
    return lp_engine() === :gurobi ?
           Model(() -> Gurobi.Optimizer(_gurobi_env())) :
           Model(Mosek.Optimizer)
end

"""
    set_lp_barrier!(model)

Force the interior-point (barrier) LP solver. This is a Gurobi-only attribute
(`Method=2`); on Mosek it is a no-op since Mosek already defaults to its
interior-point method for LPs.
"""
set_lp_barrier!(model) = lp_engine() === :gurobi && set_attribute(model, "Method", 2)

# --- Parallel-mode flag -----------------------------------------------------
# Set by run_test.jl when it fits the selected variants concurrently
# (`Threads.@spawn` per variant). Per-method inner threading (the AD good-scan)
# and the module-level linear-separation model cache consult this so they stay
# correct under outer parallelism: inner threading is suppressed (avoid
# oversubscription / nesting) and the shared linear model cache is bypassed
# (each call rebuilds, since one cached Gurobi model cannot back concurrent fits).
const _PARALLEL_VARIANTS = Ref(false)
parallel_variants() = _PARALLEL_VARIANTS[]
set_parallel_variants!(b::Bool) = (_PARALLEL_VARIANTS[] = b)
