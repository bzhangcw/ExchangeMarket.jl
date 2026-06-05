# Shared Gurobi environment singleton.
#
# Used by every Gurobi-backed model in /revealed/ — the master LP
# (redistribute.jl::solve_wealth_redist_primal) and the linear
# MILP separation oracle (linear.jl::solve_separation_linear). One env across
# all calls means:
#   - the academic-license banner prints exactly ONCE at script load,
#     not once per CG iteration,
#   - env-level OutputFlag is set so each per-model `set_attribute(md, …)`
#     call doesn't echo "Set parameter X to value Y" mid-iteration,
#   - per-model logging is still controllable via the model's own attrs
#     (e.g. `set_attribute(model, "LogToConsole", 0)`).
#
# Included FIRST from setup.jl so both redistribute.jl and the per-class
# separation files see `_gurobi_env()`.

using Gurobi

const _GRB_ENV = Ref{Gurobi.Env}()
# Set once we've discovered the license is unusable (expired / missing), so we
# don't retry Gurobi.Env() — and its multi-second timeout — on every call.
const _GRB_ENV_FAILED = Ref{Bool}(false)

function _gurobi_env()
    if _GRB_ENV_FAILED[]
        return nothing
    end
    if !isassigned(_GRB_ENV)
        try
            _GRB_ENV[] = Gurobi.Env()
            Gurobi.GRBsetintparam(_GRB_ENV[], "OutputFlag", 0)
        catch err
            # License expired / not found / token unavailable. Don't crash the
            # whole run at load time — mark Gurobi unusable and let callers fall
            # back to another solver (or skip Gurobi-only paths).
            _GRB_ENV_FAILED[] = true
            @warn "Gurobi unavailable; continuing without it" exception = (err, catch_backtrace())
            return nothing
        end
    end
    return _GRB_ENV[]
end

# Force the env (and thus the license banner) to fire at script load,
# not mid-CG-run. Without this, the banner appears the first time the
# env is actually used — which can interrupt the iteration table several
# iterations into a run.
_gurobi_env()
