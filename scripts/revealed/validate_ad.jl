# -----------------------------------------------------------------------
# Arrow–Debreu surrogate validation.
#
# adcg / adfw fit an ArrowDebreuMarket surrogate. To score it like a Fisher
# surrogate, solve its Walrasian equilibrium with the affine-scaled Newton
# solver (`afscaled_newton_equilibrium`, from the self-contained arrow_debreu.jl),
# project the price onto the real market's goods if the surrogate is
# money-lifted, then evaluate the real market's demand there (honoring the
# ground-truth wealth model via `wealth_fn`) and report the price-scaled
# excess — identical metric/return shape to the Fisher `validate_surrogate`.
#
# `arrow_debreu.jl` is a frozen, self-contained snapshot of the arrow solver
# (Mosek SOCP subproblem) so revealed/ has no arrow dependency.
# `_project_to_real`, `real_demand_at`, and `aggregate_ces_demand` come from
# validate.jl (included before this file).
# -----------------------------------------------------------------------

include(joinpath(@__DIR__, "arrow_debreu.jl"))   # afscaled_newton_equilibrium (self-contained, no arrow dep)

function validate_surrogate(
    fa_surrogate::ArrowDebreuMarket, f_real::FisherMarket;
    wealth_fn=nothing, verbose::Bool=false, kwargs...
)
    @assert fa_surrogate.n == f_real.n || fa_surrogate.n == f_real.n + 1 "AD surrogate must match the real market's good count, or have one extra (money-lifted)"
    # The potred Newton solve needs a finite AD Hessian; a linear atom (ρ=1,
    # σ=∞) makes __ces_compute_exact_hess! diverge. Skip gracefully (NaN) rather
    # than crash on a LAPACK error when the surrogate carries linear atoms.
    if any(!isfinite, fa_surrogate.σ)
        @warn "validate_surrogate(AD): surrogate has linear atom(s) (σ=∞); the potred Newton solve is undefined for them. Returning NaN excess." maxlog = 1
        return _vresult(fill(NaN, f_real.n), (orig=[NaN], lift=nothing))
    end
    res_s = afscaled_newton_equilibrium(fa_surrogate; verbose=verbose, kwargs...)
    q = _project_to_real(res_s.p, f_real.n)        # surrogate price on the real goods
    oracle = isnothing(wealth_fn) ? (p -> aggregate_ces_demand(f_real, p)) :
             (p -> real_demand_at(f_real, wealth_fn, p))
    ex = _real_excess(oracle, res_s.p, f_real.n)
    # Real-market NSW at the surrogate's clearing price, on the CLEARING
    # (supply-rationed) allocation — same as the Fisher validate_surrogate, so the
    # NSW table is populated for adcg / adfw too. The surrogate is an
    # ArrowDebreuMarket (not a CES Fisher), so its own W_surr(p^s) is left NaN.
    w_real_q = isnothing(wealth_fn) ? f_real.w : wealth_fn(q)
    D_real = [w_real_q[i] .* compute_gamma(q, Vector(f_real.c[:, i]), f_real.σ[i]) ./ q
              for i in 1:f_real.m]
    X_real = ration_to_clear(D_real, ones(f_real.n))
    return _vresult(q, ex; surr_excess=res_s.resid,
        welfare_real_ps=ces_welfare_at(f_real, w_real_q, X_real),
        nsw_supply_use=_supply_use(X_real, ones(f_real.n)))
end
