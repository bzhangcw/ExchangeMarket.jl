# -----------------------------------------------------------------------
# CES surrogate validation via price-scaled excess demand.
#
# Given a fitted surrogate FisherMarket `fa_surrogate` (output of CG /
# cgma / FW.jl on revealed-preference data) and the underlying real
# CES FisherMarket `f_real`:
#
#   1. solve the surrogate market for its Walrasian price p_s via
#      Mirror Descent (CESAnalytic best-reply, EG step, cc13 stepsize);
#   2. plug p_s into the real market and measure the price-scaled
#      excess demand  z(p_s) = p_s · (q − g_real(p_s))
#      in L∞ (and L1) — this is the dollar value of the imbalance per
#      good when the surrogate's equilibrium price is forced on reality.
#
# Note: the real market's own equilibrium price is not needed — at any
# Walrasian p*, z(p*) ≡ 0 by definition, so the metric reduces to
# "how badly does p_s clear the real market?".
#
# PLC / QL real markets are not FisherMarkets and are skipped here.
# -----------------------------------------------------------------------

using LinearAlgebra
using JuMP, MosekTools
using ExchangeMarket

# -----------------------------------------------------------------------
# Solve a CES FisherMarket for its equilibrium price. Primary solver is the
# Hessian-barrier Newton method (affine-scaling step, normal μ, CESAnalytic
# best-response) — fast, ~20 iterations to tol 1e-12. If the Newton solve errors
# or returns a non-finite price (a surrogate with very few androids / extreme σ
# can have a near-singular Hessian), it falls back to the Mosek convex program:
# the CES Eisenberg–Gale / NSW solver (`solve_ces_welfare_opt`), whose
# supply-constraint dual IS the Walrasian price. Returns a simplex-normalized
# price vector.
# -----------------------------------------------------------------------
function solve_ces_equilibrium(
    fa::FisherMarket;
    maxiter::Int=20,
    maxtime::Float64=120.0,
    tol::Float64=1e-12,
    p₀::Union{Vector{Float64},Nothing}=nothing,
    verbose::Bool=false,
    kwargs...,                      # absorb stray solver opts (e.g. legacy α)
)
    n, m = fa.n, fa.m
    p_init = isnothing(p₀) ? ones(n) ./ n : copy(p₀)
    # Warm-start allocation: HessianBar.init! (Phase I) seeds the initial bids
    # from `market.x`, so give it a uniform feasible split (Σ_i x_ij = 1 per good)
    # rather than whatever stale allocation the surrogate carried in.
    fa.x .= ones(n, m) ./ m
    # Run `alg` to convergence and read off the normalized price (silencing the
    # solver's iteration log unless verbose). Shared by both solvers.
    _run(alg, mi, mt, tl) = begin
        if verbose
            opt!(alg, fa; maxiter=mi, maxtime=mt, tol=tl, reset=true)
        else
            redirect_stdout(devnull) do
                opt!(alg, fa; maxiter=mi, maxtime=mt, tol=tl, reset=true)
            end
        end
        p = copy(alg.p)
        p ./= sum(p)
        return (p=p, iters=alg.k, dual=alg.φ, grad_norm=alg.gₙ)
    end
    # Primary: Hessian-barrier Newton.
    try
        hb = HessianBar(
            n, m, p_init;
            maxiter=maxiter, maxtime=maxtime, tol=tol,
            optimizer=CESAnalytic,
            option_mu=:normal,
            option_step=:affinesc,
            linsys=:DRq,
        )
        res = _run(hb, maxiter, maxtime, tol)
        (all(isfinite, res.p) && sum(res.p) > 0) && return res
        verbose && @warn "solve_ces_equilibrium: HessianBar gave a non-finite price; falling back to the Mosek EG program"
    catch err
        verbose && @warn "solve_ces_equilibrium: HessianBar failed; falling back to the Mosek EG program" err
    end
    # Fallback: the Mosek CES Eisenberg–Gale convex program (same solver as the
    # optimal-NSW computation). Its supply-constraint dual is the Walrasian price,
    # already simplex-normalized by solve_ces_welfare_opt.
    eg = solve_ces_welfare_opt(fa, fa.w; verbose=verbose)
    p = copy(eg.price)
    sp = sum(p)
    sp > 0 && (p ./= sp)
    return (p=p, iters=0, dual=eg.welfare, grad_norm=NaN)
end

# -----------------------------------------------------------------------
# Aggregate (over agents) CES demand g(p) = Σ_i x_i(p), where
# x_{i,j}(p) = w_i γ_{i,j}(p) / p_j and γ is the CES spending share.
# -----------------------------------------------------------------------
function aggregate_ces_demand(fa::FisherMarket, p::AbstractVector)
    n = length(p)
    g = zeros(n)
    for i in 1:fa.m
        c_i = Vector(fa.c[:, i])
        γ_i = compute_gamma(p, c_i, fa.σ[i])
        g .+= fa.w[i] .* γ_i ./ p
    end
    return g
end

# -----------------------------------------------------------------------
# Real-market aggregate demand at price p with a price-dependent wealth
# function: g(p) = Σ_i wealth_fn(p)[i] γ_i(p) / p. For constant wealth
# (wealth_fn(p) = w0) this equals aggregate_ces_demand; for first-order
# (Arrow–Debreu) wealth (wealth_fn(p) = Bᵀp) it is the endowment-valued
# demand. Used by validate_surrogate so the real market's demand at the
# surrogate's equilibrium price honors the ground-truth wealth model.
# -----------------------------------------------------------------------
function real_demand_at(fa::FisherMarket, wealth_fn, p::AbstractVector)
    w = wealth_fn(p)
    g = zeros(length(p))
    for i in 1:fa.m
        γ_i = compute_gamma(p, Vector(fa.c[:, i]), fa.σ[i])
        g .+= w[i] .* γ_i ./ p
    end
    return g
end

# -----------------------------------------------------------------------
# Proportional rationing of price-taking demands to supply, so the allocation
# CLEARS (Σ_i x_ij = s_j) before welfare is measured. `D[i]` is agent i's demand
# vector; x_ij = s_j · d_ij / Σ_i d_ij (0 where a good has no demand). Reduces to
# the raw demand when it already clears (e.g. at equilibrium). This is what makes
# the welfare-at-price W_real(p^s) a FEASIBLE-allocation value, comparable to the
# optimum W_real*: at an off-equilibrium p the raw demand does not clear, and
# scoring welfare on the unrationed (infeasible, oversupplied) bundle overstates
# it. The clearance/excess metrics are computed from the raw demand elsewhere and
# are deliberately left unchanged.
# -----------------------------------------------------------------------
function ration_to_clear(D::Vector{<:AbstractVector}, supply::AbstractVector)
    m = length(D)
    n = length(supply)
    g = zeros(n)
    for i in 1:m, j in 1:n
        g[j] += D[i][j]
    end
    return [[g[j] > 0.0 ? supply[j] * D[i][j] / g[j] : 0.0 for j in 1:n] for i in 1:m]
end

# Clearance check for an allocation `x` (vector of per-agent bundles): the
# largest fraction of any good's supply that the allocation uses,
# max_j (Σ_i x_ij / s_j). ≤ 1 means no good is oversubscribed; a rationed
# allocation hits exactly 1 on every demanded good. Surfaced in the NSW table so
# one can confirm the welfare is scored on a supply-feasible (clearing) bundle.
_supply_use(x::Vector{<:AbstractVector}, supply::AbstractVector) =
    maximum(sum(x[i][j] for i in eachindex(x)) / supply[j] for j in eachindex(supply))

# -----------------------------------------------------------------------
# Nash social welfare (NSW)  Σ_i w_i log u_i  of a CES Fisher market at the
# CLEARING allocation induced by price `p`: each agent's price-taking demand
# x_i = w_i γ_i(p) ⊘ p is proportionally rationed to unit supply
# (`ration_to_clear`) so the bundle is feasible before utilities are taken.
# Each CES utility u_i = (Σ_j c_ij x_ij^{ρ_i})^{1/ρ_i} (⟨c_i,x_i⟩ for a linear
# atom) mirrors `utility(::CESAgent, c, x)`. `w` is explicit because the
# ground-truth wealth may be price dependent (w_i = wealth_fn(p)).
# -----------------------------------------------------------------------
function ces_welfare(fa::FisherMarket, p::AbstractVector, w::AbstractVector)
    n = length(p)
    D = [w[i] .* compute_gamma(p, Vector(fa.c[:, i]), fa.σ[i]) ./ p for i in 1:fa.m]
    return ces_welfare_at(fa, w, ration_to_clear(D, ones(n)))
end

# Cardinal utility of a GES / NGES agent at bundle `x` (concave additive, r_j ∈
# (0,1)): GES u = Σ_j c_j x_j^{r_j}; NGES u = Σ_j c_j (A x)_j^{r_j}. Safe-power.
function _real_agent_u(a::GESAgent, x::AbstractVector)
    s = 0.0
    @inbounds for j in 1:a.n
        xj = x[j]
        s += a.c[j] * (xj > 0.0 ? xj^a.r[j] : 0.0)
    end
    return s
end
function _real_agent_u(a::NGESAgent, x::AbstractVector)
    z = a.A * x
    s = 0.0
    @inbounds for j in 1:a.n
        zj = z[j]
        s += a.c[j] * (zj > 0.0 ? zj^a.r[j] : 0.0)
    end
    return s
end

# Σ_i w_i log u_i(x_i) for a GES / NGES real market at an explicit allocation `x`,
# NaN-safe (mirrors ces_welfare_at). Used for W_real(p^s) on the rationed bundle.
function real_welfare_at(agents, w::AbstractVector, x::Vector{<:AbstractVector})
    W = 0.0
    for i in eachindex(agents)
        u = _real_agent_u(agents[i], x[i])
        isnan(u) && return NaN
        W += w[i] * (u > 0.0 ? log(u) : 0.0)
    end
    return W
end

# Σ_i w_i log u_i(x_i) for a CES market at an explicit allocation `x`
# (vector of per-agent bundles), with safe-power / NaN-safe log.
function ces_welfare_at(fa::FisherMarket, w::AbstractVector, x::Vector{<:AbstractVector})
    spow(a, b) = a > 0.0 ? a^b : 0.0
    W = 0.0
    for i in 1:fa.m
        c_i = Vector(fa.c[:, i])
        ρ = isinf(fa.σ[i]) ? 1.0 : fa.ρ[i]   # linear atom (σ=∞) ⇒ ρ=1 ⇒ u=⟨c,x⟩
        s = 0.0
        @inbounds for j in eachindex(x[i])
            s += c_i[j] * spow(x[i][j], ρ)
        end
        u_i = spow(s, 1.0 / ρ)
        W += w[i] * (u_i > 0.0 ? log(u_i) : 0.0)
    end
    return W
end

# -----------------------------------------------------------------------
# Project the surrogate's equilibrium price onto the real market's good
# space. When the surrogate was fit on money-lifted data it has n+1 goods
# (the extra one is money); the original n-good price is q = p/π = the
# goods prices in units of money (lem.lift). For an un-lifted surrogate
# (same n as the real market) this is the identity.
# -----------------------------------------------------------------------
function _project_to_real(p_s::AbstractVector, n_real::Int)
    if length(p_s) == n_real + 1
        π = p_s[end]
        q = p_s[1:n_real] ./ π                 # q* = p*/π*
        @printf("  [lift] money-lifted surrogate (%d goods): projecting q*=p*/π* onto %d goods (π*=%.4g, goods scale ⟨1,q*⟩=%.4g)\n",
            length(p_s), n_real, π, sum(q))
        return q
    else
        return p_s
    end
end

# -----------------------------------------------------------------------
# Price-scaled excess of the real market at the surrogate's equilibrium price.
# `d_oracle(p) -> ℝ^{n_real}` is the real market's goods demand at price p.
# Returns `(orig, lift)`:
#
# * `orig` — the ORIGINAL n-good excess `q ⊙ (1 − d(q))`, where `q` is the
#   surrogate price projected onto the real goods (`q = p̄[1:n]/π` if lifted,
#   else `q = p`). This is the "does the (projected) price clear the original
#   market" check (prop.lift.equilibrium).
#
# * `lift` — the LIFTED (n+1)-good excess when the surrogate is money-lifted,
#   else `nothing`. It scores against the lifted real demand
#   `d̄(p̄) = (d(q), M + ⟨q,1⟩ − W(q))` and supply `(1,…,1,M)`:
#   `z̄ = p̄ ⊙ ((1,…,1,M) − d̄)`. The money supply `M` cancels, leaving
#   goods `p̄_{1:n} ⊙ (1 − d(q))` and money `π (W(q) − ⟨q,1⟩)`, `W(q)=⟨q,d(q)⟩`.
#
# * `norm` — the ORIGINAL n-good excess evaluated at the projected price put
#   back on the simplex, `q̃ ⊙ (1 − d(q̃))` with `q̃ = q/⟨1,q⟩`. The bare `orig`
#   carries price units (`q = p̄[1:n]/π` is off-simplex, scale `⟨1,q⟩ ≠ 1`),
#   so its magnitude is not comparable across lift/no-lift; `norm` strips that
#   `1/π` re-scaling by fixing `⟨1,q̃⟩ = 1`, making the lifted and un-lifted
#   clearing residuals directly comparable.
# -----------------------------------------------------------------------
function _real_excess(d_oracle, p_s::AbstractVector, n_real::Int)
    if length(p_s) == n_real + 1
        π = p_s[end]
        q = p_s[1:n_real] ./ π
        d = d_oracle(q)
        orig = q .* (1.0 .- d)
        lift = vcat(p_s[1:n_real] .* (1.0 .- d), π * (dot(q, d) - sum(q)))
        q̃ = q ./ sum(q)
        d̃ = d_oracle(q̃)
        nrm = q̃ .* (1.0 .- d̃)
        return (orig=orig, lift=lift, norm=nrm)
    else
        d = d_oracle(p_s)
        q̃ = p_s ./ sum(p_s)
        d̃ = d_oracle(q̃)
        nrm = q̃ .* (1.0 .- d̃)
        return (orig=p_s .* (1.0 .- d), lift=nothing, norm=nrm)
    end
end

# Surrogate self-excess: the price-scaled residual of the SURROGATE market at
# its own equilibrium price `p` (the solver's convergence residual). ≈0 confirms
# the equilibrium was actually reached, so a large real-market excess is model
# misspecification rather than an unsolved equilibrium. For a Fisher (CES)
# surrogate it is ‖p ⊙ (1 − d_surr(p))‖∞; for an AD surrogate the solver returns
# it directly (passed in via `surr_excess`).
_surr_self_excess(fa::FisherMarket, p) = norm(p .* (1.0 .- aggregate_ces_demand(fa, p)), Inf)

# Build the validation result NamedTuple from the (orig, lift, norm) excess.
# `welfare_*` are the Nash social welfares (NSW) Σ w_i log u_i (NaN when
# not computed, e.g. non-CES ground truth): `welfare_real_ps` for the real
# market at the surrogate price p^s, `welfare_surr_ps` for the surrogate's own
# androids at p^s. (The real optimum Σ w_i log u_i(p*) is solved once by the
# driver, not per surrogate variant.)
function _vresult(p_surr, ex; iters::Int=0, surr_excess::Real=NaN,
    welfare_real_ps::Real=NaN, welfare_surr_ps::Real=NaN, nsw_supply_use::Real=NaN)
    nrm = (hasproperty(ex, :norm) && !isnothing(ex.norm)) ? ex.norm : ex.orig
    return (
        p_surrogate=p_surr,
        excess_surrogate_linf=norm(ex.orig, Inf),
        excess_surrogate_l1=norm(ex.orig, 1),
        excess_lift_linf=isnothing(ex.lift) ? NaN : norm(ex.lift, Inf),
        excess_lift_l1=isnothing(ex.lift) ? NaN : norm(ex.lift, 1),
        excess_norm_linf=norm(nrm, Inf),
        excess_norm_l1=norm(nrm, 1),
        excess_surr_self_linf=surr_excess,
        iters_surrogate=iters,
        welfare_real_ps=welfare_real_ps,
        welfare_surr_ps=welfare_surr_ps,
        # max_j Σ_i x_ij / s_j of the rationed real allocation NSW is scored on
        # (≤ 1 ⇒ supply not exceeded). NaN when welfare isn't computed.
        nsw_supply_use=nsw_supply_use,
    )
end

# -----------------------------------------------------------------------
# End-to-end validation: solve surrogate equilibrium, project to the real
# market's good space if the surrogate is money-lifted, evaluate the real
# market's aggregate demand there (honoring the ground-truth wealth model
# via `wealth_fn`), and return ‖q(1 − g)‖.
#
# Fields:
#   p_surrogate           : (projected) price on the real market's goods
#   excess_surrogate_linf : ‖q · (1 − g_real(q))‖_∞   ← main metric
#   excess_surrogate_l1   : ‖q · (1 − g_real(q))‖_1
#   iters_surrogate       : Mirror Descent iterations used
# -----------------------------------------------------------------------
function validate_surrogate(
    fa_surrogate::FisherMarket, f_real::FisherMarket;
    wealth_fn=nothing, verbose::Bool=false, kwargs...
)
    @assert fa_surrogate.n == f_real.n || fa_surrogate.n == f_real.n + 1 "surrogate must match the real market's good count, or have one extra (money-lifted)"
    # The equilibrium solvers (HessianBar / CESAnalytic best-response, and the
    # CES EG fallback) only support CES agents. If the surrogate carries any
    # non-CES (gen) agents (QL, ...), the equilibrium solve is undefined here;
    # warn once and return a NaN-shaped result so callers don't have to wrap
    # every call in try/catch.
    if fa_surrogate.storage.gen.m > 0
        @warn "validate_surrogate: surrogate has $(fa_surrogate.storage.gen.m) non-CES (gen) agent(s); equilibrium ops are CES-only and not supported for mixed surrogates. Returning NaN excess." maxlog = 1
        return _vresult(fill(NaN, f_real.n), (orig=[NaN], lift=nothing))
    end
    res_s = solve_ces_equilibrium(fa_surrogate; verbose=verbose, kwargs...)
    q = _project_to_real(res_s.p, f_real.n)         # surrogate price on the real goods
    oracle = isnothing(wealth_fn) ? (p -> aggregate_ces_demand(f_real, p)) :
             (p -> real_demand_at(f_real, wealth_fn, p))
    ex = _real_excess(oracle, res_s.p, f_real.n)
    # Nash social welfare (NSW) Σ w_i log u_i at the surrogate equilibrium price,
    # both evaluated on the CLEARING (supply-rationed) allocation via ces_welfare:
    #   * real market, with budgets honoring the ground-truth wealth model at q;
    #   * surrogate's own androids, in the surrogate's good space (n+1 if lifted).
    w_real_q = isnothing(wealth_fn) ? f_real.w : wealth_fn(q)
    # Real-market welfare on the CLEARING allocation: ration the price-taking
    # demands to unit supply, then score. `nsw_supply_use` = max_j Σ_i x_ij is
    # the clearance check (≤ 1 ⇒ supply not exceeded).
    D_real = [w_real_q[i] .* compute_gamma(q, Vector(f_real.c[:, i]), f_real.σ[i]) ./ q
              for i in 1:f_real.m]
    X_real = ration_to_clear(D_real, ones(f_real.n))
    welfare_real_ps = ces_welfare_at(f_real, w_real_q, X_real)
    welfare_surr_ps = ces_welfare(fa_surrogate, res_s.p, fa_surrogate.w)
    return _vresult(q, ex; iters=res_s.iters,
        surr_excess=_surr_self_excess(fa_surrogate, res_s.p),
        welfare_real_ps=welfare_real_ps, welfare_surr_ps=welfare_surr_ps,
        nsw_supply_use=_supply_use(X_real, ones(f_real.n)))
end

# `solve_plc_excess` (the PLC joint-LP equilibrium check used by the PLC branch
# of `validate_surrogate`) lives in `androids/plc.jl` with the rest of the PLC
# machinery; it is loaded by the drivers before this file.

# -----------------------------------------------------------------------
# Generic real-market validation for the non-CES ground-truth families,
# given as the NamedTuple `(agents=..., w=...)` / `(agents=..., b=...)` that
# build_rep_data produces (PLC, GES, SPLC, NGES). The recipe is uniform:
# solve the surrogate's equilibrium price (Fisher via Mirror Descent, AD via
# potred), project it back to the real goods if the surrogate is money-lifted,
# and plug it into the real market's aggregate demand. Demand is
# `aggregate_real_demand` (share / closed-form per class) for the single-valued
# families; PLC keeps its set-valued joint-LP `τ` check.
# -----------------------------------------------------------------------
function validate_surrogate(
    fa_surrogate, real::NamedTuple;
    wealth_fn=nothing, verbose::Bool=false, kwargs...
)
    # --- surrogate equilibrium price (dispatch on surrogate type) ---
    nan_result(n) = _vresult(fill(NaN, n), (orig=[NaN], lift=nothing))
    if fa_surrogate isa FisherMarket
        if fa_surrogate.storage.gen.m > 0
            @warn "validate_surrogate: mixed (non-CES) surrogate; equilibrium ops are CES-only. Returning NaN." maxlog = 1
            return nan_result(first(real.agents).n)
        end
        p_s = solve_ces_equilibrium(fa_surrogate; verbose=verbose, kwargs...).p
        surr_excess = _surr_self_excess(fa_surrogate, p_s)
    else  # ArrowDebreuMarket — potred Newton needs a finite Hessian
        if any(!isfinite, fa_surrogate.σ)
            @warn "validate_surrogate(AD): linear atom(s) (σ=∞); potred solve undefined. Returning NaN." maxlog = 1
            return nan_result(first(real.agents).n)
        end
        res_s = afscaled_newton_equilibrium(fa_surrogate; verbose=verbose)
        p_s = res_s.p
        surr_excess = res_s.resid     # solver's own market-clearing residual
    end

    # --- PLC ground truth: set-valued demand → joint-LP τ check ---
    if eltype(real.agents) <: PLCAgent
        @assert haskey(real, :w) "PLC validation expects Fisher budgets or a wealth function (agents, w)"
        # `w` may be fixed budgets or a price-dependent wealth function
        # (--wealth-function 2); resolve it at the surrogate equilibrium price.
        w_eval = wealth_at(real.w, p_s)
        plc_res = solve_plc_excess(p_s, real.agents, w_eval; verbose=verbose)
        # Nash welfare at the surrogate price p_s: plc_res.x are the real agents'
        # UMP-optimal demands at p_s; proportionally rationed to unit supply so
        # the allocation CLEARS before utilities are taken (W_real(p^s) must be a
        # feasible value to compare against the optimum W_real*). W_surr(p^s) is
        # the surrogate's own CES welfare (itself clearing-rationed); NaN for a
        # non-CES / AD surrogate. The optimum W_real* is solved once by the driver
        # (solve_plc_welfare_opt).
        X_real = ration_to_clear(plc_res.x, ones(length(p_s)))
        welfare_real_ps = plc_welfare(real.agents, w_eval, X_real)
        welfare_surr_ps = fa_surrogate isa FisherMarket ?
                          ces_welfare(fa_surrogate, p_s, fa_surrogate.w) : NaN
        return (
            p_surrogate=p_s,
            excess_surrogate_linf=plc_res.tau,
            excess_surrogate_l1=sum(abs.(p_s .* (1.0 .- sum(plc_res.x)))),
            excess_lift_linf=NaN, excess_lift_l1=NaN,   # PLC is never lifted
            excess_norm_linf=plc_res.tau,               # never lifted ⇒ norm ≡ orig (τ)
            excess_norm_l1=sum(abs.(p_s .* (1.0 .- sum(plc_res.x)))),
            excess_surr_self_linf=surr_excess,
            iters_surrogate=0,
            welfare_real_ps=welfare_real_ps,
            welfare_surr_ps=welfare_surr_ps,
            nsw_supply_use=_supply_use(X_real, ones(length(p_s))),
            lp_status=plc_res.status,
            lp_time=plc_res.time,
        )
    end

    # --- single-valued families (GES/SPLC/NGES): both the original n-good and
    #     the lifted (n+1)-good excess (the latter only when money-lifted) ---
    budgets = haskey(real, :w) ? real.w : real.b
    n_real = first(real.agents).n
    ex = _real_excess(p -> aggregate_real_demand(real.agents, budgets, p), p_s, n_real)
    q = _project_to_real(p_s, n_real)
    # GES / NGES NSW at the surrogate's clearing price: per-agent demand at q,
    # rationed to unit supply, scored with the concave additive utility (W_real(p^s)
    # must be a feasible/clearing value to compare against the optimum W_real*).
    # SPLC has no closed-form cardinal utility wired here, so its welfare stays NaN.
    if eltype(real.agents) <: GESAgent || eltype(real.agents) <: NGESAgent
        w_eval = wealth_at(budgets, q)
        D = [agent_demand(real.agents[i], q, w_eval[i]) for i in eachindex(real.agents)]
        X = ration_to_clear(D, ones(n_real))
        return _vresult(q, ex; surr_excess=surr_excess,
            welfare_real_ps=real_welfare_at(real.agents, w_eval, X),
            nsw_supply_use=_supply_use(X, ones(n_real)))
    end
    return _vresult(q, ex; surr_excess=surr_excess)
end
