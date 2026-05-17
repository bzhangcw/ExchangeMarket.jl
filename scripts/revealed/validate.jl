# -----------------------------------------------------------------------
# CES surrogate validation via price-scaled excess demand.
#
# Given a fitted surrogate FisherMarket `fa_surrogate` (output of CG /
# MultiCut / FW.jl on revealed-preference data) and the underlying real
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
# Solve a CES FisherMarket for its equilibrium price using Mirror Descent
# (exponentiated-gradient step, Cole-Cheung '13 step size). Returns a
# simplex-normalized price vector. We use MirrorDec rather than the
# Newton-based HessianBar because the surrogate market can have very
# few atoms with extreme σ values, where the Hessian becomes singular;
# MirrorDec is slower but globally robust.
# -----------------------------------------------------------------------
function solve_ces_equilibrium(
    fa::FisherMarket;
    α::Float64=500.0,
    maxiter::Int=10000,
    maxtime::Float64=120.0,
    tol::Float64=1e-10,
    p₀::Union{Vector{Float64},Nothing}=nothing,
    verbose::Bool=false,
)
    n, m = fa.n, fa.m
    p_init = isnothing(p₀) ? ones(n) ./ n : copy(p₀)
    alg = MirrorDec(
        n, m, p_init;
        α=α, maxiter=maxiter, maxtime=maxtime, tol=tol,
        optimizer=CESAnalytic,
        option_step=:eg, option_stepsize=:cc13,
    )
    if verbose
        opt!(alg, fa; maxiter=maxiter, maxtime=maxtime, tol=tol, reset=true)
    else
        redirect_stdout(devnull) do
            opt!(alg, fa; maxiter=maxiter, maxtime=maxtime, tol=tol, reset=true)
        end
    end
    p = copy(alg.p)
    p ./= sum(p)
    return (p=p, iters=alg.k, dual=alg.φ, grad_norm=alg.gₙ)
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
# End-to-end validation: solve surrogate equilibrium, evaluate real
# market's aggregate demand at that price, return ‖p(q-g)‖.
#
# Fields:
#   p_surrogate           : simplex-normalized surrogate equilibrium price
#   excess_surrogate_linf : ‖p_s · (q − g_real(p_s))‖_∞   ← main metric
#   excess_surrogate_l1   : ‖p_s · (q − g_real(p_s))‖_1
#   iters_surrogate       : Mirror Descent iterations used
# -----------------------------------------------------------------------
function validate_surrogate(
    fa_surrogate::FisherMarket, f_real::FisherMarket;
    verbose::Bool=false, kwargs...
)
    @assert fa_surrogate.n == f_real.n
    "surrogate and real must have the same number of goods"
    res_s = solve_ces_equilibrium(fa_surrogate; verbose=verbose, kwargs...)
    p_s = res_s.p
    q = Vector(f_real.q)
    g_at_s = aggregate_ces_demand(f_real, p_s)
    z_s = p_s .* (q .- g_at_s)
    return (
        p_surrogate=p_s,
        excess_surrogate_linf=norm(z_s, Inf),
        excess_surrogate_l1=norm(z_s, 1),
        iters_surrogate=res_s.iters,
    )
end

# -----------------------------------------------------------------------
# PLC market excess at a candidate price (single all-in-one LP from
# `eq.plc.we.check.lp` in `read-econ/choice-ump-plc.tex`).
#
# Given a price `p` and PLC agents `(agents[i], w[i])` with unit supply
# q = 1, this solves the joint LP
#
#     min τ
#     s.t. UMP primal-dual + strong duality for every agent i,
#          -τ 1 ≤ diag(p) (1 − Σ_i x_i) ≤ τ 1, τ ≥ 0.
#
# Returns (tau=τ*, x=selection, status=termination_status, time=wall).
# τ* == 0 iff p is a Walrasian equilibrium of the PLC market.
# -----------------------------------------------------------------------
function solve_plc_excess(p::AbstractVector, agents::Vector{<:PLCAgent},
    w::AbstractVector; verbose::Bool=false, timelimit::Union{Real,Nothing}=nothing)
    m = length(agents)
    n = length(p)
    @assert length(w) == m "budget vector length mismatch"
    @assert all(a -> size(a.a, 2) == n, agents) "agent gradient widths must equal n"
    # NaN/Inf in the surrogate equilibrium price would make the LP coefficients
    # invalid; fail soft so the run produces a row instead of crashing.
    if any(!isfinite, p) || any(p .<= 0)
        @warn "solve_plc_excess: non-finite or non-positive price; skipping LP" p_minmax=extrema(p)
        return (tau=NaN, x=[fill(NaN, n) for _ in 1:m], status=:bad_price, time=0.0)
    end

    model = Model(Mosek.Optimizer)
    verbose || set_silent(model)
    if !isnothing(timelimit) && timelimit > 0
        set_time_limit_sec(model, Float64(timelimit))
    end

    # Per-agent decision variables.
    x = [@variable(model, [1:n], lower_bound = 0.0, base_name = "x_$(i)") for i in 1:m]
    t = [@variable(model, base_name = "t_$(i)") for i in 1:m]
    U = [@variable(model, [1:agents[i].L], lower_bound = 0.0, base_name = "U_$(i)") for i in 1:m]
    ν = [@variable(model, [1:n], lower_bound = 0.0, base_name = "nu_$(i)") for i in 1:m]
    μ = [@variable(model, lower_bound = 0.0, base_name = "mu_$(i)") for i in 1:m]
    @variable(model, τ >= 0.0)

    for i in 1:m
        ai = agents[i]
        A_i = ai.a               # L × n
        b_i = ai.b               # L
        L_i = ai.L
        # UMP primal: A_i x_i + b_i .≥ t_i 1, ⟨p, x_i⟩ = w_i.
        @constraint(model, A_i * x[i] .+ b_i .>= t[i] .* ones(L_i))
        @constraint(model, dot(p, x[i]) == w[i])
        # UMP dual: A_i' U_i + ν_i = μ_i p, ⟨1, U_i⟩ = 1.
        @constraint(model, A_i' * U[i] .+ ν[i] .== μ[i] .* p)
        @constraint(model, sum(U[i]) == 1.0)
        # Strong duality: t_i = ⟨b_i, U_i⟩ + w_i μ_i.
        @constraint(model, t[i] == dot(b_i, U[i]) + w[i] * μ[i])
    end

    # Market clearing: -τ 1 ≤ diag(p) (1 - Σ x_i) ≤ τ 1.
    aggregate = sum(x[i] for i in 1:m)
    for j in 1:n
        @constraint(model, p[j] * (1.0 - aggregate[j]) <= τ)
        @constraint(model, p[j] * (1.0 - aggregate[j]) >= -τ)
    end

    @objective(model, Min, τ)
    t_start = time()
    optimize!(model)
    elapsed = time() - t_start

    status = termination_status(model)
    if status ∉ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.SLOW_PROGRESS, MOI.TIME_LIMIT)
        @warn "PLC excess LP terminated abnormally" status
        return (tau=NaN, x=[fill(NaN, n) for _ in 1:m], status=status, time=elapsed)
    end
    return (
        tau = value(τ),
        x   = [value.(x[i]) for i in 1:m],
        status = status,
        time = elapsed,
    )
end

# -----------------------------------------------------------------------
# Dispatch on PLC ground truth. The `real_plc` argument is the NamedTuple
# `(agents=..., w=...)` produced by `build_rep_data` for `--market-type plc`.
# Solves the surrogate CES equilibrium, then plugs that price into the
# real PLC market's joint LP.
# -----------------------------------------------------------------------
function validate_surrogate(
    fa_surrogate::FisherMarket, real_plc::NamedTuple{(:agents, :w)};
    verbose::Bool=false, kwargs...
)
    res_s = solve_ces_equilibrium(fa_surrogate; verbose=verbose, kwargs...)
    p_s = res_s.p
    plc_res = solve_plc_excess(p_s, real_plc.agents, real_plc.w; verbose=verbose)
    return (
        p_surrogate           = p_s,
        excess_surrogate_linf = plc_res.tau,        # τ* from the joint LP
        excess_surrogate_l1   = sum(abs.(p_s .* (1.0 .- sum(plc_res.x)))),
        iters_surrogate       = res_s.iters,
        lp_status             = plc_res.status,
        lp_time               = plc_res.time,
    )
end
