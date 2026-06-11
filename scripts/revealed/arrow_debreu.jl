# -----------------------------------------------------------------------
# Standalone (frozen) Arrow–Debreu equilibrium solver.
#
# A self-contained copy of the affine-scaled Newton / potential-reduction
# solver from scripts/arrow/func_new.jl (+ its SOCP trust-region subproblem
# from scripts/arrow/func.jl), so that revealed/ does NOT depend on the arrow
# scripts. Frozen: do not refactor; this is a verbatim snapshot whose only
# change is the public entry point's name (`afscaled_newton_equilibrium`,
# formerly `solve_ad_equilibrium`).
#
# Public entry:
#   afscaled_newton_equilibrium(ad; K, η, ϵ, verbose) -> (p=…)
#     Solve an ArrowDebreuMarket's Walrasian equilibrium, returning a
#     simplex-normalized price. Internally builds a HessianBar with a Σp=1
#     constraint and runs `afscaled_newton`.
#
# `proj` (x<0 ? Inf : x) is reused from ExchangeMarket; `printto` is copied
# here (it is not exported). update_budget!/play!/grad!/eval!/hess!,
# LinearConstr and HessianBar come from ExchangeMarket.
# -----------------------------------------------------------------------

using LinearAlgebra, Printf, Random
using JuMP, MosekTools
using ExchangeMarket

# Local copy (ExchangeMarket has an internal printto but does not export it).
_compute_ad_printto(ios, x) = begin
    for io in ios
        println(io, x)
    end
end

# -----------------------------------------------------------------------
# SOCP trust-region subproblem (Mosek). Copied from scripts/arrow/func.jl.
# -----------------------------------------------------------------------
function solve_socp_subp_primal(_htJ, _htz, _g, _α, _M, Δ; bool_verbose=0, mode=1, p=nothing)
    n = length(_htz)
    model = Model(Mosek.Optimizer)
    set_silent(model)
    bool_verbose != 0 && unset_silent(model)

    @variable(model, x[1:n])
    @variable(model, t)
    # t ≥ |x|²
    @constraint(model, [0.5; t; x] in RotatedSecondOrderCone())
    @constraint(model, t <= Δ^2)
    # y = _htJ * x + _htz
    @variable(model, y[1:n])
    @variable(model, δ)
    @constraint(model, _htJ * x + _htz .== y)
    # δ ≥ |y|₂
    @constraint(model, [δ; y] in SecondOrderCone())

    # use log(|I+V|₂) ≤ log(1+\|v\|₂) ≤ \|v\|₂ = xn
    @variable(model, xn)
    @constraint(model, [xn; x] in SecondOrderCone())
    @constraint(model, xn <= Δ)
    if mode == 0
        # xn ≥ |x|₂
    elseif mode == 1
        # ensure sum(x) = β / 4C
    end
    @objective(model, Min, _α * xn + _M * t + δ - _g' * x)
    optimize!(model)
    return value.(x), value(objective_value(model)), value(δ), model
end

function solve_subp_type_ii(_htJ, _htz, _g, _α, _M; Δ=0.9, p=nothing)
    _v = similar(_g)
    _x, _Q, _δ, _... = solve_socp_subp_primal(_htJ, _htz, _g, _α, _M, Δ; p=p)
    _v .= _x
    return _v, _Q, _δ
end

# -----------------------------------------------------------------------
# Affine-scaled Newton / potential-reduction loop. Copied verbatim from
# scripts/arrow/func_new.jl::potred_with_fo_new (renamed `afscaled_newton`).
# -----------------------------------------------------------------------
function afscaled_newton(
    f1, alg;
    K=3000, log_interval=10,
    scaler=1.0,
    η=1e-3,
    ϵ=1e-10,
    Δₘ=0.95,
    ff="/tmp/log.log"
)

    Random.seed!(1)
    n = f1.n
    e = ones(n)
    y = rand(n)
    alg.p .= ones(n) .* scaler ./ n
    @show alg.p

    k = 0

    log_file = open(ff, "w")
    traj_beta = zeros(K)
    traj_pot = zeros(K)
    traj_kappa = zeros(K)
    traj_c = zeros(K)
    traj_p = zeros(n, K)
    _K = K
    _η = η
    bool_break = false
    # potential
    φ(b, p) = log(b) - η * sum(log.(p))
    for k in 1:K
        update_budget!(f1, alg.p)
        play!(alg, f1)
        grad!(alg, f1)
        eval!(alg, f1)
        hess!(alg, f1)
        _J = alg.H
        _z = alg.∇
        _r = sum(_z .^ 2)
        _P = diagm(alg.p)
        _htJ = _P * _J * _P
        _htz = _P * _z

        _β = norm(_htz)
        alg.pb .= alg.p

        # how much _htz is in the direction of u₁
        U, S, V = svd(_htJ)
        u₁ = U[:, end]
        _ratio = (_β / norm(U[:, end]' * _htz))^2

        # use type II subproblem
        _α = _β
        _θ = 0.0
        _M = 0.0
        _k = 0
        _v = similar(_z)
        αₘ = 1.0
        q = similar(alg.p)
        bool_accept = false
        _u = _α
        _ℓ = 0.0
        _φb = φ(_β, alg.pb)
        while true
            if _k > 20 || bool_accept
                break
            end
            _g = zeros(n)
            v, _Q = solve_subp_type_ii(_htJ, _htz, _g, 0.0, max(alg.p...) * 1e-4; Δ=min(_β * 1e3, Δₘ), p=alg.p)
            _nv = norm(v)
            _v .= v
            Δ_acc = 0.1 * _nv^2 / (1 - _nv)
            _compute_ad_printto(
                [log_file],
                "@M: $(@sprintf("%.4e", _M)), |v|: $(@sprintf("%.1e", _nv)), k: $(_k), _α: $(@sprintf("%.4e", _α)), Δ_acc: $(@sprintf("%.1e", Δ_acc)) _θ: $(@sprintf("%.1e", _θ))"
            )
            flush(log_file)
            q .= alg.p .* v
            αₘ = min(proj.(-alg.pb ./ q)..., 0.9)
            alg.p .= alg.pb + αₘ * q
            alg.p .= alg.p ./ maximum(1 .+ v) .* scaler

            _compute_ad_printto([log_file], "before: $(@sprintf("%.1e", tr(I + diagm(v)))) after: $(@sprintf("%.1e", maximum(alg.p ./ alg.pb)))")
            update_budget!(f1, alg.p)
            play!(alg, f1)
            grad!(alg, f1)
            eval!(alg, f1)
            _β₊ = norm(alg.p .* alg.∇)
            _φ₊ = _η * log(_β₊)
            M_est = _β + norm(_htJ * _v + _htz)
            Δβ = _β - _β₊
            Δφ = _φb - _φ₊
            _compute_ad_printto(
                [log_file],
                "@M: $(@sprintf("%.4e", _M)), |v|: $(@sprintf("%.1e", _nv)), k: $(_k), _α: $(@sprintf("%.4e", _α)), Δφ: $(@sprintf("%.1e", Δφ))"
            )

            bool_accept = true

        end
        _nv = norm(_v)

        if !bool_accept
        end

        # predicted |z|_+
        spec = 0.0
        try
            spec = opnorm(_htJ)
        catch
        end

        if _β < ϵ / norm(sum(alg.p))
            bool_break = true
        end

        traj_beta[k] = _β
        traj_kappa[k] = 0.0
        traj_p[:, k] .= alg.p
        _φ = φ(_β, alg.p)
        traj_pot[k] = _φ
        msg1 = "$(k)th iteration |p|₁: $(@sprintf("%.1e", sum(alg.p))) |p|₂: $(@sprintf("%.1e", norm(alg.p))) sum(q): $(@sprintf("%.1e", sum(q)))"
        msg2 = "   |-$(k)th               φ: $(@sprintf("%.3e", _φ))  η: $(@sprintf("%.3e", _η))"
        msg3 = "   |-$(k)th         inner_k: $(_k)"
        msg4 = "   |-$(k)th               p: [$(@sprintf("%.2e", minimum(alg.p))), $(@sprintf("%.2e", maximum(alg.p)))]"
        msg5 = "   |-$(k)th               r: $(@sprintf("%.1e", _r))  λ₁: [$(@sprintf("%.1e", 0.0)), $(@sprintf("%.1e", spec))]"
        msg6 = "   |-$(k)th               α: $(@sprintf("%.1e", _α)) M: $(@sprintf("%.1e", _M)) |v|: $(@sprintf("%.1e", _nv))"
        msg7 = "   |-$(k)th            |z|ₚ: $(@sprintf("%.4e", _β)) -> $(@sprintf("%.4e", _β * norm(sum(alg.p))))"
        msg8 = "   |-$(k)th              αₘ: $(@sprintf("%.1e", αₘ))"
        msg9 = "   |-$(k)th     cond. ratio: $(@sprintf("%.1e", _ratio))  σ: [$(@sprintf("%.1e", S[end])), $(@sprintf("%.1e", S[1]))]"

        _compute_ad_printto([log_file], msg1)
        _compute_ad_printto([log_file], msg2)
        _compute_ad_printto([log_file], msg3)
        _compute_ad_printto([log_file], msg4)
        _compute_ad_printto([log_file], msg5)
        _compute_ad_printto([log_file], msg6)
        _compute_ad_printto([log_file], msg7)
        _compute_ad_printto([log_file], msg8)
        _compute_ad_printto([log_file], msg9)
        if (mod(k, log_interval) == 0) || bool_break
            _compute_ad_printto([stdout], msg1)
            _compute_ad_printto([stdout], msg2)
            _compute_ad_printto([stdout], msg3)
            _compute_ad_printto([stdout], msg4)
            _compute_ad_printto([stdout], msg5)
            _compute_ad_printto([stdout], msg6)
            _compute_ad_printto([stdout], msg7)
            _compute_ad_printto([stdout], msg8)
            _compute_ad_printto([stdout], msg9)
        end
        if bool_break
            _K = k
            break
        end
    end
    close(log_file)
    return traj_p[:, 1:_K], traj_beta[1:_K], traj_pot[1:_K], traj_kappa[1:_K]
end

# -----------------------------------------------------------------------
# Public entry (renamed from `solve_ad_equilibrium`). Solve an
# ArrowDebreuMarket's Walrasian equilibrium with the affine-scaled Newton
# / potential-reduction method, returning a simplex-normalized price.
# -----------------------------------------------------------------------
function afscaled_newton_equilibrium(ad; K=400, η=10.0, ϵ=1e-9, verbose=false,
    log_interval=verbose ? 1 : 10_000, ff="/tmp/afscaled_newton_equilibrium.log")
    n = ad.n
    linconstr = LinearConstr(1, n, ones(1, n), [1.0])
    alg = HessianBar(n, ad.m, ones(n) ./ n; linconstr=linconstr)
    alg.linsys = :direct
    run = () -> afscaled_newton(ad, alg; K=K, log_interval=log_interval,
        scaler=1.0, η=η, ϵ=ϵ, ff=ff)
    # When verbose, let the per-iteration convergence log reach stdout.
    verbose ? run() : redirect_stdout(run, devnull)
    p = copy(alg.p)
    p ./= sum(p)
    return (p=p,)
end
