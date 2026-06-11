using DRSOM
using MosekTools
import MathOptInterface as MOI

# -----------------------------------------------------------------------
# Solve an ArrowDebreuMarket's Walrasian equilibrium via the potential-
# reduction solver, returning a simplex-normalized price. Thin wrapper used
# by the AD surrogate validation (validate_ad.jl); mirrors the setup in
# test_lift.jl's backward test (HessianBar + Σp=1 + potred_with_fo_new).
# -----------------------------------------------------------------------
function solve_ad_equilibrium(ad; K=400, η=10.0, ϵ=1e-9, verbose=false,
    log_interval=verbose ? 50 : 10_000, ff="/tmp/solve_ad_equilibrium.log")
    n = ad.n
    linconstr = LinearConstr(1, n, ones(1, n), [1.0])
    alg = HessianBar(n, ad.m, ones(n) ./ n; linconstr=linconstr)
    alg.linsys = :direct
    run = () -> potred_with_fo_new(ad, alg; K=K, log_interval=log_interval,
        scaler=1.0, η=η, ϵ=ϵ, ff=ff)
    # When verbose, let potred's per-iteration convergence log reach stdout;
    # otherwise suppress it.
    verbose ? run() : redirect_stdout(run, devnull)
    p = copy(alg.p)
    p ./= sum(p)
    return (p=p,)
end

function potred_with_fo_new(
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
    # alg.p .= ones(n) ./ n .* scaler
    alg.p .= ones(n) .* scaler ./ n
    @show alg.p

    k = 0

    # Open log file

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
            # println("1'v: $(@sprintf("%.1e", sum(v)))")
            _nv = norm(v)
            _v .= v
            Δ_acc = 0.1 * _nv^2 / (1 - _nv)
            printto(
                [log_file],
                "@M: $(@sprintf("%.4e", _M)), |v|: $(@sprintf("%.1e", _nv)), k: $(_k), _α: $(@sprintf("%.4e", _α)), Δ_acc: $(@sprintf("%.1e", Δ_acc)) _θ: $(@sprintf("%.1e", _θ))"
            )
            flush(log_file)
            q .= alg.p .* v
            # println(q)
            αₘ = min(proj.(-alg.pb ./ q)..., 0.9)
            alg.p .= alg.pb + αₘ * q
            # alg.p .= alg.p .* norm(1 .+ v) .* scaler
            # alg.p .= alg.p ./ sum(alg.p) .* scaler
            # alg.p .= alg.p ./ 2
            alg.p .= alg.p ./ maximum(1 .+ v) .* scaler
            # alg.p .= alg.p ./ minimum(alg.p) .* scaler

            printto([log_file], "before: $(@sprintf("%.1e", tr(I + diagm(v)))) after: $(@sprintf("%.1e", maximum(alg.p ./ alg.pb)))")
            # alg.p .= alg.p ./ maximum(alg.p) .* scaler
            update_budget!(f1, alg.p)
            play!(alg, f1)
            grad!(alg, f1)
            eval!(alg, f1)
            _β₊ = norm(alg.p .* alg.∇)
            _φ₊ = _η * log(_β₊)
            M_est = _β + norm(_htJ * _v + _htz)
            Δβ = _β - _β₊
            Δφ = _φb - _φ₊
            printto(
                [log_file],
                "@M: $(@sprintf("%.4e", _M)), |v|: $(@sprintf("%.1e", _nv)), k: $(_k), _α: $(@sprintf("%.4e", _α)), Δφ: $(@sprintf("%.1e", Δφ))"
            )

            bool_accept = true

        end
        _nv = norm(_v)


        if !bool_accept
            # bool_break = true
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
        traj_kappa[k] = 0.0 # opnorm(PJP) / _c
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

        printto([log_file], msg1)
        printto([log_file], msg2)
        printto([log_file], msg3)
        printto([log_file], msg4)
        printto([log_file], msg5)
        printto([log_file], msg6)
        printto([log_file], msg7)
        printto([log_file], msg8)
        printto([log_file], msg9)
        if (mod(k, log_interval) == 0) || bool_break
            printto([stdout], msg1)
            printto([stdout], msg2)
            printto([stdout], msg3)
            printto([stdout], msg4)
            printto([stdout], msg5)
            printto([stdout], msg6)
            printto([stdout], msg7)
            printto([stdout], msg8)
            printto([stdout], msg9)
        end
        if bool_break
            _K = k
            break
        end
    end
    close(log_file)
    return traj_p[:, 1:_K], traj_beta[1:_K], traj_pot[1:_K], traj_kappa[1:_K]
end

