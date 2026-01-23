using DRSOM
using COPT
import MathOptInterface as MOI

proj(x) = x < 0 ? Inf : x

printto(ios, x) = begin
    for io in ios
        println(io, x)
    end
end

function solve_subp(_htJ, _htz, _g, _β, _η; ρ=0.9)

    # _D = _β / (1 + ρ) / _η
    _D = _β / _η
    _G = Symmetric(sparse(_htJ * _htJ' ./ 2))
    _r = similar(_g)
    _s = similar(_g)
    _v = similar(_g)
    _k = 0
    _λ = 0.0
    n = length(_g)
    bool_accept = false
    _M = max(1e-12, _D^2 * n)
    # _M = 1e-8
    while true
        _tA = _htJ
        _tg = _g
        # _r .= _htz
        _r .= _htz - 0.5 * _D * (_tA * _tg)
        # check if the residual is close to the gradient
        # it should be because _g = -ones(n)
        # @show norm(_tA * _tg)
        # @show norm(_r)
        # @show norm(_a)
        # _r .= _a - 0.5 * _D * (_tA * _tg) ./ _M ≡ _a
        # @assert norm(_tA * _tg) < 1e-10
        s, _λ, _... = DRSOM.TrustRegionCholesky(sparse(_G ./ _M), -_r, 1.0)
        _v .= -(_D * _tg + _tA' * s) / (2 * _M)
        if _k > 20 || bool_accept
            _s .= s
            break
        end
        # println("M: $(@sprintf("%.4e", _M)), |v|: $(@sprintf("%.1e", norm(_v))), k: $(_k)")
        if (norm(_v) < 0.95 * ρ)
            bool_accept = true
        else
            _M *= 2
            _k += 1
        end
    end
    # println("M: $(@sprintf("%.4e", _M)), |v|: $(@sprintf("%.1e", norm(_v))), k: $(_k)")
    return _v, _s, _λ, _M, _k
end

function solve_socp_subp_primal(_htJ, _htz, _g, _α, _M, Δ; bool_verbose=0, mode=1, p=nothing)
    n = length(_htz)
    model = Model(COPT.ConeOptimizer)
    set_silent(model)
    set_optimizer_attribute(model, "LogToConsole", bool_verbose)

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
        # @constraint(model, sum(x) >= _α / (4 * max(p...)))
        # @constraint(model, p'x == 0)
    end
    @objective(model, Min, _α * xn + _M * t + δ - _g' * x)
    optimize!(model)
    return value.(x), value(objective_value(model)), value(δ), model
end

function solve_subp_type_ii(
    _htJ, _htz, _g, _α, _M; Δ=0.9, p=nothing
)

    _v = similar(_g)

    # solve the subproblem
    _x, _Q, _δ, _... = solve_socp_subp_primal(_htJ, _htz, _g, _α, _M, Δ; p=p)
    _v .= _x
    return _v, _Q, _δ
end




@doc raw"""
Generate a lower triangular utility matrix with the following structure:
```
c = [
    1.0 0.0 0.0 ... 1.0
    1.0 1.0 0.0 ... 0.0
    1.0 1.0 1.0 ... 0.0
    ...
    1.0 1.0 1.0 ... 1.0
]
```
"""
function generate_triagular_market(m, n; ρ=0.0, scale=1.0)
    c = Matrix(LowerTriangular(ones(n, m)))
    c[1, n] = 1.0
    c .*= scale
    b = diagm(ones(n))

    f0 = ArrowDebreuMarket(m, n; c=c, b=b, ρ=ρ)
    return f0
end




function potred0(
    f1, alg;
    K=3000, log_interval=10,
    scaler=1.0,
    η=1e-3,
    ϵ=1e-10,
)
    Random.seed!(1)
    n = f1.n
    e = ones(n)
    y = rand(n)
    # alg.p .= ones(n) ./ n
    alg.p .= ones(n) ./ n .* scaler
    @show alg.p

    k = 0

    # Open log file

    log_file = open("/tmp/log.log", "w")
    traj_beta = zeros(K)
    traj_pot = zeros(K)
    traj_kappa = zeros(K)
    traj_c = zeros(K)
    traj_p = zeros(n, K)
    _K = K
    bool_break = false
    _η = η
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
        _g = -ones(n)
        _β = norm(_htz)
        alg.pb .= alg.p

        # how much _htz is in the direction of u₁
        # U, S, V = svd(_htJ)
        # u₁ = U[:, 1]
        # _htz_u₁ = abs(_htz' * u₁) / norm(_htz)
        # _htz_u₊ = abs(_htz' * U[:, 2:end]) / norm(_htz)

        # adjustment inner loop
        v, s, λ, M, _k = solve_subp(_htJ, _htz, _g, _β, _η)
        q = alg.p .* v
        αₘ = min(proj.(-alg.pb ./ q)..., 0.9995)
        alg.p .= alg.p + αₘ * q
        alg.p .= alg.p ./ sum(alg.p) .* scaler
        _nv = norm(v)

        if _β < ϵ
            _K = k
            bool_break = true
        end
        traj_beta[k] = _β
        traj_kappa[k] = 0.0 # opnorm(PJP) / _c
        traj_p[:, k] .= alg.p
        _φ = _η * log(_β) - sum(log.(alg.p))
        traj_pot[k] = _φ
        msg1 = "$(k)th iteration |p|₁: $(@sprintf("%.1e", sum(alg.p))) sum(q): $(@sprintf("%.1e", sum(q)))"
        msg2 = "   |-$(k)th               φ: $(@sprintf("%.3e", _φ))  η: $(@sprintf("%.3e", _η))"
        msg3 = "   |-$(k)th         inner_k: $(_k)  |s|: $(@sprintf("%.1e", norm(s)))"
        msg4 = "   |-$(k)th               p: [$(@sprintf("%.2e", minimum(alg.p))), $(@sprintf("%.2e", maximum(alg.p)))]"
        msg5 = "   |-$(k)th               r: $(@sprintf("%.1e", _r))  λ₁: [$(@sprintf("%.1e", 0.0))]"
        msg6 = "   |-$(k)th               λ: $(@sprintf("%.1e", λ)) M: $(@sprintf("%.1e", M)) |v|: $(@sprintf("%.1e", _nv))"
        msg7 = "   |-$(k)th            |z|ₚ: $(@sprintf("%.4e", _β))"
        msg8 = "   |-$(k)th              αₘ: $(@sprintf("%.1e", αₘ))"
        # msg9 = "   |-$(k)th     cond.    σ₁: $(@sprintf("%.1e", S[1]))  z_u₁: $(@sprintf("%.1e", _htz_u₁))  z_u₊: $(@sprintf("%.1e", _htz_u₊))"
        printto([log_file], msg1)
        printto([log_file], msg2)
        printto([log_file], msg3)
        printto([log_file], msg4)
        printto([log_file], msg5)
        printto([log_file], msg6)
        printto([log_file], msg7)
        printto([log_file], msg8)
        # printto([log_file], msg9)
        if (mod(k, log_interval) == 0) || bool_break
            printto([stdout], msg1)
            printto([stdout], msg2)
            printto([stdout], msg3)
            printto([stdout], msg4)
            printto([stdout], msg5)
            printto([stdout], msg6)
            printto([stdout], msg7)
            printto([stdout], msg8)
            # printto([stdout], msg9)
        end
        if bool_break
            break
        end
    end
    close(log_file)
    return traj_p[:, 1:_K], traj_beta[1:_K], traj_pot[1:_K], traj_kappa[1:_K]
end


function potred2(
    f1, alg;
    K=3000, log_interval=10,
    scaler=1.0,
    η=1e-3,
    ϵ=1e-10,
)
    """
    @note: not working yet, try to update only the first n-1 prices
    Variant of potred where the last price is always fixed to 1,
    and only the first n-1 prices are updated.
    """

    Random.seed!(1)
    n = f1.n
    e = ones(n)
    y = rand(n)

    # Initialize prices: first n-1 prices sum to (scaler-1), last price is 1
    alg.p[1:n-1] .= rand(n - 1)
    alg.p[n] = 1.0
    @show alg.p

    k = 0

    # Open log file
    log_file = open("/tmp/log.log", "w")
    traj_beta = zeros(K)
    traj_pot = zeros(K)
    traj_kappa = zeros(K)
    traj_c = zeros(K)
    traj_p = zeros(n, K)
    _K = K
    bool_break = false
    _η = η

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
        _htJ = _P[1:end-1, 1:end-1] * _J[1:end-1, 1:end-1] * _P[1:end-1, 1:end-1]
        _htz = _P[1:end-1, 1:end-1] * _z[1:end-1]
        _g = -ones(n - 1)
        _β = norm(_htz)
        alg.pb .= alg.p

        # how much _htz is in the direction of u₁
        # U, S, V = svd(_htJ)
        # u₁ = U[:, 1]
        # _htz_u₁ = abs(_htz' * u₁) / norm(_htz)
        # _htz_u₊ = abs(_htz' * U[:, 2:end]) / norm(_htz)

        # adjustment inner loop - solve for all n dimensions
        v, s, λ, M, _k = solve_subp(_htJ, _htz, _g, _β, _η)

        # Only apply update to first n-1 prices, last one stays fixed
        q = alg.p[1:n-1] .* v

        # Compute step size considering only first n-1 components
        αₘ = min(proj.(-alg.pb[1:n-1] ./ q)..., 0.9995)

        # Update only first n-1 prices
        alg.p[1:n-1] .= alg.p[1:n-1] + αₘ * q

        # # Normalize first n-1 prices to sum to (scaler - 1)
        # alg.p[1:n-1] .= alg.p[1:n-1] ./ sum(alg.p[1:n-1]) .* (scaler - 1)

        # Keep last price fixed at 1
        # alg.p[n] = 1.0

        _nv = norm(v)  # Only consider first n-1 components

        if _β < ϵ
            _K = k
            bool_break = true
        end

        traj_beta[k] = _β
        traj_kappa[k] = 0.0
        traj_p[:, k] .= alg.p
        _φ = _η * log(_β) - sum(log.(alg.p))
        traj_pot[k] = _φ

        msg1 = "$(k)th iteration |p|₁: $(@sprintf("%.1e", sum(alg.p))) sum(q): $(@sprintf("%.1e", sum(q)))"
        msg2 = "   |-$(k)th               φ: $(@sprintf("%.3e", _φ))  η: $(@sprintf("%.3e", _η))"
        msg3 = "   |-$(k)th         inner_k: $(_k)  |s|: $(@sprintf("%.1e", norm(s)))"
        msg4 = "   |-$(k)th               p: [$(@sprintf("%.2e", minimum(alg.p))), $(@sprintf("%.2e", maximum(alg.p)))]  p[n]: $(@sprintf("%.2e", alg.p[n]))"
        msg5 = "   |-$(k)th               r: $(@sprintf("%.1e", _r))  λ₁: [$(@sprintf("%.1e", 0.0))]"
        msg6 = "   |-$(k)th               λ: $(@sprintf("%.1e", λ)) M: $(@sprintf("%.1e", M)) |v|: $(@sprintf("%.1e", _nv))"
        msg7 = "   |-$(k)th            |z|ₚ: $(@sprintf("%.4e", _β))"
        msg8 = "   |-$(k)th              αₘ: $(@sprintf("%.1e", αₘ))"
        # msg9 = "   |-$(k)th     cond.    σ₁: $(@sprintf("%.1e", S[1]))  z_u₁: $(@sprintf("%.1e", _htz_u₁))  z_u₊: $(@sprintf("%.1e", _htz_u₊))"

        printto([log_file], msg1)
        printto([log_file], msg2)
        printto([log_file], msg3)
        printto([log_file], msg4)
        printto([log_file], msg5)
        printto([log_file], msg6)
        printto([log_file], msg7)
        printto([log_file], msg8)
        # printto([log_file], msg9)

        if (mod(k, log_interval) == 0) || bool_break
            printto([stdout], msg1)
            printto([stdout], msg2)
            printto([stdout], msg3)
            printto([stdout], msg4)
            printto([stdout], msg5)
            printto([stdout], msg6)
            printto([stdout], msg7)
            printto([stdout], msg8)
            # printto([stdout], msg9)
        end

        if bool_break
            break
        end
    end

    close(log_file)
    return traj_p[:, 1:_K], traj_beta[1:_K], traj_pot[1:_K], traj_kappa[1:_K]
end

# this is the old version where we fix η ≡ n.
function potred_with_fo_norm(
    f1, alg;
    K=3000, log_interval=10,
    scaler=1.0,
    η=1e-3,
    ϵ=1e-10,
    style=1,
    Δₘ=0.95,
    Δₘ_min=1e-6,
    ff="/tmp/log.log"
)

    Random.seed!(1)
    n = f1.n
    e = ones(n)
    y = rand(n)
    # alg.p .= ones(n) ./ n
    alg.p .= ones(n) ./ n .* scaler
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
    bool_break = false
    _η = η
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
        _g = ones(n)
        _β = norm(_htz)
        alg.pb .= alg.p

        # how much _htz is in the direction of u₁
        U, S, V = svd(_htJ)
        u₁ = U[:, end]
        _ratio = (_β / norm(U[:, end]' * _htz))^2
        # use type II subproblem
        _D = _β / _η
        _α = _β / 20
        # _α = 0.0
        _M = 1e-10
        # _M = sqrt(n) * _D
        _k = 0
        _v = similar(_g)
        αₘ = 1.0
        q = similar(alg.p)
        bool_accept = false
        _φb = _η * log(_β) - sum(log.(alg.pb))
        while true
            if _k > 20 || bool_accept
                break
            end
            v, _Q = solve_subp_type_ii(_htJ, _htz, _g, _α, _M)
            _nv = norm(v)
            _v .= v
            Δ_acc = 1e-2 * _β
            # Δ_acc = _nv / (1 - _nv)
            printto(
                [log_file],
                "@M: $(@sprintf("%.4e", _M)), |v|: $(@sprintf("%.1e", _nv)), k: $(_k), 
                _α: $(@sprintf("%.4e", _α))"
            )
            flush(log_file)
            if (_nv > Δₘ) # too large
                _M *= 2
                _k += 1
                continue
            end
            if _nv < 5e-1 * _β # too small
                _α *= 0.2
                _M *= 0.7
                _k += 1
                continue
            end
            q .= alg.p .* v
            αₘ = min(proj.(-alg.pb ./ q)..., 0.9995)
            alg.p .= alg.p + αₘ * q
            alg.p .= alg.p ./ sum(alg.p) .* scaler
            update_budget!(f1, alg.p)
            play!(alg, f1)
            grad!(alg, f1)
            eval!(alg, f1)
            _β₊ = norm(alg.p .* alg.∇)
            Δβ = _β - _β₊
            _φ₊ = _η * log(_β₊) - sum(log.(alg.p))
            Δφ = _φ₊ - _φb
            printto(
                [log_file],
                "@M: $(@sprintf("%.4e", _M)), |v|: $(@sprintf("%.1e", _nv)), k: $(_k), 
                _α: $(@sprintf("%.4e", _α)), Δβ: $(@sprintf("%.1e", Δβ))"
            )
            if Δβ < 0.1 * Δ_acc # too small function value decrease
                _α *= 1.2
                _k += 1
            else
                bool_accept = true
            end

        end
        _nv = norm(_v)



        # predicted |z|_+ 
        spec = 0.0
        try
            spec = opnorm(_htJ)
        catch
        end

        if _β < ϵ
            _K = k
            bool_break = true
        end
        traj_beta[k] = _β
        traj_kappa[k] = 0.0 # opnorm(PJP) / _c
        traj_p[:, k] .= alg.p
        _φ = _η * log(_β) - sum(log.(alg.p))
        traj_pot[k] = _φ
        msg1 = "$(k)th iteration |p|₁: $(@sprintf("%.1e", sum(alg.p))) sum(q): $(@sprintf("%.1e", sum(q)))"
        msg2 = "   |-$(k)th               φ: $(@sprintf("%.3e", _φ))  η: $(@sprintf("%.3e", _η))"
        msg3 = "   |-$(k)th         inner_k: $(_k)"
        msg4 = "   |-$(k)th               p: [$(@sprintf("%.2e", minimum(alg.p))), $(@sprintf("%.2e", maximum(alg.p)))]"
        msg5 = "   |-$(k)th               r: $(@sprintf("%.1e", _r))  λ₁: [$(@sprintf("%.1e", 0.0)), $(@sprintf("%.1e", spec))]"
        msg6 = "   |-$(k)th               α: $(@sprintf("%.1e", _α)) M: $(@sprintf("%.1e", _M)) |v|: $(@sprintf("%.1e", _nv))"
        msg7 = "   |-$(k)th            |z|ₚ: $(@sprintf("%.4e", _β))"
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
            break
        end
    end
    close(log_file)
    return traj_p[:, 1:_K], traj_beta[1:_K], traj_pot[1:_K], traj_kappa[1:_K]
end

function potred_with_fo_norm1(
    f1, alg;
    K=3000, log_interval=10,
    scaler=1.0,
    η=1e-3,
    ϵ=1e-10,
    style=1,
    Δₘ=0.95,
    Δₘ_min=1e-6,
    ff="/tmp/log.log"
)

    Random.seed!(1)
    n = f1.n
    e = ones(n)
    y = rand(n)
    # alg.p .= ones(n) ./ n
    alg.p .= ones(n) ./ n .* scaler
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
    bool_break = false
    _η = η
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
        _D = _β / _η
        _α = 1e-8
        # _α = 0.0
        _θ = 1.0
        # _M = 1e-10
        # _α = _β / 20
        # _M = sqrt(n) * _D
        _M = 0.0
        _k = 0
        _v = similar(_z)
        αₘ = 1.0
        q = similar(alg.p)
        bool_accept = false
        _φb = _η * log(_β) - sum(log.(alg.pb))
        _u = _α
        _ℓ = 0.0
        while true
            if _k > 20 || bool_accept
                break
            end
            # _g = _D * (ones(n) .+ (n - η) * _θ * alg.p)
            _g = _D * (ones(n) .+ 1.1 * alg.p)
            v, _Q = solve_subp_type_ii(_htJ, _htz, _g, _α, 0.0; Δ=0.9)
            _nv = norm(v)
            _v .= v
            Δ_acc = 0.1 * _nv^2 / (1 - _nv)
            printto(
                [log_file],
                "@M: $(@sprintf("%.4e", _M)), |v|: $(@sprintf("%.1e", _nv)), k: $(_k), _α: $(@sprintf("%.4e", _α)), Δ_acc: $(@sprintf("%.1e", Δ_acc)) _θ: $(@sprintf("%.1e", _θ))"
            )
            flush(log_file)
            # if (_nv > min(Δₘ, _β^0.2))# too large
            #     _M *= 5
            #     continue
            # end
            if _nv < max(1e-2 * _β, _β^2) # too small
                _u = _α
                _α = (_u + _ℓ) / 2
                continue
            end
            printto(
                [log_file],
                "step size passed"
            )
            q .= alg.p .* v
            αₘ = min(proj.(-alg.pb ./ q)..., 0.9995)
            alg.p .= alg.pb + αₘ * q

            alg.p .= alg.p ./ sum(alg.p) .* scaler
            update_budget!(f1, alg.p)
            play!(alg, f1)
            grad!(alg, f1)
            eval!(alg, f1)
            _β₊ = norm(alg.p .* alg.∇)
            _φ₊ = _η * log(_β₊) - sum(log.(alg.p))
            M_est = _β + norm(_htJ * _v + _htz)
            Δβ = _β - _β₊
            Δφ = _φb - _φ₊
            printto(
                [log_file],
                "@M: $(@sprintf("%.4e", _M)), |v|: $(@sprintf("%.1e", _nv)), k: $(_k), _α: $(@sprintf("%.4e", _α)), Δφ: $(@sprintf("%.1e", Δφ))"
            )
            # if Δβ < Δ_acc # too small function value decrease
            if Δφ < 1e-3
                if _α < _β # too small function value decrease
                    _ℓ = _α
                    _α *= 8
                else
                end
                _k += 1
            else
                bool_accept = true
            end
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

        if _β < ϵ
            bool_break = true
        end
        traj_beta[k] = _β
        traj_kappa[k] = 0.0 # opnorm(PJP) / _c
        traj_p[:, k] .= alg.p
        _φ = _η * log(_β) - sum(log.(alg.p))
        traj_pot[k] = _φ
        msg1 = "$(k)th iteration |p|₁: $(@sprintf("%.1e", sum(alg.p))) sum(q): $(@sprintf("%.1e", sum(q)))"
        msg2 = "   |-$(k)th               φ: $(@sprintf("%.3e", _φ))  η: $(@sprintf("%.3e", _η))"
        msg3 = "   |-$(k)th         inner_k: $(_k)"
        msg4 = "   |-$(k)th               p: [$(@sprintf("%.2e", minimum(alg.p))), $(@sprintf("%.2e", maximum(alg.p)))]"
        msg5 = "   |-$(k)th               r: $(@sprintf("%.1e", _r))  λ₁: [$(@sprintf("%.1e", 0.0)), $(@sprintf("%.1e", spec))]"
        msg6 = "   |-$(k)th               α: $(@sprintf("%.1e", _α)) M: $(@sprintf("%.1e", _M)) |v|: $(@sprintf("%.1e", _nv))"
        msg7 = "   |-$(k)th            |z|ₚ: $(@sprintf("%.4e", _β))"
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

include("./func_new.jl")