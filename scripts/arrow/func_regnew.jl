function regnew(f1, alg; K=3000, log_interval=10, bool_proj=true)
    Random.seed!(1)
    n = f1.n
    e = ones(n)
    # alg.p .= rand(n)
    alg.p .= ones(n) ./ n
    @show alg.p

    η₀ = 2
    bool_dynamic_η = false
    bool_trust_region = false
    pot(r, p; η=η₀) = η * log(r) - sum(log.(p))

    k = 0
    Πₚ(p) = I - p * p' / norm(p)^2
    _φ₋ = 0.0
    _npz_pred = 0.0

    traj_beta = zeros(K)
    traj_residual = zeros(K)
    traj_kappa = zeros(K)
    traj_c = zeros(K)
    traj_p = zeros(n, K)
    _K = K
    bool_break = false
    for k in 1:K
        update_budget!(f1, alg.p)
        play!(alg, f1)
        grad!(alg, f1)
        eval!(alg, f1)
        hess!(alg, f1)
        _J = alg.H
        _z = alg.∇
        _r = sum(_z .^ 2)
        Jf = _J' * _z
        if bool_dynamic_η
            _η = 1 ./ _r
            # _η = √_r
            # _η = _r .^ 2
            # _η = min(_r .^ 2, 1)
            # _η = log(_r+1)
        else
            _η = η₀
        end
        _P = diagm(alg.p)
        _pz = alg.p .* _z
        _a = _J' * _P * _pz

        PJP = _P * _J * _P

        PQP = Symmetric(PJP' * PJP)
        _pa = alg.p .* _a

        _na = norm(_pa)
        _β = _nz = norm(_pz)
        spec = eigvals(Matrix(PQP))


        # use Regularized Newton
        _λ = (_β * 2.0)
        v = -(PQP + _λ * I) \ (_pa)

        _nv = norm(v)
        alg.pb .= alg.p

        alg.p .= alg.p + alg.p .* v
        alg.p ./= sum(alg.p)

        _φ = pot(_r, alg.p)
        _c = _na / _nz

        # predicted |z|_+ 

        vl = _na / (_λ + spec[end])
        vh = _na / _λ

        _φ₋ = _φ


        traj_beta[k] = _β
        traj_residual[k] = _r
        traj_c[k] = _c
        traj_kappa[k] = opnorm(PJP) / _c
        traj_p[:, k] .= alg.p
        if _r < 1e-18
            _K = k
            bool_break = true
        end
        if (mod(k, log_interval) == 0) || bool_break
            println("$(k)th iteration φ: $(@sprintf("%.1e", _φ)) η: $(@sprintf("%.1e", _η)) |p|₁: $(@sprintf("%.1e", sum(alg.p))) \tΔφ: $(@sprintf("%.1e", _φ - _φ₋))")
            println("   |-$(k)th               r: $(@sprintf("%.1e", _r))  λ₁: [$(@sprintf("%.1e", spec[1])), $(@sprintf("%.1e", spec[end]))] : $(@sprintf("%.1e", opnorm(PJP)^2))")
            println("   |-$(k)th               λ: $(@sprintf("%.1e", _λ)) |v|: $(@sprintf("%.1e", _nv)) ∈ [$(@sprintf("%.1e", vl)), $(@sprintf("%.1e",vh)))]")
            println("   |-$(k)th            |a|ₚ: $(@sprintf("%.1e", _na)) |z|ₚ: $(@sprintf("%.4e", _nz))")
            println("   |-$(k)th       |a|ₚ/|z|ₚ: $(@sprintf("%.1e", _c)) κ: $(@sprintf("%.1e", √(spec[end])/_c))")
        end
        if bool_break
            break
        end
    end
    return traj_p[:, 1:_K], traj_beta[1:_K], traj_residual[1:_K], traj_c[1:_K], traj_kappa[1:_K]
end


function regnew2(f1, alg; K=3000, log_interval=10, bool_proj=true)
    Random.seed!(1)
    n = f1.n
    e = ones(n)
    alg.p .= ones(n) ./ n
    @show alg.p

    η₀ = 2
    pot(r, p; η=η₀) = η * log(r) - sum(log.(p))

    k = 0
    Πₚ(p) = I - p * p' / norm(p)^2

    traj_beta = zeros(K)
    traj_residual = zeros(K)
    traj_kappa = zeros(K)
    traj_c = zeros(K)
    traj_p = zeros(n, K)
    _K = K
    bool_break = false
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
        _a = _J' * _z
        _pz = alg.p .* _z

        PQP = Symmetric(_P * _J' * _J * _P)
        _pa = alg.p .* _a

        _na = norm(_pa)
        _β = _nz = norm(_pz)
        spec = eigvals(Matrix(PQP))


        # use Regularized Newton
        _λ = (_β * 2.0)
        v = -(PQP + _λ * I) \ (_pa)

        _nv = norm(v)
        alg.pb .= alg.p

        alg.p .= alg.p + alg.p .* v
        alg.p ./= sum(alg.p)

        _c = _na / _nz

        # predicted |z|_+ 

        vl = _na / (_λ + spec[end])
        vh = _na / _λ


        traj_beta[k] = _β
        traj_residual[k] = _r
        traj_c[k] = _c
        traj_kappa[k] = 0.0 # opnorm(PJP) / _c
        traj_p[:, k] .= alg.p
        if _r < 1e-18
            _K = k
            bool_break = true
        end
        if (mod(k, log_interval) == 0) || bool_break
            println("$(k)th iteration |p|₁: $(@sprintf("%.1e", sum(alg.p)))")
            println("   |-$(k)th               r: $(@sprintf("%.1e", _r))  λ₁: [$(@sprintf("%.1e", spec[1])), $(@sprintf("%.1e", spec[end]))]")
            println("   |-$(k)th               λ: $(@sprintf("%.1e", _λ)) |v|: $(@sprintf("%.1e", _nv)) ∈ [$(@sprintf("%.1e", vl)), $(@sprintf("%.1e",vh)))]")
            println("   |-$(k)th            |a|ₚ: $(@sprintf("%.1e", _na)) |z|ₚ: $(@sprintf("%.4e", _nz))")
            println("   |-$(k)th       |a|ₚ/|z|ₚ: $(@sprintf("%.1e", _c)) κ: $(@sprintf("%.1e", √(spec[end])/_c))")
        end
        if bool_break
            break
        end
    end
    return traj_p[:, 1:_K], traj_beta[1:_K], traj_residual[1:_K], traj_c[1:_K], traj_kappa[1:_K]
end

proj(x) = x < 0 ? Inf : x
function regnew3(
    f1, alg;
    K=3000, log_interval=10,
    bool_proj=false,
    scaler=10.0,
    τ=0.1
)
    Random.seed!(1)
    n = f1.n
    e = ones(n)
    # alg.p .= ones(n) ./ n
    alg.p .= ones(n) ./ n .* scaler
    @show alg.p

    η₀ = 2
    pot(r, p; η=η₀) = η * log(r) - sum(log.(p))

    k = 0
    _φ₋ = 0.0

    traj_beta = zeros(K)
    traj_residual = zeros(K)
    traj_kappa = zeros(K)
    traj_c = zeros(K)
    traj_p = zeros(n, K)
    _K = K
    bool_break = false
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
        _PJ = _P * _J
        _pz = alg.p .* _z
        _a = _PJ' * _pz
        _pa = alg.p .* _a
        _na = norm(_pa)
        _β = _nz = norm(_pz)
        # _λ = (10 * sqrt(_β) * 2.0)
        _λ = (_β * 1.01)
        # @info "sum(Pz): $(@sprintf("%.1e", sum(_pz)))"


        _Q = _PJ' * _PJ
        _PQP = _P * _Q * _P
        # option 1: use KKT system
        v = zeros(n)
        q = zeros(n)
        _j = 0
        if !bool_proj
            while true
                _kkt = [
                    (_Q+_λ*diagm(1 ./ (alg.p .^ 2))) e;
                    e' 0.0
                ]
                _rhs = [
                    -_PJ' * _pz;
                    0.0
                ]
                res = _kkt \ _rhs
                _ = res[n+1:end]
                q .= res[1:n]
                v .= q ./ alg.p
                _j += 1
                if (norm(v) < 0.9995) || (_j > 5)
                    break
                else
                    _λ *= 2
                    println("λ: $_j: $(@sprintf("%.1e", _λ))")
                    break
                end
            end
        else
            # option 2: use projection matrix
            while true
                _λ = τ * _β
                # _Pi = I - diagm(alg.p) * diagm(alg.p)' / norm(alg.p)^2
                _Pi = I - alg.p * alg.p' / norm(alg.p)^2
                v .= -(_Pi * (_PQP + _λ * I) * _Pi) \ (_Pi * _P * _PJ' * _pz)
                # println("$(alg.p'*(_Pi*v))")
                _j += 1
                if (norm(v) < 0.9995) || (_j > 5)
                    q .= alg.p .* _Pi * v
                    break
                else
                    _λ *= 2
                    println("λ: $_j: $(@sprintf("%.1e", _λ))")
                    break
                end
            end
        end
        _nv = norm(v)
        alg.pb .= alg.p

        αₘ = min(proj.(-alg.pb ./ q)..., 1.0)
        alg.p .= alg.p + αₘ * q


        _c = _na / _nz

        # predicted |z|_+ 
        spec = opnorm(_PJ * _P)
        vl = _na / (_λ + spec^2)
        vh = _na / _λ



        traj_beta[k] = _β
        traj_residual[k] = _r
        traj_c[k] = _c
        traj_kappa[k] = 0.0 # opnorm(PJP) / _c
        traj_p[:, k] .= alg.p
        if _r < 1e-18
            _K = k
            bool_break = true
        end
        if (mod(k, log_interval) == 0) || bool_break
            println("$(k)th iteration |p|₁: $(@sprintf("%.1e", sum(alg.p))) sum(q): $(@sprintf("%.1e", sum(q)))")
            println("   |-$(k)th               p: [$(@sprintf("%.1e", minimum(alg.p))), $(@sprintf("%.1e", maximum(alg.p)))]")
            println("   |-$(k)th               r: $(@sprintf("%.1e", _r))  λ₁: [$(@sprintf("%.1e", 0.0)), $(@sprintf("%.1e", spec))]")
            println("   |-$(k)th               λ: $(@sprintf("%.1e", _λ)) |v|: $(@sprintf("%.1e", _nv)) ∈ [$(@sprintf("%.1e", vl)), $(@sprintf("%.1e",vh)))]")
            println("   |-$(k)th            |a|ₚ: $(@sprintf("%.1e", _na)) |z|ₚ: $(@sprintf("%.4e", _nz))")
            println("   |-$(k)th       |a|ₚ/|z|ₚ: $(@sprintf("%.1e", _c)) κ: $(@sprintf("%.1e", (spec/_c)))")
            println("   |-$(k)th              αₘ: $(@sprintf("%.1e", αₘ))")
        end
        if bool_break
            break
        end
    end
    return traj_p[:, 1:_K], traj_beta[1:_K], traj_residual[1:_K], traj_c[1:_K], traj_kappa[1:_K]
end