# -----------------------------------------------------------------------
# subproblems
# -----------------------------------------------------------------------
@doc raw"""
    play! runs the subproblems as best-response-type mappings
    for all i ∈ I
        solve_substep!(alg, market, i; ϵᵢ=ϵᵢ)
    end
    ϵᵢ: the tolerance for the subproblem
"""
function play!(
    alg::Algorithm, market::Market;
    verbose=false,
    ϵᵢ=1e-7,
    all=false,
    timed=true,
    style=alg.optimizer.style
)
    _ts = time()
    _k = Threads.Atomic{Int}(0)
    sample!(alg.sampler, market)

    Threads.@threads for i in (all ? (1:market.m) : alg.sampler.indices)
        info = solve_substep!(
            alg, market, i;
            ϵᵢ=ϵᵢ,
            style=style
        )
        Threads.atomic_add!(_k, info.k)
    end
    timed && (alg.tₗ += time() - _ts)
    verbose && validate(market, alg.μ)
    market.sumx .= sum(market.x[:, alg.sampler.indices]; dims=2)[:]
end

function produce_functions_from_subproblem(
    alg::Algorithm, market::Market, i::Int
)
    _p = alg.p
    _u(x) = market.u(x, i)
    _∇u(x) = market.∇u(x, i)
    _f(x) = -market.w[i] * log(_u(x)) + _p' * x + alg.μ * logbar(x)
    _g(x) = -market.w[i] * _∇u(x) / _u(x) + _p - alg.μ ./ x
    _H(x) = begin
        c = _∇u(x)
        u = _u(x)
        r = market.w[i] / u^2
        return r * c * c' + alg.μ * spdiagm(1 ./ (x .^ 2))
    end
    return _f, _g, _H, _u, _∇u
end

function solve_substep!(
    alg::Algorithm, market::Market, i::Int;
    ϵᵢ=1e-7,
    style=alg.optimizer.style,
    kwargs...
)
    if style == :nlp
        # warm-start
        _x₀ = market.x[:, i]
        # provide functions
        _f, _g, _H, _u, _∇u = produce_functions_from_subproblem(alg, market, i)
        info = solve!(alg.optimizer; f=_f, g=_g, H=_H, x₀=_x₀, kwargs...)
        market.x[:, i] .= info.x
        market.val_u[i] = _u(info.x)
        market.val_∇u[:, i] = _∇u(info.x)
        return info
    elseif style == :linconic
        info = solve!(
            alg.optimizer;
            fisher=market,
            i=i,
            p=alg.p,
            μ=0.0,
            verbose=false
        )
        market.val_u[i] = market.u(market.x[:, i], i)
        market.val_f[i] = market.val_u[i]^(1 / market.ρ)
        market.val_∇u[:, i] = market.∇u(market.x[:, i], i)
        return info
    elseif style == :linconicaffine
        info = solve!(
            alg.optimizer;
            fisher=market,
            i=i,
            p=alg.p,
            μ=0.0,
            verbose=false,
            kwargs...
        )
        market.val_u[i] = market.u(market.x[:, i], i)
        market.val_f[i] = market.val_u[i]^(1 / market.ρ)

        return info


    elseif style == :analytic
        if is_linear_market(market)
            ratio = market.c[:, i] ./ alg.p
            # argmax returns the index of the maximum value,
            # it is always the smallest one among the ties.
            j₊ = argmax(ratio)
            market.x[:, i] .= 0
            market.x[j₊, i] = market.w[i] / alg.p[j₊]
            market.val_u[i] = market.u(market.x[:, i], i)
        else
            market.val_f[i], market.val_∇f[:, i], market.val_Hf[:, i] = market.f∇f(alg.p, i)
            market.x[:, i] = -market.w[i] ./ market.val_f[i] ./ market.σ[i] .* market.val_∇f[:, i]
            market.val_u[i] = market.u(market.x[:, i], i)
        end
        return ResponseInfo(
            market.val_u[i],
            0.0,
            1,
            nothing
        )
    elseif style == :bids
        # @info "use bids to recover allocation"
        # use bids to recover allocation
        if all(market.ρ .>= 0)
            market.x[:, i] = market.g[:, i] ./ alg.p
            market.val_u[i] = market.u(market.x[:, i], i)
            market.val_f[i], market.val_∇f[:, i], market.val_Hf[:, i] = market.f∇f(alg.p, i)
            cs = market.c[:, i] .* spow.(market.x[:, i], market.ρ[i])
            sumcs = sum(cs)
            # update bids
            market.g[:, i] .= market.w[i] * cs ./ sumcs
        else
            market.x[:, i] = market.g[:, i] ./ alg.p
            market.val_u[i] = market.u(market.x[:, i], i)
            market.val_f[i], market.val_∇f[:, i], market.val_Hf[:, i] = market.f∇f(alg.p, i)
            cs = spow.(market.c[:, i] ./ spow.(alg.p, market.ρ[i]), 1 / (1 - market.ρ[i]))
            sumcs = sum(cs)
            market.g[:, i] .= market.w[i] * cs ./ sumcs
        end
        return ResponseInfo(
            market.val_u[i],
            0.0,
            1,
            nothing
        )
    elseif style == :linprog
        # For piecewise linear utilities, use LP solver
        info = solve!(
            alg.optimizer;
            fisher=market,
            i=i,
            p=alg.p,
            μ=alg.μ,
            verbose=false,
            kwargs...
        )
        market.val_u[i] = market.u(market.x[:, i], i)
        return info
    end
end