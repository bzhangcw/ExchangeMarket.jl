# -----------------------------------------------------------------------
# subproblems
# -----------------------------------------------------------------------
@doc raw"""
    play! runs the subproblems as best-response-type mappings
    for all i ∈ I
        solve_substep!(alg, fisher, i; ϵᵢ=ϵᵢ)
    end
    ϵᵢ: the tolerance for the subproblem
"""
function play!(
    alg::Algorithm, fisher::FisherMarket;
    ϵᵢ=1e-7,
    verbose=false,
    all=false,
    timed=true
)
    _ts = time()
    _k = Threads.Atomic{Int}(0)
    sample!(alg.sampler, fisher)
    # @note:
    #  this will not work for sparse ones
    #  todo: maybe modify the nzval inplace.
    Threads.@threads for i in (all ? (1:fisher.m) : alg.sampler.indices)
        info = solve_substep!(
            alg, fisher, i;
            ϵᵢ=ϵᵢ,
        )
        Threads.atomic_add!(_k, info.k)
        if info.ϵ > ϵᵢ * 1e2
            @warn "subproblem $i is not converged: ϵ: $(info.ϵ)"
        end
    end
    timed && (alg.tₗ += time() - _ts)
    alg.kᵢ = _k.value / fisher.n
    verbose && validate(fisher, alg.μ)
    fisher.sumx .= sum(fisher.x[:, alg.sampler.indices]; dims=2)[:]
end

function produce_functions_from_subproblem(
    alg::Algorithm, fisher::FisherMarket, i::Int
)
    _p = alg.p
    _u(x) = fisher.u(x, i)
    _∇u(x) = fisher.∇u(x, i)
    _f(x) = -fisher.w[i] * log(_u(x)) + _p' * x + alg.μ * logbar(x)
    _g(x) = -fisher.w[i] * _∇u(x) / _u(x) + _p - alg.μ ./ x
    _H(x) = begin
        c = _∇u(x)
        u = _u(x)
        r = fisher.w[i] / u^2
        return r * c * c' + alg.μ * spdiagm(1 ./ (x .^ 2))
    end
    return _f, _g, _H, _u, _∇u
end

function solve_substep!(
    alg::Algorithm, fisher::FisherMarket, i::Int;
    ϵᵢ=1e-4
)
    if alg.optimizer.style == :nlp
        # warm-start
        _x₀ = fisher.x[:, i]
        # provide functions
        _f, _g, _H, _u, _∇u = produce_functions_from_subproblem(alg, fisher, i)
        info = solve!(alg.optimizer; f=_f, g=_g, H=_H, x₀=_x₀, tol=ϵᵢ)
        fisher.x[:, i] .= info.x
        fisher.val_u[i] = _u(info.x)
        fisher.val_∇u[:, i] = _∇u(info.x)
        return info
    elseif alg.optimizer.style == :linconic
        info = solve!(
            alg.optimizer;
            fisher=fisher, i=i, p=alg.p, μ=0.0, verbose=false
        )
        fisher.val_u[i] = fisher.u(fisher.x[:, i], i)
        fisher.val_f[i] = fisher.val_u[i]^(1 / fisher.ρ)
        fisher.val_∇u[:, i] = fisher.∇u(fisher.x[:, i], i)
        return info
    elseif alg.optimizer.style == :analytic
        if fisher.ρ == 1
            ratio = fisher.c[:, i] ./ alg.p
            # argmax returns the index of the maximum value,
            # it is always the smallest one among the ties.
            j₊ = argmax(ratio)
            fisher.x[:, i] .= 0
            fisher.x[i, j₊] = fisher.w[i] / alg.p[j₊]
            fisher.val_u[i] = fisher.u(fisher.x[:, i], i)
        else
            fisher.val_f[i], fisher.val_∇f[:, i], fisher.val_Hf[:, i] = fisher.f∇f(alg.p, i)
            fisher.x[:, i] = -fisher.w[i] ./ fisher.val_f[i] ./ fisher.σ .* fisher.val_∇f[:, i]
            fisher.val_u[i] = fisher.u(fisher.x[:, i], i)
        end
        return ResponseInfo(
            fisher.val_u[i],
            0.0,
            1,
            nothing
        )
    elseif alg.optimizer.style == :bids
        # @info "use bids to recover allocation"
        # use bids to recover allocation
        if fisher.ρ >= 0
            fisher.x[:, i] = fisher.b[:, i] ./ alg.p
            fisher.val_u[i] = fisher.u(fisher.x[:, i], i)
            fisher.val_f[i], fisher.val_∇f[:, i], fisher.val_Hf[:, i] = fisher.f∇f(alg.p, i)
            cs = fisher.c[:, i] .* spow.(fisher.x[:, i], fisher.ρ)
            sumcs = sum(cs)
            # update bids
            fisher.b[:, i] .= fisher.w[i] * cs ./ sumcs
        else
            fisher.x[:, i] = fisher.b[:, i] ./ alg.p
            fisher.val_u[i] = fisher.u(fisher.x[:, i], i)
            fisher.val_f[i], fisher.val_∇f[:, i], fisher.val_Hf[:, i] = fisher.f∇f(alg.p, i)
            cs = spow.(fisher.c[:, i] ./ spow.(alg.p, fisher.ρ), 1 / (1 - fisher.ρ))
            sumcs = sum(cs)
            fisher.b[:, i] .= fisher.w[i] * cs ./ sumcs
        end
        return ResponseInfo(
            fisher.val_u[i],
            0.0,
            1,
            nothing
        )
    end
end