# -----------------------------------------------------------------------
# differentiation utilities
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------

function grad!(alg, fisher::FisherMarket)
    if fisher.ρ == 1.0
        __linear_grad!(alg, fisher)
    else
        __ces_grad!(alg, fisher)
    end
end

function hess!(alg, fisher::FisherMarket; bool_dbg=false)
    if fisher.ρ == 1.0
        __linear_hess!(alg, fisher; bool_dbg=bool_dbg)
    else
        __ces_hess!(alg, fisher; bool_dbg=bool_dbg)
    end
end

function eval!(alg, fisher::FisherMarket)
    if fisher.ρ == 1.0
        __linear_eval!(alg, fisher)
    else
        __ces_eval!(alg, fisher)
    end
end


function __linear_grad!(alg, fisher::FisherMarket; bool_dbg=false)
    if alg.option_grad == :usex
        __linear_grad_fromx!(alg, fisher)
    else
        # __linear_grad_dual!(alg, fisher)
    end
end

function __linear_hess!(alg, fisher::FisherMarket; bool_dbg=false)
    if alg.option_grad == :usex
        __linear_hess_fromx!(alg, fisher; bool_dbg=bool_dbg)
    else
        # __linear_hess_dual!(alg, fisher)
    end
end

function __linear_eval!(alg, fisher::FisherMarket)
    if alg.option_grad == :usex
        __linear_eval_fromx!(alg, fisher)
    else
        # __linear_eval_dual!(alg, fisher)
    end
end

# -----------------------------------------------------------------------
# linear case; :usex mode
# -----------------------------------------------------------------------
function __linear_grad_fromx!(alg, fisher::FisherMarket)
    alg.∇ .= fisher.q .* (alg.sampler.batchsize / fisher.m) - sum(fisher.x[alg.sampler.indices, :]; dims=1)[:]
end

# compute Jacobian: -dx/dp
function __linear_jacxp_fromx(X₂, u, c, w, μ)
    invμ = 1 / μ
    Xc = X₂ * c
    r = w / u^2
    return invμ * X₂ - (invμ^2 * r * Xc * Xc') ./ (1 + invμ * r * c' * Xc)
end

# compute Jacobian -dp/dx
function __linear_jacpx_fromx(Xi₂, u, c, w, μ)
    r = w / u^2
    return μ * Xi₂ + r * c * c'
end

function __linear_hess_fromx!(alg, fisher::FisherMarket; bool_dbg=false)
    X2 = fisher.x[alg.sampler.indices, :] .^ 2
    Di(i) = begin
        X₂ = spdiagm(X2[i, :])
        u = fisher.val_u[i]
        c = fisher.val_∇u[i, :]
        w = fisher.w[i]
        jxp = __linear_jacxp_fromx(X₂, u, c, w, alg.μ)
        if bool_dbg
            Xi₂ = spdiagm(1 ./ X2[i, :])
            jpx = __linear_jacpx_fromx(Xi₂, u, c, w, alg.μ)
            @info "jacpx * jacxp - I" maximum(abs.(jpx * jxp - I))
        end
        return jxp
    end
    alg.H = mapreduce(Di, +, alg.sampler.indices, init=spzeros(fisher.n, fisher.n))
end

function __linear_eval_fromx!(alg, fisher::FisherMarket)
    alg.φ = (
        logbar(fisher.val_u, fisher.w) +
        alg.μ * logbar(fisher.x) +
        alg.μ * logbar(alg.p) +
        alg.p' * alg.∇ - alg.μ * fisher.n
    )
end



# -----------------------------------------------------------------------
# general CES case: ρ < 1
# -----------------------------------------------------------------------
function __ces_grad!(alg, fisher::FisherMarket; bool_dbg=false)
    if alg.option_grad == :usex
        __ces_grad_fromx!(alg, fisher; bool_dbg=bool_dbg)
    else
        __ces_grad_dual!(alg, fisher)
    end
end

function __ces_hess!(alg, fisher::FisherMarket; bool_dbg=false)
    if alg.option_grad == :usex
        __ces_hess_fromx!(alg, fisher; bool_dbg=bool_dbg)
    else
        __ces_hess_dual!(alg, fisher)
    end
end

function __ces_eval!(alg, fisher::FisherMarket)
    if alg.option_grad == :usex
        __ces_eval_fromx!(alg, fisher)
    else
        __ces_eval_dual!(alg, fisher)
    end
end

# -----------------------------------------------------------------------
# general CES case: ρ < 1, :usex mode
# -----------------------------------------------------------------------
function __ces_grad_fromx!(alg, fisher::FisherMarket; bool_dbg=false)
    @assert fisher.ρ < 1
    alg.∇ .= fisher.q .* (alg.sampler.batchsize / fisher.m) - sum(fisher.x[alg.sampler.indices, :]; dims=1)[:]
    tmp = copy(alg.∇)
    if bool_dbg
        # compare with the result from :dual mode
        __ces_grad!(alg, fisher)
        @info "(q - ∑x) - (q - w/ν ∇ν)" maximum(abs.(alg.∇ - tmp))
    end
end

# -----------------------------------------------------------------------
# general CES case: ρ < 1, :dual mode
# -----------------------------------------------------------------------
function __ces_grad_dual!(alg, fisher::FisherMarket)
    @assert fisher.ρ < 1
    # the following is not needed, move to play
    # for i in 1:fisher.m
    #     fisher.val_f[i], fisher.val_∇f[i, :], fisher.val_Hf[i, :] = fisher.f∇f(alg.p, i)
    #     fisher.x[i, :] = -fisher.w[i] ./ fisher.val_f[i] ./ fisher.σ .* fisher.val_∇f[i, :]
    #     fisher.val_u[i] = fisher.u(fisher.x[i, :], i)
    # end
    alg.∇ .= fisher.q .* (alg.sampler.batchsize / fisher.m) - sum(fisher.x[alg.sampler.indices, :]; dims=1)[:]
end

function __ces_hess_dual!(alg, fisher::FisherMarket)
    σ = fisher.σ
    w = fisher.w
    # compute 1/σ w_i * log(cs_i'p^{-σ})
    _Hi = (i) -> begin
        _H = spdiagm(1 / fisher.val_f[i] .* fisher.val_Hf[i, :]) -
             (1 / fisher.val_f[i])^2 * fisher.val_∇f[i, :] * fisher.val_∇f[i, :]'
        return _H .* (w[i] / σ)
    end
    alg.H .= mapreduce(_Hi, +, alg.sampler.indices, init=spzeros(fisher.n, fisher.n))
end

function __ces_eval_dual!(alg, fisher::FisherMarket)
    σ = fisher.σ
    w = fisher.w
    alg.φ = min(
        alg.p' * fisher.q +
        sum((w[i] / σ) * log(fisher.val_f[i]) for i in 1:fisher.m) +
        sum(w[i] * log(w[i]) for i in 1:fisher.m),
        1e8
    )
end


