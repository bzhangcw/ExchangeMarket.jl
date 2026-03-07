# -----------------------------------------------------------------------
# differentiation utilities for Arrow–Debreu exchange market
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# for ArrowDebreuMarket
# -----------------------------------------------------------------------
function __ces_hess!(alg, market::ArrowDebreuMarket)
    # compute 1/σ w_i * log(cs_i'p^{-σ})
    if alg.linsys == :direct
        __ces_compute_exact_hess!(alg, market)
    elseif alg.linsys == :direct_afcon
        # compute the exact Hessian of the affine-constrained problem
        throw(ArgumentError("linsys not supported: $(alg.linsys)"))
    elseif alg.linsys == :DRq
        throw(ArgumentError("linsys not supported: $(alg.linsys)"))
    elseif alg.linsys == :krylov
        # no preprocessing needed
        throw(ArgumentError("linsys not supported: $(alg.linsys)"))
    else
        throw(ArgumentError("linsys not supported: $(alg.linsys)"))
    end
end
# -----------------------------------------------------------------------
# compute the exact Hessian
# -----------------------------------------------------------------------
@doc raw"""
    __ces_compute_exact_hess_only_fisher!(alg, market::Market)
    Compute the exact Hessian of the problem, ∇²f, not affine-scaled
    only for the `Fisher` part. i.e., ignore fact that budget is from `<price, endowment>`.
"""
function __ces_compute_exact_hess_only_fisher!(alg, market::ArrowDebreuMarket)
    # _Hi = (i) -> begin
    #     _H = spdiagm(1 / market.val_f[i] .* market.val_Hf[:, i]) - (1 / market.val_f[i])^2 * market.val_∇f[:, i] * market.val_∇f[:, i]'
    #     return _H .* (w[i] / σ)
    # end
    # alg.H .= mapreduce(_Hi, +, alg.sampler.indices, init=spzeros(market.n, market.n))
    b = alg.p .* market.x
    γ = 1 ./ market.w' .* b
    u = market.w .* market.σ
    diag_term = b * (market.σ .+ 1)
    alg.H .= diagm(1 ./ alg.p) * (diagm(diag_term) - γ * diagm(u) * γ') * diagm(1 ./ alg.p)
    @info "use exact Hessian"
end

function __ces_compute_exact_hess!(alg, market::ArrowDebreuMarket)
    # _Hi = (i) -> begin
    #     _H = spdiagm(1 / market.val_f[i] .* market.val_Hf[:, i]) - (1 / market.val_f[i])^2 * market.val_∇f[:, i] * market.val_∇f[:, i]'
    #     return _H .* (w[i] / σ)
    # end
    # alg.H .= mapreduce(_Hi, +, alg.sampler.indices, init=spzeros(market.n, market.n))
    b = alg.p .* market.x
    γ = 1 ./ market.w' .* b
    u = market.w .* market.σ
    diag_term = b * (market.σ .+ 1)
    alg.H .= diagm(1 ./ alg.p) * (diagm(diag_term) - γ * diagm(u) * γ') * diagm(1 ./ alg.p)

    # add the endowment term
    θ = 1 ./ market.w' .* (alg.p .* market.b)
    alg.H += -diagm(1 ./ alg.p) * γ * diagm(market.w) * θ' * diagm(1 ./ alg.p)
end
