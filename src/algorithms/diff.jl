# -----------------------------------------------------------------------
# differentiation utilities
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
using LinearAlgebra
# -----------------------------------------------------------------------
# main functions
# -----------------------------------------------------------------------
function grad!(alg, market::Market)
    if is_linear_market(market)
        __linear_grad!(alg, market)
    else
        __ces_grad!(alg, market)
    end
end

function hess!(alg, market::Market; bool_dbg=false)
    if is_linear_market(market)
        __linear_hess!(alg, market; bool_dbg=bool_dbg)
    else
        __ces_hess!(alg, market; bool_dbg=bool_dbg)
    end
end

function eval!(alg, market::Market)
    if is_linear_market(market)
        __linear_eval!(alg, market)
    else
        __ces_eval!(alg, market)
    end
end

# -----------------------------------------------------------------------
# linear case
# -----------------------------------------------------------------------
function __linear_grad!(alg, market::Market; bool_dbg=false)
    if alg.option_grad == :usex
        __linear_grad_fromx!(alg, market)
    else
        # __linear_grad_dual!(alg, market)
    end
end

function __linear_hess!(alg, market::FisherMarket; bool_dbg=false)
    if alg.option_grad == :usex
        __linear_hess_fromx!(alg, market; bool_dbg=bool_dbg)
    else
        # __linear_hess_dual!(alg, market)
    end
end

function __linear_eval!(alg, market::Market)
    if alg.option_grad == :usex
        __linear_eval_fromx!(alg, market)
    else
        # __linear_eval_dual!(alg, market)
    end
end

# :usex mode
function __linear_grad_fromx!(alg, market::Market)
    alg.∇ .= market.q .* (alg.sampler.batchsize / market.m) - market.sumx
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

function __linear_hess_fromx!(alg, market::FisherMarket; bool_dbg=false)
    X2 = market.x[alg.sampler.indices, :] .^ 2
    Di(i) = begin
        X₂ = spdiagm(X2[:, i])
        u = market.val_u[i]
        c = market.val_∇u[:, i]
        w = market.w[i]
        jxp = __linear_jacxp_fromx(X₂, u, c, w, alg.μ)
        if bool_dbg
            Xi₂ = spdiagm(1 ./ X2[:, i])
            jpx = __linear_jacpx_fromx(Xi₂, u, c, w, alg.μ)
            @info "jacpx * jacxp - I" maximum(abs.(jpx * jxp - I))
        end
        return jxp
    end
    alg.H = mapreduce(Di, +, alg.sampler.indices, init=spzeros(market.n, market.n))
end

function __linear_eval_fromx!(alg, market::Market)
    alg.φ = (
        logbar(market.val_u, market.w) +
        alg.μ * logbar(market.x) +
        alg.μ * logbar(alg.p) +
        alg.p' * alg.∇ - alg.μ * market.n
    )
end



# -----------------------------------------------------------------------
# general CES case: ρ < 1
# -----------------------------------------------------------------------
function __ces_grad!(alg, market::Market; bool_dbg=false)
    __ces_grad_dual!(alg, market)
end

function __ces_hess!(alg, market::Market; bool_dbg=false)
    __ces_hess_dual!(alg, market)
end

function __ces_eval!(alg, market::Market)
    __ces_eval_dual!(alg, market)
end


# -----------------------------------------------------------------------
# general CES case: ρ < 1, :dual mode
# -----------------------------------------------------------------------
function __ces_eval_dual!(alg, market::Market)
    w = market.w
    alg.φ = min(
        alg.p' * market.q +
        sum(w[i] * log(market.val_u[i]) for i in 1:market.m),
        1e8
    )
end

function __ces_grad_dual!(alg, market::Market)
    @assert all(market.ρ .< 1)
    alg.∇ .= market.q .* (alg.sampler.batchsize / market.m) - market.sumx
end

function __ces_hess_dual!(alg, market::FisherMarket)
    # compute 1/σ w_i * log(cs_i'p^{-σ})
    if alg.linsys == :direct
        __compute_exact_hess_optimized!(alg, market)
    elseif alg.linsys == :direct_affine
        # compute the exact Hessian of the affine-constrained problem
        __compute_exact_hess_afcon!(alg, market)
    elseif alg.linsys == :DRq
        __compute_approx_hess_drq!(alg, market)
    elseif alg.linsys == :krylov
        # no preprocessing needed
    else
        throw(ArgumentError("linsys not supported: $(alg.linsys)"))
    end
end

# -----------------------------------------------------------------------
# compute the exact Hessian
# -----------------------------------------------------------------------
@doc raw"""
    __compute_exact_hess!(alg, market::FisherMarket)
    Compute the exact Hessian of the problem, ∇²f, not affine-scaled
"""
function __compute_exact_hess!(alg, market::FisherMarket)
    # bidding matrix shape: n×m
    γ = (alg.p .* market.x) ./ market.w'
    u = market.w .* market.σ
    alg.H .= diagm(1 ./ alg.p) * (diagm(γ * (market.w .+ u)) - γ * diagm(u) * γ') * diagm(1 ./ alg.p)
    @info "use exact Hessian"
end

# -----------------------------------------------------------------------
# compute the exact Hessian
# -----------------------------------------------------------------------
function __compute_exact_hess_optimized!(alg, market::FisherMarket)
    # — ensure a dense target ———————————————––
    if isa(alg.H, SparseMatrixCSC)
        alg.H = Matrix(alg.H)
    end
    H = alg.H                      # n×n dense
    # --- unpack ---------------------------------------------------------------
    p = alg.p                     # length n
    X = market.x                  # n×m
    w = market.w                  # length m  (strictly >0)
    σv = market.σ                 # length m (heterogeneous allowed)
    n, m = size(X)

    @assert length(p) == n
    @assert size(H) == (n, n)
    @assert length(w) == m

    W = similar(X)                     # n×m

    # --- 1) diagonal term ------------------------------------------------------
    # From H = P⁻¹[diag(γ*(w+u)) - γ diag(u) γ']P⁻¹, diag part becomes
    # diag_k = (∑_i X[k,i]*(1+σ_i)) / p_k
    row_weighted = X * (1 .+ σv)
    diag_term = @. row_weighted / p

    # --- 2) build W = X .* (1 ./ √w)' -----------------------------------------
    inv_sqrt_w = 1 ./ sqrt.(w)                  # m  (tiny alloc)
    @inbounds for j in 1:m
        @views W[:, j] .= X[:, j] .* inv_sqrt_w[j]
    end

    # --- 3) H ← diag(diag_term) ------------------------------------------------
    fill!(H, 0)
    @inbounds for i in 1:n
        H[i, i] = diag_term[i]
    end

    # --- 4) rank-k update: H ← H - ∑ᵢ σᵢ·W[:,i]·W[:,i]ᵀ -----------------------
    for j in 1:m
        LinearAlgebra.BLAS.syr!('U', -σv[j], view(W, :, j), H)
    end

    # --- 5) mirror upper → lower ----------------------------------------------
    @inbounds for j in 1:n-1, i in j+1:n
        H[i, j] = H[j, i]
    end

    @info "exact dense Hessian built (heterogeneous σ)"
    return nothing
end

# -----------------------------------------------------------------------
# compute the exact affine-constrained Hessian
# -----------------------------------------------------------------------
@doc raw"""
This computes the exact Hessian of the affine-constrained problem,
    namely, the log-UMP has an affine constraint on the allocation
"""
function __compute_exact_hess_afcon!(alg, market::FisherMarket)

    _Hi = (i) -> begin
        _γ = market.x[:, i] .* alg.p / market.w[i]
        ρᵢ = market.ρ[i]
        _W = begin
            ((1 - ρᵢ) * diagm(alg.p .^ 2 ./ _γ) +
             ρᵢ * alg.p * alg.p') ./ (market.w[i]^2)
        end
        _constr_x = market.constr_x[i]
        @assert _constr_x.n == market.n

        # Z = [_W _constr_x.A'; _constr_x.A spzeros(_constr_x.m, _constr_x.m)]
        # rhs = [1 / market.w[i] * I(market.n); zeros(_constr_x.m, _constr_x.n)]
        # sol = Z \ rhs
        # # first n rows
        # _H = sol[1:market.n, :]
        _iW = market.w[i]^2 / (1 - ρᵢ) * diagm(1 ./ alg.p) * (diagm(_γ) - ρᵢ * _γ * _γ') * diagm(1 ./ alg.p)
        _iH = 1 / market.w[i] .* (
            _iW - _iW * _constr_x.A' * inv(_constr_x.A * _iW * _constr_x.A' + 1e-12 * I) * _constr_x.A * _iW
        )
        return _iH
    end
    alg.H .= mapreduce(_Hi, +, alg.sampler.indices, init=spzeros(market.n, market.n))
    @info "use exact Hessian from affine-constrained UMP"
end

# -----------------------------------------------------------------------
# for ArrowDebreuMarket
# -----------------------------------------------------------------------
function __ces_hess_dual!(alg, market::ArrowDebreuMarket)
    # compute 1/σ w_i * log(cs_i'p^{-σ})
    if alg.linsys == :direct
        __compute_exact_hess!(alg, market)
    elseif alg.linsys == :direct_affine
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
    __compute_exact_hess_only_fisher!(alg, market::Market)
    Compute the exact Hessian of the problem, ∇²f, not affine-scaled
    only for the `Fisher` part. i.e., ignore fact that budget is from `<price, endowment>`.
"""
function __compute_exact_hess_only_fisher!(alg, market::ArrowDebreuMarket)
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

function __compute_exact_hess!(alg, market::ArrowDebreuMarket)
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

    # @info "use exact Hessian"
end