# -----------------------------------------------------------------------
# differentiation utilities
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
using LinearAlgebra
# -----------------------------------------------------------------------
# main functions
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

# -----------------------------------------------------------------------
# linear case
# -----------------------------------------------------------------------
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

# :usex mode
function __linear_grad_fromx!(alg, fisher::FisherMarket)
    alg.∇ .= fisher.q .* (alg.sampler.batchsize / fisher.m) - fisher.sumx
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
        X₂ = spdiagm(X2[:, i])
        u = fisher.val_u[i]
        c = fisher.val_∇u[:, i]
        w = fisher.w[i]
        jxp = __linear_jacxp_fromx(X₂, u, c, w, alg.μ)
        if bool_dbg
            Xi₂ = spdiagm(1 ./ X2[:, i])
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
    __ces_grad_dual!(alg, fisher)
end

function __ces_hess!(alg, fisher::FisherMarket; bool_dbg=false)
    __ces_hess_dual!(alg, fisher)
end

function __ces_eval!(alg, fisher::FisherMarket)
    __ces_eval_dual!(alg, fisher)
end


# -----------------------------------------------------------------------
# general CES case: ρ < 1, :dual mode
# -----------------------------------------------------------------------
function __ces_eval_dual!(alg, fisher::FisherMarket)
    σ = fisher.σ
    w = fisher.w
    alg.φ = min(
        alg.p' * fisher.q +
        sum(w[i] * log(fisher.val_u[i]) for i in 1:fisher.m),
        1e8
    )
end

function __ces_grad_dual!(alg, fisher::FisherMarket)
    @assert fisher.ρ < 1
    alg.∇ .= fisher.q .* (alg.sampler.batchsize / fisher.m) - fisher.sumx
end

function __ces_hess_dual!(alg, fisher::FisherMarket)
    # compute 1/σ w_i * log(cs_i'p^{-σ})
    if alg.linsys == :direct
        __compute_exact_hess!(alg, fisher)
    elseif alg.linsys == :direct_affine
        # compute the exact Hessian of the affine-constrained problem
        __compute_exact_hess_afcon!(alg, fisher)
    elseif alg.linsys == :DRq
        __compute_approx_hess_drq!(alg, fisher)
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
    __compute_exact_hess!(alg, fisher::FisherMarket)
    Compute the exact Hessian of the problem, ∇²f, not affine-scaled
"""
function __compute_exact_hess!(alg, fisher::FisherMarket)
    # _Hi = (i) -> begin
    #     _H = spdiagm(1 / fisher.val_f[i] .* fisher.val_Hf[:, i]) - (1 / fisher.val_f[i])^2 * fisher.val_∇f[:, i] * fisher.val_∇f[:, i]'
    #     return _H .* (w[i] / σ)
    # end
    # alg.H .= mapreduce(_Hi, +, alg.sampler.indices, init=spzeros(fisher.n, fisher.n))
    b = alg.p .* fisher.x
    pxbar = sum(b; dims=2)[:]
    γ = 1 ./ fisher.w' .* b
    u = fisher.w .* fisher.σ
    alg.H .= diagm(1 ./ alg.p) * (diagm(pxbar .* (fisher.σ + 1)) - γ * diagm(u) * γ') * diagm(1 ./ alg.p)
    @info "use exact Hessian"
end


function __compute_exact_hess_optimized!(alg, fisher::FisherMarket)
    # — ensure a dense target ———————————————––
    if isa(alg.H, SparseMatrixCSC)
        alg.H = Matrix(alg.H)
    end
    H = alg.H                      # n×n dense
    # --- unpack ---------------------------------------------------------------
    p = alg.p                     # length n
    X = fisher.x                  # n×m
    w = fisher.w                  # length m  (strictly >0)
    σ = fisher.σ                  # scalar, may be negative
    n, m = size(X)

    @assert length(p) == n
    @assert size(H) == (n, n)
    @assert length(w) == m

    W = similar(X)                     # n×m

    # --- 1) diagonal term ------------------------------------------------------
    row_sum = vec(sum(X; dims=2))           # n
    diag_term = @. (σ + 1) * row_sum / p        # n

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

    # --- 4) rank-k update: H ← H - σ·W·Wᵀ --------------------------------------
    #     syrk! only touches the chosen triangle (here 'U')
    LinearAlgebra.BLAS.syrk!('U', 'N', -σ, W, 1.0, H)

    # --- 5) mirror upper → lower ----------------------------------------------
    @inbounds for j in 1:n-1, i in j+1:n
        H[i, j] = H[j, i]
    end

    @info "exact dense Hessian built (σ = $σ)"
    return nothing
end

# -----------------------------------------------------------------------
# compute the exact affine-constrained Hessian
# -----------------------------------------------------------------------
@doc raw"""
This computes the exact Hessian of the affine-constrained problem,
    namely, the log-UMP has an affine constraint on the allocation
"""
function __compute_exact_hess_afcon!(alg, fisher::FisherMarket)

    _Hi = (i) -> begin
        _γ = fisher.x[:, i] .* alg.p / fisher.w[i]
        _W = begin
            ((1 - fisher.ρ) * diagm(alg.p .^ 2 ./ _γ) +
             fisher.ρ * alg.p * alg.p') ./ (fisher.w[i]^2)
        end
        _constr_x = fisher.constr_x[i]
        @assert _constr_x.n == fisher.n

        # Z = [_W _constr_x.A'; _constr_x.A spzeros(_constr_x.m, _constr_x.m)]
        # rhs = [1 / fisher.w[i] * I(fisher.n); zeros(_constr_x.m, _constr_x.n)]
        # sol = Z \ rhs
        # # first n rows
        # _H = sol[1:fisher.n, :]
        _iW = fisher.w[i]^2 / (1 - fisher.ρ) * diagm(1 ./ alg.p) * (diagm(_γ) - fisher.ρ * _γ * _γ') * diagm(1 ./ alg.p)
        _iH = 1 / fisher.w[i] .* (
            _iW - _iW * _constr_x.A' * inv(_constr_x.A * _iW * _constr_x.A' + 1e-12 * I) * _constr_x.A * _iW
        )
        return _iH
    end
    alg.H .= mapreduce(_Hi, +, alg.sampler.indices, init=spzeros(fisher.n, fisher.n))
    @info "use exact Hessian from affine-constrained UMP"
end
