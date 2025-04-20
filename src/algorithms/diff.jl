# -----------------------------------------------------------------------
# differentiation utilities
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
@doc raw"""
    SMWDR1{T}
    Structure for the SMW DR1 approximation of the inverse Hessian.
    @note:
        applying the Sherman–Morrison-Woodbury formula to compute the inverse Hessian,
        which is approximated by Diagonal + Rank One (DR1) form:
            diagm(this.d) + this.s * (this.a) * (this.a)'
        then the inverse Hessian, `this.Hi`, is given as the linear operator:
            Hi(x) -> (diagm(this.d) + this.s * (this.a) * (this.a)') \ x
"""
Base.@kwdef mutable struct SMWDR1{T}
    n::Int
    d::Vector{T}
    a::Vector{T}
    s::T
    # linear operator for the inverse Hessian:
    # Hi(x) -> Inverse(H) * x
    Hi::Union{Nothing,Function,SparseMatrixCSC{T,Int}}
    function SMWDR1(n::Int)
        this = new{Float64}()
        this.n = n
        this.d = zeros(n)
        this.a = zeros(n)
        this.s = 0.0
        this.Hi = nothing
        return this
    end
end

@doc raw"""
    __assemble_dr1_approx(Ha::SMWDR1)

    Assemble the DR1 approximation of the inverse Hessian from the SMWDR1 structure.
        for debugging purposes.
"""
function __assemble_dr1_approx(Ha::SMWDR1)
    return spdiagm(Ha.d) + Ha.s * Ha.a * Ha.a'
end

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
    if alg.linsys == :none
        __compute_ces_exact_hess!(alg, fisher)
    else
        # @ref only, for debugging
        # gamma = f1.w ./ sum(f1.w)
        # u = alg.p' ./ f1.w .* f1.x
        # ubar = (gamma'*u)[:]
        pxbar = alg.p .* sum(fisher.x; dims=2)[:]
        # -------------------------------------------------------------------
        # diagonal + rank-one: save unregularized part
        #   of P*(∇^2f)*P
        # -------------------------------------------------------------------
        alg.Ha.d .= (fisher.σ + 1) * pxbar
        alg.Ha.a .= pxbar ./ sum(fisher.w)
        alg.Ha.s = -sum(fisher.w) * fisher.σ
    end
end

function __compute_exact_hess!(alg, fisher::FisherMarket)
    σ = fisher.σ
    w = fisher.w
    _Hi = (i) -> begin
        _H = spdiagm(1 / fisher.val_f[i] .* fisher.val_Hf[:, i]) - (1 / fisher.val_f[i])^2 * fisher.val_∇f[:, i] * fisher.val_∇f[:, i]'
        return _H .* (w[i] / σ)
    end
    alg.H .= mapreduce(_Hi, +, alg.sampler.indices, init=spzeros(fisher.n, fisher.n))
    @info "use exact Hessian"
end

function __compute_exact_affine_scaled_hessop!(buff, alg, fisher::FisherMarket, v)
    b = alg.p .* fisher.x
    _uu = sum(v .* b; dims=2)
    buff .= _uu .* (fisher.σ + 1) - fisher.σ * b * (b' ./ fisher.w * v)
end