# -----------------------------------------------------------------------
# differentiation utilities
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
@doc raw"""
    SMWDRq{T}
    Structure for the SMW DRq approximation of the inverse Hessian.
    @note:
        applying the Sherman–Morrison–Woodbury formula to compute the inverse Hessian,
        which is approximated by Diagonal + Rank-q (DRq) form:
            diagm(this.d) + sum(this.s[i] * this.a[i] * this.a[i]' for i in 1:q)
        then the inverse Hessian, `this.Hi`, is given as the linear operator:
            H⁻¹(x) ≈ (D + ∑ sᵢ aᵢ aᵢ')⁻¹ x
"""
Base.@kwdef mutable struct SMWDRq{T}
    n::Int
    d::Vector{T}
    a::Vector{Vector{T}}
    s::Vector{T}
    Hi::Union{Nothing,Function,SparseMatrixCSC{T,Int}}
    cluster_map::Dict{Int,Vector{Int}}
    centers::Union{Nothing,Dict{Int,Vector{T}}}
    cardinality::Vector{T}
    q::Int

    function SMWDRq(n::Int, q::Int, m::Int)
        this = new{Float64}()
        this.n = n
        this.d = zeros(n)
        this.a = [zeros(n) for _ in 1:q]
        this.s = zeros(q)
        this.Hi = nothing
        this.q = q
        # cluster_map::Dict{Int, Vector{Int}}
        #   cluster/group index g = 1, 2, ..., q
        #   ∀g, this.cluster_map[g] ≡ I_g : the set of player indices in group g
        this.cluster_map = Dict(1 => [1:m...])
        # cardinality::Vector{T}
        #   the number of clusters, sᵢ, each player i belongs to
        #   this is used to reweight the x's by the cardinality of the clusters
        #   because we will use it sᵢ times in the DRq approximation
        this.cardinality = ones(m)
        return this
    end
end

@doc raw"""
    update_cluster_map!(alg::Algorithm, cluster_map::Dict{Int,Vector{Int}})
    Update the cluster map of the SMWDRq structure, this decompose the players into clusters.
"""
update_cluster_map!(alg::Algorithm, cluster_map::Dict{Int,Vector{Int}}, cardinality::Vector{Float64}; centers=nothing) = begin
    alg.Ha.q = length(cluster_map)
    alg.Ha.s = zeros(alg.Ha.q)
    alg.Ha.a = [zeros(alg.n) for _ in 1:alg.Ha.q]
    alg.Ha.cluster_map = cluster_map
    alg.Ha.cardinality = cardinality
    if centers !== nothing
        alg.Ha.centers = centers
    end
end

@doc raw"""
    __assemble_drq_approx(Ha::SMWDRq)

    Assemble the DRq approximation of the inverse Hessian from the SMWDRq structure.
        for debugging purposes.
"""
function __assemble_drq_approx(Ha::SMWDRq)
    return spdiagm(Ha.d) + sum(Ha.s[i] * Ha.a[i] * Ha.a[i]' for i in 1:Ha.q)
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
        groups = alg.Ha.cluster_map
        cards = alg.Ha.cardinality
        q = length(groups)
        n = size(fisher.x, 1)

        pxbar = alg.p .* sum(fisher.x; dims=2)[:]
        alg.Ha.d .= (fisher.σ + 1) * pxbar
        alg.Ha.a = [zeros(n) for _ in 1:q]
        alg.Ha.s = zeros(q)

        # reweight the x's by the cardinality of the clusters
        _x = fisher.x ./ cards'

        for (i, idxs) in pairs(groups)
            xsum = sum(_x[:, idxs]; dims=2)[:]
            wsum = sum(fisher.w[idxs])
            ai = (alg.p .* xsum) ./ wsum
            si = -wsum * fisher.σ
            alg.Ha.a[i] = ai
            alg.Ha.s[i] = si
        end
    elseif alg.linsys == :DRq_rep
        # this only works for CES for now
        # TODO: implement to play!
        groups = alg.Ha.cluster_map
        cards = alg.Ha.cardinality
        q = length(groups)
        n = size(fisher.x, 1)
        wealth = Dict(k => sum(fisher.w[v]) for (k, v) in groups)

        alg.Ha.d .= 0.0

        for (k, v) in groups
            _ck = alg.Ha.centers[k]
            _vf, _v∇f, _vHf = fisher.f∇f(alg.p, _ck)
            _x = -wealth[k] ./ _vf ./ fisher.σ .* _v∇f
            _γ = _x .* alg.p / wealth[k]
            alg.Ha.d += _γ * (fisher.σ + 1) * wealth[k]
            alg.Ha.a[k] = _γ
            alg.Ha.s[k] = -wealth[k] * fisher.σ
        end
    else
        throw(ArgumentError("linsys not supported: $(alg.linsys)"))
    end
end

# -----------------------------------------------------------------------
# compute the exact Hessian
# -----------------------------------------------------------------------
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

@doc raw"""
    __compute_exact_hessop_afsc!(buff, alg, fisher::FisherMarket, v)
    Compute the exact Hessian-vector product operator 
        with affine-scaling P∇²fP
"""
function __compute_exact_hessop_afscale!(buff, alg, fisher::FisherMarket, v)
    b = alg.p .* fisher.x
    _uu = sum(v .* b; dims=2)
    buff .= _uu .* (fisher.σ + 1) - fisher.σ * b * (b' ./ fisher.w * v)
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
