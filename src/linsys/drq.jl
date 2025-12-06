# -------------------------------------------------------------------
# Diagonal + Rank-q Approximation
# -------------------------------------------------------------------

using LinearAlgebra, SparseArrays


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
    __compute_approx_hess_drq!(alg, fisher::FisherMarket)
    Compute needed information of for DRq approximation of the inverse Hessian.
        the inverse Hessian is approximated by the DRq form:
            H⁻¹(x) ≈ (D + ∑ sᵢ aᵢ aᵢ')⁻¹ x
"""
function __compute_approx_hess_drq!(alg, market::FisherMarket)
    groups = alg.Ha.cluster_map
    q = length(groups)
    n = size(market.x, 1)

    # bidding matrix shape: n×m
    γ = (alg.p .* market.x) ./ market.w'
    u = market.w .* market.σ


    alg.Ha.d .= γ * (market.w .+ u)
    alg.Ha.a = [zeros(n) for _ in 1:q]
    alg.Ha.s = zeros(q)


    # for each cluster
    for (i, idxs) in pairs(groups)
        Ω = sum(u[idxs])
        ω = u[idxs] ./ Ω
        ξ = γ[:, idxs] * ω
        alg.Ha.a[i] = ξ
        alg.Ha.s[i] = -Ω
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

@doc raw"""
    smw_drq!(Ha::SMWDRq)

    Iteratively apply Sherman–Morrison updates for each (sᵢ, aᵢ) 
        to compute inverse Hessian operator.
        the inverse Hessian is approximated by the DRq form:
            H⁻¹(x) ≈ (D + ∑ sᵢ aᵢ aᵢ')⁻¹ x
    returns a linear operator of the inverse Hessian.
"""
function smw_drq!(Ha::SMWDRq)
    Dinv = 1.0 ./ Ha.d
    Hinv = x -> Dinv .* x  # start with H⁻¹ = D⁻¹

    for i in 1:length(Ha.s)
        a = Ha.a[i]
        s = Ha.s[i]
        Ha_prev = Hinv

        Hinv = x -> begin
            Ha_x = Ha_prev(x)
            Ha_a = Ha_prev(a)
            denom = 1 / s + dot(a, Ha_a)
            Ha_x .- Ha_a * (dot(a, Ha_x) / denom)
        end
    end
    Ha.Hi = Hinv
    @debug "compute SMW inverse iteratively for DRq"
end

@doc raw"""
    __drq_pd!(alg, fisher::FisherMarket)

    Diagonal + Rank-One method for linear system solver.
        applied in :affinesc mode
"""
function __drq_afsc!(alg, market::FisherMarket)
    # -------------------------------------------------------------------
    # solve
    # -------------------------------------------------------------------
    alg.Ha.d .+= alg.μ
    smw_drq!(alg.Ha)
    alg.Δ .= -alg.p .* alg.Ha.Hi(alg.p .* alg.∇ .- alg.μ)
end

@doc raw"""
    __drq_pd!(alg, fisher::FisherMarket)

    Solve the linear system in the primal-dual update
        the inverse Hessian is computed by DR1 update.
"""
function __drq_pd!(alg, market::FisherMarket)
    # rescale back to the original scale
    # n = length(alg.p)
    # invp = 1 ./ alg.p
    # Σ = invp .* alg.s
    # alg.Ha.d .= alg.Ha.d .* invp .^ 2 .+ Σ
    # alg.Ha.a .= alg.Ha.a .* invp
    # # compute the inverse operator of ∇^2 f + Σ 
    # smw_drq!(alg.Ha)
    n = length(alg.p)
    invp = 1.0 ./ alg.p
    Σ = invp .* alg.s

    # update diagonal
    alg.Ha.d .= alg.Ha.d .* invp .^ 2 .+ Σ

    # update each aᵢ vector
    for i in 1:length(alg.Ha.a)
        alg.Ha.a[i] .= alg.Ha.a[i] .* invp
    end

    # compute the inverse operator of ∇²f + Σ
    smw_drq!(alg.Ha)

    # solve 
    # |∇^2 f + Σ  A' -I | |Δ |   -|ξ₁|
    # |A          0     | |Δy| = -|ξ₂|
    # |S          0   P | |Δs|   -|ξ₃|
    A, b = alg.linconstr.A, alg.linconstr.b
    # compute the inverse of the Hessian
    GA = alg.Ha.Hi(A')
    # -------------------------------------------------------------------
    # predictor step
    # -------------------------------------------------------------------
    ξ₁ = alg.∇ + A' * alg.y - alg.s
    ξ₂ = A * alg.p - b
    ξ₃ = alg.p .* alg.s

    # compute the primal-dual update
    g = alg.Ha.Hi(ξ₁ + ξ₃ .* invp)
    alg.Δy = -inv(A * GA) * (A * g - ξ₂)
    alg.Δ = -g - alg.Ha.Hi(A' * alg.Δy)
    alg.Δs = -invp .* ξ₃ - Σ .* alg.Δ

    # stepsize for predictor
    αₘ = min(proj.(-alg.pb ./ alg.Δ)..., proj.(-alg.ps ./ alg.Δs)..., 1.0)
    α = αₘ * 0.9995

    # trial step with stepsize α
    alg.p .= alg.pb .+ α * alg.Δ
    alg.s .= alg.ps .+ α * alg.Δs
    alg.y .= alg.py .+ α * alg.Δy
    # -------------------------------------------------------------------
    # corrector step
    # -------------------------------------------------------------------
    # new complementarity
    c₁ = alg.p' * alg.s
    μ = (c₁ / sum(ξ₃))^2 * c₁ / n
    begin
        @debug "predictor stepsize: $αₘ, g: $(sum(ξ₃)), gₐ: $c₁"
        @debug "gₐ/g $(c₁/sum(ξ₃))"
        @debug "μ: $μ"
    end
    ξ₁ .= 0
    ξ₂ .= 0
    ξ₃ .= alg.Δ .* alg.Δs .- μ
    # compute the primal-dual update
    g = alg.Ha.Hi(ξ₁ + ξ₃ .* invp)

    # accumulate the corrector
    _cΔy = -inv(A * GA) * (A * g - ξ₂)
    _cΔ = -g - alg.Ha.Hi(A' * _cΔy)
    _cΔs = -invp .* ξ₃ - Σ .* _cΔ
    alg.Δy .+= _cΔy
    alg.Δ .+= _cΔ
    alg.Δs .+= _cΔs
    alg.kᵢ += 1
end

@doc raw"""
    __drq_pd!(alg, fisher::FisherMarket)

    Solve the damped Newton step, 
        see Nesterov (2018) Dvurechensky and Nesterov (2024)
        the inverse Hessian is computed by DR1 update.
"""
function __drq_damped!(alg, market::FisherMarket)
    # -------------------------------------------------------------------
    # solve
    # -------------------------------------------------------------------
    smw_drq!(alg.Ha)

    # damped Newton step
    alg.Δ .= -alg.p .* alg.Ha.Hi(alg.p .* alg.∇)
end

@doc raw"""
    __drq_pd!(alg, fisher::FisherMarket)

    Solve the path-following Newton step, 
        see Nesterov (2018) Dvurechensky and Nesterov (2024)
        the inverse Hessian is computed by DR1 update.
"""
function __drq_homo!(alg, market::FisherMarket)
    # -------------------------------------------------------------------
    # solve
    # -------------------------------------------------------------------
    smw_drq!(alg.Ha)
    γ = 0.158
    alg.k == 0 && (alg.∇₀ .= alg.∇)
    d₀ = alg.p .* alg.Ha.Hi(alg.p .* alg.∇₀)
    denom = sqrt(abs(alg.∇' * d₀))
    alg.μ = max(alg.μ - γ / denom, 0)
    alg.Δ .= -alg.p .* alg.Ha.Hi(alg.p .* (alg.∇ - alg.μ * alg.∇₀))
end
