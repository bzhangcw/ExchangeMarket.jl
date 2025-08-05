# -------------------------------------------------------------------
# Krylov mode: solving using Krylov subspace methods
# -------------------------------------------------------------------

using LinearOperators, Krylov

@doc raw"""
    LinsysKrylov{T}
    Structure for the inexact Newton using Krylov subspace methods.
    @note:
        applying the Sherman–Morrison–Woodbury formula to compute the inverse Hessian,
"""
Base.@kwdef mutable struct LinsysKrylov{T}
    n::Int
    php_hessop::LinearOperator{T}
    niter::Int
    niter_last::Int
    b::Any # compute p*x
    function LinsysKrylov(n::Int)
        this = new{Float64}()
        this.n = n
        this.php_hessop = LinearOperator(
            # dtype, nrow, ncol, symmetric, hermitian, matvec
            Float64, n, n, true, true,
            (buff, v) -> 0.0
        )
        this.niter = 0
        this.niter_last = 0
        return this
    end
end

function __update_php_hessop!(alg::Algorithm, fisher::FisherMarket)
    alg.Hk.b = alg.p .* fisher.x
    alg.Hk.php_hessop = LinearOperator(
        # dtype, nrow, ncol, symmetric, hermitian, matvec
        Float64, fisher.n, fisher.n, true, true,
        (buff, v) -> __compute_exact_hessop_afscale_optimized!(buff, alg, fisher, v; add_μ=true)
    )
end

@doc raw"""
    __compute_exact_hessop_afsc!(buff, alg, fisher::FisherMarket, v)
    Compute the exact Hessian-vector product operator 
        with affine-scaling P∇²fP + μ (if add_μ=true)
    add_μ means we are using affine-scaling to the barrier problem:
        ∇²f + μ P⁻² = ∇² (f - μ⟨1, log(p)⟩)
"""
function __compute_exact_hessop_afscale!(buff, alg, fisher::FisherMarket, v; add_μ=false)
    b = alg.Hk.b
    _uu = sum(v .* b; dims=2)
    buff .= _uu .* (fisher.σ + 1) - fisher.σ * b * (b' ./ fisher.w * v) + (add_μ * alg.μ) .* v
end

function __compute_exact_hessop_afscale!(alg, fisher::FisherMarket, v; add_μ=false)
    b = alg.Hk.b
    _uu = sum(v .* b; dims=2)
    return _uu .* (fisher.σ + 1) - fisher.σ * b * (b' ./ fisher.w * v) + (add_μ * alg.μ) .* v
end

function __compute_exact_hessop_afscale_optimized!(
    buff::Vector{T}, alg, fisher::FisherMarket{T}, v::Vector{T}; add_μ::Bool=false,
) where {T}

    b = alg.Hk.b              # n×m
    σ = fisher.σ
    μ = alg.μ
    w = fisher.w              # length-m
    n, m = size(b)

    @assert length(v) == n && length(buff) == n && length(w) == m

    row_sum = zeros(T, n)
    b_t_v = zeros(T, m)
    ones_m = ones(T, m)

    # 1) row_sum = b * ones_m     ⇒  Σ_j b[i,j]
    mul!(row_sum, b, ones_m)          # BLAS gemv

    # 2) b_t_v  = b' * v           ⇒  Σ_i b[i,j]*v[i]
    mul!(b_t_v, transpose(b), v)      # BLAS gemv

    # 3) scale   b_t_v  .=  b_t_v ./ w
    @. b_t_v /= w

    # 4) buff    = b * b_t_v       ⇒  s[i] = Σ_j b[i,j]*t[j]
    mul!(buff, b, b_t_v)             # BLAS gemv

    # 5) fuse the rest
    μ_eff = add_μ ? μ : zero(μ)
    @. buff = (σ + 1) * (row_sum * v) - σ * buff + μ_eff * v

    return buff
end

function __krylov_afsc!(alg, fisher::FisherMarket)
    __update_php_hessop!(alg, fisher)

    τ₁ = zeros(fisher.n)
    ExchangeMarket.__compute_exact_hessop_afscale_optimized!(
        τ₁, alg, fisher, ones(fisher.n); add_μ=false
    )
    # diagonal preconditioner
    M₁ = diagm(1 ./ τ₁)
    d, stats = cg(
        alg.Hk.php_hessop, alg.p .* alg.∇ .- alg.μ;
        M=M₁, # diagonal preconditioner
        rtol=max(alg.μ, 1e-10), atol=max(alg.μ, 1e-10)
    )
    alg.Hk.niter += stats.niter
    alg.Hk.niter_last = stats.niter
    @info "apply diagonal preconditioner, niter = $(stats.niter); dim = $(fisher.n)"
    alg.Δ .= -alg.p .* d
end

function __krylov_afsc_with_H!(alg, fisher::FisherMarket)
    __compute_exact_hess!(alg, fisher)
    H = diagm(alg.p) * alg.H * diagm(alg.p)

    # d, stats = cg(alg.Hk.php_hessop, alg.p .* alg.∇ .- alg.μ)
    d, stats = cg(H, alg.p .* alg.∇ .- alg.μ; rtol=alg.μ, atol=alg.μ)
    # d, stats = cg(alg.Hk.php_hessop, alg.p .* alg.∇ .- alg.μ; rtol=√alg.μ * 10, atol=√alg.μ * 10)
    alg.Hk.niter += stats.niter
    alg.Hk.niter_last = stats.niter
    alg.Δ .= -alg.p .* d
end

function __krylov_pd!(alg, fisher::FisherMarket)
    __update_php_hessop!(alg, fisher)
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

    τ₁ = zeros(fisher.n)
    ExchangeMarket.__compute_exact_hessop_afscale_optimized!(
        τ₁, alg, fisher, ones(fisher.n); add_μ=false
    )
    # diagonal preconditioner
    M₁ = diagm(1 ./ τ₁)

    Hi(v) = begin
        d, stats = cg(
            alg.Hk.php_hessop, alg.p .* v;
            M=M₁, # diagonal preconditioner
            rtol=max(alg.μ, 1e-10), atol=max(alg.μ, 1e-10),
            timemax=20.0
        )
        return d .* alg.p
    end

    # solve 
    # |∇^2 f + Σ  A' -I | |Δ |   -|ξ₁|
    # |A          0     | |Δy| = -|ξ₂|
    # |S          0   P | |Δs|   -|ξ₃|
    A, b = alg.linconstr.A, alg.linconstr.b
    # compute the inverse of the Hessian
    # !!! only support a simplex constraint for now
    # GA = alg.Ha.Hi(A')
    GA = Hi(A[:])[:]
    _iAGA = 1 / (A*GA)[]
    # -------------------------------------------------------------------
    # predictor step
    # -------------------------------------------------------------------
    ξ₁ = alg.∇ + A' * alg.y - alg.s
    ξ₂ = A * alg.p - b
    ξ₃ = alg.p .* alg.s

    # compute the primal-dual update
    # g = alg.Ha.Hi(ξ₁ + ξ₃ .* invp)
    g = Hi(ξ₁ + ξ₃ .* invp)
    # accumulate the corrector
    alg.Δy = -_iAGA * (A * g - ξ₂)
    alg.Δ = -g - Hi(A' * alg.Δy)
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
    g = Hi(ξ₁ + ξ₃ .* invp)

    # accumulate the corrector
    _cΔy = -_iAGA * (A * g - ξ₂)
    _cΔ = -g - Hi(A' * _cΔy)
    _cΔs = -invp .* ξ₃ - Σ .* _cΔ
    alg.Δy .+= _cΔy
    alg.Δ .+= _cΔ
    alg.Δs .+= _cΔs
end

function __krylov_homo!(alg, fisher::FisherMarket)
    # -------------------------------------------------------------------
    # solve
    # -------------------------------------------------------------------
    __update_php_hessop!(alg, fisher)
    γ = 0.158
    alg.k == 0 && (alg.∇₀ .= alg.∇)
    # _d = alg.Ha.Hi(alg.p .* alg.∇₀)
    _d, stats = cg(
        alg.Hk.php_hessop, alg.p .* alg.∇₀;
        rtol=max(alg.μ, 1e-10), atol=max(alg.μ, 1e-10)
    )
    alg.Hk.niter += stats.niter
    alg.Hk.niter_last = stats.niter
    # update μ
    d₀ = alg.p .* _d
    denom = sqrt(abs(alg.∇' * d₀))
    alg.μ = max(alg.μ - γ / denom, 0)
    # _d = alg.Ha.Hi(alg.p .* (alg.∇ - alg.μ * alg.∇₀))
    _d, stats = cg(
        alg.Hk.php_hessop, alg.p .* (alg.∇ - alg.μ * alg.∇₀);
        rtol=max(alg.μ, 1e-10), atol=max(alg.μ, 1e-10)
    )
    alg.Hk.niter += stats.niter
    alg.Hk.niter_last = stats.niter

    alg.Δ .= -alg.p .* _d
end
