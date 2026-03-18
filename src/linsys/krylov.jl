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

function __update_php_hessop!(alg::Algorithm, market::FisherMarket)
    alg.Hk.b = alg.p .* market.x
    alg.Hk.php_hessop = LinearOperator(
        # dtype, nrow, ncol, symmetric, hermitian, matvec
        Float64, market.n, market.n, true, true,
        (buff, v) -> __compute_exact_hessop_afscale_optimized!(buff, alg, market, v; add_μ=true)
    )
end

function __krylov_afsc!(alg, market::FisherMarket)
    if alg.optimizer.style == :lse
        return __lse_krylov_afsc!(alg, market)
    end
    __update_php_hessop!(alg, market)

    τ₁ = zeros(market.n)
    __compute_exact_hessop_afscale_optimized!(
        τ₁, alg, market, ones(market.n); add_μ=false
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
    alg.Δ .= -alg.p .* d
    alg.kᵢ = alg.Hk.niter
    if stats.niter >= 10
        alg.linsys_msg = @sprintf("        |-> apply diagonal scaling, niter = %d; dim = %d", stats.niter, market.n)
    end
    return true
end

function __krylov_pd!(alg, market::FisherMarket)
    __update_php_hessop!(alg, market)
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

    τ₁ = zeros(market.n)
    ExchangeMarket.__compute_exact_hessop_afscale_optimized!(
        τ₁, alg, market, ones(market.n); add_μ=false
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
    alg.kᵢ = alg.Hk.niter
    return true
end

function __krylov_homo!(alg, market::FisherMarket)
    # -------------------------------------------------------------------
    # solve
    # -------------------------------------------------------------------
    __update_php_hessop!(alg, market)
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
    alg.kᵢ = alg.Hk.niter
    return true
end

@doc raw"""
    __lse_krylov_afsc!(alg, market::FisherMarket)

Krylov (CG) solve for the LSE dual affine-scaled Newton system:
    H̃ d = P∇φ - μ1,  then Δ = -Pd.

Uses `__lse_hessop_afscale!` for the matrix-free Hessian-vector product
and a Neumann series preconditioner (order `p`):

```math
M_p^{-1} v = \hat{H}^{-1} \sum_{k=0}^{p} (R_{\mathrm{off}} \hat{H}^{-1})^k v
```

where `R_off v = Σᵢ βᵢ ⟨eᵢ,v⟩ eᵢ - Σ⊙v` is applied matrix-free via
pre-computed `eᵢ`, `βᵢ`, `Σ` from `__lse_precond_data`.
Setting `p=0` recovers the diagonal preconditioner.
"""
function __lse_krylov_afsc!(alg, market::FisherMarket)
    n = market.n
    # Hessian-vector product operator (with μ)
    hessop = LinearOperator(
        Float64, n, n, true, true,
        (buf, v) -> __lse_hessop_afscale!(buf, alg, market, v; add_μ=true)
    )
    # pre-compute preconditioner data
    (; diag_H, Σ, matE, β) = __lse_precond_data(alg, market)
    diag_Hμ_inv = 1.0 ./ (diag_H .+ alg.μ)

    # Neumann preconditioner M_p⁻¹ (p_order=1 by default)
    p_order = 10
    println("Neumann preconditioner M_p⁻¹ (p_order=$p_order)")
    tmp = similar(diag_H)
    M₁ = LinearOperator(Float64, n, n, true, true,
        (buf, v) -> begin
            # u₀ = Ĥ_μ⁻¹ v
            u = diag_Hμ_inv .* v
            buf .= u
            for _ in 1:p_order
                __lse_offdiag_matvec!(tmp, matE, β, Σ, u)
                u = diag_Hμ_inv .* tmp
                buf .+= u
            end
        end
    )
    d, stats = cg(
        hessop, alg.p .* alg.∇ .- alg.μ;
        M=M₁,
        rtol=max(alg.μ, 1e-10), atol=max(alg.μ, 1e-10),
        itmax=20
    )
    alg.Hk.niter += stats.niter
    alg.Hk.niter_last = stats.niter
    alg.Δ .= -alg.p .* d
    alg.kᵢ = alg.Hk.niter
    if stats.niter >= 10
        alg.linsys_msg = @sprintf("        |-> LSE Neumann(%d) precond CG, niter = %d; dim = %d", p_order, stats.niter, n)
    end
    return true
end
