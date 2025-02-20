
function linsolve!(alg, fisher::FisherMarket)
    if alg.option_step == :affine_scaling
        if alg.linsys == :none
            __direct!(alg, fisher)
        elseif alg.linsys == :dr1
            __dr1!(alg, fisher)
        else
        end
    elseif alg.option_step == :primal_dual
        if alg.linsys == :dr1
            __pd!(alg, fisher)
        else
            error("unknown linear system solver: $(alg.linsys)")
        end
    else
        error("unknown step type: $(alg.option_step)")
    end
end

function __direct!(alg, fisher::FisherMarket)
    invp = 1 ./ alg.p
    alg.Δ .= -(alg.H + alg.μ * spdiagm(invp .^ 2)) \ (alg.∇ - alg.μ * invp)
end


function smw_dr1!(Ha::SMWDR1)
    # Compute D⁻¹*a
    Dinva = Ha.a ./ Ha.d
    # Compute the scalar denominator: aᵀ D⁻¹ a - 1/s
    denom = (Ha.a' * Dinva) + 1 / Ha.s
    # inverse of by SMW formula
    Ha.Hi = (x) -> 1 ./ Ha.d .* x - Dinva * (Dinva' * x) / denom
    @debug "compute Sherman-Morrison-Woodbury on DR1"
end

@doc raw"""
    __dr1(alg, fisher::FisherMarket)

    Diagonal + Rank-One method for linear system solver.
        applied in :affine_scaling mode
"""
function __dr1!(alg, fisher::FisherMarket)
    # -------------------------------------------------------------------
    # solve
    # -------------------------------------------------------------------
    alg.Ha.d .+= alg.μ
    smw_dr1!(alg.Ha)
    alg.Δ .= -alg.p .* alg.Ha.Hi(alg.p .* alg.∇ .- alg.μ)
end

@doc raw"""
    __pd(alg, fisher::FisherMarket)

    Solve the linear system in the primal-dual update
        the inverse Hessian is computed by DR1 update.
"""
function __pd!(alg, fisher::FisherMarket)
    # rescale back to the original scale
    n = length(alg.p)
    invp = 1 ./ alg.p
    Σ = invp .* alg.s
    alg.Ha.d .= alg.Ha.d .* invp .^ 2 .+ Σ
    alg.Ha.a .= alg.Ha.a .* invp
    # compute the inverse operator of ∇^2 f + Σ 
    smw_dr1!(alg.Ha)

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
    alg.μ = max(alg.μ, 1e-30)
end
