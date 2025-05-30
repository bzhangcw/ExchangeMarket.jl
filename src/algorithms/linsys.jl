
function linsolve!(alg, fisher::FisherMarket)
    if alg.option_step == :affinesc
        if alg.linsys == :direct
            __direct!(alg, fisher)
        elseif alg.linsys == :DRq
            __drqaf(alg, fisher)
        else
        end
    elseif alg.option_step == :logbar
        if alg.linsys == :DRq
            __drqpd!(alg, fisher)
        else
            error("unsupported linear system solver: $(alg.linsys) for $(alg.option_step)")
        end
    elseif alg.option_step == :damped_ns
        if alg.linsys == :direct
            __directdamped!(alg, fisher)
        elseif alg.linsys == :DRq
            __drqdamped(alg, fisher)
        else
            error("unsupported linear system solver: $(alg.linsys) for $(alg.option_step)")
        end
    elseif alg.option_step == :homotopy
        if alg.linsys == :DRq
            __drqhomo!(alg, fisher)
        else
            error("unsupported linear system solver: $(alg.linsys) for $(alg.option_step)")
        end
    else
        error("unknown step type: $(alg.option_step)")
    end
end
# -------------------------------------------------------------------
# Direct mode: solving using exact Hessian ops
# -------------------------------------------------------------------
function __direct!(alg, fisher::FisherMarket)
    invp = 1 ./ alg.p
    alg.Δ .= -(alg.H + alg.μ * spdiagm(invp .^ 2)) \ (alg.∇ - alg.μ * invp)
end

function __directdamped!(alg, fisher::FisherMarket)
    alg.Δ .= -(alg.H) \ (alg.∇)
end

# -------------------------------------------------------------------
# Diagonal + Rank-q Approximation
# -------------------------------------------------------------------
@doc raw"""
    smw_drq!(Ha::SMWDRq)

    Iteratively apply Sherman–Morrison updates for each (sᵢ, aᵢ) to compute inverse Hessian operator.
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
    __drqpd!(alg, fisher::FisherMarket)

    Diagonal + Rank-One method for linear system solver.
        applied in :affinesc mode
"""
function __drqaf(alg, fisher::FisherMarket)
    # -------------------------------------------------------------------
    # solve
    # -------------------------------------------------------------------
    alg.Ha.d .+= alg.μ
    smw_drq!(alg.Ha)
    alg.Δ .= -alg.p .* alg.Ha.Hi(alg.p .* alg.∇ .- alg.μ)
end

@doc raw"""
    __drqpd!(alg, fisher::FisherMarket)

    Solve the linear system in the primal-dual update
        the inverse Hessian is computed by DR1 update.
"""
function __drqpd!(alg, fisher::FisherMarket)
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
end


function __drqdamped(alg, fisher::FisherMarket)
    # -------------------------------------------------------------------
    # solve
    # -------------------------------------------------------------------
    smw_drq!(alg.Ha)

    # damped Newton step
    alg.Δ .= -alg.p .* alg.Ha.Hi(alg.p .* alg.∇)
end

function __drqhomo!(alg, fisher::FisherMarket)
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
