using LinearAlgebra, SparseArrays


function linsolve!(alg, market::FisherMarket)
    alg.linsys_msg = ""
    if alg.option_step == :affinesc
        if alg.linsys ∈ [:direct, :direct_afcon]
            __direct_afsc!(alg, market)
        elseif alg.linsys == :krylov
            __krylov_afsc!(alg, market)
            # __krylov_afsc_with_H!(alg, fisher)
        elseif alg.linsys ∈ [:DRq, :DRq_rep]
            __drq_afsc!(alg, market)
        else
            error("unsupported linear system solver: $(alg.linsys) for $(alg.option_step)")
        end
    elseif alg.option_step == :logbar
        if alg.linsys ∈ [:direct]
            __direct_pd!(alg, market)
        elseif alg.linsys ∈ [:DRq, :DRq_rep]
            __drq_pd!(alg, market)
        elseif alg.linsys == :krylov
            __krylov_pd!(alg, market)
        else
            error("unsupported linear system solver: $(alg.linsys) for $(alg.option_step)")
        end
    elseif alg.option_step == :damped_ns
        if alg.linsys == :direct
            __direct_damped!(alg, market)
        elseif alg.linsys ∈ [:DRq, :DRq_rep]
            __drq_damped!(alg, market)
        else
            error("unsupported linear system solver: $(alg.linsys) for $(alg.option_step)")
        end
    elseif alg.option_step == :homotopy
        if alg.linsys ∈ [:DRq, :DRq_rep]
            __drq_homo!(alg, market)
        elseif alg.linsys == :krylov
            __krylov_homo!(alg, market)
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
function __direct!(alg, market::FisherMarket)
    invp = 1 ./ alg.p
    alg.Δ .= -(alg.H + alg.μ * spdiagm(invp .^ 2)) \ (alg.∇ - alg.μ * invp)
end

function __direct_afsc!(alg, market::FisherMarket)
    println("direct afsc solving using exact Hessian")
    dp = diagm(alg.p)
    alg.Δ .= -(dp * alg.H * dp + alg.μ * I) \ (dp * alg.∇ .- alg.μ)
    alg.Δ .= dp * alg.Δ
end

function __direct_damped!(alg, market::FisherMarket)
    alg.Δ .= -(alg.H) \ (alg.∇)
end

function __direct_pd!(alg, market::FisherMarket)
    println("direct pd using exact Hessian")
    n = length(alg.p)
    invp = 1.0 ./ alg.p
    Σ = invp .* alg.s

    # H + Σ
    HΣ = alg.H + diagm(Σ)
    Hi(v) = HΣ \ v

    # solve
    # |∇²f + Σ  A' -I | |Δ |   -|ξ₁|
    # |A         0     | |Δy| = -|ξ₂|
    # |S         0   P | |Δs|   -|ξ₃|
    A, b = alg.linconstr.A, alg.linconstr.b
    GA = Hi(A')
    # -------------------------------------------------------------------
    # predictor step
    # -------------------------------------------------------------------
    ξ₁ = alg.∇ + A' * alg.y - alg.s
    ξ₂ = A * alg.p - b
    ξ₃ = alg.p .* alg.s

    g = Hi(ξ₁ + ξ₃ .* invp)
    alg.Δy = -inv(A * GA) * (A * g - ξ₂)
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
    g = Hi(ξ₁ + ξ₃ .* invp)

    # accumulate the corrector
    _cΔy = -inv(A * GA) * (A * g - ξ₂)
    _cΔ = -g - Hi(A' * _cΔy)
    _cΔs = -invp .* ξ₃ - Σ .* _cΔ
    alg.Δy .+= _cΔy
    alg.Δ .+= _cΔ
    alg.Δs .+= _cΔs
    alg.kᵢ += 1
end

# -------------------------------------------------------------------
# DRq mode: solving using DRq approximation
# -------------------------------------------------------------------
include("drq.jl")

# -------------------------------------------------------------------
# Krylov mode: solving using Krylov subspace methods
# -------------------------------------------------------------------
include("krylov.jl")