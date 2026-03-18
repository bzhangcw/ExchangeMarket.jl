using LinearAlgebra, SparseArrays, Printf


function ninf(d::Vector{T}) where T
    return maximum(abs.(d))
end

function ninf_minus(d::Vector{T}) where T
    return maximum(abs.(min.(d, 0)))
end

function linsolve!(alg, market::FisherMarket)
    alg.linsys_msg = ""
    if alg.option_step == :affinesc
        if alg.linsys ∈ [:direct, :direct_afcon]
            return __direct_afsc!(alg, market)
        elseif alg.linsys == :krylov
            return __krylov_afsc!(alg, market)
            # __krylov_afsc_with_H!(alg, fisher)
        elseif alg.linsys ∈ [:DRq, :DRq_rep]
            return __drq_afsc!(alg, market)
        else
            error("unsupported linear system solver: $(alg.linsys) for $(alg.option_step)")
        end
    elseif alg.option_step == :logbar
        if alg.linsys ∈ [:direct]
            # return __direct_pd!(alg, market)
            return __direct_pd_mty!(alg, market)
        elseif alg.linsys ∈ [:DRq, :DRq_rep]
            return __drq_pd!(alg, market)
        elseif alg.linsys == :krylov
            return __krylov_pd!(alg, market)
        else
            error("unsupported linear system solver: $(alg.linsys) for $(alg.option_step)")
        end
    elseif alg.option_step == :damped_ns
        if alg.linsys == :direct
            return __direct_damped!(alg, market)
        elseif alg.linsys ∈ [:DRq, :DRq_rep]
            return __drq_damped!(alg, market)
        else
            error("unsupported linear system solver: $(alg.linsys) for $(alg.option_step)")
        end
    elseif alg.option_step == :homotopy
        if alg.linsys ∈ [:DRq, :DRq_rep]
            return __drq_homo!(alg, market)
        elseif alg.linsys == :krylov
            return __krylov_homo!(alg, market)
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
    return true
end

function __direct_afsc!(alg, market::FisherMarket)
    println("direct afsc solving using exact Hessian")
    dp = diagm(alg.p)
    alg.Δ .= -(dp * alg.H * dp + alg.μ * I) \ (dp * alg.∇ .- alg.μ)
    alg.Δ .= dp * alg.Δ
    return true
end

function __direct_damped!(alg, market::FisherMarket)
    alg.Δ .= -(alg.H) \ (alg.∇)
    return true
end

function __direct_pd!(alg, market::FisherMarket; bool_dbg=true)
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
    bool_dbg && begin
        println("predictor stepsize: $αₘ, g: $(sum(ξ₃)), gₐ: $c₁")
        println("gₐ/g $(c₁/sum(ξ₃))")
        println("μ: $μ")
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
    return true
end

# -------------------------------------------------------------------
# primal-dual mode: solving using Schur complement
# -------------------------------------------------------------------
function __build_schur(alg)
    invp = 1.0 ./ alg.p
    Σ = invp .* alg.s
    HΣ = alg.H + diagm(Σ)
    A, b = alg.linconstr.A, alg.linconstr.b
    Hi(v) = HΣ \ v
    GA = Hi(A')
    return (; invp, Σ, A, b, Hi, GA)
end

function __solve_kkt!(alg, ξ₁, ξ₂, ξ₃, sc)
    (; invp, Σ, A, Hi, GA) = sc
    g = Hi(ξ₁ + ξ₃ .* invp)
    alg.Δy = -inv(A * GA) * (A * g - ξ₂)
    alg.Δ = -g - Hi(A' * alg.Δy)
    alg.Δs = -invp .* ξ₃ - Σ .* alg.Δ
end

function __check_pd_equation(alg, B, ξ₁, ξ₂, ξ₃)
    A = alg.linconstr.A

    r₁ = B * alg.Δ + A' * alg.Δy - alg.Δs + ξ₁
    r₂ = A * alg.Δ + ξ₂
    r₃ = diagm(alg.s) * alg.Δ + diagm(alg.p) * alg.Δs + ξ₃

    return @sprintf("r₁=%.4e, r₂=%.4e, r₃=%.4e", norm(r₁), norm(r₂), norm(r₃))
end

function __direct_pd_mty!(alg, market::FisherMarket)
    println("direct pd using exact Hessian; MTY style")
    n = length(alg.p)
    Q = 1.1e-2
    ψ = 0.9
    # -------------------------------------------------------------------
    # predictor step
    # -------------------------------------------------------------------
    sc = __build_schur(alg)
    A, b = sc.A, sc.b
    ξ₁ = alg.∇ + A' * alg.y - alg.s
    ξ₂ = A * alg.p - b
    # ξ₃ = alg.p .* alg.s .- (1 - alg.α) * ψ * alg.μ
    # ξ₃ = alg.p .* alg.s .- (1 - alg.α) * ψ * alg.μ
    ξ₃ = alg.p .* alg.s .- (Q + √n) / (2Q + √n) * alg.μ
    __solve_kkt!(alg, ξ₁, ξ₂, ξ₃, sc)

    println("predictor step")
    println("    |--- $(__check_pd_equation(alg, alg.H, ξ₁, ξ₂, ξ₃))")

    # stage the old grad
    ∇₀ = copy(alg.∇)

    # stepsize for predictor
    αₘ = min(proj.(-alg.pb ./ alg.Δ)..., proj.(-alg.ps ./ alg.Δs)..., 1.0)
    α = αₘ * 0.9995

    # trial step with stepsize α
    c₁ = similar(alg.p)
    while true
        alg.p .= alg.pb .+ α * alg.Δ
        play!(alg, market; ϵᵢ=0.1 * α * norm(alg.Δ), verbose=false)
        eval!(alg, market)
        grad!(alg, market)
        alg.y .= alg.py .+ α * alg.Δy
        alg.s .= alg.ps .+ α * alg.Δs + (alg.∇ - ∇₀ - α * alg.H * alg.Δ)
        # entail the complementarity equation
        μ = alg.p' * alg.s / n
        c₁ = alg.p .* alg.s .- μ
        # if (norm(c₁) < Q * μ) || (α < 1e-4)
        if (ninf_minus(c₁) < Q * μ) || (α < 1e-4)
            alg.μ = μ
            alg.α = α
            # if alg.α < 0.1
            #     ψ = min(1.0, ψ * 1.1)
            # else
            #     # for large α, we can afford to be more aggressive
            #     ψ = ψ * 0.9
            # end
            break
        end
        α *= 0.8
    end
    @printf("    |--- |c₁|: %.4f\n", norm(c₁))
    @printf("    |--- stepsize: %.4f, μ: %.4f, (p's/n - μ): %.4f\n", α, alg.μ, sum(c₁) / n)
    @printf("    |--- ψ: %.4f\n", ψ)

    # -------------------------------------------------------------------
    # corrector step: motivated by the paper [1]
    # -------------------------------------------------------------------
    # @note, I feel this is not working so well, am I correct?
    # @reference: 
    #   [1].  Sun, J., Zhu, J., Zhao, G.: A predictor-corrector algorithm for a class of nonlinear saddle point problems. SIAM J. Control Optim. 35, 532–551 (1997). https://doi.org/10.1137/S0363012994276111
    # hess!(alg, market)
    # sc = __build_schur(alg)
    # ξ₁ .= alg.∇ + A' * alg.y - alg.s
    # ξ₂ .= A * alg.p - b
    # ξ₃ .= alg.p .* alg.s .- alg.μ
    # println("ξ₃: $(sum(ξ₃))")
    # __solve_kkt!(alg, ξ₁, ξ₂, ξ₃, sc)
    # println("corrector step linear equation error: $(__check_pd_equation(alg, alg.H, ξ₁, ξ₂, ξ₃))")
    # # add the corrector
    # alg.y .+= alg.Δy
    # alg.p .+= alg.Δ
    # # evaluate the new gradient and Hessian
    # ∇₀ = copy(alg.∇)
    # play!(alg, market; verbose=false)
    # grad!(alg, market)
    # alg.s .+= alg.Δs + (alg.∇ - ∇₀ - alg.H * alg.Δ)
    # c₁ = alg.p .* alg.s .- alg.μ
    # println("after corrector |c₁|: $(norm(c₁))")
    # println("    |--- stepsize: 1.0, μ: $(alg.μ), (p's/n - μ): $(sum(c₁) / n)")
    # alg.kᵢ += 1
    return false
end

# -------------------------------------------------------------------
# DRq mode: solving using DRq approximation
# -------------------------------------------------------------------
include("drq.jl")

# -------------------------------------------------------------------
# Krylov mode: solving using Krylov subspace methods
# -------------------------------------------------------------------
include("krylov.jl")