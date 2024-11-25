# -----------------------------------------------------------------------
# Hessian Barrier Method (Auction)
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------

using LinearAlgebra, SparseArrays

Base.@kwdef mutable struct HessianBar{T}
    n::Int
    m::Int
    # price
    p::Vector{T}  # current price at k
    pb::Vector{T} # backward price at k-1
    # market excess demand
    z::Vector{T}
    # barrier parameter
    μ::T

    # -------------------------------------------------------------------
    # main iterates
    # -------------------------------------------------------------------
    # dual function value
    φ::T
    # gradient
    ∇::Vector{T}
    gₙ::T # norm of gradient
    dₙ::T # size of Newton step
    α::T  # step size
    # Hessian
    H::SparseMatrixCSC{T,Int}
    # steps
    step::Symbol = :affinescaling

    # -------------------------------------------------------------------
    # timers, tolerances, counters
    # -------------------------------------------------------------------
    ts::Float64
    te::Float64
    tₗ::Float64
    t::Float64
    # iteration counters
    k::Int = 0
    # avg iterations per subproblem
    kᵢ::Float64 = 0.0
    # termination tolerance
    maxiter::Int
    maxtime::Float64
    tol::Float64

    # subproblem solvers
    optimizer::ResponseOptimizer



    function HessianBar(
        n::Int, m::Int, p::Vector{T}, μ::T;
        maxiter::Int=1000,
        maxtime::Float64=100.0,
        tol::Float64=1e-6,
        optimizer::ResponseOptimizer=OptimjlNewtonResponse,
    ) where {T}
        z = rand(n)
        this = new{T}()
        this.n = n
        this.m = m
        this.p = p
        this.φ = -1e6
        this.pb = p
        this.z = z
        this.μ = μ
        this.∇ = zeros(n)
        this.H = spzeros(n, n)
        this.ts = time()
        this.maxiter = maxiter
        this.maxtime = maxtime
        this.tol = tol
        this.k = 0
        this.kᵢ = 0.0
        this.optimizer = optimizer
        return this
    end
end

# -----------------------------------------------------------------------
# compute gradient and Hessian of Newton step of price
# -----------------------------------------------------------------------

function grad!(alg::HessianBar, fisher::FisherMarket)
    alg.∇ .= sum(fisher.x; dims=1)[:] - fisher.q
end

# compute Jacobian: dx/dp
function jacxp(X₂, u, c, w, μ)
    invμ = 1 / μ
    Xc = X₂ * c
    r = w / u^2
    return invμ * X₂ - (invμ^2 * r * Xc * Xc') ./ (1 + invμ * r * c' * Xc)
end

# compute Jacobian dp/dx
function jacpx(Xi₂, u, c, w, μ)
    r = w / u^2
    return μ * Xi₂ + r * c * c'
end

function hess!(alg::HessianBar, fisher::FisherMarket; bool_dbg=false)
    X2 = fisher.x .^ 2
    Di(i) = begin
        X₂ = spdiagm(X2[i, :])
        u = fisher.val_u[i]
        c = fisher.val_∇u[i, :]
        w = fisher.w[i]
        jxp = jacxp(X₂, u, c, w, alg.μ)
        if bool_dbg
            Xi₂ = spdiagm(1 ./ X2[i, :])
            jpx = jacpx(Xi₂, u, c, w, alg.μ)
            @info "jacpx * jacxp - I" maximum(abs.(jpx * jxp - I))
        end
        return jxp
    end
    alg.H = -mapreduce(Di, +, 1:fisher.m, init=spzeros(fisher.n, fisher.n))
end

function eval!(alg::HessianBar, fisher::FisherMarket)
    alg.φ = (
        logbar(fisher.val_u, fisher.w) +
        alg.μ * logbar(fisher.x) +
        alg.μ * logbar(alg.p) +
        alg.p' * alg.∇ - alg.μ * fisher.n
    )
end


# -----------------------------------------------------------------------
# subproblems
# -----------------------------------------------------------------------
@doc raw"""
    play! runs the subproblems as best-response-type mappings
    for all i ∈ I
        solve_substep!(alg, fisher, i; ϵᵢ=ϵᵢ)
    end
    ϵᵢ: the tolerance for the subproblem
"""
function play!(alg::HessianBar, fisher::FisherMarket; ϵᵢ=1e-7, verbose=false)
    _k = 0
    for i in 1:fisher.m
        info = solve_substep!(
            alg, fisher, i;
            ϵᵢ=ϵᵢ
        )
        _k += info.k
        if info.ϵ > ϵᵢ * 1e2
            @warn "subproblem $i is not converged: ϵ: $(info.ϵ)"
        end
    end
    alg.kᵢ = _k / fisher.n
    verbose && validate(fisher, alg.μ)
end

function produce_functions_from_subproblem(
    alg::HessianBar, fisher::FisherMarket, i::Int
)
    _p = alg.p
    _u(x) = fisher.u(x, i)
    _∇u(x) = fisher.∇u(x, i)
    _f(x) = -fisher.w[i] * log(_u(x)) + _p' * x + alg.μ * logbar(x)
    _g(x) = -fisher.w[i] * _∇u(x) / _u(x) + _p - alg.μ ./ x
    _H(x) = begin
        c = _∇u(x)
        u = _u(x)
        r = fisher.w[i] / u^2
        return r * c * c' + alg.μ * spdiagm(1 ./ (x .^ 2))
    end
    return _f, _g, _H, _u, _∇u
end

function solve_substep!(
    alg::HessianBar, fisher::FisherMarket, i::Int;
    ϵᵢ=1e-4
)
    if alg.optimizer.style == :nlp
        # warm-start
        _x₀ = fisher.x[i, :]
        # provide functions
        _f, _g, _H, _u, _∇u = produce_functions_from_subproblem(alg, fisher, i)
        info = solve!(alg.optimizer; f=_f, g=_g, H=_H, x₀=_x₀, tol=ϵᵢ)
        fisher.x[i, :] .= info.x
        fisher.val_u[i] = _u(info.x)
        fisher.val_∇u[i, :] = _∇u(info.x)
        return info
    elseif alg.optimizer.style == :structured
        solve!(alg.optimizer; fisher=fisher, i=i, p=alg.p, μ=alg.μ, verbose=false)
    end
end

# -----------------------------------------------------------------------
# main iterates
# -----------------------------------------------------------------------
function iterate!(alg::HessianBar, fisher::FisherMarket)
    alg.pb .= alg.p
    # update all sub-problems of all agents i ∈ I
    play!(alg, fisher; ϵᵢ=0.1 * alg.μ, verbose=false)

    # -------------------------------------------------------------------
    # compute dual function value, gradient and Hessian
    # !evaluate gradient first;
    grad!(alg, fisher)
    eval!(alg, fisher)
    hess!(alg, fisher)
    # -------------------------------------------------------------------

    # compute affine-scaling Newton step
    if alg.step == :affinescaling
        invp = 1 ./ alg.p
        dp = -(alg.H + alg.μ * spdiagm(invp .^ 2)) \ (alg.∇ - alg.μ ./ invp)
        alg.gₙ = gₙ = norm(alg.∇)
        alg.dₙ = dₙ = norm(dp)
        # update price
        αₘ = min(proj.(-alg.pb ./ dp)..., 1.0)
        alg.α = α = αₘ * 0.995
        alg.p .= alg.pb .+ α * dp
    else

    end

    @assert all(alg.p .> 0)
    alg.te = time()
    alg.t = alg.te - alg.ts
    alg.tₗ = alg.te - alg.ts # todo
    _logline = produce_log(
        __default_logger,
        [alg.k log10(alg.μ) alg.φ (alg.gₙ / fisher.m) alg.dₙ alg.t alg.tₗ alg.α alg.kᵢ]
    )
    println(_logline)
    alg.k += 1
    ϵ = [gₙ dₙ]

    # update barrier parameter
    # if gₙ < alg.μ * 2e1
    # alg.μ *= 0.92
    # end
    alg.μ *= 0.9

    return ϵ
end

function solve!(
    p₀::Vector{T}, alg::HessianBar, fisher::FisherMarket;
    maxiter=1000, maxtime=100.0, tol=1e-6,
    step::Symbol=:affinescaling,
    kwargs...
) where {T}
    println(__default_logger._blockheader)
    # override step choice here
    alg.step = step
    bool_default = isnothing(alg.optimizer)
    bool_default && (alg.optimizer = OptimjlNewtonResponse)
    bool_default && @warn "No optimizer specified! Will use default one: $(alg.optimizer.name)\n"
    @printf " main iterate method           := %s\n" step
    @printf " subproblem solver alias       := %s\n" alg.optimizer.name
    @printf " subproblem solver style       := %s\n" alg.optimizer.style
    # @printf " algorithm description := %s\n" t.DESC
    println(__default_logger._sep)
    _k = 0
    for _ in 1:maxiter
        mod(_k, 20) == 0 && println(__default_logger._loghead)
        ϵ = iterate!(alg, fisher)
        if (max(ϵ...) < alg.tol) || (alg.t >= alg.maxtime) || (_k >= maxiter)
            break
        end
        _k += 1
    end

    println(__default_logger._sep * "✓")
    println(__default_logger._sep)
end

