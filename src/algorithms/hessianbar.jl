# -----------------------------------------------------------------------
# Hessian Barrier Method (Auction)
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------

using LinearAlgebra, SparseArrays

Base.@kwdef mutable struct HessianBar{T} <: Algorithm
    n::Int
    m::Int
    # price
    p::Vector{T}  # current price at k
    λ::Vector{T}  # current dual variable of p at k (if needed)
    pb::Vector{T} # backward price at k-1
    pλ::Vector{T} # backward dual variable of p at k-1
    # market excess demand
    z::Vector{T}
    # barrier parameter
    μ::T

    # -------------------------------------------------------------------
    # main iterates
    # -------------------------------------------------------------------
    # dual function value
    φ::T
    # gradient and norm of scaled gradient
    ∇::Vector{T}
    gₙ::T
    gₜ::T
    # update direction and size of Newton step
    Δ::Vector{T}  # update direction at k-1
    dₙ::T
    # step size
    α::T
    # Hessian
    H::SparseMatrixCSC{T,Int}
    # steps

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

    # -------------------------------------------------------------------
    # name, subproblem solvers
    # -------------------------------------------------------------------
    name::Symbol
    optimizer::ResponseOptimizer
    # linear system solver
    linsys::Symbol
    # random sampler to choose players
    sampler::Sampler
    option_grad::Symbol
    option_step::Symbol



    function HessianBar(
        n::Int,
        m::Int,
        p::Vector{T};
        μ::Float64=1.0,
        maxiter::Int=1000,
        maxtime::Float64=100.0,
        tol::Float64=1e-6,
        optimizer::ResponseOptimizer=OptimjlNewtonResponse,
        option_grad::Symbol=:dual,
        option_step::Symbol=:affine_scaling,
        linsys::Symbol=:none,
        sampler::Sampler=NullSampler(),
    ) where {T}
        z = rand(n)
        this = new{T}()
        this.name = :HessianBar
        this.n = n
        this.m = m
        this.p = p
        this.φ = -1e6
        this.pb = p
        this.z = z
        this.μ = μ
        this.∇ = zeros(n)
        this.Δ = zeros(n)
        this.λ = μ * ones(n)
        this.pλ = μ * ones(n)
        this.H = spzeros(n, n)
        this.ts = time()
        this.maxiter = maxiter
        this.maxtime = maxtime
        this.tol = tol
        this.k = 0
        this.kᵢ = 0.0
        this.optimizer = optimizer
        this.option_grad = option_grad
        this.option_step = option_step
        this.linsys = linsys
        this.sampler = sampler
        return this
    end
end

# -----------------------------------------------------------------------
# main iterates
# -----------------------------------------------------------------------
function iterate!(alg::HessianBar, fisher::FisherMarket)
    alg.pb .= alg.p
    # update all sub-problems of all agents i ∈ I
    if alg.option_grad in [:usex, :dual]
        play!(alg, fisher; ϵᵢ=0.1 * alg.μ, verbose=false)
        # -------------------------------------------------------------------
        # compute dual function value, gradient and Hessian
        # !evaluate gradient first;
        grad!(alg, fisher)
        eval!(alg, fisher)
        hess!(alg, fisher)
    else
        throw(ArgumentError("""
        invalid option for gradient: $(alg.option_grad)\n
        only [:usex, :dual] are supported
        """))
    end


    # -------------------------------------------------------------------
    # choice for Newton step of price
    # -------------------------------------------------------------------
    if alg.option_step == :affine_scaling
        # compute affine-scaling Newton step
        linsolve!(alg, fisher)
        alg.gₙ = gₙ = norm(alg.∇)
        alg.gₜ = gₜ = norm(alg.p .* alg.∇)
        alg.dₙ = dₙ = norm(alg.Δ)
        # update price
        αₘ = min(proj.(-alg.pb ./ alg.Δ)..., 1.0)
        alg.α = α = αₘ * 0.995
        alg.p .= alg.pb .+ α * alg.Δ
    elseif alg.option_step == :augmented
        alg.pλ .= alg.λ
        # construct augmented system
        invp = 1 ./ alg.p
        # Σ = P^{-1}Λ
        σ = alg.λ .* invp
        alg.Δ = -(alg.H + spdiagm(σ)) \ (alg.∇ - alg.μ * invp)
        dλ = -alg.λ + alg.μ * invp - σ .* alg.Δ
        alg.gₙ = gₙ = norm(alg.p .* alg.∇)
        alg.dₙ = dₙ = norm(alg.Δ)
        αₘ = min(proj.(-alg.pb ./ alg.Δ)..., proj.(-alg.pλ ./ dλ)..., 1.0)
        alg.α = α = αₘ * 0.995
        alg.p .= alg.pb .+ α * alg.Δ
        alg.λ .= alg.pλ .+ α * dλ
    end

    @assert all(alg.p .> 0)
    alg.te = time()
    alg.t = alg.te - alg.ts
    alg.tₗ = alg.te - alg.ts # todo
    _logline = produce_log(
        __default_logger,
        [alg.k log10(alg.μ) alg.φ (alg.gₙ / fisher.m) alg.dₙ alg.t alg.tₗ alg.α alg.kᵢ]
    )
    alg.k += 1
    ϵ = [gₙ dₙ]

    # update barrier parameter
    # if gₙ < alg.μ * 2e1
    #     alg.μ *= 0.1
    # end
    # alg.μ = 1e-7
    # alg.μ *= 0.85
    alg.μ *= (1 - min(alg.α * 0.9, 0.98))

    alg.μ = max(alg.μ, 1e-15)
    return ϵ, _logline
end

function opt!(
    alg::HessianBar, fisher::FisherMarket;
    p₀::Union{Vector{T},Nothing}=nothing,
    maxiter=1000,
    maxtime=100.0,
    tol=1e-6,
    keep_traj::Bool=false,
    loginterval=1,
    logfile=nothing,
    reset::Bool=true,
    kwargs...
) where {T}
    ios = logfile === nothing ? [stdout] : [stdout, logfile]
    printto(ios, __default_logger._blockheader)
    traj = []
    bool_default = isnothing(alg.optimizer)
    if reset
        alg.k = 0
        alg.t = 0.0
        alg.tₗ = 0.0
        alg.ts = time()
        alg.maxtime = maxtime
        alg.maxiter = maxiter
        alg.tol = tol
    end
    !isnothing(p₀) && begin
        (alg.p .= p₀)
        printto(ios, "!!!warm-starting price")
    end

    bool_default && printto(ios, "!!!no optimizer specified! will use default one: $(alg.optimizer.name)\n")
    l = @sprintf(" subproblem solver alias       := %s", alg.optimizer.name)
    printto(ios, l)
    l = @sprintf(" subproblem solver style       := %s", alg.optimizer.style)
    printto(ios, l)
    l = @sprintf(" option for gradient           := %s", alg.option_grad)
    printto(ios, l)
    l = @sprintf(" option for step               := %s", alg.option_step)
    printto(ios, l)
    printto(ios, __default_logger._sep)
    _k = 0
    while true
        ϵ, _logline = iterate!(alg, fisher)
        keep_traj && push!(
            traj,
            StateInfo(alg.k, copy(alg.p), alg.∇, alg.gₙ, alg.dₙ, alg.φ)
        )
        mod(_k, 20 * loginterval) == 0 && printto(ios, __default_logger._loghead)
        mod(_k, loginterval) == 0 && printto(ios, _logline)
        if (alg.gₙ < alg.tol) || (alg.dₙ < alg.tol) || (alg.t >= alg.maxtime) || (_k >= alg.maxiter)
            break
        end
        _k += 1
    end


    printto(ios, __default_logger._sep)
    printto(ios, " ✓ final play")
    play!(alg, fisher; ϵᵢ=0.1 * alg.μ, verbose=false, all=true)
    printto(ios, __default_logger._sep)
    return traj
end


