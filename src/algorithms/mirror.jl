# -----------------------------------------------------------------------
# Mirror Descent Method (Auction) 
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/12/07

# references for step size choices
# [1] Cole R, Fleischer L (2008) Fast-converging tatonnement algorithms for one-time and ongoing market problems. Proceedings of the fortieth annual ACM symposium on Theory of computing. STOC ’08. (Association for Computing Machinery, New York, NY, USA), 315–324.
# [2] Cheung YK, Cole R, Devanur N (2013) Tatonnement beyond gross substitutes? gradient descent to the rescue. Proceedings of the forty-fifth annual ACM symposium on Theory of Computing. STOC ’13. (Association for Computing Machinery, New York, NY, USA), 191–200.
# [3] Cheung YK, Cole R, Tao Y (2018) Dynamics of Distributed Updating in Fisher Markets. Proceedings of the 2018 ACM Conference on Economics and Computation. EC ’18. (Association for Computing Machinery, New York, NY, USA), 351–368.

# -----------------------------------------------------------------------



using LinearAlgebra, SparseArrays

Base.@kwdef mutable struct MirrorDec{T} <: Algorithm
    n::Int
    m::Int
    # price
    p::Vector{T}  # current price at k
    pb::Vector{T} # backward price at k-1
    # market excess demand
    z::Vector{T}

    # -------------------------------------------------------------------
    # main iterates
    # -------------------------------------------------------------------
    # dual function value
    φ::T
    # gradient
    ∇::Vector{T}
    gₙ::T # norm of scaled gradient
    dₙ::T # size of Newton step
    α::Vector{T}  # step size for each j ∈ J
    μ::Float64 = 0.0  # barrier parameter (not used, keep for interface)

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
    kᵢ::Int = 0
    # termination tolerance
    maxiter::Int
    maxtime::Float64
    tol::Float64

    # -------------------------------------------------------------------
    # name, subproblem solvers
    # -------------------------------------------------------------------
    name::Symbol
    optimizer::ResponseOptimizer
    option_grad::Symbol
    option_step::Symbol = :eg
    option_stepsize::Symbol = :cc13

    # random sampler to choose players
    sampler::Sampler

    function MirrorDec(
        n::Int, m::Int,
        p::Vector{T};
        α::Float64=10.0,
        maxiter::Int=1000,
        maxtime::Float64=100.0,
        tol::Float64=1e-6,
        optimizer::ResponseOptimizer=OptimjlNewtonResponse,
        option_grad::Symbol=:dual,
        option_step::Symbol=:eg,
        option_stepsize::Symbol=:cc13,
        sampler::Sampler=NullSampler(),
        kwargs...
    ) where {T}
        z = rand(n)
        this = new{T}()
        this.name = :MirrorDec
        this.n = n
        this.m = m
        this.p = p
        this.φ = -1e6
        this.pb = zeros(n)
        this.z = z
        this.α = α * ones(n)
        this.∇ = zeros(n)
        this.ts = time()
        this.maxiter = maxiter
        this.maxtime = maxtime
        this.tol = tol
        this.k = 0
        this.kᵢ = 0
        this.optimizer = optimizer
        this.option_grad = option_grad
        this.option_step = option_step
        this.option_stepsize = option_stepsize
        this.sampler = sampler
        return this
    end
end

# -----------------------------------------------------------------------
# main iterates
# -----------------------------------------------------------------------
function iterate!(alg::MirrorDec, market::FisherMarket)
    alg.pb .= alg.p
    if (alg.k == 0) && (alg.optimizer.style == :bids)
        # initialize bids
        market.g .= market.x .* (1 ./ alg.p)
        sumb = sum(market.g, dims=2)[:]
        market.g ./= sumb
        market.g .*= market.w'
    end
    # update all sub-problems of all agents i ∈ I
    if alg.option_grad in [:usex, :dual]
        play!(alg, market; ϵᵢ=1e-4, verbose=false)
        # -------------------------------------------------------------------
        # compute dual function value, gradient and Hessian
        # !evaluate gradient first;
        grad!(alg, market)
        eval!(alg, market)
    else
        throw(ArgumentError("""
        invalid option for gradient: $(alg.option_grad)\n
        only [:usex, :dual] are supported
        """))
    end

    # compute mirror-descent step
    if alg.option_stepsize == :cc13
        # use cc'13 step size, see ref[1]
        alg.α .= 5 * max.(market.q, sum(market.x, dims=2)[:])
    elseif alg.option_stepsize == :cc08
    else
        throw(ArgumentError("""
        invalid option for step size: $(alg.option_stepsize)\n
            only [:cc13, :const] are supported
            """))
    end
    if alg.option_step == :eg
        dp = exp.(-alg.∇ ./ alg.α)
        alg.p .= alg.pb .* dp
        alg.dₙ = dₙ = norm(dp)
    elseif alg.option_step == :shmyrev
        if alg.optimizer.style == :bids
            alg.p .= sum(market.g, dims=2)[:]
            alg.dₙ = norm(alg.p - alg.pb)
        else
            dp = sum(market.x, dims=2)[:]
            alg.p .= alg.pb .* dp
            alg.dₙ = dₙ = norm(dp)
        end
    end
    alg.gₙ = gₙ = norm(alg.∇)

    # @assert all(alg.p .>= 0)
    alg.te = time()
    alg.t = alg.te - alg.ts
    alg.tₗ = alg.te - alg.ts # todo
    _logline = produce_log(
        __default_logger,
        [alg.k alg.φ alg.gₙ alg.dₙ alg.t alg.tₗ maximum(alg.α)];
        fo=true
    )
    alg.k += 1
    ϵ = [gₙ alg.dₙ]

    return ϵ, _logline
end

function opt!(
    alg::MirrorDec, market::FisherMarket;
    p₀::Union{Vector{T},Nothing}=nothing,
    maxiter=1000,
    maxtime=100.0,
    tol=1e-6,
    keep_traj::Bool=false,
    loginterval=20,
    logfile=nothing,
    reset::Bool=true,
    # -----------------------------------------------
    # stopping criterion if has a ground truth price
    pₛ::Union{Vector{T},Nothing}=nothing,
    tol_p=1e-5,
    # -----------------------------------------------
    kwargs...
) where {T}
    ios = logfile === nothing ? [stdout] : [stdout, logfile]
    printto(ios, __default_logger._blockheaderfo)
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
    l = @sprintf(" option for step size          := %s", alg.option_stepsize)
    printto(ios, l)
    printto(ios, __default_logger._sep)
    _k = 0
    for _ in 1:maxiter
        _D = Inf
        if !isnothing(pₛ)
            # p - p*
            _D = sum(abs.(alg.p - pₛ))
        end
        keep_traj && push!(
            traj,
            StateInfo(alg.k, copy(alg.p), alg.∇, alg.gₙ, _D, alg.dₙ, alg.φ, alg.t)
        )
        _, _logline = iterate!(alg, market)
        mod(_k, 20 * loginterval) == 0 && printto(ios, __default_logger._logheadfo)
        mod(_k, loginterval) == 0 && printto(ios, _logline)
        if compute_stop(_k, alg, market) || (_D < tol_p)
            # if (alg.dₙ < alg.tol) || (alg.t >= alg.maxtime) || (_k >= alg.maxiter)
            printto(ios, __default_logger._logheadfo)
            printto(ios, _logline)
            break
        end
        _k += 1
    end


    printto(ios, __default_logger._sep)
    printto(ios, " ✓ final play")
    play!(alg, market; verbose=false)
    printto(ios, __default_logger._sep)
    return traj
end

PR = ProportionalResponse = ResponseOptimizer(nothing, :bids, "ProportionalResponse")