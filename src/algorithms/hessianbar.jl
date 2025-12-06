# -----------------------------------------------------------------------
# Hessian Barrier Method (Auction)
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# @reference:
# [1] C Zhang, C He, B Jiang, Y Ye, Price updates by interior-point algorithms, working paper (2025)
# [2] Dvurechensky, P., Nesterov, Y.: Improved global performance guarantees of second-order methods in convex minimization, http://arxiv.org/abs/2408.11022, (2024)
# [3] Nesterov, Y.: Lectures on Convex Optimization. Springer International Publishing, Cham (2018)
# [4] Nesterov, Y., Nemirovskii, A.: Interior-Point Polynomial Algorithms in Convex Programming. Society for Industrial and Applied Mathematics (1994)

# -----------------------------------------------------------------------

using LinearAlgebra, SparseArrays

Base.@kwdef mutable struct HessianBar{T} <: Algorithm
    n::Int
    m::Int
    # price
    p::Vector{T}  # current price at k
    pb::Vector{T} # backward price at k-1
    s::Vector{T}  # current dual variable of p at k (if needed)
    ps::Vector{T} # backward dual variable of p at k-1
    y::Vector{T}  # current dual variable of linear constraint at k
    py::Vector{T} # backward dual variable of linear constraint at k-1
    # market excess demand
    z::Vector{T}
    # barrier parameter
    μ::T

    # -------------------------------------------------------------------
    # main iterates
    # -------------------------------------------------------------------
    # dual function value
    #  i.e., the estimate of log(sum(u)) at the best response
    φ::T
    # gradient and norm of scaled gradient
    ∇::Vector{T}
    ∇₀::Vector{T}
    gₙ::T
    gₜ::T
    dₙ::T
    # update direction and size of Newton step
    Δ::Vector{T}   # update direction of p at k-1
    Δs::Vector{T}  # update direction of s at k-1
    Δy::Vector{T}  # update direction of y at k-1
    # step size
    α::T
    # Hessian
    H::Union{SparseMatrixCSC{T,Int},Matrix{T}}
    Ha::SMWDRq{T}
    Hk::LinsysKrylov{T}

    # -------------------------------------------------------------------
    # timers, tolerances, counters
    # -------------------------------------------------------------------
    ts::Float64
    te::Float64
    tₗ::Float64 # time for collecting best responses
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
    # linear system solver
    linsys::Symbol
    # message for linear system solver
    linsys_msg::String
    # random sampler to choose players
    sampler::Sampler
    option_grad::Symbol
    option_step::Symbol
    option_mu::Symbol

    # -------------------------------------------------------------------
    # linear constraint
    # -------------------------------------------------------------------
    linconstr::Union{LinearConstr,Nothing}


    function HessianBar(
        n::Int,
        m::Int,
        p::Vector{T};
        μ::Float64=0.01,
        maxiter::Int=1000,
        maxtime::Float64=100.0,
        tol::Float64=1e-6,
        optimizer::ResponseOptimizer=CESAnalytic,
        option_grad::Symbol=:dual,
        option_step::Symbol=:affinesc,
        option_mu::Symbol=:normal,
        linconstr::Union{LinearConstr,Nothing}=nothing,
        linsys::Symbol=:DRq,
        sampler::Sampler=NullSampler(),
        # optional; needed if use primal-dual step
        s::Union{Vector{T},Nothing}=nothing,
        y::Union{Vector{T},Nothing}=nothing,
    ) where {T}
        z = rand(n)
        this = new{T}()
        this.name = :HessianBar
        this.n = n
        this.m = m
        this.φ = -1e6
        this.z = z
        this.∇ = zeros(n)
        this.∇₀ = zeros(n)
        this.H = spzeros(n, n)
        this.Ha = SMWDRq(n, 1, m)
        this.Hk = LinsysKrylov(n)
        this.ts = time()
        this.maxiter = maxiter
        this.maxtime = maxtime
        this.tol = tol
        # main iterations
        this.k = 0
        # auxiliary iterations; 
        # e.g., for CG it means the number of iterations 
        #   for the Krylov subspace method
        this.kᵢ = 0
        this.optimizer = optimizer
        # options
        this.option_grad = option_grad
        this.option_step = option_step
        this.option_mu = option_mu
        this.linsys = linsys
        this.sampler = sampler
        # linear constraint
        # todo: this is the linear constraint for prices,
        #       move to FisherMarket later
        this.linconstr = linconstr
        # initialize vectors
        this.p = copy(p)
        this.pb = zeros(n)
        this.Δ = zeros(n)
        if !isnothing(s)
            this.s = copy(s)
        else
            this.s = copy(this.p)
        end

        if this.option_mu ∈ [:pred_corr, :normal]
            this.μ = this.p' * this.s / this.n
        else
            this.μ = μ
        end

        this.ps = zeros(n)
        this.Δs = zeros(n)

        # needed if linear constraint is present
        if !isnothing(linconstr)
            if !isnothing(y)
                @assert length(y) == linconstr.m
                this.y = copy(y)
            else
                this.y = zeros(linconstr.m)
            end
            this.py = zeros(linconstr.m)
            this.Δy = zeros(linconstr.m)
        end
        return this
    end
end

function init!(alg::HessianBar, market::FisherMarket)
    market.g .= market.x .* (1 ./ alg.p)
    sumb = sum(market.g, dims=2)[:]
    market.g ./= sumb
    market.g .*= market.w'
    play!(alg, market; ϵᵢ=0.1 * alg.μ, verbose=false, style=:bids)
    alg.p .= sum(market.g, dims=2)[:]
    grad!(alg, market)
    eval!(alg, market)
    alg.gₙ = norm(alg.∇)
    alg.α = 1.0
    alg.gₜ = norm(alg.p .* alg.∇)
    alg.dₙ = norm(alg.p - alg.pb)
    _logline = produce_log(
        __default_logger,
        [alg.k log10(alg.μ) alg.φ alg.gₙ alg.dₙ alg.t alg.tₗ alg.α]
    )
    alg.k += 1


    return false, _logline
end

# -----------------------------------------------------------------------
# main iterates
# -----------------------------------------------------------------------
function iterate!(alg::HessianBar, market::FisherMarket)
    alg.pb .= alg.p
    # update all sub-problems of all agents i ∈ I
    if alg.option_grad in [:usex, :dual, :aff]
        play!(alg, market; ϵᵢ=0.1 * alg.μ, verbose=false)
        # -------------------------------------------------------------------
        # compute dual function value, gradient and Hessian
        # !evaluate gradient first;
        grad!(alg, market)
        eval!(alg, market)
        hess!(alg, market)
    else
        throw(ArgumentError("""
        invalid option for gradient: $(alg.option_grad)\n
        only [:usex, :dual] are supported
        """))
    end

    # -------------------------------------------------------------------
    # choice for Newton step of price
    #   I. methods that use the standard logarithmic barrier B
    # -------------------------------------------------------------------
    #  B(p) = -<log(p), 1>
    if alg.option_step == :affinesc
        @debug "use (primal) affine scaling step"
        # compute affine-scaling Newton step
        linsolve!(alg, market)
        alg.gₙ = gₙ = norm(alg.∇)
        alg.gₜ = gₜ = norm(alg.p .* alg.∇ .- alg.μ)
        alg.dₙ = dₙ = norm(alg.Δ)
        # update price
        αₘ = min(proj.(-(alg.pb) ./ alg.Δ)..., 1.0)
        alg.α = α = αₘ * 0.9995
        alg.p .= alg.pb .+ α * alg.Δ

    elseif alg.option_step == :logbar
        @debug "use primal-dual step"
        alg.ps .= alg.s
        alg.py .= alg.y

        linsolve!(alg, market)
        # standard ℓ₂ norm
        alg.gₙ = gₙ = norm(alg.∇)
        # local dual norm
        alg.gₜ = gₜ = norm(alg.p .* alg.∇ .- alg.μ)
        alg.dₙ = dₙ = norm(alg.Δ)
        if any(isnan.([gₜ, gₙ, dₙ]))
            return true, ""
        end

        # step size
        αₘ = min(proj.(-alg.pb ./ alg.Δ)..., proj.(-alg.ps ./ alg.Δs)..., 1.0)
        alg.α = α = αₘ * 0.9995
        alg.p .= alg.pb .+ α * alg.Δ
        alg.s .= alg.ps .+ α * alg.Δs
        alg.y .= alg.py .+ α * alg.Δy
    end

    # -------------------------------------------------------------------
    # II. methods that do not use the canonical logarithmic barrier 
    #   treat as the self-concordant function directly;
    #   see Section 2&3 in [2]
    if alg.option_step == :damped_ns
        @debug """
            use damped Newton step
        """

        linsolve!(alg, market)
        alg.gₙ = gₙ = norm(alg.∇)
        alg.gₜ = gₜ = norm(alg.p .* alg.∇)
        alg.dₙ = dₙ = norm(alg.Δ)

        # newton decrement
        λ = sqrt(-alg.∇' * alg.Δ)
        # update price
        alg.α = α = αₘ = 1.0 / (λ + 1.0)
        # alg.α = α = αₘ = 0.999
        alg.p .= alg.pb .+ α * alg.Δ

    elseif alg.option_step == :homotopy
        linsolve!(alg, market)
        alg.gₙ = gₙ = norm(alg.∇)
        alg.gₜ = gₜ = norm(alg.p .* alg.∇)
        alg.dₙ = dₙ = norm(alg.Δ)
        # update price
        # alg.α = α = αₘ = 1.0
        # alg.α = α = αₘ = 0.999
        alg.α = α = αₘ = min(proj.(-(alg.pb) ./ alg.Δ)..., 1.0)
        alg.p .= alg.pb .+ α * alg.Δ
    end

    alg.te = time()
    alg.t = alg.te - alg.ts
    _logline = produce_log(
        __default_logger,
        [alg.k log10(alg.μ) alg.φ alg.gₙ alg.dₙ alg.t alg.tₗ alg.α]
    )
    alg.k += 1
    ϵ = [alg.gₙ alg.dₙ]

    # update barrier parameter
    if alg.option_mu == :normal
        alg.μ *= (1 - min(alg.α * 0.98, 0.98))
        alg.μ = max(alg.μ, 1e-20)
    elseif alg.option_mu == :pred_corr
        alg.μ = max(alg.p' * alg.s / alg.n, 1e-25)
    elseif alg.option_mu == :strict
        println("gt: $(alg.gₜ)")
        if alg.gₜ < 1e-1 * alg.μ
            alg.μ *= 1e-1
        end
    elseif alg.option_mu == :nothing
        # not needed for e.g., Homotopy
    else
        # not needed
        @warn "unknown option for μ: $(alg.option_mu)"
        @warn "not updating μ"
    end
    return false, _logline
end

function opt!(
    alg::HessianBar, market::FisherMarket;
    p₀::Union{Vector{T},Nothing}=nothing,
    maxiter=1000,
    maxtime=100.0,
    tol=1e-6,
    keep_traj::Bool=false,
    loginterval=1,
    logfile=nothing,
    reset::Bool=true,
    bool_init_phase::Bool=true,
    # -----------------------------------------------
    # stopping criterion if has a ground truth price
    pₛ::Union{Vector{T},Nothing}=nothing,
    tol_p=1e-5,
    kwargs...
    # -----------------------------------------------
) where {T}
    logfile = logfile === nothing ? open(
        joinpath(LOGDIR, "log-hessianbar-$(current_date()).log"), "a"
    ) : logfile
    ios = [stdout, logfile]
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
    l = @sprintf(" lin-system solver alias       := %s", alg.linsys)
    if alg.linsys == :krylov
        l = l * " + optimal diagonal scaling"
        printto(ios, l)
    end
    l = @sprintf(" option for gradient           := %s", alg.option_grad)
    printto(ios, l)
    l = @sprintf(" option for step               := %s", alg.option_step)
    printto(ios, l)
    l = @sprintf(" option for μ                  := %s", alg.option_mu)
    printto(ios, l)
    if (alg.option_step == :affinesc) && (!isnothing(alg.linconstr))
        printto(ios, " !!! linear constraint is not supported for affine-scaling and will be ignored")
    end
    printto(ios, __default_logger._sep)
    flush.(ios)
    _k = 0
    bool_early_stop = false
    while true
        _D = Inf
        if !isnothing(pₛ)
            # p - p*
            _D = sum(abs.(alg.p - pₛ))
        end
        keep_traj && push!(
            traj,
            StateInfo(_k, copy(alg.p), alg.∇, alg.gₙ, _D, alg.dₙ, alg.φ, alg.t)
        )
        if alg.k == 0 && bool_init_phase == true
            printto(ios, "running Phase I...")
            printto(ios, __default_logger._loghead)
            bool_early_stop, _logline = init!(alg, market)
            mod(_k, loginterval) == 0 && printto(ios, _logline)
            printto(ios, __default_logger._sep)
            printto(ios, "running Phase II...")
        else
            mod(_k, 20 * loginterval) == 0 && printto(ios, __default_logger._loghead)
            bool_early_stop, _logline = iterate!(alg, market)
            mod(_k, loginterval) == 0 && begin
                printto(ios, _logline)
                if !isempty(alg.linsys_msg)
                    printto(ios, alg.linsys_msg)
                end
            end
            _k += 1
        end
        bool_early_stop && break

        if (alg.gₙ < alg.tol) || (alg.dₙ < alg.tol^2) || (alg.t >= alg.maxtime) || (_k >= alg.maxiter)
            break
        end
        flush.(ios)
    end


    printto(ios, __default_logger._sep)
    printto(ios, " ✓  final play")
    play!(alg, market; ϵᵢ=0.1 * alg.μ, verbose=false, all=true, timed=false)
    l = @sprintf(" ✓  finished in        %4d steps", alg.k)
    printto(ios, l)
    l = @sprintf(" ✓  using subiter.     %4d steps", alg.kᵢ)
    printto(ios, l)
    l = @sprintf("             in %.5e seconds", alg.t)
    printto(ios, l)
    l = @sprintf("  best-resp. in %.5e seconds ", alg.tₗ)
    printto(ios, l)
    l = @sprintf("            avg %.5e seconds ", alg.tₗ / alg.k)
    printto(ios, l)
    l = @sprintf("          usage %.2f%%", (alg.tₗ / alg.t) * 100)
    printto(ios, l)
    bool_early_stop && printto(ios, " !!! early stopping")
    printto(ios, __default_logger._sep)
    flush.(ios)
    close(logfile)
    return traj
end


