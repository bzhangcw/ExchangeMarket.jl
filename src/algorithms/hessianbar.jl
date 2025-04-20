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
    H::SparseMatrixCSC{T,Int}
    Ha::SMWDR1{T}

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
        option_step::Symbol=:affine_scaling,
        option_mu::Symbol=:normal,
        linconstr::Union{LinearConstr,Nothing}=nothing,
        linsys::Symbol=:dr1,
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
        this.μ = μ
        this.∇ = zeros(n)
        this.∇₀ = zeros(n)
        this.H = spzeros(n, n)
        this.Ha = SMWDR1(n)
        this.ts = time()
        this.maxiter = maxiter
        this.maxtime = maxtime
        this.tol = tol
        this.k = 0
        this.kᵢ = 0.0
        this.optimizer = optimizer
        this.option_grad = option_grad
        this.option_step = option_step
        this.option_mu = option_mu
        this.linsys = linsys
        this.sampler = sampler
        # linear constraint
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

        (this.option_mu != :nothing) && (this.μ = this.p' * this.s / this.n)

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

# -----------------------------------------------------------------------
# main iterates
# -----------------------------------------------------------------------
function iterate!(alg::HessianBar, fisher::FisherMarket)
    alg.pb .= alg.p
    # if (alg.k == 0)
    #     fisher.b .= (1 ./ alg.p') .* fisher.x
    #     sumb = sum(fisher.b, dims=2)[:]
    #     fisher.b ./= sumb
    #     fisher.b .*= fisher.w
    #     fisher.p .= sum(fisher.b, dims=1)[:]
    # end
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
    #   I. methods that use the canonical logarithmic barrier 
    # -------------------------------------------------------------------
    #  B(p) = -<log(p), 1>
    if alg.option_step == :affine_scaling
        if !isnothing(alg.linconstr)
            @warn "linear constraint is not supported for affine scaling"
        end
        # compute affine-scaling Newton step
        linsolve!(alg, fisher)
        alg.gₙ = gₙ = norm(alg.∇)
        alg.gₜ = gₜ = norm(alg.p .* alg.∇)
        alg.dₙ = dₙ = norm(alg.Δ)
        # update price
        αₘ = min(proj.(-(alg.pb) ./ alg.Δ)..., 1.0)
        alg.α = α = αₘ * 0.9995
        alg.p .= alg.pb .+ α * alg.Δ

    elseif alg.option_step == :primal_dual
        @debug "use primal-dual step"
        alg.ps .= alg.s
        alg.py .= alg.y

        linsolve!(alg, fisher)
        # standard ℓ₂ norm
        alg.gₙ = gₙ = norm(alg.∇)
        # local dual norm
        alg.gₜ = gₜ = norm(alg.p .* alg.∇)
        alg.dₙ = dₙ = norm(alg.Δ)

        # step size
        αₘ = min(proj.(-alg.pb ./ alg.Δ)..., proj.(-alg.ps ./ alg.Δs)..., 1.0)
        alg.α = α = αₘ * 0.9995
        alg.p .= alg.pb .+ α * alg.Δ
        alg.s .= alg.ps .+ α * alg.Δs
        alg.y .= alg.py .+ α * alg.Δy
    end

    # -------------------------------------------------------------------
    # II. methods that do not use the canonical logarithmic barrier 
    if alg.option_step == :damped_ns
        @debug """
            use damped Newton step
            this is very slow.
        """

        linsolve!(alg, fisher)
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
        linsolve!(alg, fisher)
        alg.gₙ = gₙ = norm(alg.∇)
        alg.gₜ = gₜ = norm(alg.p .* alg.∇)
        alg.dₙ = dₙ = norm(alg.Δ)

        # update price
        alg.α = α = αₘ = 1.0
        # alg.α = α = αₘ = 0.999
        alg.p .= alg.pb .+ α * alg.Δ
    end

    # @assert all(alg.p .> 0)
    alg.te = time()
    alg.t = alg.te - alg.ts
    _logline = produce_log(
        __default_logger,
        [alg.k log10(alg.μ) alg.φ alg.gₙ alg.dₙ alg.t alg.tₗ alg.α alg.kᵢ]
    )
    alg.k += 1
    ϵ = [gₙ dₙ]

    # update barrier parameter
    if alg.option_mu == :normal
        alg.μ *= (1 - min(alg.α * 0.9, 0.98))
        alg.μ = max(alg.μ, 1e-20)
    elseif alg.option_mu == :pred_corr
        alg.μ = max(alg.p' * alg.s / alg.n, 1e-25)
    end
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
    ground_truth::Union{Matrix{T},Nothing}=nothing,
    kwargs...
) where {T}
    logfile = logfile === nothing ? open(joinpath(LOGDIR, "log-hessianbar-$(current_date()).log"), "a") : logfile
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
    printto(ios, l)
    l = @sprintf(" option for gradient           := %s", alg.option_grad)
    printto(ios, l)
    l = @sprintf(" option for step               := %s", alg.option_step)
    printto(ios, l)
    l = @sprintf(" option for μ                  := %s", alg.option_mu)
    printto(ios, l)
    printto(ios, __default_logger._sep)
    flush.(ios)
    _k = 0
    while true
        _D = 0.0
        if !isnothing(ground_truth)
            # x - x*
            _D = sum(abs.(fisher.x - ground_truth))
        end
        keep_traj && push!(
            traj,
            StateInfo(alg.k, copy(alg.p), alg.∇, alg.gₙ, _D, alg.dₙ, alg.φ, alg.t)
        )
        _, _logline = iterate!(alg, fisher)
        mod(_k, 20 * loginterval) == 0 && printto(ios, __default_logger._loghead)
        mod(_k, loginterval) == 0 && printto(ios, _logline)
        if (alg.gₙ < alg.tol) || (alg.dₙ < alg.tol^2) || (alg.t >= alg.maxtime) || (_k >= alg.maxiter)
            break
        end
        _k += 1
        flush.(ios)
    end


    printto(ios, __default_logger._sep)
    printto(ios, " ✓  final play")
    play!(alg, fisher; ϵᵢ=0.1 * alg.μ, verbose=false, all=true, timed=false)
    l = @sprintf(" ✓  finished in        %4d steps", alg.k)
    printto(ios, l)
    l = @sprintf("             in %.5e seconds", alg.t)
    printto(ios, l)
    l = @sprintf("  best-resp. in %.5e seconds ", alg.tₗ)
    printto(ios, l)
    l = @sprintf("            avg %.5e seconds ", alg.tₗ / alg.k)
    printto(ios, l)
    l = @sprintf("          usage %.2f%%", (alg.tₗ / alg.t) * 100)
    printto(ios, l)
    printto(ios, __default_logger._sep)
    flush.(ios)
    close(logfile)
    return traj
end


