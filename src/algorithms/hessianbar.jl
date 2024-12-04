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
    # gradient
    ∇::Vector{T}
    gₙ::T # norm of gradient
    dₙ::T # size of Newton step
    α::T  # step size
    # Hessian
    H::SparseMatrixCSC{T,Int}
    # steps
    step::Symbol = :affine_scaling

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
    option_grad::Symbol



    function HessianBar(
        n::Int, m::Int,
        p::Vector{T},
        μ::T;
        maxiter::Int=1000,
        maxtime::Float64=100.0,
        tol::Float64=1e-6,
        optimizer::ResponseOptimizer=OptimjlNewtonResponse,
        option_grad::Symbol=:dual
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
        return this
    end
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
function play!(
    alg::HessianBar, fisher::FisherMarket;
    ϵᵢ=1e-7, verbose=false,
    index_set=nothing
)
    _k = 0
    index_set === nothing && (index_set = 1:fisher.m)
    for i in index_set
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
        info = solve!(
            alg.optimizer;
            fisher=fisher, i=i, p=alg.p, μ=alg.μ, verbose=false
        )
        fisher.x[i, :] .= info.x
        fisher.val_u[i] = fisher.u(info.x, i)
        fisher.val_∇u[i, :] = fisher.∇u(info.x, i)
        return info
    elseif alg.optimizer.style == :analytic
        fisher.val_ν[i], fisher.val_∇ν[i, :], fisher.val_Hν[i, :] = fisher.ν∇ν(alg.p, i)
        fisher.x[i, :] = -fisher.w[i] ./ fisher.val_ν[i] ./ fisher.σ .* fisher.val_∇ν[i, :]
        fisher.val_u[i] = fisher.u(fisher.x[i, :], i)
        return ResponseInfo(
            fisher.x[i, :],
            fisher.val_u[i],
            [0.0],
            0.0,
            1,
            nothing
        )
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
    if alg.step == :affine_scaling
        # compute affine-scaling Newton step
        invp = 1 ./ alg.p
        dp = -(alg.H + alg.μ * spdiagm(invp .^ 2)) \ (alg.∇ - alg.μ * invp)
        alg.gₙ = gₙ = norm(alg.p .* alg.∇)
        alg.dₙ = dₙ = norm(dp)
        # update price
        αₘ = min(proj.(-alg.pb ./ dp)..., 1.0)
        alg.α = α = αₘ * 0.995
        alg.p .= alg.pb .+ α * dp
    elseif alg.step == :augmented
        alg.pλ .= alg.λ
        # construct augmented system
        invp = 1 ./ alg.p
        # Σ = P^{-1}Λ
        σ = alg.λ .* invp
        dp = -(alg.H + spdiagm(σ)) \ (alg.∇ - alg.μ * invp)
        dλ = -alg.λ + alg.μ * invp - σ .* dp
        alg.gₙ = gₙ = norm(alg.p .* alg.∇)
        alg.dₙ = dₙ = norm(dp)
        αₘ = min(proj.(-alg.pb ./ dp)..., proj.(-alg.pλ ./ dλ)..., 1.0)
        alg.α = α = αₘ * 0.995
        alg.p .= alg.pb .+ α * dp
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
    println(_logline)
    alg.k += 1
    ϵ = [gₙ dₙ]

    # update barrier parameter
    if gₙ < alg.μ * 2e1
        alg.μ *= 0.1
    end
    # alg.μ *= 0.85
    # alg.μ *= (1 - min(alg.α * 0.8, 0.98))

    return ϵ
end

function solve!(
    alg::HessianBar, fisher::FisherMarket;
    p₀::Union{Vector{T},Nothing}=nothing,
    maxiter=1000, maxtime=100.0, tol=1e-6,
    step::Symbol=:affine_scaling,
    keep_traj::Bool=false,
    kwargs...
) where {T}
    println(__default_logger._blockheader)
    traj = []
    # override step choice here
    alg.step = step
    bool_default = isnothing(alg.optimizer)
    bool_default && (alg.optimizer = OptimjlNewtonResponse)
    !isnothing(p₀) && begin
        (alg.p .= p₀)
        println("!!!warm-starting price")
    end

    bool_default && @warn "!!!no optimizer specified! will use default one: $(alg.optimizer.name)\n"
    @printf " main iterate method           := %s\n" step
    @printf " subproblem solver alias       := %s\n" alg.optimizer.name
    @printf " subproblem solver style       := %s\n" alg.optimizer.style
    @printf " option for gradient           := %s\n" alg.option_grad
    # @printf " algorithm description := %s\n" t.DESC
    println(__default_logger._sep)
    _k = 0
    for _ in 1:maxiter
        mod(_k, 20) == 0 && println(__default_logger._loghead)
        ϵ = iterate!(alg, fisher)
        keep_traj && push!(traj, copy(alg.p))
        if (alg.gₙ < alg.tol) || (alg.dₙ < alg.tol) || (alg.t >= alg.maxtime) || (_k >= maxiter)
            break
        end
        _k += 1
    end


    println(__default_logger._sep)
    println(" ✓ final play")
    play!(alg, fisher; ϵᵢ=0.1 * alg.μ, verbose=false)
    println(__default_logger._sep)
    return traj
end

