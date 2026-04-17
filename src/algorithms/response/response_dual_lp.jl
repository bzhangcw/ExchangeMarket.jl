# -----------------------------------------------------------------------
# Dual LP best-response for linear markets
#   V_{i,μ}(p) = min -w_i log(λ_i) - μ Σ_j log(s_j)
#               s.t.  λ_i c_i + s_i = p,  λ_i, s ≥ 0
# Recovers x from complementarity: x_j = μ / s_j
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# -----------------------------------------------------------------------

using JuMP
import MathOptInterface as MOI

# --------------------------------------------------------------------------
# Conic version via JuMP + Mosek
# --------------------------------------------------------------------------
@doc raw"""
    Solve the dual LP per buyer via exponential cone programming:
        max  w_i log(λ) + μ Σ_j log(s_j)
        s.t. λ c_i + s = p,  λ ≥ 0, s ≥ 0
    Then x_j = μ / s_j by complementarity.
"""
function __conic_dual_lp_response(;
    i::Int=1,
    p::Vector{T}=nothing,
    market::Market=nothing,
    μ::Float64=1e-3,
    verbose::Bool=false,
    kwargs...
) where {T}
    n = market.n
    w = market.w[i]
    c = market.c[:, i]

    md = __generate_empty_jump_model(; verbose=verbose, tol=μ)

    μa = max(μ, 1e-2)

    @variable(md, λ >= 0)
    @variable(md, s[1:n] >= 0)
    @constraint(md, ls, λ .* c .+ s .== p)

    # log(λ) via exponential cone
    @variable(md, logλ)
    log_to_expcone!(λ, logλ, md)
    # log(s_j) via exponential cone
    @variable(md, logs[1:n])
    log_to_expcone!.(s, logs, md)

    @objective(md, Max, w * logλ + μa * sum(logs))

    JuMP.optimize!(md)

    market.s[:, i] .= max.(value.(s), 1e-8)
    market.x[:, i] .= abs.(dual.(ls))
    # market.x[:, i] .= μa ./ market.s[:, i]
    market.val_u[i] = c' * market.x[:, i]

    return nothing
end

DualLPConic = ResponseOptimizer(
    __conic_dual_lp_response,
    :dual_lp_conic,
    "DualLPConicResponse"
)

# --------------------------------------------------------------------------
# Analytic bisection version
# --------------------------------------------------------------------------
@doc raw"""
    Analytic dual LP best-response via bisection on λ.

    The optimality condition is:
        w / λ = Σ_j c_j μ / (p_j - λ c_j)^2 · (p_j - λ c_j) / μ
             ⟹  w / λ = Σ_j c_j / (p_j - λ c_j)
    i.e., bisect on λ: ψ(λ) = λ Σ_j c_j/(p_j - λ c_j) - w = 0.

    Then s_j = p_j - λ c_j, x_j = μ / s_j.
"""
function __dual_lp_response(;
    i::Int=1,
    p::Vector{T}=nothing,
    market::Market=nothing,
    agent::Union{AgentView,Nothing}=nothing,
    μ::Float64=1e-3,
    debug::Bool=false,
    kwargs...
) where {T}
    av = isnothing(agent) ? market.agents[i] : agent
    n = av.n
    w = market.w[av.i]
    c = av.c

    # λ must satisfy 0 < λ < min_j(p_j/c_j) for c_j > 0
    λ_max = Inf
    foreach_nz(c) do j, cj
        λ_max = min(λ_max, p[j] / cj)
    end

    # ψ(λ) = λ Σ_j c_j/(p_j - λ c_j) - w
    function ψ(λ)
        val = -w
        foreach_nz(c) do j, cj
            val += λ * cj / (p[j] - λ * cj)
        end
        return val
    end

    lo = 0.0
    hi = λ_max - 1e-15
    while ψ(hi) < 0
        hi = (hi + λ_max) / 2
        (λ_max - hi) < 1e-20 * λ_max && break
    end

    niter = 0
    for iter in 1:200
        λ_mid = (lo + hi) / 2
        v = ψ(λ_mid)
        niter = iter
        abs(v) < 1e-12 && break
        (hi - lo) < 1e-14 * hi && break
        v > 0 ? (hi = λ_mid) : (lo = λ_mid)
    end
    λ_opt = (lo + hi) / 2
    debug && @info "DualLP bisection" i niter λ_opt ψ(λ_opt)

    # recover s and x: for zero c_j, s_j = p_j, x_j = μ/p_j
    av.s .= p
    av.x .= μ ./ p
    foreach_nz(c) do j, cj
        av.s[j] = max(p[j] - λ_opt * cj, 1e-30)
        av.x[j] = μ / av.s[j]
    end
    market.val_u[av.i] = sparse_dot(c, av.x)

    return nothing
end

DualLP = ResponseOptimizer(
    __dual_lp_response,
    :dual_lp,
    "DualLPResponse"
)


# -----------------------------------------------------------------------
# compute the exact Hessian from dual LP slacks
# -----------------------------------------------------------------------
@doc raw"""
    __linear_hess_from_s!(alg, market::FisherMarket)

Compute ∇²φ_μ using dual LP slack variables s.
From the PD central path conditions:
    Σ_i⁻¹ = diag(x_i / s_i)
    r_i = w_i / u_i²
    -∇x_i = Σ_i⁻¹ - r_i Σ_i⁻¹ c c' Σ_i⁻¹ / (1 + r_i c' Σ_i⁻¹ c)
    H = Σ_i (-∇x_i)
"""
function __linear_hess_from_s!(alg, market::FisherMarket; dbg=true)
    if isa(alg.H, SparseMatrixCSC)
        alg.H = Matrix(alg.H)
    end
    alg.H .= 0.0
    n, m = size(market.x)
    X = market.x        # n×m
    S = market.s        # n×m
    w = market.w         # m
    c = market.c         # n×m

    for i in 1:m
        xᵢ = @view X[:, i]
        sᵢ = @view S[:, i]
        cᵢ = @view c[:, i]
        uᵢ = market.val_u[i]
        rᵢ = w[i] / uᵢ^2

        # Σ_i⁻¹ = diag(x_i / s_i)
        Σinv = xᵢ ./ sᵢ   # n-vector

        # Σ_i⁻¹ c
        Σinv_c = Σinv .* cᵢ  # n-vector

        # 1 + r_i c' Σ_i⁻¹ c
        denom = 1.0 + rᵢ * (cᵢ' * Σinv_c)

        # -∇x_i = diag(Σinv) - r_i Σinv_c Σinv_c' / denom
        coeff = rᵢ / denom
        alg.H .+= diagm(Σinv) .- coeff .* (Σinv_c * Σinv_c')
    end

    dbg && println("Hessian built from dual LP slacks (s-based)")
    return nothing
end