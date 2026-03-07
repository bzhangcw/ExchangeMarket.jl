# -----------------------------------------------------------------------
# run subproblems as best-response-type mappings
#   using induced utility function from Eigenberg-Gale-type potentials
#   the response mapping is captured by linear-conic programming
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
using JuMP
import MathOptInterface as MOI

# --------------------------------------------------------------------------
# primal form of CES economy in linear-conic form
# --------------------------------------------------------------------------
@doc raw"""
  solve the logarithmic utility maximization problem by JuMP + optimizer
  for CES utility function of ρ ≤ 1
    use max log(uᵢ(xᵢ))
"""
function __conic_log_response_ces(;
    i::Int=1,
    p::Vector{T}=nothing,
    market::Market=nothing,
    μ=1e-4,
    verbose=false,
    kwargs...
) where {T}
    ρ = market.ρ[i]
    ρ = market.ρ[i]
    ϵᵢ = μ * 1e-5
    md = __generate_empty_jump_model(; verbose=verbose, tol=ϵᵢ)

    @variable(md, u)
    @variable(md, logu)
    log_to_expcone!(u, logu, md)

    @variable(md, x[1:market.n] >= 0)
    @variable(md, ξ[1:market.n] >= 0)
    # budget constraint
    @constraint(md, budget, p' * x <= market.w[i])
    # utility constraint
    # Δ^{ρ} ξ^{1-ρ}≥ r 
    # ⇒ [Δ,ξ,r] ∈ P₃(ρ) [power cone]
    _c = market.c[:, i] .^ (1 / ρ)
    @constraint(md, sum(ξ) == u)
    @constraint(
        md,
        ξc[j=1:market.n],
        [_c[j] * x[j], u, ξ[j]] in MOI.PowerCone(ρ)
    )
    @objective(md, Max, logu)

    JuMP.optimize!(md)
    # ensure non-negativity
    market.x[:, i] .= max.(value.(x), 0.0)
    market.val_u[i] = market.u(market.x[:, i], i)
    market.val_f[i] = market.val_u[i]^(1 / market.ρ[i])
    market.val_∇u[:, i] = market.∇u(market.x[:, i], i)
    return nothing
end

CESConic = CESConicResponse = ResponseOptimizer(
    __conic_log_response_ces,
    :linconic,
    "CESConicResponse"
)

# --------------------------------------------------------------------------
# dual form of CES economy in linear-conic form
# --------------------------------------------------------------------------
@doc raw"""
solve the logarithmic utility maximization problem by JuMP + optimizer
  for linear utility function in the `dual form`
"""
function __conic_log_response_ces_dual(;
    i::Int=1,
    p::Vector{T}=nothing,
    market::Market=nothing,
    μ=1e-4,
    verbose=false,
    kwargs...
) where {T}
    ϵᵢ = μ * 1e-5
    md = __generate_empty_jump_model(; verbose=verbose, tol=ϵᵢ)
    @variable(md, s[1:market.n] .>= 0)
    @variable(md, logs[1:market.n])
    @variable(md, v .>= 0)
    @variable(md, logv)
    log_to_expcone!.(s, logs, md)
    log_to_expcone!(v, logv, md)
    @objective(md, Min, -market.w[i] * logv - μ * sum(logs))
    @constraint(md, xc, s + v .* market.c[:, i] - p .== 0)
    JuMP.optimize!(md)
    market.x[:, i] .= abs.(dual.(xc))
    return nothing
end
DualCESConic = DualCESConicResponse = ResponseOptimizer(
    __conic_log_response_ces_dual,
    :linconic,
    "DualCESConicResponse"
)

# --------------------------------------------------------------------------
# solve the CES utility maximization problem analytically
# --------------------------------------------------------------------------
@doc raw"""
    Analytic best-response for linear and CES markets.

    For linear markets, the optimal bundle concentrates on the good
    with the highest bang-per-buck ratio ``c_{ji}/p_j``.

    For CES markets (ρ < 1), uses the closed-form via the
    induced convex potential ``f`` and its gradient.
"""
function __analytic_response(;
    i::Int=1,
    p::Vector{T}=nothing,
    market::Market=nothing,
    kwargs...
) where {T}
    if is_linear_market(market)
        ratio = market.c[:, i] ./ p
        # argmax returns the index of the maximum value,
        # it is always the smallest one among the ties.
        j₊ = argmax(ratio)
        market.x[:, i] .= 0
        market.x[j₊, i] = market.w[i] / p[j₊]
        market.val_u[i] = market.u(market.x[:, i], i)
    else
        market.val_f[i], market.val_∇f[:, i], market.val_Hf[:, i] = market.f∇f(p, i)
        market.x[:, i] = -market.w[i] ./ market.val_f[i] ./ market.σ[i] .* market.val_∇f[:, i]
        market.val_u[i] = market.u(market.x[:, i], i)
    end
    return nothing
end

CESAnalytic = ResponseOptimizer(
    __analytic_response,
    :analytic,
    "CESAnalytic"
)


# compute Jacobian: -dx/dp
function __linear_jacxp_fromx(X₂, u, c, w, μ)
    invμ = 1 / μ
    Xc = X₂ * c
    r = w / u^2
    return invμ * X₂ - (invμ^2 * r * Xc * Xc') ./ (1 + invμ * r * c' * Xc)
end

# compute Jacobian -dp/dx
function __linear_jacpx_fromx(Xi₂, u, c, w, μ)
    r = w / u^2
    return μ * Xi₂ + r * c * c'
end

function __linear_hess_fromx!(alg, market::FisherMarket; bool_dbg=false)
    X2 = market.x[alg.sampler.indices, :] .^ 2
    Di(i) = begin
        X₂ = spdiagm(X2[:, i])
        u = market.val_u[i]
        c = market.val_∇u[:, i]
        w = market.w[i]
        jxp = __linear_jacxp_fromx(X₂, u, c, w, alg.μ)
        if bool_dbg
            Xi₂ = spdiagm(1 ./ X2[:, i])
            jpx = __linear_jacpx_fromx(Xi₂, u, c, w, alg.μ)
            @info "jacpx * jacxp - I" maximum(abs.(jpx * jxp - I))
        end
        return jxp
    end
    alg.H = mapreduce(Di, +, alg.sampler.indices, init=spzeros(market.n, market.n))
end