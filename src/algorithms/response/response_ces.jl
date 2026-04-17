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
# Utility evaluation for Linear and CES agents
# --------------------------------------------------------------------------
@inline utility(::LinearAgent, c, x) = sparse_dot(c, x)

@inline function utility(at::CESAgent, c, x)
    s = 0.0
    foreach_nz(c) do j, cj
        s += cj * spow(x[j], at.ρ)
    end
    return spow(s, 1.0 / at.ρ)
end

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
    agent::Union{AgentView,Nothing}=nothing,
    μ=1e-4,
    verbose=false,
    kwargs...
) where {T}
    av = isnothing(agent) ? market.agents[i] : agent
    n = av.n
    w = market.w[av.i]
    ρ = av.atype isa CESAgent ? av.atype.ρ : 1.0
    c = Vector(av.c)  # JuMP needs dense Vector
    ϵᵢ = μ * 1e-5
    md = __generate_empty_jump_model(; verbose=verbose, tol=ϵᵢ)

    @variable(md, u)
    @variable(md, logu)
    log_to_expcone!(u, logu, md)

    @variable(md, x[1:n] >= 0)
    @variable(md, ξ[1:n] >= 0)
    @constraint(md, budget, p' * x <= w)
    _c = c .^ (1 / ρ)
    @constraint(md, sum(ξ) == u)
    @constraint(
        md,
        ξc[j=1:n],
        [_c[j] * x[j], u, ξ[j]] in MOI.PowerCone(ρ)
    )
    @objective(md, Max, logu)

    JuMP.optimize!(md)
    av.x .= max.(value.(x), 0.0)
    market.val_u[av.i] = utility(av)
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
    agent::Union{AgentView,Nothing}=nothing,
    μ=1e-4,
    verbose=false,
    kwargs...
) where {T}
    av = isnothing(agent) ? market.agents[i] : agent
    n = av.n
    w = market.w[av.i]
    c = Vector(av.c)  # JuMP needs dense Vector
    ϵᵢ = μ * 1e-5
    md = __generate_empty_jump_model(; verbose=verbose, tol=ϵᵢ)
    @variable(md, s[1:n] .>= 0)
    @variable(md, logs[1:n])
    @variable(md, v .>= 0)
    @variable(md, logv)
    log_to_expcone!.(s, logs, md)
    log_to_expcone!(v, logv, md)
    @objective(md, Min, -w * logv - μ * sum(logs))
    @constraint(md, xc, s + v .* c - p .== 0)
    JuMP.optimize!(md)
    av.x .= abs.(dual.(xc))
    return nothing
end
DualCESConic = DualCESConicResponse = ResponseOptimizer(
    __conic_log_response_ces_dual,
    :linconic,
    "DualCESConicResponse"
)

# --------------------------------------------------------------------------
# CES closed-form demand: x_j = w c_j^(1+σ) p_j^(-σ-1) / Σ_k c_k^(1+σ) p_k^(-σ)
# --------------------------------------------------------------------------
@inline function _ces_demand!(x, c::SparseColRef, p, w, σ)
    denom = sparse_reduce(c) do j, cj
        cj^(1.0 + σ) * p[j]^(-σ)
    end
    coeff = w / denom
    x .= 0.0
    sparse_scatter!(x, c) do j, cj
        coeff * cj^(1.0 + σ) * p[j]^(-σ - 1.0)
    end
end

@inline function _ces_demand!(x, c::AbstractVector, p, w, σ)
    cs = c .^ (1.0 + σ)
    denom = dot(cs, p .^ (-σ))
    x .= (w / denom) .* cs .* p .^ (-σ - 1.0)
end

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
    agent::Union{AgentView,Nothing}=nothing,
    kwargs...
) where {T}
    av = isnothing(agent) ? market.agents[i] : agent
    w = market.w[av.i]
    if av.atype isa LinearAgent
        # linear: bang-per-buck — concentrate on best c_j/p_j
        j₊ = sparse_argmax(av.c) do j, cj
            cj / p[j]
        end
        av.x .= 0
        av.x[j₊] = w / p[j₊]
    else
        # CES closed-form demand:
        #   x_j = w · c_j^(1+σ) · p_j^(-σ-1) / Σ_k c_k^(1+σ) · p_k^(-σ)
        _ces_demand!(av.x, av.c, p, w, av.atype.σ)
    end
    market.val_u[av.i] = utility(av)
    return nothing
end

CESAnalytic = ResponseOptimizer(
    __analytic_response,
    :analytic,
    "CESAnalytic"
)
