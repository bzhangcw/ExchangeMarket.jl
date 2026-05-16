# -----------------------------------------------------------------------
# Agent type hierarchy — struct definitions only.
# Utility evaluation methods live in the corresponding response files.
# -----------------------------------------------------------------------
abstract type AgentType end

"""
    LinearAgent <: AgentType

Linear utility: u(x) = ⟨c, x⟩.
"""
struct LinearAgent <: AgentType end

"""
    CESAgent <: AgentType

CES utility: u(x) = (Σⱼ cⱼ xⱼ^ρ)^(1/ρ), with ρ < 1, σ = ρ/(1-ρ).
"""
struct CESAgent <: AgentType
    ρ::Float64
    σ::Float64
end

"""
    PLCAgent <: AgentType

Piecewise-Linear Concave utility:
  u(x) = min_{ℓ∈[L]} { aℓ'x + bℓ }
where aℓ ≥ 0 (marginal utilities) and bℓ ≥ 0 (intercepts).

Fields:
- `L`: number of hyperplanes
- `a`: gradient matrix (L × n), each row aℓ ≥ 0
- `b`: intercept vector (L,), each bℓ ≥ 0
"""
struct PLCAgent <: AgentType
    L::Int
    a::Matrix{Float64}   # L × n
    b::Vector{Float64}   # L
end

"""
    LeontiefAgent <: AgentType

Leontief (perfect-complements) utility: u(x) = min_{j∈[n]} aⱼ xⱼ.
Equivalent to PLCAgent with L = n, a = diag(weights), b = 0.

Field:
- `a`: weight vector (n,), strictly positive.
"""
struct LeontiefAgent <: AgentType
    a::Vector{Float64}
end

"""
    QuasiLinearLogAgent <: AgentType

Quasi-linear-log utility:
  u(x) = Σ_{j<n} c_j log(x_j) + x_n,
with c ∈ R^{n-1}_{++} the log weights on the first n-1 goods, and good n
entering linearly.

Fields:
- `n`: total number of goods
- `c`: log-weight vector of length n-1, strictly positive.
"""
struct QuasiLinearLogAgent <: AgentType
    n::Int
    c::Vector{Float64}   # length n-1
end

"""
    agent_type(ρ, σ) -> AgentType

Derive the agent type from CES parameter ρ.
"""
@inline function agent_type(ρ::Float64, σ::Float64)
    ρ == 1.0 ? LinearAgent() : CESAgent(ρ, σ)
end
