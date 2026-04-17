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
    agent_type(ρ, σ) -> AgentType

Derive the agent type from CES parameter ρ.
"""
@inline function agent_type(ρ::Float64, σ::Float64)
    ρ == 1.0 ? LinearAgent() : CESAgent(ρ, σ)
end
