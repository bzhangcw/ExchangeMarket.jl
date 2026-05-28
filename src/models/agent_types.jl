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
    GESAgent <: AgentType

Generalized-elasticity-of-substitution (GES) utility:
  u(x) = Σ_j c_j x_j^{r_j},  c ∈ ℝ^n_{++},  r ∈ (0, 1)^n,
with per-good elasticity σ_j := r_j / (1 - r_j) > 0. Non-homothetic
whenever r is not constant; r_j ≡ r recovers CES.

Fields:
- `n`: number of goods
- `c`: coefficient vector of length n, strictly positive
- `r`: elasticity exponents of length n, each in (0, 1)
"""
struct GESAgent <: AgentType
    n::Int
    c::Vector{Float64}   # length n, strictly positive
    r::Vector{Float64}   # length n, each in (0, 1)
end

"""
    agent_type(ρ, σ) -> AgentType

Derive the agent type from CES parameter ρ.
"""
@inline function agent_type(ρ::Float64, σ::Float64)
    @assert σ == (ρ / (1 - ρ)) "σ must be ρ/(1-ρ)"
    ρ == 1.0 ? LinearAgent() : CESAgent(ρ, σ)
end
