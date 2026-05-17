# -----------------------------------------------------------------------
# PLC market utilities for revealed-preference experiments.
#   - Random PLC agent generation
#   - Demand computation via LP (Mosek)
#   - Revealed preference production (aggregate demand at random prices)
# -----------------------------------------------------------------------

using LinearAlgebra, Random
using JuMP, MosekTools
import MathOptInterface as MOI
using ExchangeMarket

"""
    random_plc_agent(n, L; intercept=true, density=0.1, seed=nothing)

Generate a random PLCAgent with `L` hyperplanes in `n` goods.
  u(x) = min_{ℓ∈[L]} { aℓ'x + bℓ },  aℓ ≥ 0, bℓ ≥ 0.

Each entry of aℓ is kept with probability `density` (default 0.1) and
drawn from Uniform(0,1) when kept; the rest are zero. If `intercept=true`,
each bℓ is drawn from Uniform(0,1); otherwise bℓ = 0.
"""
function random_plc_agent(n::Int, L::Int; intercept=true, density::Real=0.2, seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    a = rand(L, n) .* (rand(L, n) .< density)
    b = intercept ? rand(L) : zeros(L)
    return PLCAgent(L, a, b)
end

"""
    solve_plc_demand(agent::PLCAgent, p, w; verbose=false)

Solve the PLC utility maximization problem as an LP:
  max t  s.t.  t ≤ aℓ'x + bℓ ∀ℓ,  p'x ≤ w,  x ≥ 0.

Returns (x, u).
"""
function solve_plc_demand(agent::PLCAgent, p::AbstractVector, w::Real; verbose=false)
    n = size(agent.a, 2)
    md = ExchangeMarket.__generate_empty_jump_model(; verbose=verbose, tol=1e-8)

    @variable(md, x[1:n] >= 0)
    @variable(md, t)

    for ℓ in 1:agent.L
        aℓ = view(agent.a, ℓ, :)
        @constraint(md, t <= dot(aℓ, x) + agent.b[ℓ])
    end
    @constraint(md, dot(p, x) <= w)
    @objective(md, Max, t)

    JuMP.optimize!(md)
    x_val = max.(value.(x), 0.0)
    u_val = utility(agent, nothing, x_val)
    return x_val, u_val
end

"""
    produce_revealed_preferences_plc(agents, w_vec, K, n;
        price_range=(0.5, 2.0), seed=nothing)

Generate K revealed-preference observations from a PLC market.
Each observation is a random price vector and the aggregate demand g(p) = Σᵢ xᵢ(p).

Arguments:
- agents: Vector{PLCAgent}, one per agent
- w_vec: budgets (m,)
- K: number of observations
- n: number of goods

Returns Ξ = [(p₁, g₁), ..., (p_K, g_K)].
"""
function produce_revealed_preferences_plc(
    agents::Vector{PLCAgent},
    w_vec::Vector{Float64},
    K::Int,
    n::Int;
    price_range=(0.5, 2.0),
    seed=nothing
)
    !isnothing(seed) && Random.seed!(seed)
    m = length(agents)
    Ξ = Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, K)

    for k in 1:K
        p_k = price_range[1] .+ (price_range[2] - price_range[1]) .* rand(n)
        p_k = p_k ./ sum(p_k)

        g_k = zeros(n)
        for i in 1:m
            x_i, _ = solve_plc_demand(agents[i], p_k, w_vec[i])
            g_k .+= x_i
        end
        Ξ[k] = (copy(p_k), copy(g_k))
    end
    return Ξ
end
