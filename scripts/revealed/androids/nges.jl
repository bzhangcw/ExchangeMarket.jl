# -----------------------------------------------------------------------
# Non-additive generalized elasticity of substitution (NGES).
#
# REAL-MARKET ONLY: this class is used solely to construct the ground-truth
# market; there is no NGES surrogate, separation oracle, or recovery. It
# generalizes GES by mixing the goods through a strictly positive matrix A
# before the power map (read-econ/choice-ump-utility.tex, eq.gnae.utility):
#
#   u(x) = ⟨c, (A x)^r⟩ = Σ_j c_j (A x)_j^{r_j},
#       c ∈ ℝ_{++}^n,  r ∈ (0,1)^n,  A ∈ ℝ_{++}^{n×n} (strictly positive).
#
# With A = I this is GES (androids/ges.jl). For r_j ∈ (0,1) the utility is
# strictly concave on ℝ_{++}^n (lem.gnae.concave: u = φ∘A with the separable
# φ(y)=⟨c,y^r⟩ concave and A a linear precomposition), so the UMP has a unique
# interior optimum. Unlike GES the stationarity equations couple through
# AᵀD A, so there is no per-good closed form; the demand is obtained by
# solving the concave program directly.
# -----------------------------------------------------------------------

using LinearAlgebra, Random
using JuMP, MadNLP

struct NGESAgent
    n::Int
    c::Vector{Float64}      # n, > 0
    r::Vector{Float64}      # n, ∈ (0, 1)
    A::Matrix{Float64}      # n×n, strictly positive
end

"""
    random_nges_agent(n; seed=nothing)

Random NGES agent in `n` goods:
  c_j ∈ Uniform(0.5, 2),  r_j ∈ Uniform(0.3, 0.7),  A_{jk} ∈ Uniform(0.5, 1.5).
The r-range stays interior to (0, 1) (so the utility is strictly concave),
and A is dense strictly positive (every nest (A x)_j mixes all goods).
"""
function random_nges_agent(n::Int; seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    c = 0.5 .+ 1.5 .* rand(n)
    r = 0.3 .+ 0.4 .* rand(n)
    A = 0.5 .+ rand(n, n)
    return NGESAgent(n, c, r, A)
end

"""
    nges_share_by_opt(c, r, A, p, w; verbose=false, timelimit=nothing, ε=1e-12) -> γ ∈ Δ_n

Solve the NGES UMP directly as the concave program

    max_{x ≥ ε}  Σ_j c_j (A x)_j^{r_j}   s.t.   ⟨p, x⟩ = w,

via JuMP + MadNLP, and return the spending share γ_j = p_j x_j / w. An
auxiliary nest variable `yv = A x ≥ ε` keeps the fractional powers in their
positive domain (mirroring the `x ≥ ε` guard in `ges_share_by_opt`).
"""
function nges_share_by_opt(c::AbstractVector, r::AbstractVector, A::AbstractMatrix,
    p::AbstractVector, w::Real;
    verbose::Bool=false,
    timelimit::Union{Real,Nothing}=nothing,
    ε::Real=1e-12)
    n = length(c)
    @assert size(A) == (n, n) "A must be n×n"
    @assert length(r) == n && length(p) == n
    @assert all(r .> 0) && all(r .< 1) "NGES requires r_j ∈ (0, 1)"
    @assert all(c .> 0) "NGES requires c_j > 0"
    @assert all(A .> 0) "NGES requires a strictly positive A"
    @assert w > 0

    model = new_model(nlp=true)
    verbose || set_attribute(model, "print_level", MadNLP.ERROR)
    if !isnothing(timelimit) && timelimit > 0
        set_attribute(model, "max_wall_time", Float64(timelimit))
    end

    # x: consumption (interior, budget-feasible start); yv: nest values A x.
    @variable(model, x[j=1:n] >= ε, start = w / (n * p[j]))
    @variable(model, yv[j=1:n] >= ε, start = sum(A[j, k] * w / (n * p[k]) for k in 1:n))
    @constraint(model, nest[j=1:n], yv[j] == sum(A[j, k] * x[k] for k in 1:n))
    @objective(model, Max, sum(c[j] * yv[j]^r[j] for j in 1:n))
    @constraint(model, sum(p[j] * x[j] for j in 1:n) == w)

    JuMP.optimize!(model)
    x_opt = value.(x)
    return (p .* x_opt) ./ w
end

"""
    share(agent::NGESAgent, p, w) -> γ

Per-class share dispatch (real-market demand). Delegates to
`nges_share_by_opt(agent.c, agent.r, agent.A, p, w)`.
"""
share(agent::NGESAgent, p::AbstractVector, w::Real) =
    nges_share_by_opt(agent.c, agent.r, agent.A, p, w)

"""
    produce_revealed_preferences_nges(agents, budgets, K, n; seed=nothing)

Generate K revealed-preference observations `(p_k, g_k)` from an NGES market,
`g_k = Σ_i x_i(p_k, w_i)` with `x_i = w_i γ_i / p` and `γ_i = share(agent_i, p_k, w_i)`.
Prices are uniform on the simplex. `budgets` is a Vector (m,) of fixed Fisher
budgets, or an n×m Matrix of Arrow–Debreu endowments (budget per sample is
`w_i(p_k) = ⟨p_k, b_i⟩`). Mirrors `produce_revealed_preferences_ges`.
"""
function produce_revealed_preferences_nges(
    agents::Vector{NGESAgent},
    budgets,
    K::Int,
    n::Int;
    seed=nothing
)
    !isnothing(seed) && Random.seed!(seed)
    m = length(agents)
    budgets isa AbstractMatrix &&
        @assert size(budgets) == (n, m) "endowment matrix must be n×m"
    Ξ = Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, K)
    for k in 1:K
        e_k = -log.(rand(n))
        p_k = e_k ./ sum(e_k)
        w = wealth_at(budgets, p_k)
        g_k = zeros(n)
        for i in 1:m
            γ_i = share(agents[i], p_k, w[i])
            g_k .+= w[i] .* γ_i ./ p_k
        end
        Ξ[k] = (copy(p_k), copy(g_k))
    end
    return Ξ
end
