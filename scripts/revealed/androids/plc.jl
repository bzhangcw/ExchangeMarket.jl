# -----------------------------------------------------------------------
# PLC market utilities for revealed-preference experiments.
#   - Random PLC agent generation
#   - Demand computation via LP (Mosek)
#   - Revealed preference production (aggregate demand at random prices)
# -----------------------------------------------------------------------

using LinearAlgebra, Random
using JuMP, MosekTools
using ArgParse
import MathOptInterface as MOI
using ExchangeMarket

# ---- CLI surface --------------------------------------------------------
"""
    register_cli_plc!(s::ArgParseSettings)

Add the "Market: PLC" arg group — flags that only matter when the
ground-truth market is a PLC family (`--market-type plc`).
"""
function register_cli_plc!(s::ArgParseSettings)
    add_arg_group!(s, "Market: PLC")
    @add_arg_table! s begin
        "--plc-L"
        help = "Number of PLC pieces (PLC market only)."
        arg_type = Int
        default = 5
        "--plc-no-intercept"
        help = "Disable PLC intercept (b=0); only meaningful for --market-type plc."
        action = :store_true
    end
    return s
end

"""
    plc_opt_from_cli(cli) -> NamedTuple{(:L, :intercept)}

Bundle the parsed PLC flags into the NamedTuple shape consumed by the
PLC market-builder branch in run_test.jl. Returns `(L=Int, intercept=Bool)`.
"""
plc_opt_from_cli(cli) = (L = cli["plc_L"], intercept = !cli["plc_no_intercept"])

"""
    random_plc_agent(n, L; intercept=true, density=0.1, seed=nothing)

Generate a random PLCAgent with `L` hyperplanes in `n` goods.
  u(x) = min_{ℓ∈[L]} { aℓ'x + bℓ },  aℓ ≥ 0, bℓ ≥ 0.

Each entry of aℓ is kept with probability `density` (default 0.1) and
drawn from Uniform(0,1) when kept; the rest are zero. If `intercept=true`,
each bℓ is drawn from Uniform(0,1); otherwise bℓ = 0.
"""
function random_plc_agent(n::Int, L::Int; intercept=true, sparsity::Real=0.2, seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    a = rand(L, n) .* (rand(L, n) .< sparsity)
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
    produce_revealed_preferences_plc(agents, budgets, K, n;
        price_range=(0.5, 2.0), seed=nothing)

Generate K revealed-preference observations from a PLC market.
Each observation is a random price vector and the aggregate demand g(p) = Σᵢ xᵢ(p).

Arguments:
- agents: Vector{PLCAgent}, one per agent
- budgets: either a Vector (m,) of fixed Fisher budgets w_i, or an n×m
  Matrix of Arrow–Debreu endowments — column b_i per agent, budget
  evaluated per sample as w_i(p_k) = ⟨p_k, b_i⟩.
- K: number of observations
- n: number of goods

Returns Ξ = [(p₁, g₁), ..., (p_K, g_K)].
"""
function produce_revealed_preferences_plc(
    agents::Vector{PLCAgent},
    budgets,
    K::Int,
    n::Int;
    price_range=(0.5, 2.0),
    seed=nothing
)
    !isnothing(seed) && Random.seed!(seed)
    m = length(agents)
    budgets isa AbstractMatrix &&
        @assert size(budgets) == (n, m) "endowment matrix must be n×m"
    Ξ = Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, K)

    for k in 1:K
        # Uniform on the unit simplex via Dirichlet(1,…,1): draw n iid
        # Exp(1) and normalize. Drawing uniform on `[lo, hi]^n` then
        # normalizing biases away from corners — same convention as
        # produce_revealed_preferences in setup.jl.
        e_k = -log.(rand(n))
        p_k = e_k ./ sum(e_k)

        # Per-agent wealth at this sample's price (fixed budget / endowment
        # value ⟨p_k,b_i⟩ / price-dependent wealth function), via `wealth_at`.
        w = wealth_at(budgets, p_k)
        g_k = zeros(n)
        for i in 1:m
            x_i, _ = solve_plc_demand(agents[i], p_k, w[i])
            g_k .+= x_i
        end
        Ξ[k] = (copy(p_k), copy(g_k))
    end
    return Ξ
end

# -----------------------------------------------------------------------
# PLC inspection helpers — active pieces, optimality gap, certificate.
# (Moved here from leontief.jl: each utility owns its own functions.)
# -----------------------------------------------------------------------
"""
    active_pieces(agent::PLCAgent, x; tol=1e-6)

Indices ℓ ∈ [L] where aℓ'x + bℓ is within `tol` of min_ℓ' (aℓ''x + bℓ').
"""
function active_pieces(agent::PLCAgent, x::AbstractVector; tol::Real=1e-6)
    vals = [dot(view(agent.a, ℓ, :), x) + agent.b[ℓ] for ℓ in 1:agent.L]
    umin = minimum(vals)
    return findall(v -> v - umin <= tol, vals)
end

"""
    plc_optimality_gap(plc::PLCAgent, x, p, w; u_star=nothing)

Quantify how far `x` is from being an optimal demand of `plc` at price `p` and budget `w`.
PLC demand is set-valued on flat utility regions, so `x` may differ from the LP solver's
output yet still be in the optimal demand set. The check is a *feasibility system*:

  1. Non-negativity        : x ≥ 0           → feas_neg   = max(0, -min(x))
  2. Budget                : p'x ≤ w         → feas_budget = max(0, p'x - w)
  3. Optimal utility value : u(x) = u_LP(p, w) → util_gap   = |u(x) - u_LP(p, w)|

`x` is in the optimal demand set iff all three are zero within tolerance.
Pass a precomputed `u_star = solve_plc_demand(plc, p, w)[2]` to avoid re-solving.
Returns a NamedTuple `(feas_neg, feas_budget, util_gap, u_cand, u_star)`.
"""
function plc_optimality_gap(plc::PLCAgent, x::AbstractVector, p::AbstractVector, w::Real;
    u_star::Union{Nothing,Real}=nothing)
    feas_neg = max(0.0, -minimum(x))
    feas_budget = max(0.0, dot(p, x) - w)
    u_cand = minimum(dot(view(plc.a, ℓ, :), x) + plc.b[ℓ] for ℓ in 1:plc.L)
    if u_star === nothing
        _, u_star = solve_plc_demand(plc, p, w)
    end
    util_gap = abs(u_cand - u_star)
    return (feas_neg=feas_neg, feas_budget=feas_budget, util_gap=util_gap,
        u_cand=u_cand, u_star=u_star)
end

"""
    is_plc_optimal_demand(plc::PLCAgent, x, p, w; tol=1e-6, kwargs...)

Bool wrapper around [`plc_optimality_gap`](@ref): `true` iff feasibility violations and
utility gap are all within `tol`.
"""
function is_plc_optimal_demand(plc::PLCAgent, x::AbstractVector, p::AbstractVector, w::Real;
    tol::Real=1e-6, kwargs...)
    g = plc_optimality_gap(plc, x, p, w; kwargs...)
    return max(g.feas_neg, g.feas_budget, g.util_gap) <= tol
end


# -----------------------------------------------------------------------
# PLC market excess at a candidate price (single all-in-one LP from
# `eq.plc.we.check.lp` in `read-econ/choice-ump-plc.tex`). Used by the PLC
# branch of `validate_surrogate` (validate.jl) to score a surrogate's
# equilibrium price against a PLC ground-truth market.
#
# Given a price `p` and PLC agents `(agents[i], w[i])` with unit supply
# q = 1, this solves the joint LP
#
#     min τ
#     s.t. UMP primal-dual + strong duality for every agent i,
#          -τ 1 ≤ diag(p) (1 − Σ_i x_i) ≤ τ 1, τ ≥ 0.
#
# Returns (tau=τ*, x=selection, status=termination_status, time=wall).
# τ* == 0 iff p is a Walrasian equilibrium of the PLC market.
# -----------------------------------------------------------------------
function solve_plc_excess(p::AbstractVector, agents::Vector{<:PLCAgent},
    w::AbstractVector; verbose::Bool=false, timelimit::Union{Real,Nothing}=nothing)
    m = length(agents)
    n = length(p)
    @assert length(w) == m "budget vector length mismatch"
    @assert all(a -> size(a.a, 2) == n, agents) "agent gradient widths must equal n"
    # NaN/Inf in the surrogate equilibrium price would make the LP coefficients
    # invalid; fail soft so the run produces a row instead of crashing.
    if any(!isfinite, p) || any(p .<= 0)
        @warn "solve_plc_excess: non-finite or non-positive price; skipping LP" p_minmax = extrema(p)
        return (tau=NaN, x=[fill(NaN, n) for _ in 1:m], status=:bad_price, time=0.0)
    end

    model = Model(Mosek.Optimizer)
    verbose || set_silent(model)
    if !isnothing(timelimit) && timelimit > 0
        set_time_limit_sec(model, Float64(timelimit))
    end

    # Per-agent decision variables.
    x = [@variable(model, [1:n], lower_bound = 0.0, base_name = "x_$(i)") for i in 1:m]
    t = [@variable(model, base_name = "t_$(i)") for i in 1:m]
    U = [@variable(model, [1:agents[i].L], lower_bound = 0.0, base_name = "U_$(i)") for i in 1:m]
    ν = [@variable(model, [1:n], lower_bound = 0.0, base_name = "nu_$(i)") for i in 1:m]
    μ = [@variable(model, lower_bound = 0.0, base_name = "mu_$(i)") for i in 1:m]
    @variable(model, τ >= 0.0)

    for i in 1:m
        ai = agents[i]
        A_i = ai.a               # L × n
        b_i = ai.b               # L
        L_i = ai.L
        # UMP primal: A_i x_i + b_i .≥ t_i 1, ⟨p, x_i⟩ = w_i.
        @constraint(model, A_i * x[i] .+ b_i .>= t[i] .* ones(L_i))
        @constraint(model, dot(p, x[i]) == w[i])
        # UMP dual: A_i' U_i + ν_i = μ_i p, ⟨1, U_i⟩ = 1.
        @constraint(model, A_i' * U[i] .+ ν[i] .== μ[i] .* p)
        @constraint(model, sum(U[i]) == 1.0)
        # Strong duality: t_i = ⟨b_i, U_i⟩ + w_i μ_i.
        @constraint(model, t[i] == dot(b_i, U[i]) + w[i] * μ[i])
    end

    # Market clearing: -τ 1 ≤ diag(p) (1 - Σ x_i) ≤ τ 1.
    aggregate = sum(x[i] for i in 1:m)
    for j in 1:n
        @constraint(model, p[j] * (1.0 - aggregate[j]) <= τ)
        @constraint(model, p[j] * (1.0 - aggregate[j]) >= -τ)
    end

    @objective(model, Min, τ)
    t_start = time()
    optimize!(model)
    elapsed = time() - t_start

    status = termination_status(model)
    if status ∉ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.SLOW_PROGRESS, MOI.TIME_LIMIT)
        @warn "PLC excess LP terminated abnormally" status
        return (tau=NaN, x=[fill(NaN, n) for _ in 1:m], status=status, time=elapsed)
    end
    return (
        tau=value(τ),
        x=[value.(x[i]) for i in 1:m],
        status=status,
        time=elapsed,
    )
end
