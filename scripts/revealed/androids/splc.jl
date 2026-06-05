# -----------------------------------------------------------------------
# SPLC (Separable Piecewise-Linear Concave) market utilities for
# revealed-preference experiments. Mirrors androids/plc.jl but for the
# separable subfamily (cf. § Separable piecewise linear concave utility
# in overleaf/read-econ/choice-ump-utility.tex):
#
#   u(x) = Σ_j f_j(x_j),   f_j(x_j) = min_ℓ { a_{jℓ} x_j + c_{jℓ} },
#
# each f_j nondecreasing, concave, piecewise linear in good j alone.
# This is the Vazirani–Yannakakis setting where Fisher equilibrium is
# PPAD-complete, yet individual demand is closed-form: a greedy
# bang-per-buck water-filling over segments (no LP needed).
#
#   - Random SPLC agent generation (random_splc_agent)
#   - Per-good utility evaluation (splc_fj, splc_utility)
#   - Demand via greedy segment purchase (solve_splc_demand)
#   - Revealed preference production (produce_revealed_preferences_splc)
# -----------------------------------------------------------------------

using LinearAlgebra, Random
using ArgParse

# ---- agent type ----------------------------------------------------------
"""
    SPLCAgent

Separable PLC agent. For each good j there are L_j affine pieces with
slopes `a[j]` (length L_j, sorted DECREASING) and intercepts `c[j]`
(length L_j, sorted INCREASING), so that

    f_j(x) = min_ℓ { a[j][ℓ] x + c[j][ℓ] }

is concave nondecreasing, and piece ℓ is active on the segment
[kink[j][ℓ-1], kink[j][ℓ]] where kink[j][ℓ] is the breakpoint between
pieces ℓ and ℓ+1 (kink[j][0] = 0, kink[j][L_j] = ∞).

Fields:
- `n`     : number of goods
- `a`     : Vector of slope vectors, a[j] ∈ ℝ^{L_j}_+, strictly decreasing
- `c`     : Vector of intercept vectors, c[j] ∈ ℝ^{L_j}_+, strictly increasing, c[j][1] = 0
- `kink`  : Vector of breakpoint vectors, kink[j] ∈ ℝ^{L_j-1}_+, increasing
            (kink[j][ℓ] = (c[j][ℓ+1] - c[j][ℓ]) / (a[j][ℓ] - a[j][ℓ+1]))
"""
struct SPLCAgent
    n::Int
    a::Vector{Vector{Float64}}
    c::Vector{Vector{Float64}}
    kink::Vector{Vector{Float64}}
end

# ---- CLI surface --------------------------------------------------------
"""
    register_cli_splc!(s::ArgParseSettings)

Add the "Market: SPLC" arg group — flags that only matter when the
ground-truth market is an SPLC family (`--market-type splc`).
"""
function register_cli_splc!(s::ArgParseSettings)
    add_arg_group!(s, "Market: SPLC")
    @add_arg_table! s begin
        "--splc-L"
        help = "Number of pieces per good (SPLC market only)."
        arg_type = Int
        default = 3
        "--splc-no-intercept"
        help = "Homogeneous SPLC: zero intercepts. Since min_ℓ a_ℓ x = (min_ℓ a_ℓ)x, the pieces collapse and each agent is LINEAR, u(x) = Σ_j a_j x_j (cf. rem.splc.homogeneous: SPLC's homothetic regime is exactly the linear class). Useful as a sanity check — linear androids should fit this market to ≈0 error."
        action = :store_true
    end
    return s
end

"""
    splc_opt_from_cli(cli) -> NamedTuple{(:L, :intercept)}

Bundle the parsed SPLC flags into the NamedTuple shape consumed by the
SPLC market-builder branch.
"""
splc_opt_from_cli(cli) = (L=cli["splc_L"], intercept=!cli["splc_no_intercept"])

# ---- agent generation ----------------------------------------------------
"""
    random_splc_agent(n, L; intercept=true, seed=nothing)

Generate a random SPLCAgent with `L` pieces per good. For each good j:
- slopes a[j]: L values sorted strictly decreasing, drawn from Uniform(0, 1)
  (slope of the first piece is the largest — diminishing marginal utility);
- breakpoints kink[j]: L-1 increasing values drawn from Uniform(0, 2);
- intercepts c[j]: built from continuity, c[j][1] = 0 and
  c[j][ℓ+1] = c[j][ℓ] + (a[j][ℓ] - a[j][ℓ+1]) * kink[j][ℓ],
  which makes f_j continuous and concave by construction.

By construction f_j(0) = 0, f_j is strictly increasing (all slopes > 0),
and concave (slopes decreasing).

`intercept=false` gives the homogeneous (homothetic) regime: zero
intercepts collapse min_ℓ a_ℓ x to (min_ℓ a_ℓ) x, so each f_j is a single
linear piece and the agent is linear, u(x) = Σ_j a_j x_j
(cf. rem.splc.homogeneous in choice-ump-utility.tex). Generated directly
as one piece per good with slope Uniform(0, 1).
"""
function random_splc_agent(n::Int, L::Int; intercept::Bool=true, seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    a = Vector{Vector{Float64}}(undef, n)
    c = Vector{Vector{Float64}}(undef, n)
    kink = Vector{Vector{Float64}}(undef, n)
    if !intercept
        # Homogeneous regime: one zero-intercept piece per good ⇒ linear agent.
        for j in 1:n
            a[j] = [rand() + 1e-3]
            c[j] = [0.0]
            kink[j] = Float64[]
        end
        return SPLCAgent(n, a, c, kink)
    end
    for j in 1:n
        # strictly decreasing positive slopes
        a_j = sort(rand(L) .+ 1e-3; rev=true)
        # increasing breakpoints
        k_j = sort(2.0 .* rand(L - 1))
        # continuity-built intercepts: c₁ = 0, c_{ℓ+1} = c_ℓ + (a_ℓ - a_{ℓ+1}) κ_ℓ
        c_j = zeros(L)
        for ℓ in 1:L-1
            c_j[ℓ+1] = c_j[ℓ] + (a_j[ℓ] - a_j[ℓ+1]) * k_j[ℓ]
        end
        a[j], c[j], kink[j] = a_j, c_j, k_j
    end
    return SPLCAgent(n, a, c, kink)
end

# ---- utility evaluation ---------------------------------------------------
"""
    splc_fj(agent, j, xj) -> Float64

Per-good utility f_j(x_j) = min_ℓ { a[j][ℓ] x_j + c[j][ℓ] }. Concave,
nondecreasing, f_j(0) = 0.
"""
function splc_fj(agent::SPLCAgent, j::Int, xj::Real)
    val = Inf
    for ℓ in eachindex(agent.a[j])
        val = min(val, agent.a[j][ℓ] * xj + agent.c[j][ℓ])
    end
    return val
end

"""
    splc_utility(agent, x) -> Float64

SPLC utility u(x) = Σ_j f_j(x_j).
"""
splc_utility(agent::SPLCAgent, x::AbstractVector) =
    sum(splc_fj(agent, j, x[j]) for j in 1:agent.n)

# ---- demand: greedy bang-per-buck ----------------------------------------
"""
    solve_splc_demand(agent::SPLCAgent, p, w) -> (x, u)

Closed-form SPLC demand via greedy segment purchase (the threshold/
water-filling characterization, cf. eq.splc.demand.threshold in the note):

1. Each piece (j, ℓ) covers the segment [kink[j][ℓ-1], kink[j][ℓ]] of good
   j with marginal utility a[j][ℓ]; buying the whole segment costs
   p_j * (kink[j][ℓ] - kink[j][ℓ-1]) dollars and its bang-per-buck is
   a[j][ℓ] / p_j.
2. Sort all pieces by bang-per-buck (descending) and buy greedily until
   the budget is exhausted; the marginal piece is bought fractionally.
   The last piece of each good has no upper breakpoint (extends to ∞), so
   the greedy always exhausts the budget.

Returns the demand vector x (length n) and the utility u(x). No LP solve;
cost is O(Σ_j L_j log Σ_j L_j) for the sort.
"""
function solve_splc_demand(agent::SPLCAgent, p::AbstractVector, w::Real)
    n = agent.n
    # collect all pieces: (bang-per-buck, j, segment length in goods units)
    pieces = Tuple{Float64,Int,Float64}[]
    for j in 1:n
        L = length(agent.a[j])
        for ℓ in 1:L
            lo = ℓ == 1 ? 0.0 : agent.kink[j][ℓ-1]
            hi = ℓ == L ? Inf : agent.kink[j][ℓ]
            push!(pieces, (agent.a[j][ℓ] / p[j], j, hi - lo))
        end
    end
    sort!(pieces; by=first, rev=true)

    x = zeros(n)
    budget = float(w)
    for (bpb, j, seglen) in pieces
        budget <= 0 && break
        bpb <= 0 && break                    # zero-slope pieces add no utility
        cost = p[j] * seglen                 # cost to buy the full segment
        if cost <= budget
            x[j] += seglen
            budget -= cost
        else
            x[j] += budget / p[j]            # fractional purchase
            budget = 0.0
        end
    end
    return x, splc_utility(agent, x)
end

# ---- revealed preferences -------------------------------------------------
"""
    produce_revealed_preferences_splc(agents, budgets, K, n; seed=nothing)

Generate K revealed-preference observations `(p_k, g_k)` from an SPLC
market, where `g_k = Σ_i x_i(p_k, w_i)` is the aggregate demand. Prices
are drawn uniformly on the simplex (Dirichlet(1,…,1)), matching the
CES/PLC/GES generators.

`budgets` is either a Vector (m,) of fixed Fisher budgets, or an n×m
Matrix of Arrow–Debreu endowments (column b_i per agent; budget per
sample is w_i(p_k) = ⟨p_k, b_i⟩).
"""
function produce_revealed_preferences_splc(
    agents::Vector{SPLCAgent},
    budgets::Union{Vector{Float64},Matrix{Float64}},
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
        g_k = zeros(n)
        for i in 1:m
            # Fisher: fixed budget. AD: endowment value at this sample's price.
            w_i = budgets isa AbstractMatrix ?
                  dot(p_k, view(budgets, :, i)) : budgets[i]
            x_i, _ = solve_splc_demand(agents[i], p_k, w_i)
            g_k .+= x_i
        end
        Ξ[k] = (copy(p_k), copy(g_k))
    end
    return Ξ
end
