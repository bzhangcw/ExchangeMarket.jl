# Interactive test: Arrow–Debreu PLC ground-truth market.
#
# Builds a small PLC market where each agent owns an endowment column b_i
# (n×m, unit supply per good) and faces the price-dependent budget
# w_i(p) = ⟨p, b_i⟩. Exposes `agg_demand(p)` to query the aggregate demand
# at any price, plus the raw pieces (agents, B) for inspection.
#
# Usage:
#   julia -i --project=scripts scripts/revealed/test_plc_ad.jl
#   julia> p = ones(N) ./ N
#   julia> agg_demand(p)            # aggregate demand g(p) at price p
#   julia> agg_share(p)             # expenditure share diag(p) g(p) (sums to 1)
#   julia> budgets(p)               # per-agent budgets ⟨p, b_i⟩ (sum to 1)
#   julia> walras_check(p)          # ⟨p, g(p)⟩ - Σ_i w_i(p)  (should be ~0)

using Random, LinearAlgebra
using ExchangeMarket

include("../../tools.jl")
include("../androids/plc.jl")

# ---- knobs --------------------------------------------------------------
const N = 4       # goods
const M = 5       # agents
const L = 3       # PLC pieces per agent
const SEED = 42

# ---- build the AD PLC market -------------------------------------------
Random.seed!(SEED)
const agents = [random_plc_agent(N, L; sparsity=0.99, intercept=false) for _ in 1:M]
# Endowments: n×m, column b_i per agent; each good's total endowment is 1.
const B = let b = rand(N, M)
    b ./ sum(b; dims=2)
end

# ---- accessors -----------------------------------------------------------
"""
    budgets(p) -> Vector

Per-agent Arrow–Debreu budgets w_i(p) = ⟨p, b_i⟩. Sum to ⟨p, 1⟩ (= 1 on the simplex).
"""
budgets(p::AbstractVector) = [dot(p, @view B[:, i]) for i in 1:M]

"""
    agg_demand(p) -> Vector

Aggregate Arrow–Debreu demand g(p) = Σ_i x_i(p, ⟨p, b_i⟩), one LP per agent
(solve_plc_demand).
"""
function agg_demand(p::AbstractVector)
    g = zeros(N)
    for i in 1:M
        x_i, _ = solve_plc_demand(agents[i], p, dot(p, @view B[:, i]))
        g .+= x_i
    end
    return g
end

"""
    agg_share(p) -> Vector

Market expenditure share diag(p) g(p). Sums to total wealth ⟨p, 1⟩ = 1
when every agent exhausts its budget.
"""
agg_share(p::AbstractVector) = p .* agg_demand(p)

"""
    walras_check(p) -> Float64

Walras' law residual ⟨p, g(p)⟩ − Σ_i w_i(p); ≈ 0 iff all budgets are exhausted.
"""
walras_check(p::AbstractVector) = dot(p, agg_demand(p)) - sum(budgets(p))

"""
    sample_data(K; seed=SEED) -> Vector{Tuple}

K revealed-preference samples (p_k, g_k) with simplex-uniform prices —
the same recipe build_rep_data uses (via produce_revealed_preferences_plc).
"""
sample_data(K::Int; seed=SEED) = produce_revealed_preferences_plc(agents, B, K, N; seed=seed)

# ---- demo ----------------------------------------------------------------
let p = ones(N) ./ N
    println("=== Arrow–Debreu PLC market (n=$N goods, m=$M agents, L=$L pieces) ===")
    println("endowments B (n×m, rows sum to 1):")
    display(round.(B; digits=3))
    println("\nat uniform price p = ", round.(p; digits=3), ":")
    println("  budgets w_i(p)   = ", round.(budgets(p); digits=4))
    println("  agg demand g(p)  = ", round.(agg_demand(p); digits=4))
    println("  agg share p∘g(p) = ", round.(agg_share(p); digits=4))
    println("  walras residual  = ", round(walras_check(p); digits=8))
    println("\ntry: agg_demand(rand_simplex()), sample_data(10), budgets(p), ...")
end

# Convenience: a random simplex price.
rand_simplex() = (e = -log.(rand(N)); e ./ sum(e))
