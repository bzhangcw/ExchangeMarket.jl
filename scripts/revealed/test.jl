using Pkg
Pkg.activate("../")
# Pkg.instantiate()

using Revise
using Random, SparseArrays, LinearAlgebra
using JuMP, MosekTools
using Plots, LaTeXStrings, Printf
import MathOptInterface as MOI

using ExchangeMarket

include("../tools.jl")
include("../plots.jl")
switch_to_pdf(; bool_use_html=true)

include("./util.jl")

m = 3
n = 10
ρ = 0.8
b = rand(n, m)
b ./= sum(b; dims=2)
e = ones(n)

# -----------------------------------------------------------------------
# setup market
# -----------------------------------------------------------------------
f0 = FisherMarket(m, n; ρ=ρ * ones(m), scale=30.0, sparsity=0.99)
linconstr = LinearConstr(1, n, ones(1, n), [1.0])

# -----------------------------------------------------------------------
# compute ground truth
# -----------------------------------------------------------------------
f1 = copy(f0)
p₀ = ones(n) ./ (n)
x₀ = ones(n, m) ./ m
f1.x .= x₀
alg = HessianBar(
    n, m, p₀;
    linconstr=linconstr,
)
alg.linsys = :direct

play!(alg, f1)
grad!(alg, f1)
eval!(alg, f1)
hess!(alg, f1)

idx = 1
cr = f1.c[:, idx] .^ (1 / ρ)
xc = _conic_ces_primal(;
    p=alg.p,
    n=f1.n,
    cr=cr,
    w=f1.w[idx],
    ρ=f1.ρ[1],
    verbose=true
)

# should be the same
[xc f1.x[:, idx]]

include("./util.jl")

γ = alg.p .* f1.x[:, 1] ./ f1.w[1]
pmat = zeros(n, 1)
gmat = zeros(n, 1)
pmat[:, 1] .= alg.p
gmat[:, 1] .= γ

# -----------------------------------------------------------------------
# try if I have only one gamma
y, δ, A, md = _linear_prog_ces_gamma_single(pmat=pmat, gmat=gmat, verbose=true)
objective_value(md)

K = 12
gmat = zeros(n, K)
pmat = zeros(n, K)
zmat = zeros(n, K)
# use the first play
idx = 1
for k = 1:K
    alg.p .= rand(n)
    play!(alg, f1)
    grad!(alg, f1)
    eval!(alg, f1)
    pmat[:, k] .= alg.p
    gmat[:, k] .= alg.p .* f1.x[:, idx] ./ sum(alg.p .* f1.x[:, idx])
    zmat[:, k] .= sum(f1.x, dims=2)
end

y, δ, A, md = _linear_prog_ces_gamma_single(pmat=pmat, gmat=gmat, δ₁=nothing, verbose=true)
@info "" δ f1.σ
γv(p, a) = begin
    γ = exp.(y) .* (p .^ (-δ))
    γ ./= exp(a)
end
γfit = zeros(n, K)
for k = 1:K
    γfit[:, k] .= γv(pmat[:, k], A[k])
end
@info "optimal value" objective_value(md)
@info "tightness" (gmat - γfit) .|> abs |> maximum

gmat ./ γfit

m = 3
n = 10
ρ = 0.8
b = rand(n, m)
b ./= sum(b; dims=2)
e = ones(n)

# -----------------------------------------------------------------------
# setup market
# -----------------------------------------------------------------------
f0 = FisherMarket(m, n; ρ=ρ * ones(m), scale=30.0, sparsity=0.99)
linconstr = LinearConstr(1, n, ones(1, n), [1.0])

# -----------------------------------------------------------------------
# compute ground truth
# -----------------------------------------------------------------------
f1 = copy(f0)
p₀ = ones(n) ./ (n)
x₀ = ones(n, m) ./ m
f1.x .= x₀
alg = HessianBar(
    n, m, p₀;
    linconstr=linconstr,
)
alg.linsys = :direct

include("./master.jl")
include("./pricing.jl")

# Generate revealed preferences from the market
K = 12
Ξ = produce_revealed_preferences(alg, f1, K; seed=42)

# Compute bidding matrix from market parameters
γ = compute_gamma_from_market(f1, Ξ)

w, s, model_primal = solve_master_problem(Ξ, γ; verbose=true)
u, μ, model_dual = solve_dual_problem(Ξ, γ; verbose=true)

# Validate strong duality
println("=== Strong Duality Validation ===")
println("Primal objective (Q):     ", objective_value(model_primal))
println("Dual objective (Q_*):     ", objective_value(model_dual))
println("Gap:                      ", abs(objective_value(model_primal) - objective_value(model_dual)))
println()
println("Primal solution w - ground truth:   ", abs.(w - f1.w) |> maximum)
println("Dual solution μ:          ", μ)
println("Dual solution u:          ", u[1:5])

include("./master.jl")
include("./pricing.jl")

m = 3
n = 10
K = 12 # how many revealed preferences pairs we have
ρ = 0.8
b = rand(n, m)
b ./= sum(b; dims=2)
e = ones(n)

# -----------------------------------------------------------------------
# setup market
# -----------------------------------------------------------------------
f0 = FisherMarket(m, n; ρ=ρ * ones(m), scale=30.0, sparsity=0.99)
linconstr = LinearConstr(1, n, ones(1, n), [1.0])

# -----------------------------------------------------------------------
# compute ground truth
# -----------------------------------------------------------------------
# Generate revealed preferences from the underlying market
f1 = copy(f0)
p₀ = ones(n) ./ (n)
x₀ = ones(n, m) ./ m
f1.x .= x₀
alg = HessianBar(
    n, m, p₀;
    linconstr=linconstr,
)
alg.linsys = :direct

Ξ = produce_revealed_preferences(alg, f1, K; seed=42);

# Column generation: iteratively add new CES androids to fit revealed preferences
# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
max_iters = 50
tol = 1e-6  # tolerance for reduced cost

# -----------------------------------------------------------------------
# Initialize surrogate market with one random CES agent
# -----------------------------------------------------------------------
fa = FisherMarket(1, n; ρ=rand(1), scale=30.0, sparsity=0.99)
γ_ref = Ref(compute_gamma_from_market(fa, Ξ))

# Track history
history = Dict(
    :primal_obj => Float64[],
    :dual_obj => Float64[],
    :reduced_cost => Float64[],
    :num_agents => Int[]
)

println("=== Column Generation for CES Android Fitting ===\n")

for iter in 1:max_iters
    println("--- Iteration $iter ($(fa.m) agents) ---")

    # Solve master problem (primal and dual)
    w, s, model_primal = solve_master_problem(Ξ, γ_ref[]; verbose=false)
    u, μ, model_dual = solve_dual_problem(Ξ, γ_ref[]; verbose=false)

    primal_obj = objective_value(model_primal)
    dual_obj = objective_value(model_dual)

    push!(history[:primal_obj], primal_obj)
    push!(history[:dual_obj], dual_obj)
    push!(history[:num_agents], fa.m)

    println("  Primal obj: $(round(primal_obj, digits=6))")
    println("  Dual obj:   $(round(dual_obj, digits=6))")
    println("  Weights w:  $(round.(w, digits=4))")

    # Solve pricing subproblem to find best new android
    y_opt, σ_opt, γ_new, pricing_obj = solve_pricing(Ξ, u)

    # Compute reduced cost
    rc = reduced_cost(γ_new, u, μ)
    push!(history[:reduced_cost], rc)

    println("  Pricing σ:  $(round(σ_opt, digits=4))")
    println("  Reduced cost: $(round(rc, digits=6))")

    # Check termination: if reduced cost <= 0, no improving android exists
    if rc <= tol
        println("\n✓ Converged! No improving android found (reduced cost ≤ $tol)")
        break
    end

    # Update existing agents' budgets with optimal weights from master
    fa.w .= w

    # Add new android to γ matrix (in-place via Ref)
    add_to_gamma!(γ_ref, γ_new)

    # Recover CES parameters and add to surrogate market
    c_new, ρ_new = recover_ces_params(y_opt, σ_opt)
    w_new = 0.0  # placeholder, will be updated in next iteration
    add_to_market!(fa, c_new, ρ_new, w_new)

    println("  → Added android $(fa.m) with ρ=$(round(ρ_new, digits=4))")
    println()

    if iter == max_iters
        println("\n⚠ Maximum iterations reached")
    end
end

# Final solve to get optimal weights for all agents
w_final, _, _ = solve_master_problem(Ξ, γ_ref[]; verbose=false)
fa.w .= w_final

println("\n=== Final Results ===")
println("Number of fitted agents: $(fa.m)")
println("Number of nonzero agents: $(sum(w_final .> 1e-6))")
println("Ground truth agents:     $(f1.m)")
println("Final primal objective:  $(round(history[:primal_obj][end], digits=6))")
println("Final weights: $(round.(w_final, digits=4))")

p1, g1 = Ξ[1]

alg.p .= p1
ExchangeMarket.play!(alg, fa)
ga = sum(fa.x, dims=2)
[ga g1]



fa.x

fa.ρ

γ_ref[]
