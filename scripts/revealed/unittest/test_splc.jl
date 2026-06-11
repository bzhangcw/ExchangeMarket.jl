# Interactive test: SPLC (separable PLC) market.
#
# Builds a small SPLC market, plots the per-good utility f_j(x_j) for a
# selected good j (should be concave increasing, piecewise linear), and
# sanity-checks the greedy demand (budget exhaustion + Walras' law).
#
# Usage:
#   julia -i --project=scripts scripts/revealed/test_splc.jl
#   julia> plot_fj(1)              # plot f_j for good j=1 of agent 1 → PDF
#   julia> plot_fj(2; i=3)         # good 2 of agent 3
#   julia> x, u = solve_splc_demand(agents[1], rand_simplex(), 0.2)
#   julia> agg_demand(rand_simplex())
#   julia> sample_data(10)

using Random, LinearAlgebra
using Plots, LaTeXStrings

include("../../plots.jl")            # generate_empty, switch_to_pdf — house plot style
include("../androids/splc.jl")

# pgfplotsx must be selected at top level (same constraint as run_plot.jl).
switch_to_pdf(; bool_use_html=false)

# ---- knobs --------------------------------------------------------------
const N = 4       # goods
const M = 5       # agents
const L = 4       # pieces per good
const SEED = 42
const OUTDIR = mkpath(joinpath(@__DIR__, "..", "..", "results", "test_splc"))

# ---- build the SPLC market ----------------------------------------------
Random.seed!(SEED)
const agents = [random_splc_agent(N, L, intercept=false) for _ in 1:M]
const w_vec = let w = rand(M)
    w ./ sum(w)
end

# ---- plotting -------------------------------------------------------------
"""
    plot_fj(j; i=1, xmax=nothing, npts=400) -> path

Plot the per-good utility f_j(x_j) of agent `i`, good `j`. The curve is
concave increasing piecewise linear; kinks are marked with dots. Saves a
PDF to OUTDIR and returns the path.
"""
function plot_fj(j::Int; i::Int=1, xmax=nothing, npts::Int=400)
    ag = agents[i]
    kinks = ag.kink[j]
    x_hi = isnothing(xmax) ? (isempty(kinks) ? 4.0 : 2.0 * maximum(kinks)) : float(xmax)
    xs = range(0.0, x_hi; length=npts)
    fs = [splc_fj(ag, j, x) for x in xs]

    f = generate_empty(; shape=:wide)
    plot!(f, collect(xs), fs;
        label=L"f_j(x_j)", linewidth=3, color=1,
        xlabel=L"x_j", ylabel=L"f_j(x_j)",
        yscale=:identity,
        legend=:bottomright)
    # mark the kinks
    if !isempty(kinks)
        scatter!(f, kinks, [splc_fj(ag, j, k) for k in kinks];
            label=L"\textrm{kinks}", color=2, markersize=5)
    end
    path = joinpath(OUTDIR, "splc_fj_agent$(i)_good$(j).pdf")
    savefig(f, path)
    @info "saved" path slopes = ag.a[j] kinks = kinks
    return path
end

# ---- demand accessors ------------------------------------------------------
"""
    agg_demand(p; budgets=w_vec) -> Vector

Aggregate SPLC demand g(p) = Σ_i x_i(p, w_i) (greedy per agent, no LP).
"""
function agg_demand(p::AbstractVector; budgets=w_vec)
    g = zeros(N)
    for i in 1:M
        x_i, _ = solve_splc_demand(agents[i], p, budgets[i])
        g .+= x_i
    end
    return g
end

"""
    walras_check(p) -> Float64

⟨p, g(p)⟩ − Σ_i w_i: ≈ 0 since every SPLC agent with strictly positive
slopes exhausts its budget (nondecreasing utility ⇒ non-satiation).
"""
walras_check(p::AbstractVector) = dot(p, agg_demand(p)) - sum(w_vec)

"""
    sample_data(K; seed=SEED) -> Vector{Tuple}

K revealed-preference samples (p_k, g_k), simplex-uniform prices.
"""
sample_data(K::Int; seed=SEED) = produce_revealed_preferences_splc(agents, w_vec, K, N; seed=seed)

rand_simplex() = (e = -log.(rand(N)); e ./ sum(e))

# ---- checks + demo ---------------------------------------------------------
let
    println("=== SPLC market (n=$N goods, m=$M agents, L=$L pieces/good) ===")

    # 1. concavity + monotonicity check on every (agent, good) via sampled secants
    ok_concave, ok_increasing = true, true
    for i in 1:M, j in 1:N
        xs = range(0.0, 5.0; length=200)
        fs = [splc_fj(agents[i], j, x) for x in xs]
        d1 = diff(fs)
        ok_increasing &= all(d1 .>= -1e-12)          # nondecreasing
        ok_concave &= all(diff(d1) .<= 1e-12)        # slopes nonincreasing
    end
    println("f_j concave on all (i,j):     ", ok_concave)
    println("f_j increasing on all (i,j):  ", ok_increasing)

    # 2. demand sanity at uniform price
    p = ones(N) ./ N
    g = agg_demand(p)
    println("agg demand g(p) at uniform p: ", round.(g; digits=4))
    println("walras residual ⟨p,g⟩-Σw:     ", round(walras_check(p); digits=10))

    # 3. plot good 1 of agent 1
    path = plot_fj(1)
    println("\nutility plot saved to: ", path)
    println("\ntry: plot_fj(2), plot_fj(1; i=2), agg_demand(rand_simplex()), sample_data(10)")
end
