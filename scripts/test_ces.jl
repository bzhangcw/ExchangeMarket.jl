using Revise
using SparseArrays, LinearAlgebra
using JuMP, MosekTools
import MathOptInterface as MOI
using Plots, LaTeXStrings

using ExchangeMarket

include("tools.jl")
include("plots.jl")
switch_to_pdf(;)

m = 2000
n = 200
ρ = 0.3
bool_run_conic = false

f0 = FisherMarket(m, n; ρ=ρ)
f1 = copy(f0)

if bool_run_conic
    c0 = Conic(n, m)
    create_primal_ces(c0, f0, ρ)
    validate(f0, c0)
end

μ = 1e0
p₀ = 0.1 * ones(n)
x₀ = 0.1 * ones(m, n)
f1.x .= x₀
f1.p .= p₀


alg = HessianBar(
    n, m, p₀, μ;
    optimizer=EGConicAC,
    tol=1e-10
)
traj = solve!(
    alg, f1;
    maxiter=200,
    loginterval=10,
    keep_traj=true
)
validate(f1, alg)

if bool_run_conic
    traj_pp₊ = map(pp -> norm(pp - c0.p), traj)
    fig = generate_empty()
    plot!(fig, traj_pp₊, label=L"$\|p^k - p^*\|_2$", linewidth=2, markershape=:circle)
end
