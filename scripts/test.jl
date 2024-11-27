using Revise
using SparseArrays, LinearAlgebra
using JuMP, MosekTools
import MathOptInterface as MOI
using Plots, LaTeXStrings

using ExchangeMarket

include("tools.jl")
include("plots.jl")
switch_to_pdf(;)

m = 20
n = 2
f0 = FisherMarket(m, n)
# create a copy of fisher
f1 = copy(f0)
create_dual_jump_model(f0)
validate(f0)



μ = 1e-1
p₀ = 0.1 * ones(n)
x₀ = 0.1 * ones(m, n)
f1.x .= x₀
f1.p .= p₀


alg = HessianBar(n, m, p₀, μ; optimizer=EGConic)
traj = solve!(p₀, alg, f1; maxiter=200, loginterval=10, keep_traj=true)
traj_pp₊ = map(pp -> norm(pp - f0.p), traj)


# debug
# _xx = fisher.x[1, :]
# _f, _g, _H, _u, _∇u = produce_functions_from_subproblem(alg, fisher, 1)
# info = optim_newton(_f, _g; H=_H, x₀=_xx)
fig = generate_empty()
plot!(fig, traj_pp₊, label=L"$\|p^k - p^*\|_2$")

