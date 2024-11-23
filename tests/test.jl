using Revise
using SparseArrays, LinearAlgebra
using JuMP, MosekTools
import MathOptInterface as MOI
using ExchangeMarket

include("tools.jl")

m = 500
n = 20
f0 = FisherMarket(m, n)
create_jump_model(f0)
solve_jump_model(f0);
validate(f0)


# create a copy of fisher
fisher = copy(f0)
μ = 5e-1
p₀ = 0.1 * ones(n)
x₀ = 0.1 * ones(m, n)
fisher.x .= x₀
fisher.p .= p₀


alg = HessianBar(n, m, p₀, μ)
_f, _g, _H = produce_functions_from_subproblem(alg, fisher, 1)
info = optim_newton(_f, _g; H=_H, x₀=x₀[1, :])
optimizer = optim_newton
opt!(p₀, alg, fisher; maxiter=200, optimizer=optimizer, loginterval=10)