using Revise
using SparseArrays, LinearAlgebra
using JuMP, MosekTools
import MathOptInterface as MOI
using ExchangeMarket

include("tools.jl")

m = 2000
n = 50
f0 = FisherMarket(m, n)
create_jump_model(f0)
solve_jump_model(f0);
validate(f0)


# create a copy of fisher
fisher = copy(f0)
μ = 1e-2
p₀ = 0.1 * ones(n)
x₀ = 0.1 * ones(m, n)
fisher.x .= x₀
fisher.p .= p₀


alg = HessianBar(n, m, p₀, μ)
optimizer = optim_newton
opt!(p₀, alg, fisher; maxiter=100, optimizer=optimizer, loginterval=10)


# debug
_xx = fisher.x[1, :]
_f, _g, _H, _u, _∇u = produce_functions_from_subproblem(alg, fisher, 1)
info = optim_newton(_f, _g; H=_H, x₀=_xx)