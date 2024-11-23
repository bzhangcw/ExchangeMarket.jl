using Revise
using SparseArrays, LinearAlgebra
using JuMP, MosekTools
import MathOptInterface as MOI
using ExchangeMarket

m = 5
n = 10
fisher₀ = FisherMarket(m, n)
create_jump_model(fisher₀)
solve_jump_model(fisher₀);
validate(fisher₀)


# create a copy of fisher
fisher = copy(fisher₀)
μ = 0.1
p₀ = rand(n)
x₀ = 0.1 * ones(m, n)
fisher.x .= x₀
fisher.p .= p₀

alg = HessianBar(n, m, p₀, μ)