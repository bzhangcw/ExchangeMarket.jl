using Revise
using SparseArrays, LinearAlgebra
using JuMP, MosekTools
import MathOptInterface as MOI
using Plots, LaTeXStrings

using ExchangeMarket

include("tools.jl")
include("plots.jl")
switch_to_pdf(;)

m = 200
n = 15
f0 = FisherMarket(m, n)
# create a copy of fisher
f1 = copy(f0)

c0 = Conic(n, m)
create_primal_linear(c0, f0)
validate(f0, c0)


μ = 5e-1
p₀ = 0.1 * ones(n)
x₀ = 0.1 * ones(m, n)
f1.x .= x₀
f1.p .= p₀


# alg = HessianBar(n, m, p₀, μ; optimizer=DualCESConic, option_grad=:usex)
alg = HessianBar(n, m, p₀, μ; optimizer=DualCESConic, option_grad=:usex)
traj = opt!(alg, f1; maxiter=100, loginterval=10, keep_traj=true)

