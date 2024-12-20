using Revise
using SparseArrays, LinearAlgebra
using JuMP, MosekTools
import MathOptInterface as MOI
using Plots, LaTeXStrings

using ExchangeMarket

include("tools.jl")
include("plots.jl")
switch_to_pdf(;)


ρ = 0.9
# create a copy of fisher
f0 = ExchangeMarket.toy_fisher(ρ)
m, n = f0.m, f0.n
# create a copy of fisher
f1 = copy(f0)

c0 = Conic(n, m)
create_primal_ces(c0, f0, ρ)


μ = 5e-1
p₀ = 0.1 * ones(n)
x₀ = 0.1 * ones(m, n)
f1.x .= x₀
f1.p .= p₀

alg = MirrorDec(
    n, m, p₀;
    optimizer=EGConicAC,
)
# alg = HessianBar(n, m, p₀, μ; optimizer=EGConicAC)
traj = opt!(
    alg, f1;
    maxiter=200,
    loginterval=10,
    keep_traj=true,
)
validate(f0, c0)
validate(f1, alg)