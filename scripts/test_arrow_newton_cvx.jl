
using SparseArrays, LinearAlgebra
using JuMP, MosekTools
using Plots, LaTeXStrings, Printf
import MathOptInterface as MOI

using NLPModels, NLPModelsJuMP, JuMP, MadNLPHSL, MadNLPMumps
using ExchangeMarket
using ExchangeMarket: proj

using Optim
include("tools.jl")
include("plots.jl")
switch_to_pdf(;)


include("arrow.jl")


p0 = 0.5 * ones(n)
w0 = b * p0 .- 0.01


# model 
model2 = Model()
@variable(model2, vp[i=1:n] >= 0.0, start = p0[i])
@variable(model2, vw[i=1:m] >= 0.0, start = w0[i])
@variable(model2, vv[i=1:m] >= 0.0, start = w0[i])
@constraint(model2, vvc[i=1:m], vv[i] == sum(vp[j] * b[i, j] for j in 1:n) - vw[i])
# @NLconstraint(model2, cx[i=1:m, j=1:n], vx[i, j] == vp[i] * (c[i, j] / vp[j])^(1 + σ) / sum(vp[j]^(-σ) * c[i, j]^(1 + σ) for j in 1:n))
# @constraint(model2, cz[j=1:n], sum(vx[i, j] for i in 1:m) - 1 .== vz[j])
# @constraint(model2, sum(vp) - 1 == 0)
@NLobjective(model2,
    Min,
    sum(vp[j] * bb[j] for j in 1:n) +
    # 0.01 * sum(log(sum(vp[j] * b[i, j] for j in 1:n) - vw[i]) for i in 1:m) +
    # sum(vw[i] * log(vw[i]) for i in 1:m) +
    sum(vw[i] * log(sum(vp[j]^(-σ) * c[i, j]^(1 + σ) for j in 1:n)^(1 / σ)) for i in 1:m)
)


nlp2 = MathOptNLPModel(model2)

solver2 = MadNLPSolver(nlp2,
    linear_solver=MadNLPMumps.MumpsSolver,
    max_wall_time=900.0,
    max_iter=1000,
    # print_level=MadNLP.TRACE,
    file_print_level=MadNLP.TRACE,
    tol=1e-16,
    # output_file="$fname.log",
    kkt_system=MadNLP.SparseCondensedKKTSystem,
    # nlp_scaling=false,
    mu_init=1e-4,
    # hessian_approximation=MadNLP.CompactLBFGS,
    # quasi_newton_options=MadNLP.QuasiNewtonOptions(max_history=15)
)

r2 = MadNLP.solve!(solver2)

# evaluation
pₛ = r2.solution[1:n]
wₛ = r2.solution[n+1:n+m]
vₛ = r2.solution[n+m+1:end]
f0.w .= b * pₛ
# check
alg = MirrorDec(
    n, m, rand(n);
    optimizer=EGConicAC,
    tol=1e-10,
    α=1e3,
    option_step=:eg,
    option_stepsize=:cc13
)
traj = opt!(
    alg, f0;
    keep_traj=true
)

xk = xp(alg.p)

@info """checking...
p == w?      $(norm(b * alg.p - f0.w))
sum(x) == 1? $(norm(sum(xk; dims=1) .- 1))
"""
