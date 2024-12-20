
using SparseArrays, LinearAlgebra
using JuMP, MosekTools
using Plots, LaTeXStrings, Printf
import MathOptInterface as MOI

using NLPModels, NLPModelsJuMP, JuMP, MadNLPHSL, MadNLPMumps
using ExchangeMarket

using Optim
include("tools.jl")
include("plots.jl")
switch_to_pdf(;)

include("arrow.jl")

# model 
model1 = Model()
@variable(model1, p[1:n] >= 1e-3, start = 1.0)
@variable(model1, x[1:m, 1:n], start = 1.0)
@variable(model1, θ >= 0, start = 1.0)
@NLconstraint(model1, cx[i=1:m, j=1:n], x[i, j] == p[i] * (c[i, j] / p[j])^(1 + σ) / sum(p[j]^(-σ) * c[i, j]^(1 + σ) for j in 1:n))
@constraint(model1, cs[j=1:n], sum(x[i, j] for i in 1:m) <= 1 + θ)
# @NLconstraint(model, sum(p[i] * log(p[i] / w[i]) for i in 1:n) <= 1e-5)
@NLobjective(model1,
    Min,
    θ
)


nlp1 = MathOptNLPModel(model1)

solver1 = MadNLPSolver(nlp1;
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

r1 = MadNLP.solve!(solver1)

# evaluation
pw1 = r1.solution
pₛ1 = pw1[1:n]

f0.w .= pₛ1
# check
alg1 = MirrorDec(
    n, m, rand(n);
    optimizer=EGConicAC,
    tol=1e-10,
    α=1e3,
    option_step=:eg,
    option_stepsize=:cc13
)
traj1 = opt!(
    alg1, f0;
    keep_traj=true
)

print("finished")

