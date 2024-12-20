using Revise
using SparseArrays, LinearAlgebra
using JuMP, MosekTools
using Plots, LaTeXStrings, Printf
import MathOptInterface as MOI

using NLPModels, NLPModelsJuMP, JuMP, MadNLPHSL, MadNLPMumps
using ExchangeMarket
using ExchangeMarket: proj

using Optim, Gurobi
include("tools.jl")
include("plots.jl")
switch_to_pdf(;)

include("arrow.jl")


p0 = ones(n)
w0 = b * p0 .- 0.01


res = ExchangeMarket.__optim_newton(; f=ϵ, g=gϵ, H=Hϵ, x₀=0.5 * p0, verbose=true, store_trace=true)
# res = ExchangeMarket.__optim_newton(; f=Ψ, g=gΨ, H=HΨ, x₀=0.5 * p0, verbose=true, store_trace=true)

traj = res.md.trace
fig = generate_empty(; shape=:square, settick=false)
traj_psi = map(pp -> pp.value, traj)
plot!(fig, traj_psi, label=L"\Psi", yscale=:log10, linewidth=2, markershape=:circle)
title!(L"\textrm{CES Economy:}~$\rho := %$ρfmt~(\sigma_i := %$σfmt)$")
savefig(fig, "traj_psi.pdf")


# evaluation
pₛ = res.x

f0.w .= b * pₛ

# check
alg = HessianBar(
    n, m, p0;
    μ=sum(p0 .^ 2) / f0.n,
    optimizer=EGConicAC
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



