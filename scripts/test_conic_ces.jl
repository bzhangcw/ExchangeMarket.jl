# -----------------------------------------------------------------------
# Test Conic model for CES exchange markets
# -----------------------------------------------------------------------
using Revise
using SparseArrays, LinearAlgebra
using JuMP, MosekTools
import MathOptInterface as MOI
using Plots, LaTeXStrings

using ExchangeMarket

include("tools.jl")
include("plots.jl")
switch_to_pdf(;)

ρ = 0.3
# create a copy of fisher
f0 = ExchangeMarket.toy_fisher(ρ)
m, n = f0.m, f0.n
f1 = copy(f0)
f2 = copy(f0)
f3 = copy(f0)

# primal ces EG program
c0 = Conic(n, m)
create_primal_ces(c0, f0, ρ)
validate(f0, c0)

# dual ces EG program type I
c1 = Conic(n, m)
create_dual_ces_type_i(c1, f1, ρ)
validate(f1, c1)


# dual ces EG program type II
c2 = Conic(n, m)
create_dual_ces_type_ii(c2, f2, ρ)
validate(f2, c2)

# dual ces EG program type I
c3 = Conic(n, m)
c3.p .= c1.p
create_dual_ces_type_i(c3, f3, ρ, false)
validate(f3, c3)


