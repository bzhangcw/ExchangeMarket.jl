# -----------------------------------------------------------------------
# Test Conic model for linear exchange markets
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

m = 200
n = 20
ρ = 0.5
# create a copy of fisher
f0 = ExchangeMarket.toy_fisher()
f1 = copy(f0)

# primal ces EG program
c0 = Conic(n, m)
create_primal(c0, f0)
validate(f0, c0)

# dual ces EG program
c1 = Conic(n, m)
create_dual(c1, f1)
validate(f1, c1)

