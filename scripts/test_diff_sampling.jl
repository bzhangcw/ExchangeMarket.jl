using Revise
using SparseArrays, LinearAlgebra
using JuMP, MosekTools
import MathOptInterface as MOI
using Plots, LaTeXStrings
using ProgressMeter

using ExchangeMarket

include("tools.jl")
include("plots.jl")
switch_to_pdf(;)

ℓ = 12
m = 2^ℓ
n = 15
μ = 5e-1
f0 = FisherMarket(m, n)
p₀ = 0.1 * ones(n)
x₀ = 0.1 * ones(m, n)

@info "finished setting up"
# benchmark
fₛ = copy(f0)
fₛ.x .= x₀
fₛ.p .= p₀
algₛ = HessianBar(n, m, p₀, μ; optimizer=EGConic)
algₛ.pb .= algₛ.p
# update all sub-problems of all agents i ∈ I
play!(algₛ, fₛ; ϵᵢ=0.1 * algₛ.μ, verbose=false)
@info "finished benchmark"

# -------------------------------------------------------------------
# compute dual function value, gradient and Hessian
# !evaluate gradient first;
grad!(algₛ, fₛ)
eval!(algₛ, fₛ)
hess!(algₛ, fₛ)
Hₛ = copy(algₛ.H) ./ m
gₛ = copy(algₛ.∇) ./ m
dₛ = Hₛ \ gₛ


# test different sample size
g_diff = []
H_diff = []
d_diff = []
numbers = []
p = Progress(ℓ - 1)
for k in 1:ℓ
    # create a copy of fisher
    _m = 2^k
    index_set = rand(1:m, _m)
    f1 = FisherMarket(_m, n)
    f1.q .= f0.q .* (_m / m)
    f1.x .= 0.1 * ones(_m, n)
    f1.p .= 0.1 * ones(n)
    f1.c .= f0.c[index_set, :]
    f1.val_u .= f0.val_u[index_set]
    f1.val_∇u .= f0.val_∇u[index_set, :]
    f1.w .= f0.w[index_set]
    f1.u = (x, i) -> f1.c[i, :]' * x
    f1.∇u = (x, i) -> f1.c[i, :]
    f1.val_u = zeros(_m)
    f1.val_∇u = f1.c

    alg = HessianBar(n, _m, p₀, μ; optimizer=EGConic)
    alg.pb .= alg.p
    # update all sub-problems of all agents i ∈ I
    play!(alg, f1; ϵᵢ=0.1 * alg.μ, verbose=false)

    # -------------------------------------------------------------------
    # compute dual function value, gradient and Hessian
    # !evaluate gradient first;
    grad!(alg, f1)
    eval!(alg, f1)
    hess!(alg, f1)
    Hₖ = copy(alg.H)
    gₖ = copy(alg.∇)
    push!(g_diff, norm(gₖ ./ _m - gₛ) / norm(gₛ))
    push!(H_diff, norm(Hₖ ./ _m - Hₛ) / norm(Hₛ))
    push!(numbers, _m)
    next!(p)
end

# debug
# _xx = fisher.x[1, :]
# _f, _g, _H, _u, _∇u = produce_functions_from_subproblem(alg, fisher, 1)
# info = optim_newton(_f, _g; H=_H, x₀=_xx)
fig = generate_empty()
plot!(fig, numbers, g_diff, label=L"$\|\tilde g - g\|_2/\|g\|_2$", linewidth=2, markershape=:circle)
plot!(fig, numbers, H_diff, label=L"$\|\tilde H - H\|_2/\|H\|_2$", linewidth=2, markershape=:circle, xlabel=L"$|I(\mathbf{\xi})|$")
savefig(fig, "/tmp/diff.pdf")

