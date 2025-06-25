using Revise
using SparseArrays, LinearAlgebra
using JuMP, MosekTools
using Plots, LaTeXStrings, Printf
import MathOptInterface as MOI

using ExchangeMarket

include("tools.jl")
include("plots.jl")
switch_to_pdf(;)

m = 2000
n = 200
ρ = 0.3
bool_run_conic = false
f0 = FisherMarket(m, n; ρ=ρ, bool_unit=false)

method_kwargs = [
    (:HessianBar, HessianBar, Dict(:tol => 1e-10, :linsys => :direct)),
    # (:HessianBarQN, HessianBar, Dict(:tol => 1e-10, :maxiter => 200, :linsys => :quasinewton, :sampler => sampler)),
    (:MirrorDescEG, MirrorDec, Dict(:tol => 1e-7, :α => 500.0, :option_step => :eg, :option_stepsize => :cc13)),
    # (:MirrorDescShmyrev, MirrorDec, Dict(:tol => 1e-7, :α => 500.0, :option_step => :shmyrev, :option_stepsize => :cc13)),
]

ρfmt = @sprintf("%+.1f", ρ)
σfmt = @sprintf("%+.1f", f0.σ)
results = []
for (name, method, kwargs) in method_kwargs
    f1 = copy(f0)
    p₀ = 1e2 * ones(n)
    x₀ = 1e2 * ones(m, n)
    f1.x .= x₀
    f1.p .= p₀
    alg = method(
        n, m, p₀;
        optimizer=CESAnalytic,
        kwargs...
    )
    alg.μ = sum(p₀ .^ 2) / f1.n
    traj = opt!(
        alg, f1;
        keep_traj=true
    )
    push!(results, (name, (alg, traj, f1)))
end

if bool_run_conic
    c0 = Conic(n, m)
    create_primal_ces(c0, f0, ρ)
    push!(results, (:Conic, (c0, nothing, f0)))
end

for (name, (alg, traj, f1)) in results
    validate(f1, alg)
end

fig = generate_empty(; shape=:wide)
plot!(fig, ylabel=L"$\|\mathbf{P}\nabla \varphi\|_2$")
_title = L"\textrm{CES Economy:}~$\rho := %$ρfmt~(\sigma_i := %$σfmt)$"
@info _title
title!(_title)

for (name, (alg, traj, f1)) in results
    @info "plotting $name"
    traj_pp₊ = map(pp -> pp.gₙ, traj)
    plot!(fig, traj_pp₊, label=string(name), linewidth=2, markershape=:circle)
end
savefig(fig, "traj_eps.pdf")

fig = generate_empty(; shape=:square)
plot!(fig, ylabel=L"$\|\mathbf{p}\|_2$")
for (name, (alg, traj, f1)) in results
    traj_pp₊ = map(pp -> norm(pp.p), traj)
    plot!(fig, traj_pp₊, label=string(name), yscale=:log10, linewidth=2, markershape=:circle)
    title!(L"\textrm{CES Economy:}~$\rho := %$ρfmt~(\sigma_i := %$σfmt)$")
end
savefig(fig, "traj_pnorm.pdf")

fig = generate_empty(; use_html=false, shape=:square, settick=false)
plot!(fig, ylabel=L"$\varphi$", xscale=:log2)
for (name, (alg, traj, f1)) in results
    traj_pp₊ = map(pp -> pp.φ, traj)
    plot!(fig, 1:length(traj_pp₊), traj_pp₊, label=string(name), linewidth=2)
    title!(L"\textrm{CES Economy:}~$\rho := %$ρfmt~(\sigma_i := %$σfmt)$")
end
savefig(fig, "traj_phi.pdf")