using Revise
using SparseArrays, LinearAlgebra
using JuMP, MosekTools
using Plots, LaTeXStrings, Printf, Random
import MathOptInterface as MOI

using ExchangeMarket

include("../tools.jl")
include("../plots.jl")
switch_to_pdf(;)

Random.seed!(1234)
n = 500
mrange = [10, 20, 30, 40, 50, 100, 150, 200, 500, 1000, 2000, 5000]
# m = 10000
opnorms = []
opnorms_err = []
opnorms_err_A = []
opnorms_err_A_frobe = []
opnorms_err_A_frobe_approx = []
results = []
for m in mrange

    ρ = 0.7

    f0 = FisherMarket(m, n; ρ=ρ, bool_unit=true, bool_unit_wealth=true, sparsity=0.95)
    ρfmt = @sprintf("%+.1f", ρ)
    σfmt = @sprintf("%+.1f", f0.σ)

    linconstr = LinearConstr(1, n, ones(1, n), [sum(f0.w)])

    method, kwargs = HessianBar, Dict(:tol => 1e-10, :maxiter => 20, :linsys => :DRq, :option_step => :logbar, :linconstr => linconstr)
    # for (method, kwargs) in method_kwargs
    f1 = copy(f0)
    p₀ = ones(n) ./ (2 * n)
    x₀ = ones(n, m) ./ n
    f1.x .= x₀
    f1.p .= p₀
    alg = method(
        n, m, p₀;
        optimizer=CESAnalytic,
        kwargs...
    )
    traj = opt!(
        alg, f1;
        keep_traj=true,
        maxiter=5
    )
    push!(results, (ρ, (alg, traj, f1)))
    validate(f1, alg)
    P = diagm(alg.p)
    ExchangeMarket.__compute_exact_hess!(alg, f1)
    # compare affine scaled matrices
    H = P * Matrix(alg.H) * P
    H1 = Matrix(alg.H)
    # readily scaled
    ExchangeMarket.__ces_hess_dual!(alg, f1)
    Ha = ExchangeMarket.__assemble_drq_approx(alg.Ha)
    # normalized bidding vector
    gamma = 1 ./ f1.w .* f1.x' .* alg.p'
    G = diagm((f1.w'*gamma)[:] ./ (1 - f1.ρ)) - (f1.ρ / (1 - f1.ρ)) .* (gamma' * diagm(f1.w) * gamma)
    # 
    g = (-sum(f1.x'; dims=1).+1)[:]
    # omega 
    omega = f1.w ./ sum(f1.w)
    Ω = sum(f1.w) * f1.ρ / (1 - f1.ρ)
    xi = sum(omega .* gamma, dims=1)[:]
    Ga = diagm((f1.w'*gamma)[:] ./ (1 - f1.ρ)) - Ω * xi * xi'
    gs = gamma .- xi'
    A = gs' * diagm(omega) * gs
    Af = (sum(gs .* gs; dims=2))' * omega
    @info "H - Ha" opnorm(H) opnorm(H - Ha) (opnorm(H - Ha) / opnorm(H))
    @info "G - H" opnorm(G - H)
    @info "Ga - Ha" opnorm(Ga - Ha)
    @info L"\|\gamma - \xi\|^2" Af
    push!(opnorms, opnorm(H))
    push!(opnorms_err, opnorm(H - Ha))
    push!(opnorms_err_A, opnorm(A))
    push!(opnorms_err_A_frobe, sqrt(sum(abs2, A)))
    push!(opnorms_err_A_frobe_approx, Af[])
end

fig = generate_empty(; shape=:wide)
plot!(
    fig,
    legendbackgroundcolor=RGBA(1.0, 1.0, 1.0, 0.8),
    yticks=10.0 .^ (-6:1:1),
    xticks=[100, 1000, 5000, 10000],
    linewidth=3.0,
    xlabel=L"$m$",
    xtickfont=font(18),
    ytickfont=font(18),
    xscale=:identity,
    size=(500, 400)
)

plot!(
    fig,
    mrange,
    opnorms,
    label=L"$\|\mathbf{H}\|_2$",
    markershape=:circle,
    linewidth=3.0
)
plot!(
    fig,
    mrange,
    opnorms_err,
    label=L"$\|\mathbf{H} - \widetilde{\mathbf{H}}\|_2$",
    markershape=:rect,
    linewidth=3.0
)
# plot!(
#     fig,
#     mrange,
#     opnorms_err_A,
#     label=L"$\|\mathbf{A}\|_2$",
#     linewidth=3.0
# )
# plot!(
#     fig,
#     mrange,
#     opnorms_err_A_frobe,
#     label=L"$\|\mathbf{A}\|_F$",
#     linewidth=3.0
# )
# plot!(
#     fig,
#     mrange,
#     opnorms_err_A_frobe_approx,
#     label=L"$\widehat{\|\mathbf{A}\|_F}$",
#     linewidth=3.0
# )
# Ensure standalone class
# PGFPlotsX.CUSTOM_PREAMBLE = "\\documentclass{standalone}\\usepackage{pgfplots}"

savefig(fig, "bernstein_content.tex");
savefig(fig, "bernstein_content.pdf");


