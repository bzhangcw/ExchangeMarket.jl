using Revise
using Random, SparseArrays, LinearAlgebra
using JuMP, MosekTools
using Plots, LaTeXStrings, Printf
import MathOptInterface as MOI

using ExchangeMarket

include("../tools.jl")
include("../plots.jl")
include("./setup.jl")
switch_to_pdf(;)


Random.seed!(1)
results = []
results_phi = Dict()
rrange = [0.9]
n = 1000
m = 3000

# method_filter(name) = name ∈ [:LogBar, :Tât, :PropRes]
method_filter(name) = name ∈ [:LogBar, :LogBarPCG, :PathFol, :Tât, :PropRes]
# method_filter(name) = name ∈ [:LogBar, :LogBarPCG, :PathFol, :Tât, :PropRes]
table_time = []



for ρ in rrange
    f0 = FisherMarket(m, n; ρ=ρ, bool_unit=true, scale=30.0, sparsity=0.8)
    linconstr = LinearConstr(1, n, ones(1, n), [sum(f0.w)])
    ρfmt = @sprintf("%+.2f", ρ)
    σfmt = @sprintf("%+.2f", f0.σ)
    # -----------------------------------------------------------------------
    # compute ground truth
    # -----------------------------------------------------------------------
    f1 = copy(f0)
    p₀ = ones(n) * sum(f1.w) ./ (n)
    x₀ = ones(n, m) ./ m
    f1.x .= x₀
    f1.p .= p₀
    # use log-barrier method to compute ground truth
    (name, method, kwargs) = method_kwargs[1]
    alg = method(
        n, m, p₀;
        linconstr=linconstr,
        kwargs...
    )
    traj = opt!(
        alg, f1;
        # loginterval=1,
        keep_traj=true
    )
    pₛ = copy(alg.p)
    results_phi[ρ] = pₛ
    for (name, method, kwargs) in method_kwargs
        !method_filter(name) && continue

        f1 = copy(f0)
        p₀ = ones(n) * sum(f1.w) ./ (n)
        x₀ = ones(n, m) ./ m
        f1.x .= x₀
        f1.p .= p₀
        alg = method(
            n, m, p₀;
            linconstr=linconstr,
            kwargs...
        )
        traj = opt!(
            alg, f1;
            keep_traj=true,
            pₛ=pₛ,
            tol_p=1e-7,
            maxiter=100,
        )
        push!(results, ((name, ρ), (alg, traj, f1)))
        push!(table_time, (n, m, name, ρ, traj[end].t))

    end
end


for ((name, ρ), (alg, traj, f1)) in results
    validate(f1, alg)
end

for ρ in rrange
    for attr in [:k, :t]
        ρfmt = @sprintf("%+.2f", ρ)
        σfmt = @sprintf("%+.2f", ρ / (1 - ρ))
        fig = generate_empty(; shape=:square)
        # fig = generate_empty(; shape=:wide, settick=false)
        plot!(
            fig,
            # ylabel=L"$\|\nabla \varphi\|_{\mathbf{p}}^*$",
            ylabel=L"$\|\mathbf{p} - \mathbf{p}^*\|$",
            # ylabel=L"$\Psi(\mathbf{p}) - \Psi^*$",
            title=L"$\rho := %$ρfmt~(\sigma := %$σfmt)$",
            legendbackgroundcolor=RGBA(1.0, 1.0, 1.0, 0.8),
            yticks=10.0 .^ (-13:3:3),
            xtickfont=font(18),
            ytickfont=font(18),
            # xscale=attr == :k ? :log10 : :identity,
            xscale=:identity,
            # palette=:Accent_6,
            palette=:default,
        )
        if attr == :k
            plot!(
                fig,
                xticks=[10, 20, 50, 100, 200, 500]
            )
        end

        for (i, ((mm, _ρ), (alg, traj, f1))) in enumerate(results)
            if _ρ != ρ
                continue
            end
            traj_pp₊ = map(pp -> pp.D, traj)
            traj_tt₊ = map(pp -> getfield(pp, attr), traj)
            @info "" traj[end].t
            # plot!(
            #     fig, traj_tt₊, traj_pp₊,
            #     label="",
            #     linewidth=2,
            #     linestyle=:dash,
            #     # markershape=marker_style[i],
            #     color=colors[mm]
            # )
            if mm ∈ [:Tât]
                plot!(
                    fig, traj_tt₊[1:5:end], traj_pp₊[1:5:end],
                    label=L"\texttt{%$mm}", linewidth=2,
                    linestyle=:dash,
                    markershape=marker_style[mm],
                    color=colors[mm]
                )
            elseif mm ∈ [:PropRes]
                plot!(
                    fig, traj_tt₊[1:3:end], traj_pp₊[1:3:end],
                    label=L"\texttt{%$mm}", linewidth=2,
                    linestyle=:dash,
                    markershape=marker_style[mm],
                    color=colors[mm]
                )
            else
                plot!(
                    fig, traj_tt₊, traj_pp₊,
                    label=L"\texttt{%$mm}", linewidth=2,
                    linestyle=:dash,
                    markershape=marker_style[mm],
                    color=colors[mm]
                )
            end
        end
        savefig(fig, "$(RESULTSDIR)/traj_x_$(ρfmt)_$(attr).pdf")
        savefig(fig, "$(RESULTSDIR)/traj_x_$(ρfmt)_$(attr).tex")
        println("saved to $(RESULTSDIR)/traj_x_$(ρfmt)_$(attr).tex")
    end
end

using DataFrames
df = DataFrame(table_time)
rename!(df, [:n, :m, :method, :ρ, :t])
df



