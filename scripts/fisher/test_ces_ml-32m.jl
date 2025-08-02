using Revise
using Random, SparseArrays, LinearAlgebra
using JuMP, MosekTools
using Plots, LaTeXStrings, Printf, JLD2
import MathOptInterface as MOI

using ExchangeMarket

include("../tools.jl")
include("../plots.jl")
include("./setup.jl")
# include("./segmentation.jl")

switch_to_pdf(;)

bool_part = true
Random.seed!(1)
results = []
results_phi = Dict()
rrange = [0.6]


method_filter(name) = name ∈ [:LogBar, :LogBarPCG, :Tât, :PropRes]
# method_filter(name) = name ∈ [:PropRes]
table_time = []

@load "scripts/ml-32m.jld2" S
if bool_part
    m = 2000
    n = 1000
    T = S[1:n, 1:m]
else
end
S, cols, rows = ExchangeMarket.drop_empty(T)
n, m = size(S)
# S = Matrix(S)
S = S .* 2.0
ϵₚ = 1e-6


for ρ in rrange
    # ρ = 0.6
    f0 = FisherMarket(m, n; c=S, ρ=ρ, bool_unit=true, bool_unit_wealth=true, scale=0.1, bool_ensure_nz=true)
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
    # -----------------------------------------------------------------------
    # how to do this?
    # -----------------------------------------------------------------------
    # Ω = 10
    # from kmeans
    # cluster_map, cardinality, r_kmeans, Ξ = cluster_kmeans(f1.c; k=Cc)
    # ω = (f1.c .> 1e-5) .|> Int |> sparse
    # naive clique extraction
    # cluster_map, cardinality, cliques, G = cliques_from_fillin(ω; k=Ω)
    # update_cluster_map!(alg, cluster_map, cardinality; centers=Dict(i => Ξ[:, i] for i in 1:Cc))
    # -----------------------------------------------------------------------

    traj = opt!(
        alg, f1;
        # loginterval=1,
        keep_traj=true
    )
    pₛ = copy(alg.p)
    results_phi[ρ] = pₛ
    k_max = 100
    for (k, t) in enumerate(traj)
        t.D = norm(t.p - pₛ)
        if t.D < ϵₚ
            k_max = k
            break
        end
    end
    # push!(results, ((name, ρ), (alg, traj[1:k_max], f1)))
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
            tol_p=ϵₚ,
        )
        push!(table_time, (n, m, name, ρ, traj[end].t))
        push!(results, ((name, ρ), (alg, traj, f1)))
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
        )
        if attr == :k
            plot!(
                fig,
                xticks=[10, 20, 50, 100, 200, 500]
            )
        end
        for ((mm, _ρ), (alg, traj, f1)) in results
            if _ρ != ρ
                continue
            end
            traj_pp₊ = map(pp -> pp.D, traj)
            traj_tt₊ = map(pp -> getfield(pp, attr), traj)
            @info "" traj[end].t
            @info "" traj_pp₊
            if mm ∈ [:Tât]
                plot!(
                    fig, traj_tt₊[1:2:end], traj_pp₊[1:2:end],
                    label=L"\texttt{%$mm}", linewidth=2,
                    linestyle=:dash,
                    markershape=marker_style[mm],
                    color=colors[mm]
                )
            elseif mm ∈ [:PropRes]
                plot!(
                    fig, traj_tt₊[1:2:end], traj_pp₊[1:2:end],
                    label=L"\texttt{%$mm}", linewidth=2,
                    linestyle=:dash,
                    markershape=marker_style[mm],
                    color=colors[mm]
                )
            else
                plot!(
                    fig, traj_tt₊[1:2:end], traj_pp₊[1:2:end],
                    label=L"\texttt{%$mm}", linewidth=2,
                    linestyle=:dash,
                    markershape=marker_style[mm],
                    color=colors[mm]
                )
            end
        end
        savefig(fig, "$(RESULTSDIR)/traj-mov-$(ρfmt)_$(attr).pdf")
        savefig(fig, "$(RESULTSDIR)/traj-mov-$(ρfmt)_$(attr).tex")
        println("saved to $(RESULTSDIR)/traj-mov-$(ρfmt)_$(attr).tex")
    end
end

using DataFrames
df = DataFrame(table_time)
rename!(df, [:n, :m, :method, :ρ, :t])
df



