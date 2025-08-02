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
nrange = [1000, 2000, 5000]
mrange = [10000, 50000, 100000]
maxtime = 200


open("$(RESULTSDIR)/table_time.csv", "w") do io
    println(io, "n,m,ρ,method,t,eps")
end
@info """
    Results table_time.csv is initialized 
        and will be saving to $(RESULTSDIR)/table_time.csv
"""

method_filter(name) = name ∈ [:LogBarPCG]
# method_filter(name) = name ∈ [:LogBar, :LogBarPCG, :Tât, :PropRes]
table_time = []
for n in nrange
    for m in mrange
        for ρ in rrange
            f0 = FisherMarket(m, n; ρ=ρ, bool_unit=true, scale=30.0, sparsity=0.2)
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
                    maxtime=maxtime
                )
                push!(table_time, (n, m, ρ, name, traj[end].t, traj[end].D))
                # write to a CSV file 
                open("$(RESULTSDIR)/table_time.csv", "a") do io
                    println(io, "$(n),$(m),$(ρ),$(name)," * @sprintf("%.2f", traj[end].t) * @sprintf(",%.2e", traj[end].D))
                end
            end
        end
    end
end

using DataFrames
df = DataFrame(table_time)
rename!(df, [:n, :m, :ρ, :method, :t, :ϵ])
df[!, :t] = map(x -> @sprintf("%.1f", x), df[!, :t])
df[!, :ϵ] = map(x -> @sprintf("%.2e", x), df[!, :ϵ])
show(df)



