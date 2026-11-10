#!/usr/bin/env julia
# Regenerate a comparison figure and median summary from saved measurements.
ENV["GKSwstype"]="100"
using CSV, DataFrames, Statistics, Plots
directory=isempty(ARGS) ? joinpath(@__DIR__,"jd_gpu_results") : abspath(only(ARGS))
runs=CSV.read(joinpath(directory,"runs.csv"),DataFrame)
summary=combine(groupby(runs,[:users,:goods,:nnz,:method,:criterion,:tol]),
    :target_met=>all=>:all_runs_met_target,
    :wall_seconds=>median=>:median_wall_seconds,
    :updates=>median=>:median_updates,
    :paper_error=>maximum=>:worst_paper_error,
    :shared_error=>maximum=>:worst_shared_error,
    :total_spending=>median=>:median_total_spending,
    :total_budget=>median=>:total_budget)
sort!(summary,[:users,:method]);CSV.write(joinpath(directory,"summary.csv"),summary)
timing=plot(xlabel="Users",ylabel="Solve time (seconds)",xscale=:log10,yscale=:log10,
    title="JD: raw clicks, total budget = 1",legend=:topleft)
iterations=plot(xlabel="Users",ylabel="Updates + 1",xscale=:log10,yscale=:log10,
    title="Stopping rule: $(first(summary.criterion)), tol=$(first(summary.tol))",legend=false)
for (method,color) in zip(("PR","Perturbed","PCG"),(:steelblue,:darkorange,:seagreen))
    rows=sort(filter(r->r.method==method,summary),:users)
    isempty(rows) && continue
    good=rows.all_runs_met_target
    for (panel,values) in ((timing,rows.median_wall_seconds),(iterations,rows.median_updates.+1))
        plot!(panel,rows.users,ifelse.(good,values,NaN);label=method,color,marker=:circle)
        bad=.!good .& isfinite.(values) .& (values.>0)
        any(bad) && scatter!(panel,rows.users[bad],values[bad];label="$(method): target missed",color,
            marker=:xcross,markersize=7)
    end
end
figure=plot(timing,iterations;layout=(1,2),size=(1200,440),margin=5Plots.mm)
savefig(figure,joinpath(directory,"scaling.png"))
savefig(figure,joinpath(directory,"scaling.pdf"))
println("Saved summary.csv, scaling.png and scaling.pdf in ",directory)
