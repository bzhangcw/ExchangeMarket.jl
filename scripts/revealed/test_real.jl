# Compare CG / Multicut / LSQ on a real market (CES or PLC).
# Separate training and test revealed-preference sets of the same size K,
# track train primal objective and test error per iteration, then plot.

using Revise
using Random, SparseArrays, LinearAlgebra
using JuMP, MosekTools
using Plots, LaTeXStrings, Printf
import MathOptInterface as MOI

using ExchangeMarket

include("../tools.jl")
include("../plots.jl")
include("./plc.jl")
include("./setup.jl")

switch_to_pdf(; bool_use_html=false)

bool_init = true
bool_run = true
bool_plot = true

# -----------------------------------------------------------------------
# market type: :ces or :plc
# -----------------------------------------------------------------------
# market_type = :plc
market_type = :ces

# -----------------------------------------------------------------------
# problem size
# -----------------------------------------------------------------------
if bool_init
    n = 20
    m = 50
    K = 100        # training and test set each contain K observations
    seed = 42

    # PLC-specific options
    opt_plc = (L=5, intercept=false)

    # filter the methods you want to run
    method_filter(name) = name ∈ [:CG, :Multicut]

    # -----------------------------------------------------------------------
    # build the underlying (real) market
    # -----------------------------------------------------------------------
    Random.seed!(seed)

    if market_type === :ces
        ρ_vec = -3.5 .+ 4.3 .* rand(m)   # heterogeneous ρ
        f0 = FisherMarket(m, n; ρ=ρ_vec, scale=30.0, sparsity=0.99)
        linconstr = LinearConstr(1, n, ones(1, n), [1.0])
        f1 = copy(f0)
        p₀ = ones(n) ./ n
        f1.x .= ones(n, m) ./ m
        alg = HessianBar(n, m, p₀; linconstr=linconstr)
        alg.linsys = :direct

        @info "Ground-truth CES market" n m K σ_range = extrema(f1.σ)

        Ξ_train = produce_revealed_preferences(alg, f1, K; seed=seed)
        Ξ_test = produce_revealed_preferences(alg, f1, K; seed=seed + 1)

    elseif market_type === :plc
        L = opt_plc.L
        plc_agents = [random_plc_agent(n, L; intercept=opt_plc.intercept) for _ in 1:m]
        w_vec = rand(m)
        w_vec ./= sum(w_vec)

        @info "Ground-truth PLC market" n m L K intercept = opt_plc.intercept

        @info "producing training set..."
        Ξ_train = produce_revealed_preferences_plc(plc_agents, w_vec, K, n; seed=seed)
        @info "producing test set..."
        Ξ_test = produce_revealed_preferences_plc(plc_agents, w_vec, K, n; seed=seed + 1)

    else
        error("Unknown market_type: $market_type")
    end

    tag = String(market_type)
end

if bool_run
    # -----------------------------------------------------------------------
    # run methods
    # -----------------------------------------------------------------------
    results = []
    for (name, pricing_kind, kwargs) in method_kwargs
        !method_filter(name) && continue
        Random.seed!(seed)
        @info "running $name"
        t_elapsed = @elapsed begin
            fa, γ_ref, hist = run_method_tracked(
                name, pricing_kind, kwargs, Ξ_train, Ξ_test; verbose=true
            )
        end
        push!(results, (name, fa, hist, t_elapsed))
        @info "$name done" iters = length(hist[:primal_obj]) final_train = hist[:primal_obj][end] final_test = hist[:test_err][end] time_s = t_elapsed
    end

    # -----------------------------------------------------------------------
    # summary table
    # -----------------------------------------------------------------------
    println("\n=== Summary ===")
    @printf("%-12s %8s %12s %12s %10s\n", "method", "iters", "train_obj", "test_err", "time (s)")
    for (name, fa, hist, t) in results
        @printf("%-12s %8d %12.6f %12.6f %10.3f\n",
            String(name), length(hist[:primal_obj]),
            hist[:primal_obj][end], hist[:test_err][end], t)
    end
end

# -----------------------------------------------------------------------
# plots: training and test trajectories
# -----------------------------------------------------------------------
function make_figure(ylabel_str, title_str)
    fig = generate_empty(; shape=:wide)
    plot!(fig,
        ylabel=ylabel_str,
        xlabel=L"\textrm{iteration}",
        title=title_str,
        legendbackgroundcolor=RGBA(1.0, 1.0, 1.0, 0.8),
        yticks=10.0 .^ (-10:1:3),
        xtickfont=font(18),
        ytickfont=font(18),
        xscale=:identity,
        size=(600, 500),
        legendfontsize=18,
    )
    return fig
end
if bool_plot
    if market_type === :ces
        title_train = L"Training error ($n=%$n,~m=%$m,~K=%$K$)"
        title_test = L"Test error ($n=%$n,~m=%$m,~K=%$K$)"
    else
        title_train = L"Training error ($n=%$n,~m=%$m,~L=%$L,~K=%$K$)"
        title_test = L"Test error ($n=%$n,~m=%$m,~L=%$L,~K=%$K$)"
    end

    fig_train = make_figure(L"\textrm{value}", title_train)
    fig_test = make_figure(L"\textrm{value}", title_test)

    for (name, fa, hist, _) in results
        iters = 1:length(hist[:primal_obj])
        c = get(colors, name, 1)
        mk = get(marker_style, name, :circle)
        m_end = hist[:num_agents][end]
        lbl = L"($T=%$m_end$)~\texttt{%$name}"
        plot!(fig_train, iters, hist[:primal_obj];
            label=lbl, linewidth=3,
            markershape=mk, color=c)
        plot!(fig_test, iters, hist[:test_err];
            label=lbl, linewidth=3,
            markershape=mk, color=c)
    end

    fig_combined = plot(fig_train, fig_test; layout=(1, 2), size=(2200, 600))

    savefig(fig_train, joinpath(@__DIR__, "real_$(tag)_train.pdf"))
    savefig(fig_test, joinpath(@__DIR__, "real_$(tag)_test.pdf"))
    savefig(fig_combined, joinpath(@__DIR__, "real_$(tag)_train_test.pdf"))

    @info "saved plots" joinpath(@__DIR__, "real_$(tag)_train.pdf") joinpath(@__DIR__, "real_$(tag)_test.pdf")
end
