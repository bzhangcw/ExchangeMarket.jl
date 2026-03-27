# -----------------------------------------------------------------------
# subproblems
# -----------------------------------------------------------------------
@doc raw"""
    play! runs the subproblems as best-response-type mappings
    for all i ∈ I
        solve_substep!(alg, market, i; ϵᵢ=ϵᵢ)
    end
    ϵᵢ: the tolerance for the subproblem
"""
function play!(
    alg::Algorithm, market::Market;
    verbose=false,
    ϵᵢ=1e-7,
    all=false,
    timed=true,
)
    _ts = time()
    ws = market.workspace

    if ws !== nothing
        # Batched path: GPU or CPU dense — all agents via matrix broadcasts
        sample!(alg.sampler, market)  # set batchsize for grad!
        σ_scalar = market.σ[1]  # assumes uniform ρ
        _play_batched!(ws, alg.p, σ_scalar)
    else
        # Per-agent path: CPU — AgentView + SparseColRef + @threads
        sample!(alg.sampler, market)
        if isempty(market.agents)
            init_agents!(market)
        end
        indices = all ? (1:market.m) : alg.sampler.indices
        Threads.@threads for i in indices
            solve_substep!(alg, market.agents[i], market; ϵᵢ=ϵᵢ)
        end
        market.sumx .= sum(market.x[:, alg.sampler.indices]; dims=2)[:]
    end

    timed && (alg.tₗ += time() - _ts)
    verbose && validate(market, alg.μ)
end


"""
    solve_substep!(alg, agent::AgentView, market; kwargs...)

Solve agent's best-response subproblem using the algorithm's optimizer.
The `agent` provides zero-copy access to per-agent data; `market` is
passed for scalar writes (val_u) and mutable parameters (w, ε).
"""
function solve_substep!(
    alg::Algorithm, agent::AgentView, market::Market;
    kwargs...
)
    return solve!(
        alg.optimizer;
        agent=agent,
        market=market,
        i=agent.i,
        p=alg.p,
        μ=alg.μ,
        kwargs...
    )
end

# backward-compat: old signature creates AgentView on the fly
function solve_substep!(
    alg::Algorithm, market::Market, i::Int;
    kwargs...
)
    if isempty(market.agents)
        init_agents!(market)
    end
    return solve_substep!(alg, market.agents[i], market; kwargs...)
end


@doc raw"""
    check_pareto(alg, market, optimizer)

Check Pareto optimality by temporarily swapping in `optimizer`,
running `play!` with `μ=0`, and comparing utilities.

Returns `‖u₊ - u₀‖∞` (the utility gap).
"""
function check_pareto(
    alg::Algorithm, market::FisherMarket, optimizer;
)
    u₀ = copy(market.val_u)
    # swap optimizer, zero barrier, run play!
    opt₀ = alg.optimizer
    μ₀ = alg.μ
    alg.optimizer = optimizer
    alg.μ = 0.0
    play!(alg, market; all=true, timed=false)
    alg.optimizer = opt₀
    alg.μ = μ₀
    return norm(market.val_u .- u₀, Inf)
end