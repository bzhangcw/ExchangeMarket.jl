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
    sample!(alg.sampler, market)

    Threads.@threads for i in (all ? (1:market.m) : alg.sampler.indices)
        solve_substep!(alg, market, i; ϵᵢ=ϵᵢ)
    end
    timed && (alg.tₗ += time() - _ts)
    verbose && validate(market, alg.μ)
    market.sumx .= sum(market.x[:, alg.sampler.indices]; dims=2)[:]
end


function solve_substep!(
    alg::Algorithm, market::Market, i::Int;
    kwargs...
)
    return solve!(
        alg.optimizer;
        market=market,
        i=i,
        p=alg.p,
        μ=alg.μ,
        kwargs...
    )
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