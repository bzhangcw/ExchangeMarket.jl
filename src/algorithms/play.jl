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

