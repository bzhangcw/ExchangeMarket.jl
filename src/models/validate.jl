
@doc """
    __validate(fisher::Market)
    -----------------------------------------------------------------------
    validate the equilibrium of the Market.
    use the price attached in the FisherMarket if no alg is provided.
    inner use only.
"""
function __validate(market::Market)
    validate(market, nothing)
end

function validate(market::Market, alg)
    m = market.m
    n = market.n
    u = market.u
    x = market.x
    p = isnothing(alg) ? market.p : alg.p
    μ = isnothing(alg) ? 0.0 : alg.μ
    w = market.w
    if isa(market, ArrowDebreuMarket)
        @printf(" :current budget correctly updated? [%.4e]\n", norm(w - market.b' * p))
    end

    market.df = df = DataFrame(
        :utility => market.val_u,
        :left_budget => w - x' * p,
    )
    println(__default_sep)
    @printf(" :problem size\n")

    @printf(" :    number of agents: %d\n", market.m)
    @printf(" :    number of goods: %d\n", market.n)
    @printf(" :    avg number of nonzero entries in c: %.4f\n",
        length(sparse(market.c).nzval) / (market.m * market.n)
    )
    @printf(" :equilibrium information\n")
    @printf(" :method: %s\n", alg.name)
    println(__default_sep)
    println(first(df, 10))
    println(__default_sep)
    _excess = (sum(market.x; dims=2)[:] - market.q) ./ maximum(market.q)
    @printf(" :(normalized) market excess: [%.4e, %.4e]\n", minimum(_excess), maximum(_excess))
    @printf(" :            social welfare:  %.8e\n", (log.(market.val_u))' * market.w)
    println(__default_sep)
end
