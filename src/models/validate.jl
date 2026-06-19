
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
    # Weighted Nash social welfare Σ_i w_i log u_i. The stored CES level u_i
    # = (Σ_j c_ij x_ij^{ρ})^{1/ρ} overflows to Inf for ρ ≈ 0, so score CES
    # agents through the stable `logutility` (log u = log(s)/ρ); other agent
    # types keep using the stored level via `slog`.
    _welfare = 0.0
    for i in 1:m
        a = market.agents[i]
        _welfare += w[i] * (a.atype isa CESAgent ? logutility(a) : slog(market.val_u[i]))
    end
    @printf(" :            social welfare:  %.8e\n", _welfare)
    println(__default_sep)
end
