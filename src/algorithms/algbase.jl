abstract type Algorithm end


Base.@kwdef mutable struct StateInfo{T}
    # iteration counters
    k::Int = 0
    # current price at k
    p::Vector{T}
    # current gradient at k
    ∇::Vector{T}
    # norm of scaled gradient
    gₙ::T
    # size of step
    dₙ::T
    # dual function value
    φ::T
    # time
    t::T
end

function compute_stop(k::Int, alg::Algorithm, fisher::FisherMarket)
    if alg.optimizer.style ∈ (:analytic, :linconic)
        return (alg.gₙ < alg.tol) || (alg.dₙ < alg.tol) || (alg.t >= alg.maxtime) || (k >= alg.maxiter)
    elseif alg.optimizer.style == :bids
        # this cannot ensure the subproblem is optimal for each player
        return (alg.dₙ / maximum(alg.p) < (alg.tol)) || (alg.t >= alg.maxtime) || (k >= alg.maxiter)
    end
end