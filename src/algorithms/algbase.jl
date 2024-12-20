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
end