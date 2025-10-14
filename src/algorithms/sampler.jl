using Random
abstract type Sampler end

const rng = MersenneTwister(19931017)

mutable struct NullSampler <: Sampler
    indices::Vector{Int}
    batchsize::Int
    NullSampler(;) = begin
        this = new()
        return this
    end
end

mutable struct BatchSampler <: Sampler
    batchsize::Int
    indices::Vector{Int}
    BatchSampler(; batchsize::Int) = begin
        this = new()
        this.batchsize = batchsize
        this.indices = Vector{Int}(undef, batchsize)
        return this
    end
end


function sample!(sampler::NullSampler, market::FisherMarket)
    if !isdefined(sampler, :index_set)
        sampler.indices = 1:market.m
        sampler.batchsize = market.m
    end
end

function sample!(sampler::BatchSampler, market::FisherMarket)
    sampler.indices .= shuffle(rng, 1:market.m)[1:sampler.batchsize]
end

# Arrow–Debreu overloads
function sample!(sampler::NullSampler, ad::ArrowDebreuMarket)
    if !isdefined(sampler, :index_set)
        sampler.indices = 1:ad.m
        sampler.batchsize = ad.m
    end
end

function sample!(sampler::BatchSampler, ad::ArrowDebreuMarket)
    sampler.indices .= shuffle(rng, 1:ad.m)[1:sampler.batchsize]
end
