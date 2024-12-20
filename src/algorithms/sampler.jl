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


function sample!(sampler::NullSampler, fisher::FisherMarket)
    if !isdefined(sampler, :index_set)
        sampler.indices = 1:fisher.m
        sampler.batchsize = fisher.m
    end
end

function sample!(sampler::BatchSampler, fisher::FisherMarket)
    sampler.indices .= shuffle(rng, 1:fisher.m)[1:sampler.batchsize]
end
