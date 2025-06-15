

Base.@kwdef mutable struct LinearConstr{T}
    m::Int
    n::Int
    A::Union{SparseMatrixCSC{T,Int},Matrix{T}}
    b::Vector{T}
    eps::Vector{T}
    function LinearConstr(
        m::Int, n::Int,
        A::Union{SparseMatrixCSC{T,Int},Matrix{T}},
        b::Vector{T}
    ) where {T}
        this = new{T}()
        this.n = n
        this.m = m
        @assert m <= n
        @assert size(A) == (m, n)
        @assert length(b) == m
        this.A = copy(A)
        this.b = copy(b)
        this.eps = zeros(m)
        return this
    end
end

function __evalconstr!(alg, constr::LinearConstr)
    constr.eps .= constr.A * alg.p .- constr.b
end