# -----------------------------------------------------------------------
# Sparse utilities for per-agent column access
#   SparseColRef: zero-copy view into a CSC column
#   @agent macro: syntactic sugar for per-agent data access
# -----------------------------------------------------------------------
using SparseArrays

# -----------------------------------------------------------------------
# SparseColRef: zero-copy reference to column i of a SparseMatrixCSC
# -----------------------------------------------------------------------
"""
    SparseColRef{Tv,Ti} <: AbstractVector{Tv}

Zero-copy view into column `i` of a `SparseMatrixCSC`.
Stores SubArray views into the parent's `rowval` and `nzval` arrays —
no allocation beyond the struct itself.

Construct via `sparse_col_ref(A, i)`.
"""
struct SparseColRef{Tv,Ti} <: AbstractVector{Tv}
    n::Int                           # full column length
    nzind::SubArray{Ti,1,Vector{Ti},Tuple{UnitRange{Int}},true}
    nzval::SubArray{Tv,1,Vector{Tv},Tuple{UnitRange{Int}},true}
end

Base.size(s::SparseColRef) = (s.n,)
Base.length(s::SparseColRef) = s.n

SparseArrays.nnz(s::SparseColRef) = length(s.nzind)
SparseArrays.nonzeroinds(s::SparseColRef) = s.nzind
SparseArrays.nonzeros(s::SparseColRef) = s.nzval

function Base.getindex(s::SparseColRef{Tv}, j::Int) where {Tv}
    @boundscheck 1 <= j <= s.n || throw(BoundsError(s, j))
    idx = searchsortedfirst(s.nzind, j)
    (idx <= length(s.nzind) && s.nzind[idx] == j) ? s.nzval[idx] : zero(Tv)
end

"""
    sparse_col_ref(A::SparseMatrixCSC, i::Int) -> SparseColRef

Zero-copy column reference into column `i` of CSC matrix `A`.
"""
@inline function sparse_col_ref(A::SparseMatrixCSC, i::Int)
    rng = nzrange(A, i)
    SparseColRef(size(A, 1), view(rowvals(A), rng), view(nonzeros(A), rng))
end

"""
    sparse_col_ref(A::Matrix, i::Int) -> SubArray

Dense fallback: returns `view(A, :, i)`.
"""
@inline sparse_col_ref(A::Matrix, i::Int) = view(A, :, i)

# -----------------------------------------------------------------------
# @agent macro: zero-cost per-agent data access
# -----------------------------------------------------------------------
"""
    @agent market i field

Zero-copy access to agent `i`'s data from `market`.
Column fields (`:c`, `:x`, `:s`, `:g`) return sparse-aware views.
Scalar fields (`:w`, `:ρ`, `:σ`, `:ε`) return the value directly.

# Examples
```julia
cᵢ = @agent market i c   # SparseColRef or SubArray — no allocation
wᵢ = @agent market i w   # scalar
xᵢ = @agent market i x   # writable SubArray view
```
"""
macro agent(market, i, field)
    quote
        _agent_field($(esc(market)), $(esc(i)), Val($(QuoteNode(field))))
    end
end

# Column fields (sparse-aware, zero-copy)
@inline _agent_field(m, i, ::Val{:c}) = sparse_col_ref(m.c, i)
@inline _agent_field(m, i, ::Val{:x}) = view(m.x, :, i)
@inline _agent_field(m, i, ::Val{:s}) = view(m.s, :, i)
@inline _agent_field(m, i, ::Val{:g}) = view(m.g, :, i)

# Scalar fields
@inline _agent_field(m, i, ::Val{:w}) = m.w[i]
@inline _agent_field(m, i, ::Val{:ρ}) = m.ρ[i]
@inline _agent_field(m, i, ::Val{:σ}) = m.σ[i]
@inline _agent_field(m, i, ::Val{:ε}) = m.ε_br_play[i]

# -----------------------------------------------------------------------
# Sparse iteration utilities
# -----------------------------------------------------------------------
"""
    foreach_nz(f, c::SparseColRef)

Call `f(j, cj)` for each nonzero entry `(j, cj)` in the sparse column.
For dense vectors, iterates all entries where `cj != 0`.
"""
@inline function foreach_nz(f, c::SparseColRef)
    @inbounds for k in eachindex(c.nzind)
        f(c.nzind[k], c.nzval[k])
    end
end

@inline function foreach_nz(f, c::AbstractVector)
    @inbounds for j in eachindex(c)
        cj = c[j]
        cj != 0 && f(j, cj)
    end
end

"""
    sparse_reduce(f, c::SparseColRef; init=0.0)

Accumulate `f(j, cj)` over nonzero entries. Like `mapreduce` for sparse columns.

# Example
    denom = sparse_reduce(c) do j, cj
        cj^(1+σ) * p[j]^(-σ)
    end
"""
@inline function sparse_reduce(f, c::SparseColRef; init=0.0)
    s = init
    @inbounds for k in eachindex(c.nzind)
        s += f(c.nzind[k], c.nzval[k])
    end
    return s
end

@inline function sparse_reduce(f, c::AbstractVector; init=0.0)
    s = init
    @inbounds for j in eachindex(c)
        c[j] != 0 && (s += f(j, c[j]))
    end
    return s
end

"""
    sparse_scatter!(f, x, c::SparseColRef)

Set `x[j] = f(j, cj)` at nonzero positions of `c`. Does NOT zero `x` first.

# Example
    x .= 0.0
    sparse_scatter!(x, c) do j, cj
        coeff * cj^(1+σ) * p[j]^(-σ-1)
    end
"""
@inline function sparse_scatter!(f, x, c::SparseColRef)
    @inbounds for k in eachindex(c.nzind)
        j = c.nzind[k]
        x[j] = f(j, c.nzval[k])
    end
    return x
end

@inline function sparse_scatter!(f, x, c::AbstractVector)
    @inbounds for j in eachindex(c)
        c[j] != 0 && (x[j] = f(j, c[j]))
    end
    return x
end

"""
    sparse_argmax(f, c::SparseColRef) -> Int

Return index `j` that maximizes `f(j, cj)` over nonzero entries.

# Example
    j₊ = sparse_argmax(c) do j, cj
        cj / p[j]
    end
"""
@inline function sparse_argmax(f, c::SparseColRef)
    j_best = c.nzind[1]
    v_best = f(j_best, c.nzval[1])
    @inbounds for k in 2:length(c.nzind)
        j = c.nzind[k]
        v = f(j, c.nzval[k])
        if v > v_best
            v_best = v
            j_best = j
        end
    end
    return j_best
end

@inline function sparse_argmax(f, c::AbstractVector)
    j_best = 1
    v_best = f(1, c[1])
    @inbounds for j in 2:length(c)
        c[j] == 0 && continue
        v = f(j, c[j])
        if v > v_best
            v_best = v
            j_best = j
        end
    end
    return j_best
end

"""
    sparse_dot(c::SparseColRef, v::AbstractVector)

Dot product iterating only over nonzero entries of `c`.
O(nnz) instead of O(n).
"""
@inline function sparse_dot(c::SparseColRef, v::AbstractVector)
    s = 0.0
    @inbounds for k in eachindex(c.nzind)
        s += c.nzval[k] * v[c.nzind[k]]
    end
    return s
end

@inline sparse_dot(c::AbstractVector, v::AbstractVector) = dot(c, v)

"""
    sparse_div_max(c::SparseColRef, p::AbstractVector)

Compute `maximum(cⱼ / pⱼ)` over nonzero entries of `c`.
"""
@inline function sparse_div_max(c::SparseColRef, p::AbstractVector)
    mx = -Inf
    @inbounds for k in eachindex(c.nzind)
        j = c.nzind[k]
        mx = max(mx, c.nzval[k] / p[j])
    end
    return mx
end

@inline sparse_div_max(c::AbstractVector, p::AbstractVector) = maximum(c ./ p)

"""
    sparse_outer_subtract!(S::Matrix, v::SparseColRef, α::Float64)

`S[j,l] -= α * v[j] * v[l]` for all nonzero positions of `v`.
Cost: O(nnz²) instead of O(n²).
"""
@inline function sparse_outer_subtract!(S::Matrix, v::SparseColRef, α::Float64)
    @inbounds for kl in eachindex(v.nzind)
        l = v.nzind[kl]
        vl = v.nzval[kl]
        for kj in eachindex(v.nzind)
            j = v.nzind[kj]
            S[j, l] -= α * v.nzval[kj] * vl
        end
    end
end

@inline function sparse_outer_subtract!(S::Matrix, v::AbstractVector, α::Float64)
    n = length(v)
    @inbounds for l in 1:n, j in 1:n
        S[j, l] -= α * v[j] * v[l]
    end
end
