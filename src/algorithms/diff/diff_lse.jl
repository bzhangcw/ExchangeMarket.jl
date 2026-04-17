# -----------------------------------------------------------------------
# LSE (logsumexp-smoothed) helpers and diff functions
# -----------------------------------------------------------------------
__lse_logsumexp(z) = (zmax = maximum(z); zmax + log(sum(exp.(z .- zmax))))
__lse_softmax(z) = (ez = exp.(z .- maximum(z)); ez ./ sum(ez))

@doc raw"""
    __lse_grad!(alg, market)

Gradient of the LSE dual: ∇φ_j = q_j - Σ_i (w_i/L_i) γ_{ij} c_{ij} / p_j².
"""
function __lse_grad!(alg, market::Market)
    m = market.m
    t = market.ε_br_play[1]
    p = alg.p
    c = market.c
    w = market.w
    alg.∇ .= market.q .* (alg.sampler.batchsize / m)
    for i in alg.sampler.indices
        cᵢ = @view c[:, i]
        z = cᵢ ./ p
        Li = t * __lse_logsumexp(z ./ t)
        γ = __lse_softmax(z ./ t)
        alg.∇ .-= (w[i] / Li) .* γ .* cᵢ ./ (p .^ 2)
    end
end

@doc raw"""
    __lse_eval!(alg, market)

LSE dual value: φ = ⟨p, q⟩ + Σ_i w_i log(w_i L_i), where L_i = t·LSE(c_i/(tp)).
"""
function __lse_eval!(alg, market::Market)
    m = market.m
    t = market.ε_br_play[1]
    p = alg.p
    c = market.c
    w = market.w
    alg.φ = dot(p, market.q)
    for i in 1:m
        cᵢ = @view c[:, i]
        z = cᵢ ./ p
        Li = t * __lse_logsumexp(z ./ t)
        alg.φ += w[i] * log(w[i] * Li)
    end
    alg.φ = min(alg.φ, 1e8)
end

@doc raw"""
    __lse_compute_exact_hess!(alg, market)

Compute the plain Hessian ∇²φ for the LSE dual and store in `alg.H`.

First builds the affine-scaled Hessian H̃ = P∇²φP using:
- Diagonal: w_i e_i ⊙ (z/t + 2),  where e_i = γ_i ⊙ c_i/(L_i p)
- Rank-1:  -w_i(1 + L_i/t) e_i e_i'

Then converts to plain Hessian: ∇²φ = P⁻¹ H̃ P⁻¹.
The existing `__direct_afsc!` applies the P·H·P scaling.
"""
function __lse_compute_exact_hess!(alg, market::FisherMarket)
    if isa(alg.H, SparseMatrixCSC)
        alg.H = Matrix(alg.H)
    end
    alg.H .= 0.0
    n, m = size(market.x)
    p = alg.p
    t = market.ε_br_play[1]
    c = market.c
    w = market.w

    for i in 1:m
        cᵢ = @view c[:, i]
        z = cᵢ ./ p
        Li = t * __lse_logsumexp(z ./ t)
        γ = __lse_softmax(z ./ t)
        dᵢ = cᵢ ./ (Li .* p)
        eᵢ = γ .* dᵢ
        # affine-scaled diagonal: w_i e_i ⊙ (z/t + 2)
        alg.H[diagind(alg.H)] .+= w[i] .* eᵢ .* (z ./ t .+ 2.0)
        # affine-scaled rank-1: -w_i(1 + L_i/t) e_i e_i'
        βᵢ = w[i] * (1.0 + Li / t)
        LinearAlgebra.BLAS.syr!('U', -βᵢ, eᵢ, alg.H)
    end
    # mirror upper → lower
    @inbounds for j in 1:n-1, k in j+1:n
        alg.H[k, j] = alg.H[j, k]
    end
    # convert affine-scaled H̃ = P∇²φP to plain ∇²φ
    alg.H .= alg.H ./ (p * p')

    println("LSE exact dense Hessian built")
    return nothing
end

@doc raw"""
    __lse_hessop_afscale!(buf, alg, market, v; add_μ=false)

Matrix-free Hessian-vector product `buf = H̃ v` where `H̃ = P∇²φP + μI`
(affine-scaled Hessian for the LSE dual).

Per buyer `i`:
- `eᵢ = γᵢ ⊙ cᵢ/(Lᵢ p)`
- Diagonal: `buf += wᵢ eᵢ ⊙ (z/t + 2) ⊙ v`
- Rank-1:   `buf -= wᵢ(1 + Lᵢ/t) ⟨eᵢ,v⟩ eᵢ`
- After loop: `buf += μ v`
"""
function __lse_hessop_afscale!(buf, alg, market, v; add_μ=false)
    p = alg.p
    t = market.ε_br_play[1]
    c = market.c
    w = market.w
    n, m = size(c)
    buf .= 0.0
    for i in 1:m
        cᵢ = @view c[:, i]
        z = cᵢ ./ p
        Li = t * __lse_logsumexp(z ./ t)
        γ = __lse_softmax(z ./ t)
        eᵢ = γ .* cᵢ ./ (Li .* p)
        # diagonal contribution
        buf .+= w[i] .* eᵢ .* (z ./ t .+ 2.0) .* v
        # rank-1 contribution
        βᵢ = w[i] * (1.0 + Li / t)
        buf .-= βᵢ * dot(eᵢ, v) .* eᵢ
    end
    if add_μ
        buf .+= alg.μ .* v
    end
    return buf
end

@doc raw"""
    __lse_precond_data(alg, market)

Pre-compute all data needed for the Neumann preconditioner of the
affine-scaled Hessian `H̃ = P∇²φP`.

Returns a `NamedTuple` with:
- `diag_H`: vector `Λ - Σ` (exact diagonal of `H̃`, length `n`)
- `Σ`: vector `Σᵢ βᵢ eᵢ⊙eᵢ` (length `n`)
- `matE`: `n × m` matrix whose columns are the `eᵢ` vectors
- `β`: `m`-vector of `βᵢ = wᵢ(1 + Lᵢ/t)`
"""
function __lse_precond_data(alg, market)
    p = alg.p
    t = market.ε_br_play[1]
    c = market.c
    w = market.w
    n, m = size(c)
    Λ = zeros(n)
    Σ = zeros(n)
    matE = zeros(n, m)
    β = zeros(m)
    for i in 1:m
        cᵢ = @view c[:, i]
        z = cᵢ ./ p
        Li = t * __lse_logsumexp(z ./ t)
        γ = __lse_softmax(z ./ t)
        eᵢ = γ .* cᵢ ./ (Li .* p)
        matE[:, i] .= eᵢ
        βᵢ = w[i] * (1.0 + Li / t)
        β[i] = βᵢ
        Λ .+= w[i] .* eᵢ .* (z ./ t .+ 2.0)
        Σ .+= βᵢ .* eᵢ .^ 2
    end
    return (; diag_H=Λ .- Σ, Σ, matE, β)
end

__lse_hess_diag_precond(alg, market) = __lse_precond_data(alg, market).diag_H

@doc raw"""
    __lse_offdiag_matvec!(buf, matE, β, Σ, v)

Matrix-free application of the off-diagonal residual:
`buf = R_off v = Σᵢ βᵢ ⟨eᵢ, v⟩ eᵢ - Σ ⊙ v`

Uses pre-computed `matE` (n×m), `β` (m), `Σ` (n).
"""
function __lse_offdiag_matvec!(buf, matE, β, Σ, v)
    # buf = matE * (β .* (matE' * v)) - Σ .* v
    mul!(buf, matE, β .* (matE' * v))
    buf .-= Σ .* v
    return buf
end
