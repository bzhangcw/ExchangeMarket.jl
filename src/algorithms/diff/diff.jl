# -----------------------------------------------------------------------
# differentiation utilities
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
using LinearAlgebra

# -----------------------------------------------------------------------
# main functions
# -----------------------------------------------------------------------
function grad!(alg, market::Market)
    if is_linear_market(market)
        __linear_grad!(alg, market)
    else
        __ces_grad!(alg, market)
    end
end

function hess!(alg, market::Market; bool_dbg=false)
    if is_linear_market(market)
        __linear_hess!(alg, market; bool_dbg=bool_dbg)
        # __linear_hess_fromx!(alg, market)
    else
        __ces_hess!(alg, market)
    end
end

function eval!(alg, market::Market)
    if is_linear_market(market)
        __linear_eval!(alg, market)
    else
        __ces_eval!(alg, market)
    end
end

# -----------------------------------------------------------------------
# linear case
# -----------------------------------------------------------------------
function __linear_grad!(alg, market::Market; bool_dbg=false)
    if alg.optimizer.style ∈ (:analytic, :dual_lp, :dual_lp_conic)
        alg.∇ .= market.q .* (alg.sampler.batchsize / market.m) - market.sumx
    elseif alg.optimizer.style ∈ (:approx_lin,)
        alg.∇ .= market.q .* (alg.sampler.batchsize / market.m) - market.sumx
    elseif alg.optimizer.style == :lse
        __lse_grad!(alg, market)
    end
end

function __linear_hess!(alg, market::FisherMarket; bool_dbg=false, ε_hess=1e-4)
    if alg.linsys == :direct
        if alg.optimizer.style ∈ (:dual_lp, :dual_lp_conic)
            # use slack-based Hessian directly (no ε inflation needed)
            __linear_hess_from_s!(alg, market)
            # elseif alg.optimizer.style ∈ (:linconic)
            #     __linear_hess_fromx!(alg, market)
        elseif alg.optimizer.style == :lse
            __lse_compute_exact_hess!(alg, market)
        else
            # @note: type-1
            # use a large ε for Hessian computation (better conditioned),
            # then revert to actual precision
            # ε_orig = copy(market.ε_br_play)
            # market.ε_br_play .= max.(market.ε_br_play, ε_hess)
            # play!(alg, market; all=true, timed=false)
            # __linear_compute_exact_hess!(alg, market)
            # market.ε_br_play .= ε_orig
            # play!(alg, market; all=true, timed=false)
            # @note: type-2, adaptive
            market.ε_br_play .= max.(market.ε_br_play, ε_hess)
            play!(alg, market; all=true, timed=false)
            __linear_compute_exact_hess!(alg, market)
        end
    elseif alg.linsys == :krylov
        # no preprocessing needed
    else
        throw(ArgumentError("linsys not supported for linear market: $(alg.linsys)"))
    end
end

function __linear_eval!(alg, market::Market)
    if alg.optimizer.style == :lse
        __lse_eval!(alg, market)
    else
        alg.φ = min(
            alg.p' * market.q +
            sum(market.w[i] * log(market.val_u[i]) for i in 1:market.m),
            1e8
        )
    end
end


# -----------------------------------------------------------------------
# compute the exact Hessian
# -----------------------------------------------------------------------
@doc raw"""
    __linear_compute_exact_hess!(alg, market::FisherMarket)
    Compute the exact Hessian for the linear case with log-barrier regularization.
    H = Σᵢ wᵢ Hᵢ, where per-buyer:
        H_i = (1/w_i)(λ_i/σ_i X_i² - λ_i/(σ_i(σ_i + ‖γ_i‖²)) X_i γ_i γ_i' X_i)
    so wᵢ Hᵢ = λ_i/σ_i X_i² - λ_i/(σ_i(σ_i + ‖γ_i‖²)) X_i γ_i γ_i' X_i
    where σ_i = ε_br_play[i]/n, λ_i = (1 + σ_i n)/w_i, γ_i = x_i ⊙ c_i / u_i.
"""
function __linear_compute_exact_hess!(alg, market::FisherMarket; dbg=true)
    if isa(alg.H, SparseMatrixCSC)
        alg.H = Matrix(alg.H)
    end
    alg.H .= 0.0
    n, m = size(market.x)
    X = market.x                    # n×m
    w = market.w                    # m
    ε = market.ε_br_play            # m
    σv = ε ./ n                     # m
    λv = (1 .+ σv .* n) ./ w       # m

    # diagonal coefficients: λ_i/σ_i per buyer
    diag_coeffs = λv ./ σv          # m

    # diagonal term: Σᵢ (λᵢ/σᵢ) x_i²  →  X² * diag_coeffs
    # γ = Xc/⟨c,x⟩ should equal (1+σn) Xp/w - σ1 (FOC identity)
    γ = X .* market.c ./ market.val_u'                              # n×m
    γ_foc = (1 .+ σv' .* n) .* (X .* alg.p) ./ w' .- σv'          # n×m

    XG = X .* γ                                                     # n×m
    γnorm2 = vec(sum(γ .^ 2; dims=1))                              # m
    rank1_coeffs = λv ./ (σv .* (σv .+ γnorm2))                    # m

    # verify per-buyer: γ FOC identity and W_i H_i = λ_i I
    max_err_γ, max_i_γ = 0.0, 0
    max_err_H, max_i_H = 0.0, 0
    for i in 1:m
        xᵢ = X[:, i]
        cᵢ = market.c[:, i]
        uᵢ = market.val_u[i]
        σᵢ = σv[i]
        λᵢ = λv[i]
        err_γ = maximum(abs.(γ[:, i] - γ_foc[:, i]))
        if err_γ > max_err_γ
            max_err_γ = err_γ
            max_i_γ = i
        end
        Wᵢ = cᵢ * cᵢ' / uᵢ^2 + σᵢ * diagm(1 ./ xᵢ .^ 2)
        Hᵢ = diag_coeffs[i] * diagm(xᵢ .^ 2) - rank1_coeffs[i] * (xᵢ .* γ[:, i]) * (xᵢ .* γ[:, i])'
        err_H = maximum(abs.(Wᵢ * Hᵢ - λᵢ * I(n)))
        if err_H > max_err_H
            max_err_H = err_H
            max_i_H = i
        end
        alg.H .+= Hᵢ
    end
    if dbg || (max_err_γ > 1e-4 || max_err_H > 1e-4)
        println("max_err_γ: $max_err_γ, max_err_H: $max_err_H")
        println("W_i H_i = λ_i I worst buyer $max_i_H: max error = $max_err_H @ε = $(ε[max_i_H])")
    end

    return nothing
end




# -----------------------------------------------------------------------
# general CES case: ρ < 1
# -----------------------------------------------------------------------
function __ces_eval!(alg, market::Market)
    w = market.w
    alg.φ = min(
        alg.p' * market.q +
        sum(w[i] * log(market.val_u[i]) for i in 1:market.m),
        1e8
    )
end

function __ces_grad!(alg, market::Market)
    @assert all(market.ρ .< 1)
    alg.∇ .= market.q .* (alg.sampler.batchsize / market.m) - market.sumx
end

function __ces_hess!(alg, market::FisherMarket)
    # compute 1/σ w_i * log(cs_i'p^{-σ})
    if alg.linsys == :direct
        __ces_compute_exact_hess_optimized!(alg, market)
    elseif alg.linsys == :direct_afcon
        # compute the exact Hessian of the affine-constrained problem
        __ces_compute_exact_hess_afcon!(alg, market)
    elseif alg.linsys == :DRq
        __ces_compute_approx_hess_drq!(alg, market)
    elseif alg.linsys == :krylov
        # no preprocessing needed
    else
        throw(ArgumentError("linsys not supported: $(alg.linsys)"))
    end
end

# -----------------------------------------------------------------------
# compute the exact Hessian
# -----------------------------------------------------------------------
@doc raw"""
    __ces_compute_exact_hess!(alg, market::FisherMarket)
    Compute the exact Hessian of the problem, ∇²f, not affine-scaled
"""
function __ces_compute_exact_hess!(alg, market::FisherMarket)
    # bidding matrix shape: n×m
    γ = (alg.p .* market.x) ./ market.w'
    u = market.w .* market.σ
    alg.H .= diagm(1 ./ alg.p) * (diagm(γ * (market.w .+ u)) - γ * diagm(u) * γ') * diagm(1 ./ alg.p)
    @info "use exact Hessian"
end

function __ces_compute_exact_hess_optimized!(alg, market::FisherMarket)
    # — ensure a dense target ———————————————––
    if isa(alg.H, SparseMatrixCSC)
        alg.H = Matrix(alg.H)
    end
    H = alg.H                      # n×n dense
    # --- unpack ---------------------------------------------------------------
    p = alg.p                     # length n
    X = market.x                  # n×m
    w = market.w                  # length m  (strictly >0)
    σv = market.σ                 # length m (heterogeneous allowed)
    n, m = size(X)

    @assert length(p) == n
    @assert size(H) == (n, n)
    @assert length(w) == m

    W = similar(X)                     # n×m

    # --- 1) diagonal term ------------------------------------------------------
    # From H = P⁻¹[diag(γ*(w+u)) - γ diag(u) γ']P⁻¹, diag part becomes
    # diag_k = (∑_i X[k,i]*(1+σ_i)) / p_k
    row_weighted = X * (1 .+ σv)
    diag_term = @. row_weighted / p

    # --- 2) build W = X .* (1 ./ √w)' -----------------------------------------
    inv_sqrt_w = 1 ./ sqrt.(w)                  # m  (tiny alloc)
    @inbounds for j in 1:m
        @views W[:, j] .= X[:, j] .* inv_sqrt_w[j]
    end

    # --- 3) H ← diag(diag_term) ------------------------------------------------
    fill!(H, 0)
    @inbounds for i in 1:n
        H[i, i] = diag_term[i]
    end

    # --- 4) rank-k update: H ← H - ∑ᵢ σᵢ·W[:,i]·W[:,i]ᵀ -----------------------
    for j in 1:m
        LinearAlgebra.BLAS.syr!('U', -σv[j], view(W, :, j), H)
    end

    # --- 5) mirror upper → lower ----------------------------------------------
    @inbounds for j in 1:n-1, i in j+1:n
        H[i, j] = H[j, i]
    end

    println("exact dense Hessian built (heterogeneous σ)")
    return nothing
end

# -----------------------------------------------------------------------
# compute Hessian-vector product (affine-scaled)
# -----------------------------------------------------------------------
function __compute_exact_hessop_afscale_optimized!(
    buff::Vector{T}, alg, market::FisherMarket{T}, v::Vector{T}; add_μ::Bool=false,
) where {T}

    b = alg.Hk.b              # n×m
    σ = market.σ
    μ = alg.μ
    w = market.w              # length-m
    n, m = size(b)

    @assert length(v) == n && length(buff) == n && length(w) == m

    row_sum = zeros(T, n)
    b_t_v = zeros(T, m)

    # 1) diag term vector = b * (1 .+ σ)
    mul!(row_sum, b, 1 .+ σ)          # BLAS gemv

    # 2) b_t_v  = b' * v           ⇒  Σ_i b[i,j]*v[i]
    mul!(b_t_v, transpose(b), v)      # BLAS gemv

    # 3) scale   b_t_v  .=  b_t_v ./ w
    @. b_t_v /= w

    # 4) off-diagonal product term: b * (σ .* b_t_v)
    @. b_t_v = σ * b_t_v
    mul!(buff, b, b_t_v)             # BLAS gemv

    # 5) fuse the rest
    μ_eff = add_μ ? μ : zero(μ)
    @. buff = row_sum * v - buff + μ_eff * v
    return nothing
end

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

