using LinearAlgebra, Printf, SparseArrays
using SuiteSparse
using ExchangeMarket: sparse_col_ref, SparseColRef, sparse_dot, sparse_div_max

"""
    Primal-dual interior-point method for the linear Fisher market dual:

    min_{p ≥ 0, λ ≥ 0}  ⟨p, 1⟩ - Σᵢ wᵢ log(λᵢ) - Σᵢ wᵢ log wᵢ
    s.t.  λᵢ cᵢ - p ≤ 0,  i ∈ [m]

    Variables: p ∈ Rⁿ, λ ∈ Rᵐ, sᵢ = p - λᵢcᵢ ∈ Rⁿ (slack),
               xᵢ ∈ Rⁿ (dual multiplier for constraint i), q ∈ Rⁿ (dual for p ≥ 0).

    Central path conditions:
            1 - Σᵢ xᵢ + q = 0      | r₁ = 1 - Σᵢ xᵢ + q
        -wᵢ/λᵢ + ⟨cᵢ, xᵢ⟩ = 0      | r₂
            sᵢ + λᵢcᵢ - p = 0      | r₃
               p ∘ q - μ1 = 0      | c₁
             xᵢ ∘ sᵢ - μ1 = 0      | c₂

    Newton system reduced to n×n Schur complement for Δp:
        (H - C D⁻¹ Cᵀ) Δp = r̃₁ + C D⁻¹ r̃₂
    where H = Σᵢ Wᵢ + P⁻¹Q (diagonal), D = diag(wᵢ/λᵢ² + cᵢᵀWᵢcᵢ) (diagonal),
    and Wᵢ = Xᵢ Sᵢ⁻¹ (element-wise xᵢ/sᵢ).
"""

# -----------------------------------------------------------------------
# KKT system workspace (pre-allocated for Schur complement solve)
# -----------------------------------------------------------------------
mutable struct KKTSystem
    n::Int
    m::Int
    c::Union{Matrix{Float64},SparseMatrixCSC{Float64,Int}}  # cost matrix (dense or sparse)
    w::Vector{Float64}        # budgets (m)
    c_refs::Vector             # precomputed column refs (SparseColRef or SubArray)
    H_diag::Vector{Float64}   # diagonal of H (n)
    d_diag::Vector{Float64}   # diagonal of D (m)
    Wc::Matrix{Float64}       # n × m, columns Wᵢcᵢ
    S_mat::Matrix{Float64}    # n × n dense Schur complement
    S_sparse::Union{SparseMatrixCSC{Float64,Int},Nothing}       # sparse Schur complement
    S_diag_idx::Vector{Int}   # indices into nonzeros(S_sparse) for diagonal entries
    chol_factor::Union{SuiteSparse.CHOLMOD.Factor{Float64},Nothing}  # CHOLMOD symbolic factor
    r̃1::Vector{Float64}       # modified RHS (n)
    r̃2::Vector{Float64}       # modified RHS (m)
    rhs::Vector{Float64}      # final RHS for Δp (n)
end

# Dense constructor (default)
function KKTSystem(n::Int, m::Int, c, w::Vector{Float64})
    c_refs = [sparse_col_ref(c, i) for i in 1:m]
    KKTSystem(
        n, m, c, w, c_refs,
        zeros(n), zeros(m),
        zeros(n, m), zeros(n, n),
        nothing, Int[], nothing,
        zeros(n), zeros(m), zeros(n)
    )
end

# Sparse constructor: precompute sparsity pattern and symbolic Cholesky
function KKTSystem(n::Int, m::Int, c::SparseMatrixCSC, w::Vector{Float64}, ::Val{:sparse})
    c_refs = [sparse_col_ref(c, i) for i in 1:m]
    I_idx = Int[]; J_idx = Int[]
    for j in 1:n; push!(I_idx, j); push!(J_idx, j); end
    for i in 1:m
        nzi = c_refs[i].nzind
        for r in nzi, s in nzi
            push!(I_idx, r); push!(J_idx, s)
        end
    end
    S_pattern = sparse(I_idx, J_idx, ones(length(I_idx)), n, n)
    S_pattern = S_pattern + S_pattern'
    for j in 1:n; S_pattern[j, j] = 1.0; end
    S_sparse = copy(S_pattern)
    chol_factor = cholesky(Symmetric(S_pattern); check=false)

    KKTSystem(
        n, m, c, w, c_refs,
        zeros(n), zeros(m),
        zeros(n, m), zeros(n, n),
        S_sparse, Int[], chol_factor,
        zeros(n), zeros(m), zeros(n)
    )
end

# -----------------------------------------------------------------------
# Algorithm struct
# -----------------------------------------------------------------------
Base.@kwdef mutable struct PDIPM{T}
    n::Int
    m::Int
    p::Vector{T}
    μ::Float64 = 0.0
    kkt::KKTSystem
    name::Symbol = :PDIPM
end

# -----------------------------------------------------------------------
# Data structure for the PD iterate
# -----------------------------------------------------------------------
mutable struct PDState
    n::Int          # number of goods
    m::Int          # number of buyers
    p::Vector{Float64}
    λ::Vector{Float64}
    s::Matrix{Float64}   # n × m, sᵢ = p - λᵢcᵢ
    x::Matrix{Float64}   # n × m, dual multipliers
    q::Vector{Float64}   # dual for p ≥ 0
    μ::Float64
    # directions
    Δp::Vector{Float64}
    Δλ::Vector{Float64}
    Δs::Matrix{Float64}
    Δx::Matrix{Float64}
    Δq::Vector{Float64}
end

function PDState(n::Int, m::Int)
    PDState(
        n, m,
        zeros(n), zeros(m),
        zeros(n, m), zeros(n, m), zeros(n),
        1.0,
        zeros(n), zeros(m),
        zeros(n, m), zeros(n, m), zeros(n)
    )
end

# -----------------------------------------------------------------------
# Initialize: find a strictly feasible starting point
# -----------------------------------------------------------------------
"""
    pd_init!(st, kkt; μ₀)

Initialize PDState from KKT data.
Set p = 1, λᵢ = 1/(2 max_j c_ji/p_j) so sᵢ = p - λᵢcᵢ > 0,
then set xᵢ, q from central path complementarity.
"""
function pd_init!(st::PDState, kkt::KKTSystem; μ₀::Float64=1.0)
    n, m = kkt.n, kkt.m
    c, w = kkt.c, kkt.w

    # initial prices: uniform
    st.p .= sum(w) / n
    st.μ = μ₀

    # λᵢ chosen so sᵢ = p - λᵢcᵢ > 0
    for i in 1:m
        cᵢ = @view c[:, i]
        ratio_max = maximum(cᵢ ./ st.p)
        if ratio_max > 0
            st.λ[i] = 0.5 / ratio_max
        else
            st.λ[i] = 1.0
        end
        st.s[:, i] .= st.p .- st.λ[i] .* cᵢ
    end

    # xᵢ from complementarity: xᵢ = μ / sᵢ
    for i in 1:m
        st.x[:, i] .= st.μ ./ st.s[:, i]
    end

    # q from complementarity: q = μ / p
    st.q .= st.μ ./ st.p

    return st
end

# -----------------------------------------------------------------------
# Residuals
# -----------------------------------------------------------------------
function pd_residuals(st::PDState, kkt::KKTSystem)
    c, w = kkt.c, kkt.w
    n, m = st.n, st.m
    μ = st.μ

    # r₁ = 1 - Σᵢ xᵢ + q
    r1 = ones(n) .- sum(st.x; dims=2)[:] .+ st.q

    # r₂ᵢ = -wᵢ/λᵢ + ⟨cᵢ, xᵢ⟩
    r2 = zeros(m)
    for i in 1:m
        r2[i] = -w[i] / st.λ[i] + sparse_dot(kkt.c_refs[i], st.x[:, i])
    end

    # r₃ᵢ = sᵢ + λᵢcᵢ - p
    r3 = zeros(n, m)
    for i in 1:m
        r3[:, i] .= st.s[:, i] .+ st.λ[i] .* c[:, i] .- st.p
    end

    # c₁ = p ∘ q - μ1
    c1 = st.p .* st.q .- μ

    # c₂ᵢ = xᵢ ∘ sᵢ - μ1
    c2 = zeros(n, m)
    for i in 1:m
        c2[:, i] .= st.x[:, i] .* st.s[:, i] .- μ
    end

    return (r1=r1, r2=r2, r3=r3, c1=c1, c2=c2)
end

# -----------------------------------------------------------------------
# Fill KKT workspace from current iterate
# -----------------------------------------------------------------------
function update_kkt!(kkt::KKTSystem, st::PDState, res)
    c, w = kkt.c, kkt.w
    n, m = st.n, st.m
    μ = st.μ
    r1, r2, r3 = res.r1, res.r2, res.r3

    kkt.H_diag .= st.q ./ st.p
    kkt.r̃1 .= .-r1 .+ (μ ./ st.p .- st.q)
    kkt.r̃2 .= 0.0

    for i in 1:m
        sᵢ = @view st.s[:, i]
        xᵢ = @view st.x[:, i]
        cᵢ = @view c[:, i]
        r3ᵢ = @view r3[:, i]

        Wᵢ = xᵢ ./ sᵢ
        Wᵢcᵢ = Wᵢ .* cᵢ

        kkt.H_diag .+= Wᵢ

        ξᵢ = (μ .- xᵢ .* sᵢ .+ xᵢ .* r3ᵢ) ./ sᵢ
        kkt.r̃1 .+= ξᵢ
        kkt.r̃2[i] = -r2[i] - sparse_dot(kkt.c_refs[i], ξᵢ)
        kkt.d_diag[i] = w[i] / st.λ[i]^2 + sparse_dot(kkt.c_refs[i], Wᵢcᵢ)
        kkt.Wc[:, i] .= Wᵢcᵢ
    end
    return kkt
end

# -----------------------------------------------------------------------
# Schur complement solve: (H - C D⁻¹ Cᵀ) Δp = r̃₁ + C D⁻¹ r̃₂
# -----------------------------------------------------------------------
function solve_schur_exact!(Δp::Vector{Float64}, kkt::KKTSystem)
    n, m = kkt.n, kkt.m
    kkt.S_mat .= 0.0
    @inbounds for j in 1:n
        kkt.S_mat[j, j] = kkt.H_diag[j]
    end
    kkt.rhs .= kkt.r̃1
    for i in 1:m
        Wᵢcᵢ = @view kkt.Wc[:, i]
        inv_d = 1.0 / kkt.d_diag[i]
        # rank-1 update: S -= inv_d Wᵢcᵢ (Wᵢcᵢ)ᵀ
        # Wᵢcᵢ has same sparsity as cᵢ — use c_refs for O(nnz²) instead of O(n²)
        _schur_rank1_update!(kkt.S_mat, Wᵢcᵢ, kkt.c_refs[i], inv_d)
        _schur_rhs_update!(kkt.rhs, Wᵢcᵢ, kkt.c_refs[i], kkt.r̃2[i] * inv_d)
    end
    F = cholesky!(Symmetric(kkt.S_mat))
    Δp .= F \ kkt.rhs
    return Δp
end

# Sparse: O(nnz²) — only iterate nonzero positions
@inline function _schur_rank1_update!(S, Wc, cref::SparseColRef, α)
    nzi = cref.nzind
    @inbounds for kl in eachindex(nzi)
        l = nzi[kl]
        wl = Wc[l]
        for kj in eachindex(nzi)
            S[nzi[kj], l] -= α * Wc[nzi[kj]] * wl
        end
    end
end

# Dense: O(n²)
@inline function _schur_rank1_update!(S, Wc, cref::AbstractVector, α)
    n = length(Wc)
    @inbounds for l in 1:n, j in 1:n
        S[j, l] -= α * Wc[j] * Wc[l]
    end
end

@inline function _schur_rhs_update!(rhs, Wc, cref::SparseColRef, α)
    @inbounds for k in eachindex(cref.nzind)
        j = cref.nzind[k]
        rhs[j] += α * Wc[j]
    end
end

@inline function _schur_rhs_update!(rhs, Wc, cref::AbstractVector, α)
    rhs .+= α .* Wc
end

"""
    solve_schur_dr1!(Δp, kkt)

DR1 (diagonal + rank-1) approximation of the Schur complement solve.
Replaces Σᵢ βᵢ vᵢvᵢᵀ with Ω v̄v̄ᵀ where v̄ = Σᵢ ωᵢ vᵢ, ωᵢ = βᵢ/Ω,
then applies Sherman-Morrison to invert H - Ω v̄v̄ᵀ in O(n) time.
"""
function solve_schur_dr1!(Δp::Vector{Float64}, kkt::KKTSystem)
    n, m = kkt.n, kkt.m

    # Build RHS: r̃₁ + C D⁻¹ r̃₂
    kkt.rhs .= kkt.r̃1

    # Compute v̄ = Σᵢ ωᵢ vᵢ and Ω = Σᵢ βᵢ, where βᵢ = 1/dᵢ, vᵢ = Wᵢcᵢ
    Ω = 0.0
    for i in 1:m
        Ω += 1.0 / kkt.d_diag[i]
    end

    # Accumulate v̄ (reuse Δp as temporary) and RHS
    Δp .= 0.0
    for i in 1:m
        Wᵢcᵢ = @view kkt.Wc[:, i]
        inv_d = 1.0 / kkt.d_diag[i]
        ωᵢ = inv_d / Ω
        Δp .+= ωᵢ .* Wᵢcᵢ       # v̄
        kkt.rhs .+= (kkt.r̃2[i] * inv_d) .* Wᵢcᵢ
    end
    # v̄ is now in Δp

    # Sherman-Morrison: (H - Ω v̄v̄ᵀ)⁻¹ = H⁻¹ + Ω/(1 - Ω v̄ᵀH⁻¹v̄) H⁻¹v̄ v̄ᵀH⁻¹
    # Compute H⁻¹v̄ and H⁻¹rhs
    # Use kkt.r̃1 as scratch for H⁻¹v̄ (no longer needed)
    H_inv_v̄ = kkt.r̃1
    H_inv_v̄ .= Δp ./ kkt.H_diag
    v̄ᵀH_inv_v̄ = dot(Δp, H_inv_v̄)

    H_inv_rhs = kkt.rhs ./ kkt.H_diag  # element-wise

    # Sherman-Morrison coefficient
    γ = Ω / (1.0 - Ω * v̄ᵀH_inv_v̄)

    # Δp = H⁻¹rhs + γ (v̄ᵀ H⁻¹ rhs) H⁻¹v̄
    v̄ᵀH_inv_rhs = dot(Δp, H_inv_rhs)  # Δp still holds v̄
    Δp .= H_inv_rhs .+ (γ * v̄ᵀH_inv_rhs) .* H_inv_v̄

    return Δp
end

# -----------------------------------------------------------------------
# Sparse Schur complement solve via CHOLMOD
# -----------------------------------------------------------------------
"""
    solve_schur_sparse!(Δp, kkt)

Sparse Cholesky solve of (H - Σᵢ βᵢ vᵢvᵢᵀ) Δp = rhs.
Uses pre-computed symbolic factorization in `kkt.chol_factor`; only does
numeric refactorization each call.
"""
function solve_schur_sparse!(Δp::Vector{Float64}, kkt::KKTSystem)
    n, m = kkt.n, kkt.m
    S = kkt.S_sparse

    # Zero out nonzero entries, then fill
    nzv = nonzeros(S)
    nzv .= 0.0

    # Add H_diag on diagonal
    for j in 1:n
        S[j, j] = kkt.H_diag[j]
    end

    # Build RHS and subtract rank-1 contributions at nonzero positions
    kkt.rhs .= kkt.r̃1
    for i in 1:m
        Wᵢcᵢ = @view kkt.Wc[:, i]
        inv_d = 1.0 / kkt.d_diag[i]
        kkt.rhs .+= (kkt.r̃2[i] * inv_d) .* Wᵢcᵢ
        # Subtract inv_d * Wᵢcᵢ * Wᵢcᵢᵀ at stored positions
        # Only touch entries in the sparsity pattern
        rows = rowvals(S)
        for col in 1:n
            v_col = Wᵢcᵢ[col]
            if v_col == 0.0
                continue
            end
            for idx in nzrange(S, col)
                row = rows[idx]
                nzv[idx] -= inv_d * Wᵢcᵢ[row] * v_col
            end
        end
    end

    # Numeric refactorization using existing symbolic factor
    cholesky!(kkt.chol_factor, Symmetric(S))
    Δp .= kkt.chol_factor \ kkt.rhs
    return Δp
end

# -----------------------------------------------------------------------
# Newton step via Schur complement
# -----------------------------------------------------------------------
"""
    pd_newton!(st, kkt; mode=:exact)

Compute the Newton direction for the central path system.
Reduces to an n×n system for Δp via Schur complement,
then back-substitutes for Δλ, Δs, Δx, Δq.

Solve mode (`mode`):
- `:exact`  — Cholesky factorization of the full Schur complement (O(n²m + n³))
- `:dr1`    — DR1 approximation + Sherman-Morrison (O(nm))

When KKT was built with sparse constructor, `:exact` uses sparse CHOLMOD Cholesky.
"""
function pd_newton!(st::PDState, kkt::KKTSystem; mode::Symbol=:exact)
    t_residuals = @elapsed begin
        res = pd_residuals(st, kkt)
    end
    t_update_kkt = @elapsed begin
        update_kkt!(kkt, st, res)
    end

    # Solve Schur complement system for Δp
    t_schur = @elapsed begin
        if mode == :exact
            if kkt.S_sparse !== nothing
                solve_schur_sparse!(st.Δp, kkt)
            else
                solve_schur_exact!(st.Δp, kkt)
            end
        elseif mode == :dr1
            solve_schur_dr1!(st.Δp, kkt)
        else
            error("unknown mode: $mode, expected :exact or :dr1")
        end
    end

    # Back-substitute for remaining directions
    t_backsub = @elapsed begin
        update_agent_step!(st, kkt, res)
    end
    return (res=res, t_residuals=t_residuals, t_update_kkt=t_update_kkt, t_schur=t_schur, t_backsub=t_backsub)
end

"""
    update_agent_step!(st, kkt, res)

Given Δp, back-substitute for Δλ, Δs, Δx, Δq.
"""
function update_agent_step!(st::PDState, kkt::KKTSystem, res)
    c = kkt.c
    n, m = st.n, st.m
    μ = st.μ
    r3 = res.r3

    for i in 1:m
        sᵢ = @view st.s[:, i]
        xᵢ = @view st.x[:, i]
        cᵢ = @view c[:, i]
        r3ᵢ = @view r3[:, i]

        # Δλᵢ = dᵢ⁻¹ (r̃₂ᵢ + cᵢᵀ Wᵢ Δp)
        st.Δλ[i] = (kkt.r̃2[i] + dot(kkt.Wc[:, i], st.Δp)) / kkt.d_diag[i]

        # Δsᵢ = Δp - cᵢ Δλᵢ - r₃ᵢ
        st.Δs[:, i] .= st.Δp .- cᵢ .* st.Δλ[i] .- r3ᵢ

        # Δxᵢ = Sᵢ⁻¹ (μ1 - xᵢ∘sᵢ - Xᵢ Δsᵢ)
        st.Δx[:, i] .= (μ .- xᵢ .* sᵢ .- xᵢ .* st.Δs[:, i]) ./ sᵢ
    end

    # Δq = P⁻¹ (μ1 - p∘q - Q Δp)
    st.Δq .= (μ .- st.p .* st.q .- st.q .* st.Δp) ./ st.p
    return st
end

# -----------------------------------------------------------------------
# Step size: max α ∈ (0,1] s.t. (s,x,p,q,λ) + α Δ(·) > 0
# -----------------------------------------------------------------------
function pd_stepsize(st::PDState; τ::Float64=0.9995)
    α = 1.0
    n, m = st.n, st.m

    # p + α Δp > 0
    for j in 1:n
        if st.Δp[j] < 0
            α = min(α, -st.p[j] / st.Δp[j])
        end
    end
    # q + α Δq > 0
    for j in 1:n
        if st.Δq[j] < 0
            α = min(α, -st.q[j] / st.Δq[j])
        end
    end
    # λᵢ + α Δλᵢ > 0
    for i in 1:m
        if st.Δλ[i] < 0
            α = min(α, -st.λ[i] / st.Δλ[i])
        end
    end
    # sᵢ + α Δsᵢ > 0, xᵢ + α Δxᵢ > 0
    for i in 1:m
        for j in 1:n
            if st.Δs[j, i] < 0
                α = min(α, -st.s[j, i] / st.Δs[j, i])
            end
            if st.Δx[j, i] < 0
                α = min(α, -st.x[j, i] / st.Δx[j, i])
            end
        end
    end

    return τ * α
end

# -----------------------------------------------------------------------
# Update iterate
# -----------------------------------------------------------------------
function pd_update!(st::PDState, α::Float64)
    st.p .+= α .* st.Δp
    st.q .+= α .* st.Δq
    st.λ .+= α .* st.Δλ
    for i in 1:st.m
        st.s[:, i] .+= α .* st.Δs[:, i]
        st.x[:, i] .+= α .* st.Δx[:, i]
    end
end

# -----------------------------------------------------------------------
# Complementarity gap
# -----------------------------------------------------------------------
function pd_gap(st::PDState)
    n, m = st.n, st.m
    # gap = p'q + Σᵢ xᵢ'sᵢ
    gap = dot(st.p, st.q)
    for i in 1:m
        gap += dot(st.x[:, i], st.s[:, i])
    end
    ncomp = n + n * m  # number of complementarity pairs
    return gap, gap / ncomp
end

# -----------------------------------------------------------------------
# Main solver
# -----------------------------------------------------------------------
"""
    pd_ipm(f; μ₀, maxiter, tol, mode, show_trace, show_every)

Primal-dual interior-point method for the linear Fisher market.

# Arguments
- `mode::Symbol`: Solve method
  - `:exact` — Cholesky factorization (default)
  - `:dr1`   — DR1 approximation + Sherman-Morrison
- `linalg::Symbol`: Matrix format for Schur complement
  - `:auto`   (default) — sparse if `f.c isa SparseMatrixCSC && f.sparsity ≤ 0.1`, else dense
  - `:sparse` — force sparse CHOLMOD (requires `f.c isa SparseMatrixCSC`)
  - `:dense`  — force dense

Returns (st, alg, traj).
"""
function pd_ipm(
    f;
    μ₀::Float64=1.0,
    maxiter::Int=200,
    tol::Float64=1e-10,
    mode::Symbol=:exact,
    linalg::Symbol=:auto,
    show_trace::Bool=true,
    show_every::Int=1
)
    n, m = f.n, f.m

    # Resolve linalg flag
    _use_sparse = if linalg == :auto
        f.c isa SparseMatrixCSC && f.sparsity <= 0.1
    elseif linalg == :sparse
        true
    elseif linalg == :dense
        false
    else
        error("unknown linalg: $linalg, expected :auto, :sparse, or :dense")
    end

    # Build KKT system with its own copy of c and w
    c_local = f.c isa SparseMatrixCSC ? (
        _use_sparse ? copy(f.c) : Matrix(f.c)
    ) : copy(f.c)
    w_local = copy(f.w)

    if _use_sparse
        kkt = KKTSystem(n, m, c_local, w_local, Val(:sparse))
    else
        kkt = KKTSystem(n, m, c_local, w_local)
    end
    alg = PDIPM(n=n, m=m, p=zeros(n), kkt=kkt)
    st = PDState(n, m)
    pd_init!(st, kkt; μ₀=μ₀)

    traj = []

    _linalg_str = _use_sparse ? "sparse" : "dense"
    if show_trace
        _loghead = @sprintf("%5s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s",
            "k", "μ", "|mc|", "|r₂|", "|r₃|", "|c₁|", "|c₂|", "α", "time(s)")
        _w = length(_loghead)
        _sep = "-"^_w
        _header = [
            "ExchangeMarket.jl: Primal-Dual IPM",
            "mode=:$mode, linalg=$_linalg_str (n=$n, m=$m)",
        ]
        println(_sep)
        for line in _header
            pad = max(0, _w - length(line))
            lpad = pad ÷ 2
            println(" "^lpad * line)
        end
        println(_sep)
        println(_loghead)
        println(_sep)
    end

    _t0 = time()
    for k in 1:maxiter
        # compute Newton direction
        newton_result = pd_newton!(st, alg.kkt; mode=mode)
        res = newton_result.res

        # step size
        α = pd_stepsize(st)

        # complementarity gap
        gap, μ_avg = pd_gap(st)

        # residual norms
        nr1 = norm(res.r1, Inf)
        nr2 = norm(res.r2, Inf)
        nr3 = norm(res.r3, Inf)
        nc1 = norm(res.c1, Inf)
        nc2 = norm(res.c2, Inf)

        _elapsed = time() - _t0
        if show_trace && k % show_every == 0
            @printf("%5d | %+10.3e | %10.3e | %10.3e | %10.3e | %10.3e | %10.3e | %10.4f | %10.4f\n",
                k, st.μ, nr1, nr2, nr3, nc1, nc2, α, _elapsed)
            @printf("%5s |- t_residuals=%.4f  t_update_kkt=%.4f  t_schur=%.4f  t_backsub=%.4f\n", "",
                newton_result.t_residuals, newton_result.t_update_kkt,
                newton_result.t_schur, newton_result.t_backsub)
        end

        push!(traj, (k=k, μ=st.μ, nr1=nr1, nr2=nr2, nr3=nr3, nc1=nc1, nc2=nc2, α=α, t=_elapsed))

        # check convergence
        if nr1 < tol && nr2 < √tol
            show_trace && @printf("converged at k=%d, gap=%.2e\n", k, gap)
            break
        end

        # update iterate
        pd_update!(st, α)

        # update μ: simple centering heuristic σ = (1 - α)
        _, μ_new = pd_gap(st)
        σ = (1 - α)
        st.μ = σ * μ_new
    end

    return (st=st, alg=alg, traj=traj)
end

# -----------------------------------------------------------------------
# Extract allocation from PD solution
# -----------------------------------------------------------------------
"""
    pd_allocation!(f, st)

Given converged PDState, recover allocation x and utilities.
xᵢ is the dual multiplier for the i-th constraint, which equals
the allocation at optimality.
"""
function pd_allocation!(f, st::PDState, alg::PDIPM)
    c = alg.kkt.c
    f.p .= st.p
    alg.p .= st.p
    alg.μ = st.μ
    for i in 1:f.m
        f.x[:, i] .= st.x[:, i]
        f.val_u[i] = sparse_dot(alg.kkt.c_refs[i], f.x[:, i])
    end
    f.sumx .= sum(f.x; dims=2)[:]
    return alg
end
