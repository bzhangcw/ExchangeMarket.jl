using LinearAlgebra, SparseArrays, Printf
using ExchangeMarket: spow

# GPU augmented solve requires: using CUDA, CUDA.CUSPARSE, CUDSS

"""
    GPU-compatible Primal-Dual IPM for the linear Fisher market.

    All per-agent loops replaced with batched matrix operations.
    Works on CPU (Matrix/Vector) or GPU (CuMatrix/CuVector) —
    Julia broadcasting dispatches automatically.

    Solve modes (`mode`):
    - `:schur`  — Schur complement (n×n Cholesky), default
    - `:aug`    — augmented quasi-definite system ((n+m)×(n+m) LDLᵀ via CUDSS on GPU)

    Usage:
        device = identity                     # CPU
        # using CUDA, CUDA.CUSPARSE, CUDSS    # GPU (load before include)
        # device = CuArray                    # GPU (Float64, NOT cu which gives Float32)
        result = pd_ipm_gpu(f; device=device, mode=:schur)
"""

# -----------------------------------------------------------------------
# GPU-safe helpers
# -----------------------------------------------------------------------
"""Add diagonal vector to square matrix: S[j,j] += d[j]. GPU-compatible via kernel."""
function _add_diag!(S::AbstractMatrix, d::AbstractVector)
    # Use a GPU-compatible broadcast: create index array on device
    n = length(d)
    idx = (1:n) .+ (0:n-1) .* n   # linear indices of diagonal
    view(S, idx) .= view(S, idx) .+ d
    return S
end

# CPU fast path
function _add_diag!(S::Matrix, d::Vector)
    @inbounds for j in axes(d, 1)
        S[j, j] += d[j]
    end
    return S
end

# -----------------------------------------------------------------------
# Parametric data structures — work with any array backend
# -----------------------------------------------------------------------
mutable struct KKTSystemGPU{M<:AbstractMatrix,V<:AbstractVector}
    n::Int
    m::Int
    c::M               # n × m cost matrix
    w::V               # m budgets
    # workspace (pre-allocated)
    H_diag::V          # n
    d_diag::V          # m
    Wc::M              # n × m, Wᵢcᵢ columns
    Wc_scaled::M       # n × m, Wc * diag(1/√d) for syrk
    S_mat::M           # n × n Schur complement
    r̃1::V              # n
    r̃2::V              # m
    rhs::V             # n
    W::M               # n × m workspace for x./s
    ξ::M               # n × m workspace
    # augmented system (quasi-definite)
    aug_rhs::V         # (n+m) RHS
    aug_sol::V         # (n+m) solution workspace
    aug_sparse::Any    # CuSparseMatrixCSR (or CPU SparseMatrixCSC) — set by init_aug!
    aug_solver::Any    # CudssSolver (GPU) or nothing (CPU fallback) — set by init_aug!
    aug_diag_nzidx::Vector{Int}                 # nzval indices for diagonal [H; D] (length n+m)
    aug_Wc_nzidx::Vector{Tuple{Int,Int,Int}}    # (nzval_idx, col_j, agent_i) for -Wc block
    # GPU-vectorized index arrays for build_linsys_aug_gpu! (set by init_aug!)
    aug_diag_nzidx_gpu::Any     # CuVector{Int} — nzval targets for [H_diag; d_diag]
    aug_Wc_nzidx_gpu::Any       # CuVector{Int} — nzval targets for -Wc entries
    aug_Wc_src_gpu::Any          # CuVector{Int} — linear indices into Wc (source values)
end

function KKTSystemGPU(n, m, c::M, w::V) where {M,V}
    KKTSystemGPU(
        n, m, c, w,
        similar(w, n),          # H_diag
        similar(w, m),          # d_diag
        similar(c),             # Wc
        similar(c),             # Wc_scaled
        similar(c, n, n),       # S_mat
        similar(w, n),          # r̃1
        similar(w, m),          # r̃2
        similar(w, n),          # rhs
        similar(c),             # W
        similar(c),             # ξ
        similar(w, n + m),      # aug_rhs
        similar(w, n + m),      # aug_sol
        nothing,                # aug_sparse — set by init_aug!
        nothing,                # aug_solver — set by init_aug!
        Int[],                  # aug_diag_nzidx
        Tuple{Int,Int,Int}[],   # aug_Wc_nzidx
        nothing,                # aug_diag_nzidx_gpu
        nothing,                # aug_Wc_nzidx_gpu
        nothing,                # aug_Wc_src_gpu
    )
end

mutable struct PDStateGPU{M<:AbstractMatrix,V<:AbstractVector}
    n::Int
    m::Int
    p::V
    λ::V
    s::M               # n × m
    x::M               # n × m
    q::V               # n
    μ::Float64
    μ_scale::Float64   # agent complementarity: μ_a = max(μ * μ_scale, μ_a_min)
    μ_a_min::Float64   # floor for agent complementarity (default 0.0)
    Δp::V
    Δλ::V
    Δs::M
    Δx::M
    Δq::V
    name::Symbol
end

function PDStateGPU(n, m; to_dev=identity, name::Symbol=:PDIPM_GPU,
    μ_scale::Float64=1.0, μ_a_min::Float64=1e-9)
    PDStateGPU(
        n, m,
        to_dev(zeros(n)), to_dev(zeros(m)),
        to_dev(zeros(n, m)), to_dev(zeros(n, m)),
        to_dev(zeros(n)),
        1.0,
        μ_scale,
        μ_a_min,
        to_dev(zeros(n)), to_dev(zeros(m)),
        to_dev(zeros(n, m)), to_dev(zeros(n, m)),
        to_dev(zeros(n)),
        name,
    )
end

# -----------------------------------------------------------------------
# Initialize — all batched
# -----------------------------------------------------------------------
function pd_init_gpu!(st::PDStateGPU, kkt::KKTSystemGPU; μ₀=1.0)
    c, w = kkt.c, kkt.w
    n, m = kkt.n, kkt.m

    st.p .= sum(w) / n
    st.μ = μ₀
    μ_a = max(μ₀ * st.μ_scale, st.μ_a_min)

    # λᵢ = 0.5 / max_j(c[j,i]/p[j])  — batched
    ratio_max = dropdims(maximum(c ./ st.p, dims=1); dims=1)  # m-vector
    st.λ .= ifelse.(ratio_max .> 0, 0.5 ./ ratio_max, 1.0)

    # sᵢ = p - λᵢcᵢ  — broadcasting λ as row
    st.s .= st.p .- c .* st.λ'

    # xᵢ = μ_a / sᵢ  (agent complementarity)
    st.x .= μ_a ./ st.s

    # q = μ / p  (price complementarity)
    st.q .= st.μ ./ st.p

    return st
end

# -----------------------------------------------------------------------
# Residuals — all batched
# -----------------------------------------------------------------------
function pd_residuals_gpu(st::PDStateGPU, kkt::KKTSystemGPU)
    c, w = kkt.c, kkt.w
    μ = st.μ
    μ_a = max(μ * st.μ_scale, st.μ_a_min)

    r1 = 1.0 .- dropdims(sum(st.x; dims=2); dims=2) .+ st.q
    # r2[i] = -w[i]/λ[i] + dot(c[:,i], x[:,i])
    r2 = -w ./ st.λ .+ dropdims(sum(c .* st.x; dims=1); dims=1)
    # r3 = s + λ'.*c - p
    r3 = st.s .+ c .* st.λ' .- st.p
    c1 = st.p .* st.q .- μ
    c2 = st.x .* st.s .- μ_a

    return (r1=r1, r2=r2, r3=r3, c1=c1, c2=c2)
end

# -----------------------------------------------------------------------
# Update KKT workspace — all batched
# -----------------------------------------------------------------------
function update_kkt_gpu!(kkt::KKTSystemGPU, st::PDStateGPU, res)
    c, w = kkt.c, kkt.w
    μ = st.μ
    μ_a = max(μ * st.μ_scale, st.μ_a_min)

    # W = x ./ s (n × m)
    kkt.W .= st.x ./ st.s
    # Wc = W .* c
    kkt.Wc .= kkt.W .* c

    # H_diag = q./p + sum(W, dims=2)  (n-vector)
    kkt.H_diag .= st.q ./ st.p .+ dropdims(sum(kkt.W; dims=2); dims=2)

    # ξ = (μ_a - x.*s + x.*r3) ./ s  (n × m)  — agent complementarity
    kkt.ξ .= (μ_a .- st.x .* st.s .+ st.x .* res.r3) ./ st.s

    # r̃1 = -r1 + (μ./p - q) + sum(ξ, dims=2)  — price μ for p∘q term
    kkt.r̃1 .= .-res.r1 .+ (μ ./ st.p .- st.q) .+ dropdims(sum(kkt.ξ; dims=2); dims=2)

    # r̃2[i] = -r2[i] - dot(c[:,i], ξ[:,i])  (m-vector)
    kkt.r̃2 .= .-res.r2 .- dropdims(sum(c .* kkt.ξ; dims=1); dims=1)

    # d_diag[i] = w[i]/λ[i]² + dot(c[:,i], Wc[:,i])  (m-vector)
    kkt.d_diag .= w ./ st.λ .^ 2 .+ dropdims(sum(c .* kkt.Wc; dims=1); dims=1)

    return kkt
end

# -----------------------------------------------------------------------
# Schur complement solve — batched via BLAS gemm + Cholesky
#
#   S = diag(H) - Wc * diag(1/d) * Wc'
#     = diag(H) - Wc_scaled * Wc_scaled'
#   where Wc_scaled[:,i] = Wc[:,i] / √d[i]
#
#   rhs = r̃1 + Wc * (r̃2 ./ d)
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Build linear system — Schur complement (n × n)
#   S = diag(H) - Wc_scaled * Wc_scaled',  rhs = r̃1 + Wc * (r̃2 ./ d)
# -----------------------------------------------------------------------
function build_linsys_schur_gpu!(kkt::KKTSystemGPU)
    # Scale columns: Wc_scaled = Wc * diag(1/√d)
    kkt.Wc_scaled .= kkt.Wc ./ sqrt.(kkt.d_diag)'

    # S = diag(H) - Wc_scaled * Wc_scaled'
    mul!(kkt.S_mat, kkt.Wc_scaled, kkt.Wc_scaled', -1.0, 0.0)
    _add_diag!(kkt.S_mat, kkt.H_diag)

    # RHS = r̃1 + Wc * (r̃2 ./ d)
    kkt.rhs .= kkt.r̃1
    mul!(kkt.rhs, kkt.Wc, kkt.r̃2 ./ kkt.d_diag, 1.0, 1.0)
    return kkt
end

function solve_schur_gpu!(Δp, kkt::KKTSystemGPU)
    try
        F = cholesky!(Symmetric(kkt.S_mat))
        ldiv!(Δp, F, kkt.rhs)
        return (Δp, true)
    catch e
        if e isa PosDefException
            @warn "Schur complement not PD at iteration"
            Δp .= 0
            return (Δp, false)
        end
        rethrow(e)
    end
end

# -----------------------------------------------------------------------
# Neumann series solve for the Schur complement:
#
#   S = H - C D⁻¹ C' = H(I - A),  A = H⁻¹ C D⁻¹ C'
#
# Since S ≻ 0 ⟹ ρ(A) < 1, the Neumann series converges:
#   S⁻¹ = (Σₖ Aᵏ) H⁻¹
#
# Each Neumann term: 2 GEMVs through Wc (n×m), cost O(nm).
# Total: O(K·nm) vs O(n²m + n³) for GEMM + Cholesky.
#
# Optional Chebyshev acceleration replaces the geometric rate ρ(A)ᴷ
# with the faster Chebyshev rate when spectral bounds [a, b] ⊂ [0,1)
# of A are provided.
# -----------------------------------------------------------------------

"""
    solve_schur_gpu_neumann!(Δp, kkt; Kmax=50, tol=1e-10, chebyshev=false, spec_bounds=nothing)

Solve S Δp = rhs via truncated Neumann series.  Does NOT form the
dense S matrix — only uses H_diag, Wc, d_diag, rhs.

Cost: O(K nm) for K terms, via 2K GEMVs through Wc (n × m).

# Arguments
- `Kmax`: maximum number of terms / iterations
- `tol`: relative tolerance for early termination
- `chebyshev`: if true, use Chebyshev semi-iteration (Saad, Alg. 12.1)
- `spec_bounds`: `(a, b)` with `0 ≤ a ≤ b < 1`, spectral bounds of
  `A = H⁻¹ C D⁻¹ C'`.  If `nothing` and `chebyshev=true`, 15 power
  iterations estimate `b`.
"""
function solve_schur_gpu_neumann!(
    Δp, kkt::KKTSystemGPU;
    Kmax::Int=50, tol::Real=1e-10,
    chebyshev::Bool=false,
    spec_bounds=nothing
)
    T = eltype(Δp)
    H_inv = 1 ./ kkt.H_diag
    d_inv = 1 ./ kkt.d_diag

    # RHS = r̃1 + Wc * (r̃2 ./ d)
    kkt.rhs .= kkt.r̃1
    mul!(kkt.rhs, kkt.Wc, kkt.r̃2 .* d_inv, one(T), one(T))

    # Workspace (allocated once per call)
    u = similar(kkt.d_diag)    # m-vector scratch
    v = similar(Δp)            # n-vector scratch

    # b₀ = H⁻¹ rhs
    v .= H_inv .* kkt.rhs
    b0_norm = norm(v)

    sub_iter = 0
    if !chebyshev
        # --- Plain Neumann: Δp = Σₖ Aᵏ v₀ ---
        Δp .= v
        for k in 1:Kmax
            _apply_A!(v, v, kkt.Wc, d_inv, H_inv, u)
            Δp .+= v
            sub_iter = k
            norm(v) < tol * b0_norm && break
        end
    else
        # --- Chebyshev semi-iteration for B x = b₀ ---
        # Saad, "Iterative Methods", Algorithm 12.1 (direction form).
        a_spec, b_spec = _neumann_spec_bounds(kkt, H_inv, d_inv, u, v, spec_bounds)
        α = one(T) - b_spec    # λ_min(B)
        β = one(T) - a_spec    # λ_max(B)
        θ = (β + α) / 2
        δ = (β - α) / 2
        σ₁ = θ / δ

        d_cheb = similar(Δp)   # direction vector
        r = copy(v)             # r₀ = b₀  (x₀ = 0)
        Bw = similar(Δp)       # scratch for B·w

        # k = 0: d₀ = r₀/θ, x₁ = d₀
        d_cheb .= r ./ θ
        Δp .= d_cheb
        ρ = one(T) / σ₁

        for k in 1:Kmax
            _apply_A!(Bw, d_cheb, kkt.Wc, d_inv, H_inv, u)
            r .+= Bw .- d_cheb

            sub_iter = k
            norm(r) < tol * b0_norm && break

            ρ_new = one(T) / (2σ₁ - ρ)
            d_cheb .= ρ_new .* ((2 / θ) .* r .+ ρ .* d_cheb)
            ρ = ρ_new

            Δp .+= d_cheb
        end
    end

    return (Δp, true, sub_iter)
end

"""Apply A = H⁻¹ Wc D⁻¹ Wc' to `w`, result in `out`. `u` is m-vector scratch."""
@inline function _apply_A!(out, w, Wc, d_inv, H_inv, u)
    mul!(u, Wc', w)
    u .*= d_inv
    mul!(out, Wc, u)
    out .*= H_inv
end

"""Estimate spectral bounds (a, b) of A = H⁻¹ C D⁻¹ C' via power iteration."""
function _neumann_spec_bounds(kkt, H_inv, d_inv, u, v, spec_bounds)
    T = eltype(v)
    spec_bounds !== nothing && return T(spec_bounds[1]), T(spec_bounds[2])

    w = similar(v)
    w .= one(T) / sqrt(T(length(v)))
    λ_est = zero(T)
    for _ in 1:15
        _apply_A!(v, w, kkt.Wc, d_inv, H_inv, u)
        λ_est = norm(v)
        λ_est > 0 && (w .= v ./ λ_est)
    end
    b = min(T(0.999), λ_est * T(1.05))  # slight overestimate, capped < 1
    a = zero(T)
    return a, b
end

# -----------------------------------------------------------------------
# Preconditioned CG solve for the Schur complement:
#
#   S Δp = b,   S = H - C D⁻¹ C'  (SPD, n × n)
#
# Matvec:  S v = H v - Wc (D⁻¹ (Wc' v))    — 2 GEMVs, O(nm)
# Precond: M⁻¹ = H⁻¹                        — diagonal, O(n)
#
# Convergence: O(√κ) iterations where κ = κ(H⁻¹S) = 1/(1-ρ(A)).
# Compare: Neumann needs O(κ) = O(1/(1-ρ)) terms.
# When ρ(A) = 0.99, CG needs ~10 iters vs Neumann ~460.
# -----------------------------------------------------------------------

"""
    solve_schur_gpu_pcg!(Δp, kkt; Kmax=50, tol=1e-10)

Solve S Δp = rhs via H⁻¹-preconditioned CG.  Does NOT form the
dense S matrix — only uses H_diag, Wc, d_diag, rhs.

Cost: O(K nm) for K CG iterations, via 2K GEMVs through Wc (n × m).
Converges in O(√(1/(1-ρ(A)))) iterations.
"""
function solve_schur_gpu_pcg!(Δp, kkt::KKTSystemGPU;
    Kmax::Int=50, tol::Real=1e-10)
    T = eltype(Δp)
    H_diag = kkt.H_diag
    d_inv = 1 ./ kkt.d_diag

    # Preconditioner M = diag(S) = H - Σ,  Σ_j = Σ_i Wc[j,i]² / d[i]
    # This is the exact diagonal of the Schur complement.
    # κ(M⁻¹S) = (1+δ_off)/(1-δ_off) where δ_off measures off-diagonal coupling.
    Σ_diag = dropdims(sum(kkt.Wc .^ 2 .* d_inv'; dims=2); dims=2)
    M_diag = H_diag .- Σ_diag
    # Guard: clamp to small positive to avoid division by zero
    M_diag .= max.(M_diag, T(1e-14) .* H_diag)
    M_inv = 1 ./ M_diag

    # RHS = r̃1 + Wc * (r̃2 ./ d)
    kkt.rhs .= kkt.r̃1
    mul!(kkt.rhs, kkt.Wc, kkt.r̃2 .* d_inv, one(T), one(T))

    # Workspace (allocated once)
    u = similar(kkt.d_diag)    # m-vector scratch for matvec
    Sv = similar(Δp)           # n-vector scratch for S*v
    r = similar(Δp)            # residual
    z = similar(Δp)            # preconditioned residual M⁻¹ r
    p = similar(Δp)            # search direction

    # x₀ = M⁻¹ rhs (diagonal preconditioner solve as initial guess)
    Δp .= M_inv .* kkt.rhs

    # r₀ = b - S x₀
    _apply_S!(Sv, Δp, H_diag, kkt.Wc, d_inv, u)
    r .= kkt.rhs .- Sv

    b_norm = norm(kkt.rhs)
    sub_iter = 0
    if norm(r) < tol * b_norm
        return (Δp, true, sub_iter)
    end

    # z₀ = M⁻¹ r₀
    z .= M_inv .* r
    p .= z
    rz = dot(r, z)

    for k in 1:Kmax
        _apply_S!(Sv, p, H_diag, kkt.Wc, d_inv, u)
        pSp = dot(p, Sv)
        α = rz / pSp

        Δp .+= α .* p
        r .-= α .* Sv

        sub_iter = k
        norm(r) < tol * b_norm && break

        z .= M_inv .* r
        rz_new = dot(r, z)
        β = rz_new / rz
        rz = rz_new

        p .= z .+ β .* p
    end

    return (Δp, true, sub_iter)
end

"""Apply S v = H v - Wc D⁻¹ Wc' v. `u` is m-vector scratch."""
@inline function _apply_S!(out, v, H_diag, Wc, d_inv, u)
    mul!(u, Wc', v)
    u .*= d_inv
    mul!(out, Wc, u)
    out .= H_diag .* v .- out
end

# -----------------------------------------------------------------------
# Augmented quasi-definite system ((n+m) × (n+m)):
#   [ H    -Wc ] [Δp]   [r̃1]
#   [-Wc'   D  ] [Δλ] = [r̃2]
#
# H > 0 (diagonal), D > 0 (diagonal) ⟹ quasi-definite.
#
# Stored as sparse (lower triangle) for CUDSS LDLᵀ on GPU.
# Call init_aug_gpu! once to build sparsity pattern + symbolic analysis,
# then build_linsys_aug_gpu! + solve_augmented_gpu! each iteration.
# -----------------------------------------------------------------------

"""
    init_aug_gpu!(kkt; use_cudss=false)

Build the augmented system sparsity pattern from `c` and precompute
the off-diagonal nzval index map for fast per-iteration fills.
When `use_cudss=true`, also creates a `CudssSolver` and runs
symbolic analysis (reused across all subsequent factorizations).

Called once before the first Newton step when `mode=:aug`.

Lower triangle:  [ diag(H)        ]   (n × n)
                  [ -Wc'    diag(D)]   (m × n+m)

where Wc has the same sparsity as c (values change each iteration).
"""
function init_aug_gpu!(kkt::KKTSystemGPU; use_cudss::Bool=false)
    n, m = kkt.n, kkt.m
    N = n + m
    T = eltype(kkt.w)

    # Get c as sparse on CPU to read sparsity pattern
    c_cpu = kkt.c isa SparseMatrixCSC ? kkt.c : sparse(Array(kkt.c))

    # Build lower triangle:
    #   [ I_n   ] + [ 0   0 ] + [ 0   0 ]
    #   [       ]   [ c'  0 ]   [ 0  I_m]
    #
    # (2,1) block is c' shifted to rows n+1:n+m, embedded in N×N
    ct = copy(c_cpu')  # m × n SparseMatrixCSC
    # Pad to N×N: columns n+1:N are empty, rows shifted by n
    ct_colptr = vcat(ct.colptr, fill(ct.colptr[end], m))  # m extra empty columns
    ct_embed = SparseMatrixCSC(N, N, ct_colptr, ct.rowval .+ n, ones(T, nnz(ct)))

    # Full lower triangle = diagonal + embedded c'
    A_cpu = ct_embed + spdiagm(N, N, 0 => ones(T, N))

    # Precompute nzval index maps
    diag_nz = Vector{Int}(undef, N)
    Wc_nz = Vector{Tuple{Int,Int,Int}}(undef, nnz(ct))
    kw = 0
    for col in 1:N
        for idx in A_cpu.colptr[col]:(A_cpu.colptr[col+1]-1)
            row = A_cpu.rowval[idx]
            if row == col
                diag_nz[col] = idx
            elseif col <= n  # off-diagonal in (2,1) block
                kw += 1
                Wc_nz[kw] = (idx, col, row - n)  # (nzval_idx, good_j, agent_i)
            end
        end
    end
    kkt.aug_diag_nzidx = diag_nz
    kkt.aug_Wc_nzidx = Wc_nz

    if use_cudss
        # GPU path: CuSparseMatrixCSR + CUDSS solver with symbolic analysis
        A_gpu = CuSparseMatrixCSR(A_cpu)

        # IMPORTANT: CSC→CSR conversion reorders nzvals (column-major → row-major).
        # Recompute index maps from the CSR layout on CPU.
        A_csr_cpu = SparseMatrixCSC(A_cpu')  # A' in CSC ≡ A in CSR (same arrays)
        diag_nz = Vector{Int}(undef, N)
        Wc_nz_gpu = Vector{Tuple{Int,Int,Int}}(undef, nnz(ct))
        kw2 = 0
        for row in 1:N
            for idx in A_csr_cpu.colptr[row]:(A_csr_cpu.colptr[row+1]-1)
                col2 = A_csr_cpu.rowval[idx]
                if row == col2
                    diag_nz[row] = idx
                elseif col2 <= n  # off-diagonal: row > n, col2 ≤ n  → (2,1) block
                    kw2 += 1
                    Wc_nz_gpu[kw2] = (idx, col2, row - n)  # (nzval_idx, good_j, agent_i)
                end
            end
        end
        kkt.aug_diag_nzidx = diag_nz
        kkt.aug_Wc_nzidx = Wc_nz_gpu

        solver = CudssSolver(A_gpu, "S", 'L')
        cudss("analysis", solver, kkt.aug_sol, kkt.aug_rhs)
        kkt.aug_sparse = A_gpu
        kkt.aug_solver = solver

        # Precompute GPU index arrays for vectorized build_linsys_aug_gpu!
        kkt.aug_diag_nzidx_gpu = CuVector{Int}(diag_nz)
        nz_dst = Int[t[1] for t in Wc_nz_gpu]
        nz_j = Int[t[2] for t in Wc_nz_gpu]
        nz_i = Int[t[3] for t in Wc_nz_gpu]
        # Linear index into Wc (n × m column-major): Wc[j, i] → j + (i-1)*n
        nz_src = nz_j .+ (nz_i .- 1) .* n
        kkt.aug_Wc_nzidx_gpu = CuVector{Int}(nz_dst)
        kkt.aug_Wc_src_gpu = CuVector{Int}(nz_src)
    else
        # CPU path: SparseMatrixCSC, BunchKaufman at solve time
        kkt.aug_sparse = A_cpu
        kkt.aug_solver = nothing
    end
    return kkt
end

"""Backup: scalar-indexed build (correct for CPU, slow on GPU)."""
function build_linsys_aug_gpu_bak!(kkt::KKTSystemGPU)
    n, m = kkt.n, kkt.m
    nzv = nonzeros(kkt.aug_sparse)
    didx = kkt.aug_diag_nzidx

    # Diagonal: [H; D]
    @inbounds for j in 1:n
        nzv[didx[j]] = kkt.H_diag[j]
    end
    @inbounds for i in 1:m
        nzv[didx[n+i]] = kkt.d_diag[i]
    end

    # -Wc off-diagonal block
    Wc = kkt.Wc
    @inbounds for (idx, j, i) in kkt.aug_Wc_nzidx
        nzv[idx] = -Wc[j, i]
    end

    # RHS
    kkt.aug_rhs[1:n] .= kkt.r̃1
    kkt.aug_rhs[n+1:n+m] .= kkt.r̃2
    return kkt
end

"""
    build_linsys_aug_gpu!(kkt)

Fill numerical values into the pre-allocated augmented sparse matrix and RHS.
Uses GPU-vectorized scatter via precomputed CuVector index arrays — no scalar indexing.
"""
function build_linsys_aug_gpu!(kkt::KKTSystemGPU)
    n, m = kkt.n, kkt.m
    nzv = nonzeros(kkt.aug_sparse)

    # Diagonal: scatter [H_diag; d_diag] into nzvals
    diag_vals = vcat(kkt.H_diag, kkt.d_diag)
    nzv[kkt.aug_diag_nzidx_gpu] .= diag_vals

    # -Wc off-diagonal: gather from Wc by linear index, negate, scatter into nzvals
    nzv[kkt.aug_Wc_nzidx_gpu] .= .-view(kkt.Wc, kkt.aug_Wc_src_gpu)

    # RHS
    kkt.aug_rhs[1:n] .= kkt.r̃1
    kkt.aug_rhs[n+1:n+m] .= kkt.r̃2
    return kkt
end

"""
    solve_augmented_gpu!(Δp, Δλ, kkt)

Factorize and solve the augmented system. On GPU uses CUDSS LDLᵀ
(symbolic analysis reused from init_aug_gpu!); on CPU falls back
to BunchKaufman on a dense copy.
"""
function solve_augmented_gpu!(Δp, Δλ, kkt::KKTSystemGPU)
    n, m = kkt.n, kkt.m
    if kkt.aug_solver !== nothing
        # GPU: CUDSS LDLᵀ factorization + solve (reuses symbolic analysis)
        cudss("factorization", kkt.aug_solver, kkt.aug_sol, kkt.aug_rhs; asynchronous=false)
        cudss("solve", kkt.aug_solver, kkt.aug_sol, kkt.aug_rhs; asynchronous=false)
    else
        # CPU fallback: BunchKaufman on dense copy
        A_dense = Matrix(Symmetric(kkt.aug_sparse, :L))
        F = bunchkaufman!(A_dense)
        kkt.aug_sol .= F \ kkt.aug_rhs
    end
    Δp .= @view kkt.aug_sol[1:n]
    Δλ .= @view kkt.aug_sol[n+1:n+m]
    return (Δp, Δλ, true)
end

# -----------------------------------------------------------------------
# Back-substitution — batched
# -----------------------------------------------------------------------
function update_agent_step_gpu!(st::PDStateGPU, kkt::KKTSystemGPU, res)
    μ = st.μ
    μ_a = max(μ * st.μ_scale, st.μ_a_min)

    # Δλ = (r̃2 + Wc'Δp) ./ d  — Wc'Δp is a matvec: m-vector
    mul!(st.Δλ, kkt.Wc', st.Δp)
    st.Δλ .= (kkt.r̃2 .+ st.Δλ) ./ kkt.d_diag

    # Δs = Δp - c .* Δλ' - r3
    st.Δs .= st.Δp .- kkt.c .* st.Δλ' .- res.r3

    # Δx = (μ_a - x.*s - x.*Δs) ./ s  — agent complementarity
    st.Δx .= (μ_a .- st.x .* st.s .- st.x .* st.Δs) ./ st.s

    # Δq = (μ - p.*q - q.*Δp) ./ p  — price complementarity
    st.Δq .= (μ .- st.p .* st.q .- st.q .* st.Δp) ./ st.p
    return st
end

# -----------------------------------------------------------------------
# Step size — batched min reduction
# -----------------------------------------------------------------------
function pd_stepsize_gpu(st::PDStateGPU; τ=0.9995)
    T = eltype(st.p)
    _step_ratio(v, Δv) = minimum(ifelse.(Δv .< 0, .-v ./ Δv, T(Inf)))

    α = one(T)
    α = min(α, _step_ratio(st.p, st.Δp))
    α = min(α, _step_ratio(st.q, st.Δq))
    α = min(α, _step_ratio(st.λ, st.Δλ))
    α = min(α, _step_ratio(st.s, st.Δs))
    α = min(α, _step_ratio(st.x, st.Δx))
    return T(τ) * α
end

# -----------------------------------------------------------------------
# Update — batched
# -----------------------------------------------------------------------
function pd_update_gpu!(st::PDStateGPU, α)
    st.p .+= α .* st.Δp
    st.q .+= α .* st.Δq
    st.λ .+= α .* st.Δλ
    st.s .+= α .* st.Δs
    st.x .+= α .* st.Δx
end

# -----------------------------------------------------------------------
# Gap — batched
# -----------------------------------------------------------------------
function pd_gap_gpu(st::PDStateGPU)
    # Price complementarity: p∘q targets μ (n conditions)
    # Agent complementarity: x_i∘s_i targets μ·μ_scale (n×m conditions)
    # Weighted gap so that gap/ncomp ≈ μ at the central path
    gap = dot(st.p, st.q) + sum(st.x .* st.s) * st.μ_scale
    ncomp = st.n + st.n * st.m * st.μ_scale
    return gap, gap / ncomp
end

# -----------------------------------------------------------------------
# Newton step
# -----------------------------------------------------------------------
function pd_newton_gpu!(st::PDStateGPU, kkt::KKTSystemGPU; mode::Symbol=:schur)
    t_res = @elapsed res = pd_residuals_gpu(st, kkt)
    t_kkt = @elapsed update_kkt_gpu!(kkt, st, res)
    success = true
    t_build = 0.0
    t_solve = 0.0
    t_back = 0.0
    sub_iter = 0
    if mode == :aug
        t_build = @elapsed build_linsys_aug_gpu!(kkt)
        t_solve = @elapsed (_, _, success) = solve_augmented_gpu!(st.Δp, st.Δλ, kkt)
        if success
            t_back = @elapsed begin
                μ = st.μ
                μ_a = max(μ * st.μ_scale, st.μ_a_min)
                st.Δs .= st.Δp .- kkt.c .* st.Δλ' .- res.r3
                st.Δx .= (μ_a .- st.x .* st.s .- st.x .* st.Δs) ./ st.s
                st.Δq .= (μ .- st.p .* st.q .- st.q .* st.Δp) ./ st.p
            end
        end
    elseif mode == :neumann || mode == :neumann_cheb
        _cheb = mode == :neumann_cheb
        t_solve = @elapsed (_, success, sub_iter) = solve_schur_gpu_neumann!(st.Δp, kkt;
            chebyshev=_cheb)
        if success
            t_back = @elapsed update_agent_step_gpu!(st, kkt, res)
        end
    elseif mode == :pcg
        t_solve = @elapsed (_, success, sub_iter) = solve_schur_gpu_pcg!(st.Δp, kkt)
        if success
            t_back = @elapsed update_agent_step_gpu!(st, kkt, res)
        end
    else  # :schur (default)
        t_build = @elapsed build_linsys_schur_gpu!(kkt)
        t_solve = @elapsed (_, success) = solve_schur_gpu!(st.Δp, kkt)
        if success
            t_back = @elapsed update_agent_step_gpu!(st, kkt, res)
        end
    end
    return (res=res, success=success, sub_iter=sub_iter,
        t_residuals=t_res, t_update_kkt=t_kkt, t_build=t_build, t_solve=t_solve, t_backsub=t_back)
end

# -----------------------------------------------------------------------
# Main solver
# -----------------------------------------------------------------------
function pd_ipm_gpu(
    f;
    μ₀=1.0,
    maxiter=200,
    tol=1e-10,
    device=identity,
    mode::Symbol=:schur,
    μ_scale::Float64=1.0,
    μ_a_min::Float64=0.0,
    show_trace=true,
    show_every=1
)
    n, m = f.n, f.m

    _dev_str = device === identity ? "CPU" : "GPU"

    to_dev = device
    t_transfer = @elapsed begin
        c_dev = to_dev(Matrix{Float64}(f.c))
        w_dev = to_dev(Vector{Float64}(f.w))
        kkt = KKTSystemGPU(n, m, c_dev, w_dev)
        st = PDStateGPU(n, m; to_dev=device, μ_scale=μ_scale, μ_a_min=μ_a_min)
    end
    t_init = @elapsed pd_init_gpu!(st, kkt; μ₀=μ₀)

    if mode == :aug
        use_cudss = device !== identity
        t_aug_init = @elapsed init_aug_gpu!(kkt; use_cudss=use_cudss)
    end

    traj = []

    if show_trace
        _loghead = @sprintf("%5s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s",
            "k", "μ", "|mc|", "|r₂|", "|r₃|", "|c₁|", "|c₂|", "α", "time(s)")
        _w = length(_loghead)
        _sep = "-"^_w
        println(_sep)
        println(" "^max(0, (_w - 50) ÷ 2) * "Primal-Dual IPM (batched, device=$_dev_str, n=$n, m=$m)")
        println(_sep)
        @printf(" data transfer: %.4fs | init: %.4fs\n", t_transfer, t_init)
        println(_sep)
        println(_loghead)
        println(_sep)
    end

    _t0 = time()
    status = :max_iter
    for k in 1:maxiter
        nr = pd_newton_gpu!(st, kkt; mode=mode)
        res = nr.res

        if !nr.success
            show_trace && @printf("PosDefException at k=%d — returning last iterate\n", k)
            status = :posdef_fail
            break
        end

        α = pd_stepsize_gpu(st)
        gap, μ_avg = pd_gap_gpu(st)

        nr1 = norm(res.r1, Inf)
        nr2 = norm(res.r2, Inf)
        nr3 = norm(res.r3, Inf)
        nc1 = norm(res.c1, Inf)
        nc2 = norm(res.c2, Inf)

        _elapsed = time() - _t0
        if show_trace && k % show_every == 0
            @printf("%5d | %+10.3e | %10.3e | %10.3e | %10.3e | %10.3e | %10.3e | %10.4f | %10.4f\n",
                k, st.μ, nr1, nr2, nr3, nc1, nc2, α, _elapsed)
            @printf("%5s |- t_res=%.4f  t_kkt=%.4f  t_build=%.4f  t_solve=%.4f  t_back=%.4f  sub_iter=%d\n", "",
                nr.t_residuals, nr.t_update_kkt, nr.t_build, nr.t_solve, nr.t_backsub, nr.sub_iter)
        end

        push!(traj, (k=k, μ=st.μ, nr1=nr1, nr2=nr2, nr3=nr3, nc1=nc1, nc2=nc2, α=α, t=_elapsed))

        if nr1 < tol && nr2 < √tol
            show_trace && @printf("converged at k=%d, gap=%.2e\n", k, gap)
            status = :converged
            break
        end

        if st.μ < tol * tol
            show_trace && @printf("converged (μ < tol²) at k=%d, μ=%.2e, gap=%.2e\n", k, st.μ, gap)
            status = :converged
            break
        end

        pd_update_gpu!(st, α)
        _, μ_new = pd_gap_gpu(st)
        st.μ = (1 - α) * μ_new
    end

    return (st=st, traj=traj, kkt=kkt, status=status)
end

# -----------------------------------------------------------------------
# Predictor-Corrector helpers
# -----------------------------------------------------------------------

"""
    update_kkt_rhs_corrector_gpu!(kkt, st, res, Δp_aff, Δλ_aff, Δs_aff, Δx_aff, Δq_aff, σ)

Recompute only the RHS (r̃1, r̃2) for the corrector step.
The matrix (H, Wc, D) is unchanged from the predictor.

Corrector RHS replaces the complementarity targets:
  ξ_corr = (σμ_a - x∘s + x∘r3 - Δx_aff∘Δs_aff) / s
  r̃1_corr = -r1 + (σμ/p - q) + Σ ξ_corr
  r̃2_corr = -r2 - Σ c∘ξ_corr
"""
function update_kkt_rhs_corrector_gpu!(kkt, st, res,
    Δs_aff, Δx_aff, Δq_aff, σ)
    c = kkt.c
    μ = st.μ
    σμ = σ * μ
    σμ_a = max(σ * μ * st.μ_scale, st.μ_a_min)

    # ξ_corr = (σμ_a - x∘s + x∘r3 - Δx_aff∘Δs_aff) / s
    kkt.ξ .= (σμ_a .- st.x .* st.s .+ st.x .* res.r3 .- Δx_aff .* Δs_aff) ./ st.s

    # r̃1 = -r1 + (σμ/p - q) + sum(ξ, dims=2)
    kkt.r̃1 .= .-res.r1 .+ (σμ ./ st.p .- st.q) .+ dropdims(sum(kkt.ξ; dims=2); dims=2)

    # r̃2[i] = -r2[i] - dot(c[:,i], ξ[:,i])
    kkt.r̃2 .= .-res.r2 .- dropdims(sum(c .* kkt.ξ; dims=1); dims=1)

    return kkt
end

"""Back-substitution for a given (Δp, kkt, res) — writes into provided arrays."""
function backsub_gpu!(Δλ, Δs, Δx, Δq, Δp, st, kkt, res)
    μ_a = st.μ * st.μ_scale

    mul!(Δλ, kkt.Wc', Δp)
    Δλ .= (kkt.r̃2 .+ Δλ) ./ kkt.d_diag

    Δs .= Δp .- kkt.c .* Δλ' .- res.r3

    Δx .= (μ_a .- st.x .* st.s .- st.x .* Δs) ./ st.s

    Δq .= (st.μ .- st.p .* st.q .- st.q .* Δp) ./ st.p
end

"""Back-sub for corrector: uses kkt.ξ which already has the corrector RHS."""
function backsub_corrector_gpu!(Δλ, Δs, Δx, Δq, Δp, st, kkt, res, σ)
    σμ = σ * st.μ
    σμ_a = max(σ * st.μ * st.μ_scale, st.μ_a_min)

    mul!(Δλ, kkt.Wc', Δp)
    Δλ .= (kkt.r̃2 .+ Δλ) ./ kkt.d_diag

    Δs .= Δp .- kkt.c .* Δλ' .- res.r3

    # Corrector Δx uses ξ_corr which already includes -Δx_aff∘Δs_aff
    Δx .= kkt.ξ .- st.x ./ st.s .* Δs

    Δq .= (σμ .- st.p .* st.q .- st.q .* Δp) ./ st.p
end

# -----------------------------------------------------------------------
# Predictor-Corrector main solver (Mehrotra-type)
# -----------------------------------------------------------------------
"""
    pd_ipm_gpu_pc(f; kwargs...)

Mehrotra predictor-corrector variant.  Each iteration:
1. Predictor (affine): solve with μ=0, get αᵃᶠᶠ
2. Centering parameter: σ = (μᵃᶠᶠ/μ)³
3. Corrector: solve with σμ and cross-term Δx_aff∘Δs_aff, same factorization
"""
function pd_ipm_gpu_pc(
    f;
    μ₀=1.0,
    maxiter=200,
    tol=1e-10,
    device=identity,
    mode::Symbol=:schur,
    μ_scale::Float64=1.0,
    μ_a_min::Float64=0.0,
    show_trace=true,
    show_every=1
)
    n, m = f.n, f.m
    _dev_str = device === identity ? "CPU" : "GPU"

    to_dev = device
    t_transfer = @elapsed begin
        c_dev = to_dev(Matrix{Float64}(f.c))
        w_dev = to_dev(Vector{Float64}(f.w))
        kkt = KKTSystemGPU(n, m, c_dev, w_dev)
        st = PDStateGPU(n, m; to_dev=device, μ_scale=μ_scale, μ_a_min=μ_a_min)
    end
    t_init = @elapsed pd_init_gpu!(st, kkt; μ₀=μ₀)

    if mode == :aug
        use_cudss = device !== identity
        t_aug_init = @elapsed init_aug_gpu!(kkt; use_cudss=use_cudss)
    end

    # Affine step workspace (predictor)
    Δp_aff = similar(st.p)
    Δλ_aff = similar(st.λ)
    Δs_aff = similar(st.s)
    Δx_aff = similar(st.x)
    Δq_aff = similar(st.q)

    traj = []

    if show_trace
        _loghead = @sprintf("%5s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s",
            "k", "μ", "|mc|", "|r₂|", "|r₃|", "|c₁|", "|c₂|", "α", "time(s)")
        _w = length(_loghead)
        _sep = "-"^_w
        println(_sep)
        println(" "^max(0, (_w - 50) ÷ 2) * "Primal-Dual IPM-PC (Mehrotra, device=$_dev_str, n=$n, m=$m)")
        println(_sep)
        @printf(" data transfer: %.4fs | init: %.4fs\n", t_transfer, t_init)
        println(_sep)
        println(_loghead)
        println(_sep)
    end

    T = eltype(st.p)
    _step_ratio(v, Δv) = minimum(ifelse.(Δv .< 0, .-v ./ Δv, T(Inf)))

    _t0 = time()
    status = :max_iter
    for k in 1:maxiter
        # ============================================================
        # Step 0: residuals + KKT workspace (shared by predictor & corrector)
        # ============================================================
        t_res = @elapsed res = pd_residuals_gpu(st, kkt)
        t_kkt = @elapsed update_kkt_gpu!(kkt, st, res)

        # ============================================================
        # Step 1: PREDICTOR (affine direction, μ=0)
        # ============================================================
        c_dev = kkt.c
        kkt.ξ .= (.-st.x .* st.s .+ st.x .* res.r3) ./ st.s
        kkt.r̃1 .= .-res.r1 .+ (.-st.q) .+ dropdims(sum(kkt.ξ; dims=2); dims=2)
        kkt.r̃2 .= .-res.r2 .- dropdims(sum(c_dev .* kkt.ξ; dims=1); dims=1)

        t_pred = @elapsed begin
            _solve_schur_dispatch!(Δp_aff, kkt, mode)
            mul!(Δλ_aff, kkt.Wc', Δp_aff)
            Δλ_aff .= (kkt.r̃2 .+ Δλ_aff) ./ kkt.d_diag
            Δs_aff .= Δp_aff .- kkt.c .* Δλ_aff' .- res.r3
            Δx_aff .= (.-st.x .* st.s .- st.x .* Δs_aff) ./ st.s
            Δq_aff .= (.-st.p .* st.q .- st.q .* Δp_aff) ./ st.p
        end

        # ============================================================
        # Step 2: Centering parameter σ (Mehrotra heuristic)
        # ============================================================
        α_aff = one(T)
        α_aff = min(α_aff, _step_ratio(st.p, Δp_aff))
        α_aff = min(α_aff, _step_ratio(st.q, Δq_aff))
        α_aff = min(α_aff, _step_ratio(st.λ, Δλ_aff))
        α_aff = min(α_aff, _step_ratio(st.s, Δs_aff))
        α_aff = min(α_aff, _step_ratio(st.x, Δx_aff))
        α_aff *= T(0.9995)  # back off from boundary

        _, μ_cur = pd_gap_gpu(st)
        gap_aff = dot(st.p .+ α_aff .* Δp_aff, st.q .+ α_aff .* Δq_aff) +
                  sum((st.x .+ α_aff .* Δx_aff) .* (st.s .+ α_aff .* Δs_aff)) * st.μ_scale
        ncomp = st.n + st.n * st.m * st.μ_scale
        μ_aff = max(gap_aff / ncomp, T(1e-300))  # guard against negative gap
        σ = clamp((μ_aff / μ_cur)^3, T(1e-6), T(0.5))

        # ============================================================
        # Step 3: CORRECTOR (centering + cross-term)
        # ============================================================
        σμ = σ * st.μ
        σμ_a = max(σ * st.μ * st.μ_scale, st.μ_a_min)
        kkt.ξ .= (σμ_a .- st.x .* st.s .+ st.x .* res.r3 .- Δx_aff .* Δs_aff) ./ st.s
        kkt.r̃1 .= .-res.r1 .+ (σμ ./ st.p .- st.q) .+ dropdims(sum(kkt.ξ; dims=2); dims=2)
        kkt.r̃2 .= .-res.r2 .- dropdims(sum(c_dev .* kkt.ξ; dims=1); dims=1)

        t_corr = @elapsed begin
            _solve_schur_dispatch!(st.Δp, kkt, mode)
            mul!(st.Δλ, kkt.Wc', st.Δp)
            st.Δλ .= (kkt.r̃2 .+ st.Δλ) ./ kkt.d_diag
            st.Δs .= st.Δp .- kkt.c .* st.Δλ' .- res.r3
            st.Δx .= kkt.ξ .- kkt.W .* st.Δs
            st.Δq .= (σμ .- st.p .* st.q .- st.q .* st.Δp) ./ st.p
        end

        # ============================================================
        # Step 4: Step size + update
        # ============================================================
        α = pd_stepsize_gpu(st)

        # Safeguard: if corrector step is worse than predictor, use predictor
        if α < T(0.1) * α_aff
            st.Δp .= Δp_aff
            st.Δλ .= Δλ_aff
            st.Δs .= Δs_aff
            st.Δx .= Δx_aff
            st.Δq .= Δq_aff
            α = α_aff
        end

        gap, μ_avg = pd_gap_gpu(st)

        nr1 = norm(res.r1, Inf)
        nr2 = norm(res.r2, Inf)
        nr3 = norm(res.r3, Inf)
        nc1 = norm(res.c1, Inf)
        nc2 = norm(res.c2, Inf)

        _elapsed = time() - _t0
        if show_trace && k % show_every == 0
            @printf("%5d | %+10.3e | %10.3e | %10.3e | %10.3e | %10.3e | %10.3e | %10.4f | %10.4f\n",
                k, st.μ, nr1, nr2, nr3, nc1, nc2, α, _elapsed)
            @printf("%5s |- t_res=%.4f  t_kkt=%.4f  t_pred=%.4f  t_corr=%.4f  σ=%.4f  α_aff=%.4f\n", "",
                t_res, t_kkt, t_pred, t_corr, σ, α_aff)
        end

        push!(traj, (k=k, μ=st.μ, nr1=nr1, nr2=nr2, nr3=nr3, nc1=nc1, nc2=nc2,
            α=α, σ=σ, α_aff=α_aff, t=_elapsed))

        if nr1 < tol && nr2 < √tol
            show_trace && @printf("converged at k=%d, gap=%.2e\n", k, gap)
            status = :converged
            break
        end

        # Numerical convergence: μ too small for further progress
        if st.μ < tol * tol
            show_trace && @printf("converged (μ < tol²) at k=%d, μ=%.2e, gap=%.2e\n", k, st.μ, gap)
            status = :converged
            break
        end

        pd_update_gpu!(st, α)
        _, μ_new = pd_gap_gpu(st)
        st.μ = (one(T) - α + α * σ) * μ_cur  # μ_new ≈ (1 - α(1-σ)) μ
    end

    return (st=st, traj=traj, kkt=kkt, status=status)
end

"""Dispatch the Schur complement solve by mode (shared by predictor & corrector)."""
function _solve_schur_dispatch!(Δp, kkt, mode::Symbol)
    if mode == :pcg
        solve_schur_gpu_pcg!(Δp, kkt)
    elseif mode == :neumann || mode == :neumann_cheb
        solve_schur_gpu_neumann!(Δp, kkt; chebyshev=(mode == :neumann_cheb))
    elseif mode == :aug
        # aug already has Δp,Δλ in aug_sol — caller handles separately
        error("aug mode: use solve_augmented_gpu! directly")
    else  # :schur
        build_linsys_schur_gpu!(kkt)
        solve_schur_gpu!(Δp, kkt)
    end
end

# -----------------------------------------------------------------------
# Extract allocation
# -----------------------------------------------------------------------
function pd_allocation_gpu!(f, st::PDStateGPU, kkt::KKTSystemGPU)
    f.p .= Array(st.p)
    for i in 1:f.m
        f.x[:, i] .= Array(st.x[:, i])
    end
    f.val_u .= dropdims(sum(Array(kkt.c) .* f.x; dims=1); dims=1)
    f.sumx .= dropdims(sum(f.x; dims=2); dims=2)
    return f
end
