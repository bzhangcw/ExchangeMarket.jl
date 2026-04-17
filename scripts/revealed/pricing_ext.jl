using JuMP, MosekTools, Gurobi
using LinearAlgebra
using Statistics: mean
using Printf
import MathOptInterface as MOI

"""
    solve_pricing_shor_fix_σ(Ξ, u, σ; verbose=false)

Solve the pricing problem with fixed σ via Shor SDP relaxation.

The problem (for fixed σ):
    max_q  Σ_k ⟨q, ν_k⟩ / ⟨q, f_k⟩
where q = exp(y), f_k = p_k^{-σ}, ν_k = u_k ⊙ f_k.

Reformulation (fc2): introduce ρ_k = 1/⟨q, f_k⟩
    max_{q, ρ}  Σ_k ρ_k ⟨q, ν_k⟩
    s.t.        ρ_k ⟨q, f_k⟩ ≤ 1,  ∀k
                q ≥ 0, ⟨q, 1⟩ = 1

Shor relaxation: lift w = (q; ρ), W = w w^T, drop rank-1.
    Bilinear terms ρ_k q_j → W[j, n+k]

Returns: (q_opt, obj_upper, obj_rounded, status)
    - obj_upper: SDP upper bound
    - obj_rounded: feasible objective from rounded solution
"""
function solve_pricing_shor_fix_σ(
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T},
    σ::T;
    verbose=false
) where T

    K = length(Ξ)
    n = length(Ξ[1][1])
    d = n + K  # dimension of w = (q; ρ)

    # Precompute f_k = p_k^{-σ} and ν_k = u_k ⊙ f_k, normalized per k
    # The ratio ⟨q,ν_k⟩/⟨q,f_k⟩ is invariant to scaling f_k → f_k/s_k
    # (since ν_k = u_k ⊙ f_k scales the same way)
    f = zeros(T, K, n)
    ν = zeros(T, K, n)
    for k in 1:K
        p_k = Ξ[k][1]
        f_raw = p_k .^ (-σ)
        s = sum(f_raw)  # normalize so entries are O(1)
        f[k, :] = f_raw ./ s
        ν[k, :] = u[k, :] .* f[k, :]  # ν_k = u_k ⊙ (f_k / s)
    end

    # Build SDP
    md = Model(MosekTools.Optimizer)
    verbose || set_silent(md)

    # Variables: w ∈ R^d, W ∈ S^d
    @variable(md, w[1:d] >= 0)
    @variable(md, W[1:d, 1:d], Symmetric)

    # PSD constraint: [1 w'; w W] ≽ 0
    @constraint(md, psd,
        [1 w'; w W] in PSDCone()
    )

    # Simplex for q: Σ_j w[j] = 1 (first n entries)
    # @constraint(md, simplex, sum(w[j] for j in 1:n) == 1)

    # Bilinear constraint (relaxed): ρ_k ⟨q, f_k⟩ ≤ 1
    # = Σ_j f[k,j] * W[j, n+k] ≤ 1
    for k in 1:K
        @constraint(md, sum(f[k, j] * W[j, n+k] for j in 1:n) <= 1)
    end

    # Diagonal bounds: W_{jj} ≤ 1 for q, W_{n+k,n+k} ≤ M_k^2 for ρ
    for j in 1:n
        @constraint(md, W[j, j] <= 1.0)
    end
    for k in 1:K
        M_k = 1.0 / minimum(f[k, :])  # f already normalized, so this is O(n)
        @constraint(md, W[n+k, n+k] <= min(1e3, M_k^2))
    end

    # Objective: Σ_k Σ_j ν[k,j] * W[j, n+k]
    @objective(md, Max, sum(ν[k, j] * W[j, n+k] for j in 1:n for k in 1:K))

    optimize!(md)

    status = termination_status(md)
    obj_upper = objective_value(md)

    w_val = value.(w)
    W_val = value.(W)

    # Extract q from w (first n entries)
    q_sdp = max.(w_val[1:n], 0.0)
    q_sdp ./= sum(q_sdp)  # re-normalize

    # Compute rounded feasible objective: Σ_k ⟨q, ν_k⟩ / ⟨q, f_k⟩
    obj_rounded = 0.0
    for k in 1:K
        obj_rounded += dot(q_sdp, ν[k, :]) / dot(q_sdp, f[k, :])
    end

    # Recover y = log(q) for compatibility
    y_opt = log.(max.(q_sdp, 1e-15))

    γ_new = produce_gamma(Ξ, y_opt, σ)

    # Check rank of W
    eigs = eigvals(Symmetric(W_val))
    rank_W = sum(eigs .> 1e-6 * maximum(eigs))

    if verbose
        println("SDP Shor relaxation (fixed σ=$σ):")
        println("  Status:      $status")
        println("  Upper bound: $obj_upper")
        println("  Rounded obj: $obj_rounded")
        println("  Gap:         $(obj_upper - obj_rounded)")
        println("  rank(W):     $rank_W")
    end

    return y_opt, σ, γ_new, obj_upper, obj_rounded, rank_W
end

"""
    solve_pricing_expcone(Ξ, u; verbose=false)

Solve the pricing problem via exponential-cone relaxation (convex):
    max_{γ,v,A,y,σ}  Σ_k u_k^T γ_k
    s.t.  v_k = y - σ log(p_k) - A_k 1,   ∀k
          (v_{k,j}, 1, γ_{k,j}) ∈ K_exp,   ∀k,j
          1^T γ_k = 1,                      ∀k

Since 1^T γ_k = 1, we can freely shift u_k → u_k - max_j(u_{k,j}) · 1
without changing the optimizer (objective shifts by a constant).
This makes u_k ≤ 0, ensuring the cone constraints bind at optimum,
so the relaxation is exact (not just a relaxation).

Returns: (y_opt, σ_opt, γ_new, obj_val)
"""
function solve_pricing_expcone(
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    σ_bounds::Tuple{T,T}=(-0.95, 30.0),
    y_bound::T=100.0,
    verbose=false
) where T

    K = length(Ξ)
    n = length(Ξ[1][1])

    # Normalize u_k ≤ 0: shift by max_j(u_{k,j}) so cone is tight
    u_shift = [maximum(u[k, :]) for k in 1:K]
    u_norm = copy(u)
    for k in 1:K
        u_norm[k, :] .-= u_shift[k]
    end
    obj_offset = sum(u_shift)  # constant to add back

    # Precompute log prices
    log_p = [log.(Ξ[k][1]) for k in 1:K]

    md = Model(MosekTools.Optimizer)
    verbose || set_silent(md)

    # Variables
    @variable(md, y[1:n])
    @variable(md, σ_bounds[1] <= σ <= σ_bounds[2])
    @variable(md, γ[1:K, 1:n] >= 0)
    @variable(md, v[1:K, 1:n])
    @variable(md, A[1:K])

    # Bound y for numerical stability
    @constraint(md, [j = 1:n], -y_bound <= y[j] <= y_bound)

    # Linking constraint: v_k = y - σ log(p_k) - A_k 1
    for k in 1:K
        for j in 1:n
            @constraint(md, v[k, j] == y[j] - σ * log_p[k][j] - A[k])
        end
    end

    # Exponential cone: (v_{k,j}, 1, γ_{k,j}) ∈ K_exp
    # MOI convention: K_exp = {(x,y,z) : y exp(x/y) ≤ z, y > 0}
    # We need γ_{k,j} ≥ exp(v_{k,j}), i.e., (v_{k,j}, 1, γ_{k,j}) ∈ K_exp
    for k in 1:K
        for j in 1:n
            @constraint(md, [v[k, j], 1, γ[k, j]] in MOI.ExponentialCone())
        end
    end

    # Simplex: 1^T γ_k = 1
    for k in 1:K
        @constraint(md, sum(γ[k, j] for j in 1:n) == 1)
    end

    # Objective: max Σ_k u_norm_k^T γ_k (+ obj_offset constant)
    @objective(md, Max, sum(u_norm[k, j] * γ[k, j] for k in 1:K for j in 1:n))

    optimize!(md)

    status = termination_status(md)
    obj_val = objective_value(md) + obj_offset

    y_opt = value.(y)
    σ_opt = value(σ)
    γ_opt = value.(γ)

    # Recompute γ from softmax for consistency
    γ_new = produce_gamma(Ξ, y_opt, σ_opt)
    obj_softmax = sum(dot(u[k, :], γ_new[k, :]) for k in 1:K)

    # Check tightness: compare cone γ vs softmax γ
    if verbose
        gap = maximum(abs.(γ_opt .- γ_new))
        println("Exp-cone pricing:")
        println("  Status:       $status")
        println("  Objective (cone):    $obj_val")
        println("  Objective (softmax): $obj_softmax")
        println("  σ:            $σ_opt")
        println("  u_shift:      $u_shift")
        println("  Tightness gap (max |γ_cone - γ_softmax|): $gap")
    end

    return y_opt, σ_opt, γ_new, obj_softmax
end


"""
    solve_pricing_mip(Ξ, u; L=30, B=50.0, σ_bounds=(-0.95, 30.0), time_limit=60.0, verbose=false)

Solve the pricing problem to global optimality via MIP with piecewise-linear
approximation of the exponential. Uses Gurobi with SOS2 constraints.

The logits θ_{k,j} = y_j - σ log p_{k,j} are affine in (y, σ).
The exponential a_{k,j} ≈ exp(θ_{k,j}) is piecewise-linear (SOS2).
Normalization γ_{k,j} = a_{k,j} / D_k uses Gurobi's bilinear constraints.

Returns: (y, σ, γ, obj)
"""
function solve_pricing_mip(
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    L::Int=30,
    B::T=50.0,
    σ_bounds::Tuple{T,T}=(-0.95, 30.0),
    time_limit::T=60.0,
    verbose=false
) where T

    K = length(Ξ)
    n = length(Ξ[1][1])

    log_p = [log.(Ξ[k][1]) for k in 1:K]

    # Compute logit bounds
    θ_L = -B - σ_bounds[2] * maximum(maximum(log_p[k]) for k in 1:K)
    θ_U = B - σ_bounds[1] * minimum(minimum(log_p[k]) for k in 1:K)
    # Clamp to avoid numerical issues
    θ_L = max(θ_L, -B * 2)
    θ_U = min(θ_U, B * 2)

    # Breakpoints for piecewise-linear exp
    breakpoints = range(θ_L, θ_U, length=L)

    md = Model(() -> Gurobi.Optimizer())
    if !verbose
        set_silent(md)
    end
    set_attribute(md, "TimeLimit", time_limit)
    set_attribute(md, "NonConvex", 2)

    # Variables
    @variable(md, -B <= y[1:n] <= B)
    @variable(md, σ_bounds[1] <= σ <= σ_bounds[2])
    @variable(md, θ[1:K, 1:n])
    @variable(md, a[1:K, 1:n] >= 0)
    @variable(md, D[1:K] >= 1e-10)
    @variable(md, γ_var[1:K, 1:n] >= 0)
    @variable(md, λ[1:K, 1:n, 1:L] >= 0)

    # Logit constraints: θ_{k,j} = y_j - σ log p_{k,j}
    for k in 1:K, j in 1:n
        @constraint(md, θ[k, j] == y[j] - σ * log_p[k][j])
    end

    # Piecewise-linear exp via SOS2
    for k in 1:K, j in 1:n
        # Convexity: Σ_l λ_{k,j,l} = 1
        @constraint(md, sum(λ[k, j, l] for l in 1:L) == 1)
        # θ interpolation
        @constraint(md, θ[k, j] == sum(λ[k, j, l] * breakpoints[l] for l in 1:L))
        # a interpolation
        @constraint(md, a[k, j] == sum(λ[k, j, l] * exp(breakpoints[l]) for l in 1:L))
        # SOS2
        @constraint(md, [λ[k, j, l] for l in 1:L] in MOI.SOS2(collect(1.0:L)))
    end

    # Denominator
    for k in 1:K
        @constraint(md, D[k] == sum(a[k, j] for j in 1:n))
    end

    # Normalization: γ_{k,j} * D_k = a_{k,j} (bilinear, handled by Gurobi NonConvex=2)
    for k in 1:K
        @constraint(md, sum(γ_var[k, j] for j in 1:n) == 1)
        for j in 1:n
            @constraint(md, γ_var[k, j] * D[k] == a[k, j])
        end
    end

    # Objective
    @objective(md, Max, sum(u[k, j] * γ_var[k, j] for k in 1:K for j in 1:n))

    optimize!(md)

    status = termination_status(md)
    obj_mip = objective_value(md)

    y_opt = value.(y)
    σ_opt = value(σ)

    # Recompute γ from softmax for consistency
    γ_new = produce_gamma(Ξ, y_opt, σ_opt)
    obj_softmax = sum(dot(u[k, :], γ_new[k, :]) for k in 1:K)

    if verbose
        gap_mip = maximum(abs.(value.(γ_var) .- γ_new))
        println("MIP pricing:")
        println("  Status:       $status")
        println("  Objective (MIP):     $obj_mip")
        println("  Objective (softmax): $obj_softmax")
        println("  σ:            $σ_opt")
        println("  Breakpoints:  $L on [$(round(θ_L,digits=1)), $(round(θ_U,digits=1))]")
        println("  Gap (max |γ_mip - γ_softmax|): $gap_mip")
    end

    return y_opt, σ_opt, γ_new, obj_softmax
end


"""
    solve_pricing_admm(Ξ, u; ρ=1.0, max_iters=500, tol=1e-8, verbose=false)

Linearized ADMM for the pricing problem. Consensus formulation:
    max  Σ_k ⟨u_k, softmax(y_k - σ_k log p_k)⟩
    s.t. y_k = y, σ_k = σ,  ∀k

Each local step fixes γ_k = u_k/‖u_k‖₁ and inverts the softmax
to get y_{k,j} = L_{k,j} + σ_k ℓ_{k,j}, choosing σ_k by proximal
least-squares (closed-form scalar problem).

Returns: (y, σ, γ, obj, history)
"""
function solve_pricing_admm(
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    ρ::T=1.0,
    max_iters::Int=500,
    tol::T=1e-8,
    verbose=false
) where T

    K = length(Ξ)
    n = length(Ξ[1][1])

    # Precompute log-price ratios ℓ_{k,j} = log(p_{k,j}/p_{k,1})
    log_p = [log.(Ξ[k][1]) for k in 1:K]
    ℓ = [log_p[k][j] - log_p[k][1] for k in 1:K, j in 2:n]  # K × (n-1)

    # Precompute L_{k,j} = log(u_{k,j}/u_{k,1})
    L = zeros(T, K, n - 1)
    valid_k = trues(K)
    for k in 1:K
        if any(u[k, :] .<= 1e-15)
            valid_k[k] = false
            continue
        end
        for j in 2:n
            L[k, j-1] = log(u[k, j] / u[k, 1])
        end
    end

    # Initialize consensus variables
    y = zeros(T, n)
    σ = one(T)

    # Local copies
    y_loc = zeros(T, K, n)
    σ_loc = ones(T, K)

    # Dual variables (scaled form: μ_k/ρ, ν_k/ρ)
    μ_y = zeros(T, K, n)   # dual for y_k = y
    ν_σ = zeros(T, K)      # dual for σ_k = σ

    history = Dict(
        :obj => T[],
        :primal_res => T[],
        :dual_res => T[]
    )

    for iter in 1:max_iters
        y_old = copy(y)
        σ_old = σ

        # --- Step 1: Local updates (parallel, closed-form) ---
        for k in 1:K
            if !valid_k[k]
                y_loc[k, :] .= y .- μ_y[k, :]
                σ_loc[k] = σ - ν_σ[k]
                continue
            end

            # Augmented targets
            ȳ = y .- μ_y[k, :]   # y - μ_k/ρ
            σ̄ = σ - ν_σ[k]       # σ - ν_k/ρ

            # Solve for σ_k: min Σ_{j≥2} (L_{k,j} + σ_k ℓ_{k,j} - ȳ_j)² + (σ_k - σ̄)²
            num = zero(T)
            den = one(T)  # from (σ_k - σ̄)² term
            for j in 1:(n-1)
                num += ℓ[k, j] * (ȳ[j+1] - L[k, j])
                den += ℓ[k, j]^2
            end
            num += σ̄
            σ_loc[k] = num / den

            # Set y_k from inversion: y_{k,1} = 0, y_{k,j} = L_{k,j} + σ_k ℓ_{k,j}
            y_loc[k, 1] = 0.0
            for j in 1:(n-1)
                y_loc[k, j+1] = L[k, j] + σ_loc[k] * ℓ[k, j]
            end
        end

        # --- Step 2: Consensus (averaging) ---
        for j in 1:n
            y[j] = mean(y_loc[k, j] + μ_y[k, j] for k in 1:K)
        end
        σ = mean(σ_loc[k] + ν_σ[k] for k in 1:K)

        # --- Step 3: Dual update ---
        for k in 1:K
            for j in 1:n
                μ_y[k, j] += y_loc[k, j] - y[j]
            end
            ν_σ[k] += σ_loc[k] - σ
        end

        # --- Convergence check ---
        primal_res = sqrt(
            sum((y_loc[k, j] - y[j])^2 for k in 1:K for j in 1:n) +
            sum((σ_loc[k] - σ)^2 for k in 1:K)
        )
        dual_res = ρ * sqrt(
            K * sum((y[j] - y_old[j])^2 for j in 1:n) +
            K * (σ - σ_old)^2
        )

        γ = produce_gamma(Ξ, y, σ)
        obj = sum(dot(u[k, :], γ[k, :]) for k in 1:K)

        push!(history[:obj], obj)
        push!(history[:primal_res], primal_res)
        push!(history[:dual_res], dual_res)

        if verbose && (iter <= 5 || iter % 50 == 0 || max(primal_res, dual_res) < tol)
            @printf("  ADMM iter %4d: obj=%.6f  σ=%.4f  r_p=%.2e  r_d=%.2e\n",
                iter, obj, σ, primal_res, dual_res)
        end

        if max(primal_res, dual_res) < tol
            verbose && println("  ✓ ADMM converged at iter $iter")
            break
        end
    end

    γ = produce_gamma(Ξ, y, σ)
    obj = sum(dot(u[k, :], γ[k, :]) for k in 1:K)

    return y, σ, γ, obj, history
end

