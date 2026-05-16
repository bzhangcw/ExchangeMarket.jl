# Stochastic Frank-Wolfe / gradient boosting variant for the surrogate-fitting master.
#
#   minimize  F(h) = (1/K) Σ_k ‖P_k g_k - h(p_k)‖_∞   over   h ∈ conv(H)
#
# Each FW step:
#   1. Sample mini-batch S ⊂ [K] (full batch if batch_size == 0 or ≥ K).
#   2. Subgradient r_k = sign(target_k - h(p_k))[j*] · e_{j*} for k ∈ S.
#   3. LMO ≡ existing pricing solver:  γ_ℓ = argmax_γ Σ_k ⟨r_k, γ(p_k)⟩.
#   4. Update  h_{ℓ+1} = (1 - η_ℓ) h_ℓ + η_ℓ γ_ℓ,  η_ℓ = 2/(ℓ+2).
#
# Reuses pricing.jl: `solve_pricing`, `solve_pricing_dual_lp`,
# `recover_ces_params`, `add_to_gamma!`, `add_to_market!`.

using LinearAlgebra
using Random
using FrankWolfe

# pricing.jl is included by setup.jl; this file is included after setup.jl.

"""
    compute_h_at_prices(γ_ref, w)

Return matrix H of size (K, n) with rows h(p_k) = Σ_t w_t γ_t(p_k).
"""
function compute_h_at_prices(γ_ref::Ref{Array{T,3}}, w::AbstractVector{T}) where {T}
    γ = γ_ref[]
    m, K, n = size(γ)
    @assert length(w) == m "w length $(length(w)) ≠ m $(m) in γ"
    H = zeros(T, K, n)
    @inbounds for t in 1:m
        wt = w[t]
        for k in 1:K, j in 1:n
            H[k, j] += wt * γ[t, k, j]
        end
    end
    return H
end

"""
    linfty_subgradient(r)

A subgradient of x ↦ ‖x‖_∞ at x = r: returns sign(r[j*]) e_{j*} where
j* = argmax_j |r_j|. The result satisfies ‖s‖_1 = 1 and ⟨s, r⟩ = ‖r‖_∞.
"""
function linfty_subgradient(r::AbstractVector{T}) where {T}
    j_star = argmax(abs.(r))
    s = zeros(T, length(r))
    s[j_star] = sign(r[j_star])
    return s
end

"""
    compute_residual_subgrad(Ξ, γ_ref, w, batch)

Return u ∈ ℝ^{K×n} with rows u[k,:] = ∂‖target_k - h_ℓ(p_k)‖_∞ for k ∈ batch
(zero outside batch). This is the LMO direction: maximize Σ_k ⟨u_k, γ(p_k)⟩.
"""
function compute_residual_subgrad(Ξ, γ_ref::Ref{Array{T,3}},
    w::AbstractVector{T}, batch::AbstractVector{Int}) where {T}
    K = length(Ξ)
    n = length(Ξ[1][1])
    H = compute_h_at_prices(γ_ref, w)
    u = zeros(T, K, n)
    for k in batch
        p_k, g_k = Ξ[k]
        target = p_k .* g_k
        r = target .- @view H[k, :]
        u[k, :] .= linfty_subgradient(r)
    end
    return u
end

"""
    evaluate_train_loss(Ξ, γ_ref, w)

Mean L∞ residual over Ξ:  (1/K) Σ_k ‖P_k g_k - h(p_k)‖_∞.
"""
function evaluate_train_loss(Ξ, γ_ref::Ref{Array{T,3}}, w::AbstractVector{T}) where {T}
    K = length(Ξ)
    H = compute_h_at_prices(γ_ref, w)
    total = zero(T)
    for k in 1:K
        p_k, g_k = Ξ[k]
        target = p_k .* g_k
        total += norm(target .- @view(H[k, :]), Inf)
    end
    return total / K
end

"""
    fw_step_size(iter; rule=:diminishing)

Step size schedule.  :diminishing → η_ℓ = 2/(ℓ+2).
"""
function fw_step_size(iter::Int; rule::Symbol=:diminishing)
    if rule === :diminishing
        return 2.0 / (iter + 2)
    else
        error("Unknown FW step rule: $rule")
    end
end

"""
    run_method_tracked_fw(name, kwargs, Ξ_train, Ξ_test=nothing; verbose=true)

Stochastic Frank-Wolfe / gradient boosting runner. Same return signature
as `run_method_tracked`: returns `(fa, γ_ref, history)`.

`kwargs` entries:
  :max_iters    (default 200)
  :batch_size   (default 0 → full batch; otherwise the mini-batch size b ≤ K)
  :tol_obj      (default 1e-3)        — stop when train loss < tol_obj
  :tol_delta    (default 1e-5)        — stop when running improvement stalls
  :step_rule    (default :diminishing)
  :seed         (default 0)
"""
function run_method_tracked_fw(name::Symbol, kwargs::Dict,
    Ξ_train, Ξ_test=nothing; verbosity::Int=1)

    # 0 = silent; ≥1 = per-iteration table. FW's inner pricing call is
    # already silent so the level-2 step adds no extra detail here.
    verbose = verbosity >= 1

    max_iters = get(kwargs, :max_iters, 200)
    batch_size = get(kwargs, :batch_size, 0)
    tol_obj = get(kwargs, :tol_obj, 1e-3)
    tol_delta = get(kwargs, :tol_delta, 1e-5)
    step_rule = get(kwargs, :step_rule, :diminishing)
    rng_seed = get(kwargs, :seed, 0)
    timelimit = get(kwargs, :timelimit, Inf)        # wall-clock cap, seconds

    n = length(Ξ_train[1][1])
    K_train = length(Ξ_train)
    has_test = !isnothing(Ξ_test)
    rng = MersenneTwister(rng_seed)
    bs = batch_size <= 0 ? K_train : min(batch_size, K_train)

    # Initialize surrogate market with one random CES agent (mirror the CG code path).
    fa = FisherMarket(1, n; ρ=rand(1), scale=30.0, sparsity=0.99)
    fa.w[1] = 1.0
    γ_ref = Ref(compute_gamma_from_market(fa, Ξ_train))
    w_vec = Float64[1.0]                        # FW iterate weights (own copy)

    # Track best iterate (FW objective is non-monotone; averaged iterate is theoretical,
    # best-seen is the practical surrogate).
    best_obj = Inf
    best_w = copy(w_vec)
    best_m = fa.m

    history = Dict(
        :primal_obj => Float64[],
        :test_err => Float64[],
        :num_agents => Int[],
    )

    _t0 = time()
    if verbose
        println("=== $(name) (FW, batch=$(bs)/$K_train) ===")
        @printf("%5s | %10s | %10s | %10s | %10s | %5s | %10s\n",
            "iter", "train", "test", "step η", "Δ(F)", "T", "t(s)")
        @printf("%5s-+-%10s-+-%10s-+-%10s-+-%10s-+-%5s-+-%10s\n",
            "-----", "----------", "----------", "----------", "----------", "-----", "----------")
    end

    for iter in 1:max_iters
        if time() - _t0 > timelimit
            verbose && @printf("time limit reached (%.1fs > %.1fs)\n", time() - _t0, timelimit)
            break
        end
        # ---- evaluate / record ---------------------------------------------------------
        primal_obj = evaluate_train_loss(Ξ_train, γ_ref, w_vec)
        # Sync fa.w to current FW iterate before evaluating test error.
        if length(fa.w) != length(w_vec)
            # Should not happen post-add_to_market! but stay defensive.
            resize!(fa.w, length(w_vec))
        end
        fa.w .= w_vec
        te = has_test ? evaluate_test_error(fa, Ξ_test) : NaN
        push!(history[:primal_obj], primal_obj)
        push!(history[:test_err], te)
        push!(history[:num_agents], fa.m)

        if primal_obj < best_obj
            best_obj = primal_obj
            best_w = copy(w_vec)
            best_m = fa.m
        end

        improvement = length(history[:primal_obj]) >= 2 ?
                      history[:primal_obj][end-1] - primal_obj : NaN

        function _log_row(η_val=NaN)
            _elapsed = time() - _t0
            !verbose && return
            te_str = isnan(te) ? @sprintf("%10s", "-") : @sprintf("%10.3e", te)
            η_str = isnan(η_val) ? @sprintf("%10s", "-") : @sprintf("%10.3e", η_val)
            Δ_str = @sprintf("%10.3e", isnan(improvement) ? 0.0 : improvement)
            @printf("%5d | %10.3e | %s | %s | %s | %5d | %10.4f\n",
                iter, primal_obj, te_str, η_str, Δ_str, fa.m, _elapsed)
        end

        # ---- convergence checks --------------------------------------------------------
        # FW iterates are non-monotone under stochastic / subgradient updates; we only
        # check absolute tolerance and the running mean of the last `window` iterates.
        if primal_obj < tol_obj
            _log_row()
            verbose && @printf("converged (obj/K = %.2e < tol_obj=%g)\n", primal_obj, tol_obj)
            break
        end
        window = 10
        if length(history[:primal_obj]) >= 2 * window
            recent = history[:primal_obj][end-window+1:end]
            prior = history[:primal_obj][end-2*window+1:end-window]
            mean_drop = sum(prior) / window - sum(recent) / window
            if mean_drop < tol_delta
                _log_row()
                verbose && @printf(
                    "stalled (mean drop over last %d iters = %.2e < tol_delta=%g)\n",
                    window, mean_drop, tol_delta
                )
                # break
            end
        end

        # ---- mini-batch + subgradient --------------------------------------------------
        batch = bs == K_train ? collect(1:K_train) :
                sort!(randperm(rng, K_train)[1:bs])
        u = compute_residual_subgrad(Ξ_train, γ_ref, w_vec, batch)

        # ---- LMO ≡ pricing oracle ------------------------------------------------------
        y_lp, σ_lp, _, _ = solve_pricing_dual_lp(Ξ_train, u)
        y_opt, σ_opt, γ_new, _ = solve_pricing(Ξ_train, u; y_init=y_lp, σ_init=σ_lp)
        c_new, ρ_new = recover_ces_params(y_opt, σ_opt)

        # ---- FW update -----------------------------------------------------------------
        η = fw_step_size(iter; rule=step_rule)
        w_vec .*= (1 - η)
        push!(w_vec, η)

        add_to_gamma!(γ_ref, γ_new)
        add_to_market!(fa, c_new, ρ_new, η)
        # add_to_market! appended weight η to fa.w but didn't scale older entries.
        fa.w .= w_vec

        _log_row(η)
    end

    # Restore best iterate before returning (matches CG's "drop then return" practice).
    if best_m == length(best_w) <= fa.m
        # Truncate γ_ref and fa to the best iterate size, restore best weights.
        m_cur = fa.m
        if best_m < m_cur
            γ_ref[] = γ_ref[][1:best_m, :, :]
            fa.m = best_m
            fa.c = fa.c[:, 1:best_m]
            fa.ρ = fa.ρ[1:best_m]
            fa.σ = fa.σ[1:best_m]
            fa.x = fa.x[:, 1:best_m]
            fa.g = fa.g[:, 1:best_m]
            fa.s = fa.s[:, 1:best_m]
            if length(fa.val_u) >= best_m
                fa.val_u = fa.val_u[1:best_m]
            end
            if length(fa.ε_br_play) >= best_m
                fa.ε_br_play = fa.ε_br_play[1:best_m]
            end
        end
        if length(fa.w) != best_m
            resize!(fa.w, best_m)
        end
        fa.w .= best_w
    end

    _elapsed = time() - _t0
    if verbose
        @printf("--- done: %d agents, best obj/K=%.3e, t=%.4fs ---\n",
            fa.m, best_obj, _elapsed)
    end
    return fa, γ_ref, history
end

# ----------------------------------------------------------------------------
# Variant that delegates the FW iteration to the FrankWolfe.jl package.
# ----------------------------------------------------------------------------
# Decision variable lives in ℝ^{K×n}: the matrix `H` of values h(p_k) =
# Σ_t w_t γ_t(p_k). Feasible set is the convex hull of CES atoms, with
# the LMO returning a one-atom evaluation matrix γ ∈ ℝ^{K×n} (rows are
# γ(p_k) for k ∈ [K]).
#
# Objective f(H) = (1/K) Σ_k ‖P_k g_k − H[k,:]‖_∞.
# Gradient: row-k subgradient is −sign(target_k − H[k,:])[j*]·e_{j*}/K.
#
# Atom-to-params lookup uses a Dict keyed by objectid of the LMO-returned
# matrix; FrankWolfe.jl stores LMO outputs by identity in ActiveSet, so
# the cache survives the AFW iteration. Initial atom params are computed
# from the random FisherMarket used to seed the iterate.

"""
    CESPricingLMO(Ξ)

Linear minimization oracle for the CES pricing subproblem. Calls
`solve_pricing_dual_lp` + `solve_pricing` (from ces.jl) to find the CES
atom whose evaluation matrix γ ∈ ℝ^{K×n} minimizes ⟨direction, γ⟩, and
caches the recovered `(y, σ)` for downstream `add_to_market!`.
"""
mutable struct CESPricingLMO{T} <: FrankWolfe.LinearMinimizationOracle
    Ξ::Vector{Tuple{Vector{T},Vector{T}}}
    cache::Dict{UInt64,NamedTuple}
end

CESPricingLMO(Ξ::Vector{Tuple{Vector{T},Vector{T}}}) where {T} =
    CESPricingLMO{T}(Ξ, Dict{UInt64,NamedTuple}())

function FrankWolfe.compute_extreme_point(lmo::CESPricingLMO{T},
    direction::AbstractMatrix; kwargs...) where {T}
    # min ⟨direction, γ⟩  ⇔  max ⟨−direction, γ⟩, which is the dual-pricing
    # problem with cost rows u_k := −direction[k,:].
    u = Matrix{T}(-direction)
    y_lp, σ_lp, _, _ = solve_pricing_dual_lp(lmo.Ξ, u)
    y_opt, σ_opt, γ_new, _ = solve_pricing(lmo.Ξ, u; y_init=y_lp, σ_init=σ_lp)
    γ_dense = Matrix{T}(γ_new)
    lmo.cache[objectid(γ_dense)] = (y=y_opt, σ=σ_opt)
    return γ_dense
end

"""
    run_method_tracked_fwjl(name, kwargs, Ξ_train, Ξ_test=nothing; verbosity=1)

Drive the FW iteration with FrankWolfe.jl's `away_frank_wolfe`, which
maintains an active set so per-atom weights can be recovered post-run.
Returns `(fa, γ_ref, history)` matching the runners in cpm.jl /
`run_method_tracked_fw`.

`kwargs` entries:
- `:max_iters`  (200)  forwarded as `max_iteration` to FrankWolfe.jl
- `:tol_obj`    (1e-3) forwarded as `epsilon`
- `:timelimit`  (Inf)  stops via callback when wall-clock exceeds
- `:seed`       (0)    seed for the initial random CES atom
- `:line_search` (FrankWolfe.Agnostic())   2/(t+2) classical rule

History is recorded at every FW iteration via FrankWolfe.jl's `callback`
interface, then converted into the dictionary shape used by the CG / FW
runners (`:primal_obj`, `:test_err`, `:num_agents`).
"""
function run_method_tracked_fwjl(name::Symbol, kwargs::Dict,
    Ξ_train, Ξ_test=nothing; verbosity::Int=1)

    verbose = verbosity >= 1

    max_iters = get(kwargs, :max_iters, 200)
    tol_obj   = get(kwargs, :tol_obj, 1e-3)
    timelimit = get(kwargs, :timelimit, Inf)
    rng_seed  = get(kwargs, :seed, 0)
    line_search = get(kwargs, :line_search, FrankWolfe.Agnostic())

    K = length(Ξ_train)
    n = length(Ξ_train[1][1])

    # Pre-compute targets P_k g_k once.
    targets = [Ξ_train[k][1] .* Ξ_train[k][2] for k in 1:K]

    f = function (H::AbstractMatrix)
        s = 0.0
        @inbounds for k in 1:K
            m = 0.0
            for j in 1:n
                m = max(m, abs(targets[k][j] - H[k, j]))
            end
            s += m
        end
        return s / K
    end
    grad! = function (G::AbstractMatrix, H::AbstractMatrix)
        fill!(G, 0.0)
        @inbounds for k in 1:K
            j_star = 1
            best = -1.0
            for j in 1:n
                v = abs(targets[k][j] - H[k, j])
                if v > best
                    best = v
                    j_star = j
                end
            end
            # ∂‖·‖_∞ at r := target − H[k,:] is sign(r_{j*})·e_{j*};
            # we differentiate ‖target − H[k,:]‖_∞ w.r.t. H so the sign flips.
            G[k, j_star] = -sign(targets[k][j_star] - H[k, j_star]) / K
        end
        return G
    end

    # Initial CES atom — seeded so runs are reproducible.
    Random.seed!(rng_seed)
    fa0 = FisherMarket(1, n; ρ=rand(1), scale=30.0, sparsity=0.99)
    fa0.w[1] = 1.0
    γ_init = compute_gamma_from_market(fa0, Ξ_train)        # (1, K, n)
    H0 = Matrix{Float64}(reshape(γ_init[1, :, :], K, n))

    # Cache the initial atom's (y, σ) so the active-set lookup post-FW
    # can recover its params just like an LMO-produced atom.
    σ0 = fa0.σ[1]
    c0 = Vector(fa0.c[:, 1])
    y0 = (1 + σ0) .* log.(c0)

    lmo = CESPricingLMO(Ξ_train)
    lmo.cache[objectid(H0)] = (y=y0, σ=σ0)

    # History recorded per FW iteration via callback.
    history = Dict(
        :primal_obj => Float64[],
        :test_err   => Float64[],
        :num_agents => Int[],
    )
    _t0 = time()

    # Light-weight callback: record per-iteration primal & active-set
    # size; test_err is set NaN during iteration and filled in once at
    # the end from the final active set. In FrankWolfe.jl v0.6,
    # `away_frank_wolfe` calls the callback as `cb(state, active_set)`,
    # so the live active set comes in via `args[1]`.
    function cb(state, args...)
        as = isempty(args) ? nothing : args[1]
        T_cur = isnothing(as) ? 1 : length(as.atoms)
        push!(history[:primal_obj], state.primal)
        push!(history[:test_err],   NaN)
        push!(history[:num_agents], T_cur)
        # Wall-clock cap: returning `false` halts FrankWolfe.jl's loop.
        return (time() - _t0) ≤ timelimit
    end

    common_kwargs = (
        max_iteration = max_iters,
        epsilon       = tol_obj,
        line_search   = line_search,
        callback      = cb,
        verbose       = false,           # we drive our own logging via cb
        print_iter    = max_iters + 1,   # suppress FW's stride prints
        trajectory    = false,
    )

    # away_frank_wolfe returns a 7-tuple in FrankWolfe.jl v0.6:
    # (x, v, primal, dual_gap, ExecutionStatus, trajectory, active_set).
    x_opt, _, primal, dual_gap, _, _, active_set =
        FrankWolfe.away_frank_wolfe(f, grad!, lmo, H0; common_kwargs...)

    fa = _build_fa_from_active_set(active_set, lmo, n)
    γ_ref = Ref(compute_gamma_from_market(fa, Ξ_train))

    # Fill the final test_err into the trailing history slot — single
    # FisherMarket evaluation against the final active set.
    if !isnothing(Ξ_test) && !isempty(history[:test_err])
        history[:test_err][end] = evaluate_test_error(fa, Ξ_test)
    end

    _elapsed = time() - _t0
    if verbose
        println("=== $(name) (FrankWolfe.jl away_frank_wolfe) ===")
        @printf("--- done: %d atoms, primal=%.3e, dual_gap=%.3e, t=%.4fs ---\n",
            fa.m, primal, dual_gap, _elapsed)
    end

    return fa, γ_ref, history
end

"""
    _build_fa_from_active_set(as, lmo, n)

Reconstruct a `FisherMarket` from a FrankWolfe.jl `ActiveSet`. Each atom
in the active set is a γ matrix produced by `CESPricingLMO`; its (y, σ)
were stashed in `lmo.cache` keyed by `objectid`. Atoms whose params
aren't cached are skipped (their weight is folded into a uniform CES
fallback so the FisherMarket remains non-empty).
"""
function _build_fa_from_active_set(as, lmo::CESPricingLMO{T}, n::Int) where {T}
    atoms = as.atoms
    weights = as.weights
    fa = FisherMarket(0, n; ρ=Float64[], scale=30.0, sparsity=0.99)
    for (γ_matrix, w_t) in zip(atoms, weights)
        w_t <= 0 && continue
        params = get(lmo.cache, objectid(γ_matrix), nothing)
        isnothing(params) && continue
        add_column_to_market!(fa, params, :ces, T(w_t))
    end
    # If everything dropped (unlikely), seed with a single uniform atom.
    if fa.m == 0
        fa = FisherMarket(1, n; ρ=[0.0], scale=30.0, sparsity=0.99)
        fa.w[1] = 1.0
    end
    return fa
end
