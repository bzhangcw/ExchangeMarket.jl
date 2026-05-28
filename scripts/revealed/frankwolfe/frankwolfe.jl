# Stochastic Frank-Wolfe / gradient boosting variant for the surrogate-fitting master.
#
#   minimize  F(h) = (1/K) Σ_k ‖P_k g_k - h(p_k)‖_∞   over   h ∈ conv(H)
#
# Each FW step:
#   1. Sample mini-batch S ⊂ [K] (full batch if batch_size == 0 or ≥ K).
#   2. Subgradient r_k = sign(target_k - h(p_k))[j*] · e_{j*} for k ∈ S.
#   3. LMO ≡ existing separation solver:  γ_ℓ = argmax_γ Σ_k ⟨r_k, γ(p_k)⟩.
#   4. Update  h_{ℓ+1} = (1 - η_ℓ) h_ℓ + η_ℓ γ_ℓ,  η_ℓ = 2/(ℓ+2).
#
# Reuses separation.jl: `solve_separation_lbfgs_ces`, `solve_separation_dual_lp_ces`,
# `recover_ces_params`, `add_to_gamma!`, `add_to_market!`.

using LinearAlgebra
using Random

# separation.jl is included by setup.jl; this file is included after setup.jl.
# The third-party FrankWolfe.jl wrapper lives in third-party/wrapper_frankwolfe.jl
# (included from setup.jl after validate.jl).

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

# Iteration-table layout for run_method_tracked_fw. Widths are taken from
# the formatted dummy row so the header lines up without hand-tuned %5/%10
# magic in the per-iter call site.
const FW_TABLE = IterTable(
    ["iter", "train",  "test",   "step η", "Δ(F)",   "T",   "t(s)"],
    ["%5d",  "%10.3e", "%10.3e", "%10.3e", "%10.3e", "%5d", "%10.4f"],
    Any[1,   1.0e-3,   1.0e-3,   1.0e-3,   1.0e-3,   1,     1.234],
)

"""
    run_method_tracked_fw(name, kwargs, Ξ_train, Ξ_test=nothing; verbose=true)

Stochastic Frank-Wolfe / gradient boosting runner. Same return signature
as `run_method_tracked`: returns `(fa, γ_ref, history)`.

!!! note "Prefer `run_method_tracked_fwjl`"
    This is a minimal hand-rolled FW loop kept for reference and for
    cases where pulling in FrankWolfe.jl is undesirable. For production
    use prefer `run_method_tracked_fwjl` (in
    `third-party/wrapper_frankwolfe.jl`), which delegates to the
    FrankWolfe.jl package and supports away-steps, line search, and
    active-set bookkeeping that this manual loop does not.

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

    # One-shot nudge toward the FrankWolfe.jl wrapper. `maxlog=1` so a
    # batch sweep that calls this runner many times doesn't spam the log.
    @warn """run_method_tracked_fw is a minimal hand-rolled FW loop \
             (diminishing step only, no away-steps, no line search). \
             For production runs prefer run_method_tracked_fwjl from \
             third-party/wrapper_frankwolfe.jl, which delegates to the \
             FrankWolfe.jl package.""" maxlog=1

    # 0 = silent; ≥1 = per-iteration table. FW's inner separation call is
    # already silent so the level-2 step adds no extra detail here.
    verbose = verbosity >= 1

    max_iters = get(kwargs, :max_iters, 200)
    batch_size = get(kwargs, :batch_size, 0)
    tol_obj = get(kwargs, :tol_obj, 1e-3)
    tol_delta = get(kwargs, :tol_delta, 1e-5)
    step_rule = get(kwargs, :step_rule, :diminishing)
    rng_seed = get(kwargs, :seed, 0)
    timelimit = get(kwargs, :timelimit, Inf)        # wall-clock cap, seconds
    interval_eval_test = get(kwargs, :interval_eval_test, 1)
    f_real = get(kwargs, :f_real, nothing)
    interval_eval_excess = get(kwargs, :interval_eval_excess, 0)

    n = length(Ξ_train[1][1])
    K_train = length(Ξ_train)
    has_test = !isnothing(Ξ_test)
    rng = MersenneTwister(rng_seed)
    bs = batch_size <= 0 ? K_train : min(batch_size, K_train)

    # Initialize surrogate market with one random CES agent (mirror the CG code path).
    ws = cpu_workspace(n)
    add_ces!(ws, 1; ρ=rand(1), scale=30.0, sparsity=0.99)
    fa = FisherMarket(ws)
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
        :excess => Float64[],
        :num_agents => Int[],
    )
    last_test_err = Ref(NaN)
    last_excess = Ref(NaN)
    has_excess = !isnothing(f_real) && interval_eval_excess > 0

    _t0 = time()
    if verbose
        print_banner(FW_TABLE, BANNER_TITLE)
        print_config("method",          String(name))
        print_config("alias",           "FW (manual)")
        print_config("batch / K_train", @sprintf("%d / %d", bs, K_train))
        print_config("max_iters",       max_iters)
        print_config("timelimit (s)",   @sprintf("%g", Float64(timelimit)))
        print_config("step_rule",       String(step_rule))
        print_config("tol_obj",         isnothing(tol_obj)   ? "off" : @sprintf("%g", tol_obj))
        print_config("tol_delta",       isnothing(tol_delta) ? "off" : @sprintf("%g", tol_delta))
        println("-"^table_width(FW_TABLE))
        print_header(FW_TABLE)
    end

    for iter in 1:max_iters
        if time() - _t0 > timelimit
            verbose && print_continuation(FW_TABLE,
                @sprintf("time limit reached (%.1fs > %.1fs)", time() - _t0, timelimit))
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
        # test_err policy: N>0 ⇒ every N iters; N==-1 ⇒ only after the loop.
        if has_test && interval_eval_test > 0 && (iter % interval_eval_test == 0)
            last_test_err[] = evaluate_test_error(fa, Ξ_test)
        end
        te = last_test_err[]   # alias for the logging closure below
        if has_excess && fa.m > 0 && (iter % interval_eval_excess == 0)
            try
                v = validate_surrogate(fa, f_real; verbose=false)
                last_excess[] = v.excess_surrogate_linf
            catch err
                @warn "[$name iter $iter] validate_surrogate failed" err
            end
        end
        push!(history[:primal_obj], primal_obj)
        push!(history[:test_err], te)
        push!(history[:excess], last_excess[])
        push!(history[:num_agents], fa.m)

        if primal_obj < best_obj
            best_obj = primal_obj
            best_w = copy(w_vec)
            best_m = fa.m
        end

        improvement = length(history[:primal_obj]) >= 2 ?
                      history[:primal_obj][end-1] - primal_obj : NaN

        function _log_row(η_val=NaN)
            verbose || return
            print_row(FW_TABLE, Any[iter, primal_obj, te, η_val,
                                    isnan(improvement) ? 0.0 : improvement,
                                    fa.m, time() - _t0])
        end

        # ---- convergence checks --------------------------------------------------------
        # FW iterates are non-monotone under stochastic / subgradient updates; we only
        # check absolute tolerance and the running mean of the last `window` iterates.
        # `nothing` on either tolerance disables the corresponding check.
        if !isnothing(tol_obj) && primal_obj < tol_obj
            _log_row()
            verbose && print_continuation(FW_TABLE,
                @sprintf("converged (obj/K = %.2e < tol_obj=%g)", primal_obj, tol_obj))
            break
        end
        window = 10
        if !isnothing(tol_delta) && length(history[:primal_obj]) >= 2 * window
            recent = history[:primal_obj][end-window+1:end]
            prior = history[:primal_obj][end-2*window+1:end-window]
            mean_drop = sum(prior) / window - sum(recent) / window
            if mean_drop < tol_delta
                _log_row()
                verbose && print_continuation(FW_TABLE,
                    @sprintf("stalled (mean drop over last %d iters = %.2e < tol_delta=%g)",
                             window, mean_drop, tol_delta))
                # break
            end
        end

        # ---- mini-batch + subgradient --------------------------------------------------
        batch = bs == K_train ? collect(1:K_train) :
                sort!(randperm(rng, K_train)[1:bs])
        u = compute_residual_subgrad(Ξ_train, γ_ref, w_vec, batch)

        # ---- LMO ≡ separation oracle ------------------------------------------------------
        y_lp, σ_lp, _, _ = solve_separation_dual_lp_ces(Ξ_train, u)
        y_opt, σ_opt, γ_new, _ = solve_separation_lbfgs_ces(Ξ_train, u; y_init=y_lp, σ_init=σ_lp)
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
    # Final test_err evaluation always overwrites the trailing slot, so the
    # last entry reflects the converged surrogate regardless of `interval_eval_test`.
    if has_test && !isempty(history[:test_err])
        history[:test_err][end] = evaluate_test_error(fa, Ξ_test)
    end
    if has_excess && fa.m > 0 && !isempty(history[:excess])
        try
            v = validate_surrogate(fa, f_real; verbose=false)
            history[:excess][end] = v.excess_surrogate_linf
        catch err
            @warn "[$name final] validate_surrogate failed" err
        end
    end
    if verbose
        @printf("--- done: %d agents, best obj/K=%.3e, t=%.4fs ---\n",
            fa.m, best_obj, _elapsed)
    end
    return fa, γ_ref, history
end
