# Hand-rolled away-step Frank–Wolfe runner for the surrogate-fitting master.
#
#   minimize  F(h) = (1/K) Σ_k ‖P_k g_k - h(p_k)‖_∞   over   h ∈ conv(H)
#
# This is our own FW loop (no FrankWolfe.jl dependency), so WE own the stopping
# rule: it stops on tol_obj / tol_delta / max_iters / timelimit — never on a
# duality gap. The gap-based stop in FrankWolfe.jl's away_frank_wolfe is unsafe
# here because the CES separation LMO is LBFGS-local (non-global) and, with
# subsampling, stochastic; an inexact LMO under-estimates the gap and can quit
# far from the optimum (see frankwolfe/wrapper_frankwolfe.jl).
#
# Each iteration:
#   1. H = Σ_t w_t γ_t,  subgradient u_k = sign(target_k - H_k)[j*]·e_{j*}.
#   2. LMO ≡ separation oracle:  s = argmax_γ Σ_k ⟨u_k, γ(p_k)⟩  (find_cut_single).
#   3. Away vertex a = active atom minimizing ⟨u, γ_a⟩; pick FW vs away by gap.
#   4. Line search η on the convex 1-D restriction; weight update + optional drop.
#
# FW vs SFW is NOT a separate code path: it is implied by subsampling alone.
# `:sample_size` (CLI --sample-size; legacy alias `:batch_size`) > 0 makes the LMO
# stochastic — find_cut_single subsamples (Ξ, u) jointly, solves on the subset,
# and re-expands the cut over the full data. sample_size == 0 ⇒ full-batch FW.
#
# Reuses separation.jl (`find_cut_single`, `add_column_to_market!`) and
# setup.jl (`compute_gamma_from_market`, `evaluate_test_error`).

using LinearAlgebra
using Random

# separation.jl is included by setup.jl; this file is included after setup.jl.

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

# Iteration-table layout for run_method_tracked_fw.
const FW_TABLE = IterTable(
    ["iter", "train",  "test",   "step η", "Δ(F)",   "T",   "step", "t(s)"],
    ["%5d",  "%10.3e", "%10.3e", "%10.3e", "%10.3e", "%5d", "%6s",  "%10.4f"],
    Any[1,   1.0e-3,   1.0e-3,   1.0e-3,   1.0e-3,   1,     "FW",   1.234],
)

"""
    fw_line_search(φ, η_max; iters=45) -> η

Line search for a CONVEX 1-D restriction `φ(η) = f(x + η·d)` on `[0, η_max]`, by
golden-section search. Both FW runners (Fisher ℓ∞ and Arrow–Debreu ℓ1) minimize a
convex objective, so the restriction is convex and golden-section converges to its
minimizer; the endpoints `{0, η_max}` are compared too, so a min at 0 (no improving
step) or at `η_max` (a full FW step / a weight-zeroing away "drop") is captured.
Shared by `run_method_tracked_fw` and `run_ad_tracked_fw`.
"""
function fw_line_search(φ, η_max::Real; iters::Int=45)
    η_max <= 0 && return 0.0
    r = (sqrt(5.0) - 1) / 2
    a, b = 0.0, float(η_max)
    c = b - r * (b - a)
    d = a + r * (b - a)
    fc, fd = φ(c), φ(d)
    for _ in 1:iters
        if fc < fd
            b, d, fd = d, c, fc
            c = b - r * (b - a)
            fc = φ(c)
        else
            a, c, fc = c, d, fd
            d = a + r * (b - a)
            fd = φ(d)
        end
        (b - a) < 1e-12 && break
    end
    η = (a + b) / 2
    fη, f0, fM = φ(η), φ(0.0), φ(float(η_max))
    best, fb = η, fη
    f0 < fb && ((best, fb) = (0.0, f0))
    fM < fb && (best = float(η_max))
    return best
end

"""
    run_method_tracked_fw(name, kwargs, Ξ_train, Ξ_test=nothing; verbosity=1)

Hand-rolled away-step Frank–Wolfe runner for the Fisher surrogate-fitting master.
Same return signature as `run_method_tracked`: `(fa, γ_ref, history)`.

Maintains an explicit active set `(γ_t, params_t, class_t, w_t)` with FW steps,
away steps, and weight-zeroing drops; the step size comes from `fw_line_search`
(or the diminishing `2/(ℓ+2)` rule). Stopping is ours alone — tol_obj /
tol_delta / max_iters / timelimit, never a duality gap. The best full-batch
iterate is snapshotted and restored (FW is non-monotone under subsampling).

`kwargs` entries:
  :max_iters    (200)
  :tol_obj      (1e-3)            stop when train loss < tol_obj   (nothing ⇒ off)
  :tol_delta    (1e-5)            stop when the windowed mean drop stalls (nothing ⇒ off)
  :step_rule    (:linesearch)     :linesearch | :diminishing
  :away_steps   (true)            enable away steps + drops
  :classes      ([:ces])          LMO class menu (find_cut_single)
  :sample_size  (0)               >0 ⇒ stochastic LMO (SFW); legacy alias :batch_size
  :sample_hard  (false)           boosting-style residual-weighted subsample
  :seed         (0)
  :timelimit    (Inf)
Any other key is forwarded to `find_cut_single` (e.g. ces_sigma_lower).
"""
function run_method_tracked_fw(name::Symbol, kwargs::Dict,
    Ξ_train, Ξ_test=nothing; verbosity::Int=1)

    verbose = verbosity >= 1
    verbose_separation = verbosity >= 2

    max_iters  = get(kwargs, :max_iters, 200)
    tol_obj    = get(kwargs, :tol_obj, 1e-3)
    tol_delta  = get(kwargs, :tol_delta, 1e-5)
    step_rule  = Symbol(get(kwargs, :step_rule, :linesearch))
    away_steps = get(kwargs, :away_steps, true) === true
    rng_seed   = get(kwargs, :seed, 0)
    timelimit  = get(kwargs, :timelimit, Inf)
    interval_eval_test = get(kwargs, :interval_eval_test, 1)
    log_interval = Int(get(kwargs, :log_interval, 1))   # print the iter row every N iters
    classes    = Vector{Symbol}(get(kwargs, :classes, Symbol[:ces]))
    # FW vs SFW is implied by subsampling alone (no separate path): sample_size>0
    # makes the LMO stochastic. :batch_size is the legacy alias.
    sample_size = Int(get(kwargs, :sample_size, get(kwargs, :batch_size, 0)))
    sample_hard = get(kwargs, :sample_hard, false) === true

    # Everything not consumed by the loop is forwarded to find_cut_single.
    _control = (:max_iters, :tol_obj, :tol_delta, :tol_rc, :step_rule, :away_steps,
        :seed, :timelimit, :interval_eval_test, :interval_eval_excess, :f_real,
        :log_interval, :classes, :sample_size, :batch_size, :sample_hard, :drop,
        :interval_dropping, :ad_delta, :ad_endow_mode, :ad_mask_size)
    oracle_kw = Dict{Symbol,Any}(k => v for (k, v) in kwargs if !(k in _control))

    K = length(Ξ_train)
    n = length(Ξ_train[1][1])
    has_test = !isnothing(Ξ_test)
    Random.seed!(rng_seed)

    targets = [Ξ_train[k][1] .* Ξ_train[k][2] for k in 1:K]
    _t0 = time()
    reset_cg_timers!()   # per-android separation accumulators (FW has no LP master)
    _remaining() = isfinite(timelimit) ? max(1.0, timelimit - (time() - _t0)) : nothing

    # ---- active set + objective helpers ----------------------------------------
    Γ = Matrix{Float64}[]      # γ_t over the training prices (K×n)
    P = NamedTuple[]           # oracle params per atom (y, σ)
    CL = Symbol[]              # class per atom
    w = Float64[]              # convex weights, Σ = 1

    f_obj = function (H)
        s = 0.0
        @inbounds for k in 1:K
            mx = 0.0
            for j in 1:n
                dval = abs(targets[k][j] - H[k, j]); dval > mx && (mx = dval)
            end
            s += mx
        end
        return s / K
    end
    Hmat = function ()
        H = zeros(Float64, K, n)
        @inbounds for t in eachindex(w)
            wt = w[t]; Γt = Γ[t]
            for k in 1:K, j in 1:n
                H[k, j] += wt * Γt[k, j]
            end
        end
        return H
    end
    subgrad = function (H)
        u = zeros(Float64, K, n)
        @inbounds for k in 1:K
            r = targets[k] .- @view H[k, :]
            js = argmax(abs.(r))
            u[k, js] = sign(r[js])
        end
        return u
    end
    lmo = function (u, spw)
        return find_cut_single(Ξ_train, u, 0.0, classes;
            sample_size=sample_size, sample_weights=spw,
            verbose=verbose_separation, timelimit=_remaining(), oracle_kw...)
    end

    # Seed vertex: LMO best response to the all-zero predictor.
    H0 = zeros(Float64, K, n)
    c0 = lmo(subgrad(H0), nothing)
    push!(Γ, Matrix{Float64}(c0.γ_new)); push!(P, c0.params)
    push!(CL, c0.class); push!(w, 1.0)

    history = Dict(
        :primal_obj => Float64[],
        :test_err => Float64[],
        :excess => Float64[],
        :num_agents => Int[],
    )
    last_test_err = Ref(NaN)
    best_f = Inf
    best_P = copy(P); best_CL = copy(CL); best_w = copy(w)

    if verbose
        print_banner(FW_TABLE, BANNER_TITLE)
        print_config("method",          String(name))
        print_config("alias",           "away-step FW (manual)")
        print_config("classes",         join(String.(classes), ", "))
        print_config("K (training samples)", K)
        print_config("n (goods)",       n)
        print_config("subsample (SFW)", sample_size > 0 ?
            (sample_hard ? "$(sample_size), residual-weighted" : "$(sample_size), uniform") : "off (full batch)")
        print_config("away_steps",      string(away_steps))
        print_config("step_rule",       String(step_rule))
        print_config("max_iters",       max_iters)
        print_config("timelimit (s)",   @sprintf("%g", Float64(timelimit)))
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

        H = Hmat()
        fcur = f_obj(H)
        # Test error on the CURRENT iterate (build a market from the active set).
        if has_test && interval_eval_test > 0 && (iter % interval_eval_test == 0)
            last_test_err[] = evaluate_test_error(_build_fa(P, CL, w, n), Ξ_test)
        end
        te = last_test_err[]
        push!(history[:primal_obj], fcur)
        push!(history[:test_err], te)
        push!(history[:excess], NaN)
        push!(history[:num_agents], length(w))

        if fcur < best_f
            best_f = fcur
            best_P = copy(P); best_CL = copy(CL); best_w = copy(w)
        end

        improvement = length(history[:primal_obj]) >= 2 ?
                      history[:primal_obj][end-1] - fcur : NaN
        _log_row(η_val, kind) = verbose && print_row(FW_TABLE,
            Any[iter, fcur, te, η_val, isnan(improvement) ? 0.0 : improvement,
                length(w), kind, time() - _t0])

        if !isnothing(tol_obj) && fcur < tol_obj
            _log_row(0.0, "-")
            verbose && print_continuation(FW_TABLE,
                @sprintf("converged (obj/K = %.2e < tol_obj=%g)", fcur, tol_obj))
            break
        end
        window = 10
        if !isnothing(tol_delta) && length(history[:primal_obj]) >= 2 * window
            recent = @view history[:primal_obj][end-window+1:end]
            prior = @view history[:primal_obj][end-2*window+1:end-window]
            mean_drop = sum(prior) / window - sum(recent) / window
            if mean_drop < tol_delta
                _log_row(0.0, "-")
                verbose && print_continuation(FW_TABLE,
                    @sprintf("stalled (mean drop over last %d iters = %.2e < tol_delta=%g)",
                             window, mean_drop, tol_delta))
                break
            end
        end

        # ---- LMO + away vertex ------------------------------------------------
        u = subgrad(H)
        spw = (sample_hard && sample_size > 0) ?
              [norm(targets[k] .- @view(H[k, :]), Inf) for k in 1:K] : nothing
        cand = lmo(u, spw)
        if isnothing(cand)
            _log_row(0.0, "-")
            verbose && print_continuation(FW_TABLE, "LMO returned no cut; terminating")
            break
        end
        s = cand.γ_new
        g_fw = 0.0
        @inbounds for k in 1:K, j in 1:n
            g_fw += u[k, j] * (s[k, j] - H[k, j])
        end

        use_away = false
        a_idx = 0
        if away_steps && length(w) >= 2
            a_idx = 1; a_val = Inf
            for t in eachindex(w)
                v = 0.0
                Γt = Γ[t]
                @inbounds for k in 1:K, j in 1:n
                    v += u[k, j] * Γt[k, j]
                end
                v < a_val && (a_val = v; a_idx = t)
            end
            g_away = 0.0
            Γa = Γ[a_idx]
            @inbounds for k in 1:K, j in 1:n
                g_away += u[k, j] * (H[k, j] - Γa[k, j])
            end
            use_away = g_away > g_fw
        end

        if use_away
            Γa = Γ[a_idx]
            D = H .- Γa
            wa = w[a_idx]
            η_max = wa < 1.0 ? wa / (1 - wa) : 1.0
            kind = "away"
        else
            D = s .- H
            η_max = 1.0
            kind = "FW"
        end

        if step_rule === :linesearch
            η = fw_line_search(η -> f_obj(H .+ η .* D), η_max)
        else
            η = min(fw_step_size(iter), η_max)
        end

        if use_away
            w .*= (1 + η)
            w[a_idx] -= η
            if w[a_idx] <= 1e-10
                deleteat!(Γ, a_idx); deleteat!(P, a_idx)
                deleteat!(CL, a_idx); deleteat!(w, a_idx)
                kind = "drop"
            end
        else
            w .*= (1 - η)
            push!(Γ, Matrix{Float64}(s)); push!(P, cand.params)
            push!(CL, cand.class); push!(w, η)
        end
        sw = sum(w); sw > 0 && (w ./= sw)   # defensive renormalization

        (log_interval <= 1 || iter % log_interval == 0) && _log_row(η, kind)
    end

    # ---- build the returned market from the best-seen iterate ------------------
    isempty(best_w) && (best_P = copy(P); best_CL = copy(CL); best_w = copy(w))
    fa = _build_fa(best_P, best_CL, best_w, n)
    γ_ref = Ref(compute_gamma_from_market(fa, Ξ_train))

    if has_test && !isempty(history[:test_err])
        history[:test_err][end] = evaluate_test_error(fa, Ξ_test)
    end
    _elapsed = time() - _t0
    if verbose
        @printf("--- done: %d atoms, best obj/K=%.3e, t=%.4fs ---\n",
            fa.m, best_f, _elapsed)
    end
    print_cg_timing_summary()
    return fa, γ_ref, history
end

"""
    _build_fa(params, classes, w, n) -> FisherMarket

Construct a FisherMarket from an active set: one column per atom with its
(y, σ) params, class, and convex weight. Mirrors the active-set reconstruction
in wrapper_frankwolfe.jl.
"""
function _build_fa(params::AbstractVector, classes::AbstractVector, w::AbstractVector, n::Int)
    fa = FisherMarket(cpu_workspace(n, 0))
    for (p, cl, wt) in zip(params, classes, w)
        wt <= 0 && continue
        add_column_to_market!(fa, p, cl, Float64(wt))
    end
    if fa.m == 0   # everything zero-weight (degenerate): seed one uniform CES atom
        add_column_to_market!(fa, (y=zeros(n), σ=0.5), :ces, 1.0)
    end
    return fa
end
