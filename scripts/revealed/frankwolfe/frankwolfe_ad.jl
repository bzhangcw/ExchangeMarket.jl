# Hand-rolled away-step Frank–Wolfe runner for the Arrow–Debreu surrogate-fitting
# master (δ = 1). Arrow–Debreu sibling of frankwolfe/frankwolfe.jl.
#
#   minimize  F(M) = (1/K) Σ_k ‖P_k g_k − M_k‖_1   over the bundle hull
#
# where the predictor M[k,j] = Σ_t ⟨p_k, b_t⟩ γ_t(p_k)_j and, with the supply
# scale δ FIXED at 1 (Σ_t b_t = 1), the feasible set is a COMPACT convex set
# (a product of n simplices over conv(ℋ)). Its vertices are "one preference type
# per good" bundles: good l is owned outright (b = e_l) by a single type γ^{(l)},
# giving  M_vertex[k,j] = Σ_l p_k[l]·γ^{(l)}(p_k)_j.
#
# Like the Fisher manual runner this is OUR loop (no FrankWolfe.jl): stopping is
# tol_obj / tol_delta / max_iters / timelimit, never a duality gap (the gap is
# unreliable for this LBFGS-local / subsampled, inexact LMO — the whole reason we
# moved off the package; see frankwolfe/wrapper_frankwolfe_ad.jl).
#
# FW vs SFW is implied by subsampling alone: :sample_size > 0 makes the per-good
# LMO stochastic (find_cut_single subsamples and re-expands). No separate path.
#
# Reuses: find_cut_single, _gamma_over_full_from_cand (separation.jl);
# ad_market_from_atoms, evaluate_test_error_ad (redistribute_ad.jl); fw_line_search,
# AD_TABLE/banner helpers; loaded after frankwolfe.jl and redistribute_ad.jl.

using LinearAlgebra
using Random

# Iteration table: iter, train (ℓ1/K), test, |gap| (diagnostic), Δ(F), T
# (active bundles), step type, t(s).
const FWAD_TABLE = IterTable(
    ["iter", "train",  "test",   "gap",    "Δ(F)",   "T",   "step", "t(s)"],
    ["%5d",  "%10.3e", "%10.3e", "%10.3e", "%10.3e", "%5d", "%6s",  "%10.4f"],
    Any[1,   1.0e-3,   1.0e-3,   1.0e-3,   1.0e-3,   1,     "FW",   1.234],
)

"""
    _ad_expand_bundles(BC, w, n) -> (cands, B)

Turn an active set of bundle vertices into the AD atom list + endowment matrix.
Each active bundle `i` (weight `w_i > 0`) owns good `l` via its cached type
`BC[i][l]`, contributing an atom with endowment `w_i` on good `l`. Stacking over
(bundle, good) gives `Σ_t B[t,l] = Σ_i w_i = 1` for every good (δ = 1).
"""
function _ad_expand_bundles(BC::AbstractVector, w::AbstractVector, n::Int)
    cands = NamedTuple[]
    rows = Vector{Float64}[]
    for (bundle, α) in zip(BC, w)
        α <= 1e-12 && continue
        for l in 1:n
            c = bundle[l]
            push!(cands, (class=c.class, params=c.params))
            row = zeros(Float64, n); row[l] = Float64(α)
            push!(rows, row)
        end
    end
    if isempty(cands)   # degenerate: keep every good owned by one CES atom
        for l in 1:n
            push!(cands, (class=:ces, params=(y=zeros(n), σ=0.5)))
            row = zeros(Float64, n); row[l] = 1.0
            push!(rows, row)
        end
    end
    B = Matrix{Float64}(undef, length(cands), n)
    for (t, r) in enumerate(rows)
        B[t, :] .= r
    end
    return cands, B
end

"""
    run_ad_tracked_fw(name, kwargs, Ξ_train, Ξ_test=nothing; verbosity=1)
        -> (fa::ArrowDebreuMarket, γ_ref, history)

Hand-rolled away-step Frank–Wolfe runner for the Arrow–Debreu master (δ = 1).
Mirrors `run_method_tracked_fw` but optimizes the AD predictor over the bundle
hull with the ℓ1 objective, and returns a fitted `ArrowDebreuMarket` — the same
`(fa, γ_ref, history)` triple the shared drivers expect from `run_ad_tracked`.

`kwargs`: as `run_method_tracked_fw`, plus `:classes` defaults to `[:ces]`
(homothetic only; no :leontief — it cannot round-trip through the AD market's
CES (c, ρ) storage). `:ad_delta` / `:ad_endow_mode` are ignored (δ ≡ 1; ownership
is single-good).
"""
function run_ad_tracked_fw(name::Symbol, kwargs::Dict,
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
    classes    = Vector{Symbol}(get(kwargs, :classes, Symbol[:ces]))
    :leontief in classes &&
        error("run_ad_tracked_fw: :leontief atoms cannot be stored in an " *
              "ArrowDebreuMarket (CES (c, ρ) form diverges at σ = -1). " *
              "Use --classes from {ces,linear}.")
    sample_size = Int(get(kwargs, :sample_size, get(kwargs, :batch_size, 0)))
    sample_hard = get(kwargs, :sample_hard, false) === true

    _control = (:max_iters, :tol_obj, :tol_delta, :tol_rc, :step_rule, :away_steps,
        :seed, :timelimit, :interval_eval_test, :interval_eval_excess, :f_real,
        :classes, :sample_size, :batch_size, :sample_hard, :drop, :interval_dropping,
        :ad_delta, :ad_endow_mode, :ad_mask_size)
    oracle_kw = Dict{Symbol,Any}(k => v for (k, v) in kwargs if !(k in _control))

    K = length(Ξ_train)
    n = length(Ξ_train[1][1])
    has_test = !isnothing(Ξ_test)
    Random.seed!(rng_seed)

    targets = [Ξ_train[k][1] .* Ξ_train[k][2] for k in 1:K]   # P_k g_k (δ=1 ⇒ no shift)
    prices = [Ξ_train[k][1] for k in 1:K]
    _t0 = time()
    _remaining() = isfinite(timelimit) ? max(1.0, timelimit - (time() - _t0)) : nothing

    # ---- active set (bundle vertices) + objective helpers ----------------------
    M = Matrix{Float64}[]            # predictor X_t over training prices (K×n)
    BC = Vector{NamedTuple}[]        # n per-good cands per bundle
    w = Float64[]

    f_obj = function (X)
        s = 0.0
        @inbounds for k in 1:K, j in 1:n
            s += abs(targets[k][j] - X[k, j])
        end
        return s / K
    end
    Xmat = function ()
        X = zeros(Float64, K, n)
        @inbounds for t in eachindex(w)
            wt = w[t]; Mt = M[t]
            for k in 1:K, j in 1:n
                X[k, j] += wt * Mt[k, j]
            end
        end
        return X
    end
    # u_pred[k,j] = sign(target − X): the ascent direction whose ⟨u, M⟩ the LMO
    # maximizes. Returns the K×n sign matrix.
    signs = function (X)
        U = Matrix{Float64}(undef, K, n)
        @inbounds for k in 1:K, j in 1:n
            U[k, j] = sign(targets[k][j] - X[k, j])
        end
        return U
    end
    # Bundle LMO: per good l reweight u^l = p_k[l]·U and call find_cut_single;
    # assemble M_vertex[k,j] = Σ_l p_k[l]·γ^{(l)}(p_k)_j and the n cands.
    lmo = function (U, spw)
        Mv = zeros(Float64, K, n)
        bundle = Vector{NamedTuple}(undef, n)
        ul = Matrix{Float64}(undef, K, n)
        for l in 1:n
            @inbounds for k in 1:K
                pkl = prices[k][l]
                for j in 1:n
                    ul[k, j] = pkl * U[k, j]
                end
            end
            c = find_cut_single(Ξ_train, ul, 0.0, classes;
                sample_size=sample_size, sample_weights=spw,
                verbose=verbose_separation, timelimit=_remaining(), oracle_kw...)
            isnothing(c) && return (nothing, bundle)
            bundle[l] = (class=c.class, params=c.params)
            γl = c.γ_new
            @inbounds for k in 1:K
                pkl = prices[k][l]
                for j in 1:n
                    Mv[k, j] += pkl * γl[k, j]
                end
            end
        end
        return (Mv, bundle)
    end

    # Seed vertex: LMO best response to the all-zero predictor.
    X0 = zeros(Float64, K, n)
    Mv0, bundle0 = lmo(signs(X0), nothing)
    isnothing(Mv0) && error("run_ad_tracked_fw: seed LMO produced no cut.")
    push!(M, Mv0); push!(BC, bundle0); push!(w, 1.0)

    history = Dict(
        :primal_obj => Float64[],
        :test_err => Float64[],
        :excess => Float64[],          # AD surrogate-equilibrium excess not wired
        :num_agents => Int[],          # active-set size (number of bundles)
        :delta => Float64[],           # constant 1.0 (δ fixed)
    )
    last_test_err = Ref(NaN)
    best_f = Inf
    best_BC = copy(BC); best_w = copy(w)

    if verbose
        print_banner(FWAD_TABLE, BANNER_TITLE)
        print_config("method",          String(name))
        print_config("alias",           "away-step FW (manual, Arrow–Debreu, δ=1)")
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
        println("-"^table_width(FWAD_TABLE))
        print_header(FWAD_TABLE)
    end

    for iter in 1:max_iters
        if time() - _t0 > timelimit
            verbose && print_continuation(FWAD_TABLE,
                @sprintf("time limit reached (%.1fs > %.1fs)", time() - _t0, timelimit))
            break
        end

        X = Xmat()
        fcur = f_obj(X)
        if has_test && interval_eval_test > 0 && (iter % interval_eval_test == 0)
            cands_t, B_t = _ad_expand_bundles(BC, w, n)
            last_test_err[] = evaluate_test_error_ad(cands_t, B_t, Ξ_test; delta=1.0)
        end
        te = last_test_err[]
        push!(history[:primal_obj], fcur)
        push!(history[:test_err], te)
        push!(history[:excess], NaN)
        push!(history[:num_agents], length(w))
        push!(history[:delta], 1.0)

        if fcur < best_f
            best_f = fcur
            best_BC = [copy(b) for b in BC]; best_w = copy(w)
        end

        improvement = length(history[:primal_obj]) >= 2 ?
                      history[:primal_obj][end-1] - fcur : NaN
        _log_row(gap, kind) = verbose && print_row(FWAD_TABLE,
            Any[iter, fcur, te, gap, isnan(improvement) ? 0.0 : improvement,
                length(w), kind, time() - _t0])

        if !isnothing(tol_obj) && fcur < tol_obj
            _log_row(0.0, "-")
            verbose && print_continuation(FWAD_TABLE,
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
                verbose && print_continuation(FWAD_TABLE,
                    @sprintf("stalled (mean drop over last %d iters = %.2e < tol_delta=%g)",
                             window, mean_drop, tol_delta))
                break
            end
        end

        # ---- LMO + away vertex ------------------------------------------------
        U = signs(X)
        spw = (sample_hard && sample_size > 0) ?
              [sum(abs, targets[k] .- @view(X[k, :])) for k in 1:K] : nothing
        Mv, bundle = lmo(U, spw)
        if isnothing(Mv)
            _log_row(0.0, "-")
            verbose && print_continuation(FWAD_TABLE, "LMO returned no cut; terminating")
            break
        end
        g_fw = 0.0
        @inbounds for k in 1:K, j in 1:n
            g_fw += U[k, j] * (Mv[k, j] - X[k, j])
        end

        use_away = false
        a_idx = 0
        if away_steps && length(w) >= 2
            a_idx = 1; a_val = Inf
            for t in eachindex(w)
                v = 0.0; Mt = M[t]
                @inbounds for k in 1:K, j in 1:n
                    v += U[k, j] * Mt[k, j]
                end
                v < a_val && (a_val = v; a_idx = t)
            end
            g_away = 0.0; Ma = M[a_idx]
            @inbounds for k in 1:K, j in 1:n
                g_away += U[k, j] * (X[k, j] - Ma[k, j])
            end
            use_away = g_away > g_fw
        end

        if use_away
            Ma = M[a_idx]
            D = X .- Ma
            wa = w[a_idx]
            η_max = wa < 1.0 ? wa / (1 - wa) : 1.0
            kind = "away"
        else
            D = Mv .- X
            η_max = 1.0
            kind = "FW"
        end

        if step_rule === :linesearch
            η = fw_line_search(η -> f_obj(X .+ η .* D), η_max)
        else
            η = min(fw_step_size(iter), η_max)
        end

        if use_away
            w .*= (1 + η)
            w[a_idx] -= η
            if w[a_idx] <= 1e-10
                deleteat!(M, a_idx); deleteat!(BC, a_idx); deleteat!(w, a_idx)
                kind = "drop"
            end
        else
            w .*= (1 - η)
            push!(M, Mv); push!(BC, bundle); push!(w, η)
        end
        sw = sum(w); sw > 0 && (w ./= sw)

        _log_row(g_fw, kind)
    end

    # ---- build the returned market from the best-seen iterate ------------------
    isempty(best_w) && (best_BC = [copy(b) for b in BC]; best_w = copy(w))
    cands, B = _ad_expand_bundles(best_BC, best_w, n)
    fa = ad_market_from_atoms(cands, B)
    m = length(cands)
    γ_tensor = Array{Float64,3}(undef, m, K, n)
    for t in 1:m
        γ_tensor[t, :, :] .= _gamma_over_full_from_cand(Ξ_train, cands[t])
    end
    γ_ref = Ref(γ_tensor)

    if has_test && !isempty(history[:test_err])
        history[:test_err][end] = evaluate_test_error_ad(cands, B, Ξ_test; delta=1.0)
    end
    history[:cands] = cands

    _elapsed = time() - _t0
    if verbose
        @printf("--- done: %d atoms (%d bundles), best obj/K=%.3e, t=%.4fs ---\n",
            m, length(best_w), best_f, _elapsed)
    end
    return fa, γ_ref, history
end
