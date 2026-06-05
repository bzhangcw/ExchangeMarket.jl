# Arrow–Debreu column-generation iteration runner (`run_ad_tracked`).
#
# Sibling of cpm.jl (`run_method_tracked`): same CG loop shape, but driven by
# the Arrow–Debreu master/dual/pricing primitives in redistribute_ad.jl
# (endowments b_t ∈ ℝⁿ₊, Σ_t b_t = δ·1, price-dependent budget ⟨p, b_t⟩;
# supply scale δ fixed via :ad_delta or a free LP variable via :ad_delta = :free).
# Homothetic classes only — no pinning machinery. `adcg` (separation_kind
# :cg_ad) dispatches here from run_one_method (setup.jl), returning the same
# (fa, γ_ref, history) triple as the Fisher CG runners.
#
# Depends on:
#   - solve_wealth_redist_primal_ad, extract_duals_ad, find_cut_single_ad,
#     evaluate_test_error_ad, drop_zero_atoms_ad, ad_market_from_atoms   (redistribute_ad.jl)
#   - add_to_gamma!, _gamma_over_full_from_cand, format_class,
#     nonhomothetic_classes                                     (separation.jl)
#   - BANNER_TITLE, print_banner/print_config/print_header/
#     print_row/print_continuation, table_width, IterTable     (cpm.jl/logging.jl)
#   - AD_TABLE (defined below): the Fisher table plus a supply-scale δ column.

using Printf
using Random

# AD iteration table: the Fisher CPM_TABLE columns plus a supply-scale δ
# column (sec.wealth.ad.scale). Defined here rather than reusing CPM_TABLE so
# the Fisher runner's table stays unchanged (no dead δ column). The δ column
# is constant when --ad-delta is fixed and tracks the master's choice when
# --ad-delta-free.
const AD_TABLE = IterTable(
    ["k", "train", "test", "|∇|", "Δ(fixed-pt)", "T", "δ", "class", "t(s)"],
    ["%5d", "%10.3e", "%10.3e", "%10.3e", "%10.3e", "%5d", "%5.1e", "%14s", "%10.4f"],
    Any[1, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1, 1.0, "ces+lin×10", 1.234],
)

"""
    run_ad_tracked(kwargs, Ξ_train, Ξ_test=nothing; verbosity=1) -> (fa, γ_ref, history)

Arrow–Debreu column-generation loop. Homothetic classes only. Mirrors
`run_method_tracked` (cpm.jl) but with the AD master/dual/pricing.

`kwargs` keys (with defaults): `:classes` ([:ces]), `:max_iters` (500),
`:tol_obj` (1e-3), `:tol_rc` (1e-5), `:tol_delta` (1e-5), `:timelimit` (Inf),
`:drop` (true), `:interval_dropping` (1), `:ad_delta` (1.0 — supply scale,
a fixed number or `:free` for an LP-variable δ, sec.wealth.ad.scale).
A `nothing` tolerance disables the corresponding stop. Any other key is
forwarded to the separation oracle.
Warm restart: `:initial_cands` + `:initial_γ_ref` resume from a prior
phase's atoms / bidding tensor (else bootstrap one random CES atom).

Returns:
- `fa::ArrowDebreuMarket` — the fitted surrogate (CES (c, ρ) atoms + endowments b).
- `γ_ref::Ref` — the (m, K, n) bidding tensor.
- `history::Dict` — `:primal_obj`, `:test_err`, `:num_agents`, `:delta`.
"""
function run_ad_tracked(kwargs::Dict, Ξ_train, Ξ_test=nothing; verbosity::Int=1)
    verbose = verbosity >= 1
    verbose_separation = verbosity >= 2

    classes = Vector{Symbol}(get(kwargs, :classes, Symbol[:ces]))
    @assert !isempty(classes) "AD kwargs :classes must be non-empty"
    nonhomo = nonhomothetic_classes(classes)
    isempty(nonhomo) ||
        error("run_ad_tracked: homothetic classes only; got non-homothetic $(nonhomo). " *
              "Use --classes from {ces,linear}.")
    # The fitted surrogate is returned as an ArrowDebreuMarket, whose atoms
    # live in the CES (c, ρ) parameterization — Leontief sits at the σ → -1
    # boundary where that recovery diverges, so it cannot be stored.
    :leontief in classes &&
        error("run_ad_tracked: :leontief atoms cannot be stored in an " *
              "ArrowDebreuMarket (CES (c, ρ) form diverges at σ = -1). " *
              "Use --classes from {ces,linear}.")

    max_iters = get(kwargs, :max_iters, 500)
    tol_obj = get(kwargs, :tol_obj, 1e-3)
    tol_rc = get(kwargs, :tol_rc, 1e-5)
    tol_delta = get(kwargs, :tol_delta, 1e-5)
    timelimit = get(kwargs, :timelimit, Inf)
    drop = get(kwargs, :drop, true)
    # Prune zero-endowment atoms every `interval_dropping` iterations
    # (regularization; smaller master LP). 1 ⇒ every iteration.
    interval_dropping = get(kwargs, :interval_dropping, 1)
    # Endowment-mask mode (sec.wealth.ad.mask): each new atom's endowment is
    # restricted to a support S_t fixed at generation time.
    #   :single (default) — S_t = {winning good}: master of Fisher size.
    #   :full             — S_t = [n]: the unmasked master (eq.ad.master).
    #   :random           — S_t = {winning good} ∪ (ad_mask_size-1 random goods).
    ad_endow_mode = Symbol(get(kwargs, :ad_endow_mode, :single))
    ad_mask_size = Int(get(kwargs, :ad_mask_size, 1))
    ad_endow_mode in (:single, :full, :random) ||
        error("run_ad_tracked: unknown :ad_endow_mode :$(ad_endow_mode) (use :single, :full, :random)")
    # Supply scale δ (sec.wealth.ad.scale): a fixed number (1.0 = unit-supply
    # master eq.ad.master), or :free (δ ≥ 0 is an LP variable in the master,
    # eq.ad.master.scaled). Cuts/pricing are unchanged in both cases.
    ad_delta = get(kwargs, :ad_delta, 1.0)
    ad_delta === :free || (ad_delta isa Real && ad_delta >= 0) ||
        error("run_ad_tracked: :ad_delta must be a nonnegative number or :free (got $(ad_delta))")
    # Mini-batch size + boosting-style residual weighting (--sample-size /
    # --sample-hard). sample_size still flows to the oracle via oracle_kw
    # (it is NOT in _control); read here only to gate the weight computation.
    sample_size = Int(get(kwargs, :sample_size, 0))
    sample_hard = get(kwargs, :sample_hard, false) === true

    K = length(Ξ_train)
    n = length(Ξ_train[1][1])
    has_test = !isnothing(Ξ_test)

    # Loop/master control keys consumed here; everything else (per-class
    # oracle knobs like ces_sigma_lower/upper, linear M/mip_*, AND
    # :sample_size) is forwarded to the separation oracle, exactly as
    # run_method_tracked splats kwargs into find_cut_single. Subsampling
    # (--sample-size) works per oracle call: find_cut_single subsamples
    # (Ξ, ũ) jointly, solves on the subset, and re-expands + re-scores the
    # candidate over the full data — critical for the linear MIP oracle,
    # whose cost is super-linear in K.
    _control = (:classes, :max_iters, :tol_obj, :tol_rc, :tol_delta, :timelimit,
        :drop, :interval_dropping, :interval_eval_test, :interval_eval_excess,
        :f_real, :seed, :redist_use_nlp, :redist_nonh_w, :tol_stage_2,
        :initial_cands, :initial_γ_ref, :initial_fa, :initial_masks,
        :ad_endow_mode, :ad_mask_size, :ad_delta, :sample_hard)
    oracle_kw = Dict{Symbol,Any}(k => v for (k, v) in kwargs if !(k in _control))

    # Warm restart (phased schedules, run_plc_phased.jl): resume from the
    # previous phase's atoms + bidding tensor when both are supplied, so
    # earlier columns stay in play across a class-menu expansion — the AD
    # analog of run_method_tracked's `initial_fa` / `initial_γ_ref`. The
    # final prune in the prior phase keeps `cands` and `γ_ref` row-aligned.
    # Otherwise bootstrap one random CES atom.
    #
    # The bootstrap atom ALWAYS carries the full mask [n]: it guarantees that
    # every good has at least one owner (the masked supply rows need ≥1
    # variable each), and — because a sole owner of a good is forced to
    # b_{t,l} = 1 by the supply row — it can never be dropped while it is the
    # only owner. Coverage is therefore invariant under dropping.
    cands_init = get(kwargs, :initial_cands, nothing)
    γ_ref_init = get(kwargs, :initial_γ_ref, nothing)
    masks_init = get(kwargs, :initial_masks, nothing)
    if isnothing(cands_init) || isnothing(γ_ref_init)
        atom0 = (class=:ces, params=(y=0.1 .* randn(n), σ=0.5))
        γ0 = _gamma_over_full_from_cand(Ξ_train, atom0)          # K×n
        γ_ref = Ref(reshape(γ0, 1, K, n))
        cands = Any[atom0]
        masks = Vector{Int}[collect(1:n)]
    else
        cands = collect(Any, cands_init)   # mutable copy: push!/cands[keep] below
        γ_ref = γ_ref_init
        # Warm restart without recorded masks ⇒ treat prior atoms as full-mask.
        masks = isnothing(masks_init) ?
                Vector{Int}[collect(1:n) for _ in 1:length(cands)] :
                collect(Vector{Int}, masks_init)
    end

    history = Dict(
        :primal_obj => Float64[],
        :test_err => Float64[],
        # AD surrogate-equilibrium excess isn't wired; kept (NaN-filled) so the
        # history schema matches the Fisher runners (run_test / run_plc_phased
        # read :excess uniformly).
        :excess => Float64[],
        :num_agents => Int[],
        # Supply scale per iteration: constant when :ad_delta is fixed, the
        # master's optimal δ when :ad_delta = :free.
        :delta => Float64[],
    )

    _t0 = time()
    if verbose
        print_banner(AD_TABLE, BANNER_TITLE)
        print_config("method", "cg (Arrow–Debreu)")
        print_config("classes", join(String.(classes), ", "))
        print_config("master", "AD LP; endowments b_t ∈ ℝⁿ₊, Σ_t b_t = δ·1")
        print_config("endow mask", ad_endow_mode === :random ?
                                   "random (|S_t| = $(ad_mask_size))" : String(ad_endow_mode))
        print_config("supply scale δ", ad_delta === :free ?
                                       "free (LP variable, eq.ad.master.scaled)" : @sprintf("%g (fixed)", ad_delta))
        print_config("K (training samples)", K)
        print_config("n (goods)", n)
        sample_size > 0 && print_config("mini-batch", sample_hard ?
            "$(sample_size), residual-weighted (--sample-hard)" : "$(sample_size), uniform")
        print_config("max_iters", max_iters)
        print_config("timelimit (s)", @sprintf("%g", Float64(timelimit)))
        print_config("tol_obj", isnothing(tol_obj) ? "off" : @sprintf("%g", tol_obj))
        print_config("tol_delta", isnothing(tol_delta) ? "off" : @sprintf("%g", tol_delta))
        print_config("tol_|∇|", isnothing(tol_rc) ? "off" : @sprintf("%g", tol_rc))
        println("-"^table_width(AD_TABLE))
        print_header(AD_TABLE)
    end

    # Persistent AD master: appends b-variable blocks for new atoms across
    # iterations instead of rebuilding the K·n balance rows (the dominant
    # per-iteration cost at moderate n). Invalidated whenever dropping
    # reshuffles γ rows (cached b-var ↔ atom-index mapping breaks).
    master_cache = Ref{Any}(nothing)

    B = zeros(Float64, 1, n)
    for iter in 1:max_iters
        if time() - _t0 > timelimit
            verbose && print_continuation(AD_TABLE,
                @sprintf("time limit reached (%.1fs > %.1fs)", time() - _t0, timelimit))
            break
        end
        _remaining = isfinite(timelimit) ? max(1.0, timelimit - (time() - _t0)) : nothing

        B, s_slack, model_primal, balance, supply, δ_val = solve_wealth_redist_primal_ad(
            Ξ_train, γ_ref[]; masks=masks, delta=ad_delta,
            verbose=(iter == 1) && verbose, timelimit=_remaining,
            cache=master_cache)
        primal_obj = objective_value(model_primal) / K
        u, ν = extract_duals_ad(model_primal, balance, supply, K, n)
        # Pricing scan priority: goods with the largest master residual
        # Σ_k |s_{k,j}| first — these are the coordinates where the surrogate
        # mismatches most, hence the most likely to yield a violated cut.
        # With early exit on, pricing typically stops after the first few.
        residual_order = sortperm(vec(sum(abs, s_slack; dims=1)); rev=true)
        # Boosting-style mini-batch (--sample-hard): per-sample residual ‖s_k‖₁
        # = Σ_j |s_{k,j}| (sum over goods). Forwarded to the oracle so the
        # subsample concentrates on hard examples; uniform (nothing) by default,
        # and a no-op unless --sample-size subsamples. The good-scan priority
        # above (residual_order) and this sample priority are orthogonal: one
        # picks coordinates, the other picks samples.
        sample_w = (sample_hard && sample_size > 0) ?
                   vec(sum(abs, s_slack; dims=2)) : nothing

        # Drop atoms whose endowment collapsed to ~0. Duals (u, ν) are
        # unaffected (zero-mass columns don't change the dual solution), and
        # the dropped atoms contribute ~0 to the predictor, so this leaves
        # primal_obj / test_err intact. cands, B, masks are re-sliced to stay
        # aligned with the pruned γ_ref. (Coverage is safe: a sole owner of a
        # good is pinned to b_{t,l} = 1 by the supply row, so it never drops.)
        if drop && interval_dropping > 0 && (iter % interval_dropping == 0)
            ndrop, keep = drop_zero_atoms_ad(cands, γ_ref, B)
            if ndrop > 0
                cands = cands[keep]
                B = B[keep, :]
                masks = masks[keep]
                # Reshuffled γ rows invalidate the cached b-var ↔ atom mapping.
                master_cache[] = nothing
            end
        end

        te = has_test ? evaluate_test_error_ad(cands, B, Ξ_test; delta=δ_val) : NaN
        push!(history[:primal_obj], primal_obj)
        push!(history[:test_err], te)
        push!(history[:excess], NaN)        # surrogate-equilibrium excess not wired for AD
        push!(history[:num_agents], length(cands))
        push!(history[:delta], δ_val)

        improvement = length(history[:primal_obj]) >= 2 ?
                      history[:primal_obj][end-1] - primal_obj : NaN

        _log_row(rc_val=NaN, class_str="-") = verbose && print_row(AD_TABLE,
            Any[iter, primal_obj, te, rc_val, isnan(improvement) ? 0.0 : improvement,
                length(cands), δ_val, class_str, time()-_t0])

        # convergence: average per-sample error.
        if !isnothing(tol_obj) && primal_obj < tol_obj
            _log_row()
            verbose && print_continuation(AD_TABLE,
                @sprintf("converged (obj/K = %.2e < tol_obj=%g)", primal_obj, tol_obj))
            break
        end
        # convergence: fixed-point stall.
        if length(history[:primal_obj]) >= 3 && !isnothing(tol_delta)
            imp2 = max(history[:primal_obj][end-2] - history[:primal_obj][end-1],
                history[:primal_obj][end-1] - primal_obj)
            if imp2 < tol_delta
                _log_row()
                verbose && print_continuation(AD_TABLE,
                    @sprintf("converged (Δ = %.2e < tol_delta=%g)", imp2, tol_delta))
                break
            end
        end

        # AD pricing: up to n oracle calls (one per good), scanned in
        # residual-priority order with early exit — any good whose cut
        # violates by more than the threshold is good enough to add (CG
        # needs *a* violated cut, not the most violated); the full scan only
        # happens near convergence, when no good early-exits (which is what
        # certifies termination). The threshold is tol_rc when the rc-stop
        # is active, else 0 (tol_rc = nothing disables the rc-based STOP,
        # but any strictly violated cut still warrants an early exit —
        # without this, disabling tol_rc silently makes every iteration pay
        # all n oracle calls).
        _early_exit = isnothing(tol_rc) ? zero(eltype(ν)) : tol_rc
        best = find_cut_single_ad(Ξ_train, u, ν, classes;
            verbose=verbose_separation, timelimit=_remaining,
            early_exit_rc=_early_exit, scan_order=residual_order,
            sample_weights=sample_w, oracle_kw...)
        if isnothing(best) || !isfinite(best.rc)
            _log_row()
            verbose && print_continuation(AD_TABLE,
                "no improving column from any good (oracles failed or |∇| = -Inf); terminating")
            break
        end
        rc_val = best.rc
        class_str = format_class(best.class, best.params)
        # homothetic ⇒ rc ≤ 0 means no violated cut anywhere; rc ≤ tol_rc stops.
        if !isnothing(tol_rc) && rc_val <= tol_rc
            _log_row(rc_val, class_str)
            verbose && print_continuation(AD_TABLE,
                @sprintf("converged (|∇| = %.2e ≤ tol_|∇|=%g, %s)", rc_val, tol_rc, class_str))
            break
        end

        add_to_gamma!(γ_ref, best.γ_new)
        push!(cands, (class=best.class, params=best.params))
        # Endowment mask of the new atom: always contains the winning good
        # (so the atom can serve the cut it violates), padded per the mode.
        push!(masks, make_ad_mask(ad_endow_mode, n, best.good; mask_size=ad_mask_size))
        _log_row(rc_val, class_str)
    end

    # Final master solve with all columns, for the returned B / test error,
    # then a final prune so the returned surrogate is minimal.
    B, _, _, _, _, δ_final = solve_wealth_redist_primal_ad(Ξ_train, γ_ref[]; masks=masks,
        delta=ad_delta, verbose=false, cache=master_cache)
    if drop
        ndrop, keep = drop_zero_atoms_ad(cands, γ_ref, B)
        if ndrop > 0
            cands = cands[keep]
            B = B[keep, :]
            masks = masks[keep]
        end
    end
    if has_test && !isempty(history[:test_err])
        history[:test_err][end] = evaluate_test_error_ad(cands, B, Ξ_test; delta=δ_final)
    end
    if !isempty(history[:delta])
        history[:delta][end] = δ_final
    end
    if verbose
        @printf("--- done: %d atoms, obj/K=%.3e, δ=%.4f, t=%.4fs ---\n",
            length(cands), history[:primal_obj][end], δ_final, time() - _t0)
    end

    # Stash the atom list + masks in the history so phased drivers can
    # warm-restart via :initial_cands / :initial_masks (the returned
    # ArrowDebreuMarket stores only the converted (c, ρ) columns, not the
    # oracle params or the masks).
    history[:cands] = cands
    history[:masks] = masks

    # Return the (fa, γ_ref, history) triple expected by run_one_method, so
    # `adcg` dispatches through the same path as the Fisher CG methods.
    # The fitted surrogate is a first-class ArrowDebreuMarket: atoms as CES
    # (c, ρ) columns, the master's endowments B as the market's b.
    fa = ad_market_from_atoms(cands, B)
    return fa, γ_ref, history
end
