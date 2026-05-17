# Cutting-plane / column-generation iteration runner (`run_method_tracked`).
#
# Stage semantics:
#   stage = 2 ⇒ multicut (K per-sample inversion candidates, dispatched by
#               solve_pricing_inversion_multiclass across :ces and :linear)
#   stage = 1 ⇒ single-cut (one improving column from solve_pricing_multiclass)
# A run that starts in multicut auto-demotes to single-cut on stall to
# clean up; single-cut treats a stall as terminal convergence.
#
# Verbosity: 0 silent, 1 per-iteration table, 2 + per-class pricing detail
# (forwarded to the pricing functions, e.g., Gurobi MIP log to console).
#
# Depends on:
#   - solve_wealth_redistribution_primal, solve_wealth_redistribution_dual, extract_duals      (redistribute.jl)
#   - compute_gamma_from_market                                    (setup.jl)
#   - evaluate_test_error                                          (setup.jl)
#   - drop_zero_columns!, add_to_gamma!, add_column_to_market!,
#     format_class, solve_pricing_multiclass,
#     solve_pricing_inversion_multiclass                           (pricing.jl)

# Iteration-table layout for run_method_tracked. Widths are computed by
# IterTable from the dummy row; the "class" column's dummy is a realistic
# worst case ("ces+lin×10") so multicut tags don't overflow.
const CPM_TABLE = IterTable(
    ["k", "train", "test", "∇(grad)", "Δ(fixed-pt)", "T", "class", "t(s)"],
    ["%5d", "%10.3e", "%10.3e", "%10.3e", "%10.3e", "%5d", "%14s", "%10.4f"],
    Any[1, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1, "ces+lin×10", 1.234],
)

# Re-expand a pricing-oracle candidate (class, y, σ) to its full K × n
# bidding matrix over Ξ_full. Used when the pricing call ran on a
# random subset of prices but the master needs the full-shape γ.
# CES / Leontief reuse the softmax recipe in `produce_gamma`; :linear
# (σ → +∞) requires the bang-per-buck argmax form from `compute_gamma`.
function _gamma_over_full(Ξ_full, y::AbstractVector, σ::Real, class::Symbol)
    K = length(Ξ_full)
    n = length(y)
    γ = zeros(eltype(y), K, n)
    if class === :linear
        c = exp.(y)
        for k in 1:K
            γ[k, :] = compute_gamma(Ξ_full[k][1], c, σ)
        end
    else
        for k in 1:K
            z_k = y .- σ .* log.(Ξ_full[k][1])
            ez = exp.(z_k .- maximum(z_k))
            γ[k, :] = ez ./ sum(ez)
        end
    end
    return γ
end

"""
    run_method_tracked(name, pricing_kind, kwargs, Ξ_train, Ξ_test=nothing;
                       verbosity=1, sanity=false)

Reimplements the CG loops with per-iteration tracking of primal objective
and test error. Returns `(fa, γ_ref, history)`.
"""
function run_method_tracked(name::Symbol, pricing_kind::Symbol, kwargs::Dict,
    Ξ_train, Ξ_test=nothing; verbosity::Int=1, sanity=false)

    # verbosity levels:
    #   0 = silent (no per-iter table, no pricing detail)
    #   1 = per-iteration table only
    #   2 = + per-class pricing detail (Gurobi MIP log, CES/Leontief lines)
    verbose = verbosity >= 1
    verbose_pricing = verbosity >= 2

    max_iters = get(kwargs, :max_iters, 50)
    tol_obj = get(kwargs, :tol_obj, 5e-3)
    tol_rc = get(kwargs, :tol_rc, 1e-3)
    tol_delta = get(kwargs, :tol_delta, 1e-3)
    tol_stage_2 = get(kwargs, :tol_stage_2, tol_delta)
    drop = get(kwargs, :drop, true)
    # Linear-MILP knobs that the header echoes under the `linear` class.
    use_indicators_kw = get(kwargs, :use_indicators, false)
    # Drop zero-weight columns from the surrogate market on a fixed stride.
    # `interval_dropping = 1` (default) is every iteration as before;
    # set >1 to keep zero-weight atoms longer (cheaper LPs lose this protection,
    # but the master LP can still re-activate dropped atoms via positive weights).
    interval_dropping = get(kwargs, :interval_dropping, 1)
    classes = get(kwargs, :classes, Symbol[:ces])
    # MultiCut-only: after stage-2 demotion, restrict the CES pricer to a
    # fixed ρ (corresponding σ = ρ/(1-ρ)) instead of the free-σ search.
    # ρ near 1 (e.g. 0.97) yields a near-linear CES boundary cleanup.
    # `nothing` ⇒ no restriction; CES pricing keeps the free-σ behavior.
    stage1_ces_rho = get(kwargs, :stage1_ces_rho, nothing)
    timelimit = get(kwargs, :timelimit, Inf)        # wall-clock cap, seconds
    interval_eval_test = get(kwargs, :interval_eval_test, 1)
    # Mini-batch / subsampling for the pricing oracle. When 0 < sample_size < K,
    # each iteration draws a fresh random subset S ⊂ [K] of size sample_size
    # and runs the pricing call on (Ξ_train[S], u[S, :]). For stage-1 single-cut
    # the candidate (y, σ) is then re-expanded to the full K-shape γ via
    # `_gamma_over_full` before being added to the master. For stage-2 multicut
    # only the |S| per-sample inversions are computed (one atom per sampled k),
    # cutting per-iteration cost roughly K / sample_size on the pricing side.
    # See Higle-Sen '91 / Joachims '09 for the cutting-plane subsampling literature.
    sample_size = get(kwargs, :sample_size, 0)
    # Per-iteration market-excess tracking: solves the surrogate equilibrium
    # and evaluates ‖p_s · (q − g_real(p_s))‖_∞ in the real market.
    # `f_real::Union{FisherMarket,Nothing}` — pass via kwargs from test_real.jl.
    # `interval_eval_excess`: 0 or -1 disables, >0 evaluates every N iters.
    f_real = get(kwargs, :f_real, nothing)
    interval_eval_excess = get(kwargs, :interval_eval_excess, 0)
    @assert !isempty(classes) "method kwargs :classes must be non-empty"

    n = length(Ξ_train[1][1])

    # initialize surrogate market with one random CES agent
    ws = cpu_workspace(n)
    add_ces!(ws, 1; ρ=rand(1), scale=30.0, sparsity=0.99)
    ws.ces.w ./= sum(ws.ces.w)
    fa = FisherMarket(ws)
    γ_ref = Ref(compute_gamma_from_market(fa, Ξ_train))
    # Persistent warm-start for the :linear MILP: stashes the (y, γ) of the
    # last winning linear column so the next pricing call hands it to Gurobi
    # as a MIPstart. Often 3–10× faster on subsequent CG rounds.
    linear_warm = Ref{Union{Nothing,NamedTuple{(:y, :γ),Tuple{Vector{Float64},Matrix{Float64}}}}}(nothing)
    # Persistent MILP model: the linear pricing call keeps its Gurobi model
    # alive across iterations and only rewrites the objective when log-prices
    # are unchanged. Wiped automatically when `--sample-size` is active (the
    # cache misses on every call because Ξ_pr changes shape).
    linear_model_cache = Ref{Any}(nothing)
    # Persistent master LP: solve_wealth_redistribution_primal keeps its Gurobi model
    # alive across iterations and appends new w variables (with their
    # balance/budget coefficients) for each freshly added atom. We invalidate
    # the cache whenever `drop_zero_columns!` runs, because dropping
    # reshuffles γ rows and breaks the cached w-variable ↔ atom-index mapping.
    master_cache = Ref{Any}(nothing)

    K_train = length(Ξ_train)
    n_train = length(Ξ_train[1][1])
    has_test = !isnothing(Ξ_test)

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
    # `stage` is a small ordinal that selects the pricing strategy:
    #   stage = 2 ⇒ multicut (K per-sample inversion candidates)
    #   stage = 1 ⇒ single-cut (one improving column from the multiclass dispatcher)
    # A run that starts in multicut auto-demotes to single-cut on stall
    # to clean up; single-cut treats a stall as terminal convergence.
    stage = pricing_kind === :cg_multicut ? 2 : 1
    if verbose
        print_banner(CPM_TABLE, BANNER_TITLE)
        print_config("method", String(name))
        print_config("alias", String(pricing_kind))
        print_config("classes", join(String.(classes), ", "))
        # Per-class parameter sub-block. Only print for classes that are
        # actually selected, so a CES-only run isn't cluttered with linear
        # / leontief defaults.
        if :ces in classes
            ces_extras = if pricing_kind === :cg_multicut && !isnothing(stage1_ces_rho)
                @sprintf("free σ via LBFGS (stage 1) + fixed ρ=%g (stage 2)", stage1_ces_rho)
            elseif pricing_kind === :cg_multicut
                "per-sample inversion (σ-grid [-1.0, 30] × LBFGS refine)"
            else
                "free σ via LBFGS, warm-started by dual-LP"
            end
            print_config("ces", ces_extras; indent=true)
        end
        if :linear in classes
            print_config("linear",
                "Gurobi MILP, " *
                (use_indicators_kw ? "indicator constraints" : "big-M (2·max|log p|)") *
                ", warm-start + model cache"; indent=true)
        end
        :leontief in classes && print_config("leontief", "fixed σ = -1 via LBFGS"; indent=true)
        :ql in classes && print_config("ql", "piecewise-linear-concave QL (w = 1)"; indent=true)
        print_config("K (training samples)", K_train)
        print_config("n (goods)", n)
        print_config("max_iters", max_iters)
        print_config("timelimit (s)", @sprintf("%g", Float64(timelimit)))
        print_config("tol_obj", isnothing(tol_obj) ? "off" : @sprintf("%g", tol_obj))
        print_config("tol_delta", isnothing(tol_delta) ? "off" : @sprintf("%g", tol_delta))
        print_config("tol_rc", isnothing(tol_rc) ? "off" : @sprintf("%g", tol_rc))
        print_config("sample_size", sample_size)
        print_config("interval_dropping", interval_dropping)
        println("-"^table_width(CPM_TABLE))
        print_header(CPM_TABLE)
    end

    for iter in 1:max_iters
        if time() - _t0 > timelimit
            verbose && print_continuation(CPM_TABLE,
                @sprintf("time limit reached (%.1fs > %.1fs)", time() - _t0, timelimit))
            break
        end
        # Pass the remaining time budget to Gurobi so a single master solve
        # can't overshoot the per-method cap by more than its own slack.
        _remaining = isfinite(timelimit) ? max(1.0, timelimit - (time() - _t0)) : nothing
        w, _, model_primal, balance, budget = solve_wealth_redistribution_primal(Ξ_train, γ_ref[];
            verbose=false, timelimit=_remaining, cache=master_cache)
        primal_obj = objective_value(model_primal) / K_train
        u, μ = extract_duals(model_primal, balance, budget, K_train, n_train)

        if sanity || (iter == 1)
            u_dual, μ_dual, model_dual = solve_wealth_redistribution_dual(Ξ_train, γ_ref[]; verbose=false)
            dual_obj = objective_value(model_dual) / K_train
            gap = abs(primal_obj - dual_obj)
            @assert gap < 1e-6 "Strong duality violated: primal=$primal_obj, dual=$dual_obj, gap=$gap"
            # check dual variable sign convention
            u_err = norm(u .- u_dual, Inf)
            μ_err = abs(μ - μ_dual)
            if verbose && (u_err > 1e-4 || μ_err > 1e-4)
                print_continuation(CPM_TABLE,
                    @sprintf("⚠ dual mismatch: ‖u - u_dual‖∞=%.2e, |μ - μ_dual|=%.2e",
                             u_err, μ_err))
            end
        end

        if drop && interval_dropping > 0 && (iter % interval_dropping == 0)
            drop_zero_columns!(fa, γ_ref, w)
            # Dropping reshuffles γ rows; the cached master would point at
            # the wrong w-variable for each atom, so wipe and rebuild next iter.
            master_cache[] = nothing
        else
            fa.w .= w
        end

        # record after weights are set on fa.
        # test_err policy by `interval_eval_test`:
        #   N > 0 → recompute every N iters (forward-fill in between)
        #   N == -1 → never inside the loop; final value filled after the loop
        if has_test && interval_eval_test > 0 && (iter % interval_eval_test == 0)
            last_test_err[] = evaluate_test_error(fa, Ξ_test)
        end
        te = last_test_err[]   # alias for the logging closure below
        # Market-excess (same forward-fill cadence as test).
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

        # fixed-point improvement
        improvement = length(history[:primal_obj]) >= 2 ?
                      history[:primal_obj][end-1] - primal_obj : NaN

        # helper: format one log row. NaN handling for `te` / `rc_val` and
        # the dash-fallback for `improvement` are all delegated to print_row.
        function _log_row(rc_val=NaN, class_str::AbstractString="-")
            verbose || return
            print_row(CPM_TABLE, Any[iter, primal_obj, te, rc_val,
                isnan(improvement) ? 0.0 : improvement,
                fa.m, class_str, time()-_t0])
        end

        # convergence: average per-sample error. `tol_obj === nothing`
        # disables the check (non-stop until tol_delta / max_iters / timelimit).
        if !isnothing(tol_obj) && primal_obj < tol_obj
            _log_row()
            verbose && print_continuation(CPM_TABLE,
                @sprintf("converged (obj/K = %.2e < tol_obj=%g)", primal_obj, tol_obj))
            break
        end

        # convergence: fixed-point stall.
        # In multicut mode, a stall isn't a stop signal — it's a switch
        # signal: hand off to single-cut for cleanup. Single-cut treats
        # the stall as terminal convergence as before.
        # `nothing` on either threshold disables the corresponding action:
        #   tol_stage_2 === nothing ⇒ MultiCut never auto-demotes
        #   tol_delta   === nothing ⇒ single-cut never stall-stops
        if length(history[:primal_obj]) >= 3
            imp2 = max(history[:primal_obj][end-2] - history[:primal_obj][end-1],
                history[:primal_obj][end-1] - primal_obj)
            threshold = stage > 1 ? tol_stage_2 : tol_delta
            if !isnothing(threshold) && imp2 < threshold
                if stage > 1
                    verbose && print_continuation(CPM_TABLE,
                        @sprintf("stage %d stalled (Δ = %.2e < tol_stage_2=%g); demoting to stage %d",
                                 stage, imp2, tol_stage_2, stage - 1))
                    stage -= 1
                    # fall through: this iteration's pricing runs at the
                    # lower stage and logs a single row.
                else
                    _log_row()
                    verbose && print_continuation(CPM_TABLE,
                        @sprintf("converged (Δ = %.2e < tol_delta=%g)", imp2, tol_delta))
                    break
                end
            end
        end

        # pricing / column(s) addition — dispatched by `stage`.
        rc_val = NaN
        class_str = "-"
        if stage == 1
            # Single-cut: one improving column per pass from the multiclass
            # dispatcher. Terminates on small reduced cost.
            #
            # MultiCut post-demotion override: if stage1_ces_rho is set
            # AND we got here by demoting from stage 2 (pricing_kind ===
            # :cg_multicut), run CES at a fixed σ = ρ/(1-ρ) instead of
            # the free-σ search; non-CES classes still go through the
            # regular dispatcher. The best candidate by reduced cost wins.
            apply_fixed_rho = (pricing_kind === :cg_multicut) &&
                              !isnothing(stage1_ces_rho) && (:ces in classes)
            _pricing_remaining = isfinite(timelimit) ?
                                 max(1.0, timelimit - (time() - _t0)) : nothing

            # Subsample for the pricing oracle (Higle-Sen / Joachims style).
            # Master always uses the full K samples; only the separation oracle
            # sees a random subset, then we re-expand γ to K-shape via the
            # candidate's (y, σ) before adding it to the master.
            do_sample = sample_size > 0 && sample_size < K_train
            Ξ_pr, u_pr = if do_sample
                S = randperm(K_train)[1:sample_size]
                Ξ_train[S], u[S, :]
            else
                Ξ_train, u
            end

            # Warm-start payload for the :linear MILP (only used by
            # solve_pricing_linear; other classes ignore the extra kwargs).
            # When --sample-size is on we can't reuse the previous γ as a
            # MIPstart because it was sized for a different Ξ_pr; we still
            # pass y as a heuristic seed.
            _warm_y = isnothing(linear_warm[]) ? nothing : linear_warm[].y
            _warm_γ = (isnothing(linear_warm[]) || do_sample) ? nothing : linear_warm[].γ

            if apply_fixed_rho
                σ_fixed = stage1_ces_rho / (1 - stage1_ces_rho)
                y_opt, σ_out, γ_new, obj_val = solve_pricing_fix_σ(Ξ_pr, u_pr, σ_fixed;
                    timelimit=_pricing_remaining)
                # rc / γ-expansion use the FULL K-shape γ so the master sees
                # the candidate's full bidding profile.
                γ_full = do_sample ? _gamma_over_full(Ξ_train, y_opt, σ_out, :ces) : γ_new
                rc_ces = reduced_cost(γ_full, u, μ)
                cand = (γ_new=γ_full, params=(y=y_opt, σ=σ_out), obj=obj_val,
                    class=:ces, rc=rc_ces)
                other_classes = filter(c -> c !== :ces, classes)
                if !isempty(other_classes)
                    cand_other_sub = solve_pricing_multiclass(Ξ_pr, u_pr, μ, other_classes;
                        verbose=verbose_pricing, timelimit=_pricing_remaining,
                        linear_y_warm=_warm_y, linear_γ_warm=_warm_γ,
                        linear_model_cache=(do_sample ? nothing : linear_model_cache))
                    γ_other_full = do_sample ?
                                   _gamma_over_full(Ξ_train, cand_other_sub.params.y,
                        cand_other_sub.params.σ, cand_other_sub.class) :
                                   cand_other_sub.γ_new
                    cand_other = (γ_new=γ_other_full, params=cand_other_sub.params,
                        obj=cand_other_sub.obj, class=cand_other_sub.class,
                        rc=reduced_cost(γ_other_full, u, μ))
                    if cand_other.rc > cand.rc
                        cand = cand_other
                    end
                end
            else
                cand_sub = solve_pricing_multiclass(Ξ_pr, u_pr, μ, classes;
                    verbose=verbose_pricing, timelimit=_pricing_remaining,
                    linear_y_warm=_warm_y, linear_γ_warm=_warm_γ,
                    linear_model_cache=(do_sample ? nothing : linear_model_cache))
                if do_sample
                    γ_full = _gamma_over_full(Ξ_train, cand_sub.params.y,
                        cand_sub.params.σ, cand_sub.class)
                    cand = (γ_new=γ_full, params=cand_sub.params, obj=cand_sub.obj,
                        class=cand_sub.class, rc=reduced_cost(γ_full, u, μ))
                else
                    cand = cand_sub
                end
            end
            # Stash the winning linear column for next round's MIPstart.
            if cand.class === :linear && all(isfinite, cand.params.y)
                linear_warm[] = (y=Vector{Float64}(cand.params.y),
                    γ=Matrix{Float64}(cand.γ_new))
            end
            rc_val = cand.rc
            class_str = format_class(cand.class, cand.params)
            # `tol_rc === nothing` disables the reduced-cost stop entirely.
            if !isnothing(tol_rc) && (rc_val > 0.0) && (rc_val <= tol_rc)
                _log_row(rc_val, class_str)
                verbose && print_continuation(CPM_TABLE,
                    @sprintf("converged (rc = %.2e ≤ tol_rc=%g, %s)",
                             rc_val, tol_rc, class_str))
                break
            end
            add_to_gamma!(γ_ref, cand.γ_new)
            add_column_to_market!(fa, cand.params, cand.class, 0.0)

        elseif stage == 2
            # Multicut: per-sample inversion across the inversion-capable
            # classes. With `sample_size`, only |S| of the K samples are
            # inverted this round; each candidate's γ is re-expanded to
            # the full K-shape before joining the master so γ_ref stays
            # rectangular. Total atoms added per pass = |S| (or K when
            # sample_size is 0 / >= K). Leontief gets picked up later by
            # the single-cut cleanup phase.
            do_sample = sample_size > 0 && sample_size < K_train
            Ξ_pr, u_pr = if do_sample
                S = randperm(K_train)[1:sample_size]
                Ξ_train[S], u[S, :]
            else
                Ξ_train, u
            end
            candidates = solve_pricing_inversion_multiclass(Ξ_pr, u_pr, classes)
            counts = Dict{Symbol,Int}(:ces => 0, :linear => 0)
            for cand in candidates
                cand.class === :none && continue
                γ_full = do_sample ?
                         _gamma_over_full(Ξ_train, cand.y, cand.σ, cand.class) :
                         cand.γ_new
                add_to_gamma!(γ_ref, γ_full)
                add_column_to_market!(fa, (y=cand.y, σ=cand.σ), cand.class, 0.0)
                counts[cand.class] = get(counts, cand.class, 0) + 1
            end
            tags = String[]
            counts[:ces] > 0 && push!(tags, "ces×$(counts[:ces])")
            counts[:linear] > 0 && push!(tags, "lin×$(counts[:linear])")
            class_str = isempty(tags) ? "-" : join(tags, "+")

        else
            error("Unknown pricing stage: $stage")
        end

        # log after pricing
        _log_row(rc_val, class_str)
    end

    # final master solve with latest columns
    w_final, _, _, _, _ = solve_wealth_redistribution_primal(Ξ_train, γ_ref[];
        verbose=false, cache=master_cache)
    if drop
        drop_zero_columns!(fa, γ_ref, w_final)
        master_cache[] = nothing
    else
        fa.w .= w_final
    end

    _elapsed = time() - _t0
    # Final test_err evaluation: always overwrite the trailing slot with the
    # converged surrogate, so callers see an exact final value regardless of
    # `interval_eval_test`.
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
        @printf("--- done: %d agents, obj/K=%.3e, t=%.4fs ---\n", fa.m, history[:primal_obj][end], _elapsed)
    end

    return fa, γ_ref, history
end
