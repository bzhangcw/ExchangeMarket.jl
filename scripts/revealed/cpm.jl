# Cutting-plane / column-generation iteration runner (`run_method_tracked`).
#
# Stage semantics:
#   stage = 2 ⇒ multicut (K per-sample inversion candidates, handled by
#               solve_separation_multicut across :ces and :linear)
#   stage = 1 ⇒ single-cut (one improving column from solve_separation)
# A run that starts in multicut auto-demotes to single-cut on stall to
# clean up; single-cut treats a stall as terminal convergence.
#
# The master is always the LP `solve_wealth_redist_primal`
# (eq.wealth.hybrid.lp in overleaf/read-econ/wealth-dist.tex). Homothetic
# atoms have free non-negative weights; non-homothetic atoms are pinned
# at a shared wealth w_0 (`--redist-nonh-w`, default 1/max_iters) and
# contribute γ rows precomputed at that w_0 by the separation oracle.
# The routing — which class is homothetic, which isn't — is driven by
# `is_homothetic(::Symbol)` declared in each android file (currently only
# :ql is non-homothetic). The CLI flag `--redist-use-nlp` is reserved
# for a future MadNLP-based nonlinear master and currently errors with
# NotImplemented.
#
# Verbosity: 0 silent, 1 per-iteration table, 2 + per-class separation detail
# (forwarded to the separation oracles, e.g., Gurobi MIP log to console).
#
# Depends on:
#   - solve_wealth_redist_primal, solve_wealth_redist_dual, extract_duals      (redistribute.jl)
#   - compute_gamma_from_market                                    (setup.jl)
#   - evaluate_test_error                                          (setup.jl)
#   - drop_zero_columns!, add_to_gamma!, add_column_to_market!,
#     format_class, solve_separation, solve_separation_multicut,
#     is_homothetic, nonhomothetic_classes                  (separation.jl)
#   - clear_linear_separation_cache!                       (androids/linear.jl)

using ArgParse

# ---- CLI surface --------------------------------------------------------
"""
    register_cli_cpm!(s::ArgParseSettings)

Add CG/cgma's CLI flags (`--stage-2-tol`, `--android-dropping-interval`)
to the given ArgParseSettings as a "Method: CG/cgma" arg group.
"""
function register_cli_cpm!(s::ArgParseSettings)
    add_arg_group!(s, "Method: CG/cgma")
    @add_arg_table! s begin
        "--stage-2-tol"
        help = "cgma only: improvement threshold below which stage 2 auto-demotes to stage 1 (single-cut cleanup). > 0 overrides the per-method :tol_stage_2; ≤ 0 (default) keeps the in-code default (typically equals :tol_delta)."
        arg_type = Float64
        default = -1.0
        "--android-dropping-interval"
        help = "Drop zero-weight androids (columns) from the surrogate market every N iterations (working like regularization). 5 (default) is every iter; larger keeps dormant androids longer (cheaper LP, but more clutter in the surrogate)."
        arg_type = Int
        default = 5
    end
    return s
end

"""
    apply_cli_cpm!(local_extra::Dict, cli)

Forward CG/cgma-specific CLI values into the runner kwargs.
"""
function apply_cli_cpm!(local_extra::Dict, cli)
    if cli["android_dropping_interval"] > 0
        local_extra[:interval_dropping] = cli["android_dropping_interval"]
    end
    if cli["stage_2_tol"] > 0
        local_extra[:tol_stage_2] = cli["stage_2_tol"]
    end
    return local_extra
end

# Iteration-table layout for run_method_tracked. Widths are computed by
# IterTable from the dummy row; the "class" column's dummy is a realistic
# worst case ("ces+lin×10") so multicut tags don't overflow.
const CPM_TABLE = IterTable(
    # `|∇|` is the per-iteration reduced cost (a.k.a. CG "gradient
    # magnitude") of the winning candidate column — what the CLI knob
    # `--tol-rc` thresholds.
    ["k", "train", "test", "|∇|", "Δ(fixed-pt)", "T", "class", "t(s)"],
    ["%5d", "%10.3e", "%10.3e", "%10.3e", "%10.3e", "%5d", "%14s", "%10.4f"],
    Any[1, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1, "ces+lin×10", 1.234],
)

# Separation wrappers (find_cut_single / find_cuts_multi / _gamma_over_full)
# live in separation.jl so accpm.jl can share them.

"""
    run_method_tracked(name, separation_kind, kwargs, Ξ_train, Ξ_test=nothing;
                       verbosity=1, sanity=false)

Reimplements the CG loops with per-iteration tracking of primal objective
and test error. Returns `(fa, γ_ref, history)`.
"""
function run_method_tracked(name::Symbol, separation_kind::Symbol, kwargs::Dict,
    Ξ_train, Ξ_test=nothing; verbosity::Int=1, sanity=false)

    # verbosity levels:
    #   0 = silent (no per-iter table, no separation detail)
    #   1 = per-iteration table only
    #   2 = + per-class separation detail (Gurobi MIP log, CES/Leontief lines)
    verbose = verbosity >= 1
    verbose_separation = verbosity >= 2

    max_iters = get(kwargs, :max_iters, 50)
    tol_obj = get(kwargs, :tol_obj, 5e-3)
    tol_rc = get(kwargs, :tol_rc, 1e-3)
    tol_delta = get(kwargs, :tol_delta, 1e-3)
    tol_stage_2 = get(kwargs, :tol_stage_2, tol_delta)
    drop = get(kwargs, :drop, true)
    # Drop zero-weight columns from the surrogate market on a fixed stride.
    # `interval_dropping = 1` (default) is every iteration as before;
    # set >1 to keep zero-weight androids longer (cheaper LPs lose this protection,
    # but the master LP can still re-activate dropped androids via positive weights).
    interval_dropping = get(kwargs, :interval_dropping, 1)
    classes = get(kwargs, :classes, Symbol[:ces])
    # CLI-driven master selection. The NLP master is reserved for a future
    # MadNLP integration; until then, --redist-use-nlp is a hard error so
    # callers don't silently fall into a stub.
    use_nlp = get(kwargs, :redist_use_nlp, false)
    if use_nlp
        error("--redist-use-nlp: NLP master not implemented yet (planned via MadNLP). " *
              "Run without --redist-use-nlp to use the LP master with pinned non-homothetic weights.")
    end
    # Stage-2 (multicut) per-sample inversion is only defined for the
    # homothetic inversion-capable classes (:ces / :linear). Non-homothetic
    # classes (e.g., :ql) are silently skipped in stage 2 but come back
    # online after the auto-demotion to stage 1 (single-cut), where the LP
    # pinning machinery handles them. Computed once here so banner and
    # stage-2 dispatch share one list.
    classes_homo = homothetic_classes(classes)
    if separation_kind === :cg_multicut
        nh = nonhomothetic_classes(classes)
        if !isempty(nh) && verbosity >= 1
            @info "cgma stage 2 (multicut) only inverts homothetic classes; " *
                  "$(nh) deferred to stage 1 (single-cut) cleanup."
        end
    end
    # Pinned wealth w_0 for non-homothetic atoms in the LP master.
    # CLI default -1 ⇒ resolve here to 1/max_iters (the simplex constraint
    # Σ_t w_t = 1 then bounds |N| ≤ ⌊1/w_0⌋ = max_iters).
    nonh_w = let cli_w = get(kwargs, :redist_nonh_w, -1.0)
        Float64(cli_w > 0 ? cli_w : 1.0 / max_iters)
    end
    timelimit = get(kwargs, :timelimit, Inf)        # wall-clock cap, seconds
    interval_eval_test = get(kwargs, :interval_eval_test, 1)
    # Mini-batch / subsampling for the separation oracle. When 0 < sample_size < K,
    # each iteration draws a fresh random subset S ⊂ [K] of size sample_size
    # and runs the separation call on (Ξ_train[S], u[S, :]). For stage-1 single-cut
    # the candidate (y, σ) is then re-expanded to the full K-shape γ via
    # `_gamma_over_full` before being added to the master. For stage-2 multicut
    # only the |S| per-sample inversions are computed (one android per sampled k),
    # cutting per-iteration cost roughly K / sample_size on the separation side.
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

    # Warm-restart support: callers can pass `:initial_fa => fa` to resume CG
    # with an existing surrogate market (e.g. phased schedules that switch
    # `classes` between calls while preserving previously generated columns).
    # When absent, bootstrap as before with a single random CES atom.
    #
    # `:initial_γ_ref => γ_ref` carries the matching bidding tensor. Without
    # it, we'd have to rebuild γ via `compute_gamma_from_market`, which only
    # knows the CES analytic demand — wrong for any Leontief/Linear/GES atom
    # that lives in `fa.storage.gen`. The previous phase's γ_ref is already
    # row-aligned with `fa.storage.routing` (drop_zero_columns! prunes both
    # together at the post-loop tail), so threading it through preserves
    # primal-objective continuity across phase boundaries.
    fa = get(kwargs, :initial_fa, nothing)
    γ_ref_init = get(kwargs, :initial_γ_ref, nothing)
    if isnothing(fa)
        ws = cpu_workspace(n)
        add_ces!(ws, 1; ρ=rand(1), scale=30.0, sparsity=0.99)
        ws.ces.w ./= sum(ws.ces.w)
        fa = FisherMarket(ws)
    end
    γ_ref = isnothing(γ_ref_init) ?
            Ref(compute_gamma_from_market(fa, Ξ_train)) :
            γ_ref_init
    # Persistent master LP: solve_wealth_redist_primal keeps its Gurobi model
    # alive across iterations and appends new w variables (with their
    # balance/budget coefficients and any pinning constraints) for each
    # freshly added android. We invalidate the cache whenever
    # `drop_zero_columns!` runs, because dropping reshuffles γ rows and
    # breaks the cached w-variable ↔ android-index mapping.
    master_cache = Ref{Any}(nothing)

    # Helper: indices of pinned (non-homothetic) atoms, derived from the
    # storage routing. The :ces substore holds (genuine) homothetic CES /
    # linear atoms; the :gen substore is a typed-agent channel that holds
    # both non-homothetic atoms (e.g., QuasiLinearLogAgent) AND some
    # homothetic atoms that can't ride the CES (c, ρ) form (e.g.,
    # LeontiefAgent — the σ → -1 limit blows up under (c, ρ) recovery).
    # Pinning is needed only for the non-homothetic ones, so we dispatch
    # on the agent type via `agent_is_homothetic` rather than blanket-pinning
    # everything in `:gen`. Recomputed each iteration since the routing
    # grows with each appended atom.
    agent_is_homothetic(::LeontiefAgent) = true
    agent_is_homothetic(::QuasiLinearLogAgent) = false
    agent_is_homothetic(::GESAgent) = false
    agent_is_homothetic(a) = error("agent_is_homothetic: undeclared for $(typeof(a))")
    _pinned_idx() = [i for (i, (sub, j)) in enumerate(fa.storage.routing)
                     if sub === :gen && !agent_is_homothetic(fa.storage.gen.agents[j])]

    # Scatter master-solution weights `w` back into ces.w / gen.w via
    # routing. Replaces the old `fa.w .= w` shim, which only routes to
    # ces.w and would length-mismatch once gen.m > 0.
    function _scatter_weights!(w::AbstractVector)
        ws = fa.storage
        @assert length(w) == length(ws.routing) "w length $(length(w)) ≠ routing length $(length(ws.routing))"
        for (i, (sub, j)) in enumerate(ws.routing)
            if sub === :ces
                ws.ces.w[j] = w[i]
            else
                ws.gen.w[j] = w[i]
            end
        end
    end

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
    # `stage` is a small ordinal that selects the separation strategy:
    #   stage = 2 ⇒ multicut (K per-sample inversion candidates)
    #   stage = 1 ⇒ single-cut (one improving column from the per-class separation oracle)
    # A run that starts in multicut auto-demotes to single-cut on stall
    # to clean up; single-cut treats a stall as terminal convergence.
    stage = separation_kind === :cg_multicut ? 2 : 1
    if verbose
        print_banner(CPM_TABLE, BANNER_TITLE)
        print_config("method", String(name))
        print_config("alias", String(separation_kind))
        print_config("classes", join(String.(classes), ", "))
        # Per-class parameter sub-block. Only print for classes that are
        # actually selected, so a CES-only run isn't cluttered with linear
        # / leontief defaults.
        # Per-class banner rows — one `*_config_summary` per class lives next
        # to its oracle (e.g., androids/ces.jl); the shared dispatcher loops
        # over `classes` and routes to each. Same call shape in accpm.jl.
        print_class_configs(classes, kwargs;
            is_multicut=(separation_kind === :cg_multicut), nonh_w=nonh_w)
        print_config("master",
            @sprintf("LP; non-homothetic androids with fixed wealth"))
        print_config("K (training samples)", K_train)
        print_config("n (goods)", n)
        print_config("max_iters", max_iters)
        print_config("timelimit (s)", @sprintf("%g", Float64(timelimit)))
        print_config("tol_obj", isnothing(tol_obj) ? "off" : @sprintf("%g", tol_obj))
        print_config("tol_delta", isnothing(tol_delta) ? "off" : @sprintf("%g", tol_delta))
        print_config("tol_|∇|", isnothing(tol_rc) ? "off" : @sprintf("%g", tol_rc))
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
        # Show Gurobi's barrier log on the FIRST master solve only — useful
        # to confirm Method=2 / Crossover=0 are taking effect and to spot
        # numerical issues early — and stay silent afterward.
        w, _, model_primal, balance, budget = solve_wealth_redist_primal(
            Ξ_train, γ_ref[];
            verbose=(iter == 1) && verbose,
            timelimit=_remaining,
            pinned_idx=_pinned_idx(),
            pinned_w=nonh_w,
            cache=master_cache,
        )
        primal_obj = objective_value(model_primal) / K_train
        u, μ = extract_duals(model_primal, balance, budget, K_train, n_train)

        if drop && interval_dropping > 0 && (iter % interval_dropping == 0)
            # drop_zero_columns! now prunes both substores, rebuilds
            # routing, and returns the kept indices so we can re-scatter
            # the surviving weights without a tolerance mismatch.
            _, _keep = drop_zero_columns!(fa, γ_ref, w)
            # Routing-aware scatter: writes each w[i] to ws.ces.w[j] or
            # ws.gen.w[j] per the rebuilt routing. `fa.w .= w` would only
            # work for pure-CES (shim → ces.w of length ces.m); with gen
            # agents the lengths disagree.
            _scatter_weights!(w[_keep])
            # Reshuffled γ rows invalidate both the master cache (w-var ↔
            # atom-index mapping changes) and the linear separation cache
            # (warm-start (y, γ) refers to old column positions).
            master_cache[] = nothing
            clear_linear_separation_cache!()
        else
            _scatter_weights!(w)
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
        #   tol_stage_2 === nothing ⇒ cgma never auto-demotes
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
                    # fall through: this iteration's separation runs at the
                    # lower stage and logs a single row.
                else
                    _log_row()
                    verbose && print_continuation(CPM_TABLE,
                        @sprintf("converged (Δ = %.2e < tol_delta=%g)", imp2, tol_delta))
                    break
                end
            end
        end

        # Separation / column(s) addition — selected by `stage`. Both
        # branches delegate to the shared find_cut_* wrappers in separation.jl.
        rc_val = NaN
        class_str = "-"
        _separation_remaining = isfinite(timelimit) ?
                                max(1.0, timelimit - (time() - _t0)) : nothing
        if stage == 1
            # Single-cut: one improving column per pass across all classes.
            cand = find_cut_single(
                Ξ_train, u, μ, classes;
                # Shared pinning wealth for non-homothetic oracles
                # (eq.wealth.hybrid.lp). Forwarded explicitly through every
                # layer; homothetic oracles ignore it.
                nonh_w=nonh_w,
                sample_size=sample_size,
                verbose=verbose_separation,
                timelimit=_separation_remaining,
                kwargs...
            )   # forward class-specific knobs (:nn_hidden, :nn_iters, ...)
            # find_cut_single returns nothing when every class's separator
            # either failed or produced a NaN/-Inf reduced cost (e.g.,
            # MadNLP non-convergence on the GES oracle). Treat that as
            # terminal convergence: no improving column exists at the current
            # duals.
            if isnothing(cand) || !isfinite(cand.rc)
                _log_row()
                verbose && print_continuation(CPM_TABLE,
                    "no improving column from any class (separators failed or |∇| = -Inf); terminating")
                break
            end
            rc_val = cand.rc
            class_str = format_class(cand.class, cand.params)
            # `tol_rc === nothing` disables the reduced-cost stop entirely.
            if !isnothing(tol_rc) && (rc_val > 0.0) && (rc_val <= tol_rc)
                _log_row(rc_val, class_str)
                verbose && print_continuation(CPM_TABLE,
                    @sprintf("converged (|∇| = %.2e ≤ tol_|∇|=%g, %s)",
                        rc_val, tol_rc, class_str))
                break
            end
            # Hard guard for non-homothetic candidates: only add when
            # rc > 0. Wealth-dist.tex (\eqref{eq.wealth.hybrid.pricing})
            # shows that for a non-homothetic atom pinned at w_0, adding
            # a column with rc ≤ 0 strictly worsens the master objective
            # by w_0 · |rc| — the dual must set ν_t = rc to maintain
            # feasibility, and that ν_t enters the objective with
            # coefficient -w_0. Homothetic atoms with free w_t ≥ 0 are
            # safe at rc ≤ 0 (master sets w_t = 0 in the new optimum),
            # so the gate is class-specific.
            if !is_homothetic(cand.class) && rc_val <= 0.0
                _log_row(rc_val, class_str)
                verbose && print_continuation(CPM_TABLE,
                    @sprintf("skipped non-homothetic %s with |∇| = %.2e ≤ 0 (would worsen master under pinning)",
                        class_str, rc_val))
                # Don't add the column, but don't terminate either —
                # the next iteration's separator may still find an
                # improving homothetic column.
                continue
            end
            add_to_gamma!(γ_ref, cand.γ_new)
            # add_column_to_market! routes homothetic classes into ws.ces
            # (via add_to_market!) and non-homothetic into ws.gen (via
            # add_gen!), extending ws.routing in both cases so γ_ref's
            # first dim stays in sync with ws.m. Non-homothetic atoms enter
            # at the pinning wealth nonh_w; their γ row was evaluated at
            # the same w_0 inside the separation oracle (the `nonh_w` kwarg
            # forwarded above), matching the model in
            # eq.wealth.hybrid.predictor.
            w_for_add = is_homothetic(cand.class) ? 0.0 : nonh_w
            add_column_to_market!(fa, cand.params, cand.class, w_for_add)

        elseif stage == 2
            # Multicut: per-sample inversion across the inversion-capable
            # classes. Restricted to homothetic classes — solve_separation_multicut
            # only inverts :ces / :linear, and non-homothetic atoms would
            # need their own pinning machinery in the inversion. They are
            # picked up after the auto-demotion to stage 1 (single-cut),
            # where the LP-with-pinning path handles them. Each candidate
            # added with weight 0; the master LP can re-activate dropped
            # androids via positive weights. Leontief is also deferred to
            # the stage-1 cleanup phase (no per-sample inverter for it).
            # `kwargs...` forwards class-specific knobs (e.g., ces_sigma_lower /
            # ces_sigma_upper for the CES inversion σ-grid) to the per-class
            # inverters; unrelated classes silently absorb them.
            cands = find_cuts_multi(Ξ_train, u, classes_homo;
                sample_size=sample_size, kwargs...)
            for c in cands
                add_to_gamma!(γ_ref, c.γ_new)
                # Multicut produces only homothetic classes (:ces, :linear,
                # :leontief). :ces / :linear ride the CES (c, ρ) storage
                # via add_to_market!; :leontief is routed natively into
                # GenStore as a LeontiefAgent by add_column_to_market!.
                add_column_to_market!(fa, c.params, c.class, 0.0)
            end
            class_str = format_cuts_tag(cands)

        else
            error("Unknown separation stage: $stage")
        end

        # log after separation
        _log_row(rc_val, class_str)
    end

    # final master solve with latest columns
    w_final, _, _, _, _ = solve_wealth_redist_primal(
        Ξ_train, γ_ref[];
        verbose=false,
        pinned_idx=_pinned_idx(),
        pinned_w=nonh_w,
        cache=master_cache,
    )
    if drop
        _, _keep = drop_zero_columns!(fa, γ_ref, w_final)
        _scatter_weights!(w_final[_keep])
        master_cache[] = nothing
        clear_linear_separation_cache!()
    else
        _scatter_weights!(w_final)
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
