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
#   - solve_master_problem, solve_dual_problem, extract_duals      (redistribute.jl)
#   - compute_gamma_from_market                                    (setup.jl)
#   - evaluate_test_error                                          (setup.jl)
#   - drop_zero_columns!, add_to_gamma!, add_column_to_market!,
#     format_class, solve_pricing_multiclass,
#     solve_pricing_inversion_multiclass                           (pricing.jl)

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
    drop = get(kwargs, :drop, true)
    classes = get(kwargs, :classes, Symbol[:ces])
    timelimit = get(kwargs, :timelimit, Inf)        # wall-clock cap, seconds
    @assert !isempty(classes) "method kwargs :classes must be non-empty"

    n = length(Ξ_train[1][1])

    # initialize surrogate market with one random CES agent
    fa = FisherMarket(1, n; ρ=rand(1), scale=30.0, sparsity=0.99)
    γ_ref = Ref(compute_gamma_from_market(fa, Ξ_train))

    K_train = length(Ξ_train)
    n_train = length(Ξ_train[1][1])
    has_test = !isnothing(Ξ_test)

    history = Dict(
        :primal_obj => Float64[],
        :test_err => Float64[],
        :num_agents => Int[],
    )

    _t0 = time()
    # `stage` is a small ordinal that selects the pricing strategy:
    #   stage = 2 ⇒ multicut (K per-sample inversion candidates)
    #   stage = 1 ⇒ single-cut (one improving column from the multiclass dispatcher)
    # A run that starts in multicut auto-demotes to single-cut on stall
    # to clean up; single-cut treats a stall as terminal convergence.
    stage = pricing_kind === :cg_multicut ? 2 : 1
    if verbose
        println("=== $(name) ($(pricing_kind)) ===")
        @printf("%5s | %10s | %10s | %10s | %10s | %5s | %14s | %10s\n",
            "k", "train", "test", "∇(grad)", "Δ(fixed-pt)", "T", "class", "t(s)")
        @printf("%5s-+-%10s-+-%10s-+-%10s-+-%10s-+-%5s-+-%14s-+-%10s\n",
            "-----", "----------", "----------", "----------", "----------", "-----", "--------------", "----------")
    end

    for iter in 1:max_iters
        if time() - _t0 > timelimit
            verbose && @printf("time limit reached (%.1fs > %.1fs)\n", time() - _t0, timelimit)
            break
        end
        w, _, model_primal, balance, budget = solve_master_problem(Ξ_train, γ_ref[]; verbose=false)
        primal_obj = objective_value(model_primal) / K_train
        u, μ = extract_duals(model_primal, balance, budget, K_train, n_train)

        if sanity || (iter == 1)
            u_dual, μ_dual, model_dual = solve_dual_problem(Ξ_train, γ_ref[]; verbose=false)
            dual_obj = objective_value(model_dual) / K_train
            gap = abs(primal_obj - dual_obj)
            @assert gap < 1e-6 "Strong duality violated: primal=$primal_obj, dual=$dual_obj, gap=$gap"
            # check dual variable sign convention
            u_err = norm(u .- u_dual, Inf)
            μ_err = abs(μ - μ_dual)
            if verbose && (u_err > 1e-4 || μ_err > 1e-4)
                @printf("  ⚠ dual mismatch: ‖u - u_dual‖∞=%.2e, |μ - μ_dual|=%.2e\n", u_err, μ_err)
            end
        end

        if drop
            drop_zero_columns!(fa, γ_ref, w)
        else
            fa.w .= w
        end

        # record after weights are set on fa
        te = has_test ? evaluate_test_error(fa, Ξ_test) : NaN
        push!(history[:primal_obj], primal_obj)
        push!(history[:test_err], te)
        push!(history[:num_agents], fa.m)

        # fixed-point improvement
        improvement = length(history[:primal_obj]) >= 2 ?
                      history[:primal_obj][end-1] - primal_obj : NaN

        # helper: format one log row
        function _log_row(rc_val=NaN, class_str::AbstractString="-")
            _elapsed = time() - _t0
            if !verbose
                return
            end
            rc_str = isnan(rc_val) ? @sprintf("%10s", "-") : @sprintf("%10.3e", rc_val)
            te_str = isnan(te) ? @sprintf("%10s", "-") : @sprintf("%10.3e", te)
            Δ_str = @sprintf("%10.3e", isnan(improvement) ? 0.0 : improvement)
            @printf("%5d | %10.3e | %s | %s | %s | %5d | %14s | %10.4f\n",
                iter, primal_obj, te_str, rc_str, Δ_str, fa.m, class_str, _elapsed)
        end

        # convergence: average per-sample error
        if primal_obj < tol_obj
            _log_row()
            verbose && @printf("converged (obj/K = %.2e < tol_obj=%g)\n", primal_obj, tol_obj)
            break
        end

        # convergence: fixed-point stall.
        # In multicut mode, a stall isn't a stop signal — it's a switch
        # signal: hand off to single-cut for cleanup. Single-cut treats
        # the stall as terminal convergence as before.
        if length(history[:primal_obj]) >= 3
            imp2 = max(history[:primal_obj][end-2] - history[:primal_obj][end-1],
                history[:primal_obj][end-1] - primal_obj)
            if imp2 < tol_delta
                if stage > 1
                    verbose && @printf("stage %d stalled (Δ = %.2e < tol_delta=%g); demoting to stage %d\n",
                        stage, imp2, tol_delta, stage - 1)
                    stage -= 1
                    # fall through: this iteration's pricing runs at the
                    # lower stage and logs a single row.
                else
                    _log_row()
                    verbose && @printf("converged (Δ = %.2e < tol_delta=%g)\n", imp2, tol_delta)
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
            cand = solve_pricing_multiclass(Ξ_train, u, μ, classes; verbose=verbose_pricing)
            rc_val = cand.rc
            class_str = format_class(cand.class, cand.params)
            if (rc_val > 0.0) && (rc_val <= tol_rc)
                _log_row(rc_val, class_str)
                verbose && @printf("converged (rc = %.2e ≤ tol_rc=%g, %s)\n",
                    rc_val, tol_rc, class_str)
                break
            end
            add_to_gamma!(γ_ref, cand.γ_new)
            add_column_to_market!(fa, cand.params, cand.class, 0.0)

        elseif stage == 2
            # Multicut: per-sample inversion across the inversion-capable
            # classes. For each k, compute the candidate in every allowed
            # class (CES inverts (y,σ); linear has no σ) and add the one
            # with the higher pricing-oracle objective. Total K atoms per
            # pass — not |classes|·K. Leontief gets picked up later by
            # the single-cut cleanup phase.
            candidates = solve_pricing_inversion_multiclass(Ξ_train, u, classes)
            counts = Dict{Symbol,Int}(:ces => 0, :linear => 0)
            for cand in candidates
                cand.class === :none && continue
                add_to_gamma!(γ_ref, cand.γ_new)
                add_column_to_market!(fa, (y=cand.y, σ=cand.σ), cand.class, 0.0)
                counts[cand.class] = get(counts, cand.class, 0) + 1
            end
            tags = String[]
            counts[:ces]    > 0 && push!(tags, "ces×$(counts[:ces])")
            counts[:linear] > 0 && push!(tags, "lin×$(counts[:linear])")
            class_str = isempty(tags) ? "-" : join(tags, "+")

        else
            error("Unknown pricing stage: $stage")
        end

        # log after pricing
        _log_row(rc_val, class_str)
    end

    # final master solve with latest columns
    w_final, _, _, _, _ = solve_master_problem(Ξ_train, γ_ref[]; verbose=false)
    if drop
        drop_zero_columns!(fa, γ_ref, w_final)
    else
        fa.w .= w_final
    end

    _elapsed = time() - _t0
    if verbose
        @printf("--- done: %d agents, obj/K=%.3e, t=%.4fs ---\n", fa.m, history[:primal_obj][end], _elapsed)
    end

    return fa, γ_ref, history
end
