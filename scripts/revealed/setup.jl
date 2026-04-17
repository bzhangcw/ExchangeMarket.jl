# Methods and tracking utilities for revealed-preference CES surrogate fitting.
# Mirrors the structure of scripts/fisher/setup.jl.

using Printf

include("./master.jl")
include("./pricing.jl")
include("./main.jl")

# -----------------------------------------------------------------------
# methods: (name, pricing_kind, kwargs)
# pricing_kind ∈ {:cg_single, :cg_multicut, :cg_lsq}
# -----------------------------------------------------------------------
method_kwargs = [
    [:CG, :cg_single,
        Dict(
            :max_iters => 300,
            :tol_obj => 1e-3,
            :tol_rc => 1e-5,
            :tol_delta => 1e-5,
            :drop => false,
        )
    ],
    [:Multicut, :cg_multicut,
        Dict(
            :max_iters => 50,
            :tol_obj => 1e-3,
            :tol_rc => 1e-3,
            :tol_delta => 1e-5,
            :drop => true,
        )
    ],
    [:LSQ, :cg_lsq,
        Dict(
            :max_iters => 50,
            :tol_obj => 1e-3,
            :tol_rc => 1e-3,
            :tol_delta => 1e-3,
            :drop => true,
        )
    ],
]

colors = Dict(
    :CG => 1,
    :Multicut => 2,
    :LSQ => 3,
)

marker_style = Dict(
    :CG => :circle,
    :Multicut => :rect,
    :LSQ => :dtriangle,
)

# -----------------------------------------------------------------------
# evaluation on test set: mean L∞ error of Σ w_i γ_i(p) vs target P g
# -----------------------------------------------------------------------
function evaluate_test_error(fa, Ξ_test)
    isempty(fa.w) && return NaN
    K = length(Ξ_test)
    n = length(Ξ_test[1][1])
    errs = Float64[]
    for (p, g) in Ξ_test
        target = p .* g
        fitted = zeros(n)
        for i in 1:fa.m
            c_i = Vector(fa.c[:, i])
            fitted .+= fa.w[i] .* compute_gamma(p, c_i, fa.σ[i])
        end
        push!(errs, norm(fitted .- target, Inf))
    end
    return sum(errs) / K
end

# -----------------------------------------------------------------------
# iteration-level tracked runner. Reimplements the CG loops from main.jl
# but records (primal_obj, test_err) after every master solve.
# -----------------------------------------------------------------------
function run_method_tracked(name::Symbol, pricing_kind::Symbol, kwargs::Dict,
    Ξ_train, Ξ_test; verbose=true, sanity=false)

    max_iters = get(kwargs, :max_iters, 50)
    tol_obj = get(kwargs, :tol_obj, 5e-3)
    tol_rc = get(kwargs, :tol_rc, 1e-3)
    tol_delta = get(kwargs, :tol_delta, 1e-3)
    drop = get(kwargs, :drop, true)

    n = length(Ξ_train[1][1])

    # initialize surrogate market with one random CES agent
    fa = FisherMarket(1, n; ρ=rand(1), scale=30.0, sparsity=0.99)
    γ_ref = Ref(compute_gamma_from_market(fa, Ξ_train))

    K_train = length(Ξ_train)
    n_train = length(Ξ_train[1][1])

    history = Dict(
        :primal_obj => Float64[],
        :test_err => Float64[],
        :num_agents => Int[],
    )

    _t0 = time()
    if verbose
        println("=== $(name) ($(pricing_kind)) ===")
        @printf("%5s | %10s | %10s | %10s | %10s | %5s | %10s\n",
            "k", "train", "test", "rc", "Δ(fp)", "T", "t(s)")
        @printf("%5s-+-%10s-+-%10s-+-%10s-+-%10s-+-%5s-+-%10s\n",
            "-----", "----------", "----------", "----------", "----------", "-----", "----------")
    end

    for iter in 1:max_iters
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
        te = evaluate_test_error(fa, Ξ_test)
        push!(history[:primal_obj], primal_obj)
        push!(history[:test_err], te)
        push!(history[:num_agents], fa.m)

        # fixed-point improvement
        improvement = length(history[:primal_obj]) >= 2 ?
                      history[:primal_obj][end-1] - primal_obj : NaN

        # convergence: average per-sample error
        if primal_obj < tol_obj
            _elapsed = time() - _t0
            if verbose
                @printf("%5d | %10.3e | %10.3e | %10s | %10.3e | %5d | %10.4f\n",
                    iter, primal_obj, te, "-", isnan(improvement) ? 0.0 : improvement, fa.m, _elapsed)
                @printf("converged (obj/K = %.2e < tol_obj=%g)\n", primal_obj, tol_obj)
            end
            break
        end

        # convergence: fixed-point stall
        if length(history[:primal_obj]) >= 3
            imp2 = max(history[:primal_obj][end-2] - history[:primal_obj][end-1],
                history[:primal_obj][end-1] - primal_obj)
            if imp2 < tol_delta
                _elapsed = time() - _t0
                if verbose
                    @printf("%5d | %10.3e | %10.3e | %10s | %10.3e | %5d | %10.4f\n",
                        iter, primal_obj, te, "-", improvement, fa.m, _elapsed)
                    @printf("converged (Δ = %.2e < tol_delta=%g)\n", imp2, tol_delta)
                end
                break
            end
        end

        # pricing / column(s) addition
        rc_val = NaN
        if pricing_kind === :cg_single
            y_lp, σ_lp, _, _ = solve_pricing_dual_lp(Ξ_train, u)
            y_opt, σ_opt, γ_new, _ = solve_pricing(Ξ_train, u; y_init=y_lp, σ_init=σ_lp)
            rc_val = reduced_cost(γ_new, u, μ)
            if rc_val <= tol_rc
                _elapsed = time() - _t0
                if verbose
                    @printf("%5d | %10.3e | %10.3e | %10.3e | %10.3e | %5d | %10.4f\n",
                        iter, primal_obj, te, rc_val, isnan(improvement) ? 0.0 : improvement, fa.m, _elapsed)
                    @printf("converged (rc = %.2e ≤ tol_rc=%g)\n", rc_val, tol_rc)
                end
                break
            end
            add_to_gamma!(γ_ref, γ_new)
            c_new, ρ_new = recover_ces_params(y_opt, σ_opt)
            add_to_market!(fa, c_new, ρ_new, 0.0)

        elseif pricing_kind === :cg_multicut
            candidates = solve_pricing_inversion(Ξ_train, u)
            for (y_opt, σ_opt, γ_new, _) in candidates
                add_to_gamma!(γ_ref, γ_new)
                c_new, ρ_new = recover_ces_params(y_opt, σ_opt)
                add_to_market!(fa, c_new, ρ_new, 0.0)
            end

        elseif pricing_kind === :cg_lsq
            y_opt, σ_opt, γ_new, _ = solve_pricing_leastsq(Ξ_train, u; verbose=false)
            add_to_gamma!(γ_ref, γ_new)
            c_new, ρ_new = recover_ces_params(y_opt, σ_opt)
            add_to_market!(fa, c_new, ρ_new, 0.0)

        else
            error("Unknown pricing_kind: $pricing_kind")
        end

        # log after pricing
        _elapsed = time() - _t0
        if verbose
            rc_str = isnan(rc_val) ? @sprintf("%10s", "-") : @sprintf("%10.3e", rc_val)
            @printf("%5d | %10.3e | %10.3e | %s | %10.3e | %5d | %10.4f\n",
                iter, primal_obj, te, rc_str, isnan(improvement) ? 0.0 : improvement, fa.m, _elapsed)
        end
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
