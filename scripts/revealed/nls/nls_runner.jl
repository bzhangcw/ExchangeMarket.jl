# Nonlinear-least-squares (NLS) surrogate fit — the "fixed number of agents"
# alternative to column generation.
#
# Where CRM (cpm.jl) *grows* the surrogate one android at a time (alternating
# an LP master with a nonconvex separation oracle), the NLS runner fixes the
# count t up front and solves for ALL CES parameters (yᵢ, σᵢ) AND the wealth
# split w ∈ Δᵗ jointly in one nonconvex program (see nls.jl::solve_nls, solved
# by MadNLP). This is the natural baseline for the question "can the ground
# truth be matched by t CES agents at all, irrespective of how CG discovers
# them?".
#
# Contract: `run_method_tracked_nls` returns `(fa, γ_ref, history)`, the same
# triple every other runner (run_method_tracked, *_fwjl, *_accpm) returns, so
# the driver (run_nls.jl) reuses validate_surrogate / evaluate_test_error /
# save_run unchanged. The fitted (Y, σ, w) is materialized as a genuine
# CES `FisherMarket` via the same add_column_to_market! path CG uses, so the
# surrogate is structurally identical to a CG surrogate (all atoms in the
# `ces` substore, `gen.m == 0`) and validates the same way.
#
# A `:budget => :ad` kwarg switches the surrogate to Arrow–Debreu: the
# fit solves for endowments b_i ∈ ℝⁿ₊ (Σ_i b_i = 1) instead of scalar weights
# w ∈ Δᵗ (nls_ad.jl::solve_nls_ad), the surrogate is an `ArrowDebreuMarket`
# (ad_market_from_atoms), and error is scored by `evaluate_test_error_ad`.
# This mirrors how `adcg` is the AD sibling of `cg`. Equilibrium validation is
# CES-Fisher-only in this codebase, so the driver skips it for AD surrogates.
#
# Depends on (all from ../setup.jl and its includes):
#   - solve_nls, evaluate_nls                         (nls.jl)
#   - solve_nls_ad, evaluate_nls_ad                   (nls_ad.jl)
#   - produce_gamma, compute_gamma_from_market        (androids/ces.jl, setup.jl)
#   - add_to_gamma!, add_column_to_market!, drop_zero_columns!   (separation.jl)
#   - ad_market_from_atoms, evaluate_test_error_ad    (redistribute_ad.jl)
#   - cpu_workspace, add_ces!, FisherMarket           (ExchangeMarket)
#   - evaluate_test_error                             (setup.jl)

using Printf

# Scatter a master/NLS weight vector back into the workspace substores via
# routing. All NLS atoms are CES (→ ws.ces), but we route generically to
# match the cpm.jl convention and stay correct if that ever changes.
function _nls_scatter_weights!(fa::FisherMarket, w::AbstractVector)
    ws = fa.storage
    @assert length(w) == length(ws.routing) "w length $(length(w)) ≠ routing length $(length(ws.routing))"
    for (i, (sub, j)) in enumerate(ws.routing)
        if sub === :ces
            ws.ces.w[j] = w[i]
        else
            ws.gen.w[j] = w[i]
        end
    end
    return fa
end

"""
    build_fa_from_nls(Ξ_train, Y, σ_vec, w) -> (fa::FisherMarket, γ_ref)

Materialize an NLS solution as a CES surrogate market. Each agent i is the
CES android γᵢ(p) = softmax(Yᵢ - σᵢ log p) with wealth wᵢ — exactly the
parameterization the CES separation oracle emits, so we reuse the blessed
`add_column_to_market!(...; :ces)` path (which applies the gauge fix and the
(c, ρ) recovery). We bootstrap with one random CES placeholder (weight 0) so
`add_column_to_market!` has a market to expand, then `drop_zero_columns!`
removes it — the same bootstrap-then-prune pattern as run_method_tracked.

Returns the market and the row-aligned bidding tensor `γ_ref` (used by the
driver only if it wants to warm-start a CG phase from the NLS fit).
"""
function build_fa_from_nls(Ξ_train, Y::Matrix{T}, σ_vec::Vector{T}, w::AbstractVector) where T
    n = length(Ξ_train[1][1])
    t = size(Y, 1)
    @assert length(σ_vec) == t && length(w) == t

    # Bootstrap placeholder (weight normalized to 1, later zeroed and dropped).
    ws = cpu_workspace(n)
    add_ces!(ws, 1; ρ=rand(1), scale=30.0, sparsity=0.99)
    ws.ces.w ./= sum(ws.ces.w)
    fa = FisherMarket(ws)
    γ_ref = Ref(compute_gamma_from_market(fa, Ξ_train))   # 1×K×n: the placeholder row

    for i in 1:t
        # produce_gamma uses the RAW yᵢ; add_column_to_market! stores the
        # gauge-shifted (max(y)=0) atom. softmax is shift-invariant, so the
        # γ_ref row and the stored atom agree at every price.
        add_to_gamma!(γ_ref, produce_gamma(Ξ_train, Y[i, :], σ_vec[i]))
        add_column_to_market!(fa, (y=Y[i, :], σ=σ_vec[i]), :ces, T(w[i]))
    end

    # Drop the placeholder (weight 0). NLS weights come from a softmax so they
    # are strictly positive; any android whose wealth underflows the drop
    # tolerance (1e-8) contributes nothing and is pruned too.
    w_full = vcat(zero(T), T.(collect(w)))
    _, keep = drop_zero_columns!(fa, γ_ref, w_full)
    _nls_scatter_weights!(fa, w_full[keep])
    return fa, γ_ref
end

# Pad a previous (smaller) NLS solution to warm-start the next, larger fit.
# solve_nls fills the first `size(Y_init, 1)` agents from the warm start and
# randomizes the rest, so we pass the previous (Y, σ) straight through and let
# the new agents start fresh. We deliberately pass `w_init = nothing` (uniform
# α₀) rather than padding the weights, since the previous split is meaningless
# for the enlarged agent set.

"""
    run_method_tracked_nls(kwargs, Ξ_train, Ξ_test=nothing; verbosity=1)
        -> (fa, γ_ref, history)

Fit a fixed-agent CES surrogate by NLS for each `t` in `kwargs[:t_list]`,
recording train / test error (mean ℓ₁, the same metric as the CG master's
`primal_obj`, so the two are directly comparable) plus the raw NLS ℓ₂²
objective. Returns the surrogate fitted at the LAST `t` in the list.

kwargs:
- `:t_list`      :: Vector{Int}  — agent counts to sweep (required)
- `:seed`        :: Int|Nothing  — RNG seed forwarded to solve_nls
- `:warmstart`   :: Bool         — warm-start each t from the previous fit (default true)
- `:budget`      :: Symbol       — `:fisher` (scalar w ∈ Δᵗ, default) or `:ad`
                                   (endowments b_i, Σ_i b_i = δ·1; surrogate is an
                                   ArrowDebreuMarket)
- `:nls_max_iter`:: Int          — MadNLP iteration cap per fit (default 100)
- `:ad_delta`    :: Real|Symbol  — AD-only supply scale δ: a fixed number
                                   (default 1.0) or `:free` (fit δ ≥ 0 jointly)
"""
function run_method_tracked_nls(kwargs::Dict, Ξ_train, Ξ_test=nothing; verbosity::Int=1)
    t_list = kwargs[:t_list]
    @assert !isempty(t_list) "run_method_tracked_nls: :t_list must be non-empty"
    seed = get(kwargs, :seed, nothing)
    warmstart = get(kwargs, :warmstart, true)
    budget = get(kwargs, :budget, :fisher)
    budget in (:fisher, :ad) || error("run_method_tracked_nls: :budget must be :fisher or :ad, got $budget")
    max_iter = get(kwargs, :nls_max_iter, 100)
    # Supply scale δ for the AD surrogate (sec.wealth.ad.scale): a fixed number
    # or :free (δ ≥ 0 fitted jointly). Ignored for the Fisher surrogate.
    ad_delta = get(kwargs, :ad_delta, 1.0)
    has_test = !isnothing(Ξ_test)
    is_ad = budget === :ad
    verbose = verbosity >= 1
    K_train = length(Ξ_train)   # normalize the reported NLS objective by K

    history = Dict(
        :primal_obj => Float64[],   # train, mean ℓ₁ (≡ CG primal_obj)
        :test_err => Float64[],     # test, mean ℓ₁
        :excess => Float64[],       # filled by the driver's validation pass
        :num_agents => Int[],       # effective androids after prune (= fa.m)
        :t_requested => Int[],      # requested agent count for this fit
        :nls_obj => Float64[],      # mean NLS objective (Σ_k ‖·‖₂²)/K — the
                                    # solver's ℓ₂² metric, normalized by K so it
                                    # is comparable across sample sizes (distinct
                                    # from the mean-ℓ₁ train/test columns).
        :delta => Float64[],        # AD supply scale used (NaN for Fisher)
        :time => Float64[],
    )

    fa_last = nothing
    γ_ref_last = nothing
    Y_prev = nothing
    σ_prev = nothing

    if verbose
        if is_ad
            @printf("%5s | %10s | %10s | %12s | %5s | %8s | %9s\n",
                "t", "train", "test", "nls(ℓ₂²/K)", "T_eff", "δ", "time(s)")
            @printf("%5s-+-%10s-+-%10s-+-%12s-+-%5s-+-%8s-+-%9s\n",
                "-----", "----------", "----------", "------------", "-----", "--------", "---------")
        else
            @printf("%5s | %10s | %10s | %12s | %5s | %9s\n",
                "t", "train", "test", "nls(ℓ₂²/K)", "T_eff", "time(s)")
            @printf("%5s-+-%10s-+-%10s-+-%12s-+-%5s-+-%9s\n",
                "-----", "----------", "----------", "------------", "-----", "---------")
        end
    end

    for t in t_list
        Y_init = (warmstart && Y_prev !== nothing) ? Y_prev : nothing
        σ_init = (warmstart && σ_prev !== nothing) ? σ_prev : nothing

        local fa, γ_ref, Y, σ_vec, nls_obj, tr, te
        δ_used = NaN
        if budget === :fisher
            t_fit = @elapsed begin
                Y, σ_vec, w, _γ_fitted, _res = solve_nls(
                    Ξ_train, t;
                    max_iter=max_iter,
                    Y_init=Y_init, σ_init=σ_init,
                    verbose=verbosity >= 2, seed=seed,
                )
            end
            nls_obj_sum, _, _ = evaluate_nls(Ξ_train, Y, σ_vec, w)
            nls_obj = nls_obj_sum / K_train
            fa, γ_ref = build_fa_from_nls(Ξ_train, Y, σ_vec, w)
            tr = evaluate_test_error(fa, Ξ_train)
            te = has_test ? evaluate_test_error(fa, Ξ_test) : NaN
        else   # :ad
            t_fit = @elapsed begin
                Y, σ_vec, B, δ_used, _res = solve_nls_ad(
                    Ξ_train, t;
                    delta=ad_delta, max_iter=max_iter,
                    Y_init=Y_init, σ_init=σ_init,
                    verbose=verbosity >= 2, seed=seed,
                )
            end
            nls_obj_sum, _ = evaluate_nls_ad(Ξ_train, Y, σ_vec, B; delta=δ_used)
            nls_obj = nls_obj_sum / K_train
            # Atom list in the shape ad_market_from_atoms / evaluate_test_error_ad
            # expect (one CES atom per agent); B is t×n (agent-by-good).
            cands = [(class=:ces, params=(y=Y[i, :], σ=σ_vec[i])) for i in 1:t]
            fa = ad_market_from_atoms(cands, B)
            γ_ref = nothing
            tr = evaluate_test_error_ad(cands, B, Ξ_train; delta=δ_used)
            te = has_test ? evaluate_test_error_ad(cands, B, Ξ_test; delta=δ_used) : NaN
        end

        push!(history[:primal_obj], tr)
        push!(history[:test_err], te)
        push!(history[:excess], NaN)
        push!(history[:num_agents], fa.m)
        push!(history[:t_requested], t)
        push!(history[:nls_obj], nls_obj)
        push!(history[:delta], δ_used)
        push!(history[:time], t_fit)

        fa_last, γ_ref_last = fa, γ_ref
        Y_prev, σ_prev = Y, σ_vec

        if verbose
            if is_ad
                @printf("%5d | %10.3e | %10.3e | %12.3e | %5d | %8.4f | %9.3f\n",
                    t, tr, te, nls_obj, fa.m, δ_used, t_fit)
            else
                @printf("%5d | %10.3e | %10.3e | %12.3e | %5d | %9.3f\n",
                    t, tr, te, nls_obj, fa.m, t_fit)
            end
        end
    end

    return fa_last, γ_ref_last, history
end
