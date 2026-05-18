# FrankWolfe.jl-package wrapper for the surrogate-fitting FW iteration.
#
# Mirrors the manual Frank-Wolfe runner in ../frankwolfe.jl
# (`run_method_tracked_fw`) but delegates the active-set bookkeeping and
# step-size schedule to the third-party FrankWolfe.jl package.
#
# Loaded last from setup.jl (after validate.jl) so the helpers it needs
# (`compute_gamma_from_market`, `evaluate_test_error`, `validate_surrogate`,
# `add_column_to_market!`, `solve_separation_lbfgs_ces*`, `recover_ces_params`) are all
# already defined.
#
# Decision variable lives in ℝ^{K×n}: the matrix `H` of values h(p_k) =
# Σ_t w_t γ_t(p_k). Feasible set is the convex hull of CES androids, with
# the LMO returning a one-android evaluation matrix γ ∈ ℝ^{K×n} (rows are
# γ(p_k) for k ∈ [K]).
#
# Objective f(H) = (1/K) Σ_k ‖P_k g_k − H[k,:]‖_∞.
# Gradient: row-k subgradient is −sign(target_k − H[k,:])[j*]·e_{j*}/K.
#
# Android-to-params lookup uses a Dict keyed by objectid of the LMO-returned
# matrix; FrankWolfe.jl stores LMO outputs by identity in ActiveSet, so
# the cache survives the AFW iteration. Initial android params are computed
# from the random FisherMarket used to seed the iterate.

using FrankWolfe
using Random
using LinearAlgebra

# Iteration-table for the FrankWolfe.jl wrapper's banner. The package
# prints its own per-iteration log, so this table is only used to size
# the banner width consistently with cpm.jl / frankwolfe.jl.
const FWJL_TABLE = IterTable(
    ["iter", "primal", "dual_gap", "T", "t(s)"],
    ["%6d", "%10.3e", "%10.3e", "%5d", "%10.4f"],
    Any[1, 1.0e-3, 1.0e-3, 1, 1.234],
)

"""
    CESSeparationLMO(Ξ)

Linear minimization oracle for the CES separation subproblem. Calls
`solve_separation_dual_lp_ces` + `solve_separation_lbfgs_ces` (from ces.jl) to find the CES
android whose evaluation matrix γ ∈ ℝ^{K×n} minimizes ⟨direction, γ⟩, and
caches the recovered `(y, σ)` for downstream `add_to_market!`.
"""
mutable struct CESSeparationLMO{T} <: FrankWolfe.LinearMinimizationOracle
    Ξ::Vector{Tuple{Vector{T},Vector{T}}}
    cache::Dict{UInt64,NamedTuple}
end

CESSeparationLMO(Ξ::Vector{Tuple{Vector{T},Vector{T}}}) where {T} =
    CESSeparationLMO{T}(Ξ, Dict{UInt64,NamedTuple}())

function FrankWolfe.compute_extreme_point(lmo::CESSeparationLMO{T},
    direction::AbstractMatrix; kwargs...) where {T}
    # min ⟨direction, γ⟩  ⇔  max ⟨−direction, γ⟩, which is the dual-separation
    # problem with cost rows u_k := −direction[k,:].
    u = Matrix{T}(-direction)
    y_lp, σ_lp, _, _ = solve_separation_dual_lp_ces(lmo.Ξ, u)
    y_opt, σ_opt, γ_new, _ = solve_separation_lbfgs_ces(lmo.Ξ, u; y_init=y_lp, σ_init=σ_lp)
    γ_dense = Matrix{T}(γ_new)
    lmo.cache[objectid(γ_dense)] = (y=y_opt, σ=σ_opt)
    return γ_dense
end

"""
    run_method_tracked_fwjl(name, kwargs, Ξ_train, Ξ_test=nothing; verbosity=1)

Drive the FW iteration with FrankWolfe.jl's `away_frank_wolfe`, which
maintains an active set so per-android weights can be recovered post-run.
Returns `(fa, γ_ref, history)` matching the runners in cpm.jl /
`run_method_tracked_fw`.

`kwargs` entries:
- `:max_iters`  (200)  forwarded as `max_iteration` to FrankWolfe.jl
- `:tol_obj`    (1e-3) forwarded as `epsilon`
- `:timelimit`  (Inf)  stops via callback when wall-clock exceeds
- `:seed`       (0)    seed for the initial random CES android
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

    # Initial CES android — seeded so runs are reproducible.
    Random.seed!(rng_seed)
    ws0 = cpu_workspace(n)
    add_ces!(ws0, 1; ρ=rand(1), scale=30.0, sparsity=0.99)
    fa0 = FisherMarket(ws0)
    fa0.w[1] = 1.0
    γ_init = compute_gamma_from_market(fa0, Ξ_train)        # (1, K, n)
    H0 = Matrix{Float64}(reshape(γ_init[1, :, :], K, n))

    # Cache the initial android's (y, σ) so the active-set lookup post-FW
    # can recover its params just like an LMO-produced android.
    σ0 = fa0.σ[1]
    c0 = Vector(fa0.c[:, 1])
    y0 = (1 + σ0) .* log.(c0)

    lmo = CESSeparationLMO(Ξ_train)
    lmo.cache[objectid(H0)] = (y=y0, σ=σ0)

    # History recorded per FW iteration via callback.
    history = Dict(
        :primal_obj => Float64[],
        :test_err   => Float64[],
        :excess     => Float64[],
        :num_agents => Int[],
    )
    _t0 = time()
    f_real = get(kwargs, :f_real, nothing)
    interval_eval_excess = get(kwargs, :interval_eval_excess, 0)
    last_excess = Ref(NaN)
    has_excess = !isnothing(f_real) && interval_eval_excess > 0

    # Light-weight callback: record per-iteration primal & active-set
    # size. test_err is recomputed every `test_err_stride` iterations
    # (rebuilding a FisherMarket from the active set is O(T) and would
    # dominate the time budget if done each iter); intervening iters
    # carry forward the most recent evaluation so the plotted curve is
    # continuous. In FrankWolfe.jl v0.6, `away_frank_wolfe` calls the
    # callback as `cb(state, active_set)`, so the live active set comes
    # in via `args[1]`.
    interval_eval_test = get(kwargs, :interval_eval_test, get(kwargs, :test_err_stride, 25))
    iter_counter = Ref(0)
    last_test_err = Ref(NaN)
    function cb(state, args...)
        iter_counter[] += 1
        as = isempty(args) ? nothing : args[1]
        T_cur = isnothing(as) ? 1 : length(as.androids)
        push!(history[:primal_obj], state.primal)
        if !isnothing(as) && !isnothing(Ξ_test) &&
           interval_eval_test > 0 && (iter_counter[] % interval_eval_test == 0)
            fa_tmp = _build_fa_from_active_set(as, lmo, n)
            last_test_err[] = evaluate_test_error(fa_tmp, Ξ_test)
        end
        if has_excess && !isnothing(as) &&
           (iter_counter[] % interval_eval_excess == 0)
            try
                fa_tmp2 = _build_fa_from_active_set(as, lmo, n)
                v = validate_surrogate(fa_tmp2, f_real; verbose=false)
                last_excess[] = v.excess_surrogate_linf
            catch err
                @warn "[$name iter $(iter_counter[])] validate_surrogate failed" err
            end
        end
        push!(history[:test_err], last_test_err[])
        push!(history[:excess],   last_excess[])
        push!(history[:num_agents], T_cur)
        # Wall-clock cap: returning `false` halts FrankWolfe.jl's loop.
        return (time() - _t0) ≤ timelimit
    end

    # `tol_obj === nothing` ⇒ disable FW.jl's epsilon stop by passing a
    # vanishingly small value (the package expects a real number).
    epsilon_eff = isnothing(tol_obj) ? eps(Float64) : tol_obj
    if verbose
        print_banner(FWJL_TABLE, BANNER_TITLE)
        print_config("method",        String(name))
        print_config("alias",         "FrankWolfe.jl away_frank_wolfe")
        print_config("K (training samples)", K)
        print_config("n (goods)",     n)
        print_config("max_iters",     max_iters)
        print_config("timelimit (s)", @sprintf("%g", Float64(timelimit)))
        print_config("epsilon",       @sprintf("%g", epsilon_eff))
        print_config("line_search",   string(line_search))
        println("-"^table_width(FWJL_TABLE))
    end
    # FW.jl's own per-iteration log is enabled when the caller passes
    # verbosity >= 1; the stride matches `interval_eval_test` so the
    # printed iters line up with our test-error rows.
    common_kwargs = (
        max_iteration = max_iters,
        epsilon       = epsilon_eff,
        line_search   = line_search,
        callback      = cb,
        verbose       = verbose,
        print_iter    = max(1, interval_eval_test),
        trajectory    = false,
    )

    # away_frank_wolfe returns a 7-tuple in FrankWolfe.jl v0.6:
    # (x, v, primal, dual_gap, ExecutionStatus, trajectory, active_set).
    _, _, primal, dual_gap, _, _, active_set =
        FrankWolfe.away_frank_wolfe(f, grad!, lmo, H0; common_kwargs...)

    fa = _build_fa_from_active_set(active_set, lmo, n)
    γ_ref = Ref(compute_gamma_from_market(fa, Ξ_train))

    # Fill the final test_err into the trailing history slot — single
    # FisherMarket evaluation against the final active set.
    if !isnothing(Ξ_test) && !isempty(history[:test_err])
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

    _elapsed = time() - _t0
    if verbose
        println("=== $(name) (FrankWolfe.jl away_frank_wolfe) ===")
        @printf("--- done: %d androids, primal=%.3e, dual_gap=%.3e, t=%.4fs ---\n",
            fa.m, primal, dual_gap, _elapsed)
    end

    return fa, γ_ref, history
end

"""
    _build_fa_from_active_set(as, lmo, n)

Reconstruct a `FisherMarket` from a FrankWolfe.jl `ActiveSet`. Each android
in the active set is a γ matrix produced by `CESSeparationLMO`; its (y, σ)
were stashed in `lmo.cache` keyed by `objectid`. Androids whose params
aren't cached are skipped (their weight is folded into a uniform CES
fallback so the FisherMarket remains non-empty).
"""
function _build_fa_from_active_set(as, lmo::CESSeparationLMO{T}, n::Int) where {T}
    androids = as.androids
    weights = as.weights
    fa = FisherMarket(cpu_workspace(n))
    for (γ_matrix, w_t) in zip(androids, weights)
        w_t <= 0 && continue
        params = get(lmo.cache, objectid(γ_matrix), nothing)
        isnothing(params) && continue
        add_column_to_market!(fa, params, :ces, T(w_t))
    end
    # If everything dropped (unlikely), seed with a single uniform android.
    if fa.m == 0
        ws = cpu_workspace(n)
        add_ces!(ws, 1; ρ=[0.0], scale=30.0, sparsity=0.99)
        fa = FisherMarket(ws)
        fa.w[1] = 1.0
    end
    return fa
end
