# FrankWolfe.jl-package wrapper for the Arrow–Debreu surrogate-fitting iteration.
#
# Arrow–Debreu sibling of frankwolfe/wrapper_frankwolfe.jl
# (`run_method_tracked_fwjl`, the Fisher FW.jl runner). It drives the AD
# master's problem (cpm_ad.jl / redistribute_ad.jl) with FrankWolfe.jl's
# `away_frank_wolfe` instead of the Gurobi master LP — the AD analogue of what
# `fwjl` is to `cg`.
#
# Loaded from setup.jl after wrapper_frankwolfe.jl (for `FWJL_TABLE`) and after
# redistribute_ad.jl (for `ad_market_from_atoms`, `evaluate_test_error_ad`) and
# separation.jl (for `find_cut_single`, `_gamma_over_full_from_cand`).
#
# ---- Why FW applies, and the δ = 1 restriction --------------------------------
# Decision variable lives in predictor space `M ∈ ℝ^{K×n}` (the AD analogue of
# the Fisher wrapper's `H`):
#
#     M[k,j] = Σ_t ⟨p_k, b_t⟩ · γ_t(p_k)_j .
#
# With the supply scale δ FIXED at 1 (Σ_t b_t = 1), the feasible set of M is a
# COMPACT convex set — a product of n simplices over conv(ℋ) — so FW applies
# directly. Its vertices are "one preference type per good" bundles: good l is
# owned outright (b = e_l) by a single type γ^{(l)}, giving
#
#     M_vertex[k,j] = Σ_l p_k[l] · γ^{(l)}(p_k)_j .
#
# Free δ would make the endowment set an unbounded cone (no compact FW domain),
# so this runner is δ = 1 only; `:ad_delta` is intentionally ignored. The
# (δ-1)p_k target shift vanishes, so the objective is just
#
#     f(M) = (1/K) Σ_k ‖P_k g_k − M_k‖_1     (entrywise ℓ1, = the master's Σ|s|).
#
# ---- LMO ---------------------------------------------------------------------
# min ⟨G, M⟩ over the bundle hull DECOUPLES per good: for each good l,
#     min_{γ∈ℋ} Σ_k ⟨ p_k[l]·G[k,:], γ(p_k) ⟩ ,
# which is `find_cut_single` on the reweighted dual u_k = −p_k[l]·G[k,:] (same
# −direction sign flip the Fisher CESSeparationLMO uses; offset μ = 0 since the
# LMO needs the argmin, not a reduced cost). `find_cut_single_ad`
# (redistribute_ad.jl) does this reweighting too but COLLAPSES to the single
# best good (it is a pricing oracle); the LMO must keep all n per-good
# minimizers to assemble M_vertex, so we run the per-good loop here.

using FrankWolfe
using Random
using LinearAlgebra

"""
    ADBundleLMO(Ξ, classes)

Linear minimization oracle for the Arrow–Debreu bundle hull. Each
`compute_extreme_point` call runs the homothetic separation oracle once per
good (good l on the price-reweighted direction p_k[l]·direction[k,:]) and
returns the assembled bundle predictor `M_vertex ∈ ℝ^{K×n}`. The n per-good
winning candidates (each `(class, params, good)`) are cached by
`objectid(M_vertex)` so the post-run active set can be turned back into an
`ArrowDebreuMarket` (FrankWolfe.jl stores LMO outputs by identity).
"""
mutable struct ADBundleLMO{T} <: FrankWolfe.LinearMinimizationOracle
    Ξ::Vector{Tuple{Vector{T},Vector{T}}}
    classes::Vector{Symbol}
    oracle_kw::Dict{Symbol,Any}
    cache::Dict{UInt64,Vector{NamedTuple}}
end

ADBundleLMO(Ξ::Vector{Tuple{Vector{T},Vector{T}}}, classes::Vector{Symbol};
    oracle_kw::Dict{Symbol,Any}=Dict{Symbol,Any}()) where {T} =
    ADBundleLMO{T}(Ξ, classes, oracle_kw, Dict{UInt64,Vector{NamedTuple}}())

function FrankWolfe.compute_extreme_point(lmo::ADBundleLMO{T},
    direction::AbstractMatrix; kwargs...) where {T}
    Ξ = lmo.Ξ
    K = length(Ξ)
    n = length(Ξ[1][1])
    cands = Vector{NamedTuple}(undef, n)
    M = zeros(T, K, n)
    u = Matrix{T}(undef, K, n)
    for l in 1:n
        # Reweighted dual for good l: u_k = −p_k[l]·direction[k,:]. The −sign
        # turns FrankWolfe's min⟨direction,·⟩ into the oracle's max⟨u,·⟩.
        @inbounds for k in 1:K
            pkl = Ξ[k][1][l]
            for j in 1:n
                u[k, j] = -pkl * direction[k, j]
            end
        end
        c = find_cut_single(Ξ, u, zero(T), lmo.classes; lmo.oracle_kw...)
        cands[l] = merge(c, (good=l,))
        # Good l is owned outright (b = e_l) by this type, budget ⟨p_k,e_l⟩ =
        # p_k[l], so it contributes p_k[l]·γ^{(l)}(p_k) to the bundle predictor.
        γl = c.γ_new                                   # K×n
        @inbounds for k in 1:K
            pkl = Ξ[k][1][l]
            for j in 1:n
                M[k, j] += pkl * γl[k, j]
            end
        end
    end
    lmo.cache[objectid(M)] = cands
    return M
end

"""
    run_ad_tracked_fwjl(name, kwargs, Ξ_train, Ξ_test=nothing; verbosity=1)
        -> (fa::ArrowDebreuMarket, γ_ref, history)

Arrow–Debreu Frank–Wolfe runner (δ = 1). Mirrors `run_method_tracked_fwjl`
but optimizes the AD predictor over the bundle hull and returns a fitted
`ArrowDebreuMarket`, matching the `(fa, γ_ref, history)` triple the shared
drivers expect from `run_ad_tracked` (`:cg_ad`).

`kwargs` entries:
- `:max_iters`  (200)  forwarded as `max_iteration`
- `:tol_obj`    (1e-3) forwarded as `epsilon`
- `:timelimit`  (Inf)  stops via callback when wall-clock exceeds
- `:seed`       (0)    seed for the initial random separation solve
- `:classes`    ([:ces]) homothetic classes for the oracle (no :leontief)
- `:line_search` (FrankWolfe.Agnostic())   2/(t+2) classical rule
Any other key is forwarded to `find_cut_single` (e.g. `ces_sigma_lower`).
`:ad_delta` is ignored (δ ≡ 1). Ownership is single-good by construction, so
`:ad_endow_mode` has no multi-good realization: `:full` (the CLI default) maps
silently to `:single`, and `:random` (mask) warns that masking is ignored.
"""
function run_ad_tracked_fwjl(name::Symbol, kwargs::Dict,
    Ξ_train, Ξ_test=nothing; verbosity::Int=1)

    verbose = verbosity >= 1

    max_iters   = get(kwargs, :max_iters, 200)
    tol_obj     = get(kwargs, :tol_obj, 1e-3)
    timelimit   = get(kwargs, :timelimit, Inf)
    rng_seed    = get(kwargs, :seed, 0)
    line_search = get(kwargs, :line_search, FrankWolfe.Agnostic())
    classes     = Vector{Symbol}(get(kwargs, :classes, Symbol[:ces]))
    :leontief in classes &&
        error("run_ad_tracked_fwjl: :leontief atoms cannot be stored in an " *
              "ArrowDebreuMarket (CES (c, ρ) form diverges at σ = -1). " *
              "Use --classes from {ces,linear}.")
    # Endowment mode (--ad-endow-mode): like adfw, this δ=1 FW path owns each good
    # by its own best-fit type (= :single), so multi-good ownership has no
    # realization. `:full` (the CLI default) maps silently to `:single`; only
    # `:random` (mask), which is always explicit, warns that masking is ignored.
    ad_endow_mode = Symbol(get(kwargs, :ad_endow_mode, :single))
    if ad_endow_mode === :random
        @warn "run_ad_tracked_fwjl: --ad-endow-mode=mask is not supported by adfwjl; the δ=1 " *
              "bundle-hull master owns each good by its own best-fit type (single-good " *
              "ownership), so masking has no effect. Using single-good ownership." maxlog = 1
    end

    # Oracle-only kwargs: everything not consumed by the FW loop itself is
    # forwarded to find_cut_single (mirrors run_ad_tracked's oracle_kw split).
    _control = (:max_iters, :tol_obj, :timelimit, :seed, :line_search, :classes,
        :interval_eval_test, :test_err_stride, :interval_eval_excess, :f_real,
        :ad_delta, :ad_endow_mode, :ad_mask_size, :drop, :interval_dropping,
        :tol_rc, :tol_delta)
    oracle_kw = Dict{Symbol,Any}(k => v for (k, v) in kwargs if !(k in _control))

    # The per-good separation LMO is LBFGS-local (non-global), and --sample-size
    # (which leaks into oracle_kw) further subsamples find_cut_single on every
    # call, making the oracle stochastic. Either way FrankWolfe's gap-based stop
    # can terminate early; flag it so a short run isn't mistaken for a bug.
    warn_fwjl_inexact_lmo(String(name); subsampled=Int(get(oracle_kw, :sample_size, 0)) > 0)

    K = length(Ξ_train)
    n = length(Ξ_train[1][1])

    # Targets P_k g_k (δ = 1 ⇒ no (δ-1)p_k shift).
    targets = [Ξ_train[k][1] .* Ξ_train[k][2] for k in 1:K]

    f = function (M::AbstractMatrix)
        s = 0.0
        @inbounds for k in 1:K, j in 1:n
            s += abs(targets[k][j] - M[k, j])
        end
        return s / K
    end
    grad! = function (G::AbstractMatrix, M::AbstractMatrix)
        @inbounds for k in 1:K, j in 1:n
            # ∂/∂M of ‖target − M‖_1 is −sign(target − M).
            G[k, j] = -sign(targets[k][j] - M[k, j]) / K
        end
        return G
    end

    Random.seed!(rng_seed)
    lmo = ADBundleLMO(Ξ_train, classes; oracle_kw=oracle_kw)

    # Seed vertex: the LMO's best response to the all-zero predictor. This is a
    # valid bundle, populates the objectid cache, and needs no FisherMarket
    # bootstrap.
    M0 = zeros(Float64, K, n)
    G0 = zeros(Float64, K, n)
    grad!(G0, M0)
    H0 = FrankWolfe.compute_extreme_point(lmo, G0)

    history = Dict(
        :primal_obj => Float64[],
        :test_err   => Float64[],
        :excess     => Float64[],          # AD surrogate-equilibrium excess not wired
        :num_agents => Int[],              # active-set size (number of bundles)
    )
    _t0 = time()
    interval_eval_test = get(kwargs, :interval_eval_test, get(kwargs, :test_err_stride, 25))
    iter_counter = Ref(0)
    last_test_err = Ref(NaN)
    function cb(state, args...)
        iter_counter[] += 1
        as = isempty(args) ? nothing : args[1]
        T_cur = isnothing(as) ? 1 : length(as.atoms)
        push!(history[:primal_obj], state.primal)
        if !isnothing(as) && !isnothing(Ξ_test) &&
           interval_eval_test > 0 && (iter_counter[] % interval_eval_test == 0)
            cands, B = _ad_atoms_from_active_set(as, lmo, n)
            last_test_err[] = evaluate_test_error_ad(cands, B, Ξ_test; delta=1.0)
        end
        push!(history[:test_err], last_test_err[])
        push!(history[:excess], NaN)
        push!(history[:num_agents], T_cur)
        return (time() - _t0) ≤ timelimit
    end

    # `tol_obj === nothing` (--tol-obj 0.0) ⇒ disable FW.jl's dual-gap stop
    # (epsilon = 0.0); the run goes to max_iteration / the time limit.
    epsilon_eff = isnothing(tol_obj) ? 0.0 : tol_obj
    if verbose
        print_banner(FWJL_TABLE, BANNER_TITLE)
        print_config("method",        String(name))
        print_config("alias",         "FrankWolfe.jl away_frank_wolfe (Arrow–Debreu, δ=1)")
        print_config("classes",       join(String.(classes), ", "))
        print_config("endow mode",     ad_endow_mode === :random ?
            "single (mask requested → ignored; δ=1 bundle hull)" : "single (δ=1 bundle hull)")
        print_config("K (training samples)", K)
        print_config("n (goods)",     n)
        print_config("max_iters",     max_iters)
        print_config("timelimit (s)", @sprintf("%g", Float64(timelimit)))
        print_config("epsilon",       @sprintf("%g", epsilon_eff))
        print_config("line_search",   string(line_search))
        println("-"^table_width(FWJL_TABLE))
    end

    common_kwargs = (
        max_iteration = max_iters,
        epsilon       = epsilon_eff,
        line_search   = line_search,
        callback      = cb,
        verbose       = verbose,
        print_iter    = max(1, interval_eval_test),
        trajectory    = false,
    )

    _, _, primal, dual_gap, _, _, active_set =
        FrankWolfe.away_frank_wolfe(f, grad!, lmo, H0; common_kwargs...)

    cands, B = _ad_atoms_from_active_set(active_set, lmo, n)
    fa = ad_market_from_atoms(cands, B)
    # γ_ref as the (m, K, n) bidding tensor, one K×n slice per atom — same shape
    # run_ad_tracked returns.
    m = length(cands)
    γ_tensor = Array{Float64,3}(undef, m, K, n)
    for t in 1:m
        γ_tensor[t, :, :] .= _gamma_over_full_from_cand(Ξ_train, cands[t])
    end
    γ_ref = Ref(γ_tensor)

    if !isnothing(Ξ_test) && !isempty(history[:test_err])
        history[:test_err][end] = evaluate_test_error_ad(cands, B, Ξ_test; delta=1.0)
    end
    # Stash atoms so phased drivers can warm-restart, mirroring run_ad_tracked.
    history[:cands] = cands

    _elapsed = time() - _t0
    if verbose
        println("=== $(name) (FrankWolfe.jl away_frank_wolfe, Arrow–Debreu) ===")
        @printf("--- done: %d atoms (%d bundles), primal=%.3e, dual_gap=%.3e, t=%.4fs ---\n",
            m, length(active_set.atoms), primal, dual_gap, _elapsed)
    end

    return fa, γ_ref, history
end

"""
    _ad_atoms_from_active_set(as, lmo, n) -> (cands, B)

Turn a FrankWolfe.jl `ActiveSet` of bundle vertices into the AD atom list +
endowment matrix. Each active bundle `i` (weight α_i > 0) owns good `l` via the
cached type `cand_{i,l}`, contributing an atom with endowment α_i on good `l`.
Stacking over (bundle, good) gives `Σ_t B[t,l] = Σ_i α_i = 1` for every good
(δ = 1). Bundles whose cands aren't cached are skipped.
"""
function _ad_atoms_from_active_set(as, lmo::ADBundleLMO{T}, n::Int) where {T}
    cands = NamedTuple[]
    rows = Vector{Float64}[]
    for (M_vertex, α) in zip(as.atoms, as.weights)
        α <= 0 && continue
        bundle = get(lmo.cache, objectid(M_vertex), nothing)
        isnothing(bundle) && continue
        for l in 1:n
            c = bundle[l]
            push!(cands, (class=c.class, params=c.params))
            row = zeros(Float64, n)
            row[l] = Float64(α)
            push!(rows, row)
        end
    end
    # Degenerate fallback (everything dropped): a single full-endowment CES atom
    # keeps the returned market non-empty and every good owned.
    if isempty(cands)
        push!(cands, (class=:ces, params=(y=zeros(n), σ=0.5)))
        push!(rows, ones(Float64, n))
    end
    B = Matrix{Float64}(undef, length(cands), n)
    for (t, r) in enumerate(rows)
        B[t, :] .= r
    end
    return cands, B
end
