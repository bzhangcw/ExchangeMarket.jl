# -----------------------------------------------------------------------
# Per-class separation oracle for the column-generation framework.
#
# The separation subproblem in CG is:
#   given dual variables (u, μ) from the master, find a column γ_new with
#   reduced cost rc = Σ_k <u_k, γ_{new,k}> - μ > 0.
#
# Different function classes (CES, linear, Leontief, ...) parameterize γ
# differently. This file:
#   - keeps the class-agnostic utilities (reduced_cost, add_to_gamma!,
#     drop_zero_columns!),
#   - includes per-class separation files (androids/ces.jl, androids/linear.jl, androids/leontief.jl),
#   - provides solve_separation_class / solve_separation, which run each class's
#     separation routine and pick the column with the largest rc,
#   - provides solve_separation_multicut for per-sample inversion (one cut
#     per sample, K cuts per round),
#   - provides runner-facing wrappers find_cut_single / find_cuts_multi
#     that handle sample-size subsampling re-expansion and (for the
#     single-cut path) optional fixed-ρ CES override. Both runners
#     (cpm.jl, accpm.jl) call only these wrappers.
#
# The class-specific solvers all return a NamedTuple
#   (γ_new::Matrix{T}, params::NamedTuple, obj::T, class::Symbol)
# so the separation oracle can pick the best one without knowing the class.
# -----------------------------------------------------------------------

using LinearAlgebra
using ArgParse
using ExchangeMarket

# ---- Android homotheticity declarations ------------------------------------
# Each android file declares its own `is_homothetic(::Val{:foo})` method;
# the Val-based dispatch lets us pass the class symbol around without
# coupling separation.jl to any particular android module. The generic
# entry point is `is_homothetic(cls::Symbol)`; callers use that plus the
# `homothetic_classes` / `nonhomothetic_classes` helpers when partitioning
# a user-supplied class list (e.g., for the multicut/non-homothetic guard
# in cpm.jl and the pinning-index construction in solve_wealth_redist_primal).
is_homothetic(cls::Symbol) = is_homothetic(Val(cls))
is_homothetic(::Val{C}) where {C} =
    error("is_homothetic: class :$C has not declared homotheticity. " *
          "Add `is_homothetic(::Val{:$C}) = true|false` in androids/$(C).jl.")

homothetic_classes(classes::AbstractVector{Symbol}) = filter(is_homothetic, classes)
nonhomothetic_classes(classes::AbstractVector{Symbol}) = filter(c -> !is_homothetic(c), classes)

include("./androids/ces.jl")
include("./androids/linear.jl")
include("./androids/leontief.jl")
include("./androids/ql.jl")
include("./androids/ges.jl")
include("./androids/nn.jl")

# ---- CLI surface (shared across separation classes) ------------------------
"""
    register_cli_separation!(s::ArgParseSettings)

Add the "Separation (shared)" arg group — knobs that apply across all
separation classes (currently just `--sample-size`).
"""
function register_cli_separation!(s::ArgParseSettings)
    add_arg_group!(s, "Separation (shared)")
    @add_arg_table! s begin
        "--sample-size"
        help = "Mini-batch size for the separation oracle (Higle-Sen / Joachims style). If 0 (default) or >= K, uses the full training set; otherwise each separation call sees a random subset of this size, master uses full K."
        arg_type = Int
        default = 0
        "--sample-hard"
        help = "Boosting-style mini-batch: when --sample-size subsamples, draw the batch with probability proportional to each sample's current master residual ‖s_k‖₁ (column-generation = LPBoost; the dual already weights the oracle, this concentrates the *batch* on hard examples). No effect without --sample-size. Default: uniform."
        action = :store_true
    end
    return s
end

"""
    apply_cli_separation!(local_extra::Dict, cli)

Forward shared-separation CLI values into the runner kwargs.
"""
function apply_cli_separation!(local_extra::Dict, cli)
    if cli["sample_size"] > 0
        local_extra[:sample_size] = cli["sample_size"]
    end
    if get(cli, "sample_hard", false)
        local_extra[:sample_hard] = true
    end
    return local_extra
end

"""
    print_class_configs(classes, kwargs; is_multicut, nonh_w)

Emit the per-class banner rows (one `print_config` per class actually in
`classes`) by dispatching to the per-class `*_config_summary` defined in
the corresponding `androids/<class>.jl`. Centralizes the printing logic
so cpm.jl, accpm.jl, and any future driver share one source of truth.

`is_multicut` is forwarded to `ces_config_summary` (the only summary
sensitive to the separation regime). `nonh_w` is forwarded to the
non-homothetic summaries (`ql`, `ges`) — drivers without a pinning
weight (e.g., accpm.jl pins at 1.0) pass that value explicitly.
"""
function print_class_configs(classes, kwargs::Dict;
                             is_multicut::Bool, nonh_w::Real)
    :ces      in classes && print_config("ces",
        ces_config_summary(kwargs; is_multicut=is_multicut); indent=true)
    :linear   in classes && print_config("linear",
        linear_config_summary(kwargs); indent=true)
    :leontief in classes && print_config("leontief",
        leontief_config_summary(kwargs); indent=true)
    :ql       in classes && print_config("ql",
        ql_config_summary(kwargs; nonh_w=nonh_w); indent=true)
    :ges      in classes && print_config("ges",
        ges_config_summary(kwargs; nonh_w=nonh_w); indent=true)
    return nothing
end

# -----------------------------------------------------------------------
# Class-agnostic CG utilities
# -----------------------------------------------------------------------
"""
    reduced_cost(γ_new, u, μ)

Compute the reduced cost for a new agent with bidding vectors γ_new:
    reduced_cost = Σ_k <u_k, γ_new_k> - μ.

If reduced_cost > 0, adding this agent can improve the master problem.
"""
function reduced_cost(γ_new::Matrix{T}, u::Matrix{T}, μ::T) where T
    K = size(γ_new, 1)
    rc = sum(dot(u[k, :], γ_new[k, :]) for k in 1:K) - μ
    return rc
end

"""
    add_to_gamma!(γ_ref::Ref{Array{T,3}}, γ_new)

Append a new agent's bidding vectors to γ in-place.
γ_ref holds γ with shape (m, K, n); γ_new has shape (K, n).
After return, γ_ref[] has shape (m+1, K, n).
"""
function add_to_gamma!(γ_ref::Ref{Array{T,3}}, γ_new::Matrix{T}) where T
    γ = γ_ref[]
    m, K, n = size(γ)
    @assert size(γ_new) == (K, n) "γ_new must be (K, n) = ($K, $n)"
    γ_expanded = zeros(T, m + 1, K, n)
    γ_expanded[1:m, :, :] .= γ
    γ_expanded[m+1, :, :] .= γ_new
    γ_ref[] = γ_expanded
    return nothing
end

"""
    drop_zero_columns!(fa::FisherMarket, γ_ref, w; tol=1e-8)

Remove agents with weight ≤ tol across both substores (`ces` and `gen`)
and prune `γ_ref` to match. Partitions `keep = findall(w .> tol)` by the
workspace routing tag, calls `_prune_ces!` / `_prune_gen!` with the
substore-local indices to retain, then rebuilds `routing` so the
remaining rows of γ_ref still index into the right substore slots
(local indices are renumbered).

Returns `(ndrop, keep)` where `keep::Vector{Int}` is the (sorted, into
`1:m_γ`) index list that survived the threshold. Callers that need to
re-scatter the post-drop weights should index `w[keep]` to stay in sync
with the rewritten routing — using a different tolerance afterward
would length-mismatch.
"""
function drop_zero_columns!(fa::FisherMarket{T}, γ_ref::Ref{Array{T,3}}, w::Vector{T}; tol=1e-8) where T
    ws = fa.storage
    m_γ = size(γ_ref[], 1)
    @assert length(w) == m_γ "w length ($(length(w))) must match γ_ref agent dim ($m_γ)"
    @assert length(ws.routing) == m_γ "routing length ($(length(ws.routing))) must match γ_ref agent dim ($m_γ)"

    keep = findall(w .> tol)
    ndrop = m_γ - length(keep)
    ndrop == 0 && return (0, keep)

    # Partition kept indices by substore, preserving CG-insertion order
    # within each substore so the renumbered local indices stay
    # contiguous from 1.
    ces_local_keep = Int[]
    gen_local_keep = Int[]
    for i in keep
        sub, j = ws.routing[i]
        if sub === :ces
            push!(ces_local_keep, j)
        else
            push!(gen_local_keep, j)
        end
    end

    # Single-shot workspace prune: rewrites CES params, gen agents,
    # universal per-agent arrays (ws.w/x/g/val_u/ε_br_play), routing,
    # and re-slices substore views in one pass.
    prune_workspace!(ws; ces_keep=ces_local_keep, gen_keep=gen_local_keep)
    fa.m = ws.m
    # AgentView registry may hold stale views into the pre-prune ws
    # arrays — clear it so init_agents! rebuilds on next use.
    setfield!(fa, :agents, Vector{Any}())

    γ_ref[] = γ_ref[][keep, :, :]
    return (ndrop, keep)
end

# -----------------------------------------------------------------------
# Per-class separation oracle
# -----------------------------------------------------------------------
"""
    solve_separation_class(class::Symbol, Ξ, u; nonh_w, kwargs...)

Run the separation subproblem for a single class. Returns a NamedTuple
    (γ_new::Matrix{T}, params::NamedTuple, obj::T, class::Symbol).

Supported classes:
- `:ces`      — full CES, free σ; LP warm-start + LBFGS refinement. (homothetic)
- `:linear`   — linear utility class H(1); big-M MIP (eq.cg.sep.linear). (homothetic)
- `:leontief` — CES boundary σ → -1⁺; concave fix-σ LBFGS at σ = σ_leontief. (homothetic)
- `:nn`       — softmax-MLP android γ_θ(p); LBFGS over the weights θ. (homothetic)
- `:ql`       — quasi-linear-log android γ(c, w); regime-enumeration at the
                pinned wealth w = nonh_w. (non-homothetic)

`nonh_w` is the shared pinning wealth w_0 for non-homothetic classes
(eq.wealth.hybrid.lp). Passed positionally to each non-homothetic
oracle; homothetic oracles ignore it (their γ has no w-dependence — see
`is_homothetic(::Val{:foo})` declared in each android file).
"""
function solve_separation_class(class::Symbol,
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    nonh_w::Real=one(T),
    kwargs...) where T
    if class === :ces
        return solve_separation_ces(Ξ, u; kwargs...)
    elseif class === :linear
        return solve_separation_linear(Ξ, u; kwargs...)
    elseif class === :leontief
        return solve_separation_leontief(Ξ, u; kwargs...)
    elseif class === :ql
        # Non-homothetic class: γ depends on the pinning wealth, so
        # `nonh_w` is a mandatory positional argument to the oracle.
        # Any future non-homothetic class added here should follow the same
        # `solve_separation_<class>(Ξ, u, nonh_w; kwargs...)` calling pattern.
        return solve_separation_ql(Ξ, u, nonh_w; kwargs...)
    elseif class === :ges
        # Non-homothetic class: γ depends on the pinning wealth through the
        # implicit budget root λ. Pricing is a generic NLP solved by MadNLP
        # in the log-variable parameterization (eq.ges.pricing.log).
        return solve_separation_ges(Ξ, u, nonh_w; kwargs...)
    elseif class === :nn
        # Forward NN-specific knobs from the caller's kwargs (the
        # signatures don't overlap with the others, so other classes
        # ignore them silently).
        hidden = get(kwargs, :nn_hidden, NN_HIDDEN_DEFAULT)
        max_iters = get(kwargs, :nn_iters, NN_ITERS_DEFAULT)
        return solve_separation_nn(Ξ, u; hidden=hidden, max_iters=max_iters,
            kwargs...)
    else
        error("Unknown separation class: $class. Supported: :ces, :linear, :leontief, :ql, :ges, :nn.")
    end
end

"""
    solve_separation(Ξ, u, μ, classes; nonh_w, verbose, kwargs...)

Solve the separation subproblem for each class in `classes`, compute the reduced
cost of each candidate column, and return the candidate with the largest rc.

`nonh_w` is the shared pinning wealth for non-homothetic classes; it is
forwarded explicitly (not via `kwargs...`) so every layer in the stack
documents the parameter rather than passing it through opaquely.
Homothetic classes ignore it.

Returns a NamedTuple
    (γ_new, params, obj, class, rc)
where `params` and `class` are passed through from the winning class's
solver and `rc = Σ_k <u_k, γ_new_k> - μ`.
"""
function solve_separation(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T}, μ::T, classes::Vector{Symbol};
    nonh_w::Real=one(T),
    verbose::Bool=false,
    kwargs...) where T
    @assert !isempty(classes) "classes must be non-empty"
    best = nothing
    best_rc = T(-Inf)
    for class in classes
        cand = @time_sep class solve_separation_class(
            class, Ξ, u;
            nonh_w=nonh_w, verbose=verbose, kwargs...
        )
        # A per-class oracle is allowed to return `nothing` to signal
        # "skip me this iteration" (e.g., GES when its NLP fails or the
        # recovered (c, r) wouldn't reproduce the NLP γ at test prices).
        # Just continue — other classes may still find an improving column.
        if isnothing(cand)
            verbose && println("  class=$class: skipped (oracle returned nothing)")
            continue
        end
        rc = reduced_cost(cand.γ_new, u, μ)
        verbose && println("  class=$class: obj=$(cand.obj), rc=$rc")
        if rc > best_rc
            best_rc = rc
            best = (γ_new=cand.γ_new, params=cand.params, obj=cand.obj, class=cand.class, rc=rc)
        end
    end
    return best
end

"""
    solve_separation_multicut(Ξ, u, classes; σ_grid=range(-1.0, 30.0, length=50))

Per-sample multicut inversion across the inversion-capable classes
(currently `:ces` and `:linear`). For each sample `k`, compute the
inverted candidate in every allowed class and **pick the one with the
largest separation-oracle objective** `Σ_{k'} ⟨u_{k'}, γ(p_{k'}; y, σ)⟩`.
The objective is on the same scale across classes, so the per-`k`
argmax is a valid greedy choice for the K columns added per multicut
pass — at most K (not |classes|·K) androids enter the surrogate.

CES uses `solve_separation_inversion_ces` (σ-grid + Brent refinement);
linear uses `solve_separation_inversion_linear` (no σ — read off the
bang-per-buck winner from `argmax_j u[k, j]`).

Returns a `Vector` of K NamedTuples
    `(class::Symbol, y, σ, γ_new::Matrix, obj)`.
A sample whose value of K is `:none` indicates no allowed class
produced an inverted candidate (e.g., neither `:ces` nor `:linear`
was in `classes`); the caller should skip such entries.
"""
function solve_separation_multicut(
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T},
    classes::Vector{Symbol};
    # Inversion σ-grid is constructed inside solve_separation_inversion_ces
    # from its `ces_sigma_lower` / `ces_sigma_upper` kwargs (defaults match
    # the previous file-level extents). Forwarded via `kwargs...`.
    kwargs...
) where T
    K = length(Ξ)
    n = length(Ξ[1][1])
    ces_results = (:ces in classes) ? (@time_sep :ces solve_separation_inversion_ces(Ξ, u; kwargs...)) : nothing
    lin_results = (:linear in classes) ? (@time_sep :linear solve_separation_inversion_linear(Ξ, u)) : nothing
    leon_results = (:leontief in classes) ? (@time_sep :leontief solve_separation_inversion_leontief(Ξ, u)) : nothing

    # Per-sample winner across all inversion-capable classes. We carry a
    # class-specific `params::NamedTuple` so the runner can pass it
    # straight to `add_column_to_market!` regardless of class — same
    # convention as find_cut_single's return shape.
    chosen = Vector{NamedTuple}(undef, K)
    for k in 1:K
        best = (class=:none, params=NamedTuple(), γ_new=zeros(T, K, n), obj=T(-Inf))
        if !isnothing(ces_results)
            y, σ, γ_new, obj = ces_results[k]
            obj > best.obj && (best = (class=:ces, params=(y=y, σ=σ),
                γ_new=γ_new, obj=obj))
        end
        if !isnothing(lin_results)
            y, γ_new, obj = lin_results[k]
            obj > best.obj && (best = (class=:linear, params=(y=y, σ=T(Inf)),
                γ_new=γ_new, obj=obj))
        end
        if !isnothing(leon_results)
            a, γ_new, obj = leon_results[k]
            obj > best.obj && (best = (class=:leontief, params=(a=a,),
                γ_new=γ_new, obj=obj))
        end
        chosen[k] = best
    end
    return chosen
end

# -----------------------------------------------------------------------
# Class-aware market expansion
# -----------------------------------------------------------------------
"""
    add_column_to_market!(fa::FisherMarket, params::NamedTuple, class::Symbol, w_new=0.0)

Convert a class-specific separation solution `params=(y, σ)` into a CES agent
and append it to `fa`. All classes are stored as CES: linear sits at
σ → ∞ (large σ_linear), Leontief at σ → -1⁺ (σ_leontief ≈ -0.9).

The map `(y, σ) → (c, ρ)` is `c = exp(y / (1+σ))`, `ρ = σ / (1+σ)`. Because
γ is softmax-invariant under `y → y + α·1` (equivalently `c → c · κ`), the
separation solver only pins `y` up to an additive constant. We normalize by
shifting so `max(y) = 0`, which gives the canonical representative `c ∈
(0, 1]^n` (`max(c) = 1`). At extreme σ → -1 the small entries may underflow
to 0; `compute_gamma` handles that correctly via log-space softmax.
"""
function add_column_to_market!(fa::FisherMarket{T}, params::NamedTuple, class::Symbol, w_new::T=zero(T)) where T
    if class === :nn
        # MVP v1: NN androids register a uniform-CES PLACEHOLDER so fa.m
        # stays in sync with γ_ref[]'s first dim. The LP master pulls γ
        # from γ_ref (which holds the actual γ_θ(p_k) row added by
        # add_to_gamma!), so primal_obj is NN-correct. But
        # evaluate_test_error walks fa.c, fa.σ and will evaluate the
        # placeholder (uniform CES, ρ=0) at test prices, NOT the NN.
        # Test error is therefore incorrect for NN atoms — fix in v2 via
        # a side-table keyed by atom index that holds θ.
        n = size(fa.c, 1)
        add_to_market!(fa, ones(T, n), T(0), w_new)
        return fa
    elseif class === :ql
        # QL androids go into the GenStore via add_gen!: a typed
        # QuasiLinearLogAgent holds the params (c, n), and routing tags
        # this row as (:gen, j) so evaluate_test_error dispatches per-class.
        # No CES placeholder needed.
        agent = QuasiLinearLogAgent(fa.storage.n, Vector{Float64}(params.c))
        add_gen!(fa.storage, agent, Float64(w_new))
        fa.m = fa.storage.m
        return fa
    elseif class === :leontief
        # Leontief atoms are stored natively (NOT as a near-Leontief CES
        # approximation): the CES (c, ρ) form has 1/(1+σ) blowing up at
        # σ = -1, and compute_gamma's (1+σ)·log(c) factor zeros out at the
        # boundary — so a true Leontief atom cannot round-trip through CES
        # storage. We route through GenStore as a LeontiefAgent, with
        # share dispatch via `share(::LeontiefAgent, p, w) = leontief_share(a, p)`.
        # Although Leontief is homothetic, it shares the GenStore channel
        # with QL because that's how the per-class share dispatch works.
        agent = LeontiefAgent(Vector{Float64}(params.a))
        add_gen!(fa.storage, agent, Float64(w_new))
        fa.m = fa.storage.m
        return fa
    elseif class === :ges
        # GES atoms are non-homothetic and have no closed-form CES (c, ρ)
        # representation — the per-good elasticities σ_j can't collapse
        # into a single ρ. They go into GenStore as a typed GESAgent,
        # with share dispatch via `share(::GESAgent, p, w) = ges_share(c, r, p, w)`.
        # The LP master pins w_t = nonh_w for these atoms (see _pinned_idx
        # in cpm.jl and the agent_is_homothetic(::GESAgent) declaration).
        agent = GESAgent(fa.storage.n,
            Vector{Float64}(params.c),
            Vector{Float64}(params.r))
        add_gen!(fa.storage, agent, Float64(w_new))
        fa.m = fa.storage.m
        return fa
    end
    y, σ = params.y, params.σ
    y_shifted = y .- maximum(y)                              # max(y) = 0  ⇒  max(c) = 1
    if class === :linear
        # Linear utility: y = log c (raw), stored as a true LinearAgent via ρ = 1.
        # σ becomes Inf inside the FisherMarket via expand_players!; agent_type
        # dispatches on ρ == 1.0 to return LinearAgent(), and compute_gamma
        # special-cases σ = Inf to the bang-per-buck argmax.
        c_new = exp.(y_shifted)
        ρ_new = T(1)
    elseif class === :ces
        c_new = exp.(y_shifted ./ (1 + σ))
        ρ_new = T(σ / (1 + σ))
    else
        error("Unknown class for market expansion: $class.")
    end
    add_to_market!(fa, c_new, ρ_new, w_new)
    return fa
end

# -----------------------------------------------------------------------
# Pretty-print a column's class. Each class is labeled by its own name —
# we deliberately do NOT collapse `:linear` / `:leontief` into "ces(±∞)"
# even though they sit on the CES boundary in parameter space, because
# both are stored as their own atom types (LinearAgent / LeontiefAgent),
# not via the CES (c, ρ) channel. Only `:ces` carries an explicit ρ.
# This is the indicator the user sees in the iteration log.
# -----------------------------------------------------------------------
function format_class(class::Symbol, params::NamedTuple)
    if class === :linear
        return "linear"
    elseif class === :leontief
        return "leon"
    elseif class === :ces
        ρ = params.σ / (1 + params.σ)
        return "ces($(round(ρ; digits=2)))"
    elseif class === :nn
        return "nn(H=$(get(params, :hidden, NN_HIDDEN_DEFAULT)))"
    elseif class === :ql
        return "ql"
    elseif class === :ges
        return "ges"
    else
        return String(class)
    end
end

# -----------------------------------------------------------------------
# Runner-facing cut wrappers — used by cpm.jl (LP-CG) and accpm.jl
# (analytic-center CG) so both runners share the same oracle interface.
#
#   - find_cut_single(Ξ, u, μ, classes; ...)  → one improving column
#     (NamedTuple with rc), wraps solve_separation / fix-σ dispatch and
#     the sample-size subsampling re-expansion.
#   - find_cuts_multi(Ξ, u, classes; ...)     → vector of per-sample
#     inversion columns (multicut), wraps solve_separation_multicut.
#
# Both return γ matrices already expanded to the FULL K × n shape over
# the master's training set, so the caller can `add_to_gamma!` directly
# regardless of whether subsampling was used.
# -----------------------------------------------------------------------

using Random  # for randperm

# Re-expand a candidate's γ from subsampled to full K. Dispatches on
# `cand.class` so each android class can use its own parametric form
# (NN uses θ, CES uses (y, σ), linear uses y at σ=Inf).
function _gamma_over_full_from_cand(Ξ_full, cand)
    if cand.class === :nn
        n = length(Ξ_full[1][1])
        K = length(Ξ_full)
        γ = Matrix{Float64}(undef, K, n)
        H = get(cand.params, :hidden, NN_HIDDEN_DEFAULT)
        @inbounds for k in 1:K
            γ[k, :] .= nn_share(cand.params.θ, Ξ_full[k][1]; hidden=H)
        end
        return γ
    elseif cand.class === :ql
        n = length(Ξ_full[1][1])
        K = length(Ξ_full)
        γ = Matrix{Float64}(undef, K, n)
        c = cand.params.c
        w = cand.params.w
        @inbounds for k in 1:K
            γ[k, :] .= ql_share(c, Ξ_full[k][1], w)
        end
        return γ
    elseif cand.class === :leontief
        n = length(Ξ_full[1][1])
        K = length(Ξ_full)
        γ = Matrix{Float64}(undef, K, n)
        a = cand.params.a
        @inbounds for k in 1:K
            γ[k, :] .= leontief_share(a, Ξ_full[k][1])
        end
        return γ
    elseif cand.class === :ges
        n = length(Ξ_full[1][1])
        K = length(Ξ_full)
        γ = Matrix{Float64}(undef, K, n)
        c = cand.params.c
        r = cand.params.r
        w = cand.params.w
        @inbounds for k in 1:K
            γ[k, :] .= ges_share(c, r, Ξ_full[k][1], w)
        end
        return γ
    else
        return _gamma_over_full(Ξ_full, cand.params.y, cand.params.σ, cand.class)
    end
end

# Re-expand a single separation-oracle candidate (class, y, σ) back to the
# full K × n bidding matrix over `Ξ_full`. Used when the separation call ran
# on a random subset but the master needs the full-shape γ.
# CES uses the softmax recipe in `produce_gamma`; :linear (σ → +∞) needs
# the bang-per-buck argmax form from `compute_gamma`.
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

# Weighted sampling of `k` indices from 1:length(w) WITHOUT replacement,
# with selection probability proportional to w (Efraimidis–Spirakis A-Res:
# key_i = randexp()/w_i, take the k smallest keys). No StatsBase dependency.
# Nonpositive / non-finite weights are floored to a tiny ε so every sample
# stays eligible (AdaBoost keeps nonzero weight everywhere); if the whole
# weight vector is ~0 the caller falls back to uniform.
function _weighted_sample_no_replace(w::AbstractVector, k::Int)
    m = length(w)
    k >= m && return collect(1:m)
    wmax = maximum(w)
    ε = wmax > 0 ? 1e-12 * wmax : 1.0
    keys = [randexp() / max(w[i], ε) for i in 1:m]
    return partialsortperm(keys, 1:k)
end

# Internal: build the (Ξ_pr, u_pr) pair the separation oracle actually sees,
# given an optional sample-size cap. Returns (Ξ_pr, u_pr, do_sample::Bool).
#
# `weights` (per-sample, length K) turns the uniform mini-batch into a
# residual-weighted one (--sample-hard / boosting): the batch is drawn
# proportional to `weights` so the oracle concentrates on hard examples.
# This biases only *which candidate is proposed*; the reduced cost is always
# re-scored on the full data (see find_cut_single), so the CG stop is exact.
function _maybe_subsample(Ξ_train, u, sample_size::Int;
    weights::Union{AbstractVector,Nothing}=nothing)
    K_train = length(Ξ_train)
    do_sample = sample_size > 0 && sample_size < K_train
    if do_sample
        S = if isnothing(weights) || !any(>(0), weights)
            randperm(K_train)[1:sample_size]            # uniform (default / degenerate weights)
        else
            _weighted_sample_no_replace(weights, sample_size)
        end
        return Ξ_train[S], u[S, :], true
    else
        return Ξ_train, u, false
    end
end

"""
    find_cut_single(Ξ_train, u, μ, classes; kwargs...) -> NamedTuple

One improving cut from the per-class separation oracle.

The returned NamedTuple has fields `(γ_new, params, obj, class, rc)`,
with `γ_new` always shaped `K_train × n` (full master shape — sample-size
subsampling expansion is handled internally).

Keyword arguments:
- `nonh_w::Real` — shared pinning wealth w_0 for non-homothetic classes
  (eq.wealth.hybrid.lp). Forwarded explicitly through every layer down
  to each non-homothetic oracle (e.g., `solve_separation_ql(Ξ, u, w; …)`).
  Homothetic classes ignore it.
- `sample_size::Int = 0` — subsample size for the separation oracle (0 ⇒ full).
- `sample_weights::Union{AbstractVector,Nothing} = nothing` — per-sample
  weights for a residual-weighted mini-batch (--sample-hard / boosting). Only
  used when `sample_size` subsamples; `nothing` ⇒ uniform.
- `verbose::Bool = false`, `timelimit::Union{Real,Nothing} = nothing`.

The persistent caches for the :linear MILP (model + (y, γ) warm-start)
live inside `androids/linear.jl` and are managed there via
`clear_linear_separation_cache!()`; subsampling invalidates them by
calling that helper when the shape of `Ξ_pr` is unstable.
"""
function find_cut_single(Ξ_train, u::AbstractMatrix, μ::Real,
    classes::Vector{Symbol};
    nonh_w::Real=1.0,
    sample_size::Int=0,
    sample_weights::Union{AbstractVector,Nothing}=nothing,
    verbose::Bool=false,
    timelimit::Union{Real,Nothing}=nothing,
    kwargs...)   # forwards class-specific knobs (e.g. :nn_hidden, :nn_iters)
    # to solve_separation_class.

    Ξ_pr, u_pr, do_sample = _maybe_subsample(Ξ_train, u, sample_size; weights=sample_weights)
    # When subsampling, the cached linear MILP from the previous (larger
    # Ξ) call has the wrong shape — wipe it so this call rebuilds.
    do_sample && clear_linear_separation_cache!()

    sub = solve_separation(Ξ_pr, u_pr, μ, classes;
        nonh_w=nonh_w, verbose=verbose, timelimit=timelimit, kwargs...)
    if do_sample
        γ_full = _gamma_over_full_from_cand(Ξ_train, sub)
        return (γ_new=γ_full, params=sub.params, obj=sub.obj,
            class=sub.class, rc=reduced_cost(γ_full, u, μ))
    else
        return sub
    end
end

"""
    find_cuts_multi(Ξ_train, u, classes; sample_size=0) -> Vector{NamedTuple}

Multicut: per-sample inversion across the inversion-capable classes
(:ces and :linear). Each candidate's `γ_new` is expanded to the full
K_train × n shape; candidates with `class === :none` are filtered out.

Each returned NamedTuple has fields `(γ_new, y, σ, class)`. The caller
adds each via `add_to_gamma!` / `add_column_to_market!` with weight 0.
"""
function find_cuts_multi(Ξ_train, u::AbstractMatrix, classes::Vector{Symbol};
    sample_size::Int=0,
    sample_weights::Union{AbstractVector,Nothing}=nothing,
    kwargs...)   # forwards CES σ-bound kwargs (ces_sigma_lower, ces_sigma_upper)
    # to solve_separation_inversion_ces; other classes silently ignore.

    Ξ_pr, u_pr, do_sample = _maybe_subsample(Ξ_train, u, sample_size; weights=sample_weights)
    raw = solve_separation_multicut(Ξ_pr, u_pr, classes; kwargs...)
    out = NamedTuple[]
    for cand in raw
        cand.class === :none && continue
        γ_full = do_sample ?
                 _gamma_over_full_from_cand(Ξ_train, cand) :
                 cand.γ_new
        push!(out, (γ_new=γ_full, params=cand.params, class=cand.class))
    end
    return out
end

"""
    format_cuts_tag(cands) -> String

"ces×5+lin×3+leon×1"-style tag for the iteration-log "class" column from
a multicut candidate list. Returns "-" if the list is empty.
"""
function format_cuts_tag(cands)
    counts = Dict{Symbol,Int}(:ces => 0, :linear => 0, :leontief => 0)
    for c in cands
        counts[c.class] = get(counts, c.class, 0) + 1
    end
    tags = String[]
    counts[:ces] > 0 && push!(tags, "ces×$(counts[:ces])")
    counts[:linear] > 0 && push!(tags, "lin×$(counts[:linear])")
    counts[:leontief] > 0 && push!(tags, "leon×$(counts[:leontief])")
    return isempty(tags) ? "-" : join(tags, "+")
end
