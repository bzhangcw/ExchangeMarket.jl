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

include("./androids/ces.jl")
include("./androids/linear.jl")
include("./androids/leontief.jl")
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
    return local_extra
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

Remove agents with weight ≤ tol from both the FisherMarket and γ.
First syncs fa with w, then drops zero-weight agents. Returns the number dropped.
"""
function drop_zero_columns!(fa::FisherMarket{T}, γ_ref::Ref{Array{T,3}}, w::Vector{T}; tol=1e-8) where T
    m_γ = size(γ_ref[], 1)
    @assert length(w) == m_γ "w length ($(length(w))) must match γ_ref agent dim ($m_γ)"

    keep = findall(w .> tol)
    ndrop = m_γ - length(keep)
    ndrop == 0 && return 0

    γ_ref[] = γ_ref[][keep, :, :]

    fa.m = length(keep)
    fa.c = fa.c[:, keep]
    fa.ρ = fa.ρ[keep]
    fa.σ = fa.σ[keep]
    fa.w = w[keep]
    fa.x = fa.x[:, keep]
    fa.g = fa.g[:, keep]
    fa.s = fa.s[:, keep]
    while length(fa.val_u) < m_γ
        push!(fa.val_u, zero(T))
    end
    while length(fa.ε_br_play) < m_γ
        push!(fa.ε_br_play, fa.ε_br_play[1])
    end
    fa.val_u = fa.val_u[keep]
    fa.ε_br_play = fa.ε_br_play[keep]

    return ndrop
end

# -----------------------------------------------------------------------
# Per-class separation oracle
# -----------------------------------------------------------------------
"""
    solve_separation_class(class::Symbol, Ξ, u; kwargs...)

Run the separation subproblem for a single class. Returns a NamedTuple
    (γ_new::Matrix{T}, params::NamedTuple, obj::T, class::Symbol).

Supported classes:
- `:ces`      — full CES, free σ; LP warm-start + LBFGS refinement.
- `:linear`   — linear utility class H(1); big-M MIP (eq.cg.sep.linear).
- `:leontief` — CES boundary σ → -1⁺; concave fix-σ LBFGS at σ = σ_leontief.
"""
function solve_separation_class(class::Symbol,
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    kwargs...) where T
    if class === :ces
        return solve_separation_ces(Ξ, u; kwargs...)
    elseif class === :linear
        return solve_separation_linear(Ξ, u; kwargs...)
    elseif class === :leontief
        return solve_separation_leontief(Ξ, u; kwargs...)
    elseif class === :ql
        return solve_separation_ql(Ξ, u; kwargs...)
    elseif class === :nn
        # Forward NN-specific knobs from the caller's kwargs (the
        # signatures don't overlap with the others, so other classes
        # ignore them silently).
        hidden = get(kwargs, :nn_hidden, NN_HIDDEN_DEFAULT)
        max_iters = get(kwargs, :nn_iters, NN_ITERS_DEFAULT)
        return solve_separation_nn(Ξ, u; hidden=hidden, max_iters=max_iters,
                                    kwargs...)
    else
        error("Unknown separation class: $class. Supported: :ces, :linear, :leontief, :ql, :nn.")
    end
end

"""
    solve_separation(Ξ, u, μ, classes; kwargs...)

Solve the separation subproblem for each class in `classes`, compute the reduced
cost of each candidate column, and return the candidate with the largest rc.

Returns a NamedTuple
    (γ_new, params, obj, class, rc)
where `params` and `class` are passed through from the winning class's
solver and `rc = Σ_k <u_k, γ_new_k> - μ`.
"""
function solve_separation(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T}, μ::T, classes::Vector{Symbol};
    verbose::Bool=false,
    kwargs...) where T
    @assert !isempty(classes) "classes must be non-empty"
    best = nothing
    best_rc = T(-Inf)
    for class in classes
        cand = solve_separation_class(class, Ξ, u; verbose=verbose, kwargs...)
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
    σ_grid=range(LOWER_SIGMA_BOUND, UPPER_SIGMA_BOUND, length=50)
) where T
    K = length(Ξ)
    n = length(Ξ[1][1])
    ces_results = (:ces in classes) ?
                  solve_separation_inversion_ces(Ξ, u; σ_grid=σ_grid) : nothing
    lin_results = (:linear in classes) ?
                  solve_separation_inversion_linear(Ξ, u) : nothing

    chosen = Vector{NamedTuple}(undef, K)
    for k in 1:K
        best = (class=:none, y=zeros(T, n), σ=T(NaN),
            γ_new=zeros(T, K, n), obj=T(-Inf))
        if !isnothing(ces_results)
            y, σ, γ_new, obj = ces_results[k]
            if obj > best.obj
                best = (class=:ces, y=y, σ=σ, γ_new=γ_new, obj=obj)
            end
        end
        if !isnothing(lin_results)
            y, γ_new, obj = lin_results[k]
            if obj > best.obj
                best = (class=:linear, y=y, σ=T(Inf), γ_new=γ_new, obj=obj)
            end
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
    elseif class === :ces || class === :leontief
        c_new = exp.(y_shifted ./ (1 + σ))
        ρ_new = T(σ / (1 + σ))
    elseif class === :ql
        error(":ql androids cannot be stored in FisherMarket — the QL share is " *
              "piecewise and budget-dependent, not CES. A parallel container " *
              "or a per-android class tag in FisherMarket is needed before " *
              "QL androids can participate in the master LP. The separation oracle " *
              "(solve_separation_ql) works standalone for off-line analysis.")
    else
        error("Unknown class for market expansion: $class.")
    end
    add_to_market!(fa, c_new, ρ_new, w_new)
    return fa
end

# -----------------------------------------------------------------------
# Pretty-print a column's class as ces(ρ).
#   :linear   → ces(1)
#   :leontief → ces(-∞)
#   :ces      → ces(0.42) with the recovered ρ
# This is the indicator the user sees in the iteration log.
# -----------------------------------------------------------------------
function format_class(class::Symbol, params::NamedTuple)
    if class === :linear
        return "ces(1)"
    elseif class === :leontief
        return "ces(-∞)"
    elseif class === :ces
        ρ = params.σ / (1 + params.σ)
        return "ces($(round(ρ; digits=2)))"
    elseif class === :nn
        return "nn(H=$(get(params, :hidden, NN_HIDDEN_DEFAULT)))"
    else
        return String(class)
    end
end

format_class_from_yσ(y, σ) = "ces($(round(σ / (1 + σ); digits=2)))"

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

# Internal: build the (Ξ_pr, u_pr) pair the separation oracle actually sees,
# given an optional sample-size cap. Returns (Ξ_pr, u_pr, do_sample::Bool).
function _maybe_subsample(Ξ_train, u, sample_size::Int)
    K_train = length(Ξ_train)
    do_sample = sample_size > 0 && sample_size < K_train
    if do_sample
        S = randperm(K_train)[1:sample_size]
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
- `sample_size::Int = 0` — subsample size for the separation oracle (0 ⇒ full).
- `fixed_rho_ces::Union{Real,Nothing} = nothing` — if set AND `:ces in classes`,
  run CES at a fixed ρ via `solve_separation_fix_σ_ces` and pick the best of
  that vs. the per-class separation-oracle result (by reduced cost). Used by
  cpm.jl's cgma post-demotion cleanup.
- `linear_y_warm::Union{Vector,Nothing} = nothing`,
  `linear_γ_warm::Union{Matrix,Nothing} = nothing` — warm-start payload
  for the :linear MILP. Other classes ignore.
- `linear_model_cache::Union{Ref,Nothing} = nothing` — persistent Gurobi
  model cache for the :linear MILP (disable when subsampling, since Ξ_pr
  changes shape across iterations).
- `verbose::Bool = false`, `timelimit::Union{Real,Nothing} = nothing`.
"""
function find_cut_single(Ξ_train, u::AbstractMatrix, μ::Real,
    classes::Vector{Symbol};
    sample_size::Int=0,
    fixed_rho_ces::Union{Real,Nothing}=nothing,
    linear_y_warm=nothing,
    linear_γ_warm=nothing,
    linear_model_cache=nothing,
    verbose::Bool=false,
    timelimit::Union{Real,Nothing}=nothing,
    kwargs...)   # forwards class-specific knobs (e.g. :nn_hidden, :nn_iters)
                 # through to solve_separation_class

    Ξ_pr, u_pr, do_sample = _maybe_subsample(Ξ_train, u, sample_size)
    # When subsampling, the cached linear MILP from the previous (larger
    # Ξ) call has the wrong shape — disable cache for this call.
    _cache = do_sample ? nothing : linear_model_cache
    # γ from previous round was sized for the previous Ξ_pr; only reuse
    # as MIPstart when the shape is stable (no subsampling).
    _γ_warm = do_sample ? nothing : linear_γ_warm

    apply_fixed_rho = !isnothing(fixed_rho_ces) && (:ces in classes)

    if apply_fixed_rho
        σ_fixed = fixed_rho_ces / (1 - fixed_rho_ces)
        y_opt, σ_out, γ_new, obj_val = solve_separation_fix_σ_ces(Ξ_pr, u_pr, σ_fixed;
            timelimit=timelimit)
        γ_full = do_sample ? _gamma_over_full(Ξ_train, y_opt, σ_out, :ces) : γ_new
        cand = (γ_new=γ_full, params=(y=y_opt, σ=σ_out), obj=obj_val,
            class=:ces, rc=reduced_cost(γ_full, u, μ))
        other_classes = filter(c -> c !== :ces, classes)
        if !isempty(other_classes)
            sub = solve_separation(Ξ_pr, u_pr, μ, other_classes;
                verbose=verbose, timelimit=timelimit,
                linear_y_warm=linear_y_warm, linear_γ_warm=_γ_warm,
                linear_model_cache=_cache, kwargs...)
            γ_other_full = do_sample ?
                           _gamma_over_full(Ξ_train, sub.params.y, sub.params.σ, sub.class) :
                           sub.γ_new
            cand_other = (γ_new=γ_other_full, params=sub.params, obj=sub.obj,
                class=sub.class, rc=reduced_cost(γ_other_full, u, μ))
            cand_other.rc > cand.rc && (cand = cand_other)
        end
        return cand
    end

    sub = solve_separation(Ξ_pr, u_pr, μ, classes;
        verbose=verbose, timelimit=timelimit,
        linear_y_warm=linear_y_warm, linear_γ_warm=_γ_warm,
        linear_model_cache=_cache, kwargs...)
    if do_sample
        γ_full = _gamma_over_full(Ξ_train, sub.params.y, sub.params.σ, sub.class)
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
    sample_size::Int=0)

    Ξ_pr, u_pr, do_sample = _maybe_subsample(Ξ_train, u, sample_size)
    raw = solve_separation_multicut(Ξ_pr, u_pr, classes)
    out = NamedTuple[]
    for cand in raw
        cand.class === :none && continue
        γ_full = do_sample ?
                 _gamma_over_full(Ξ_train, cand.y, cand.σ, cand.class) :
                 cand.γ_new
        push!(out, (γ_new=γ_full, y=cand.y, σ=cand.σ, class=cand.class))
    end
    return out
end

"""
    format_cuts_tag(cands) -> String

"ces×5+lin×3"-style tag for the iteration-log "class" column from a
multicut candidate list. Returns "-" if the list is empty.
"""
function format_cuts_tag(cands)
    counts = Dict{Symbol,Int}(:ces => 0, :linear => 0)
    for c in cands
        counts[c.class] = get(counts, c.class, 0) + 1
    end
    tags = String[]
    counts[:ces] > 0 && push!(tags, "ces×$(counts[:ces])")
    counts[:linear] > 0 && push!(tags, "lin×$(counts[:linear])")
    return isempty(tags) ? "-" : join(tags, "+")
end
