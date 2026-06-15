# Arrow–Debreu wealth redistribution — master (primal), dual, n-call
# pricing oracle, test-error, and the zero-atom drop primitive. The
# column-generation loop (`run_ad_tracked`) lives in cpm_ad.jl.
#
# This is the Arrow–Debreu sibling of the Fisher master/dual file
# (redistribute.jl; the loop sibling is cpm.jl). It mirrors `§ The Arrow–Debreu case` in
# overleaf/read-econ/wealth-dist.tex. The difference from Fisher is the
# atom variable and the supply constraint:
#
#   Fisher : scalar budget w_t ≥ 0, Σ_t w_t = 1 (scalar dual μ),
#            predictor Σ_t w_t γ_t(p_k).
#   AD     : endowment b_t ∈ ℝⁿ₊,   Σ_t b_t = δ·1 (n-vector dual ν),
#            price-dependent budget w_{t,k} = ⟨p_k, b_t⟩,
#            predictor Σ_t ⟨p_k, b_t⟩ γ_t(p_k).
#
# The supply scale δ (sec.wealth.ad.scale / eq.ad.master.scaled) defaults to
# the unit supply δ = 1; it can be fixed at another value (--ad-delta) or left
# to the master as an LP variable (--ad-delta-free). The balance residual then
# carries the known shift (δ-1)p_k; cuts and pricing are unchanged.
#
# Homothetic classes only (ces / linear / leontief): there is no pinning
# machinery here, so the loop is simpler than run_method_tracked.
#
# Key reuse: the AD reduced cost for good j is exactly the Fisher reduced
# cost evaluated on the price-reweighted dual ũ^{(j)}_k = (p_k)_j · u_k
# with the scalar offset μ = ν_j. So `find_cut_single` / `solve_separation`
# (separation.jl) and `_gamma_over_full_from_cand` are reused verbatim;
# AD only swaps the master and runs the oracle n times (one per good).
#
# Depends on (in scope once setup.jl has included engine.jl and
# separation.jl): `new_model`, `set_lp_barrier!`, `find_cut_single`,
# `_gamma_over_full_from_cand`.

using JuMP
using LinearAlgebra
using Random   # randperm/shuffle — good-scan order + random masks (find_cut_single_ad / make_ad_mask)
# Master models built via `new_model()` (engine.jl): Gurobi when available, else Mosek.
using ArgParse # register_cli_ad! / apply_cli_ad!

# The fitted AD surrogate is a first-class ExchangeMarket.ArrowDebreuMarket:
# atoms become CES columns (c, ρ) and the master's endowment matrix B becomes
# the market's endowments b (n×m). See `ad_market_from_atoms` below. The
# shared drivers (run_one_method.jl / run_test.jl) read `fa.m`, deepcopy, and
# serialize it like any market object.
#
# NOTE: the drivers read the test error from `history[:test_err]` (computed
# in-loop via evaluate_test_error_ad), and surrogate-equilibrium validation
# (`validate_surrogate`) is not wired for AD (neither as surrogate nor as
# ground truth); the drivers skip it via `isa ArrowDebreuMarket` guards.

"""
    ad_market_from_atoms(cands, B) -> ArrowDebreuMarket

Convert the AD CG loop's internal state — the atom list `cands` (each a
NamedTuple with `class`/`params`) and the m×n endowment matrix `B` from the
final master solve — into an `ArrowDebreuMarket`.

Atom storage uses the CES (c, ρ) parameterization (same map as
`add_column_to_market!` in separation.jl):
- `:ces`    — c = exp(y / (1+σ)), ρ = σ/(1+σ), with y shifted so max(y) = 0.
- `:linear` — c = exp(y), ρ = 1 (compute_gamma/ces_share special-case σ = ∞).
- `:leontief` cannot ride the (c, ρ) form (σ → -1 diverges); run_ad_tracked
  rejects it up front.

The market's endowments are `b = Matrix(B')` (n×m, note transpose: B is m×n).
"""
function ad_market_from_atoms(cands, B::AbstractMatrix)
    m = length(cands)
    n = size(B, 2)
    @assert size(B, 1) == m "B must be m×n with m = length(cands)"
    C = zeros(n, m)
    ρ_vec = zeros(m)
    for (t, cand) in enumerate(cands)
        if cand.class === :ces
            y, σ = cand.params.y, cand.params.σ
            y_shifted = y .- maximum(y)
            C[:, t] .= exp.(y_shifted ./ (1 + σ))
            ρ_vec[t] = σ / (1 + σ)
        elseif cand.class === :linear
            y = cand.params.y
            C[:, t] .= exp.(y .- maximum(y))
            ρ_vec[t] = 1.0
        else
            error("ad_market_from_atoms: class :$(cand.class) cannot be stored " *
                  "in an ArrowDebreuMarket's CES (c, ρ) form.")
        end
    end
    return ArrowDebreuMarket(m, n; c=C, ρ=ρ_vec, b=Matrix(B'), verbose=false)
end

"""
    solve_wealth_redist_primal_ad(Ξ, γ; masks=nothing, delta=1.0, verbose=false, timelimit=nothing)

Solve the Arrow–Debreu master (`eq.ad.master` / scaled `eq.ad.master.scaled`
in wealth-dist.tex), optionally with endowment masks (`eq.ad.mask`,
sec.wealth.ad.mask) and a supply scale δ (sec.wealth.ad.scale):

    L^AD = min_{b, s, [δ]}  Σ_{k,j} |s_{k,j}|
    s.t.  s_k + Σ_t ⟨p_k, b_t⟩ γ_{t,k} = P_k g_k + (δ-1) p_k,   ∀k∈[K]
          Σ_t b_t = δ·1   (componentwise, n rows)
          b_t ∈ ℝⁿ₊,  (b_t)_l = 0 ∀ l ∉ S_t,  s_k ∈ ℝⁿ,  [δ ≥ 0]

The predictor coefficient on `b[t,l]` in balance row (k,j) is
`γ[t,k,j]·p_k[l]` (since ⟨p_k,b_t⟩ = Σ_l p_k[l] b[t,l]); the program is an
LP in `b` (and δ — both the shifted balance and the scaled supply are
jointly linear in (b, δ)). The ‖·‖∞-per-sample → Σ|·| objective is lifted
via auxiliary `tt ≥ ±s`, matching the Fisher master.

Arguments:
- Ξ: Vector of (p_k, g_k) tuples, K observations.
- γ: bidding tensor of size (m, K, n) (atom t's share γ_t(p_k)).

Keyword arguments:
- `masks::Union{Vector{Vector{Int}},Nothing}` — per-atom endowment supports
  S_t ⊆ [n] (sec.wealth.ad.mask). Atom t gets variables only for goods in
  masks[t]; `nothing` ⇒ full support for every atom (the unmasked master).
  Every good must be owned by at least one atom, else the supply row is
  infeasible (asserted).
- `delta::Union{Real,Symbol}` — surrogate supply scale δ (eq.ad.master.scaled).
  A number fixes δ (default 1.0 recovers eq.ad.master exactly: the shift
  (δ-1)p_k vanishes and supply is unit). `:free` makes δ ≥ 0 a decision
  variable; the master stays an LP whose dual gains the scalar constraint
  Σ_k⟨u_k,p_k⟩ ≤ ⟨ν,1⟩ and whose objective becomes the excess-demand
  comparison Σ_k⟨u_k, P_k g_k - p_k⟩ (eq.ad.dual.scaled) — all handled
  automatically by LP duality; the pricing cuts are unchanged.
- `cache::Union{Ref,Nothing}` — persistent JuMP model (same protocol as the
  Fisher master's cache). On hit, only the new atoms' masked b variables are
  appended via `set_normalized_coefficient`; the K·n balance rows (the
  O(K·m·n·|S|) expression-construction cost of a rebuild), the slack/lift
  variables, the supply rows, the δ variable, and the objective are all
  reused. The caller wipes the Ref to force a rebuild (e.g. after
  `drop_zero_atoms_ad` reshuffles γ rows). The δ spec must not change across
  cached solves (asserted).
- `timelimit::Union{Real,Nothing}` — accepted for API compatibility but
  intentionally NOT applied to the master solve. Gurobi's barrier (Method=2,
  no crossover) returns no solution if cut off before convergence, which makes
  the downstream `value(...)` reads crash; the CG loop enforces the wall-clock
  budget between iterations instead, so the master always runs to completion.

Returns `(B, s, model, balance, supply, δ_val)`:
- B: m×n endowment matrix (zeros outside each atom's mask).
- s: K×n slack matrix.
- model, balance (K×n ConstraintRef), supply (length-n ConstraintRef vector).
- δ_val: the supply scale used (the fixed number, or the optimal value of
  the free variable).
"""
function solve_wealth_redist_primal_ad(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    γ::Array{T,3};
    masks::Union{Vector{Vector{Int}},Nothing}=nothing,
    delta::Union{Real,Symbol}=1.0,
    verbose=false,
    timelimit::Union{Real,Nothing}=nothing,
    cache::Union{Ref,Nothing}=nothing,
) where T
    m, K, n = size(γ)
    # Supply-scale spec: a fixed number δ ≥ 0, or :free (δ becomes an LP variable).
    delta_free = delta === :free
    if !delta_free
        @assert delta isa Real && delta >= 0 "delta must be a nonnegative number or :free (got $delta)"
    end
    # Owned goods of atom t (full support when unmasked).
    _owned(t) = isnothing(masks) ? collect(1:n) : masks[t]
    # Feasibility: every good needs ≥ 1 owner, else Σ_t b_{t,l} = δ has no
    # variables on its left-hand side.
    if !isnothing(masks)
        @assert length(masks) == m "masks must have one entry per atom"
        owned_union = sort(unique(vcat(masks...)))
        @assert owned_union == collect(1:n) "every good must be owned by ≥1 atom " *
                                            "(unowned: $(setdiff(1:n, owned_union)))"
    end

    # ----------------------------------------------------------------
    # Cache hit: same (K, n), and γ grew by appending atoms. Append the
    # masked b variables per new atom and extend the balance / supply
    # coefficients — O(K·n·|S_t|) per new atom instead of a full rebuild.
    # ----------------------------------------------------------------
    cache_hit = false
    model = nothing
    b_vars = nothing       # Vector of per-atom Dict(good => VariableRef)
    s_var = nothing
    balance = nothing
    supply = nothing
    δ_var = nothing        # VariableRef when delta === :free, else nothing
    if !isnothing(cache) && !isnothing(cache[])
        c = cache[]
        if c.K == K && c.n == n && c.last_m <= m
            # The δ spec is fixed for the lifetime of a cached model: its terms
            # sit inside the balance/supply rows and cannot be retrofitted.
            @assert c.delta === delta "delta spec changed across cached solves " *
                                      "($(c.delta) → $delta); wipe the cache to rebuild"
            model = c.model
            b_vars = c.b_vars
            s_var = c.s
            balance = c.balance
            supply = c.supply
            δ_var = c.δ_var
            for t in (c.last_m+1):m
                # Apply the lower bound explicitly with set_lower_bound: the
                # `lower_bound = 0.0` kwarg on an anonymous @variable is
                # silently dropped by some JuMP versions (same pitfall noted
                # in the Fisher master cache).
                d = Dict{Int,VariableRef}()
                for l in _owned(t)
                    v = @variable(model, base_name = "b_$(t)_$(l)")
                    set_lower_bound(v, 0.0)
                    d[l] = v
                end
                push!(b_vars, d)
                for k in 1:K
                    p_k = Ξ[k][1]
                    for j in 1:n, (l, v) in d
                        coef = γ[t, k, j] * p_k[l]
                        iszero(coef) && continue
                        set_normalized_coefficient(balance[k, j], v, coef)
                    end
                end
                for (l, v) in d
                    set_normalized_coefficient(supply[l], v, 1.0)
                end
            end
            cache_hit = true
            # NOTE: `timelimit` is intentionally NOT applied to the master
            # solve. Gurobi's barrier (Method=2, no crossover) returns *no*
            # solution if it is cut off before converging, and the downstream
            # `value(...)` reads then crash ("0 solution(s) in the model").
            # The CG loop (run_ad_tracked) already enforces the wall-clock
            # budget by checking the elapsed time before each iteration, so the
            # master is left to run to completion and always returns a point.
        end
    end

    if !cache_hit
        model = new_model()
        # Barrier without crossover, same rationale as the Fisher master: a
        # near-optimal interior (u, ν) is all the separation oracle consumes,
        # and the smoother dual trajectory yields more diverse columns.
        # (Gurobi-only; no-op on Mosek, whose LP default is already IPM.)
        set_lp_barrier!(model)
        # See the note above: no per-solve time limit on the master.

        # Masked endowment variables: atom t owns only goods in _owned(t).
        b_vars = Vector{Dict{Int,VariableRef}}()
        for t in 1:m
            d = Dict{Int,VariableRef}()
            for l in _owned(t)
                v = @variable(model, base_name = "b_$(t)_$(l)")
                set_lower_bound(v, 0.0)
                d[l] = v
            end
            push!(b_vars, d)
        end
        @variable(model, s[1:K, 1:n])
        s_var = s
        @variable(model, tt[1:K, 1:n] >= 0)

        # Supply scale δ (eq.ad.master.scaled): a decision variable when :free.
        if delta_free
            δ_var = @variable(model, base_name = "δ")
            set_lower_bound(δ_var, 0.0)
        end

        # Per-(sample, good) balance with the (δ-1)p_k shift (eq.ad.shifted.data):
        #   s_{k,j} + Σ_t Σ_{l∈S_t} γ[t,k,j] p_k[l] b[t,l] = (p_k⊙g_k)_j + (δ-1)(p_k)_j
        # For fixed δ the shift goes to the RHS; for free δ the variable term
        # -δ(p_k)_j moves to the LHS and the constant -(p_k)_j to the RHS.
        balance = Matrix{ConstraintRef}(undef, K, n)
        for k in 1:K
            p_k, g_k = Ξ[k]
            Pg = p_k .* g_k
            for j in 1:n
                lhs = s_var[k, j] +
                      sum(γ[t, k, j] * p_k[l] * v for t in 1:m for (l, v) in b_vars[t])
                if delta_free
                    balance[k, j] = @constraint(model, lhs - p_k[j] * δ_var == Pg[j] - p_k[j])
                else
                    balance[k, j] = @constraint(model, lhs == Pg[j] + (delta - 1) * p_k[j])
                end
            end
        end

        # tt_{k,j} ≥ |s_{k,j}|.
        for k in 1:K, j in 1:n
            @constraint(model, tt[k, j] >= s_var[k, j])
            @constraint(model, tt[k, j] >= -s_var[k, j])
        end

        # Supply: Σ_{t: l∈S_t} b_{t,l} = δ for every good l (n rows; dual ν ∈ ℝⁿ).
        supply = Vector{ConstraintRef}(undef, n)
        for l in 1:n
            owners = [b_vars[t][l] for t in 1:m if haskey(b_vars[t], l)]
            if delta_free
                supply[l] = @constraint(model, sum(owners) - δ_var == 0)
            else
                supply[l] = @constraint(model, sum(owners) == delta)
            end
        end

        @objective(model, Min, sum(tt))
    end

    # Per-call verbose toggle on every solve (rebuild OR cache hit), so a
    # cached model doesn't leak iter-1's verbose flags into later iters.
    # Solver-portable (Gurobi "OutputFlag"/"LogToConsole" do not exist on Mosek).
    verbose ? unset_silent(model) : set_silent(model)
    optimize!(model)

    if !isnothing(cache)
        cache[] = (model=model, b_vars=b_vars, s=s_var, balance=balance,
            supply=supply, δ_var=δ_var, delta=delta, last_m=m, K=K, n=n)
    end

    # No primal point available (e.g. barrier numerical failure or
    # infeasibility): reading `value(...)` would throw the cryptic MOI
    # "0 solution(s) in the model". Return a sentinel `B = nothing` instead and
    # let the caller decide whether to fall back to the last good surrogate or
    # error. `model`/`balance`/`supply` are still returned so the caller can
    # inspect `termination_status(model)` for the message.
    if !has_values(model)
        return nothing, nothing, model, balance, supply, NaN
    end

    # Assemble B (m×n) from the per-atom variable dicts (zeros off-mask).
    B = zeros(Float64, m, n)
    for t in 1:m
        for (l, v) in b_vars[t]
            B[t, l] = value(v)
        end
    end
    δ_val = delta_free ? value(δ_var) : Float64(delta)
    return B, value.(s_var), model, balance, supply, δ_val
end

"""
    extract_duals_ad(model, balance, supply, K, n)

Extract `(u, ν)` from a solved AD master:
- u[k,j] = dual of balance constraint (k,j).
- ν[l]   = -dual of supply constraint l (MOI sign convention, matching
  `extract_duals`'s μ = -dual(budget)). ν ∈ ℝⁿ.
"""
function extract_duals_ad(model, balance::Matrix{ConstraintRef},
    supply::Vector{ConstraintRef}, K::Int, n::Int)
    u = [dual(balance[k, j]) for k in 1:K, j in 1:n]
    ν = [-dual(supply[l]) for l in 1:n]
    return u, ν
end

"""
    find_cut_single_ad(Ξ, u, ν, classes; verbose, timelimit, early_exit_rc=nothing,
                       kwargs...)

Arrow–Debreu pricing (`eq.ad.pricing`). The most-violated cut is

    π^AD = max_{j∈[n]} [ sup_{γ∈ℋ_H} Σ_k ⟨(p_k)_j u_k, γ(p_k)⟩ − ν_j ],

i.e. for each good j run the homothetic separation oracle on the
price-reweighted dual ũ^{(j)}_k = (p_k)_j · u_k with scalar offset ν_j,
and keep the candidate with the largest reduced cost across goods. The
inner sup is exactly `find_cut_single` (separation.jl) with μ = ν_j.

Two accelerations over the naive n-call loop (CG correctness needs *a*
violated cut, not the most violated one):

1. Early exit (`early_exit_rc`): goods are scanned in random order and the
   scan stops at the first candidate with `rc > early_exit_rc`. Pass the CG
   loop's `tol_rc`; `nothing` disables and scans all n goods (exact max).
2. Threading: when Julia has >1 thread AND `:linear ∉ classes`, the n calls
   run under `Threads.@threads`. The linear MIP oracle keeps a module-level
   Gurobi-model cache (`_linear_model_cache`) that is not thread-safe, so any
   class list containing `:linear` falls back to the serial scan. (CES /
   Leontief oracles are pure Optim.jl + thread-local allocations except the
   CES dual-LP warm start, which builds a fresh Mosek model per call.)
   Threading and early exit compose: each thread checks a shared atomic flag.

A third reduction (pricing side, independent of the endowment masks): the
`scan_order` argument prioritizes the goods whose cuts are most likely
violated — e.g. sorted by the master's per-good residual. Combined with
early exit, only those "interesting" coordinates actually get an oracle
call; the master keeps full endowments B. The order always covers all n
goods, so a scan with no violated cut still certifies convergence.

Returns the winning candidate NamedTuple `(γ_new, params, obj, class, rc, good)`
(γ_new over the full Ξ; `good` is the index j whose reweighted oracle won —
also used to build the atom's endowment mask, cf. sec.wealth.ad.mask), or
`nothing` if every scanned good's oracle failed.
"""
function find_cut_single_ad(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T}, ν::Vector{T}, classes::Vector{Symbol};
    verbose::Bool=false,
    timelimit::Union{Real,Nothing}=nothing,
    early_exit_rc::Union{Real,Nothing}=nothing,
    scan_order::Union{Vector{Int},Nothing}=nothing,
    kwargs...) where T
    K = length(Ξ)
    n = length(ν)

    # Reweighted dual for good j: ũ_k = (p_k)_j u_k.
    _reweight(j) = begin
        ũ = similar(u)
        for k in 1:K
            ũ[k, :] .= Ξ[k][1][j] .* @view u[k, :]
        end
        ũ
    end
    # Tag each candidate with the good whose oracle produced it.
    _solve(j) = begin
        c = find_cut_single(Ξ, _reweight(j), T(ν[j]), classes;
            verbose=verbose, timelimit=timelimit, kwargs...)
        isnothing(c) ? nothing : merge(c, (good=j,))
    end
    _ok(c) = !isnothing(c) && isfinite(c.rc)

    # Scan priority: caller-provided (e.g. residual-sorted, "interesting"
    # coordinates first) or random — random avoids early exit systematically
    # favoring low-index goods, which would starve the others' cuts.
    order = isnothing(scan_order) ? randperm(n) : scan_order
    @assert length(order) == n "scan_order must be a permutation of 1:n"

    # Suppress the inner good-scan threading when run_test.jl is already fitting
    # variants concurrently (one thread per method), to avoid nested @threads and
    # oversubscription.
    use_threads = Threads.nthreads() > 1 && !(:linear in classes) && !parallel_variants()

    if !use_threads
        # Serial scan with early exit.
        best = nothing
        best_rc = T(-Inf)
        for j in order
            cand = _solve(j)
            verbose && _ok(cand) &&
                println("  good j=$j: class=$(cand.class), rc=$(cand.rc)")
            if _ok(cand) && cand.rc > best_rc
                best_rc = cand.rc
                best = cand
            end
            if !isnothing(early_exit_rc) && best_rc > early_exit_rc
                verbose && println("  early exit at good j=$j (rc=$(best_rc) > $(early_exit_rc))")
                break
            end
        end
        return best
    end

    # Threaded scan. Each task writes its candidate into a slot; a shared
    # atomic flag lets later tasks skip work once some thread found a
    # violated cut (cooperative early exit — already-running oracles finish).
    cands = Vector{Any}(nothing, n)
    found = Threads.Atomic{Bool}(false)
    Threads.@threads for j in order
        if !isnothing(early_exit_rc) && found[]
            continue
        end
        c = _solve(j)
        cands[j] = c
        if !isnothing(early_exit_rc) && _ok(c) && c.rc > early_exit_rc
            found[] = true   # atomic store
        end
    end

    best = nothing
    best_rc = T(-Inf)
    for j in 1:n
        c = cands[j]
        verbose && _ok(c) &&
            println("  good j=$j: class=$(c.class), rc=$(c.rc)")
        if _ok(c) && c.rc > best_rc
            best_rc = c.rc
            best = c
        end
    end
    return best
end

"""
    make_ad_mask(mode, n, j_win; mask_size=1, rng=Random.GLOBAL_RNG) -> Vector{Int}

Build the endowment mask S_t of a freshly generated AD atom
(sec.wealth.ad.mask). `j_win` is the good whose pricing oracle produced the
atom (it is always included, so the atom can serve the cut it violates).

Modes:
- `:single` — S_t = {j_win}: one variable per atom, the cheapest master.
- `:full`   — S_t = [n]: the unmasked master (eq.ad.master).
- `:random` — S_t = {j_win} ∪ (mask_size - 1 goods sampled uniformly
  without replacement from the rest).
"""
function make_ad_mask(mode::Symbol, n::Int, j_win::Int;
    mask_size::Int=1, rng=Random.GLOBAL_RNG)
    if mode === :single
        return [j_win]
    elseif mode === :full
        return collect(1:n)
    elseif mode === :random
        size_eff = clamp(mask_size, 1, n)
        rest = shuffle(rng, setdiff(1:n, j_win))
        return sort(vcat(j_win, rest[1:min(size_eff - 1, length(rest))]))
    else
        error("make_ad_mask: unknown mode :$mode (use :single, :full, :random)")
    end
end

# ---- CLI surface --------------------------------------------------------
"""
    register_cli_ad!(s::ArgParseSettings)

Add the "Master: Arrow–Debreu" arg group: `--ad-endow-mode`, `--ad-mask-size`,
`--ad-delta`, `--ad-delta-free`. All flow through `local_extra` into the adcg
runner kwargs (run_ad_tracked's :ad_endow_mode / :ad_mask_size / :ad_delta).
"""
function register_cli_ad!(s::ArgParseSettings)
    add_arg_group!(s, "Master: Arrow–Debreu (adcg)")
    @add_arg_table! s begin
        "--ad-endow-mode"
        help = "Endowment mask of each new AD atom (sec.wealth.ad.mask), adcg only: " *
               "single (atom owns only the good whose oracle generated it — master of Fisher size), " *
               "full (default; atom may own all goods — the unmasked AD master), " *
               "mask (atom owns the winning good + random others; size via --ad-mask-size). " *
               "adfw / adfwjl ignore this — their δ=1 bundle-hull master is single-good by " *
               "construction, so `mask` warns and falls back to single."
        arg_type = String
        default = "full"
        range_tester = x -> x in ("single", "full", "mask")
        "--ad-mask-size"
        help = "Mask size |S_t| for --ad-endow-mode mask: the winning good + (S-1) goods sampled uniformly at random. Clamped to [1, n]."
        arg_type = Int
        default = 2
        "--ad-delta"
        help = "Surrogate supply scale δ ≥ 0 (sec.wealth.ad.scale): supply rows become Σ_t b_t = δ·1 " *
               "and the balance residual gains the shift (δ-1)p_k. Default 1.0 (the unit-supply master eq.ad.master). " *
               "Ignored when --ad-delta-free is set."
        arg_type = Float64
        default = 1.0
        "--ad-delta-free"
        help = "Let the master choose δ ≥ 0 as an LP decision variable (eq.ad.master.scaled). " *
               "Overrides --ad-delta."
        action = :store_true
    end
    return s
end

"""
    apply_cli_ad!(local_extra::Dict, cli)

Forward `--ad-endow-mode` / `--ad-mask-size` / `--ad-delta` / `--ad-delta-free`
into the runner kwargs. The CLI token `mask` maps to run_ad_tracked's `:random`
mode; `--ad-delta-free` maps to `:ad_delta => :free`.
"""
function apply_cli_ad!(local_extra::Dict, cli)
    mode = cli["ad_endow_mode"]
    local_extra[:ad_endow_mode] = mode == "mask" ? :random : Symbol(mode)
    local_extra[:ad_mask_size] = cli["ad_mask_size"]
    local_extra[:ad_delta] = cli["ad_delta_free"] ? :free : cli["ad_delta"]
    return local_extra
end

"""
    evaluate_test_error_ad(cands, B, Ξ_test; delta=1.0)

Mean ℓ₁ expenditure-share error of the AD surrogate on `Ξ_test`:
for each (p, g), the fitted expenditure is Σ_t ⟨p, b_t⟩ γ_t(p), compared
to the target p⊙g + (δ-1)p (eq.ad.shifted.data; at δ = 1 this is just p⊙g).
`cands[t]` is the winning oracle NamedTuple (carries `class`/`params`),
`B` is the m×n endowment matrix, `delta` is the supply scale the master was
solved with. Each atom's share at the test prices is recomputed via
`_gamma_over_full_from_cand` (the same class-aware expansion the oracle
uses), so all homothetic classes work.
"""
function evaluate_test_error_ad(cands, B::AbstractMatrix{T}, Ξ_test;
    delta::Real=1.0) where T
    K = length(Ξ_test)
    n = length(Ξ_test[1][1])
    m = length(cands)
    γt = [_gamma_over_full_from_cand(Ξ_test, cands[t]) for t in 1:m]   # each K×n
    err = zero(T)
    fitted = zeros(T, n)
    for k in 1:K
        p, g = Ξ_test[k]
        fill!(fitted, zero(T))
        for t in 1:m
            fitted .+= dot(p, @view B[t, :]) .* @view γt[t][k, :]
        end
        err += norm(fitted .- (p .* g .+ (delta - 1) .* p), 1)
    end
    return err / K
end

"""
    drop_zero_atoms_ad(cands, γ_ref, B; tol=1e-8) -> (ndrop, keep)

Arrow–Debreu analog of `drop_zero_columns!`: drop atoms whose endowment
collapsed to ~0 (the master set `b_t ≈ 0`, so the atom contributes nothing
to the predictor `Σ_t ⟨p,b_t⟩ γ_t(p)`). An atom is kept iff
`max_l |B[t,l]| > tol`. Prunes the `γ_ref` tensor in place; the caller
re-slices `cands` and `B` by the returned `keep` to stay aligned. With unit
supply, `Σ_t b_t = 1` guarantees `max_t B[t,l] ≥ 1/m > tol` for each good,
so `keep` is never empty. With a free supply scale, δ* ≈ 0 collapses every
endowment; in that case nothing is dropped (the atoms' preferences are
still needed if δ moves away from 0 in a later iteration).
"""
function drop_zero_atoms_ad(cands, γ_ref::Ref, B::AbstractMatrix; tol=1e-8)
    m = length(cands)
    keep = findall(t -> maximum(abs, @view B[t, :]) > tol, 1:m)
    # δ* ≈ 0 (free supply scale): all endowments are ~0 — keep everything.
    isempty(keep) && return 0, collect(1:m)
    ndrop = m - length(keep)
    ndrop > 0 && (γ_ref[] = γ_ref[][keep, :, :])
    return ndrop, keep
end
