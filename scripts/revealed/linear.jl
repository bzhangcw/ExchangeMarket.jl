# -----------------------------------------------------------------------
# Linear-utility market utilities for revealed-preference experiments.
#   - Random linear agent generation
#   - Closed-form bang-per-buck demand
#   - LP recovery of c from price-share samples (eq.plc.mixture.lin.fit)
# -----------------------------------------------------------------------

using LinearAlgebra, Random
using JuMP, MosekTools, Gurobi
import MathOptInterface as MOI
using ExchangeMarket

# Single shared Gurobi env: the license banner ("Set parameter LicenseID
# …, Academic license …") is printed at env creation, so reusing one env
# across all MIP solves emits the banner exactly once (at script load)
# instead of once per CG iteration. We also disable env-level OutputFlag
# so the "Set parameter X to value Y" echo for `set_attribute(md, …)`
# doesn't print at the console for every CG iteration; per-model logging
# is reactivated via `LogFile` on the JuMP model.
const _GRB_ENV = Ref{Gurobi.Env}()
function _gurobi_env()
    if !isassigned(_GRB_ENV)
        _GRB_ENV[] = Gurobi.Env()
        Gurobi.GRBsetintparam(_GRB_ENV[], "OutputFlag", 0)
    end
    return _GRB_ENV[]
end

# Force the env (and thus the license banner) to fire at script load,
# not mid-CG-run. Without this, the banner appears the first time linear
# pricing is actually invoked — which under the multicut→single-cut
# transition is several iterations in, interrupting the iteration table.
_gurobi_env()

"""
    random_linear_agent(n; normalize=true, seed=nothing)

Generate a random linear-utility coefficient vector `c ∈ ℝⁿ_{++}`.
The `LinearAgent` struct in `ExchangeMarket.jl` is a marker only, so we return
the coefficient vector directly (utility `u(x) = ⟨c, x⟩`).

If `normalize=true`, `c` is rescaled to sum to 1 (the bang-per-buck argmax is
invariant under positive rescaling, so this is just for comparability).
"""
function random_linear_agent(n::Int; normalize=true, seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    c = rand(n)
    normalize && (c ./= sum(c))
    return c
end

"""
    solve_linear_demand(c, p, w)

Closed-form bang-per-buck demand for a linear consumer with coefficients `c`
at price `p` and budget `w`:
  j_star = argmax_j c_j / p_j,
  x      = (w / p_{j_star}) * e_{j_star},
  γ      = e_{j_star}  (spending share, a vertex of Δₙ).

Returns `(x, γ, j_star)`.
"""
function solve_linear_demand(c::AbstractVector, p::AbstractVector, w::Real)
    n = length(c)
    @assert length(p) == n "c and p must have the same length"
    j_star = argmax(c ./ p)
    γ = zeros(n)
    γ[j_star] = 1.0
    x = zeros(n)
    x[j_star] = w / p[j_star]
    return x, γ, j_star
end

"""
    fit_linear_lp(pmat, gmat; verbose=false, gauge_index=nothing)

Solve the feasibility LP (eq.plc.mixture.lin.fit) for fitting a single linear
agent to observed price-share samples:

  find y ∈ ℝⁿ
  s.t. y[j_k] - log p[k, j_k] ≥ y[j] - log p[k, j]   ∀ j ∈ [n], k ∈ [K],
       y[gauge_index] = 0                              (gauge: fix one coord)

where `j_k := argmax_l gmat[l, k]` is read off the observed share `gmat[:, k]`
(assumed to be a vertex of Δₙ — true for samples from a real linear agent).

Inputs:
- `pmat::Matrix{Float64}` of shape `(n, K)`: K price columns.
- `gmat::Matrix{Float64}` of shape `(n, K)`: K vertex spending-share columns.
- `gauge_index` defaults to `n` (pin `y[n] = 0`).

Returns `(y, c, feasible, model)` where `c = exp.(y)` is the recovered linear
coefficient (identifiable up to positive scaling; the gauge pins one entry).
"""
function fit_linear_lp(
    pmat::AbstractMatrix, gmat::AbstractMatrix;
    verbose=false, gauge_index::Union{Nothing,Int}=nothing
)
    n, K = size(pmat)
    @assert size(gmat) == (n, K) "pmat and gmat must have the same shape (n, K)"

    md = ExchangeMarket.__generate_empty_jump_model(; verbose=verbose, tol=1e-8)
    @variable(md, y[1:n])
    # Chebyshev slack δ ≥ 0 — the minimum margin by which the observed winner
    # beats every losing alternative. Maximizing δ picks an interior y, so the
    # recovered argmax matches the observed winner strictly (no ties).
    @variable(md, δ >= 0)

    # Read off the observed bang-per-buck winners per price.
    j_winners = [argmax(view(gmat, :, k)) for k in 1:K]

    # Bang-per-buck constraints with slack:
    # y[j_k] - log p[k, j_k] ≥ y[j] - log p[k, j] + δ   for j ≠ j_k.
    for k in 1:K
        jk = j_winners[k]
        log_p_jk = log(pmat[jk, k])
        for j in 1:n
            j == jk && continue  # trivially satisfied for j = jk
            @constraint(md, y[jk] - log_p_jk >= y[j] - log(pmat[j, k]) + δ)
        end
    end

    # Gauge: pin one coordinate (default y[n] = 0).
    gi = isnothing(gauge_index) ? n : gauge_index
    @constraint(md, y[gi] == 0.0)

    # Maximize the slack — finds an interior solution; δ = 0 iff some
    # observed winners are at a knife-edge (unbounded above means cone scaling).
    @objective(md, Max, δ)

    JuMP.optimize!(md)
    status = termination_status(md)
    feasible = status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.FEASIBLE_POINT)
    y_val = feasible ? value.(y) : fill(NaN, n)
    c_val = feasible ? exp.(y_val) : fill(NaN, n)

    return y_val, c_val, feasible, md
end

# -----------------------------------------------------------------------
# Linear pricing as a big-M mixed-integer program (eq.cg.sep.linear)
# -----------------------------------------------------------------------
"""
    solve_pricing_linear(Ξ, u; M=nothing, mip_timelimit=60.0,
                          mip_relgap=1e-4,
                          mip_logfile=joinpath(tempdir(), "gurobi_linear_mip.log"),
                          verbose=false, kwargs...)

Pricing subproblem for the linear function class, solved exactly as the
big-M MIP in eq.cg.sep.linear of the paper:

    max_{y, γ_1,...,γ_K}  Σ_k ⟨u_k, γ_k⟩
    s.t.  γ_k ∈ {0,1}^n,  ⟨1, γ_k⟩ = 1,                       ∀ k
          y_j - log p_{k,j} ≥ y_{j'} - log p_{k,j'}
                              - M(1 - γ_{k,j}),               ∀ j, j', k
          y_n = 0.

The constraints encode that γ_k is the indicator of the bang-per-buck
winner under the linear utility with parameter y (cf. fact.demand.linear)
for the *same* y across all K samples. The big-M is taken as
`M = 2 max_{k,j} |log p_{k,j}|` per fact.nphard.sep.linear.

Solved with **Gurobi**. The MIP is NP-hard in K (fact.nphard.sep.linear)
and dominates CG wall-clock at moderate K, so `mip_timelimit` (Gurobi
`TimeLimit`, seconds) and `mip_relgap` (Gurobi `MIPGap`) bound the
per-pricing call; on time-limit termination the best incumbent is
returned (and is still a valid improving column when its reduced cost
is positive). Gurobi's solver log goes to `mip_logfile` by default (set
to `""` to disable file logging), so the console stays quiet across the
many pricing calls in a CG run; pass `verbose=true` to send the log to
the console instead. The license banner from `Gurobi.Env()` prints once
at first call (a shared env is cached) and not per call.

The returned `params.y` is exactly `log(c)` (the linear coefficient up to
the gauge `y[n] = 0`); `params.σ = Inf` signals the linear regime so
downstream `add_column_to_market!` stores a true `LinearAgent` (ρ = 1.0)
rather than a large-σ CES stand-in.

Returns a NamedTuple compatible with the multi-class dispatcher:
    (γ_new::Matrix{T} of shape (K, n), params=(y=log c, σ=Inf), obj, class=:linear).
"""
function solve_pricing_linear(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    M::Union{T,Nothing}=nothing,
    mip_timelimit::Real=60.0,
    mip_relgap::Real=1e-4,
    mip_logfile::String=joinpath(tempdir(), "gurobi_linear_mip.log"),
    verbose::Bool=false,
    kwargs...) where T
    K = length(Ξ)
    n = length(Ξ[1][1])
    @assert size(u) == (K, n) "u must have shape (K, n)"

    # Log-prices and big-M from observed data.
    log_p = Matrix{T}(undef, K, n)
    for k in 1:K
        log_p[k, :] = log.(Ξ[k][1])
    end
    bigM = isnothing(M) ? T(2 * maximum(abs.(log_p))) : M

    # Gurobi handles the big-M MIP much faster than Mosek; this pricing
    # subproblem is NP-hard in K (fact.nphard.sep.linear), so it dominates
    # CG wall-clock for moderate K and benefits from Gurobi's MIP engine.
    # Reuse the shared env so the license banner doesn't reprint per call.
    md = Model(() -> Gurobi.Optimizer(_gurobi_env()))
    if verbose
        # Default Gurobi behavior: log goes to console.
    else
        # Silence the console, but keep the per-solve log for later
        # inspection by writing it to mip_logfile (appended across calls).
        set_attribute(md, "LogToConsole", 0)
        isempty(mip_logfile) || set_attribute(md, "LogFile", mip_logfile)
    end
    set_attribute(md, "TimeLimit", float(mip_timelimit))
    set_attribute(md, "MIPGap", float(mip_relgap))
    set_attribute(md, "LogFile", "/tmp/gurobi.log")
    @variable(md, y[1:n])
    @variable(md, γ[1:K, 1:n], Bin)

    # Simplex constraint per sample.
    @constraint(md, [k = 1:K], sum(γ[k, :]) == 1)

    # Bang-per-buck big-M inequalities: when γ[k,j] = 1, force
    # y_j - log p_{k,j} ≥ y_{j'} - log p_{k,j'} for all j'; otherwise the
    # -M(1 - γ_{k,j}) slack renders the constraint redundant.
    @constraint(md, [k = 1:K, j = 1:n, jp = 1:n],
        y[j] - log_p[k, j] >=
        y[jp] - log_p[k, jp] - bigM * (1 - γ[k, j]))

    # Gauge: pin one coordinate (matches fit_linear_lp's default).
    @constraint(md, y[n] == 0)

    # Objective: max Σ_k ⟨u_k, γ_k⟩.
    @objective(md, Max, sum(u[k, j] * γ[k, j] for k in 1:K, j in 1:n))

    JuMP.optimize!(md)
    status = termination_status(md)
    # Accept any termination that produced a feasible incumbent — under a
    # time limit Gurobi may stop at TIME_LIMIT with `primal_status =
    # FEASIBLE_POINT`, which is still a usable (sub-optimal) column.
    primal = primal_status(md)
    has_incumbent = primal == MOI.FEASIBLE_POINT || primal == MOI.NEARLY_FEASIBLE_POINT

    y_opt = has_incumbent ? value.(y) : fill(T(NaN), n)
    γ_new = has_incumbent ? Matrix{T}(value.(γ)) : fill(T(NaN), K, n)
    obj_val = has_incumbent ? T(JuMP.objective_value(md)) : T(NaN)

    verbose && println("Linear pricing MIP: obj=$obj_val (M=$bigM, status=$status, primal=$primal)")
    return (γ_new=γ_new, params=(y=y_opt, σ=T(Inf)), obj=obj_val, class=:linear)
end

# -----------------------------------------------------------------------
# Per-sample linear inversion (the linear analogue of solve_pricing_inversion)
# -----------------------------------------------------------------------
"""
    solve_pricing_inversion_linear(Ξ, u)

Produce K linear-class candidates by inverting the bang-per-buck winner
at each sample. Linear has no σ to search over — the structure is fully
fixed by `fact.demand.linear`:

    γ(p_k) = e_{j_k^∗},     j_k^∗ ∈ arg max_j (y_j - log p_{k,j}).

For sample k we pick `j_k^∗ = argmax_j u[k, j]` (the dual's most-rewarded
good at that price) and construct `y` so that `j_k^∗` is the strict
bang-per-buck winner at price `p_k`. Following the proof of
`fact.y.exists.linear`, gauging `y[j_k^∗] = 0` and taking

    y_j = log(p_{k,j} / p_{k,j_k^∗}) - 1,     j ≠ j_k^∗

realizes the desired vertex with unit slack. We then evaluate the
spending share at every sample k' under this y (a one-hot vertex per
row of `γ_new`) and report the pricing objective.

Returns a `Vector{Tuple{Vector{T}, Matrix{T}, T}}` with entries
`(y_opt, γ_new::Matrix of shape (K, n), obj_val)`. The σ slot is fixed
at `Inf` for the linear class and supplied by the caller when forming
`add_column_to_market!(fa, (y=y_opt, σ=T(Inf)), :linear, …)`.
"""
function solve_pricing_inversion_linear(
    Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T}
) where T
    K = length(Ξ)
    n = length(Ξ[1][1])
    log_p = [log.(Ξ[k][1]) for k in 1:K]
    results = Vector{Tuple{Vector{T},Matrix{T},T}}()

    for k in 1:K
        j_star = argmax(view(u, k, :))

        # Construct y so j_star is the bang-per-buck winner at p_k.
        # Gauge y[j_star] = 0; unit-margin Chebyshev-style fill.
        y_opt = zeros(T, n)
        for j in 1:n
            j == j_star && continue
            y_opt[j] = log_p[k][j] - log_p[k][j_star] - one(T)
        end

        # Spending share at every sample under this y: argmax over j.
        γ_new = zeros(T, K, n)
        for k2 in 1:K
            j_star_k2 = argmax(y_opt .- log_p[k2])
            γ_new[k2, j_star_k2] = one(T)
        end

        obj_val = zero(T)
        for k2 in 1:K
            obj_val += dot(view(u, k2, :), view(γ_new, k2, :))
        end
        push!(results, (y_opt, γ_new, obj_val))
    end
    return results
end
