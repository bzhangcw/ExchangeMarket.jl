# -----------------------------------------------------------------------
# Generalized-elasticity-of-substitution (GES) market utilities.
#   - Random GES agent generation
#   - Closed-form share via Newton on the scalar budget root (lem.ges.budget.root)
#   - CG separation oracle: log-variable NLP (eq.ges.pricing.log) via MadNLP
# Non-homothetic class — γ depends on the pinning wealth w through the
# implicit KKT multiplier λ that solves b(λ; c, r) = w (eq.ges.budget.scalar).
# -----------------------------------------------------------------------

using LinearAlgebra, Random, Statistics
using JuMP, MadNLP
import MathOptInterface as MOI
using ArgParse
using ExchangeMarket

# Non-homothetic: γ depends on the pinning wealth w through the implicit λ.
is_homothetic(::Val{:ges}) = false

# ---- CLI surface --------------------------------------------------------
# Defaults for the GES NLP search box. The σ-upper / y-bound numerics mirror
# CES (cf. _CES_SIGMA_UPPER_DEFAULT = 30 in androids/ces.jl,
# LOWER_Y_BOUND = -100, UPPER_Y_BOUND = 100); the σ-lower default differs
# because GES requires σ > 0 strictly (polynomial-utility concavity:
# r_j ∈ (0, 1) ⇔ σ_j > 0), whereas CES allows σ > -1 (Leontief boundary).
# Note: the recovered c_j = exp(y_j / (σ_j + 1)) can grow rapidly when
# σ_j is small AND |y_j| is large; if test-time `ges_share` then misbehaves,
# lower --ges-sigma-upper (e.g. to 5) or tighten --ges-y-{lower,upper}.
const _GES_SIGMA_LOWER_DEFAULT = 1e-2
const _GES_SIGMA_UPPER_DEFAULT = 10.0
const _GES_Y_LOWER_DEFAULT = -10.0
const _GES_Y_UPPER_DEFAULT = 10.0

"""
    register_cli_ges!(s::ArgParseSettings)

Adds the "Separation: GES" arg group: `--ges-sigma-lower`, `--ges-sigma-upper`,
`--ges-y-lower`, `--ges-y-upper`. These set the GES NLP search box; their
defaults mirror CES (`--ces-sigma-upper`, `LOWER/UPPER_Y_BOUND`) except the
σ lower bound is strictly positive (GES requires σ > 0).
"""
function register_cli_ges!(s::ArgParseSettings)
    add_arg_group!(s, "Separation: GES")
    @add_arg_table! s begin
        "--ges-sigma-lower"
        help = "Strict-positive lower bound on σ_j for GES (polynomial-utility concavity needs σ > 0). Default $(_GES_SIGMA_LOWER_DEFAULT)."
        arg_type = Float64
        default = _GES_SIGMA_LOWER_DEFAULT
        "--ges-sigma-upper"
        help = "Upper bound on σ_j (CES-matching default $(_GES_SIGMA_UPPER_DEFAULT)). Lower (e.g. 5) keeps the recovered c_j = exp(y_j/(σ_j+1)) numerically tame."
        arg_type = Float64
        default = _GES_SIGMA_UPPER_DEFAULT
        "--ges-y-lower"
        help = "Lower bound on y_j (CES-matching default $(_GES_Y_LOWER_DEFAULT))."
        arg_type = Float64
        default = _GES_Y_LOWER_DEFAULT
        "--ges-y-upper"
        help = "Upper bound on y_j (CES-matching default $(_GES_Y_UPPER_DEFAULT))."
        arg_type = Float64
        default = _GES_Y_UPPER_DEFAULT
    end
    return s
end

"""
    apply_cli_ges!(local_extra::Dict, cli)

Forward GES NLP bounds into the runner kwargs.
"""
function apply_cli_ges!(local_extra::Dict, cli)
    local_extra[:ges_sigma_lower] = cli["ges_sigma_lower"]
    local_extra[:ges_sigma_upper] = cli["ges_sigma_upper"]
    local_extra[:ges_y_lower] = cli["ges_y_lower"]
    local_extra[:ges_y_upper] = cli["ges_y_upper"]
    return local_extra
end

"""
    ges_config_summary(kwargs::Dict; nonh_w::Real) -> String

Banner line for the GES class — shows the NLP search box plus the pinning
wealth used by the master. The recovered c_j scales as exp(y_j/(σ_j+1)),
so narrow [σ_lo, σ_hi] / [y_lo, y_hi] is the main numerical-safety knob.
"""
function ges_config_summary(kwargs::Dict; nonh_w::Real)
    σ_lo = get(kwargs, :ges_sigma_lower, _GES_SIGMA_LOWER_DEFAULT)
    σ_hi = get(kwargs, :ges_sigma_upper, _GES_SIGMA_UPPER_DEFAULT)
    y_lo = get(kwargs, :ges_y_lower, _GES_Y_LOWER_DEFAULT)
    y_hi = get(kwargs, :ges_y_upper, _GES_Y_UPPER_DEFAULT)
    return @sprintf("NLP, σ ∈ [%g, %g], y ∈ [%g, %g] (fixed w₀ = %g)",
        σ_lo, σ_hi, y_lo, y_hi, nonh_w)
end

# -----------------------------------------------------------------------
# Ground-truth generator
# -----------------------------------------------------------------------
"""
    random_ges_agent(n; seed=nothing)

Generate a random GESAgent in `n` goods.
  u(x) = Σ_j c_j x_j^{r_j},  c_j ∈ Uniform(0.5, 2),  r_j ∈ Uniform(0.3, 0.7).

The r-range stays comfortably interior to (0, 1) so the elasticity
σ_j = r_j/(1−r_j) ∈ (0.43, 2.33) avoids both the linear (r→1, σ→∞) and
the perfect-complement (r→0, σ→0) boundaries.
"""
function random_ges_agent(n::Int; seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    c = 0.5 .+ 1.5 .* rand(n)
    r = 0.3 .+ 0.4 .* rand(n)
    return GESAgent(n, c, r)
end

# -----------------------------------------------------------------------
# Closed-form share at fixed (c, r, p, w)
# -----------------------------------------------------------------------
"""
    ges_budget(c, r, p, λ) -> Float64

Implied-wealth function b(λ; c, r) = Σ_j (c_j r_j)^{σ_j+1} p_j^{-σ_j} λ^{-(σ_j+1)}
from eq.ges.budget.scalar, evaluated at multiplier λ. Strictly convex,
strictly decreasing, log-convex on (0, ∞).
"""
function ges_budget(c::AbstractVector, r::AbstractVector, p::AbstractVector, λ::Real)
    n = length(c)
    s = 0.0
    @inbounds for j in 1:n
        σj = r[j] / (1 - r[j])
        s += (c[j] * r[j])^(σj + 1) * p[j]^(-σj) * λ^(-(σj + 1))
    end
    return s
end

# Derivative b'(λ) = −Σ_j (σ_j+1) A_j λ^{-(σ_j+2)}; strictly negative.
function _ges_budget_deriv(c::AbstractVector, r::AbstractVector, p::AbstractVector, λ::Real)
    n = length(c)
    s = 0.0
    @inbounds for j in 1:n
        σj = r[j] / (1 - r[j])
        s -= (σj + 1) * (c[j] * r[j])^(σj + 1) * p[j]^(-σj) * λ^(-(σj + 2))
    end
    return s
end

"""
    ges_budget_root(c, r, p, w; max_iter=50, tol=1e-12) -> λ

Newton on the scalar budget root b(λ; c, r) = w (lem.ges.budget.root).
Quadratic local convergence from any λ_0 with b(λ_0) ≥ w (tangent below
graph, strictly decreasing → monotone increasing iterates). The initial
guess uses a median-σ closed form:

  σ_mid := median(σ),  C := Σ_j A_j,  λ_0 := (C/w)^{1/(σ_mid + 1)}.

When `σ_j ≡ σ` (CES collapse), this is exact and Newton terminates in one
iteration. Otherwise it's a one-shot approximation; the safeguard halves
λ inward whenever a Newton step would cross zero.
"""
function ges_budget_root(c::AbstractVector, r::AbstractVector,
    p::AbstractVector, w::Real;
    max_iter::Int=50, tol::Real=1e-12)
    n = length(c)
    σ = [r[j] / (1 - r[j]) for j in 1:n]
    A = [(c[j] * r[j])^(σ[j] + 1) * p[j]^(-σ[j]) for j in 1:n]
    σ_mid = median(σ)
    λ = (sum(A) / w)^(1 / (σ_mid + 1))

    for _ in 1:max_iter
        b = ges_budget(c, r, p, λ)
        if abs(b - w) ≤ tol * max(w, 1.0)
            return λ
        end
        bp = _ges_budget_deriv(c, r, p, λ)
        λ_new = λ - (b - w) / bp
        # Safeguard: stay in (0, ∞); halve toward zero if Newton overshoots.
        λ = λ_new > 0 ? λ_new : λ / 2
    end
    return λ
end

"""
    ges_share(c, r, p, w) -> γ ∈ Δ_n

Closed-form GES spending share γ_j = p_j x_j(λ)/w with x_j(λ) from
eq.ges.demand.lambda and λ solving eq.ges.budget.scalar at the supplied
budget w. Requires solving a smooth scalar root for λ; otherwise everything
is elementary.
"""
function ges_share(c::AbstractVector, r::AbstractVector,
    p::AbstractVector, w::Real)
    n = length(c)
    λ = ges_budget_root(c, r, p, w)
    γ = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        σj = r[j] / (1 - r[j])
        # x_j = (c_j r_j / (λ p_j))^{σ_j+1}; γ_j = p_j x_j / w.
        γ[j] = p[j] * (c[j] * r[j] / (λ * p[j]))^(σj + 1) / w
    end
    return γ
end

"""
    ges_share_by_opt(c, r, p, w; verbose=false, timelimit=nothing) -> γ ∈ Δ_n

Closed-form alternative to `ges_share`: solve the UMP directly as a smooth
concave-maximization in `x`,

    max_{x ≥ ε}  Σ_j c_j x_j^{r_j}   s.t.   ⟨p, x⟩ = w,

via JuMP + MadNLP, then return γ_j = p_j x_j / w. This stays in `x`-space
and therefore never evaluates the budget-function form
`b(λ) = Σ (c_j r_j)^{σ_j+1} p_j^{-σ_j} λ^{-(σ_j+1)}` (which overflows at
extreme `c_j` / σ_j combinations and is the failure mode of `ges_share`'s
Newton solver). With r_j ∈ (0, 1) the utility is strictly concave on
ℝ^n_{++}, so the NLP has a unique interior optimum, and MadNLP handles
the boundary slopes (r_j c_j x_j^{r_j-1} → ∞ as x_j → 0) via the
strict-positive lower bound `ε`.

Use this when `ges_share` returns garbage (e.g., as a fallback inside
`share(::GESAgent, p, w)`, or as the reference share in the GES-separator
sanity check).
"""
function ges_share_by_opt(c::AbstractVector, r::AbstractVector,
    p::AbstractVector, w::Real;
    verbose::Bool=false,
    timelimit::Union{Real,Nothing}=nothing,
    ε::Real=1e-12)
    n = length(c)
    @assert length(r) == n "c and r must have the same length"
    @assert length(p) == n "c and p must have the same length"
    @assert all(r .> 0) && all(r .< 1) "GES requires r_j ∈ (0, 1)"
    @assert all(c .> 0) "GES requires c_j > 0"
    @assert w > 0

    model = Model(MadNLP.Optimizer)
    if !verbose
        set_attribute(model, "print_level", MadNLP.ERROR)
    end
    if !isnothing(timelimit) && timelimit > 0
        set_attribute(model, "max_wall_time", Float64(timelimit))
    end

    # Initial guess: uniform x_j = w/(n p_j) (interior, budget-feasible).
    @variable(model, x[j=1:n] >= ε, start = w / (n * p[j]))

    # u(x) = Σ_j c_j x_j^{r_j} — smooth concave in the strictly-interior
    # domain x > 0; r_j ∈ (0, 1) keeps each term concave.
    @objective(model, Max, sum(c[j] * x[j]^r[j] for j in 1:n))

    # Single linear budget constraint.
    @constraint(model, sum(p[j] * x[j] for j in 1:n) == w)

    JuMP.optimize!(model)
    x_opt = value.(x)
    return (p .* x_opt) ./ w
end

"""
    share(agent::GESAgent, p, w) -> γ

Per-class share dispatch.
Delegates to `ges_share_by_opt(agent.c, agent.r, p, w)` — the
optimization-based share evaluator.
"""
share(agent::GESAgent, p::AbstractVector, w::Real) =
    ges_share_by_opt(agent.c, agent.r, p, w)

# -----------------------------------------------------------------------
# Separation oracle — (y, σ) NLP via MadNLP (eq.ges.pricing)
# -----------------------------------------------------------------------
"""
    solve_separation_ges(Ξ, u, w; init=nothing, σ_max=30.0,
                         verbose=false, timelimit=nothing, kwargs...)

CG separation oracle for the GES function class at fixed pinning wealth
`w`. Solves the NLP (eq.ges.pricing) in the CES-compatible
parameterization

    max_{y ∈ ℝⁿ, σ ∈ (0, σ_max]ⁿ, λ ∈ ℝᴷ_++}   Σ_k Σ_j u_{k,j} γ_{k,j}(y, σ; λ_k)
    s.t.                                         Σ_j γ_{k,j} = 1,  ∀ k,

with γ_{k,j} = (1/w) · exp(y_j − σ_j log p_{k,j}) · (σ_j/((σ_j+1) λ_k))^{σ_j+1}.
By lem.ges.simplex.budget the K simplex equalities ARE the K budget
bindings b(λ_k) = w, so λ is pinned implicitly.

Recovery (eq.ges.yparam): σ_j is stored directly; r_j = σ_j/(σ_j+1);
c_j = exp(y_j / (σ_j+1)). The CES limit σ_j ≡ σ recovers a CES atom with
the same (y, σ) form.

`init::NamedTuple{(:c, :r, :λ)}` overrides the generic Cobb-Douglas
initial guess (σ = 1, y = 0 ⇒ c_j r_j = 1; λ_k from the C-D closed form).

Returns a NamedTuple compatible with the per-class separation oracle:
    (γ_new::Matrix{T} of shape (K, n), params=(c, r, w), obj, class=:ges).
"""
function solve_separation_ges(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T},
    w::Real;
    init::Union{Nothing,NamedTuple}=nothing,
    # NLP search box — all four bounds are CLI knobs (see `register_cli_ges!`).
    # Defaults mirror CES (σ-upper = 30, y ∈ [-100, 100]) except σ-lower is
    # strictly positive (polynomial-utility concavity).
    ges_sigma_lower::Real=_GES_SIGMA_LOWER_DEFAULT,
    ges_sigma_upper::Real=_GES_SIGMA_UPPER_DEFAULT,
    ges_y_lower::Real=_GES_Y_LOWER_DEFAULT,
    ges_y_upper::Real=_GES_Y_UPPER_DEFAULT,
    verbose::Bool=false,
    timelimit::Union{Real,Nothing}=nothing,
    kwargs...) where T
    K = length(Ξ)
    n = length(Ξ[1][1])
    @assert size(u) == (K, n) "u must have shape (K, n)"
    @assert w > 0 "pinning wealth w must be positive"
    @assert ges_sigma_lower > 0 "ges_sigma_lower must be positive (GES requires σ > 0)"
    @assert ges_sigma_upper > ges_sigma_lower "ges_sigma_upper must exceed ges_sigma_lower"
    @assert ges_y_upper > ges_y_lower "ges_y_upper must exceed ges_y_lower"

    log_p = [log.(Ξ[k][1]) for k in 1:K]
    w_f = Float64(w)
    σ_lo, σ_hi = Float64(ges_sigma_lower), Float64(ges_sigma_upper)
    y_lo, y_hi = Float64(ges_y_lower), Float64(ges_y_upper)
    ϵ_λ = 1e-6                                    # strict-positive λ floor

    model = Model(MadNLP.Optimizer)
    if !verbose
        set_attribute(model, "print_level", MadNLP.ERROR)
    end
    if !isnothing(timelimit) && timelimit > 0
        set_attribute(model, "max_wall_time", Float64(timelimit))
    end

    # CES-compatible (y, σ, λ) parameterization (eq.ges.yparam).
    # y_j ∈ [y_lo, y_hi]      — CES intercept (y_j = (σ_j+1) log c_j)
    # σ_j ∈ [σ_lo, σ_hi]      — per-good elasticity (eq.ges.pricing)
    # λ_k > 0                 — per-sample KKT multiplier (pinned by simplex)
    @variable(model, y_lo <= y[1:n] <= y_hi)
    @variable(model, σ_lo <= σ[1:n] <= σ_hi)
    @variable(model, λ[1:K] >= ϵ_λ)

    # γ as a bounded NLP variable rather than a free @expression. The
    # formula γ_{k,j} = (1/w) · exp(y_j − σ_j log p_{k,j}) ·
    # (σ_j/((σ_j+1) λ_k))^{σ_j+1} is enforced via an equality constraint.
    # The per-entry bounds γ_{k,j} ∈ [0, 1] are *redundant* with the
    # simplex equality Σ_j γ_{k,j} = 1 at any TRUE optimum (positive
    # formula + sum-to-one ⇒ each entry ≤ 1), but they cut off the
    # spurious MadNLP iterates where the simplex residual hasn't tightened
    # yet and individual entries float outside [0, 1] (these caused the
    # master to blow up by orders of magnitude when accepted). When MadNLP
    # can't satisfy both the formula equality and the bounds, it
    # terminates infeasible and our sentinel returns `nothing`.
    @variable(model, 0.0 <= γ[1:K, 1:n] <= 1.0)
    @constraint(model, γ_def[k=1:K, j=1:n],
        γ[k, j] == (1.0 / w_f) *
                   exp(y[j] - σ[j] * log_p[k][j]) *
                   (σ[j] / ((σ[j] + 1) * λ[k]))^(σ[j] + 1))

    # Simplex constraint per sample = budget binding (lem.ges.simplex.budget).
    @constraint(model, simplex[k=1:K], sum(γ[k, j] for j in 1:n) == 1)

    # Objective — separation reduced cost (without the −μ constant, which
    # the caller subtracts via `reduced_cost` in separation.jl).
    @objective(model, Max, sum(u[k, j] * γ[k, j] for k in 1:K, j in 1:n))

    # Initial guess.
    if isnothing(init)
        # Generic Cobb-Douglas: σ_j = 1 (clamped into the user box), y_j = 0.
        # Under σ ≡ 1, y ≡ 0, the budget equation b(λ_k) = Σ_j p_{k,j}^{-1} / (4λ_k²) = w
        # gives λ_k = √(Σ_j 1/p_{k,j} / (4w)) = √(Σ_j 1/p_{k,j}) / (2√w).
        σ_init = clamp(1.0, σ_lo, σ_hi)
        y_init = clamp(0.0, y_lo, y_hi)
        for j in 1:n
            set_start_value(y[j], y_init)
            set_start_value(σ[j], σ_init)
        end
        for k in 1:K
            λ0 = sqrt(sum(1.0 / Ξ[k][1][j] for j in 1:n)) / (2 * sqrt(w_f))
            set_start_value(λ[k], max(λ0, ϵ_λ))
        end
    else
        @assert haskey(init, :c) && haskey(init, :r) && haskey(init, :λ) \
                                                        "init must have fields (c, r, λ)"
        for j in 1:n
            σ_init = init.r[j] / (1 - init.r[j])
            set_start_value(σ[j], clamp(σ_init, σ_lo, σ_hi))
            y_init = (σ_init + 1) * log(init.c[j])
            set_start_value(y[j], clamp(y_init, y_lo, y_hi))
        end
        for k in 1:K
            set_start_value(λ[k], max(init.λ[k], ϵ_λ))
        end
    end
    # Seed γ start values from the (y, σ, λ) start by evaluating the
    # NLP's own defining formula at the start point, then clamping into
    # the variable bounds. Without this MadNLP starts γ at 0 (the
    # default for [0, 1]-bounded variables), which is far from the
    # simplex and degrades IPM convergence.
    for k in 1:K, j in 1:n
        σj = start_value(σ[j])
        yj = start_value(y[j])
        λk = start_value(λ[k])
        g0 = (1.0 / w_f) *
             exp(yj - σj * log_p[k][j]) *
             (σj / ((σj + 1) * λk))^(σj + 1)
        set_start_value(γ[k, j], clamp(g0, 1e-8, 1.0 - 1e-8))
    end

    JuMP.optimize!(model)
    status = termination_status(model)
    primal = primal_status(model)

    # Return `nothing` whenever this separation call should be skipped
    # (MadNLP failed to produce a usable incumbent, or the recovered
    # (c, r) doesn't reproduce the NLP γ at test prices). `solve_separation`
    # handles `nothing` by continuing to the next class — CES runs in the
    # same CG round, so if GES is unusable on this iteration the runner
    # falls back to CES's candidate automatically.
    has_incumbent = primal == MOI.FEASIBLE_POINT || primal == MOI.NEARLY_FEASIBLE_POINT
    if !has_incumbent
        @warn "GES separation: MadNLP returned no usable incumbent (status=$status, primal=$primal); skipping this iteration's GES candidate"
        return nothing
    end

    obj_val = T(JuMP.objective_value(model))

    # Recover (c, r) from (y, σ) via eq.ges.yparam.
    σ_opt = value.(σ)
    r_opt = σ_opt ./ (σ_opt .+ 1)              # r_j = σ_j / (σ_j + 1)
    c_opt = exp.(value.(y) ./ (σ_opt .+ 1))    # c_j = exp(y_j / (σ_j + 1))

    # Read γ directly from the NLP (matches the simplex constraint to solver
    # tolerance). Recomputing via `ges_share(c_opt, r_opt, p_k, w)` would drift
    # away from this at extreme σ_j / c_j (where (c_j r_j)^{σ_j+1} ≈ 10^{30+}
    # makes Newton on b(λ) = w numerically delicate). The training-time
    # column is then self-consistent for the master; the question is
    # whether test-time `ges_share` agrees — checked below.
    γ_new = Matrix{T}(undef, K, n)
    for k in 1:K, j in 1:n
        γ_new[k, j] = T(value(γ[k, j]))
    end

    # Per-entry bound check: MadNLP's `tol_constr_viol` only enforces
    # `Σ_j γ_{k,j} = 1` to ~1e-4; individual γ_{k,j} can drift far outside
    # [0, 1] while still summing to 1 (e.g., γ = [10, -9, …] satisfies the
    # simplex but corresponds to no real share). Dropping such a column
    # into the LP master with w_t pinned at nonh_w then forces the slack
    # `s_k` to absorb the out-of-simplex mass, accumulating to a master-obj
    # blow-up of orders of magnitude. Reject if any γ_{k,j} falls outside
    # [-γ_slack, 1 + γ_slack], where γ_slack is a couple of NLP tolerances.
    let γ_lo = T(-1e-3), γ_hi = T(1) + T(1e-3)
        γmin, γmax = extrema(γ_new)
        if !all(isfinite, γ_new) || γmin < γ_lo || γmax > γ_hi
            @warn "GES separation: NLP γ has entries outside [γ_lo, γ_hi] (min=$γmin, max=$γmax) — sums-to-1 only at the row level, but individual values are unphysical. Skipping (MadNLP likely stopped at ITERATION_LIMIT with a loose feasibility tolerance; try tightening --ges-sigma-upper / --ges-y-{lower,upper})." σ_range = (minimum(σ_opt), maximum(σ_opt)) c_range = (minimum(c_opt), maximum(c_opt))
            return nothing
        end
    end

    # Sanity check: the recovered (c, r) must reproduce the NLP-stored γ
    # via the test-time evaluator `ges_share`, AND have c magnitudes that
    # won't overflow Newton on b(λ) = w at unseen prices. We check ALL
    # training samples (sample-1-only isn't enough: an atom can drift at
    # some k > 1 even if k = 1 looks fine), and we cap max(c) so that
    # `share(::GESAgent, p, w)` at TEST prices doesn't blow up later in
    # evaluate_test_error (a recovered c_j ≈ 10^{30+} makes
    # (c_j r_j)^{σ_j+1} overflow Float64, so Newton diverges and the
    # test-error metric becomes meaningless).
    let drift_tol = 1e-3, c_max = 1e10
        if maximum(c_opt) > c_max || !all(isfinite, c_opt)
            @warn "GES separation: recovered max(c) = $(maximum(c_opt)) > c_max=$c_max; skipping this iteration's GES candidate (recovery in numerically delicate regime — try tightening --ges-y-{lower,upper} or raising --ges-sigma-lower)." σ_range = (minimum(σ_opt), maximum(σ_opt)) c_range = (minimum(c_opt), maximum(c_opt))
            return nothing
        end
        max_drift = 0.0
        worst_k = 0
        for k in 1:K
            γ_recomp = ges_share(c_opt, r_opt, Ξ[k][1], w_f)
            d = maximum(abs.(γ_new[k, :] .- γ_recomp))
            if !isfinite(d) || d > max_drift
                max_drift = d
                worst_k = k
            end
            if !isfinite(d) || d > drift_tol
                break   # one bad sample is enough; no need to keep checking
            end
        end
        if !isfinite(max_drift) || max_drift > drift_tol
            @warn "GES separation: NLP γ vs. ges_share γ drift = $max_drift at sample $worst_k > tol=$drift_tol; recovered (c, r) in a numerically delicate regime — skipping this iteration's GES candidate (try lowering --ges-sigma-upper or tightening --ges-y-{lower,upper})." σ_range = (minimum(σ_opt), maximum(σ_opt)) c_range = (minimum(c_opt), maximum(c_opt))
            return nothing
        end
    end

    verbose && println("GES separation: status=$status, primal=$primal, obj=$obj_val")
    return (γ_new=γ_new,
        params=(c=Vector{Float64}(c_opt), r=Vector{Float64}(r_opt), w=w_f),
        obj=obj_val, class=:ges)
end
