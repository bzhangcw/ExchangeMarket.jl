# Quasi-Newton tâtonnement preconditioned by a CRM surrogate's curvature.
#
# Question: can a CRM-fitted *surrogate* market supply useful second-order
# information (an approximate Hessian) for solving the *real* market's
# equilibrium? We run tâtonnement on the REAL market,
#
#     p  ←  p + α · H⁻¹ ( Σᵢ xᵢ_real(p) − 1 ),
#
# where the excess demand Σx−1 is the real market's, but the curvature H is
# taken from the surrogate. This is Newton's method on the Eisenberg–Gale
# dual potential φ(p): ∇φ = q − Σx (unit supply q = 1) and H = ∇²φ = −∂g/∂p.
#   • H = I        ⇒  plain additive tâtonnement (raise price on excess demand)
#   • H = H_surr   ⇒  quasi-Newton step, curvature borrowed from the surrogate
# If H_surr accelerates convergence over H = I, the surrogate is providing an
# approximate high-order derivative of the real market.
#
# The tâtonnement loop here is self-contained (NOT ExchangeMarket.jl's
# solvers); only the surrogate fit (CG) and the closed-form CES share reuse
# the package / revealed-preference plumbing.
#
# CLI: inherits `parse_args_for_test_real` (run_test.jl's parser). The CG
# surrogate is always method :CG regardless of --methods. Experiment knobs
# are the `QN_*` constants below.

using Revise
using Random, SparseArrays, LinearAlgebra
using ArgParse
using JuMP, MosekTools
using Printf, Serialization
using Plots, LaTeXStrings
import MathOptInterface as MOI

using ExchangeMarket

include("../../tools.jl")
include("../../plots.jl")          # generate_empty, switch_to_pdf — house plot style
include("../androids/plc.jl")    # PLC ground-truth generator + solve_plc_demand
include("../setup.jl")           # build_rep_data, run_one_method, compute_gamma, aggregate_ces_demand, save_run

# Backend must be selected at top level (pgfplotsx from inside a function
# leaves savefig writing empty files — same constraint as run_plot.jl).
switch_to_pdf(; bool_use_html=false)

# -----------------------------------------------------------------------
# Experiment knobs.
# -----------------------------------------------------------------------
const QN_ITERS = 100    # max tâtonnement iterations per run
const QN_TOL = 1e-7   # stop early once the merit ‖diag(p)(q−g)‖_∞ < QN_TOL
const QN_STEP_MAX = 1.0    # initial (max) step; full (quasi-)Newton step when accepted
const QN_LINESEARCH = true   # backtracking line search on ‖diag(p)(q−g)‖_∞ (halve until the merit drops)
const QN_LS_BACKTRACK = 30    # max backtracks per iteration
const QN_RIDGE = 1e-12   # ridge added to the surrogate Hessian for safe inversion
const QN_NORMALIZE = true   # renormalize p onto the simplex each iter (share-form equilibrium price ∈ Δₙ)
const QN_VERBOSE = true   # print a per-iteration progress table (like the CG runner)

# Per-iteration progress table (shared logging.jl helpers, as in cpm.jl).
# Columns: iteration, merit ‖diag(p)(q−g)‖_∞ entering the iter, accepted
# step size α, and number of line-search backtracks.
const QN_TABLE = IterTable(
    ["k", "‖diag(p)(1-g)‖∞", "α", "bt"],
    ["%5d", "%16.6e", "%9.2e", "%4d"],
    [1, 1e-3, 1.0, 3],
)

# -----------------------------------------------------------------------
# CLI + real market + CG surrogate
# -----------------------------------------------------------------------
const cli = parse_args_for_test_real()
cfg = build_run_config(cli)
# qn_tat's real_aggregate_demand and surrogate-Hessian logic assume fixed
# (Fisher) budgets; first-order (AD) wealth is not supported here.
cfg.wealth_function != 0 &&
    error("qn_tat.jl does not support --wealth-function 1 (fixed-budget demand evaluation only).")

@info "configuration" market_type = cfg.market_type cfg.n cfg.m cfg.K cfg.seed cfg.timelimit classes = cfg.allowed_classes qn_iters = QN_ITERS qn_step_max = QN_STEP_MAX linesearch = QN_LINESEARCH

rd = build_rep_data(cfg, 1, cfg.seed)

# Always fit a single CG surrogate (ignore --methods; qn_tat only needs one fa).
const CG_NAME = :CG
spec = first(s for s in method_kwargs if s[1] == CG_NAME)
(_, sep_kind, cg_kwargs) = spec
@info "fitting CG surrogate..."
res = run_one_method(cfg, cli, 1, rd.rep_seed, rd.Ξ_train, rd.Ξ_test, rd.f_real,
    CG_NAME, sep_kind, cg_kwargs)
fa = res.fa
@info "surrogate fitted" androids = fa.m train_obj = res.hist[:primal_obj][end]

# Curvature audit: only finite-σ, positive-weight CES androids carry smooth
# second-order information. A surrogate made of linear (σ=∞, vertex) or
# Leontief (kink) androids is piecewise-linear and has ZERO Hessian almost
# everywhere — the quasi-Newton step then degenerates to (1/ridge)·plain and
# is useless. Warn loudly so the user re-fits with `--classes ces`.
let nce = count(t -> t[1] === :ces && isfinite(fa.storage.ces.σ[t[2]]) && fa.storage.ces.w[t[2]] > 0,
        fa.storage.routing)
    @info "surrogate curvature" smooth_ces_androids = nce total_androids = fa.m
    nce == 0 && @warn "Surrogate has NO smooth (finite-σ CES) androids: it is piecewise-linear, " *
                      "so its Hessian is ~0 and the quasi-Newton step carries no curvature. " *
                      "Re-fit with `--classes ces` (optionally `--sep-ces-sigma-upper`) for a curved surrogate."
end

# -----------------------------------------------------------------------
# Real-market aggregate demand g(p) = Σᵢ xᵢ(p, wᵢ) at an arbitrary price.
#   CES : closed form (reuses aggregate_ces_demand, validate.jl).
#   PLC : one LP per agent per call (solve_plc_demand, androids/plc.jl) —
#         expensive, but the headline intractable case.
# -----------------------------------------------------------------------
function real_aggregate_demand(p::AbstractVector)
    if cfg.market_type === :ces
        return aggregate_ces_demand(rd.f_real, p)
    elseif cfg.market_type === :plc
        n = length(p)
        g = zeros(n)
        for (agent, w) in zip(rd.f_real.agents, rd.f_real.w)
            x_i, _ = solve_plc_demand(agent, p, w)
            g .+= x_i
        end
        return g
    else  # :ges — non-homothetic; share(agent,p,w) is the GES spending share
        n = length(p)
        g = zeros(n)
        for (agent, w) in zip(rd.f_real.agents, rd.f_real.w)
            g .+= w .* share(agent, p, w) ./ p   # x_i = w_i γ_i / p
        end
        return g
    end
end

# -----------------------------------------------------------------------
# Surrogate curvature H(p) = ∇²φ_surr(p) = −∂g_surr/∂p (symmetric PD).
#
# Per CES android i (coefficients c, elasticity σ, weight w), with share
# s = γ(p) (softmax) and P = diag(p):
#     −∂xᵢ/∂p = w·[ σ·P⁻¹(diag(s) − ssᵀ)P⁻¹ + diag(s)P⁻² ]   ⪰ 0,
# summing over androids gives H. Non-CES androids (linear/leontief/ges,
# routed to the `gen` substore) have vertex/kink demands with ~zero curvature
# almost everywhere, so they contribute nothing here; the quasi-Newton
# benefit comes from the CES androids. A small ridge guarantees invertibility
# even if no CES android carries positive weight.
# -----------------------------------------------------------------------
function surrogate_hessian(fa::FisherMarket, p::AbstractVector; ridge::Real=QN_RIDGE)
    n = length(p)
    ws = fa.storage
    H = zeros(n, n)
    pinv = 1.0 ./ p
    for (sub, j) in ws.routing
        sub === :ces || continue           # only CES androids carry smooth curvature
        c = Vector(ws.ces.c[:, j])
        σ = ws.ces.σ[j]
        w = ws.ces.w[j]
        (w > 0 && isfinite(σ)) || continue  # skip zero-weight / linear (σ=∞) androids
        s = compute_gamma(p, c, σ)          # CES spending share at p
        # term1 = σ · P⁻¹(diag(s) − ssᵀ)P⁻¹ ; term2 = diag(s)P⁻²
        @inbounds for a in 1:n, b in 1:n
            cov = (a == b ? s[a] : 0.0) - s[a] * s[b]
            H[a, b] += w * σ * pinv[a] * cov * pinv[b]
        end
        @inbounds for a in 1:n
            H[a, a] += w * s[a] * pinv[a]^2
        end
    end
    H .= 0.5 .* (H .+ H')                    # symmetrize against round-off
    H[diagind(H)] .+= ridge
    return H
end

# -----------------------------------------------------------------------
# Diagonal (Jacobi) preconditioner: keep only the diagonal of H_surr, i.e.
# the per-good own-curvature −∂gⱼ/∂pⱼ summed over CES androids,
#     H_jj = Σᵢ wᵢ · pⱼ⁻² · sᵢⱼ · (1 + σᵢ(1 − sᵢⱼ))  > 0.
# Returned as a `Diagonal` so `Hfn(p) \ z` is an elementwise rescale of the
# excess demand (a cheap, coordinatewise curvature correction between full
# quasi-Newton and plain tâtonnement). Computed directly (no n×n matrix).
# -----------------------------------------------------------------------
function surrogate_hessian_diag(fa::FisherMarket, p::AbstractVector; ridge::Real=QN_RIDGE)
    n = length(p)
    ws = fa.storage
    d = fill(float(ridge), n)
    pinv2 = (1.0 ./ p) .^ 2
    for (sub, j) in ws.routing
        sub === :ces || continue
        σ = ws.ces.σ[j]
        w = ws.ces.w[j]
        (w > 0 && isfinite(σ)) || continue
        s = compute_gamma(p, Vector(ws.ces.c[:, j]), σ)
        @inbounds for a in 1:n
            d[a] += w * pinv2[a] * s[a] * (1 + σ * (1 - s[a]))
        end
    end
    return Diagonal(d)
end

# -----------------------------------------------------------------------
# Self-contained tâtonnement. The search direction is `d = H⁻¹ z` with
# excess demand `z = Σx_real(p) − q`; `Hfn === nothing` ⇒ H = I, i.e.
# `d = z`, plain tâtonnement. A backtracking line search picks the step:
# start at `step_max` and halve until the merit `m(p) = ‖diag(p)(q−g)‖_∞`
# strictly decreases (capped at `QN_LS_BACKTRACK` halvings). The same merit
# and search are used for both H=I and H=H_surr, so the comparison isolates
# the *direction*. `linesearch=false` takes the fixed full step `step_max`.
# `hist[k]` is the merit at the price entering iteration k (hist[1] = start).
# -----------------------------------------------------------------------
function project(p, normalize)
    p = max.(p, 1e-12)                  # keep prices strictly positive
    normalize && (p = p ./ sum(p))      # share-form equilibrium price ∈ Δₙ
    return p
end

function tatonnement(p0::AbstractVector, q::AbstractVector, Hfn;
    step_max::Real=QN_STEP_MAX, iters::Int=QN_ITERS,
    normalize::Bool=QN_NORMALIZE, linesearch::Bool=QN_LINESEARCH,
    verbose::Bool=QN_VERBOSE, label::AbstractString="")
    merit(p) = (g = real_aggregate_demand(p); (norm(p .* (q .- g), Inf), g))
    p = collect(float.(p0))
    f0, g = merit(p)
    hist = Float64[]
    if verbose
        println("\n", "─"^3, " ", label, " ", "─"^max(1, table_width(QN_TABLE) - length(label) - 5))
        print_header(QN_TABLE)
    end
    for k in 1:iters
        push!(hist, f0)
        if f0 < QN_TOL                          # converged: merit below tolerance
            verbose && print_continuation(QN_TABLE,
                @sprintf("converged (‖diag(p)(1-g)‖∞ = %.2e < tol = %.0e) in %d iters", f0, QN_TOL, k - 1))
            break
        end
        z = g .- q                              # excess demand  Σx − q
        d = Hfn === nothing ? z : (Hfn(p) \ z)  # H = I ⇒ d = z (plain)
        α = float(step_max)
        p_next, f_next, g_next = p, f0, g
        accepted = false
        nbt = 0
        for b in 1:(linesearch ? QN_LS_BACKTRACK : 1)
            pt = project(p .+ α .* d, normalize)
            ft, gt = merit(pt)
            if isfinite(ft) && (!linesearch || ft < f0)
                p_next, f_next, g_next = pt, ft, gt
                accepted = true
                nbt = b - 1
                break
            end
            α /= 2
        end
        verbose && print_row(QN_TABLE, Any[k, f0, accepted ? α : NaN, accepted ? nbt : nothing])
        if !accepted                            # no decreasing step found
            verbose && print_continuation(QN_TABLE, "stalled: line search found no improving step")
            linesearch && k == 1 && @warn "line search found no decreasing step at iter 1; check curvature/normalization."
            break                               # stagnated; stop (hist already records f0)
        end
        p, f0, g = p_next, f_next, g_next
    end
    return p, hist
end

# -----------------------------------------------------------------------
# Run both schemes from the same start price.
# -----------------------------------------------------------------------
n = cfg.n
p0 = fill(1.0 / n, n)
q = cfg.market_type === :ces ? Vector(rd.f_real.q) : ones(n)

_, hist_plain = tatonnement(p0, q, nothing; label="plain  (H = I)")
p_diag, hist_diag = tatonnement(p0, q, p -> surrogate_hessian_diag(fa, p); label="diagonal  (H = diag H_surr)")
p_qn, hist_qn = tatonnement(p0, q, p -> surrogate_hessian(fa, p); label="quasi-Newton  (H = H_surr)")

function _report(name, h)
    status = h[end] < QN_TOL ? "(converged)" :
             length(h) >= QN_ITERS ? "(max iters)" :
             "(stalled)"     # line search found no improving step
    @printf("%-8s : iters=%3d / %d  %-12s  start ‖z‖∞=%.3e  final ‖z‖∞=%.3e\n",
        name, length(h), QN_ITERS, status, h[1], h[end])
end
@printf("\n=== qn_tat done (tol=%.0e, max_iters=%d) ===\n", QN_TOL, QN_ITERS)
_report("plain", hist_plain)
_report("diagonal", hist_diag)
_report("qn", hist_qn)

# -----------------------------------------------------------------------
# Serialize + plot. Style mirrors revealed/run_plot.jl (pgfplotsx, log-y,
# dense ticks, large fonts) so the PDF slots into the paper-figure pipeline.
# -----------------------------------------------------------------------
payload = (
    fa=deepcopy(fa),
    hist_plain=hist_plain,
    hist_diag=hist_diag,
    hist_qn=hist_qn,
    p0=p0, p_diag=p_diag, p_qn=p_qn, q=q,
    market_type=cfg.market_type,
    n=cfg.n, m=cfg.m, K=cfg.K, seed=cfg.seed,
    qn_iters=QN_ITERS, qn_step_max=QN_STEP_MAX, qn_linesearch=QN_LINESEARCH,
    qn_ridge=QN_RIDGE, qn_normalize=QN_NORMALIZE,
    cli=cli,
)

out_path = if cli["no_data_file"]
    ""
elseif !isempty(cli["data_file"])
    abspath(cli["data_file"])
else
    joinpath(cfg.out_dir, "run_qn_tat.jls")
end
isempty(out_path) || save_run(out_path, payload)

let
    floor_ = 1e-12
    # Pad an early-stopped (converged) trajectory to the full axis, holding
    # its last value, so both curves span 1:QN_ITERS.
    padlast(v, L) = length(v) >= L ? v[1:L] : vcat(v, fill(v[end], L - length(v)))
    yp = max.(padlast(hist_plain, QN_ITERS), floor_)
    yd = max.(padlast(hist_diag, QN_ITERS), floor_)
    yq = max.(padlast(hist_qn, QN_ITERS), floor_)

    fig = generate_empty(; shape=:wide)
    plot!(fig,
        ylabel=L"\max_j\,|p_j(1-g_j)|",
        xlabel=L"\textrm{iteration}",
        legendbackgroundcolor=RGBA(1.0, 1.0, 1.0, 0.8),
        yscale=:log10,
        xtickfont=font(18),
        ytickfont=font(18),
        size=(600, 500),
        left_margin=12Plots.mm,
        bottom_margin=8Plots.mm,
        legendfontsize=18,
    )
    plot!(fig, 1:length(yp), yp; label=L"\textrm{plain}~(H=I)", linewidth=3, color=1)
    plot!(fig, 1:length(yd), yd; label=L"\textrm{diagonal}~(H=\mathrm{diag})", linewidth=3, color=3)
    plot!(fig, 1:length(yq), yq; label=L"\textrm{quasi-Newton}", linewidth=3, color=2)

    plot_pdf = isempty(out_path) ?
               joinpath(cfg.out_dir, "run_qn_tat.pdf") :
               string(first(splitext(out_path)), ".pdf")
    savefig(fig, plot_pdf)
    @info "saved convergence plot" plot_pdf
end
