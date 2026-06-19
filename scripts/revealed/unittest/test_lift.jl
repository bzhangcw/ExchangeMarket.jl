# =======================================================================
# Forward and backward tests for the money-lift of a Fisher CES market.
#
# Step-0 lift (overleaf/read-econ/rational-literature-iii.tex, `lem.lift` /
# `prop.lift.equilibrium`): adjoin money as an (n+1)-th commodity so that
# scale-dependent Fisher demand becomes a homogeneous-degree-0,
# Walras-law-satisfying excess demand on the lifted simplex
# 𝛥̄ = {(p,π) : ⟨1,p⟩ + π = 1}. The bijection (p,π) ↦ q = p/π relates the
# lifted price to the original (unnormalized) price.
#
# forward_test  — construct the lifted equilibrium A′=(p*,π*) FROM the
#   original equilibrium A=q* via the bijection, and check the lifted
#   excess demand vanishes there (cheap, closed-form lifted demand).
#
# backward_test — the real round trip: build the lifted market as a genuine
#   (n+1)-good, (m+1)-agent Arrow–Debreu economy, solve it FROM SCRATCH with
#   `afscaled_newton`, project q* = p*/π* back, and verify q* clears the
#   original Fisher market (and recovers its equilibrium scale).
#
# The lifted economy (faithful realization of d̄(p,π)=(d(q), money)):
#   • buyers i=1..m — endowment w_i units of money (good n+1), CES utility
#     over goods 1..n only (zero coefficient on money). Budget π w_i ⇒ goods
#     demand π w_i γ_i(p)/p = w_i γ_i(q)/q = original Fisher demand.
#   • rentier (agent m+1) — endows one unit of each good (the unit supply,
#     exogenous in Fisher), wants only money. Sells goods, holds money.
# Supply (1,…,1, Σw_i): goods clear ⟺ d(q)=1, money clears ⟺ ⟨1,q⟩=Σw_i.
#
# Self-contained for the forward path (only ExchangeMarket). The backward
# path additionally pulls the AD potential-reduction solver from
# scripts/arrow (JuMP + COPT). The CES helpers below mirror the canonical
# definitions in setup.jl (`compute_gamma`) and validate.jl
# (`solve_ces_equilibrium`, `aggregate_ces_demand`), inlined to avoid a
# solver-license load dependency for the forward test.
#
# A (the Fisher market) and A′ (its lift) are built ONCE by
# build_lift_markets; forward_test and backward_test then run on the same
# (fa, ad) pair. run_lift_tests wires it all together.
#
# Usage:
#   cd /Users/brent/workspace/ExchangeMarket.jl/scripts
#   julia --project=. revealed/test_lift.jl
#   julia> run_lift_tests(n=5, m=8, budget_total=2.0, seed=7)
#   # or drive the shared data yourself:
#   julia> mk = build_lift_markets(n=5, m=8, budget_total=2.0, seed=7);
#   julia> forward_test(mk.fa, mk.ad); backward_test(mk.fa, mk.ad)
# =======================================================================

using Random
using LinearAlgebra
using Printf
using JuMP                # arrow_debreu.jl uses @variable/@constraint (Mosek SOCP)
using ExchangeMarket

# AD affine-scaled Newton equilibrium solver (self-contained; no arrow dep).
include("../arrow_debreu.jl")   # afscaled_newton, afscaled_newton_equilibrium

# -----------------------------------------------------------------------
# CES helpers (mirrors of setup.jl / validate.jl; see header).
# -----------------------------------------------------------------------

# γ_j = (c_j^{1+σ} p_j^{-σ}) / Σ_ℓ(c_ℓ^{1+σ} p_ℓ^{-σ})   (setup.jl)
function compute_gamma(p::AbstractVector, c::AbstractVector, σ::Real)
    if isinf(σ) && σ > 0                       # linear regime: bang-per-buck vertex
        γ = zeros(eltype(p), length(c))
        γ[argmax(c ./ p)] = one(eltype(p))
        return γ
    end
    z = (1 + σ) .* log.(c) .- σ .* log.(p)     # log-space softmax for numerical safety
    ez = exp.(z .- maximum(z))
    return ez ./ sum(ez)
end

# Aggregate CES demand g(p) = Σ_i w_i γ_i(p) ./ p. Valid at any positive
# (unnormalized) price q since γ is homogeneous of degree 0. (validate.jl)
function aggregate_ces_demand(fa::FisherMarket, p::AbstractVector)
    g = zeros(length(p))
    for i in 1:fa.m
        γ_i = compute_gamma(p, Vector(fa.c[:, i]), fa.σ[i])
        g .+= fa.w[i] .* γ_i ./ p
    end
    return g
end

# Equilibrium price of a CES FisherMarket via Mirror Descent (EG step,
# cc13 stepsize, CESAnalytic best-reply). Returns a simplex-normalized
# price direction. (validate.jl)
function solve_ces_equilibrium(fa::FisherMarket; α=500.0, maxiter=10000,
    maxtime=120.0, tol=1e-12, verbose=false)
    n, m = fa.n, fa.m
    alg = MirrorDec(n, m, ones(n) ./ n; α=α, maxiter=maxiter, maxtime=maxtime,
        tol=tol, optimizer=CESAnalytic, option_step=:eg, option_stepsize=:cc13)
    run = () -> opt!(alg, fa; maxiter=maxiter, maxtime=maxtime, tol=tol, reset=true)
    verbose ? run() : redirect_stdout(run, devnull)
    p = copy(alg.p)
    p ./= sum(p)
    return (p=p, iters=alg.k, grad_norm=alg.gₙ)
end

# -----------------------------------------------------------------------
# The lift (theory `lem.lift`).
# -----------------------------------------------------------------------

# Inverse bijection q ↦ (p, π) onto the interior of 𝛥̄.
lift_price(q::AbstractVector) = (s = 1 + sum(q); (p=q ./ s, π=1 / s))

# Forward bijection (p, π) ↦ q = p/π.
unlift_price(p::AbstractVector, π::Real) = p ./ π

# Lifted excess demand Z̄(p,π) against supply (1, M):
#   Z̄_goods = d(q) − 1,   Z̄_money = (M + ⟨q,1⟩ − W(q)) − M = ⟨q,1⟩ − W(q),
# with q = p/π and W(q) = ⟨q, d(q)⟩. (eq.def.lifted.demand)
function lifted_excess_demand(fa::FisherMarket, p::AbstractVector, π::Real; M::Real)
    q = unlift_price(p, π)
    d = aggregate_ces_demand(fa, q)
    W = dot(q, d)
    Z_goods = d .- Vector(fa.q)              # supply fa.q is the unit vector
    Z_money = sum(q) - W                     # money demand − money supply M
    return (Z_goods=Z_goods, Z_money=Z_money, q=q, money_demand=M + sum(q) - W)
end

# -----------------------------------------------------------------------
# Build a small Fisher CES market with total budget `budget_total`
# (= equilibrium scale W). Pattern from setup.jl build_rep_data `:ces`.
# -----------------------------------------------------------------------
function build_fisher_ces(; n, m, seed, budget_total=1.0,
    ces_rho_range=(-3.5, 0.9), sparsity=0.99)
    Random.seed!(seed)
    ρ_lo, ρ_hi = ces_rho_range
    ρ_vec = ρ_lo .+ (ρ_hi - ρ_lo) .* rand(m)
    ws = cpu_workspace(n)
    add_ces!(ws, m; ρ=ρ_vec, scale=30.0, sparsity=sparsity)
    ws.ces.w ./= sum(ws.ces.w)               # normalize, then scale to W
    ws.ces.w .*= budget_total
    return FisherMarket(ws)
end

# -----------------------------------------------------------------------
# Lift a Fisher CES market to a genuine (n+1)-good Arrow–Debreu economy.
# -----------------------------------------------------------------------
function lift_to_ad_market(fa::FisherMarket; ρ_rentier::Float64=0.5)
    n, m = fa.n, fa.m
    w = collect(fa.w)
    nl, ml = n + 1, m + 1          # money = good n+1; rentier = agent m+1
    C = zeros(nl, ml)              # CES coefficients (n+1 × m+1)
    B = zeros(nl, ml)              # endowments       (n+1 × m+1)
    ρv = zeros(ml)
    for i in 1:m                   # buyers: CES over goods, money endowment w_i
        C[1:n, i] .= Vector(fa.c[:, i])     # 0 coefficient on money ⇒ buy no money
        B[nl, i] = w[i]
        ρv[i] = fa.ρ[i]
    end
    C[nl, ml] = 1.0                # rentier: utility in money only
    B[1:n, ml] .= 1.0             # rentier endows the unit supply of goods
    ρv[ml] = ρ_rentier
    return ArrowDebreuMarket(ml, nl; c=C, b=B, ρ=ρv,
        bool_force_dense=true, verbose=false)
end

# -----------------------------------------------------------------------
# Build the shared data once: the Fisher market A and its lift A′. Both
# forward_test and backward_test then run on this same (fa, ad) pair.
# -----------------------------------------------------------------------
function build_lift_markets(; n=4, m=5, seed=42, budget_total=1.0, ρ_rentier=0.5)
    fa = build_fisher_ces(; n=n, m=m, seed=seed, budget_total=budget_total)
    ad = lift_to_ad_market(fa; ρ_rentier=ρ_rentier)
    return (fa=fa, ad=ad)
end

# -----------------------------------------------------------------------
# Banner: describe A and A′ before a run.
# -----------------------------------------------------------------------
function print_lift_banner(mode::Symbol; n, m, W)
    println("=======================================================================")
    println(" Money-lift test  [$(mode)]")
    println("-----------------------------------------------------------------------")
    @printf(" A   = original Fisher CES market: %d goods, %d agents, fixed budgets\n", n, m)
    @printf("       summing to W=%.3g (the equilibrium scale). Demand d(q) is\n", W)
    println("       homogeneous of degree −1; supply is 1 per good.")
    println(" A′  = lift of A: adjoin money as good $(n+1) so demand becomes")
    println("       degree-0 and Walras-closed on 𝛥̄={(p,π):⟨1,p⟩+π=1}; q=p/π.")
    if mode === :forward
        println("Forward: build A′'s equilibrium FROM A's via q*↦(p*,π*) and")
        println("       check the closed-form lifted excess Z̄(p*,π*)=0.")
    else
        @printf("Backward: realize A′ as a genuine %d-good, %d-agent Arrow–\n", n + 1, m + 1)
        println("       Debreu economy (buyers hold money + CES over goods; one")
        println("       rentier endows the goods, wants money), solve it from")
        println("       scratch with afscaled_newton, then project q*=p*/π* back")
        println("       and check it clears A.")
    end
    println("=======================================================================")
end

# =======================================================================
# Forward test: construct A′ from A and check the lifted excess vanishes.
# =======================================================================
function forward_test(fa::FisherMarket, ad::ArrowDebreuMarket;
    M=2.0, tol=1e-6, seed=42, verbose=true)
    n, m = fa.n, fa.m
    W = sum(fa.w)                                    # equilibrium scale = total budget
    verbose && print_lift_banner(:forward; n=n, m=m, W=W)

    # --- equilibrium A: simplex direction p*, equilibrium-scale price q* ---
    res = solve_ces_equilibrium(fa)
    p_star = res.p                                   # Σ p_star = 1
    q_star = (W / dot(p_star, ones(n))) .* p_star    # scale so ⟨1,q*⟩ = W
    d_star = aggregate_ces_demand(fa, q_star)
    goods_clear = norm(d_star .- 1.0, Inf)
    @assert goods_clear < tol "A is not an equilibrium: ‖d(q*) − 1‖∞ = $goods_clear"

    # --- lift A → A′ = (p_lift, π_lift) on int 𝛥̄ ---
    p_lift, π_lift = lift_price(q_star)
    simplex_resid = abs(sum(p_lift) + π_lift - 1.0)
    @assert all(p_lift .> 0) && π_lift > 0 "A′ not in interior of lifted simplex"
    @assert simplex_resid < tol "A′ off the lifted simplex: residual = $simplex_resid"
    recovered_scale = 1 / π_lift - 1                 # = ⟨1,q*⟩, should equal W

    # --- test A′ is an equilibrium of the lifted market ---
    lz = lifted_excess_demand(fa, p_lift, π_lift; M=M)
    goods_excess = norm(lz.Z_goods, Inf)
    money_excess = abs(lz.Z_money)
    roundtrip = norm(lz.q .- q_star, Inf)
    walras = dot(p_lift, lz.Z_goods) + π_lift * lz.Z_money
    @assert goods_excess < tol "lifted goods market does not clear: $goods_excess"
    @assert money_excess < tol "lifted money market does not clear: $money_excess"
    @assert roundtrip < tol "bijection round-trip failed: $roundtrip"
    @assert abs(walras) < tol "Walras' law violated on the lift: $walras"
    @assert lz.money_demand > 0 "money demand nonpositive; raise M"

    # --- the constructed price must clear the GENUINE lifted AD market A′ ---
    # lift_price(q*) = (q*/(1+W), 1/(1+W)) is exactly A′'s equilibrium price
    # under the Σp̄=1 normalization, so the genuine economy clears there too
    # (money supply Σw_i = W). This is the same `ad` the backward test solves.
    p_bar = vcat(p_lift, π_lift)
    d_ad = aggregate_demand(ad, p_bar)
    ad_excess = norm(p_bar .* (Vector(ad.q) .- d_ad), Inf)
    @assert ad_excess < tol "forward-constructed price does not clear genuine A′: $ad_excess"

    # --- negative control: an off-equilibrium price must be rejected ---
    Random.seed!(seed + 100)
    q_bad = q_star .* exp.(0.3 .* randn(n))
    pb, πb = lift_price(q_bad)
    lzb = lifted_excess_demand(fa, pb, πb; M=M)
    bad_excess = norm(lzb.Z_goods, Inf)
    @assert bad_excess > 100 * tol "negative control failed: off-eq excess $bad_excess too small"

    if verbose
        @printf("\n┌─ forward lift test ─ n=%d m=%d W=%.3g M=%.3g\n", n, m, W, M)
        @printf("│  eq solve:        %d iters, grad_norm=%.2e\n", res.iters, res.grad_norm)
        @printf("│  A  goods clear:  ‖d(q*)−1‖∞      = %.3e\n", goods_clear)
        @printf("│  equilibrium q*:  [%s]\n", join((@sprintf("%.4f", x) for x in q_star), ", "))
        @printf("│  lifted price p*: [%s]   π* = %.4f\n",
            join((@sprintf("%.4f", x) for x in p_lift), ", "), π_lift)
        @printf("│  scale recovery:  1/π*−1 = %.6g   (W = %.6g)\n", recovered_scale, W)
        @printf("|  money price π*:  %.3e  \n", π_lift)
        @printf("│  A′ goods excess: ‖Z̄_goods‖∞     = %.3e\n", goods_excess)
        @printf("│  A′ money excess: |Z̄_money|      = %.3e\n", money_excess)
        @printf("│  bijection rt:    ‖q−q*‖∞         = %.3e\n", roundtrip)
        @printf("│  Walras on lift:  ⟨p,Z̄g⟩+πZ̄m     = %.3e\n", walras)
        @printf("│  clears genuine A′ (AD):  ‖p̄(q̄−d̄)‖∞ = %.3e\n", ad_excess)
        @printf("│  neg control:     off-eq ‖Z̄g‖∞   = %.3e  (≫ tol)\n", bad_excess)
        @printf("└─ PASS\n")
    end
    return (pass=true, goods_excess=goods_excess, money_excess=money_excess,
        recovered_scale=recovered_scale, walras=walras, ad_excess=ad_excess,
        bad_excess=bad_excess)
end

# =======================================================================
# Backward test: solve A′ from scratch, project back, validate against A.
# =======================================================================
function backward_test(fa::FisherMarket, ad::ArrowDebreuMarket;
    K=400, log_interval=100, η=10.0, ϵ=1e-9, tol=1e-4, verbose=true)
    n, m = fa.n, fa.m
    W = sum(fa.w)
    verbose && print_lift_banner(:backward; n=n, m=m, W=W)

    # --- lifted AD market A′ (built from the same fa) ---
    nl, ml = ad.n, ad.m
    @assert nl == n + 1 && ml == m + 1
    @assert isapprox(sum(ad.q[1:n]), n; atol=1e-9)         # goods supply = 1 each
    @assert isapprox(ad.q[nl], W; atol=1e-9)               # money supply = Σ w_i = W

    # --- solve A′ from scratch via afscaled_newton ---
    linconstr = LinearConstr(1, nl, ones(1, nl), [1.0])     # Σ p̄ = 1
    alg = HessianBar(nl, ml, ones(nl) ./ nl; linconstr=linconstr)
    alg.linsys = :direct
    run = () -> afscaled_newton(ad, alg; K=K, log_interval=log_interval,
        scaler=1.0, η=η, ϵ=ϵ, ff="/tmp/test_lift_backward.log")
    verbose ? run() : redirect_stdout(run, devnull)
    p_bar = copy(alg.p)
    ad_excess = norm(p_bar .* (ad.q .- ad.sumx), Inf)       # price-scaled excess of A′

    # --- project back: q* = p*/π* ---
    π_star = p_bar[nl]
    q_rec = p_bar[1:n] ./ π_star

    # --- validate against the ORIGINAL market A (clearing is the criterion) ---
    d_rec = aggregate_ces_demand(fa, q_rec)
    goods_excess = norm(d_rec .- 1.0, Inf)                  # original goods clearing
    scale_rec = sum(q_rec)                                  # should be W

    @assert ad_excess < 1e-3 "A′ did not reach equilibrium (excess $ad_excess); raise K"
    @assert goods_excess < tol "projected price does not clear original A: $goods_excess"
    @assert abs(scale_rec - W) < tol "scale not recovered: $scale_rec vs W=$W"

    if verbose
        @printf("\n┌─ backward lift test ─ n=%d m=%d W=%.3g\n", n, m, W)
        @printf("│  A′ (lifted AD):  %d goods, %d agents\n", nl, ml)
        @printf("│  A′ excess:       ‖p̄(q̄−Σx)‖∞   = %.3e\n", ad_excess)
        @printf("│  money price π*:  %.6g   (q* scale ⟨1,q*⟩ recovered = %.6g, W = %.6g)\n",
            π_star, scale_rec, W)
        @printf("│  projected clears original A:   ‖d(q*)−1‖∞ = %.3e\n", goods_excess)
        @printf("│  lifted price p*: [%s]/sum(p*) = [%s]  π* = %.4f\n",
            join((@sprintf("%.4f", x) for x in p_bar[1:n]), ", "), sum(p_bar[1:n]), p_bar[nl])
        @printf("│  recovered q* = p*/π*:  [%s]\n", join((@sprintf("%.4f", x) for x in q_rec), ", "))
        @printf("└─ PASS\n")
    end
    return (pass=true, ad_excess=ad_excess, goods_excess=goods_excess,
        scale_rec=scale_rec, q_rec=q_rec)
end

# =======================================================================
# Driver: build (fa, ad) once, run forward and backward on the same data.
# =======================================================================
function run_lift_tests(; n=4, m=5, seed=42, budget_total=1.0, ρ_rentier=0.5,
    M=2.0, K=400, log_interval=100, η=10.0, ϵ=1e-9,
    forward_tol=1e-6, backward_tol=1e-4, verbose=true)
    mk = build_lift_markets(; n=n, m=m, seed=seed,
        budget_total=budget_total, ρ_rentier=ρ_rentier)
    fwd = forward_test(mk.fa, mk.ad; M=M, tol=forward_tol, seed=seed, verbose=verbose)
    bwd = backward_test(mk.fa, mk.ad; K=K, log_interval=log_interval,
        η=η, ϵ=ϵ, tol=backward_tol, verbose=verbose)
    return (fa=mk.fa, ad=mk.ad, forward=fwd, backward=bwd)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_lift_tests()
    run_lift_tests(n=5, m=8, budget_total=1.0, seed=7)
end
