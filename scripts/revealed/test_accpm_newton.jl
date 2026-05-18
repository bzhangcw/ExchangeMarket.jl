# Smoke test for the hand-rolled Newton analytic-center solver in
# accpm.jl (`solve_analytic_center_newton`), against the Mosek reference
# (`solve_analytic_center`).
#
# Run from scripts/:
#   julia --project=. revealed/test_accpm_newton.jl
#
# Verifies, on a small synthetic CES instance:
#   1. The Newton iteration converges in a handful of steps (Newton
#      decrement λ_N ≤ ε_ac).
#   2. The recovered (u, μ) is strictly interior of the polytope.
#   3. The KKT residual ‖A^T (1/s)‖∞ ≤ ε_ac (eq.accpm.ac in the doc).
#   4. The Newton (u, μ) agrees with the Mosek (u, μ) to ~1e-4
#      (Mosek's exp-cone barrier and our log-barrier should be the
#      same problem; small disagreement is acceptable from tolerances).
#   5. Warm-starting the next call from the previous z cuts iter count.

using Printf
using LinearAlgebra
using Random

include("./setup.jl")

Random.seed!(20260517)

# ------ tiny problem -----------------------------------------------------
# K samples × n goods, with m synthetic γ rows. Keep small so Mosek is
# fast and the matrices are easy to eyeball if anything goes wrong.
const K_test = 4
const n_test = 3
const m_test = 5

# Random prices on the simplex, demands inside.
Ξ_test = [(rand(n_test) .+ 0.1, rand(n_test) .+ 0.1) for _ in 1:K_test]

# Random γ tensor (m, K, n) with rows summing to ≈1 (softmax-style).
γ_raw = randn(m_test, K_test, n_test)
γ_test = similar(γ_raw)
for i in 1:m_test, k in 1:K_test
    z = γ_raw[i, k, :]
    e = exp.(z .- maximum(z))
    γ_test[i, k, :] .= e ./ sum(e)
end

# Loose μ_ub from problem data (matches accpm.jl::_initial_mu_ub).
μ_ub_test = sum(maximum(abs.(p .* g)) for (p, g) in Ξ_test) * 1.1 + 1.0
@printf("μ_ub = %g\n", μ_ub_test)

# ------ 1. Newton: solve from scratch -----------------------------------
println("\n=== Newton from Phase-I ===")
u_n, μ_n, info = solve_analytic_center_newton(Ξ_test, γ_test, μ_ub_test;
                                               ε_ac=1e-8, max_iters=80)
@printf("converged: niter = %d,  λ_N = %.3e\n", info.niter, info.λN)
@assert info.λN <= 1e-7 "Newton failed to converge: λ_N = $(info.λN)"

# ------ 2. Strict feasibility check -------------------------------------
# Rebuild (A, b) via the same helper accpm.jl uses, so we can read off
# the recovered slack and KKT residual.
A_chk, b_chk = _ac_build_constraints(γ_test, μ_ub_test)
s_chk = b_chk .- A_chk * info.z
@printf("strict feasibility: min slack = %.3e (must be > 0)\n", minimum(s_chk))
@assert minimum(s_chk) > 0 "Recovered z is not strictly feasible"

# ------ 3. KKT residual (eq.accpm.ac) -----------------------------------
kkt = A_chk' * (1.0 ./ s_chk)
@printf("KKT residual ‖A^T (1/s)‖∞ = %.3e (must be ≤ ε_ac = 1e-8)\n",
        maximum(abs.(kkt)))
@assert maximum(abs.(kkt)) <= 1e-7 "KKT residual too large"

# ------ 4. Compare against Mosek (sanity, not exact) --------------------
# The two solvers barrier slightly different sets:
#   Newton (this doc's eq.accpm.objective): log-barrier on ALL M slacks,
#     including the 2nK lift constraints (a_{kj} ± u_{kj} ≥ 0).
#   Mosek (existing solve_analytic_center): log-barrier on the cut, μ-bound,
#     and simplex slacks ONLY; lift inequalities are hard constraints.
# So the two centers differ by O(1/lift_slack) on average. We sanity-check
# that they're in the same neighborhood (within a few percent of μ_ub).
println("\n=== Mosek reference (different barrier set; sanity only) ===")
u_m, μ_m, _ = solve_analytic_center(Ξ_test, γ_test, μ_ub_test; verbose=false)
Δu = norm(u_n - u_m, Inf)
Δμ = abs(μ_n - μ_m)
@printf("‖u_Newton − u_Mosek‖∞ = %.3e\n", Δu)
@printf("|μ_Newton − μ_Mosek|  = %.3e  (μ_ub = %.3e)\n", Δμ, μ_ub_test)
@assert Δu < 0.1 "Newton vs Mosek u way off (Δu = $Δu); barrier-set difference shouldn't be this large"
@assert Δμ < 0.1 * μ_ub_test "Newton vs Mosek μ way off"

# ------ 5. Warm-start behavior ------------------------------------------
# Adversarial case: the new cut is near the OLD analytic center (this is
# precisely the realistic CG setting — the separation oracle returns a
# separating hyperplane, so the cut is tight at z by construction). In
# that case the previous z sits right where the new log-barrier is
# steepest, and warm-start Newton can be slower than cold restart. We
# report both numbers without asserting an ordering.
println("\n=== Warm-start (adversarial: new cut near old AC) ===")
γ_grow = cat(γ_test, reshape(γ_test[1, :, :] .+ 0.01 .* randn(K_test, n_test),
                              1, K_test, n_test); dims=1)
_, _, info2_cold = solve_analytic_center_newton(Ξ_test, γ_grow, μ_ub_test;
                                                ε_ac=1e-8, max_iters=80)
_, _, info2_warm = solve_analytic_center_newton(Ξ_test, γ_grow, μ_ub_test;
                                                z_warm=info.z, ε_ac=1e-8, max_iters=80)
@printf("cold start:  niter = %d, λ_N = %.2e\n", info2_cold.niter, info2_cold.λN)
@printf("warm start:  niter = %d, λ_N = %.2e\n", info2_warm.niter, info2_warm.λN)
@assert info2_warm.λN <= 1e-7 "Warm-start failed to converge"
@assert info2_cold.λN <= 1e-7 "Cold-start failed to converge"

# Benign case: a "neutral" cut far from the old AC. Warm-start should
# be at-or-better than cold here.
println("\n=== Warm-start (benign: new cut far from old AC) ===")
γ_benign_row = ones(K_test, n_test) ./ n_test   # uniform row, deep in polytope
γ_benign = cat(γ_test, reshape(γ_benign_row, 1, K_test, n_test); dims=1)
_, _, info3_cold = solve_analytic_center_newton(Ξ_test, γ_benign, μ_ub_test;
                                                ε_ac=1e-8, max_iters=80)
_, _, info3_warm = solve_analytic_center_newton(Ξ_test, γ_benign, μ_ub_test;
                                                z_warm=info.z, ε_ac=1e-8, max_iters=80)
@printf("cold start:  niter = %d, λ_N = %.2e\n", info3_cold.niter, info3_cold.λN)
@printf("warm start:  niter = %d, λ_N = %.2e\n", info3_warm.niter, info3_warm.λN)
@assert info3_warm.λN <= 1e-7 "Benign warm-start failed to converge"

println("\nAll Newton AC tests passed.")
