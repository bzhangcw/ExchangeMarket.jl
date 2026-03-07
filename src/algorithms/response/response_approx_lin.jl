# -----------------------------------------------------------------------
# approximate best-response for linear markets
#   with log-barrier regularization (σ > 0)
# @reference: Lemma in extend.tex (lump-sum FOC for linear utility)
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# -----------------------------------------------------------------------

@doc raw"""
    Approximate analytic best-response for linear markets with
    log-barrier regularization parameter ϵᵢ > 0 (passed from play!).

    Given linear utility u = ⟨c, x⟩, the regularized best response satisfies:
        x_j = (σ/(λ p_j)) · u/(u - b_j),  λ = (1 + σn) / w,  b_j = c_j/(λ p_j)
    where σ = ϵᵢ/n and u is the root of the convex equation
        φ(u) = σ Σ_j b_j / (u - b_j) - 1 = 0,   u > max(b_j).
    φ is convex and monotone decreasing, so Newton converges monotonically.

    Stores ϵᵢ into market.ε_br_play[i] for use by the Hessian computation.
"""
function __approx_lin_response(;
    i::Int=1,
    p::Vector{T}=nothing,
    market::Market=nothing,
    ϵᵢ::Float64=1e-3,
    debug::Bool=true,
    kwargs...
) where {T}
    n = market.n
    w = market.w[i]
    # ε = max(ϵᵢ, market.ε_br_play[i])
    ε = market.ε_br_play[i]
    σ = ε / n
    c = market.c[:, i]
    λ = (1 + σ * n) / w

    # solve φ(u) = σ Σ b_j/(u - b_j) - 1 = 0 by bisection.
    b = c ./ (λ .* p)
    b_max = maximum(b)
    φ(u) = σ * sum(b ./ (u .- b)) - 1
    # bisection: φ > 0 near b_max, φ → -1 as u → ∞
    lo = b_max + 1e-15
    hi = b_max + sum(b)
    while φ(hi) > 0
        hi *= 2
    end
    niter = 0
    for iter in 1:200
        u = (lo + hi) / 2
        v = φ(u)
        niter = iter
        abs(v) < 1e-12 && break
        (hi - lo) < 1e-14 * hi && break
        v > 0 ? (lo = u) : (hi = u)
    end
    u = (lo + hi) / 2
    debug && (abs(φ(u)) > ε) && @info "ApproxLin bisection" ε i niter u φ(u)

    # allocation: x_j = (σ/(λ p_j)) · u/(u - b_j)
    x = (σ ./ (λ .* p)) .* u ./ (u .- b)
    market.x[:, i] .= x
    market.val_u[i] = c' * x

    return nothing
end

ApproxLin = ResponseOptimizer(
    __approx_lin_response,
    :approx_lin,
    "ApproxLinResponse"
)

# -----------------------------------------------------------------------
# JuMP conic version: max log(c'x) + σ Σ log(x_j) s.t. p'x ≤ w
# -----------------------------------------------------------------------
@doc raw"""
    Solve the σ-regularized log-UMP with linear utility via conic programming:
        max  log(⟨c, x⟩) + σ Σ_j log(x_j)
        s.t. ⟨p, x⟩ ≤ w,  x ≥ 0.
    Uses exponential cone: log(x) ≥ t  ⟺  [t, 1, x] ∈ ExpCone.
"""
function __conic_approx_lin_response(;
    i::Int=1,
    p::Vector{T}=nothing,
    market::Market=nothing,
    ϵᵢ::Float64=1e-3,
    verbose::Bool=false,
    kwargs...
) where {T}
    n = market.n
    w = market.w[i]
    ε = max(ϵᵢ, market.ε_br_play[i])
    σ = ε / n
    c = market.c[:, i]

    md = __generate_empty_jump_model(; verbose=verbose, tol=1e-12)

    @variable(md, x[1:n] >= 0)
    # u = c'x and log(u)
    @variable(md, u)
    @variable(md, logu)
    @constraint(md, u == c' * x)
    log_to_expcone!(u, logu, md)
    # log(x_j) via exponential cone
    @variable(md, logx[1:n])
    log_to_expcone!.(x, logx, md)
    # budget
    @constraint(md, bgt, p' * x <= w)

    @objective(md, Max, logu + σ * sum(logx))

    JuMP.optimize!(md)
    market.x[:, i] .= max.(value.(x), 0.0)
    market.val_u[i] = c' * market.x[:, i]
    return md
end

ApproxLinConic = ResponseOptimizer(
    __conic_approx_lin_response,
    :approx_lin_conic,
    "ApproxLinConicResponse"
)
