# Arrow–Debreu nonlinear-least-squares surrogate fit — the AD sibling of
# nls.jl::solve_nls.
#
# Fisher (nls.jl): fit t CES agents with FIXED scalar budgets w ∈ Δᵗ, predictor
#   h(p) = Σ_i w_i γ_i(p).
# Arrow–Debreu (here): fit t CES agents with ENDOWMENTS b_i ∈ ℝⁿ₊, Σ_i b_i = 1,
#   so the budget is price-dependent w_i(p) = ⟨p, b_i⟩ and the predictor is
#   h(p) = Σ_i ⟨p, b_i⟩ γ_i(p)   (eq.cg.predictor.ad in method.tex / the AD
#   master in wealth-dist.tex).
#
# Both share the CES share γ_i(p) = softmax(y_i − σ_i log p). Only the budget
# differs: a scalar w_i (Fisher) vs the n-vector inner product ⟨p, b_i⟩ (AD).
# The simplex/supply constraint is likewise lifted: w ∈ Δᵗ (one column-softmax
# over agents) becomes Σ_i b_i = 1 per good (n column-softmaxes, one per good).
#
# Like solve_nls, the program is solved by MadNLP over an unconstrained
# reparameterization (the constraints are folded into softmaxes), with
# ADNLPModels supplying the derivatives.

using LinearAlgebra
using Random
using MadNLP
using ADNLPModels
using LogExpFunctions: logsumexp

"""
    solve_nls_ad(Ξ, t; delta=1.0, ...) -> (Y, σ_vec, B, δ, result)

Solve the Arrow–Debreu NLS problem for t CES agents jointly, with supply scale
δ (sec.wealth.ad.scale, mirroring adcg's `--ad-delta` / `--ad-delta-free`):
    min_{y_i, σ_i, b_i, [δ]}  Σ_k ‖ Σ_i ⟨p_k, b_i⟩ γ_i(p_k) − (P_k g_k + (δ−1)p_k) ‖²
    s.t.                 γ_i(p_k) = softmax(y_i − σ_i log p_k),
                         b_i ≥ 0,  Σ_i b_i = δ·1  (per good),  [δ ≥ 0].

Endowments are parameterized by a free t×n matrix β whose COLUMNS are pushed
through a softmax over agents (colsum 1), then scaled by δ so Σ_i b_i = δ·1 —
the AD analog of solve_nls's `w = softmax(α)`. `delta` is either a fixed
nonnegative number (default 1.0 ⇒ unit supply, the shift (δ−1)p_k vanishes) or
`:free`, in which case δ ≥ 0 becomes an extra decision variable.

Arguments / keywords mirror solve_nls (Y_init / σ_init warm-start the agent
parameters; B and δ are always started at the uniform / unit-supply point).
Returns the agent parameters `Y` (t×n), `σ_vec` (t), the endowment matrix
`B` (t×n, agent-by-good, columns summing to δ), the supply scale `δ` used, and
the MadNLP result.
"""
function solve_nls_ad(Ξ::Vector{Tuple{Vector{T},Vector{T}}}, t::Int;
    delta::Union{Real,Symbol}=1.0,
    max_iter::Int=100,
    Y_init::Union{Matrix{T},Nothing}=nothing,
    σ_init::Union{Vector{T},Nothing}=nothing,
    verbose::Bool=false,
    seed::Union{Int,Nothing}=nothing) where T

    delta_free = delta === :free
    delta_free || (delta isa Real && delta >= 0) ||
        error("solve_nls_ad: delta must be a nonnegative number or :free (got $delta)")

    K = length(Ξ)
    n = length(Ξ[1][1])

    # Precompute targets P_k g_k, prices, and log-prices.
    targets = zeros(T, K, n)
    prices = zeros(T, K, n)
    log_prices = zeros(T, K, n)
    for k in 1:K
        p_k, g_k = Ξ[k]
        targets[k, :] = p_k .* g_k
        prices[k, :] = p_k
        log_prices[k, :] = log.(p_k)
    end

    # Parameter layout:
    #   θ = [y_1(n); σ_1; …; y_t(n); σ_t; β(t·n); [δ]]
    # Agent block as in solve_nls; β is the t×n endowment logits, column-softmax
    # over the t agents → b̃ (colsum 1), then scaled by δ → B (Σ_i B[i,j] = δ).
    # δ is the trailing variable only when delta === :free.
    dim_agent = n + 1
    dim_endow = t * dim_agent + t * n
    dim_total = dim_endow + (delta_free ? 1 : 0)

    function unpack(θ)
        S = eltype(θ)
        Y = zeros(S, t, n)
        σ_vec = zeros(S, t)
        for i in 1:t
            off = (i - 1) * dim_agent
            Y[i, :] = θ[off+1:off+n]
            σ_vec[i] = θ[off+n+1]
        end
        β = reshape(θ[t*dim_agent+1:t*dim_agent+t*n], t, n)   # t×n
        δ = delta_free ? θ[dim_endow+1] : S(delta)
        B = zeros(S, t, n)
        for j in 1:n
            col = @view β[:, j]
            # softmax over agents (colsum 1), scaled by δ ⇒ Σ_i B[i,j] = δ.
            B[:, j] = δ .* exp.(col .- logsumexp(col))
        end
        return Y, σ_vec, B, δ
    end

    function objective(θ)
        S = eltype(θ)
        Y, σ_vec, B, δ = unpack(θ)
        val = zero(S)
        for k in 1:K
            # Shifted target P_k g_k + (δ-1) p_k (eq.ad.shifted.data); at δ=1
            # this is just P_k g_k.
            res_k = -(S.(targets[k, :]) .+ (δ - 1) .* prices[k, :])
            for i in 1:t
                budget = dot(@view(prices[k, :]), @view(B[i, :]))   # ⟨p_k, b_i⟩
                z = Y[i, :] .- σ_vec[i] .* log_prices[k, :]
                γ_ik = exp.(z .- logsumexp(z))
                res_k .+= budget .* γ_ik
            end
            val += dot(res_k, res_k)
        end
        return val
    end

    !isnothing(seed) && Random.seed!(seed)

    θ0 = zeros(T, dim_total)
    delta_free && (θ0[dim_endow+1] = one(T))   # start δ at unit supply
    if !isnothing(Y_init) && !isnothing(σ_init)
        for i in 1:min(t, size(Y_init, 1))
            off = (i - 1) * dim_agent
            θ0[off+1:off+n] = Y_init[i, :]
            θ0[off+n+1] = σ_init[i]
        end
        # New agents (i > size(Y_init,1)) start random, matching solve_nls.
        for i in (size(Y_init, 1)+1):t
            off = (i - 1) * dim_agent
            θ0[off+1:off+n] = randn(T, n) * 0.5
            θ0[off+n+1] = rand(T) * 2.0
        end
    else
        for i in 1:t
            off = (i - 1) * dim_agent
            θ0[off+1:off+n] = randn(T, n) * 0.5
            θ0[off+n+1] = rand(T) * 2.0
        end
    end
    # β starts at 0 ⇒ uniform endowments b_{i,j} = 1/t (interior start).

    lower = fill(T(-Inf), dim_total)
    upper = fill(T(Inf), dim_total)
    for i in 1:t
        off = (i - 1) * dim_agent
        lower[off+n+1] = T(-0.98)   # σ > -1 for ρ < 1
        upper[off+n+1] = T(400.0)
    end
    delta_free && (lower[dim_endow+1] = zero(T))   # δ ≥ 0

    nlp = ADNLPModel(objective, θ0, lower, upper;
        hessian_backend=ADNLPModels.EmptyADbackend,
    )
    best_result = madnlp(nlp;
        max_iter=max_iter, tol=1e-6,
        hessian_approximation=MadNLP.CompactLBFGS,
        linear_solver=LapackCPUSolver,
        print_level=verbose ? MadNLP.INFO : MadNLP.ERROR,
    )

    Y_opt, σ_opt, B_opt, δ_opt = unpack(best_result.solution)
    return Y_opt, σ_opt, B_opt, δ_opt, best_result
end

"""
    evaluate_nls_ad(Ξ, Y, σ_vec, B; delta=1.0) -> (obj, max_err)

Raw AD-NLS residual statistics: the squared ℓ₂ objective Σ_k ‖·‖₂² and the
ℓ∞ residual, for the predictor Σ_i ⟨p, b_i⟩ γ_i(p) against the shifted target
p⊙g + (δ-1)p (eq.ad.shifted.data; just p⊙g at δ=1). Sibling of evaluate_nls;
reported alongside the mean-ℓ₁ train/test error.
"""
function evaluate_nls_ad(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    Y::Matrix{T}, σ_vec::Vector{T}, B::Matrix{T}; delta::Real=1.0) where T

    K = length(Ξ)
    n = length(Ξ[1][1])
    t = length(σ_vec)
    obj = zero(T)
    max_err = zero(T)
    for k in 1:K
        p_k, g_k = Ξ[k]
        log_p = log.(p_k)
        target_k = p_k .* g_k .+ (T(delta) - 1) .* p_k
        fitted_k = zeros(T, n)
        for i in 1:t
            budget = dot(p_k, @view B[i, :])
            z = Y[i, :] .- σ_vec[i] .* log_p
            γ_ik = exp.(z .- logsumexp(z))
            fitted_k .+= budget .* γ_ik
        end
        r = fitted_k .- target_k
        obj += dot(r, r)
        max_err = max(max_err, maximum(abs, r))
    end
    return obj, max_err
end
