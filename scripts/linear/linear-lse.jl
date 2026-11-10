using Optim, LineSearches, ForwardDiff
using LinearOperators, Krylov

logsumexp(z) = (zmax = maximum(z); zmax + log(sum(exp.(z .- zmax))))
softmax(z) = (ez = exp.(z .- maximum(z)); ez ./ sum(ez))

"""
    Logsumexp-smoothed dual for linear Fisher market.

    φᵗ(p) = ⟨p, 1⟩ + Σᵢ wᵢ log(wᵢ · LSEₜ(cᵢ/p))

    where LSEₜ(z) = t log(Σⱼ exp(zⱼ/t)).
"""

# type-generic: works with ForwardDiff.Dual
function lse_dual(p, f; t=1e-1)
    n, m = f.n, f.m
    T = eltype(p)
    val = sum(p)
    c = f.c isa SparseMatrixCSC ? Matrix(f.c) : f.c
    for i in 1:m
        # bang-per-buck: z_j = c[j,i] / p[j]
        z = c[:, i] ./ p
        # numerically stable logsumexp
        zmax = maximum(z)
        Li = t * (zmax / t + log(sum(exp.((z .- zmax) ./ t))))
        val += f.w[i] * log(f.w[i] * Li)
    end
    return val
end

function lse_grad!(g::Vector{T}, p::Vector{T}, f; t::T=1e-1) where {T}
    n, m = f.n, f.m
    c = f.c isa SparseMatrixCSC ? Matrix(f.c) : f.c
    g .= one(T)
    for i in 1:m
        z = c[:, i] ./ p
        zmax = maximum(z)
        ez = exp.((z .- zmax) ./ t)
        S = sum(ez)
        α = ez ./ S
        Li = t * (zmax / t + log(S))
        d = α .* c[:, i] ./ (p .^ 2)
        g .-= (f.w[i] / Li) .* d
    end
    return g
end

function lse_hess!(H::Matrix{T}, p::Vector{T}, f; t::T=1e-1, bool_diag_only::Bool=false) where {T}
    n, m = f.n, f.m
    c = f.c isa SparseMatrixCSC ? Matrix(f.c) : f.c
    H .= zero(T)
    for i in 1:m
        cᵢ = @view c[:, i]
        z = cᵢ ./ p
        Li = t * logsumexp(z ./ t)
        γ = softmax(z ./ t)
        dᵢ = cᵢ ./ (Li .* p)         # d_i = P⁻¹ c_i / L_i
        eᵢ = γ .* dᵢ                  # e_i = Γ_i d_i
        # diagonal: w_i e_i ⊙ (z / t + 2)
        Λᵢ = f.w[i] .* eᵢ .* (z ./ t .+ 2.0)
        H[diagind(H)] .+= Λᵢ
        if !bool_diag_only
            # rank-one: -w_i (1 + L_i/t) e_i e_iᵀ
            βᵢ = f.w[i] * (1.0 + Li / t)
            H .-= βᵢ .* (eᵢ * eᵢ')
        end
    end
    return H
end

"""
    lse_dual_newton(f; t, p₀, hessian, maxiter, gtol, show_trace, p_order)

Minimize φᵗ(p) using affine-scaled Newton with backtracking.
  hessian = :exact       — exact Hessian (dense solve)
  hessian = :dr1         — DR1 approximation (Sherman-Morrison)
  hessian = :diag        — diagonal approximation
  hessian = :krylov      — CG with diagonal preconditioner
  hessian = :vonneumann  — Von Neumann series (p_order iterations)
Returns (p, traj) where traj is the gradient norm history.
"""
function lse_dual_newton(
    f;
    t::Float64=1e-1,
    p₀::Union{Vector{Float64},Nothing}=nothing,
    hessian::Symbol=:dr1,
    maxiter::Int=500,
    gtol::Float64=1e-10,
    show_trace::Bool=true,
    show_every::Int=2,
    p_order::Int=4
)
    n = f.n
    p = p₀ === nothing ? ones(n) * sum(f.w) / n : copy(p₀)

    traj = Float64[]
    for k in 1:maxiter
        φ = lse_dual(p, f; t=t)
        res = lse_newton_step(p, f; t=t, hessian=hessian, p_order=p_order)
        gnorm = norm(res.g, Inf)
        push!(traj, gnorm)

        if gnorm < gtol
            show_trace && @printf("converged at k=%d, |∇φ|=%.2e\n", k, gnorm)
            break
        end

        # backtracking line search (Armijo): p₊ = p - α q
        α = 1.0
        slope = -dot(res.g, res.q)
        for _ in 1:30
            p_new = p .- α .* res.q
            if all(p_new .> 0) && lse_dual(p_new, f; t=t) <= φ + 1e-4 * α * slope
                break
            end
            α *= 0.5
        end
        p .-= α .* res.q

        if show_trace && k % show_every == 0
            if hessian == :krylov
                @printf("k=%3d  φ=%.8e  |∇φ|=%.2e  α=%.2e  cg=%d\n", k, φ, gnorm, α, res.niter)
            else
                @printf("k=%3d  φ=%.8e  |∇φ|=%.2e  α=%.2e\n", k, φ, gnorm, α)
            end
        end
    end

    return (p=p, traj=traj)
end

"""
    lse_allocation!(f, p; t)

Compute the entropy-regularized (softmax) allocation in place.
    αᵢⱼ = softmax(cᵢ/(t·p))ⱼ
    xᵢⱼ = wᵢ αᵢⱼ / pⱼ
Updates `f.x`, `f.p`, `f.val_u`, `f.sumx`.
"""
function lse_allocation!(f, p::Vector{Float64}; t::Float64=1e-1)
    c = f.c isa SparseMatrixCSC ? Matrix(f.c) : f.c
    for i in 1:f.m
        z = c[:, i] ./ p
        zmax = maximum(z)
        ez = exp.((z .- zmax) ./ t)
        α = ez ./ sum(ez)
        f.x[:, i] .= f.w[i] .* α ./ p
        f.val_u[i] = dot(c[:, i], f.x[:, i])
    end
    f.p .= p
    f.sumx .= sum(f.x; dims=2)[:]
    return f
end

struct OptimAlg <: ExchangeMarket.Algorithm
    p::Vector{Float64}
    μ::Float64
    name::String
end
OptimAlg(p) = OptimAlg(p, 0.0, "LSE-IPNewton")

"""
    lse_hess_dr1(p, f; t, bool_diag_only, bool_full) -> (Λ, ξ, Ω, matG, matD, matE, vecL)

DR1 components of the affine-scaled Hessian H(p) = P∇²φᵗP.
    H̃ = diag(Λ) - Ω ξξᵀ
"""
function lse_hess_dr1(p::Vector{Float64}, f; t::Float64=1e-1, bool_diag_only::Bool=false, bool_full::Bool=false)
    n, m = f.n, f.m
    c = f.c isa SparseMatrixCSC ? Matrix(f.c) : f.c
    Λ = zeros(n)
    Σ = zeros(n)
    ξ = zeros(n)
    Ω = 0.0

    matG = bool_full ? zeros(n, m) : nothing
    matD = bool_full ? zeros(n, m) : nothing
    matE = bool_full ? zeros(n, m) : nothing
    vecL = bool_full ? zeros(m) : nothing

    for i in 1:m
        cᵢ = @view c[:, i]
        z = cᵢ ./ p
        Li = t * logsumexp(z ./ t)
        γ = softmax(z ./ t)
        dᵢ = cᵢ ./ (Li .* p)
        eᵢ = γ .* dᵢ

        if bool_full
            matG[:, i] .= γ
            matD[:, i] .= dᵢ
            matE[:, i] .= eᵢ
            vecL[i] = Li
        end

        # diagonal: w_i e_i ⊙ (z / t + 2)
        Λ .+= f.w[i] .* eᵢ .* (z ./ t .+ 2.0)

        if !bool_diag_only
            βi = f.w[i] * (1.0 + Li / t)
            Ω += βi
            ξ .+= βi .* eᵢ
            Σ .+= βi .* eᵢ .^ 2
        end
    end
    if !bool_diag_only
        ξ ./= Ω
    end

    return (Λ=Λ, Σ=Σ, ξ=ξ, Ω=Ω, matG=matG, matD=matD, matE=matE, vecL=vecL)
end

"""
    lse_hess_dr1_inv(Λ, ξ, Ω, r) -> H̃⁻¹ r

Apply the DR1 inverse to a vector r via Sherman-Morrison.
"""
function lse_hess_dr1_inv(Λ::Vector{Float64}, ξ::Vector{Float64}, Ω::Float64, r::Vector{Float64})
    Λinv_r = r ./ Λ
    Λinv_ξ = ξ ./ Λ
    denom = 1.0 - Ω * dot(ξ, Λinv_ξ)
    return Λinv_r .+ (Ω / denom) .* Λinv_ξ .* dot(Λinv_ξ, r)
end

"""
    lse_hess_diag_inv(Λ, Σ, r) -> Ĥ⁻¹ r

Apply the diagonal approximation inverse: (Λ - Σ)⁻¹ r,
where Σ = Σᵢ βᵢ eᵢ⊙eᵢ is the exact diagonal of the rank-one sum.
"""
lse_hess_diag_inv(Λ::Vector{Float64}, Σ::Vector{Float64}, r::Vector{Float64}) = r ./ (Λ .- Σ)

"""
    lse_hess_matvec!(buf, v, p, f; t)

Compute buf = H(p) v where H = P∇²φᵗP (affine-scaled Hessian), matrix-free.
"""
function lse_hess_matvec!(buf::Vector{Float64}, v::Vector{Float64}, p::Vector{Float64}, f; t::Float64=1e-1)
    n, m = f.n, f.m
    c = f.c isa SparseMatrixCSC ? Matrix(f.c) : f.c
    buf .= 0.0
    for i in 1:m
        cᵢ = @view c[:, i]
        z = cᵢ ./ p
        Li = t * logsumexp(z ./ t)
        γ = softmax(z ./ t)
        dᵢ = cᵢ ./ (Li .* p)
        eᵢ = γ .* dᵢ
        # diagonal contribution
        buf .+= f.w[i] .* eᵢ .* (z ./ t .+ 2.0) .* v
        # rank-one contribution
        βᵢ = f.w[i] * (1.0 + Li / t)
        buf .-= βᵢ * dot(eᵢ, v) .* eᵢ
    end
    return buf
end

"""
    lse_newton_step(p, f; t, hessian=:dr1, p_order=4)

Solves H̃ d = P∇φᵗ, returns q = Pd (the unscaled direction).
  hessian = :exact       — exact Hessian (dense solve)
  hessian = :dr1         — DR1 approximation (Sherman-Morrison)
  hessian = :diag        — diagonal approximation
  hessian = :krylov      — CG with diagonal preconditioner
  hessian = :vonneumann  — Von Neumann series (p_order iterations)
"""
function lse_newton_step(
    p::Vector{Float64}, f; t::Float64=1e-1, hessian::Symbol=:dr1,
    cg_rtol::Float64=1e-8, cg_atol::Float64=1e-10, p_order::Int=4
)
    n = f.n
    g = zeros(n)
    lse_grad!(g, p, f; t=t)
    Pr = p .* g

    if hessian == :exact
        H = zeros(n, n)
        lse_hess!(H, p, f; t=t)
        H += sqrt(norm(Pr)) * I
        F = cholesky(Symmetric(H); check=false)
        d = issuccess(F) ? F \ Pr : pinv(H) * Pr
        return (q=p .* d, g=g, niter=0)
    end

    (; Λ, Σ, ξ, Ω) = lse_hess_dr1(p, f; t=t)

    if hessian == :diag
        d = lse_hess_diag_inv(Λ, Σ, Pr)
        return (q=p .* d, g=g, niter=0)
    elseif hessian == :krylov
        # Hessian-vector product operator
        hessop = LinearOperator(Float64, n, n, true, true,
            (buf, v) -> lse_hess_matvec!(buf, v, p, f; t=t)
        )
        # diagonal preconditioner: Λ - Σ (exact diagonal of H)
        M = opDiagonal(1 ./ (Λ .- Σ))
        d, stats = cg(hessop, Pr; M=M, rtol=cg_rtol, atol=cg_atol, itmax=20)
        return (q=p .* d, g=g, niter=stats.niter)
    elseif hessian == :vonneumann
        # M_p⁻¹ r via Von Neumann series: dₖ₊₁ = 2dₖ - Ĥ⁻¹ H dₖ
        Hhat_inv = 1 ./ (Λ .- Σ)
        d = Hhat_inv .* Pr          # d₀ = Ĥ⁻¹ r
        Hd = zeros(n)
        for _ in 1:p_order
            lse_hess_matvec!(Hd, d, p, f; t=t)
            d .= 2 .* d .- Hhat_inv .* Hd
        end
        return (q=p .* d, g=g, niter=p_order)
    else  # :dr1
        d = lse_hess_dr1_inv(Λ, ξ, Ω, Pr)
        return (q=p .* d, g=g, niter=0)
    end
end

"""
    lse_dual_regnewton(f; t, p₀, σ₀, maxiter, gtol, show_trace, show_every)

Minimize φᵗ(p) using regularized Newton with Mishchenko AdaN+ strategy.
Solves the affine-scaled system (H̃ + σI)d = P∇φ at each step,
where H̃ = P∇²φP is the affine-scaled Hessian. Step: p₊ = p - α P d.

σ is adapted via Mishchenko's rule [Algorithm 2.3, SIOPT 2023]:
  - Initialize: estimate Hessian Lipschitz M via finite difference, σ = √M
  - Update: σ = max(σ/√2, √M) using consecutive gradient/Hessian differences

Since the step is affine-scaled (Δ = Pd), any α < 1 with α‖d‖∞ < 1
guarantees p₊ > 0 automatically.

Returns (p, traj, σ) where traj is the gradient norm history.
"""
function lse_dual_regnewton(
    f;
    t::Float64=1e-1,
    p₀::Union{Vector{Float64},Nothing}=nothing,
    σ₀::Float64=0.0,
    maxiter::Int=500,
    gtol::Float64=5e-7,
    show_trace::Bool=true,
    show_every::Int=2
)
    n = f.n
    p = p₀ === nothing ? ones(n) * sum(f.w) / n : copy(p₀)

    H = zeros(n, n)
    g = zeros(n)
    g_prev = zeros(n)
    Pr = zeros(n)
    σ = σ₀
    traj = Float64[]

    for k in 1:maxiter
        # gradient and affine-scaled Hessian
        lse_grad!(g, p, f; t=t)
        lse_hess!(H, p, f; t=t)
        Pr .= p .* g
        gnorm = norm(g, Inf)
        push!(traj, gnorm)

        if gnorm < gtol
            show_trace && @printf("converged at k=%d, |∇φ|=%.2e, σ=%.2e\n", k, gnorm, σ)
            break
        end

        # Mishchenko adaptive σ initialization/update
        if k == 1
            dx = randn(n)
            dx .*= 1e-5 * norm(p) / norm(dx)
            g_pert = zeros(n)
            lse_grad!(g_pert, p .+ dx, f; t=t)
            H_pert = zeros(n, n)
            lse_hess!(H_pert, p .+ dx, f; t=t)
            # Lipschitz estimate in affine-scaled space
            Pr_pert = (p .+ dx) .* g_pert
            Hdx = H * dx
            M = norm(Pr_pert - Pr - Hdx) / norm(dx)^2
            σ = max(σ₀, sqrt(M))
        else
            # use consecutive iterates: dx = p - p_prev (encoded via d from last step)
            # approximate Lipschitz from gradient difference
            Hdx = H * (Pr .- p .* g_prev)  # H * (change in scaled gradient direction)
            Pr_diff = Pr .- p .* g_prev
            if norm(Pr_diff) > 1e-15
                M = norm(Pr - p .* g_prev - Hdx) / norm(Pr_diff)
                σ = max(σ / sqrt(2.0), sqrt(M))
            end
        end
        g_prev .= g

        # solve (H̃ + σI) d = Pr
        F = cholesky(Symmetric(H + σ * I); check=false)
        d = issuccess(F) ? F \ Pr : (Symmetric(H + σ * I)) \ Pr

        # affine-scaled step: q = Pd, then p₊ = p - α q
        q = p .* d
        φ = lse_dual(p, f; t=t)
        α = 0.9995
        slope = -dot(g, q)
        for _ in 1:30
            p_new = p .- α .* q
            if lse_dual(p_new, f; t=t) <= φ + 1e-4 * α * slope
                break
            end
            α *= 0.5
        end
        p .-= α .* q

        if show_trace && k % show_every == 0
            @printf("k=%3d  φ=%.8e  |∇φ|=%.2e  α=%.2e  σ=%.2e\n", k, φ, gnorm, α, σ)
        end
    end

    return (p=p, traj=traj, σ=σ)
end

"""
    lse_dual_pfh(f; t, p₀, μ₀, μ_factor, hessian, p_order, maxiter, gtol, ...)

Minimize φᵗ(p) using a path-following homotopy (PFH) Newton method.
Solves a sequence of regularized subproblems (H̃ + μI)d = P∇φ with
decreasing μ, adapted from DRSOM.jl's PFH algorithm with step=:newton.

hessian modes:
  :direct  — dense Hessian + Cholesky (default)
  :vncg    — matrix-free CG with Von Neumann series preconditioner (p_order terms)

μ schedule: when the homotopy residual ‖P∇φ - μ·1‖ < min(0.5, 10μ)
or inner iterations exceed 10, set μ ← μ_factor · μ.

Returns (p, traj, μ).
"""
function lse_dual_pfh(
    f;
    t::Float64=1e-1,
    p₀::Union{Vector{Float64},Nothing}=nothing,
    μ₀::Float64=5e-2,
    μ_factor::Float64=0.02,
    hessian::Symbol=:direct,
    p_order::Int=4,
    maxiter::Int=500,
    gtol::Float64=5e-7,
    show_trace::Bool=true,
    show_every::Int=2
)
    n = f.n
    p = p₀ === nothing ? ones(n) * sum(f.w) / n : copy(p₀)

    H = hessian == :direct ? zeros(n, n) : nothing
    g = zeros(n)
    Pr = zeros(n)
    Hd = zeros(n)
    invP = diagm(1 ./ p)
    μ = μ₀
    inner_k = 0
    traj = Float64[]

    for k in 1:maxiter
        # gradient and affine-scaled gradient
        lse_grad!(g, p, f; t=t)

        # gf = p .* g - μ .* p
        gf = p .* g .- μ
        # gf = g + μ .* p
        gnorm = norm(gf, Inf)
        push!(traj, gnorm)

        # Hf = invP .* H .* invP + μ * I
        Hf = H + μ * I
        if gnorm < gtol
            show_trace && @printf("converged at k=%d, |∇φ|=%.2e, μ=%.2e\n", k, gnorm, μ)
            break
        end

        if hessian == :direct
            # dense Hessian + Cholesky
            lse_hess!(H, p, f; t=t)
            F = cholesky(Symmetric(Hf); check=false)
            d = issuccess(F) ? F \ gf : Symmetric(Hf) \ gf
        else  # :vncg — CG with Von Neumann preconditioner
            # operator for (H̃ + μI)v via matrix-free matvec
            hessop = LinearOperator(Float64, n, n, true, true,
                (buf, v) -> (lse_hess_matvec!(buf, v, p, f; t=t); buf .+= μ .* v)
            )
            # Von Neumann preconditioner: M_p⁻¹ r ≈ (H̃ + μI)⁻¹ r
            # Ĥ = diag(Λ - Σ), Ĥ_μ = diag(Λ - Σ + μ)
            (; Λ, Σ) = lse_hess_dr1(p, f; t=t, bool_diag_only=true)
            Hhat_μ_inv = 1 ./ (Λ .- Σ .+ μ)
            M = LinearOperator(Float64, n, n, true, true,
                function (buf, r)
                    buf .= Hhat_μ_inv .* r
                    for _ in 1:p_order
                        lse_hess_matvec!(Hd, buf, p, f; t=t)
                        Hd .+= μ .* buf
                        buf .= 2 .* buf .- Hhat_μ_inv .* Hd
                    end
                end
            )
            d, _ = cg(hessop, gf; M=M, rtol=1e-6, atol=1e-10, itmax=50)
        end

        # affine-scaled step: q = Pd, then p₊ = p - α q
        # d ./= norm(d)
        q = p .* d
        φ = lse_dual(p, f; t=t)
        α = 0.9995
        slope = -dot(g, q)
        for _ in 1:30
            p_new = p .- α .* q
            if all(p_new .> 0) && lse_dual(p_new, f; t=t) <= φ + 1e-4 * α * slope
                break
            end
            α *= 0.5
        end
        p .-= α .* q
        inner_k += 1

        # homotopy residual: ‖gf‖
        ϵ_μ = norm(gf)

        if ϵ_μ < min(0.5, 10 * μ) || inner_k > 15
            μ = μ_factor * μ
            inner_k = 0
        end

        if show_trace && k % show_every == 0
            @printf("k=%3d  φ=%.8e  |∇φ|=%.2e  α=%.2e  μ=%.2e\n", k, φ, gnorm, α, μ)
        end
    end

    return (p=p, traj=traj, μ=μ)
end
