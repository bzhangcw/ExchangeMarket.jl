# NN-parameterized android class (homothetic).
#
#   γ_θ(p) = softmax( W₂ · tanh(W₁ · log(p) + b₁) + b₂ )
#
# Naturally homothetic: γ depends on p only (no `w` input), so the same
# polytope-and-LP-master plumbing the CES / linear / Leontief classes use
# applies unchanged. The separation problem at iteration T is the usual
#
#     π_{T+1} = max_θ  Σ_k ⟨u_k, γ_θ(p_k)⟩
#
# solved here by Optim.jl + ForwardDiff (smooth, ~10²–10³ params; LBFGS
# converges in well under a second for the K, n sizes we use).
#
# ---- v1 / MVP scope ------------------------------------------------------
# Only the LP master + outer CG primal are NN-correct. `add_column_to_market!`
# stores a PLACEHOLDER CES atom in the FisherMarket so `fa.m` stays in sync
# with `γ_ref[]`, but `evaluate_test_error` walks `fa.c, fa.σ` and so will
# evaluate the placeholder (uniform CES), not the actual NN. That means:
#
#   ✓ primal_obj (training fit) — correct, useful for A/B vs CES separation
#   ✗ test_err (held-out fit)   — wrong (NN atoms evaluated as uniform CES)
#
# A v2 side-table that registers θ keyed by atom index, plus a hook in
# evaluate_test_error to use nn_share(θ, p_test) when present, would fix
# this. Skipped for now to keep the MVP small.

using LinearAlgebra
using Optim
using LogExpFunctions: logsumexp
using ArgParse

# Homothetic by construction: γ_θ depends on p only, no `w` input (see file header).
is_homothetic(::Val{:nn}) = true

# Default MLP architecture. Small on purpose: the polytope it has to push
# the LP master through is bounded by ‖u_k‖_1 ≤ 1, so high capacity per
# atom is overkill — more atoms each with modest expressiveness wins.
const NN_HIDDEN_DEFAULT = 16
const NN_ITERS_DEFAULT  = 200

# Unpack a flat θ into the four MLP parameter blocks. Layout:
#   W₁ ∈ ℝ^{H×n}, b₁ ∈ ℝ^H, W₂ ∈ ℝ^{n×H}, b₂ ∈ ℝ^n
# Total length = nH + H + nH + n = 2nH + n + H.
@inline _nn_dim(n::Int, H::Int) = 2 * n * H + n + H

function _nn_unpack(θ::AbstractVector, n::Int, H::Int)
    @assert length(θ) == _nn_dim(n, H) "θ has wrong size for (n=$n, H=$H)"
    i = 0
    W1 = reshape(view(θ, i+1:i+H*n), H, n);  i += H * n
    b1 = view(θ, i+1:i+H);                   i += H
    W2 = reshape(view(θ, i+1:i+n*H), n, H);  i += n * H
    b2 = view(θ, i+1:i+n)
    return W1, b1, W2, b2
end

"""
    nn_share(θ, p; hidden=NN_HIDDEN_DEFAULT)

Evaluate the homothetic NN share function γ_θ(p) ∈ Δⁿ at price `p`.
"""
function nn_share(θ::AbstractVector{T}, p::AbstractVector;
    hidden::Int=NN_HIDDEN_DEFAULT) where T
    n = length(p)
    W1, b1, W2, b2 = _nn_unpack(θ, n, hidden)
    h = tanh.(W1 * log.(p) .+ b1)
    z = W2 * h .+ b2
    return exp.(z .- logsumexp(z))
end

# Initialize θ from a small Gaussian. Width comes from He-init for tanh.
function _nn_init(n::Int, H::Int; seed::Union{Int,Nothing}=nothing)
    !isnothing(seed) && Random.seed!(seed)
    σ1 = sqrt(2.0 / n)
    σ2 = sqrt(2.0 / H)
    W1 = σ1 .* randn(H, n)
    b1 = zeros(H)
    W2 = σ2 .* randn(n, H)
    b2 = zeros(n)
    return vcat(vec(W1), b1, vec(W2), b2)
end

"""
    solve_separation_nn(Ξ, u; hidden, max_iters, θ_init, verbose) -> NamedTuple

Separation oracle for the NN-android class. Maximizes

    Σ_k ⟨u_k, γ_θ(p_k)⟩

over θ via Optim.jl's LBFGS with forward-mode autodiff. Returns a
NamedTuple compatible with the per-class separation oracle:

    (γ_new::Matrix{T} of shape (K, n), params=(θ=θ_opt, hidden=H), obj, class=:nn).

Keyword arguments:
- `hidden::Int = NN_HIDDEN_DEFAULT` — MLP hidden width.
- `max_iters::Int = NN_ITERS_DEFAULT` — Optim outer iteration cap.
- `θ_init::Union{Vector,Nothing} = nothing` — warm start (default He init).
- `verbose::Bool = false`, `timelimit::Union{Real,Nothing} = nothing`.
"""
function solve_separation_nn(Ξ::Vector{Tuple{Vector{T},Vector{T}}},
    u::Matrix{T};
    hidden::Int=NN_HIDDEN_DEFAULT,
    max_iters::Int=NN_ITERS_DEFAULT,
    θ_init::Union{Vector,Nothing}=nothing,
    verbose::Bool=false,
    timelimit::Union{Real,Nothing}=nothing,
    kwargs...) where T

    K = length(Ξ)
    n = length(Ξ[1][1])
    log_p = [log.(Ξ[k][1]) for k in 1:K]

    function neg_obj(θ)
        W1, b1, W2, b2 = _nn_unpack(θ, n, hidden)
        s = zero(eltype(θ))
        @inbounds for k in 1:K
            h = tanh.(W1 * log_p[k] .+ b1)
            z = W2 * h .+ b2
            γ_k = exp.(z .- logsumexp(z))
            s += dot(u[k, :], γ_k)
        end
        return -s
    end

    θ0 = isnothing(θ_init) ? _nn_init(n, hidden) : Vector{Float64}(θ_init)
    _tlim = isnothing(timelimit) || timelimit <= 0 ? NamedTuple() :
            (time_limit=Float64(timelimit),)
    result = optimize(
        neg_obj, θ0,
        LBFGS(; m=10),
        Optim.Options(; show_trace=verbose, iterations=max_iters,
                        g_tol=1e-6, _tlim...);
        autodiff=:forward,
    )

    θ_opt = Optim.minimizer(result)
    obj   = -Optim.minimum(result)
    # Build the K × n γ matrix that the LP master / γ_ref consumes.
    γ_new = Matrix{Float64}(undef, K, n)
    for k in 1:K
        γ_new[k, :] .= nn_share(θ_opt, Ξ[k][1]; hidden=hidden)
    end
    verbose && println("NN separation (H=$hidden): obj=$obj, |θ|=$(length(θ_opt))")
    return (γ_new=γ_new, params=(θ=θ_opt, hidden=hidden), obj=Float64(obj), class=:nn)
end

# ---- CLI surface --------------------------------------------------------
"""
    register_cli_nn!(s::ArgParseSettings)

Add the "Separation: NN" arg group (`--sep-nn-hidden`, `--sep-nn-iters`).
"""
function register_cli_nn!(s::ArgParseSettings)
    add_arg_group!(s, "Separation: NN")
    @add_arg_table! s begin
        "--sep-nn-hidden"
        help = "Hidden width H of the per-android MLP used by `:nn` separation. Total θ dimension is 2nH + n + H. Default 16."
        arg_type = Int
        default = NN_HIDDEN_DEFAULT
        "--sep-nn-iters"
        help = "Outer LBFGS iteration cap for the NN separation subproblem. Each call is autodiffed (ForwardDiff) and warm-started from He init."
        arg_type = Int
        default = NN_ITERS_DEFAULT
    end
    return s
end

"""
    apply_cli_nn!(local_extra::Dict, cli)

Forward NN-separation CLI values into the runner kwargs.
"""
function apply_cli_nn!(local_extra::Dict, cli)
    if cli["sep_nn_hidden"] != NN_HIDDEN_DEFAULT
        local_extra[:nn_hidden] = cli["sep_nn_hidden"]
    end
    if cli["sep_nn_iters"] != NN_ITERS_DEFAULT
        local_extra[:nn_iters] = cli["sep_nn_iters"]
    end
    return local_extra
end
