using COPT

# get CES demand function x from linear-conic optimization
function _conic_ces_primal(;
    p::Vector{T}=nothing,
    n,
    cr,
    w,
    ρ,
    verbose=false
) where {T}
    md = ExchangeMarket.__generate_empty_jump_model(; verbose=verbose, tol=1e-8)
    @variable(md, u)
    @variable(md, logu)
    log_to_expcone!(u, logu, md)

    @variable(md, x[1:n] >= 0)
    @variable(md, ξ[1:n] >= 0)
    # budget constraint
    @constraint(md, budget, p' * x <= w)
    # utility constraint
    # Δ^{ρ} ξ^{1-ρ}≥ r 
    # ⇒ [Δ,ξ,r] ∈ P₃(ρ) [power cone]
    @constraint(md, sum(ξ) == u)
    @constraint(
        md,
        ξc[j=1:n],
        [cr[j] * x[j], u, ξ[j]] in MOI.PowerCone(ρ)
    )
    @objective(md, Max, logu)

    JuMP.optimize!(md)
    # ensure non-negativity
    x = max.(value.(x), 0.0)
    return x
end

@doc raw"""
given a CES bidding vector γ, produce the CES coefficients (δ, c)
    from linear-programming optimization
    if there are more than one bidding vector, 
    this may not be tight.
"""
function _linear_prog_ces_gamma_single(;
    pmat::Union{SparseMatrixCSC{T},Matrix{T}}=nothing,
    gmat::Union{SparseMatrixCSC{T},Matrix{T}}=nothing,
    δ₁::Union{Float64,Nothing}=nothing,
    verbose=false
) where {T}
    md = ExchangeMarket.__generate_empty_jump_model(; verbose=verbose, tol=1e-8)
    # md = Model(
    #     optimizer_with_attributes(
    #         () -> COPT.Optimizer(),
    #     )
    # )

    n, K = size(pmat)
    @variable(md, y[1:n])
    @variable(md, A[1:K])
    @variable(md, r[1:n, 1:K])
    @variable(md, rmax >= 0)
    @variable(md, δ >= -1)

    if !isnothing(δ₁)
        @constraint(md, δ == δ₁)
    end
    @constraint(md,
        fitc[k=1:K],
        r[:, k] .+ y .- δ .* log.(pmat[:, k]) .- A[k] .== log.(gmat[:, k])
    )
    @constraint(md, rmax .>= r)
    @constraint(md, rmax .>= -r)

    # anchoring, because non-uniqueness
    @objective(md, Min, rmax)

    JuMP.optimize!(md)
    println(termination_status(md))
    return value.(y), value.(δ), value.(A), md
end

