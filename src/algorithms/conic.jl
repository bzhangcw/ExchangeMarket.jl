# -----------------------------------------------------------------------
# directly run market problem
#   using induced utility function from Eigenberg-Gale-type potentials
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------

Base.@kwdef mutable struct Conic
    n::Int
    m::Int
    # price
    p::Vector{Float64}  # current price at k
    λ::Vector{Float64}  # current dual variable of p at k (if needed)

    model::Model

    name::Symbol
    # barrier parameter
    μ::Float64 = 0.0

    function Conic(
        n::Int, m::Int;
        tol=1e-12
    )
        this = new()
        this.name = :Conic
        this.n = n
        this.m = m
        this.p = zeros(n)
        this.λ = zeros(n)
        this.model = __generate_empty_jump_model(; verbose=true, tol=tol)
        return this
    end
end


function __create_primal(alg::Conic, fisher::FisherMarket)
    model = alg.model
    m = fisher.m
    n = fisher.n
    u = fisher.u
    q = fisher.q
    w = fisher.w
    @variable(model, x[1:m, 1:n] >= 0)
    @variable(model, v[1:m])
    @variable(model, ℓ[1:m])
    @constraint(model, limit, x' * ones(m) .<= q)
    for i in 1:m
        @constraint(model, ℓ[i] == u(x[i, :], i))
        log_to_expcone!(ℓ[i], v[i], model)
    end
    @objective(model, Min, -sum([w[i] * v[i] for i in 1:m]))
    # -----------------------------------------------------------------------
    # optimize
    JuMP.optimize!(model)
    # price saved in alg
    alg.p = -dual.(model[:limit])
    # allocation saved in fisher
    fisher.x = value.(model[:x])
    fisher.val_u = value.(model[:ℓ])
end

function __create_dual(alg::Conic, fisher::FisherMarket)
    model = alg.model
    @variable(model, s[1:fisher.m, 1:fisher.n] .>= 0)
    @variable(model, p[1:fisher.n])
    @variable(model, λ[1:fisher.m])
    @variable(model, logλ[1:fisher.m])
    log_to_expcone!.(λ, logλ, model)
    @objective(model, Min, p' * fisher.q - sum([fisher.w[i] * logλ[i] for i in 1:fisher.m]))

    @constraint(model, xc[i=1:fisher.m], s[i, :] + λ[i] * fisher.c[i, :] - p .== 0)
    JuMP.optimize!(model)
    alg.p = value.(p)
    fisher.x = hcat([abs.(dual.(xc[i])) for i in 1:fisher.m]...)'
    fisher.val_u = map(i -> fisher.u(fisher.x[i, :], i), 1:fisher.m)
    return
end

function create_primal_linear(alg::Conic, fisher::FisherMarket)
    __create_primal(alg, fisher)
end

function create_dual_linear(alg::Conic, fisher::FisherMarket)
    __create_dual(alg, fisher)
end

function create_primal_ces(alg::Conic, fisher::FisherMarket, ρ::Float64=0.5)
    model = alg.model
    m = fisher.m
    n = fisher.n
    q = fisher.q
    w = fisher.w
    @variable(model, x[1:m, 1:n] >= 0)
    @variable(model, xp[1:m, 1:n] >= 0)
    # xp = x^ρ
    powerp_to_cone.(xp, x, model, ρ)

    @variable(model, v[1:m])
    @variable(model, ℓ[1:m])
    @constraint(model, limit, x' * ones(m) .<= q)
    for i in 1:m
        @constraint(model, ℓ[i] == sum(fisher.c[i, :] .* xp[i, :]))
        log_to_expcone!(ℓ[i], v[i], model)
    end
    @objective(model, Min, -sum([w[i] * v[i] for i in 1:m]) / ρ) - fisher.w' * log.(fisher.w)
    # -----------------------------------------------------------------------
    # optimize
    JuMP.optimize!(model)
    # price saved in alg
    alg.p = -dual.(model[:limit])
    # allocation saved in fisher
    fisher.x = value.(model[:x])
    fisher.val_u = value.(model[:ℓ] .^ (1 / ρ))
end

function create_dual_ces(alg::Conic, fisher::FisherMarket, ρ::Float64=0.5, bool_solve_p=true)
    create_dual_ces_type_i(alg, fisher, ρ, bool_solve_p)
end

function create_dual_ces_type_i(alg::Conic, fisher::FisherMarket, ρ::Float64=0.5, bool_solve_p=true)
    model = alg.model
    @variable(model, p[1:fisher.n] .>= 0)
    if !bool_solve_p
        @info "fix price p"
        set_lower_bound.(p, alg.p)
        set_upper_bound.(p, alg.p)
    end
    @variable(model, λ[1:fisher.m])
    @variable(model, logλ[1:fisher.m])
    log_to_expcone!.(λ, logλ, model)

    # Δ^{ρ} ξ^{1-ρ}≥ r 
    # ⇒ [Δ,ξ,r] ∈ P₃(ρ) [power cone]
    @variable(model, ξ[1:fisher.m, 1:fisher.n])
    @constraint(
        model,
        λc[i=1:fisher.m],
        sum(ξ[i, :]) <= 1
    )
    @constraint(
        model,
        ξc[i=1:fisher.m, j=1:fisher.n],
        [p[j], ξ[i, j], fisher.c[i, j] * λ[i]] in MOI.PowerCone(ρ)
    )
    @objective(model, Min,
        p' * fisher.q -
        1 / ρ * sum([fisher.w[i] * logλ[i] for i in 1:fisher.m]) +
        fisher.w' * log.(fisher.w)
    )

    JuMP.optimize!(model)
    alg.p = value.(p)
    fisher.x = first.(dual.(alg.model[:ξc]))
    fisher.val_u = map(i -> sum(fisher.c[i, :] .* (fisher.x[i, :] .^ ρ))^(1 / ρ), 1:fisher.m)
    return
end

# --------------------------------------------------------------------------
# solve the dual problem of CES EG program
# this is formulation II (non-standard form)
# I did this via conjugate dual, please see type I, which is more elegant
# --------------------------------------------------------------------------
function create_dual_ces_type_ii(alg::Conic, fisher::FisherMarket, ρ::Float64=0.5)
    model = alg.model
    @variable(model, s[1:fisher.m, 1:fisher.n] .>= 0)
    @variable(model, p[1:fisher.n] .>= 0)
    @variable(model, Δ[1:fisher.m, 1:fisher.n])
    # Δ_{ij} = p_j - s_{ij}
    @constraint(model, Δc[i=1:fisher.m, j=1:fisher.n], Δ[i, j] == p[j] - s[i, j])
    @variable(model, λ[1:fisher.m])
    @variable(model, logλ[1:fisher.m])
    log_to_expcone!.(λ, logλ, model)
    # r_{ij} = λ_i * ρ * c_{ij}
    @variable(model, r[1:fisher.m, 1:fisher.n])
    @constraint(model, rc[i=1:fisher.m, j=1:fisher.n], r[i, j] == λ[i] * ρ * fisher.c[i, j])

    # Δ^{ρ} ξ^{1-ρ}≥ r 
    # ⇒ [Δ,ξ,r] ∈ P₃(ρ) [power cone]
    @variable(model, ξ[1:fisher.m, 1:fisher.n])
    @constraint(
        model,
        ξc[i=1:fisher.m, j=1:fisher.n],
        [Δ[i, j], ξ[i, j], r[i, j]] in MOI.PowerCone(ρ)
    )
    @objective(model, Min,
        p' * fisher.q -
        1 / ρ * sum([fisher.w[i] * logλ[i] for i in 1:fisher.m]) +
        (1 - ρ) / ρ * sum(ξ)
    )

    JuMP.optimize!(model)
    alg.p = value.(p)
    fisher.x = first.(dual.(alg.model[:ξc]))
    fisher.val_u = map(i -> sum(fisher.c[i, :] .* (fisher.x[i, :] .^ ρ))^(1 / ρ), 1:fisher.m)
    return
end

