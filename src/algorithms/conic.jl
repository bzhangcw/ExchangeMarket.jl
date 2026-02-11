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


function __create_primal(alg::Conic, market::FisherMarket)
    model = alg.model
    m = market.m
    n = market.n
    u = market.u
    q = market.q
    w = market.w
    @variable(model, x[1:m, 1:n] >= 0)
    @variable(model, v[1:m])
    @variable(model, ℓ[1:m])
    @constraint(model, limit, x' * ones(m) .<= q)
    for i in 1:m
        @constraint(model, ℓ[i] == u(x[:, i], i))
        log_to_expcone!(ℓ[i], v[i], model)
    end
    @objective(model, Min, -sum([w[i] * v[i] for i in 1:m]))
    # -----------------------------------------------------------------------
    # optimize
    JuMP.optimize!(model)
    # price saved in alg
    alg.p = -dual.(model[:limit])
    # allocation saved in market
    market.x = value.(model[:x])
    market.val_u = value.(model[:ℓ])
end

function __create_dual(alg::Conic, market::FisherMarket)
    model = alg.model
    @variable(model, s[1:market.m, 1:market.n] .>= 0)
    @variable(model, p[1:market.n])
    @variable(model, λ[1:market.m])
    @variable(model, logλ[1:market.m])
    log_to_expcone!.(λ, logλ, model)
    @objective(model, Min, p' * market.q - sum([market.w[i] * logλ[i] for i in 1:market.m]))

    @constraint(model, xc[i=1:market.m], s[:, i] + λ[i] * market.c[:, i] - p .== 0)
    JuMP.optimize!(model)
    alg.p = value.(p)
    market.x = hcat([abs.(dual.(xc[i])) for i in 1:market.m]...)'
    market.val_u = map(i -> market.u(market.x[:, i], i), 1:market.m)
    return
end

function create_primal_linear(alg::Conic, market::FisherMarket)
    __create_primal(alg, market)
end

function create_dual_linear(alg::Conic, market::FisherMarket)
    __create_dual(alg, market)
end

function create_primal_ces(alg::Conic, market::FisherMarket, ρ::Float64=0.5)
    model = alg.model
    m = market.m
    n = market.n
    q = market.q
    w = market.w
    @variable(model, x[1:m, 1:n] >= 0)
    @variable(model, xp[1:m, 1:n] >= 0)
    # xp = x^ρ
    powerp_to_cone.(xp, x, model, ρ)

    @variable(model, v[1:m])
    @variable(model, ℓ[1:m])
    @constraint(model, limit, x' * ones(m) .<= q)
    for i in 1:m
        @constraint(model, ℓ[i] == sum(market.c[:, i] .* xp[:, i]))
        log_to_expcone!(ℓ[i], v[i], model)
    end
    @objective(model, Min, -sum([w[i] * v[i] for i in 1:m]) / ρ) - market.w' * log.(market.w)
    # -----------------------------------------------------------------------
    # optimize
    JuMP.optimize!(model)
    # price saved in alg
    alg.p = -dual.(model[:limit])
    # allocation saved in market
    market.x = value.(model[:x])
    market.val_u = value.(model[:ℓ] .^ (1 / ρ))
end

function create_primal_pwl(alg::Conic, market::FisherMarket)  # TODO  
    model = alg.model
    m = market.m
    n = market.n
    q = market.q
    w = market.w

    @variable(model, x[1:n, 1:m] >= 0) # allocation
    @variable(model, z[1:m] >= 0)      # utility
    @variable(model, t[1:m])           # log(utility)

    @objective(model, Max, sum(w[i] * t[i] for i in 1:m))

    @constraint(model, util[i=1:m, l=1:size(market.A_planes,2)], z[i] <= dot(@view(market.A_planes[:, l, i]), x[:, i]) + market.b_planes[l, i])
    @constraint(model, log_util[i=1:m], [t[i], 1, z[i]] in MOI.ExponentialCone())
    @constraint(model, supply, x * ones(m) .<= q)
    # -----------------------------------------------------------------------
    # optimize
    JuMP.optimize!(model)
    # price saved in alg
    alg.p = -dual.(model[:supply])
    # allocation saved in market
    market.x = value.(model[:x])
    market.val_u = value.(model[:z])
    return alg.p
end

function create_dual_pwl(alg::Conic, market::FisherMarket)  # TODO:  p with fixed surrogate price,
    pass # TODO: calculate optimized x
end
function create_dual_ces(alg::Conic, market::FisherMarket, ρ::Float64=0.5, bool_solve_p=true, bool_optimize=true)
    create_dual_ces_type_i(alg, market, ρ, bool_solve_p)
end

function create_dual_ces_type_i(alg::Conic, market::FisherMarket, ρ::Float64=0.5, bool_solve_p=true, bool_optimize=true)
    model = alg.model
    @variable(model, p[1:market.n] .>= 0)
    if !bool_solve_p
        @info "fix price p"
        set_lower_bound.(p, alg.p)
        set_upper_bound.(p, alg.p)
    end
    @variable(model, λ[1:market.m])
    @variable(model, logλ[1:market.m])
    log_to_expcone!.(λ, logλ, model)

    # Δ^{ρ} ξ^{1-ρ}≥ r 
    # ⇒ [Δ,ξ,r] ∈ P₃(ρ) [power cone]
    @variable(model, ξ[1:market.m, 1:market.n])
    @constraint(
        model,
        λc[i=1:market.m],
        sum(ξ[:, i]) <= 1
    )
    @constraint(
        model,
        ξc[i=1:market.m, j=1:market.n],
        [p[j], ξ[i, j], market.c[i, j] * λ[i]] in MOI.PowerCone(ρ)
    )
    @objective(model, Min,
        p' * market.q -
        1 / ρ * sum([market.w[i] * logλ[i] for i in 1:market.m]) +
        market.w' * log.(market.w)
    )
    if bool_optimize
        JuMP.optimize!(model)
        alg.p = value.(p)
        market.x = first.(dual.(alg.model[:ξc]))
        market.val_u = map(i -> sum(market.c[:, i] .* (market.x[:, i] .^ ρ))^(1 / ρ), 1:market.m)
    end
    return
end

# --------------------------------------------------------------------------
# solve the dual problem of CES EG program
# this is formulation II (non-standard form)
# I did this via conjugate dual, please see type I, which is more elegant
# --------------------------------------------------------------------------
function create_dual_ces_type_ii(alg::Conic, market::FisherMarket, ρ::Float64=0.5)
    model = alg.model
    @variable(model, s[1:market.m, 1:market.n] .>= 0)
    @variable(model, p[1:market.n] .>= 0)
    @variable(model, Δ[1:market.m, 1:market.n])
    # Δ_{ij} = p_j - s_{ij}
    @constraint(model, Δc[i=1:market.m, j=1:market.n], Δ[i, j] == p[j] - s[i, j])
    @variable(model, λ[1:market.m])
    @variable(model, logλ[1:market.m])
    log_to_expcone!.(λ, logλ, model)
    # r_{ij} = λ_i * ρ * c_{ij}
    @variable(model, r[1:market.m, 1:market.n])
    @constraint(model, rc[i=1:market.m, j=1:market.n], r[i, j] == λ[i] * ρ * market.c[i, j])

    # Δ^{ρ} ξ^{1-ρ}≥ r 
    # ⇒ [Δ,ξ,r] ∈ P₃(ρ) [power cone]
    @variable(model, ξ[1:market.m, 1:market.n])
    @constraint(
        model,
        ξc[i=1:market.m, j=1:market.n],
        [Δ[i, j], ξ[i, j], r[i, j]] in MOI.PowerCone(ρ)
    )
    @objective(model, Min,
        p' * market.q -
        1 / ρ * sum([market.w[i] * logλ[i] for i in 1:market.m]) +
        (1 - ρ) / ρ * sum(ξ)
    )

    JuMP.optimize!(model)
    alg.p = value.(p)
    market.x = first.(dual.(alg.model[:ξc]))
    market.val_u = map(i -> sum(market.c[:, i] .* (market.x[:, i] .^ ρ))^(1 / ρ), 1:market.m)
    return
end

