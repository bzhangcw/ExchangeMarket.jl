# -----------------------------------------------------------------------
# modeling tools for Fisher Market
# @author:Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
using LinearAlgebra, SparseArrays, Random
using JuMP, MosekTools, MadNLP
# using COPT
import MathOptInterface as MOI

@doc """
    __validate(fisher::Market)
    -----------------------------------------------------------------------
    validate the equilibrium of the Market.
    use the price attached in the FisherMarket if no alg is provided.
    inner use only.
"""
function __validate(market::Market)
    validate(market, nothing)
end

function validate(market::Market, alg)
    m = market.m
    n = market.n
    u = market.u
    x = market.x
    p = isnothing(alg) ? market.p : alg.p
    μ = isnothing(alg) ? 0.0 : alg.μ
    w = market.w
    if isa(market, ArrowDebreuMarket)
        @printf(" :current budget correctly updated? [%.4e]\n", norm(w - market.b' * p))
    end

    market.df = df = DataFrame(
        :utility => market.val_u,
        :left_budget => w - x' * p,
    )
    println(__default_sep)
    @printf(" :problem size\n")

    @printf(" :    number of agents: %d\n", market.m)
    @printf(" :    number of goods: %d\n", market.n)
    @printf(" :    avg number of nonzero entries in c: %.4f\n",
        length(sparse(market.c).nzval) / (market.m * market.n)
    )
    @printf(" :equilibrium information\n")
    @printf(" :method: %s\n", alg.name)
    println(__default_sep)
    println(first(df, 10))
    println(__default_sep)
    _excess = (sum(market.x; dims=2)[:] - market.q) ./ maximum(market.q)
    @printf(" :(normalized) market excess: [%.4e, %.4e]\n", minimum(_excess), maximum(_excess))
    @printf(" :            social welfare:  %.8e\n", (log.(market.val_u))' * market.w)
    println(__default_sep)
end



@doc raw"""
    generate an empty JuMP model with the specified optimizer
    bool_nlp: whether to use NLP solver, default false, use a linear conic solver.
"""
function __generate_empty_jump_model(; bool_nlp=false, verbose=false, tol=1e-7)
    if bool_nlp
        return Model(
            optimizer_with_attributes(
                () -> MadNLP.Optimizer(),
                "tol" => tol,
            )
        )
        # elseif __default_jump_solver == :copt
        #     return Model(
        #         optimizer_with_attributes(
        #             () -> COPT.ConeOptimizer(),
        #             "LogToConsole" => verbose,
        #             "FeasTol" => tol,
        #             "DualTol" => tol
        #         )
        #     )
    else
        md = Model(MosekTools.Optimizer)
        set_attribute(md, "MSK_IPAR_LOG", verbose ? 10 : 0)
        set_attribute(md, "MSK_DPAR_INTPNT_CO_TOL_MU_RED", tol)
        set_attribute(md, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", tol)
        return md
    end
end

@doc raw"""
    translate log utility to exponential cone
    v <= log(x) ==> [v, 1, x] in MOI.ExponentialCone()
"""
log_to_expcone!(x, logx, model) = @constraint(
    model, [logx, 1, x] in MOI.ExponentialCone()
)

@doc raw"""
    translate power to power cone 
    t = |x|^p
    if p ≥ 1
        t ≥ |x|^p ⇒ [t, 1, x] in MOI.PowerCone(1/p)
    for 0 < p < 1
        t ≤ |x|^p ⇒ [x, 1, t] in MOI.PowerCone(p)
    end
"""
function powerp_to_cone(t, x, model, p::Float64)
    if (p > 1)
        @constraint(
            model, [v, 1, x] in MOI.PowerCone(1 / p)
        )
    elseif (0 < p < 1)
        @constraint(
            model, [x, 1, t] in MOI.PowerCone(p)
        )
    elseif (p < 0)
        @constraint(
            model, [t, x, 1] in MOI.PowerCone(1 / (1 - p))
        )
    else
        @warn "unsupported power $p"
    end
end

function test_powerp_to_cone()
    model = __generate_empty_jump_model()
    @variable(model, t)
    @variable(model, x)
    powerp_to_cone(t, x, model, 0.5)
    set_lower_bound(x, 4.0)
    set_upper_bound(x, 4.0)
    @objective(model, Max, t)
    optimize!(model)
    @show value(t) value(x)
end


function drop_empty(c::SparseMatrixCSC)
    # sum over rows/cols to see which have any nonzeros
    row_sums = sum(abs.(c), dims=2)[:]    # Vector of length m
    col_sums = sum(abs.(c), dims=1)[1, :]  # Vector of length n

    rows = findall(!iszero, row_sums)
    cols = findall(!iszero, col_sums)
    return c[rows, cols], rows, cols
end


@doc raw"""
    extract_standard_form(md::Model)

Given a JuMP LP `md`, returns `(A, b, c)` such that  
  maximize cᵀ x  
  subject to A x = b, x ≥ 0,  
allowing both `≤` and `=` constraints in `md`.
"""
function extract_standard_form(md::Model)
    backend = JuMP.backend(md)
    n = MOI.get(backend, MOI.NumberOfVariables())

    # objective
    sense = MOI.get(backend, MOI.ObjectiveSense())
    obj = MOI.get(backend, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    c = zeros(n)
    for term in obj.terms
        c[term.variable.value] = term.coefficient
    end
    if sense == MOI.MAX_SENSE
        c .*= -1
    end

    # ≤‐constraints
    F, S = MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}
    ineq = MOI.get(backend, MOI.ListOfConstraintIndices{F,S}())
    m_ineq = length(ineq)
    rows₁ = Int[]
    cols₁ = Int[]
    vals₁ = Float64[]
    b_ineq = zeros(m_ineq)
    for (i, ci) in enumerate(ineq)
        f = MOI.get(backend, MOI.ConstraintFunction(), ci)
        for term in f.terms
            push!(rows₁, i)
            push!(cols₁, term.variable.value)
            push!(vals₁, term.coefficient)
        end
        b_ineq[i] = MOI.get(backend, MOI.ConstraintSet(), ci).upper
    end
    A_ineq = sparse(rows₁, cols₁, vals₁, m_ineq, n)

    # =‐constraints
    E = MOI.EqualTo{Float64}
    eq = MOI.get(backend, MOI.ListOfConstraintIndices{F,E}())
    m_eq = length(eq)
    rows₂ = Int[]
    cols₂ = Int[]
    vals₂ = Float64[]
    b_eq = zeros(m_eq)
    for (i, ci) in enumerate(eq)
        f = MOI.get(backend, MOI.ConstraintFunction(), ci)
        for term in f.terms
            push!(rows₂, i)
            push!(cols₂, term.variable.value)
            push!(vals₂, term.coefficient)
        end
        b_eq[i] = MOI.get(backend, MOI.ConstraintSet(), ci).value
    end
    A_eq0 = sparse(rows₂, cols₂, vals₂, m_eq, n)

    # add slack for ≤ → =
    S_slack = spdiagm(0 => ones(m_ineq))
    top = hcat(A_eq0, spzeros(Float64, m_eq, m_ineq))
    bottom = hcat(A_ineq, S_slack)
    A_full = vcat(top, bottom)
    b_full = vcat(b_eq, b_ineq)
    c_full = vcat(c, zeros(m_ineq))

    return A_full, b_full, c_full
end