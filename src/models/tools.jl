# -----------------------------------------------------------------------
# modeling tools for Fisher Market
# @author:Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
using LinearAlgebra, SparseArrays, Random
using JuMP, MosekTools, MadNLP
# using COPT
import MathOptInterface as MOI

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