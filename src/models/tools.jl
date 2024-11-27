# -----------------------------------------------------------------------
# modeling tools for Fisher Market
# @author:Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------
using LinearAlgebra, SparseArrays
using JuMP, COPT, MosekTools, Random
import MathOptInterface as MOI
@doc raw"""
"""
function __generate_empty_jump_model(; verbose=false, tol=1e-8)
    if __default_jump_solver == :copt
        return Model(
            optimizer_with_attributes(
                () -> COPT.ConeOptimizer(),
                "LogToConsole" => verbose,
                "FeasTol" => tol,
                "DualTol" => tol
            )
        )
    else
        md = Model(MosekTools.Optimizer)
        set_attribute(md, "MSK_IPAR_LOG", verbose ? 10 : 0)
        # set_attribute(md, "MSK_DPAR_INTPNT_CO_TOL_INFEAS", tol)
        # set_attribute(md, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", tol)
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
