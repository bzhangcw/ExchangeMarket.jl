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
__generate_empty_jump_model(; verbose=false) = begin
    __default_jump_solver == :copt ? Model(
        optimizer_with_attributes(
            () -> COPT.ConeOptimizer(),
            "LogToConsole" => verbose
        )
    ) : Model(
        optimizer_with_attributes(
            () -> MosekTools.Optimizer(),
            "MSK_IPAR_LOG" => verbose ? 10 : 0
        )
    )
end

@doc raw"""
    translate log utility to exponential cone
    v <= log(x) ==> [v, 1, x] in MOI.ExponentialCone()
"""
log_to_expcone!(x, v, model) = @constraint(
    model, [v, 1, x] in MOI.ExponentialCone()
)
