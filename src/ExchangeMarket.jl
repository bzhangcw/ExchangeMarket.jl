# -----------------------------------------------------------------------
# A Package for Competitive Exchange Market
# @author:Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# 
# -----------------------------------------------------------------------

module ExchangeMarket

greet() = print("Competitive Exchange Market!")
__default_sep = repeat("-", 100)

include("utils.jl")

# models
include("models/fisher.jl")

# algorithms
include("algorithms/response.jl")
include("algorithms/hessianbar.jl")

export FisherMarket, create_jump_model, solve_jump_model, validate
export logbar
export HessianBar, eval!, grad!, hess!, iterate!, opt!, play!

export ResponseInfo, solve_substep!, produce_functions_from_subproblem, default_newton_response

end # module ExchangeMarket
