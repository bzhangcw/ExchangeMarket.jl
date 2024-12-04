# -----------------------------------------------------------------------
# A Package for Competitive Exchange Market
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------

module ExchangeMarket

greet() = print("Competitive Exchange Market!")

# -----------------------------------------------------------------------
# default settings
# -----------------------------------------------------------------------
__default_jump_solver = :mosek
__default_sep = repeat("-", 100)

include("utils.jl")

# models
include("models/tools.jl")
include("models/fisher.jl")

# algorithms
include("algorithms/diff.jl")
include("algorithms/conic.jl")
include("algorithms/response.jl")
include("algorithms/response_eg.jl")
include("algorithms/response_nlp.jl")
include("algorithms/hessianbar.jl")


export logbar, log_to_expcone!, powerp_to_cone!
export FisherMarket, validate
export Conic
export create_primal_linear, create_dual_linear
export create_primal_ces, create_dual_ces_type_i, create_dual_ces_type_ii
export HessianBar, solve!
export eval!, grad!, hess!, iterate!, play!

export ResponseInfo, solve_substep!, produce_functions_from_subproblem
export BR
export EGConic, EGConicCES, EGConicCESTypeI, EGConicAC
export NR, ONR

end
# end of module `ExchangeMarket`
