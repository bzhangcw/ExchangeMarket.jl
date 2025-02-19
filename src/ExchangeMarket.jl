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

# constraints
include("models/constrs.jl")

# algorithms
include("algorithms/algbase.jl")
include("algorithms/diff.jl")
include("algorithms/play.jl")
include("algorithms/response.jl")
include("algorithms/response_eg.jl")
include("algorithms/response_nlp.jl")
include("algorithms/primals.jl")
include("algorithms/linsys.jl")
include("algorithms/sampler.jl")

# main algorithms
include("algorithms/conic.jl")
include("algorithms/hessianbar.jl")
include("algorithms/mirror.jl")


export pprint
export logbar, log_to_expcone!, powerp_to_cone!, proj
export FisherMarket, validate
export create_primal_linear, create_dual_linear
export create_primal_ces, create_dual_ces
export eval!, grad!, hess!, iterate!, play!, opt!
export Conic
export HessianBar
export MirrorDec

export ResponseInfo, solve!, solve_substep!, produce_functions_from_subproblem
export BR
export PR # proportional response
export EGConic, EGConicCES, EGConicCESTypeI, EGConicAC
export ONR

# DR1: Diagonal + Rank-One
export SMWDR1, smw_dr1!

# randomization utilities
export NullSampler, BatchSampler

# primal methods
export boxed_allocation!

# constraints
export LinearConstr

end
# end of module `ExchangeMarket`
