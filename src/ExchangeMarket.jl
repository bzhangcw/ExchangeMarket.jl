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
__default_sep = repeat("-", 60)

include("utils.jl")

# models
include("models/tools.jl")
include("models/constrs.jl")
include("models/fisher.jl")


# algorithms
include("algorithms/algbase.jl")
include("algorithms/diff.jl")
include("algorithms/play.jl")
include("algorithms/response.jl")
include("algorithms/response_ces.jl")
include("algorithms/response_ces_af.jl")
include("algorithms/response_nlp.jl")
include("algorithms/primals.jl")
include("algorithms/sampler.jl")

# linear systems, solving all kinds of update
#   using ∇²
include("linsys/linsys.jl")

# main algorithms
include("algorithms/conic.jl")
include("algorithms/hessianbar.jl")
include("algorithms/mirror.jl")

export LOGDIR, RESULTSDIR
export pprint
export logbar, log_to_expcone!, powerp_to_cone!, proj, extract_standard_form
export FisherMarket, validate
export create_primal_linear, create_dual_linear
export create_primal_ces, create_dual_ces
export eval!, grad!, hess!, iterate!, play!, opt!
export Conic
export HessianBar
export MirrorDec

export ResponseInfo, solve!, solve_substep!, produce_functions_from_subproblem
export CESConic, DualCESConic, CESAnalytic
export PR # proportional response
export ONR
# affine-constrained response
export AFCESConic

# DRq: Diagonal + Rank-q
export SMWDRq, smw_drq!, update_cluster_map!

# randomization utilities
export NullSampler, BatchSampler

# primal methods
export boxed_allocation!

# constraints
export LinearConstr

end
# end of module `ExchangeMarket`
