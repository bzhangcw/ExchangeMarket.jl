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
abstract type AbstractMarket end

include("models/tools.jl")
include("models/constrs.jl")
include("linalg/sparse.jl")
include("models/fisher.jl")
include("models/arrow.jl")

const Market = AbstractMarket
include("models/agent_view.jl")
include("models/validate.jl")

# algorithms
include("algorithms/algbase.jl")
include("algorithms/diff/diff.jl")
include("algorithms/diff/diff_lse.jl")
include("algorithms/diff/diff_afcon.jl")
include("algorithms/diff/diff_arrow.jl")
include("algorithms/play.jl")
include("algorithms/response/response.jl")
include("algorithms/response/response_ces.jl")
include("algorithms/response/response_ces_af.jl")
include("algorithms/response/response_nlp.jl")
include("algorithms/response/response_bids.jl")
include("algorithms/response/response_approx_lin.jl")
include("algorithms/response/response_dual_lp.jl")
include("algorithms/response/response_lse.jl")
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
export AbstractMarket, Market, FisherMarket, ArrowDebreuMarket, validate, update_budget!, update_supply!, expand_players!

# agent interface
export AgentType, LinearAgent, CESAgent, AgentView, @agent
export SparseColRef, sparse_col_ref
export foreach_nz, sparse_reduce, sparse_scatter!, sparse_argmax
export sparse_dot, sparse_div_max
export utility
export init_agents!
export create_primal_linear, create_dual_linear
export create_primal_ces, create_dual_ces
export eval!, grad!, hess!, iterate!, play!, opt!
export Conic
export HessianBar
export MirrorDec

export solve!, solve_substep!, produce_functions_from_subproblem
export CESConic, DualCESConic, CESAnalytic
export Bids # proportional response
export ApproxLin, ApproxLinConic
export DualLP, DualLPConic
export LSEResponse
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
