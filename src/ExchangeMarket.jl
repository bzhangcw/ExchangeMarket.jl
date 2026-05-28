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
# agent_types.jl defines the AgentType hierarchy; loaded before workspace
# so GenStore can hold a typed Vector{AgentType}.
include("models/agent_types.jl")
include("models/workspace.jl")
include("models/fisher.jl")
include("models/arrow.jl")

const Market = AbstractMarket
include("models/validate.jl")
include("models/agent_view.jl")
include("linalg/cuda.jl")

# algorithms
include("algorithms/algbase.jl")
include("algorithms/diff/diff.jl")
include("algorithms/diff/diff_lse.jl")
include("algorithms/diff/diff_arrow.jl")
include("algorithms/play.jl")
include("algorithms/response/response.jl")
include("algorithms/response/response_ces.jl")
include("algorithms/response/response_nlp.jl")
include("algorithms/response/response_bids.jl")
include("algorithms/response/response_lse.jl")
include("algorithms/response/response_plc.jl")
include("algorithms/response/response_ql.jl")
include("algorithms/primals.jl")
include("algorithms/sampler.jl")

# legacy parts, to be removed later.
include("algorithms/legacy/response_approx_lin.jl")
include("algorithms/legacy/response_dual_lp.jl")
include("algorithms/legacy/response_ces_conic.jl")

# linear systems, solving all kinds of update
include("linsys/linsys.jl")

# main algorithms
include("algorithms/conic.jl")
include("algorithms/hessianbar.jl")
include("algorithms/mirror.jl")

export LOGDIR, _RESULTS_ROOT, current_results_dir
export pprint
export logbar, log_to_expcone!, powerp_to_cone!, proj, extract_standard_form
export AbstractMarket, Market, FisherMarket, ArrowDebreuMarket, validate, update_budget!, update_supply!, expand_players!

# agent interface
export AgentType, LinearAgent, CESAgent, PLCAgent, LeontiefAgent, QuasiLinearLogAgent, GESAgent, AgentView, @agent
export solve_ql_demand
export SparseColRef, sparse_col_ref
export foreach_nz, sparse_reduce, sparse_scatter!, sparse_argmax
export sparse_dot, sparse_div_max
# workspace (canonical storage)
export AgentWorkspace, CESStore, GenStore
export cpu_workspace, cpu_workspace!, gpu_workspace,
    expand_ces!, add_ces!, add_gen!,
    prune_workspace!, _prune_ces!, _prune_gen!
export utility
export init_agents!
export create_primal_linear, create_dual_linear
export create_primal_ces, create_dual_ces
export eval!, grad!, hess!, iterate!, play!, opt!
export Conic
export HessianBar
export MirrorDec

export solve!, solve_substep!, produce_functions_from_subproblem

# best-response/utility type
export CESAnalytic
export Bids
export LSEResponse
export OptimNewtonResponse
export PLCResponse
export QLResponse

# legacy CES conic responses (kept for scripts/fisher experiments)
export CESConic, CESConicResponse, DualCESConic, DualCESConicResponse

# DRq: Diagonal + Rank-q
export SMWDRq, smw_drq!, update_cluster_map!

# randomization utilities
export NullSampler, BatchSampler

# primal methods
export boxed_allocation!

# constraints
export LinearConstr

# package-level paths (defined in utils.jl)
export LOGDIR, _RESULTS_ROOT, current_results_dir

end
# end of module `ExchangeMarket`
