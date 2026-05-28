# -----------------------------------------------------------------------
# (intentionally empty)
#
# The GPU / batched-compute path (MarketWorkspace, to_device!, to_host!,
# _play_batched!, cpu_workspace(::FisherMarket), gpu_workspace(::FisherMarket))
# was removed in the 2026/05 workspace refactor. `play!` now runs the
# per-agent CPU path unconditionally. The AgentWorkspace-level GPU
# storage constructor `gpu_workspace(n, m; device, ...)` lives in
# `src/models/workspace.jl`.
# -----------------------------------------------------------------------
