# -----------------------------------------------------------------------
# MarketWorkspace: universal data layout for CPU batched and GPU computation
#
# Core algorithm is universal. Workspace holds arrays that can be
# Array (CPU) or CuArray (GPU). Broadcasting dispatches automatically.
#
# CPU per-agent mode: workspace = nothing → AgentView + @threads
# GPU/batched mode:   workspace = MarketWorkspace → matrix broadcasts
# -----------------------------------------------------------------------

using LinearAlgebra

"""
    MarketWorkspace{M, V}

Holds market data and pre-allocated buffers for batched computation.
Parametric on array type: `{Matrix, Vector}` for CPU, `{CuMatrix, CuVector}` for GPU.
"""
mutable struct MarketWorkspace{M<:AbstractMatrix, V<:AbstractVector}
    n::Int
    m::Int
    # Problem data (read-only after setup)
    c::M           # n × m cost matrix
    w::V           # m budgets
    q::V           # n goods supply
    σ::V           # m CES parameters σᵢ = ρᵢ/(1-ρᵢ)
    # Iterates (modified during solve)
    x::M           # n × m allocation
    sumx::V        # n aggregate demand
    val_u::V       # m utility values
    # Pre-allocated workspace for play!
    C_pow::M       # n × m for c^(1+σ)
    denom::M       # 1 × m for denominator
end

"""
    cpu_workspace(market::FisherMarket)

Create a CPU workspace with aliases into the market's arrays (zero copy for iterates).
`c` is densified since the batched path uses broadcasting.
"""
function cpu_workspace(market::FisherMarket{T}) where {T}
    n, m = market.n, market.m
    MarketWorkspace(
        n, m,
        Matrix{T}(market.c),                    # dense copy of c
        market.w, market.q, market.σ,            # aliases
        market.x, market.sumx, market.val_u,     # aliases — writes go to market
        zeros(T, n, m), zeros(T, 1, m),          # workspace buffers
    )
end

"""
    gpu_workspace(market::FisherMarket; device)

Create a GPU workspace. `device` should be `cu` from CUDA.jl.
All data is copied to device. No host↔device transfers during solve.
"""
function gpu_workspace(market::FisherMarket{T}; device) where {T}
    n, m = market.n, market.m
    MarketWorkspace(
        n, m,
        device(Matrix{T}(market.c)),
        device(Vector{T}(market.w)),
        device(Vector{T}(market.q)),
        device(Vector{T}(market.σ)),
        device(Matrix{T}(market.x)),
        device(zeros(T, n)),
        device(zeros(T, m)),
        device(zeros(T, n, m)),
        device(zeros(T, 1, m)),
    )
end

# -----------------------------------------------------------------------
# to_device! / to_host! — keep GPU workspace alive across calls
# -----------------------------------------------------------------------
"""
    to_device!(market; device)

Move market data to GPU. Reuses existing GPU workspace if available.
After this, `play!` will use the batched GPU path.
"""
function to_device!(market::FisherMarket; device)
    if market.gpu_workspace_cache === nothing
        market.gpu_workspace_cache = gpu_workspace(market; device=device)
    else
        ws = market.gpu_workspace_cache
        copyto!(ws.x, market.x)
        copyto!(ws.sumx, market.sumx)
        copyto!(ws.val_u, market.val_u)
    end
    market.workspace = market.gpu_workspace_cache
    return market
end

"""
    to_host!(market)

Copy GPU results back to host arrays. Keeps GPU workspace alive for reuse.
Switches `play!` back to CPU per-agent mode.
"""
function to_host!(market::FisherMarket)
    ws = market.workspace
    ws === nothing && return market
    copyto!(market.x, Array(ws.x))
    copyto!(market.val_u, Array(ws.val_u))
    market.sumx .= Array(ws.sumx)
    market.workspace = nothing   # back to CPU per-agent mode
    return market
end

# -----------------------------------------------------------------------
# Batched play! — same code for CPU Matrix and GPU CuMatrix
# -----------------------------------------------------------------------
"""
    _play_batched!(ws::MarketWorkspace, p, σ_scalar)

Compute CES demand for ALL agents via matrix broadcasts.
Assumes all agents share the same σ (uniform ρ).

    x[j,i] = w[i] · c[j,i]^(1+σ) · p[j]^(-σ-1) / Σₖ c[k,i]^(1+σ) · p[k]^(-σ)
"""
function _play_batched!(ws::MarketWorkspace, p::AbstractVector, σ_scalar::Float64)
    σ = σ_scalar
    ρ = σ / (1.0 + σ)
    # C_pow = c^(1+σ)
    ws.C_pow .= ws.c .^ (1.0 + σ)
    # denom[i] = Σⱼ C_pow[j,i] · p[j]^(-σ)
    p_neg = p .^ (-σ)
    ws.denom .= sum(ws.C_pow .* p_neg, dims=1)
    # x = (w/denom) · C_pow · p^(-σ-1)
    p_neg1 = p .^ (-σ - 1.0)
    ws.x .= (ws.w' ./ ws.denom) .* ws.C_pow .* p_neg1
    # utility: u[i] = (Σⱼ c[j,i] · x[j,i]^ρ)^(1/ρ)
    ws.val_u .= dropdims(
        sum(ws.c .* ws.x .^ ρ, dims=1) .^ (1.0 / ρ);
        dims=1
    )
    # aggregate demand
    ws.sumx .= dropdims(sum(ws.x, dims=2); dims=2)
    return ws
end
