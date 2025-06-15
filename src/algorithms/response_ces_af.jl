# -----------------------------------------------------------------------
# homogenous affine-constrained CES utility maximization problem
#   using affine-constrained flow model
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# @date: 2024/11/22
# -----------------------------------------------------------------------


using JuMP
import MathOptInterface as MOI

# --------------------------------------------------------------------------
# primal form of CES economy in linear-conic form
# --------------------------------------------------------------------------
@doc raw"""
  solve the logarithmic utility maximization problem by JuMP + optimizer
  for CES utility function of ρ ≤ 1
    use max log(uᵢ(xᵢ))
"""
function __af_conic_log_response_ces(;
    i::Int=1,
    p::Vector{T}=nothing,
    fisher::FisherMarket=nothing,
    μ=1e-4,
    verbose=false,
    kwargs...
) where {T}
    ρ = fisher.ρ
    ϵᵢ = μ * 1e-5
    md = __generate_empty_jump_model(; verbose=verbose, tol=ϵᵢ)

    @variable(md, u)
    @variable(md, logu)
    log_to_expcone!(u, logu, md)

    @variable(md, x[1:fisher.n] >= 0)
    @variable(md, ξ[1:fisher.n] >= 0)

    if !isnothing(fisher.constr_x)
        @constraint(md, flow_balance, fisher.constr_x[i].A * x == fisher.constr_x[i].b)
    else
        @warn "no constraint imposed on x, where is the affine constraint?"
    end

    # budget constraint
    @constraint(md, budget, p' * x <= fisher.w[i])
    # utility constraint
    # Δ^{ρ} ξ^{1-ρ}≥ r 
    # ⇒ [Δ,ξ,r] ∈ P₃(ρ) [power cone]
    _c = fisher.c[:, i] .^ (1 / ρ)
    @constraint(md, sum(ξ) == u)
    @constraint(
        md,
        ξc[j=1:fisher.n],
        [_c[j] * x[j], u, ξ[j]] in MOI.PowerCone(ρ)
    )
    @objective(md, Max, logu)

    JuMP.optimize!(md)
    fisher.x[:, i] .= value.(x)
    return ResponseInfo(
        objective_value(md),
        # the rest is dummy
        ϵᵢ,
        1,
        md
    )
end

AFCESConic = AFCESConicResponse = ResponseOptimizer(
    __af_conic_log_response_ces,
    :linconicaffine,
    "AFCESConicResponse"
)

# --------------------------------------------------------------------------
# build the AF-CES model; reference only
# --------------------------------------------------------------------------
@doc raw"""
  query the homogeneous constraint Ax=0 from 
    the standard form of the `affine-constrained flow model`:
        the maximum revenue source-sink flow with the CES utility function
    A: adjacency matrix of the network, 
    E: edge set of the network,
    p: price vector,
    r: revenue vector,
    nᵥ: number of vertices in the network
"""
function __query_Abc_standard(source, sink, A, E, nᵥ)
    model = __generate_empty_jump_model(;)
    @variable(model, 0 <= f[E])

    # sum of in/out flows
    sum_flows(i; out=false) = begin
        edges = out ? A[i, :].nzind : A[:, i].nzind
        length(edges) == 0 && return 0
        out && return sum(f[(i, j)] for j in edges)
        return sum(f[(j, i)] for j in edges)
    end

    # balance constraints
    for i in 1:nᵥ
        if i == source || i == sink
            continue
        end
        @constraint(model,
            sum_flows(i; out=true) == sum_flows(i; out=false)
        )
    end
    # source-sink balance
    @constraint(
        model,
        sum_flows(source; out=false) - sum_flows(source; out=true) == sum_flows(sink; out=true) - sum_flows(sink; out=false)
    )
    # assign a dummy objective
    @objective(model, Max, sum(f[e] for (edx, e) in enumerate(E)))
    return model, extract_standard_form(model)...
end