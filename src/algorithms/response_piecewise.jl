using JuMP, LinearAlgebra
import MathOptInterface as MOI

"""
PiecewiseLPResponse for General Piecewise Linear Utilities.
Utility: u_i(x) = min_{l} { A_planes[:, l, i]' * x + b_planes[l, i] }
"""
function PiecewiseLPResponse()
    function __piecewise_response(;
        i::Int=1,
        p::Vector{T}=nothing,
        fisher=nothing,
        market=nothing,
        verbose=false,
        kwargs...
    ) where T
        market_obj = isnothing(fisher) ? market : fisher
        if isnothing(market_obj)
            error("Either 'fisher' or 'market' parameter must be provided")
        end

        if market_obj.A_planes === nothing
            error("Agent $i: A_planes not found. This solver requires general piecewise linear parameters.")
        end

        n = market_obj.n
        L = size(market_obj.A_planes, 2)
        b_planes = market_obj.b_planes === nothing ? zeros(T, L, market_obj.m) : market_obj.b_planes

        md = try
            ExchangeMarket.__generate_empty_jump_model(; verbose=verbose, tol=1e-8)
        catch
            Model()
        end

        @variable(md, x[1:n] >= 0)
        @variable(md, u_val)

        for l in 1:L
            a_vec = market_obj.A_planes[:, l, i]
            b_val = b_planes[l, i]
            @constraint(md, u_val <= dot(a_vec, x) + b_val)
        end
        @constraint(md, dot(p, x) <= market_obj.w[i])
        @objective(md, Max, u_val)

        JuMP.optimize!(md)

        if termination_status(md) == MOI.OPTIMAL || termination_status(md) == MOI.SLOW_PROGRESS
            x_vals = [max(0.0, JuMP.value(x[j])) for j in 1:n]
            market_obj.x[:, i] .= x_vals
            market_obj.val_u[i] = JuMP.objective_value(md)
        else
            @warn "Agent $i LP optimization failed with status: $(termination_status(md))"
        end

        return ExchangeMarket.ResponseInfo(market_obj.val_u[i], 0.0, 1, md)
    end

    return ExchangeMarket.ResponseOptimizer(__piecewise_response, :linprog, "GeneralPiecewiseLPResponse")
end
