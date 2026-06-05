# -----------------------------------------------------------------------
# differentiation utilities for affine-constrained CES
# @author: Chuwen Zhang <chuwzhang@gmail.com>
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# compute the exact affine-constrained Hessian
# -----------------------------------------------------------------------
@doc raw"""
This computes the exact Hessian of the affine-constrained problem,
    namely, the log-UMP has an affine constraint on the allocation
"""
function __ces_compute_exact_hess_afcon!(alg, market::FisherMarket)

    _Hi = (i) -> begin
        _γ = market.x[:, i] .* alg.p / market.w[i]
        ρᵢ = market.ρ[i]
        _W = begin
            ((1 - ρᵢ) * diagm(alg.p .^ 2 ./ _γ) +
             ρᵢ * alg.p * alg.p') ./ (market.w[i]^2)
        end
        _constr_x = market.constr_x[i]
        @assert _constr_x.n == market.n

        # Z = [_W _constr_x.A'; _constr_x.A spzeros(_constr_x.m, _constr_x.m)]
        # rhs = [1 / market.w[i] * I(market.n); zeros(_constr_x.m, _constr_x.n)]
        # sol = Z \ rhs
        # # first n rows
        # _H = sol[1:market.n, :]
        _iW = market.w[i]^2 / (1 - ρᵢ) * diagm(1 ./ alg.p) * (diagm(_γ) - ρᵢ * _γ * _γ') * diagm(1 ./ alg.p)
        _iH = 1 / market.w[i] .* (
            _iW - _iW * _constr_x.A' * inv(_constr_x.A * _iW * _constr_x.A' + 1e-12 * I) * _constr_x.A * _iW
        )
        return _iH
    end
    alg.H .= mapreduce(_Hi, +, alg.sampler.indices, init=spzeros(market.n, market.n))
    @info "use exact Hessian from affine-constrained UMP"
end
