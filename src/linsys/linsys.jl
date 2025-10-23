using LinearAlgebra, SparseArrays


function linsolve!(alg, market::FisherMarket)
    alg.linsys_msg = ""
    if alg.option_step == :affinesc
        if alg.linsys ∈ [:direct, :direct_affine]
            __direct!(alg, market)
        elseif alg.linsys == :krylov
            __krylov_afsc!(alg, market)
            # __krylov_afsc_with_H!(alg, fisher)
        elseif alg.linsys ∈ [:DRq, :DRq_rep]
            __drq_afsc!(alg, market)
        else
            error("unsupported linear system solver: $(alg.linsys) for $(alg.option_step)")
        end
    elseif alg.option_step == :logbar
        if alg.linsys ∈ [:DRq, :DRq_rep]
            __drq_pd!(alg, market)
        elseif alg.linsys == :krylov
            __krylov_pd!(alg, market)
        else
            error("unsupported linear system solver: $(alg.linsys) for $(alg.option_step)")
        end
    elseif alg.option_step == :damped_ns
        if alg.linsys == :direct
            __direct_damped!(alg, market)
        elseif alg.linsys ∈ [:DRq, :DRq_rep]
            __drq_damped!(alg, market)
        else
            error("unsupported linear system solver: $(alg.linsys) for $(alg.option_step)")
        end
    elseif alg.option_step == :homotopy
        if alg.linsys ∈ [:DRq, :DRq_rep]
            __drq_homo!(alg, market)
        elseif alg.linsys == :krylov
            __krylov_homo!(alg, market)
        else
            error("unsupported linear system solver: $(alg.linsys) for $(alg.option_step)")
        end
    else
        error("unknown step type: $(alg.option_step)")
    end
end

# -------------------------------------------------------------------
# Direct mode: solving using exact Hessian ops
# -------------------------------------------------------------------
function __direct!(alg, market::FisherMarket)
    invp = 1 ./ alg.p
    alg.Δ .= -(alg.H + alg.μ * spdiagm(invp .^ 2)) \ (alg.∇ - alg.μ * invp)
end

function __direct_damped!(alg, market::FisherMarket)
    alg.Δ .= -(alg.H) \ (alg.∇)
end


# -------------------------------------------------------------------
# DRq mode: solving using DRq approximation
# -------------------------------------------------------------------
include("drq.jl")

# -------------------------------------------------------------------
# Krylov mode: solving using Krylov subspace methods
# -------------------------------------------------------------------
include("krylov.jl")