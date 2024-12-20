
function linsolve!(alg, fisher::FisherMarket)
    if alg.linsys == :none
        __plainlinsolve!(alg, fisher)
    else
        __quasinewton(alg, fisher)
    end
end

function __plainlinsolve!(alg, fisher::FisherMarket)
    invp = 1 ./ alg.p
    alg.Δ .= -(alg.H + alg.μ * spdiagm(invp .^ 2)) \ (alg.∇ - alg.μ * invp)
end
