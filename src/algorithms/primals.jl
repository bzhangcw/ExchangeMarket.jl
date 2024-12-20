function boxed_allocation!(alg, fisher::FisherMarket)
    q = copy(fisher.q)

    for i in 1:fisher.m
        ratio = fisher.c[i, :] ./ alg.p
        fisher.x[i, :] .= 0
        idx = sortperm(ratio, rev=true)
        budget = fisher.w[i]
        for j in idx
            fisher.x[i, j] = min(budget / alg.p[j], q[j])
            q[j] -= fisher.x[i, j]
            budget -= fisher.x[i, j] * alg.p[j]
            if budget == 0
                break
            end
        end
    end
end
