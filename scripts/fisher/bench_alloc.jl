# Allocation profiling for the agent interface.
# Run with: julia --project=scripts scripts/fisher/bench_alloc.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random, SparseArrays, LinearAlgebra, Printf
using ExchangeMarket

include(joinpath(@__DIR__, "setup.jl"))

Random.seed!(1)
n = 500
m = 1000

f0 = FisherMarket(m, n; ρ=0.5, bool_unit=true, scale=30.0, sparsity=0.05)
println("c: $(typeof(f0.c)), x: $(typeof(f0.x))")
println("sparsity=$(f0.sparsity), nnz per col ≈ $(nnz(f0.c) / m)")

linconstr = LinearConstr(1, n, ones(1, n), [sum(f0.w)])
f1 = copy(f0)
p₀ = ones(n) * sum(f1.w) / n
f1.x .= ones(n, m) ./ m
f1.p .= p₀

# warm up: run one full solve so everything is compiled
(_, method, kwargs) = method_kwargs[1]
alg = method(n, m, p₀; linconstr=linconstr, kwargs...)
opt!(alg, f1; keep_traj=false, maxiter=3)

# --- warmup all code paths before benchmarking ---
agent = f1.agents[1]
f1.x[:, 1] .= rand(n) .+ 0.01
c1 = f1.agents[1].c
x1 = view(f1.x, :, 1)
c1_dense = Vector(f1.c[:, 1])
p = alg.p
w1 = f1.w[1]

# warmup calls
for i in 1:m; @agent f1 i c; end
for i in 1:m; @agent f1 i x; end
[AgentView(f1, i) for i in 1:m]
utility(agent)
sparse_dot(c1, x1)
dot(c1_dense, x1)
foreach_nz(c1) do j, cj; cj * x1[j]; end
indirect_f∇f(agent, w1, p)
f1.f∇f(p, 1)
play!(alg, f1; all=true)
solve_substep!(alg, f1.agents[1], f1)
eval!(alg, f1)
grad!(alg, f1)

# --- benchmark (after warmup) ---
println("\n" * "="^60)
println("Allocation profiling (n=$n, m=$m, sparse c, nnz/col≈$(nnz(f1.c)÷m))")
println("="^60)

# 1. @agent macro access
print("\n@agent(c) ×$m:                  ")
@time for i in 1:m
    @agent f1 i c
end

print("@agent(x) ×$m:                  ")
@time for i in 1:m
    @agent f1 i x
end

# 2. AgentView construction
print("\nAgentView ×$m:                  ")
@time [AgentView(f1, i) for i in 1:m]

print("AgentView ×1:                   ")
@time AgentView(f1, 1)

# 3. init_agents!
f1.agents = []
print("init_agents!:                   ")
@time init_agents!(f1)

# 4. utility computation
print("\nutility(agent) ×1000 [CES]:     ")
@time for _ in 1:1000; utility(agent); end

# 5. sparse_dot vs dot
print("\nsparse_dot ×10000 (nnz=$(nnz(c1))):  ")
@time for _ in 1:10000; sparse_dot(c1, x1); end

print("dot(dense) ×10000 (n=$n):       ")
@time for _ in 1:10000; dot(c1_dense, x1); end

# 6. foreach_nz
print("\nforeach_nz ×10000 (sparse):     ")
@time for _ in 1:10000
    s = 0.0
    foreach_nz(c1) do j, cj
        s += cj * x1[j]
    end
end

# 7. indirect_f∇f
print("\nindirect_f∇f(agent) ×1000:      ")
@time for _ in 1:1000; indirect_f∇f(agent, w1, p); end

print("market.f∇f(p,1) ×1000 [closure]:")
@time for _ in 1:1000; f1.f∇f(p, 1); end

# 8. Full play! round
print("\nplay!(all=$m):                  ")
@time play!(alg, f1; all=true)

print("play!(all=$m) 2nd:              ")
@time play!(alg, f1; all=true)

# 9. solve_substep! single agent
print("\nsolve_substep! ×100 (1 agent):  ")
@time for _ in 1:100; solve_substep!(alg, f1.agents[1], f1); end

# 10. eval! and grad!
print("\neval! ×1000:                    ")
@time for _ in 1:1000; eval!(alg, f1); end

print("grad! ×1000:                    ")
@time for _ in 1:1000; grad!(alg, f1); end

println("\n" * "="^60)
println("Done. Look for high allocation counts above.")
println("="^60)
