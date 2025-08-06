# Fisher Market Example

This example demonstrates how to use ExchangeMarket.jl to solve Fisher market equilibrium problems with CES utilities.

## Problem Setup

We'll solve a Fisher market with:
- 1000 agents (buyers)
- 500 goods
- CES utilities with different elasticity parameters
- Multiple optimization algorithms

## Complete Example

```julia
using ExchangeMarket
using Random, SparseArrays, LinearAlgebra
using JuMP, MosekTools
using Plots, LaTeXStrings, Printf

# Set random seed for reproducibility
Random.seed!(1)

# Market parameters
rrange = [-0.9, 0.9]  # CES elasticity parameters to test
n = 500   # number of goods
m = 1000  # number of agents

# Filter for specific methods
method_filter(name) = name ∈ [:LogBar, :PathFol, :Tât, :PropRes]

# Store results
table_time = []
results = []
results_phi = Dict()
results_ground = Dict()

# Test different ρ values
for ρ in rrange
    # Create Fisher market with CES utilities
    f0 = FisherMarket(m, n; ρ=ρ, bool_unit=true, scale=30.0, sparsity=0.2)
    linconstr = LinearConstr(1, n, ones(1, n), [sum(f0.w)])
    
    ρfmt = @sprintf("%+.2f", ρ)
    σfmt = @sprintf("%+.2f", f0.σ)
    
    println("Testing ρ = $ρfmt (σ = $σfmt)")
    
    # -----------------------------------------------------------------------
    # Compute ground truth using HessianBar
    # -----------------------------------------------------------------------
    f1 = copy(f0)
    p₀ = ones(n) * sum(f1.w) ./ n
    x₀ = ones(n, m) ./ m
    f1.x .= x₀
    f1.p .= p₀
    
    # Use HessianBar to compute ground truth
    alg = HessianBar(n, m, p₀; linconstr=linconstr)
    traj = opt!(alg, f1; keep_traj=true)
    
    pₛ = copy(alg.p)  # Ground truth prices
    results_phi[ρ] = pₛ
    results_ground[ρ] = (alg, traj, f1)
    
    # Test different algorithms
    for (name, method) in [
        (:HessianBar, HessianBar),
        (:PathFol, PathFol),
        (:Tât, Tât),
        (:PropRes, PropRes)
    ]
        !method_filter(name) && continue
        
        f1 = copy(f0)
        p₀ = ones(n) * sum(f1.w) ./ n
        x₀ = ones(n, m) ./ m
        f1.x .= x₀
        f1.p .= p₀
        
        alg = method(n, m, p₀; linconstr=linconstr)
        traj = opt!(alg, f1; keep_traj=true, pₛ=pₛ, maxiter=500)
        
        push!(results, ((name, ρ), (alg, traj, f1)))
        push!(table_time, (n, m, name, ρ, traj[end].t))
        
        println("  $name: $(traj[end].t) seconds")
    end
end
```

## Validation

After computing solutions, validate the results:

```julia
# Validate ground truth solutions
for (ρ, (alg, traj, f1)) in results_ground
    println("Validating ρ = $ρ")
    validate(f1, alg)
end
```

This will output validation information including:
- Problem size
- Equilibrium validation results
- Market excess (should be near zero)
- Social welfare

## Convergence Analysis

### Plotting Convergence Trajectories

```julia
# Plot distance to equilibrium for each ρ and method
for ρ in rrange
    ρfmt = @sprintf("%+.2f", ρ)
    σfmt = @sprintf("%+.2f", ρ / (1 - ρ))
    
    fig = plot(
        ylabel=L"$\|\mathbf{p} - \mathbf{p}^*\|$",
        title=L"$\rho := %$ρfmt~(\sigma := %$σfmt)$",
        legendbackgroundcolor=RGBA(1.0, 1.0, 1.0, 0.8),
        yticks=10.0 .^ (-16:4:3),
        xtickfont=font(18),
        ytickfont=font(18),
        xscale=:identity,
        size=(1200, 600)
    )
    
    for ((name, _ρ), (alg, traj, f1)) in results
        if _ρ != ρ
            continue
        end
        
        traj_pp₊ = map(pp -> pp.D, traj)
        traj_tt₊ = map(pp -> pp.k, traj)
        
        plot!(fig, traj_tt₊, traj_pp₊, 
              label=L"\texttt{%$name}", 
              linewidth=2, 
              linestyle=:dash, 
              markershape=:circle)
    end
    
    display(fig)
end
```

## Performance Comparison

### Timing Analysis

```julia
# Create timing table
using DataFrames

df = DataFrame(
    n=Int[],
    m=Int[],
    method=Symbol[],
    ρ=Float64[],
    time=Float64[]
)

for (n, m, method, ρ, time) in table_time
    push!(df, (n, m, method, ρ, time))
end

# Display results
println("Performance Summary:")
println(df)
```

### Algorithm Comparison

```julia
# Compare convergence rates
for ρ in rrange
    println("ρ = $ρ:")
    for ((name, _ρ), (alg, traj, f1)) in results
        if _ρ == ρ
            final_dist = traj[end].D
            iterations = length(traj)
            println("  $name: $iterations iterations, final distance: $final_dist")
        end
    end
end
```

## Key Insights

### CES Parameter Effects

- **ρ = -0.9**: High substitution elasticity (σ ≈ 0.47), goods are close substitutes
- **ρ = 0.0**: Cobb-Douglas utilities (σ = 1), moderate substitution
- **ρ = 0.9**: Low substitution elasticity (σ ≈ 9), goods are complements

### Algorithm Performance

1. **HessianBar**: Best for large problems, quadratic convergence
2. **PathFol**: Robust convergence, good for poor initialization
3. **Tât**: Simple and fast for small problems
4. **PropRes**: Intuitive interpretation, slower convergence

### Convergence Patterns

- All methods converge to the same equilibrium
- Convergence rate depends on ρ value
- HessianBar shows fastest convergence for well-conditioned problems
- PathFol is most robust to different ρ values

## Advanced Features

### Custom Utility Parameters

```julia
# Test with different utility parameters
f0 = FisherMarket(m, n; 
    ρ=0.5,           # CES elasticity
    bool_unit=true,   # Unit utilities
    scale=30.0,       # Utility scaling
    sparsity=0.2      # Matrix sparsity
)
```

### Linear Constraints

```julia
# Add custom linear constraints
A = [ones(1, n); rand(2, n)]  # 3 constraints
b = [sum(f0.w); rand(2)]      # Right-hand sides
linconstr = LinearConstr(3, n, A, b)
```

This example demonstrates the full workflow of using ExchangeMarket.jl for Fisher market equilibrium computation, from problem setup to solution validation and performance analysis. 