# Getting Started

This guide will help you get started with ExchangeMarket.jl using real examples from the package.

## Installation

### Prerequisites

ExchangeMarket.jl requires Julia 1.6 or later. You can download Julia from [julialang.org](https://julialang.org/).

### Installing the Package

```julia
using Pkg
Pkg.add("ExchangeMarket")
```

### Required Dependencies

For the examples in this guide, you'll also need:

```julia
using Pkg
Pkg.add("JuMP")
Pkg.add("MosekTools")  # For MOSEK solver
Pkg.add("Plots")       # For visualization
Pkg.add("LaTeXStrings") # For LaTeX labels
```

## Basic Usage

### Loading the Package

```julia
using ExchangeMarket
using Random, SparseArrays, LinearAlgebra
using JuMP, MosekTools
using Plots, LaTeXStrings, Printf
```

### Creating a Fisher Market

The main market type in ExchangeMarket.jl is `FisherMarket`, which supports CES (Constant Elasticity of Substitution) utilities.

```julia
# Set random seed for reproducibility
Random.seed!(1)

# Market parameters
n = 500  # number of goods
m = 1000 # number of agents  
ρ = 0.5  # CES elasticity parameter

# Create Fisher market with CES utilities
f0 = FisherMarket(m, n; ρ=ρ, bool_unit=true, scale=30.0, sparsity=0.2)
```

### Setting Up Constraints

ExchangeMarket.jl supports linear constraints on the market:

```julia
# Create linear constraint (e.g., total budget constraint)
linconstr = LinearConstr(1, n, ones(1, n), [sum(f0.w)])
```

### Initializing the Market

```julia
# Initialize prices and allocations
p₀ = ones(n) * sum(f0.w) ./ n  # Uniform initial prices
x₀ = ones(n, m) ./ m           # Uniform initial allocations
f0.x .= x₀
f0.p .= p₀
```

### Solving the Market

ExchangeMarket.jl provides several algorithms for finding market equilibrium:

```julia
# Method 1: Hessian Barrier Method
alg = HessianBar(n, m, p₀; linconstr=linconstr)
traj = opt!(alg, f0; keep_traj=true, maxiter=500)

# Method 2: Path Following
alg = PathFol(n, m, p₀; linconstr=linconstr)
traj = opt!(alg, f0; keep_traj=true, maxiter=500)

# Method 3: Tâtonnement
alg = Tât(n, m, p₀; linconstr=linconstr)
traj = opt!(alg, f0; keep_traj=true, maxiter=500)

# Method 4: Proportional Response
alg = PropRes(n, m, p₀; linconstr=linconstr)
traj = opt!(alg, f0; keep_traj=true, maxiter=500)
```

### Validating Results

```julia
# Validate the computed equilibrium
validate(f0, alg)
```

This will output:
- Problem size information
- Equilibrium validation results
- Market excess (should be near zero)
- Social welfare

## Advanced Usage

### Testing Different Elasticity Parameters

```julia
# Test different ρ values
rrange = [-0.9, 0.9]
results = []

for ρ in rrange
    f0 = FisherMarket(m, n; ρ=ρ, bool_unit=true, scale=30.0, sparsity=0.2)
    linconstr = LinearConstr(1, n, ones(1, n), [sum(f0.w)])
    
    # Initialize
    p₀ = ones(n) * sum(f0.w) ./ n
    x₀ = ones(n, m) ./ m
    f0.x .= x₀
    f0.p .= p₀
    
    # Solve with different methods
    for (name, method) in [(:HessianBar, HessianBar), (:PathFol, PathFol)]
        alg = method(n, m, p₀; linconstr=linconstr)
        traj = opt!(alg, f0; keep_traj=true, maxiter=500)
        push!(results, (name, ρ, traj))
    end
end
```

### Analyzing Convergence

```julia
# Plot convergence trajectories
for (name, ρ, traj) in results
    iterations = [t.k for t in traj]
    distances = [t.D for t in traj]
    
    plot(iterations, distances, 
         label="$name (ρ=$ρ)", 
         xlabel="Iteration", 
         ylabel="Distance to equilibrium",
         yscale=:log10)
end
```

## Troubleshooting

### Common Issues

1. **Solver not found**: Make sure you have MOSEK installed and licensed
2. **Convergence issues**: Try different initial conditions or algorithm parameters
3. **Memory issues**: For large markets, consider using sparse matrices

### Performance Tips

- Use `sparsity` parameter to control matrix density
- Adjust `scale` parameter for different problem sizes
- Monitor convergence with `keep_traj=true`

## Next Steps

- Check out the **Examples** for more detailed use cases
- Explore the **API Reference** for complete function documentation
- Read the **Tutorials** for advanced features and techniques 