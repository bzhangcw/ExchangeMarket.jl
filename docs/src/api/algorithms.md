# Algorithms

This page documents the optimization algorithms available in ExchangeMarket.jl.

## Available Algorithms

ExchangeMarket.jl provides several algorithms for finding market equilibrium:

### HessianBar

Hessian barrier method for interior point optimization.

```julia
alg = HessianBar(n, m, p₀; linconstr=linconstr)
traj = opt!(alg, f0; keep_traj=true, maxiter=500)
```

### PathFol

Path following method for market equilibrium.

```julia
alg = PathFol(n, m, p₀; linconstr=linconstr)
traj = opt!(alg, f0; keep_traj=true, maxiter=500)
```

### Tât

Tâtonnement (price adjustment) method.

```julia
alg = Tât(n, m, p₀; linconstr=linconstr)
traj = opt!(alg, f0; keep_traj=true, maxiter=500)
```

### PropRes

Proportional response method.

```julia
alg = PropRes(n, m, p₀; linconstr=linconstr)
traj = opt!(alg, f0; keep_traj=true, maxiter=500)
```

## Algorithm Comparison

### Testing Different Methods

```julia
# Test different algorithms on the same problem
methods = [
    (:HessianBar, HessianBar),
    (:PathFol, PathFol),
    (:Tât, Tât),
    (:PropRes, PropRes)
]

results = []
for (name, method) in methods
    alg = method(n, m, p₀; linconstr=linconstr)
    traj = opt!(alg, f0; keep_traj=true, maxiter=500)
    push!(results, (name, traj))
end
```

### Convergence Analysis

```julia
# Plot convergence trajectories
for (name, traj) in results
    iterations = [t.k for t in traj]
    distances = [t.D for t in traj]
    
    plot(iterations, distances, 
         label=name, 
         xlabel="Iteration", 
         ylabel="Distance to equilibrium",
         yscale=:log10)
end
```

## Algorithm Parameters

### Common Parameters

All algorithms accept these common parameters:

- `n`: Number of goods
- `m`: Number of agents
- `p₀`: Initial prices
- `linconstr`: Linear constraints

### opt! Function

The main optimization function:

```julia
opt!(alg, f0; keep_traj=true, maxiter=500, pₛ=nothing)
```

**Parameters:**
- `alg`: Algorithm object
- `f0`: Fisher market object
- `keep_traj`: Whether to keep trajectory (default: true)
- `maxiter`: Maximum iterations (default: 500)
- `pₛ`: Ground truth prices for comparison (optional)

## Trajectory Analysis

### Trajectory Object

The `opt!` function returns a trajectory containing:

- `t.k`: Iteration number
- `t.t`: Time elapsed
- `t.D`: Distance to equilibrium
- `t.φ`: Objective value
- `t.∇φ`: Gradient norm
- `t.Δp`: Price change norm

### Example Trajectory Output

```
running Phase I...
      k |  lg(μ) |             φ |    |∇φ| |    |Δp| |       t |      tₗ |       α |     kᵢ 
      0 |  -5.40 | -1.450859e+01 | 1.1e+01 | 4.5e-02 | 0.0e+00 | 4.0e-01 | 1.0e+00 | 2.0e+00 
--------------------------------------------------------------------------------------------
running Phase II...
      k |  lg(μ) |             φ |    |∇φ| |    |Δp| |       t |      tₗ |       α |     kᵢ 
      1 |  -5.40 | -1.208262e+01 | 8.4e-01 | 3.2e-03 | 1.0e+00 | 4.1e-01 | 1.0e+00 | 2.0e+00 
      2 |  -7.48 | -1.208395e+01 | 6.2e-02 | 2.1e-04 | 1.0e+00 | 4.1e-01 | 1.0e+00 | 2.0e+00 
      3 | -10.72 | -1.208395e+01 | 6.1e-04 | 2.1e-06 | 1.0e+00 | 4.2e-01 | 1.0e+00 | 2.0e+00 
      4 | -14.02 | -1.208395e+01 | 8.6e-06 | 3.3e-08 | 1.0e+00 | 4.2e-01 | 1.0e+00 | 2.0e+00 
      5 | -17.32 | -1.208395e+01 | 1.9e-07 | 7.3e-10 | 1.0e+00 | 4.2e-01 | 1.0e+00 | 2.0e+00 
```

## Performance Comparison

### Algorithm Characteristics

1. **HessianBar**: Interior point method, good for large problems
2. **PathFol**: Path following, robust convergence
3. **Tât**: Simple price adjustment, fast for small problems
4. **PropRes**: Proportional response, intuitive interpretation

### Convergence Properties

- **HessianBar**: Quadratic convergence, requires good initial point
- **PathFol**: Linear convergence, robust to poor initialization
- **Tât**: Linear convergence, simple implementation
- **PropRes**: Sublinear convergence, intuitive behavior

## Advanced Usage

### Custom Algorithm Parameters

```julia
# Custom parameters for HessianBar
alg = HessianBar(n, m, p₀; 
    linconstr=linconstr,
    μ₀=1e-3,      # Initial barrier parameter
    tol=1e-8,     # Tolerance
    maxiter=1000   # Maximum iterations
)
```

### Algorithm Selection

Choose algorithm based on problem characteristics:

- **Large problems** (n, m > 1000): Use HessianBar or PathFol
- **Small problems** (n, m < 100): Use Tât or PropRes
- **Poor initialization**: Use PathFol
- **Good initialization**: Use HessianBar 