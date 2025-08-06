# Models

This page documents the market models available in ExchangeMarket.jl.

## Fisher Market

The main market type in ExchangeMarket.jl is `FisherMarket`, which supports CES (Constant Elasticity of Substitution) utilities.

### Constructor

```julia
FisherMarket(m, n; ρ=0.0, bool_unit=true, scale=1.0, sparsity=1.0)
```

**Parameters:**
- `m`: Number of agents/buyers
- `n`: Number of goods
- `ρ`: CES elasticity parameter (default: 0.0 for Cobb-Douglas)
- `bool_unit`: Whether to use unit utilities (default: true)
- `scale`: Scaling factor for utilities (default: 1.0)
- `sparsity`: Sparsity of utility matrix (default: 1.0)

### Example

```julia
# Create a Fisher market with CES utilities
Random.seed!(1)
n = 500  # number of goods
m = 1000 # number of agents
ρ = 0.5  # CES elasticity parameter

f0 = FisherMarket(m, n; ρ=ρ, bool_unit=true, scale=30.0, sparsity=0.2)
```

### Market Properties

The `FisherMarket` object contains:

- `f0.w`: Agent endowments
- `f0.c`: Utility matrix (sparse)
- `f0.p`: Current prices
- `f0.x`: Current allocations
- `f0.σ`: Substitution elasticity (σ = ρ/(1-ρ))

## Linear Constraints

### LinearConstr

Linear constraints for market optimization problems.

```julia
LinearConstr(n_constraints, n_vars, A, b)
```

**Parameters:**
- `n_constraints`: Number of constraint equations
- `n_vars`: Number of variables
- `A`: Constraint matrix
- `b`: Constraint right-hand side

### Example

```julia
# Create budget constraint: sum of prices equals total wealth
linconstr = LinearConstr(1, n, ones(1, n), [sum(f0.w)])
```

## Market Initialization

### Setting Initial Prices and Allocations

```julia
# Initialize with uniform prices and allocations
p₀ = ones(n) * sum(f0.w) ./ n  # Uniform initial prices
x₀ = ones(n, m) ./ m           # Uniform initial allocations
f0.x .= x₀
f0.p .= p₀
```

## Market Validation

### validate

Validate market equilibrium to ensure all constraints are satisfied.

```julia
validate(f0, alg)
```

This function outputs:
- Problem size information
- Equilibrium validation results  
- Market excess (should be near zero)
- Social welfare

### Example Output

```
------------------------------------------------------------
 :problem size
 :    number of agents: 1000
 :    number of goods: 500
 :    avg number of nonzero entries in c: 0.1997
 :equilibrium information
 :method: HessianBar
------------------------------------------------------------
10×2 DataFrame
 Row │ utility    left_budget  
     │ Float64    Float64      
─────┼──────────────────────────
   1 │ 1.24158e-7   0.0
   2 │ 3.80644e-6  -2.1684e-19
   ...
------------------------------------------------------------
 :(normalized) market excess: [-5.6658e-10, 5.1747e-10]
 :            social welfare:  -1.30839518e+01
------------------------------------------------------------
```

## Utility Functions

### CES Utilities

The package supports CES (Constant Elasticity of Substitution) utilities:

```julia
U(x) = (∑ᵢ αᵢ xᵢ^ρ)^(1/ρ)
```

where:
- `ρ` is the elasticity parameter
- `σ = ρ/(1-ρ)` is the substitution elasticity
- `αᵢ` are the CES parameters

### Parameter Ranges

- `ρ ∈ [-∞, 1)`: CES parameter
- `ρ = 0`: Cobb-Douglas utilities (limit)
- `ρ = -∞`: Leontief utilities (limit)
- `ρ = 1`: Linear utilities (limit)

## Market Properties

### Equilibrium Conditions

A valid market equilibrium satisfies:

1. **Budget Balance**: Each agent spends exactly their endowment value
2. **Market Clearing**: Total demand equals total supply for each good
3. **Individual Rationality**: Each agent gets non-negative utility
4. **Pareto Efficiency**: No reallocation can make everyone better off

### Economic Interpretation

- **Prices**: Reflect relative scarcity and preferences
- **Allocations**: Optimal consumption bundles given prices
- **Equilibrium**: No agent wants to change their consumption given prices 