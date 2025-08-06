# ExchangeMarket.jl

A Julia package for competitive exchange market optimization and analysis.

## Overview

ExchangeMarket.jl provides a comprehensive framework for modeling and solving competitive exchange market problems. The package includes:

- **Fisher Market Models**: Implementation of Fisher market equilibrium computation
- **CES Market Models**: Constant Elasticity of Substitution market formulations  
- **Conic Optimization**: Advanced optimization algorithms using conic programming
- **Linear Systems**: Efficient solvers for linear systems arising in market computations
- **Mirror Descent**: Implementation of mirror descent algorithms for market optimization

## Quick Start

```julia
using ExchangeMarket
using Random, SparseArrays, LinearAlgebra
using JuMP, MosekTools

# Create a Fisher market with CES utilities
Random.seed!(1)
n = 500  # number of goods
m = 1000 # number of agents
ρ = 0.5  # CES elasticity parameter

# Create Fisher market
f0 = FisherMarket(m, n; ρ=ρ, bool_unit=true, scale=30.0, sparsity=0.2)

# Set up linear constraints
linconstr = LinearConstr(1, n, ones(1, n), [sum(f0.w)])

# Initialize prices and allocations
p₀ = ones(n) * sum(f0.w) ./ n
x₀ = ones(n, m) ./ m
f0.x .= x₀
f0.p .= p₀

# Solve using Hessian barrier method
alg = HessianBar(n, m, p₀; linconstr=linconstr)
traj = opt!(alg, f0; keep_traj=true, maxiter=500)

# Validate results
validate(f0, alg)
```

## Installation

```julia
using Pkg
Pkg.add("ExchangeMarket")
```

## Key Features

- **Multiple Market Types**: Support for Fisher markets, CES markets, and general exchange markets
- **Advanced Algorithms**: Conic optimization, mirror descent, and response-based algorithms
- **Efficient Solvers**: Specialized linear system solvers for market computations
- **Extensible Framework**: Modular design allowing easy extension to new market types
- **Comprehensive Utilities**: Logging, validation, and analysis tools

## Documentation

- **Getting Started**: Learn the basics of using ExchangeMarket.jl
- **API Reference**: Complete documentation of all functions and types
- **Examples**: Step-by-step examples for common use cases
- **Tutorials**: In-depth tutorials for advanced features

## Citation

If you use ExchangeMarket.jl in your research, please cite:

```bibtex
@software{zhang2024exchangemarket,
  title={ExchangeMarket.jl: A Julia Package for Competitive Exchange Market Optimization},
  author={Zhang, Chuwen},
  year={2024},
  url={https://github.com/yourusername/ExchangeMarket.jl}
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](https://github.com/yourusername/ExchangeMarket.jl/blob/main/CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/yourusername/ExchangeMarket.jl/blob/main/LICENSE) file for details. 