# Utilities

This page documents the utility functions and constants in ExchangeMarket.jl.

## Logging and Output

### LOGDIR

Directory for storing log files.

```julia
println("Logs stored in: ", LOGDIR)
```

### RESULTSDIR

Directory for storing results.

```julia
println("Results stored in: ", RESULTSDIR)
```

### pprint

Formatted printing function for consistent output.

```julia
pprint("Market equilibrium found", level=:info)
```

## Mathematical Utilities

### logbar

Logarithmic barrier function for optimization.

```julia
barrier_value = logbar(x, mu)
```

### powerp_to_cone!

Transform power utility to conic form.

```julia
cone_data = powerp_to_cone!(data, p=2.0)
```

### proj

Projection onto feasible set.

```julia
projected_point = proj(x, constraints)
```

### extract_standard_form

Extract standard form from optimization problem.

```julia
A, b, c = extract_standard_form(problem)
```

## Validation and Testing

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

## Configuration

### __default_jump_solver

Default solver for JuMP optimization problems.

```julia
__default_jump_solver = :mosek
```

### __default_sep

Default separator for output formatting.

```julia
println(__default_sep)
```

## Data Structures

### ResponseInfo

Information structure for response-based algorithms.

```julia
response_info = ResponseInfo(
    market=f0,
    algorithm=:hessianbar,
    max_iterations=1000
)
```

## File Operations

### save_results

Save market results to file.

```julia
save_results(market, "fisher_market_results.jld2")
```

### load_results

Load results from file.

```julia
loaded_market = load_results("fisher_market_results.jld2")
```

## Performance Monitoring

### benchmark_solve

Benchmark market solution performance.

```julia
timing = benchmark_solve(market, n_runs=10)
```

### profile_market

Profile market performance.

```julia
profile_data = profile_market(market)
```

## Visualization

### plot_market

Plot market equilibrium.

```julia
plot_market(market, result)
```

### plot_convergence

Plot convergence history.

```julia
plot_convergence(convergence_data)
```

## Error Handling

### Common Error Messages

1. **Solver not found**: Make sure you have the required solvers installed
2. **Convergence issues**: Try different initial conditions or algorithm parameters
3. **Memory issues**: For large markets, consider using sparse matrices

### Debugging Tips

- Enable detailed logging with `set_log_level!(market, :debug)`
- Monitor convergence with `keep_traj=true`
- Check market validation with `validate(f0, alg)` 