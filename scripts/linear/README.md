# Standalone linear Fisher comparison

`test-linear.jl` shares market generation, reporting, and accuracy checks for
three methods: PR, Perturbed (direct IPM), and PCG (the same IPM with IC(0)).
The benchmark does not import or call ExchangeMarket.

```sh
julia test-cpu.jl --tol 1e-4
julia test-gpu.jl --tol 1e-4
```

Both launchers default to **n=500, m=1000**, one timed repetition, and densities
0.01, 0.02, 0.05. Use `--densities 0.02` for one case. `--sizes_n`, `--sizes_m`,
`--methods`, `--tol`, and `--output-dir` remain configurable. The shared runner
also accepts `--device cpu|gpu` and `--direct-solver auto|cholesky|cudss`.

## GPU environment

On the NVIDIA machine, instantiate the separate environment once:

```sh
julia --project=scripts/linear/gpu -e 'using Pkg; Pkg.instantiate()'
julia scripts/linear/test-gpu.jl --densities 0.02
```

The GPU project pins CUDA.jl 5.4.2 and CUDSS.jl 0.1.6 to the native interfaces
used here and supports Julia 1.9 (including the existing turing1 installation).
It does not depend on ExchangeMarket or a licensed optimizer. The CPU launcher
uses the existing `scripts` environment. `LINEAR_PROJECT` can select another
compatible environment, including the GPU project for CPU reference runs.
The optional optimal-diagonal SDP diagnostics load JuMP/MOSEK only on request.

## Device ownership and solvers

The generated read-only valuations and budgets are transferred once per case.
Each method owns independent mutable state. The outer IPM and line search are
shared; the array types select its response, assembly, and solver workspaces.
No matrix or iterate is transferred to the CPU during a GPU solve. Scalar
reductions/status checks synchronize with the host. CUDA scalar indexing is
disabled, and incompatible solver/device combinations raise an error.

- CPU `:cholesky` uses dense Cholesky; GPU `:cudss` uses sparse SPD cuDSS.
- `:pcg_ic` selects CPU IC/CG or native cuSPARSE IC(0)/GPU CG by array type.
- GPU cuDSS analyzes the fixed structural CSR pattern once per solve.
- GPU IC preserves that pattern and recomputes the numerical factor per Newton
  system. SpSV analysis/buffers are reused across all CG iterations in that system.
- The batched oracle/Hessian use the same safeguarded root and cancellation-
  resistant diagonal formulas as the CPU implementation. IC inverse normalization,
  zero initial direction, and the inner relative tolerance (1e-9) are unchanged.

`wall_seconds` includes solver/workspace initialization, matrix analysis,
factorizations, iteration diagnostics, and GPU synchronization; warm-up and
market transfer are excluded. `transfer_seconds` is reported separately, and
`end_to_end_seconds` adds it to the solve time. `allocated_MiB` counts host Julia
allocations only. All methods use the same raw-allocation accuracy predicate.
Budget reporting compares total spending with total budget; individual budget
errors are still checked by `meets_target`.

## JD click-data experiment

See [JD.md](JD.md) for [`dataset/jd.sh`](../../dataset/jd.sh) (download and preprocess), `test-jd-gpu.jl`
(the three-method sparse GPU comparison), and `plot-jd-gpu.jl`. JD utilities
remain raw click counts; budgets sum to one within each selected market.
