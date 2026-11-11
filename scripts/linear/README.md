# Standalone linear Fisher comparison

`test-linear.jl` shares market generation, reporting, and accuracy checks for
three methods: PR, Factorization (direct IPM), and PCG (the same IPM with IC(0)).
New runs and resumed-run reports use **Factorization (CPU)** or
**Factorization (cuDSS)** in console logs, CSVs, and plot legends.
The CLI method identifier remains `perturbed` for compatibility with saved commands.
The benchmark does not import or call ExchangeMarket.

```sh
julia test-cpu.jl --tol 1e-4
julia test-gpu.jl --tol 1e-4
```

Both launchers default to **n=500, m=1000**, one timed repetition, and densities
0.01, 0.02, 0.05. Use `--densities 0.02` for one case. `--sizes_n`, `--sizes_m`,
`--methods`, `--tol`, and `--output-dir` remain configurable. The shared runner
also accepts `--device cpu|gpu`, `--storage sparse|dense` (default: sparse), and
`--direct-solver auto|cholesky|cudss`. PR defaults to **40,000 updates**;
**all three methods have a 300-second timed runtime limit** (`--time-limit`).
The IPM update cap remains 200. Use `--verbose` for iteration logs.

Random markets are generated directly as sparse CSC matrices. Both random
and JD inputs pass through `linear_prepare_market(c,w; device, storage)` and
`run_linear(method,c,w; ...)`. `linear-sparse.jl` contains shared sparse storage
and arithmetic; `linear-gpu-sparse.jl` supplies CUDA dispatch and native sparse
assembly. Sparse allocations preserve the positive `sigma/price` entries at
zero utility implicitly. There is no JD-specific solver. Paper KKT diagnostics are opt-in and are not
computed by the synthetic comparisons.

Results default to `linear-cpu-sparse-results` or `linear-gpu-sparse-results`.
Each CSV row records storage, input SHA256, iteration/runtime limits, status,
and accuracy. `cmd.sh`, `config.toml`, `environment.txt`, and per-job checkpoints
record the run. Existing output directories require `--resume` with the same
configuration. Earlier dense results remain separate; compare CPU/GPU rows
with identical input hashes and storage.

## GPU environment

On the NVIDIA machine, instantiate the separate environment once:

```sh
julia --project=scripts/linear/gpu -e 'using Pkg; Pkg.instantiate()'
julia scripts/linear/test-gpu.jl --densities 0.02
```

The GPU project pins CUDA.jl 6.5.1 and CUDSS.jl 1.0.0 and was verified on
turing1 with Julia 1.13.3 and the RTX A6000 (package minimum: Julia 1.10).
Its project preferences select CUDA runtime and compiler 12.6 for the native
cuSPARSE IC(0) backend. Use `julia` from your normal login shell.
When upgrading an existing checkout with an older manifest, run
`julia --project=scripts/linear/gpu -e 'using Pkg; Pkg.update(); Pkg.precompile()'`
once before launching a comparison.
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

- Sparse CPU `:cholesky` uses CHOLMOD and reuses symbolic analysis; dense mode
  uses dense Cholesky. GPU `:cudss` uses sparse SPD cuDSS.
- Sparse CPU IC(0) updates only the lower CSC pattern, without a dense workspace.
- `:pcg_ic` selects CPU IC/CG or native cuSPARSE IC(0)/GPU CG by array type.
- GPU cuDSS analyzes the fixed structural CSR pattern once per solve.
- GPU IC preserves that pattern and recomputes the numerical factor per Newton
  system. SpSV analysis/buffers are reused across all CG iterations in that system.
- The batched oracle/Hessian use the same safeguarded root and cancellation-
  resistant diagonal formulas as the CPU implementation. IC inverse normalization,
  zero initial direction, and the inner relative tolerance (1e-9) are unchanged.

Each job warms up two updates and its reporting at the actual problem shape,
then starts from a fresh solver state. The supervisor terminates a worker if a
native solver call exceeds the timed runtime limit (polling every 0.1 seconds,
with up to one second for shutdown). Cooperative stops return `time_limit`;
forced stops retain the last checkpoint and cannot count as convergence.
Startup and warm-up have separate limits of 600 and 300 seconds. Failed jobs
are recorded and the worker restarts for the remaining jobs.

`wall_seconds` includes solver/workspace initialization, matrix analysis,
factorizations, iteration diagnostics, and GPU synchronization; warm-up and
market preparation are excluded. `transfer_seconds` includes sparse storage
preparation and, on GPU, the initial transfer. It is reported separately, and
`end_to_end_seconds` adds it to the solve time. `allocated_MiB` counts host Julia
allocations only. All methods use the same raw-allocation accuracy predicate.
Budget reporting compares total spending with total budget; individual budget
errors are still checked by `meets_target`.

## Saved comparison statistics and plots

From the repository root, regenerate reports from the existing CPU/GPU CSVs:

```sh
julia --startup-file=no scripts/linear/plot-linear.jl
```

`--cpu-dir`, `--gpu-dir`, and `--output-dir` select alternative result folders.
The default output is `linear-gpu-sparse-results/comparison/`. The script verifies
matching market hashes and settings before comparing devices. It exports:

- `statistics.csv`: convergence counts, runtime means/medians, update counts,
  market-clearing error, regret, and aggregate budget error, overall and per sweep.
- `paired-speedups.csv`: CPU/GPU ratios using only cases that converged on both
  devices. The ratio of mean times and the geometric mean of per-case speedups
  are reported separately.
- `fastest-counts.csv` and `failures.csv`: fastest converged methods and all
  unsuccessful runs.
- `runtime-vary-n`, `runtime-vary-m`, `iterations-vary-n`, and
  `iterations-vary-m`: CPU/GPU figures in PNG and PDF. Crosses mark missed targets.
- `performance-profile`: the fraction of cases each method solves within a
  given factor of the fastest converged method on the same device. Failed runs
  remain unsolved throughout the profile.

The earlier `scaling-v2.png` / `summary-v2.csv` (varying goods) and
`scaling-v3.png` / `summary-v3.csv` (varying buyers) names are also restored in
each sparse result folder. Earlier dense result folders are preserved.

Runtime figures exclude preparation and warm-up. All-run statistics include
capped runs; converged-only statistics are labeled explicitly. Each sweep
measurement counts once, including configurations present in both sweeps.
The current results have one timed repetition per point, so the plots show
observations without uncertainty bands.

## JD click-data experiment

See [JD.md](JD.md) for [`dataset/jd.sh`](../../dataset/jd.sh) (download and preprocess), `test-jd-gpu.jl`
(the three-method sparse GPU comparison), and `plot-jd-gpu.jl`. JD utilities
remain raw click counts; budgets sum to one within each selected market.
