# JD click-data experiment

The three methods are PR, Perturbed (IPM with cuDSS), and PCG (the same IPM
with native CUDA IC(0)). They share the existing outer algorithms and do not
use the ExchangeMarket interface. `test_jd_gpu.jl` follows the instance scales
and displayed relative KKT criterion in §5/Table 3 of
[Liu et al. (2025)](https://arxiv.org/pdf/2506.06258).

## Data and budgets

For product `j` and user `i`, `c[j,i]` is the **number of click records** for
that pair, including repeated clicks. Unobserved pairs have utility zero.
Utilities are not scaled, normalized, clipped, or imputed. This deliberately
differs from the paper's per-user utility normalization, as requested.

Users are ranked by the number of distinct products they clicked, descending,
with user ID ascending to break ties. A selected prefix retains every product
clicked by at least one selected user, with unit supply. This activity ranking
reproduces the published nonzero totals; ranking by total repeated clicks does
not. The paper does not specify its tie-break.

Independent `Uniform(0,1)` budget draws use Python's `random.Random(seed)`, with
default seed 1. **The Julia loader divides the selected users' budgets by their
sum, so every selected market has total budget 1.** Prefixes use the same raw
draws before renormalization. No budget comes from the click counts.

The source is the pinned
[Hugging Face dataset](https://huggingface.co/datasets/a6687543/MSOM_Data_Driven_Challenge_2020/tree/cbad25bd35202e98e6ae7fc23de4913695c28927).
Only `JD_click_data.csv` is needed. Download verification requires 920,317,454
bytes and SHA256
`7e1ea745f1a49a0a223832b97247964f817e485d32df0fb332b85fb3f363d728`.
Rows without a user/product ID are excluded. The complete published file is
retained, including a few boundary-date records. No month filter is applied
by default. `--month 2018-03` is an explicit alternative, not the reproduction.

The verified source produces these dimensions (goods × users):

| Users | Goods | Nonzeros | Paper goods |
|---:|---:|---:|---:|
| 2,000 | 13,472 | 136,877 | 13,541 |
| 5,000 | 15,855 | 265,220 | 15,887 |
| 20,000 | 19,447 | 698,579 | 19,449 |
| 50,000 | 21,432 | 1,278,391 | 21,410 |
| 200,000 | 23,842 | 2,937,906 | 23,853 |
| 500,000 | 25,202 | 4,679,648 | 25,214 |
| 1,000,000 | 26,083 | 6,189,526 | 26,115 |
| 2,557,836 (all) | 27,330 | 8,073,837 | 27,330 |

Every nonzero total matches Table 3. The full source contains five fewer
identified users than the paper's 2,557,841. Subset product counts also differ;
the available description is insufficient to reproduce its exact user ties.
The pipeline reports actual dimensions without adding dummy users/products.
The repository uses `n=goods, m=buyers`, opposite the paper; this driver labels
both dimensions explicitly as `goods` and `users`.

## Pipeline

From the repository root, locally or in the same turing1 checkout:

```bash
bash scripts/linear/jd.sh
```

Python 3.8+ and its standard library are sufficient. The script downloads and
checks the raw data, then aggregates counts with SQLite and writes:

- `dataset/jd/utilities.bin`: sparse utility data and reproducible raw budget draws.
- `manifest.json`: source revision/hash, matrix hash, selection and budget rules.
- `instances.csv`: actual and published dimensions.
- `goods.csv`, `users.csv.gz`: row/column identities and activity counts.
- `click_counts.sqlite`: reusable aggregation cache.
- `download.log`, `preprocess.log`: pipeline logs.

`JD_DATA_DIR` selects another output directory, `PYTHON` selects Python, and
arguments to `jd.sh` are passed to preprocessing (e.g. `--seed 7`). Downloading
resumes an incomplete `.part` file. A matching completed dataset is verified
and reused. Generated data is Git-ignored. If the remote machine cannot reach
Hugging Face, copy the checksum-verified raw file into its `dataset/jd` first;
the same pipeline verifies and preprocesses that local input.

## GPU comparison

On turing1, from the existing repository root:

```bash
/home/chuwen/julia-1.9.4/bin/julia --project=scripts/linear/gpu \
  scripts/linear/test-jd-gpu-correctness.jl

/home/chuwen/julia-1.9.4/bin/julia --project=scripts/linear/gpu \
  scripts/linear/test_jd_gpu.jl --tol 1e-4
```

Default prefixes are `2000,5000,20000,50000,200000,500000,1000000,all`.
Use `--users 2000 --outdir scripts/linear/jd_gpu_smoke` for a first case.
Other controls include `--methods pr,perturbed,pcg`, `--repeats 1`,
`--time-limit 7200`, `--maxiter-pr 1000000`, `--maxiter-ipm 100000`,
`--linear-rtol 1e-9`, and `--linear-maxiter 0` (automatic: 5 × goods).
The larger IPM update cap lets the paper's wall-time limit govern difficult
real-data instances instead of the synthetic benchmark's 200-update cap.
IPM uses `sigma_min = tol * minimum(w) / goods`, as in the synthetic tests.

By default all methods stop on the paper's displayed relative KKT error,
setting `t_i = c_i' x_i` and `y_i = w_i / t_i`. The three residuals and their
maximum are saved. The literal displayed dual denominator is used; it differs
from the denominator in the authors' paper-era code. `--criterion shared`
instead uses the existing maximum of market-clearing error, individual
relative budget error, and relative regret. Both sets of errors are recorded
regardless of the stopping rule. Budget reporting uses total spending versus
total budget. Normalizing budgets affects the numerical size of the paper's
KKT residuals, so its threshold need not imply the same relative individual
accuracy as `--criterion shared`.

Results go to `scripts/linear/jd_gpu_results/runs.csv`, with per-job logs,
trajectories, settings, source hash, environment, and `cmd.sh`. Completed rows
are saved immediately. `--resume` requires identical settings and input hash,
and skips completed jobs. Each method runs in an isolated Julia process to
contain GPU failures. Two updates at the actual problem size and a diagnostic call are warmed up
before initializing a fresh timed state. Warm-up time is reported separately;
using only a tiny market leaves shape-specialized CUDA compilation in timing.
Host-to-device preparation is reported separately; solve time includes
workspace allocation, Hessian pattern construction, analysis/factorization,
iterations, diagnostics, and final synchronization. The 7,200-second limit
starts at solve initialization; a supervisor enforces it with at most a short
shutdown grace period. Startup has a separate 600-second limit. Errors and
timeouts are recorded as failures, and do not stop the remaining cases.

Valuation and iterate arrays remain on the GPU during the algorithm. Scalar
reductions synchronize with the CPU. Sparse allocation storage includes an
analytic representation of the positive allocations on zero-utility entries
(`sigma / price`); omitting those would change the perturbed model. The
Hessian is assembled with native cuSPARSE sparse multiplication, then solved
by cuDSS or IC(0)/PCG. Factor fill can still be expensive at large scales.

## Binary layout

`JDUTIL01` is little-endian, with a 64-byte header: eight magic bytes followed
by seven UInt64 values (version=1, goods, users, nnz, seed, flags=0, reserved=0).
It then stores CSC column pointers (UInt64, users+1), row indices (UInt32,
nnz), raw click counts (Float64, nnz), and raw budget draws (Float64, users).
Indices are one-based. `jd-data.jl` reads a user prefix, removes unused goods,
and normalizes its budgets without changing the utility values.

```julia
include("scripts/linear/jd-data.jl")
market = JDData.load_market("dataset/jd/utilities.bin"; users=2000)
@assert sum(market.w) ≈ 1
# market.c is SparseMatrixCSC{Float64}; rows = goods, columns = users.
```

Regenerate the summary table and plots from saved CSV data (no GPU needed):

```bash
julia --project=scripts scripts/linear/plot_jd_gpu.jl scripts/linear/jd_gpu_results
```
