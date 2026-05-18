# Revealed-preference experiments — runbook

This directory contains the implementation and scripts for the
column-generation surrogate-fitting experiments reported in
`overleaf/arxiv/numeric.tex`.

The shell snippets below use the following placeholders — substitute
your own absolute paths:

- `/tmp/ExchangeMarket.jl/scripts` — the `scripts/` subproject of the
  `ExchangeMarket.jl` Julia package. This is the directory you `cd`
  into so that `julia --project=.` picks up the scripts'
  `Project.toml` and `tools.jl` / `plots.jl` are reachable.
- `/tmp/repo` — the **paper** repository root, containing this
  `revealed/` directory and the `overleaf/arxiv/` LaTeX tree.

Two notes about paths:

- `revealed/` is hard-linked between `/tmp/repo/revealed/` and
  `/tmp/ExchangeMarket.jl/scripts/revealed/`. Edits to either are
  visible at the other.
- Julia resolves `@__DIR__` differently for the two paths because of
  the link. Always run the scripts after `cd`-ing into
  `/tmp/ExchangeMarket.jl/scripts`, then invoke
  `revealed/test_real.jl` (or `revealed/example_plc_plots.jl`)
  as a relative path so `include("../tools.jl")` resolves correctly.
  Pass absolute paths to `--pdf-dir` / `--tex-dir` so output lands
  under `/tmp/repo/` regardless of which side of the hard link Julia
  resolved to.

## CLI

`test_real.jl` accepts the following flags (see `julia test_real.jl -h`):

| flag                      | default                | description |
|---|---|---|
| `--market-type`, `-t`     | `ces`                  | Ground-truth family: `ces` or `plc` |
| `--n`, `-n`               | `5`                    | Number of goods |
| `--m`, `-m`               | `50`                   | Number of agents in the real market |
| `--k`, `-k`               | `100`                  | Training (and test) observation count |
| `--seed`, `-s`            | `42`                   | Master random seed |
| `--timelimit`, `-T`       | `60.0`                 | Wall-clock cap per method (seconds) |
| `--methods`               | `CG,cgma,FW,SFW`   | Comma list of methods to run |
| `--classes`               | `ces,linear,leontief`  | Comma list of CG dispatch classes |
| `--csv`                   | (empty)                | If set, append per-method rows to this CSV |
| `--no-plot`               | off                    | Skip plotting |
| `--verbosity`, `-v`       | `0`                    | `0` silent · `1` per-iteration table · `2` + per-pricing detail |
| `--pdf-dir`               | the script's own dir   | Where PDF panels are written |
| `--tex-dir`               | (empty — skip .tex)    | Where pgfplots `.tex` panels are written |
| `--plc-L`                 | `5`                    | PLC pieces per agent (PLC only) |
| `--plc-no-intercept`      | off                    | Force `b_l = 0` (homogeneous PLC) |

The methods are launched in parallel via `Threads.@spawn`. Set
`--threads=N` on the Julia command line for true concurrency
(`N >= |methods|` is enough).

## Running a single configuration

Dropping plots straight into the paper's `figs/` folder so
`numeric.tex` can `\input{}` the `.tex` panels:

```sh
cd /tmp/ExchangeMarket.jl/scripts
julia --project=. --threads=4 \
      revealed/test_real.jl \
      --market-type ces --n 5 --timelimit 60 \
      --pdf-dir /tmp/repo/revealed \
      --tex-dir /tmp/repo/overleaf/arxiv/figs
```

For the table-only / CSV-only sweep (no figures), keep `--no-plot`:

```sh
cd /tmp/ExchangeMarket.jl/scripts
julia --project=. --threads=4 \
      revealed/test_real.jl \
      --market-type ces --n 50 --timelimit 60 \
      --csv real_results.csv --no-plot
```

A single CES run at `n = 50` with all four methods takes about 60 s
wall-clock (the per-method cap binds for `n >= 50`).

## Running the scaling sweep (used for `tab.real.scaling`)

Outer parallelism over `(n, market_type)` via `xargs -P`; inner
parallelism over methods via `Threads.@spawn`.

```sh
cd /tmp/ExchangeMarket.jl/scripts
rm -f /tmp/repo/revealed/real_results.csv
rm -rf /tmp/scaling_csv && mkdir -p /tmp/scaling_csv

printf '%s\n' \
  "10 ces" "10 plc" \
  "50 ces" "50 plc" \
  "100 ces" "100 plc" \
  "200 ces" "200 plc" \
| xargs -P 2 -L 1 bash -c '
    N="$1"; MT="$2"
    echo "→ $MT n=$N start" >&2
    julia --project=. --threads=4 \
          revealed/test_real.jl \
          --market-type "$MT" --n "$N" \
          --timelimit 60 \
          --csv /tmp/scaling_csv/${N}_${MT}.csv --no-plot \
        > /tmp/scaling_${N}_${MT}.log 2>&1
    echo "← $MT n=$N done" >&2
  ' _
```

`-P 2` runs two `(n, market_type)` configurations concurrently;
`--threads=4` inside each Julia process runs the four methods
concurrently. On the 14-core M4 Pro this uses about 8 cores plus Mosek's
own threading.

Concatenate the per-config CSVs into the master one:

```sh
head -1 /tmp/scaling_csv/10_ces.csv > real_results.csv
for f in /tmp/scaling_csv/*.csv; do tail -n +2 "$f" >> real_results.csv; done
```

## Two-commodity illustration PDFs

The notebook `test-example.ipynb` and the equivalent standalone script
`example_plc_plots.jl` (in this directory) produce
`example_plc_homo.pdf`, `example_plc_nonhomo.pdf`, and `example_plc.pdf`
in `--pdf-dir`, and optionally pgfplotsx `.tex` panels in `--tex-dir`
(consumed by `numeric.tex` via `\input{}`).

CLI:

| flag        | default                  | description                                |
|---|---|---|
| `--pdf-dir` | the script's own dir     | Where the PDF panels are written           |
| `--tex-dir` | (empty — skip .tex)      | Where pgfplots `.tex` panels are written   |

To regenerate both PDFs and the pgfplots `.tex` consumed by
`overleaf/arxiv/numeric.tex`:

```sh
cd /tmp/ExchangeMarket.jl/scripts
julia --project=. \
      revealed/example_plc_plots.jl \
      --pdf-dir /tmp/repo/revealed \
      --tex-dir /tmp/repo/overleaf/arxiv/figs
```

## CSV schema

`real_results.csv` columns: `market_type, n, m, K, method, iters,
atoms_T, train_obj, test_err, time_s`. One row per `(market_type, n,
method)` combination; multiple sweeps append rows.

## Reproducing `tab.real.scaling`

The numbers in `overleaf/arxiv/numeric.tex` Table 1 were generated by
the sweep above with `m = 50`, `K = 100`, `seed = 42`, `--timelimit 60`.
The table shows `(market, n) in {(ces, 10/50/100), (plc, 10/50)}`. The
remaining rows (PLC at `n = 100`, both at `n = 200`) were not completed
because the data-set construction LP at large `n` already exhausts the
60 s budget; rerun without the `--no-plot` flag if you want the
per-iteration plots updated.

## Time-limit semantics

The `--timelimit` value is passed as the `:timelimit` kwarg to both
`run_method_tracked` (`setup.jl`) and `run_method_tracked_fw`
(`frankwolfe.jl`). Each function checks `time() - _t0 > timelimit` at
the top of each iteration and breaks. The reported `time_s` is the
`@elapsed` wall-clock of the function call, which can exceed the limit
by up to one iteration's worth.

## File map

- `setup.jl`            — method dispatch, `run_method_tracked`, master + duals
- `frankwolfe.jl`       — `run_method_tracked_fw` (FW / SFW)
- `pricing.jl`          — multi-class dispatcher + utilities (reduced cost, atom
                          storage, class-aware market expansion)
- `ces.jl`              — CES pricing primitives (LP warmstart, LBFGS, inversion)
- `linear.jl`           — linear-class pricing (sigma -> infinity boundary)
- `leontief.jl`         — Leontief-class pricing (sigma -> -1 boundary)
- `plc.jl`              — PLC market construction and demand LP
- `master.jl`           — primal/dual master LPs
- `test_real.jl`        — scaling benchmark driver
- `test-example.ipynb`  — two-commodity illustration notebook
- `test_basic.ipynb`    — fitting-LP sanity checks (linear / Leontief)
- `real_results.csv`    — current sweep output (regenerated by the sweep)

Stale paths:
- `nls.jl`, `pricing_ext.jl` are legacy / experimental; not used by `setup.jl`.
