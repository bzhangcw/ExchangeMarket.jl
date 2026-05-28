#!/usr/bin/env bash
# Full-grid ablation runner for run_test.jl on cgma.
#
# Fixes  market=ces, n=50, m=50, method=cgma, rep=1, and the stopping
# tolerances from the seed command. Sweeps the Cartesian product of
# SAMPLE_GRID × CLASSES_GRID × K_GRID — i.e., every combination of the
# three knobs (NOT one-at-a-time).
#
# Per-run output: <RESULTS_DIR>/<tag>/run.log
# Aggregate CSV : <RESULTS_DIR>/results.csv  (one row per method per rep,
#                 produced by run_test.jl --csv)
#
# Usage (run from the `scripts/` directory — the Project.toml `--project=.`
# activates is scripts/Project.toml):
#   bash revealed/ablation_multicut.sh                  # full grid, serial
#   RESULTS_DIR=results/my_run bash revealed/ablation_multicut.sh
#
# Parallel execution — list mode emits one shell command per line on
# stdout WITHOUT executing, so xargs / parallel can fan it out:
#
#   # 4 runs at a time via xargs. On macOS the -S 4096 is REQUIRED;
#   # BSD xargs's default replstr buffer (-S 255) is too small for the
#   # ~400-char emitted commands and errors with
#   # "command line cannot be assembled, too long". GNU xargs (Linux)
#   # has a larger default and accepts the same invocation without -S.
#   LIST_ONLY=1 bash revealed/ablation_multicut.sh \
#     | xargs -S 4096 -I CMD -P 4 sh -c 'CMD'
#
#   # ...or via GNU parallel (no buffer-size knob needed):
#   LIST_ONLY=1 bash revealed/ablation_multicut.sh \
#     | parallel -j 4
#
# Each emitted command redirects to its own <logdir>/run.log so the
# parallel jobs don't interleave on the console. The manifest.tsv is
# populated synchronously at emission time, so it always exists by the
# time the jobs start. Note: each run already uses --threads=6, so
# total threads ≈ 6 × P — tune P to your core count.

set -euo pipefail

# Refuse to run from anywhere other than scripts/: --project=. and the
# relative `revealed/run_test.jl` path below both assume that cwd.
if [[ ! -f revealed/run_test.jl ]]; then
  echo "error: run this script from the scripts/ directory" >&2
  echo "  current dir: $(pwd)" >&2
  exit 2
fi

# ---------------------------------------------------------------- paths
RESULTS_DIR=${RESULTS_DIR:-results/ablation_multicut_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RESULTS_DIR"
# Resolve to absolute. run_test.jl's --csv handling joins relative paths
# with @__DIR__ (= revealed/), NOT cwd, so a relative path like
# `results/ablation_multicut_<ts>/results.csv` lands at
# `scripts/revealed/results/...` and fails. Absolute paths bypass that.
RESULTS_DIR=$(cd "$RESULTS_DIR" && pwd)
CSV_PATH="$RESULTS_DIR/results.csv"

# ---------------------------------------------------------------- fixed args
# Mirrors the seed invocation; `-h` from that invocation is dropped (would
# just print --help and exit). Per-run --data-file is injected by run_one
# (run_test.jl defaults to <out-dir>/real_<market>.jls, which would
# overwrite across the sweep otherwise — we suffix each dump by its tag
# so the per-run trajectories survive for replotting with run_plot.jl).
COMMON_ARGS=(
  --market-type ces
  --n 50
  --m 50
  --methods cgma
  --verbosity 1
  -T 2000
  -I 1000
  --tol-obj 1e-6
  --tol-delta 0
  --tol-rc 0
  --rep 1
  --csv "$CSV_PATH"
  --out-dir "$RESULTS_DIR"
)

# ---------------------------------------------------------------- sweep grids
# Full Cartesian product: |SAMPLE| × |CLASSES| × |K| runs.
SAMPLE_GRID=(25 50)
CLASSES_GRID=("ces" "ces,linear")
K_GRID=(500)

# ---------------------------------------------------------------- runner
# Job counter — gives each emitted/run job a unique suffix even when
# multiple run_one calls land in the same second (which happens
# trivially in LIST_ONLY mode, where the for-loop emits all jobs in
# well under a second).
JOB_COUNTER=0

run_one() {
  local tag="$1"; shift
  JOB_COUNTER=$((JOB_COUNTER + 1))
  local ts; ts=$(date +%Y%m%d_%H%M%S)
  local serial; serial=$(printf '%02d' "$JOB_COUNTER")
  local logdir="$RESULTS_DIR/$tag"
  # Per-run timestamped data file. The trailing 2-digit serial guards
  # against same-second collisions in list/parallel mode.
  local data_file="$RESULTS_DIR/real_ces_${ts}_${serial}.jls"

  if [[ "${LIST_ONLY:-0}" == 1 ]]; then
    # Record manifest synchronously now; emit the runnable command on
    # stdout so xargs / parallel can pick it up.
    mkdir -p "$logdir"
    printf '%s\t%s\t%s\n' "$ts" "$tag" "$data_file" >> "$RESULTS_DIR/manifest.tsv"
    {
      printf 'julia --project=. --threads=6 revealed/run_test.jl'
      printf ' %q' "${COMMON_ARGS[@]}" --data-file "$data_file" "$@"
      printf ' > %q 2>&1\n' "$logdir/run.log"
    }
    return
  fi

  mkdir -p "$logdir"
  # Wrapper status to stderr so it doesn't pollute stdout (matches the
  # discipline used in LIST_ONLY mode; the terminal still shows it).
  {
    echo "============================================================"
    echo "[$ts] $tag    →  $data_file"
    printf '  julia ...run_test.jl'
    printf ' %s' "${COMMON_ARGS[@]}" --data-file "$data_file" "$@"
    echo
    echo "============================================================"
  } >&2
  julia --project=. --threads=6 revealed/run_test.jl \
    "${COMMON_ARGS[@]}" --data-file "$data_file" "$@" \
    2>&1 | tee "$logdir/run.log"
  # Record the tag↔file pairing in a manifest so analysis scripts can
  # recover "which jls came from which sweep value" without grepping logs.
  printf '%s\t%s\t%s\n' "$ts" "$tag" "$data_file" >> "$RESULTS_DIR/manifest.tsv"
}

# ---------------------------------------------------------------- dispatch
# Cartesian product of the three grids. Tag encodes all three knob
# values so downstream analysis can pivot on any axis.
NRUNS=$(( ${#SAMPLE_GRID[@]} * ${#CLASSES_GRID[@]} * ${#K_GRID[@]} ))
echo "### sweep: full grid — ${#SAMPLE_GRID[@]} × ${#CLASSES_GRID[@]} × ${#K_GRID[@]} = $NRUNS run(s)" >&2

for s in "${SAMPLE_GRID[@]}"; do
  for c in "${CLASSES_GRID[@]}"; do
    ctag=$(echo "$c" | tr ',' '+')
    for k in "${K_GRID[@]}"; do
      run_one "s${s}_c${ctag}_k${k}" \
        --sample-size "$s" \
        --classes     "$c" \
        --k           "$k"
    done
  done
done

if [[ "${LIST_ONLY:-0}" == 1 ]]; then
  # In list mode the trailing summary goes to stderr so the stdout
  # stream contains nothing but runnable shell commands (xargs would
  # otherwise try to execute these echo lines).
  {
    echo
    echo "Emitted $JOB_COUNTER job(s)."
    echo "  pipe to: xargs -S 4096 -I CMD -P <N> sh -c 'CMD'  (macOS needs -S; GNU xargs doesn't)"
    echo "       or: parallel -j <N>"
    echo "  results : $RESULTS_DIR/"
    echo "  manifest: $RESULTS_DIR/manifest.tsv (already populated)"
  } >&2
else
  echo
  echo "Done."
  echo "  per-run logs : $RESULTS_DIR/<tag>/run.log"
  echo "  per-run .jls : $RESULTS_DIR/real_ces_<timestamp>_<NN>.jls (run_plot.jl -f <path>)"
  echo "  tag↔file map : $RESULTS_DIR/manifest.tsv"
  echo "  aggregate CSV: $CSV_PATH"
fi
