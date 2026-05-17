#!/usr/bin/env bash
# One-at-a-time (OAT) ablation runner for run_test.jl on MultiCut.
#
# Fixes  market=ces, n=50, m=50, method=MultiCut, rep=1, and (mostly) the
# stopping tolerances from the seed command. Sweeps sample-size, classes,
# and k one axis at a time, each holding the others at the baseline below.
#
# Per-run output: <RESULTS_DIR>/<tag>/run.log
# Aggregate CSV : <RESULTS_DIR>/results.csv  (one row per method per rep,
#                 produced by run_test.jl --csv)
#
# Usage (run from the `scripts/` directory — the Project.toml `--project=.`
# activates is scripts/Project.toml):
#   bash revealed/ablation_multicut.sh                  # all sweeps
#   bash revealed/ablation_multicut.sh sample           # one axis
#   bash revealed/ablation_multicut.sh classes k        # multiple
#   RESULTS_DIR=results/my_run bash revealed/ablation_multicut.sh
#
# Selectable axes: sample | classes | k    (default: all three)

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
  --methods MultiCut
  --verbosity 1
  -T 1000
  -I 1000
  --tol-obj 1e-6
  --tol-delta 0
  --tol-rc 0
  --rep 1
  --csv "$CSV_PATH"
  --out-dir "$RESULTS_DIR"
)

# ---------------------------------------------------------------- baselines
# When an axis is NOT being swept, it sits at these values.
K_BASELINE=300
CLASSES_BASELINE="ces,leontief"
SAMPLE_BASELINE=25

# ---------------------------------------------------------------- sweep grids
SAMPLE_GRID=(0 10 25 50 100)
CLASSES_GRID=("ces" "ces,leontief" "ces,linear" "ces,linear,leontief")
K_GRID=(100 300 500)

# ---------------------------------------------------------------- runner
run_one() {
  local tag="$1"; shift
  local ts; ts=$(date +%Y%m%d_%H%M%S)
  local logdir="$RESULTS_DIR/$tag"
  mkdir -p "$logdir"
  # Per-run timestamped data file — sequential runs always differ at
  # second resolution, so this won't collide. The tag→file mapping is
  # logged inside <logdir>/run.log and printed here.
  local data_file="$RESULTS_DIR/real_ces_${ts}.jls"
  echo "============================================================"
  echo "[$ts] $tag    →  $data_file"
  printf '  julia ...run_test.jl'
  printf ' %s' "${COMMON_ARGS[@]}" --data-file "$data_file" "$@"
  echo
  echo "============================================================"
  julia --project=. --threads=6 revealed/run_test.jl \
    "${COMMON_ARGS[@]}" --data-file "$data_file" "$@" \
    2>&1 | tee "$logdir/run.log"
  # Record the tag↔file pairing in a manifest so analysis scripts can
  # recover "which jls came from which sweep value" without grepping logs.
  printf '%s\t%s\t%s\n' "$ts" "$tag" "$data_file" >> "$RESULTS_DIR/manifest.tsv"
}

# ---------------------------------------------------------------- sweeps
sweep_sample() {
  echo "### sweep: sample-size (k=$K_BASELINE, classes=$CLASSES_BASELINE)"
  for s in "${SAMPLE_GRID[@]}"; do
    run_one "sample-${s}" \
      --k "$K_BASELINE" \
      --classes "$CLASSES_BASELINE" \
      --sample-size "$s"
  done
}

sweep_classes() {
  echo "### sweep: classes (k=$K_BASELINE, sample-size=$SAMPLE_BASELINE)"
  for c in "${CLASSES_GRID[@]}"; do
    local ctag; ctag=$(echo "$c" | tr ',' '+')
    run_one "classes-${ctag}" \
      --k "$K_BASELINE" \
      --classes "$c" \
      --sample-size "$SAMPLE_BASELINE"
  done
}

sweep_k() {
  echo "### sweep: k (classes=$CLASSES_BASELINE, sample-size=$SAMPLE_BASELINE)"
  for k in "${K_GRID[@]}"; do
    run_one "k-${k}" \
      --k "$k" \
      --classes "$CLASSES_BASELINE" \
      --sample-size "$SAMPLE_BASELINE"
  done
}

# ---------------------------------------------------------------- dispatch
AXES=("$@")
if [[ ${#AXES[@]} -eq 0 ]]; then
  AXES=(sample classes k)
fi

for axis in "${AXES[@]}"; do
  case "$axis" in
    sample)  sweep_sample  ;;
    classes) sweep_classes ;;
    k)       sweep_k       ;;
    *) echo "unknown axis: $axis (expected: sample | classes | k)" >&2; exit 2 ;;
  esac
done

echo
echo "Done."
echo "  per-run logs : $RESULTS_DIR/<tag>/run.log"
echo "  per-run .jls : $RESULTS_DIR/real_ces_<timestamp>.jls (run_plot.jl -f <path>)"
echo "  tag↔file map : $RESULTS_DIR/manifest.tsv"
echo "  aggregate CSV: $CSV_PATH"
