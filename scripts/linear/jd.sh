#!/usr/bin/env bash
# Run from any directory: bash scripts/linear/jd.sh [preprocessor options].
# Python >=3.8 with SQLite >=3.25; no third-party Python packages required.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
data_dir=${JD_DATA_DIR:-"$repo_dir/dataset/jd"}
python_bin=${PYTHON:-python3}
mkdir -p "$data_dir"
"$python_bin" "$script_dir/jd_download.py" --data-dir "$data_dir" 2>&1 | tee "$data_dir/download.log"
"$python_bin" "$script_dir/jd_preprocess.py" --data-dir "$data_dir" "$@" 2>&1 | tee "$data_dir/preprocess.log"
