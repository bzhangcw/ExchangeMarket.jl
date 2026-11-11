#!/usr/bin/env bash
# Run from any directory: bash dataset/jd.sh [preprocessor options].
# Python >=3.8 with SQLite >=3.25; no third-party Python packages required.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
data_dir=${JD_DATA_DIR:-"$script_dir/jd"}
python_bin=${PYTHON:-python3}
mkdir -p "$data_dir"
"$python_bin" "$script_dir/jd_download.py" --data-dir "$data_dir" 2>&1 | tee "$data_dir/download.log"
"$python_bin" "$script_dir/jd_preprocess.py" --data-dir "$data_dir" "$@" 2>&1 | tee "$data_dir/preprocess.log"
