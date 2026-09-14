#!/usr/bin/env bash
set -euo pipefail
RUNFILES=/home/harel/code/robot-offline-loc/bazel-bin/experimental/overhead_matching/swag/farfield/localization/grid_filter.runfiles
TASK_REPO=/home/harel/code/robot-regression-loc
TASK_PYTHONPATH="$TASK_REPO:/data/farfield_matching/runs/260913_accuracy_recovery/tools"
for task_dep in "$RUNFILES"/pip_3_12_*/site-packages; do
  TASK_PYTHONPATH="$TASK_PYTHONPATH:$task_dep"
done
export PYTHONPATH="$TASK_PYTHONPATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TORCHINDUCTOR_CACHE_DIR=/data/farfield_matching/runs/260913_overnight/inductor_cache
export TORCHINDUCTOR_COMPILE_THREADS=2
exec "$RUNFILES/python_3_12_x86_64-unknown-linux-gnu/bin/python3" "$@"
