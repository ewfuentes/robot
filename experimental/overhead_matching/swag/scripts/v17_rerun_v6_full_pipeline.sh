#!/bin/bash
# Full path-eval pipeline for the v17 retrain on the cleaned-up train.py pipeline
# (commit 51e3d24, output dir 260521_v17_rerun_v6).
#
#   1. Export v17_rerun_v6 sim matrices for 9 cities (named v17_rerun_v6.pt)
#   2. Calibrate sigma on Seattle
#   3. Write per-city aggregator configs
#   4. Run histogram-filter path eval on all 9 cities
#
# Mirrors the historical v17 pipeline at
#   /data/overhead_matching/training_outputs/260518_090000_early_fusion_attempts/_pipeline_artifacts/scripts/v17_full_pipeline.sh
# with MODEL/NAME/RESULTS_BASE/CFG_DIR/SIGMA_OUT swapped.
#
# Idempotent: Phase 1 skips an export if the .pt already exists;
# Phase 4 skips an eval if summary_statistics.json already exists.

set -u
MODEL=/data/overhead_matching/training_outputs/260521_v17_rerun_v6/260521_084703_safa_v17_temp01_cv05
NAME=v17_rerun_v6
EXPORT=bazel-bin/experimental/overhead_matching/swag/scripts/export_similarity_matrix
SIGMA_BIN=bazel-bin/experimental/overhead_matching/swag/scripts/calibrate_sigma
EVAL_BIN=bazel-bin/experimental/overhead_matching/swag/scripts/evaluate_histogram_on_paths
RESULTS_BASE=/data/overhead_matching/evaluation/results/260521_v17_rerun_v6_path_eval
CFG_DIR=/tmp/v17_rerun_v6_configs
SIGMA_OUT=/tmp/sigma_calibration/v17_rerun_v6_seattle
mkdir -p "$CFG_DIR"
cd /home/ekf/code/robot-safa-extractors

# Tuples: city  dataset_path  landmark_version  paths_path  use_cache_flag
declare -a CITIES=(
  "Seattle /data/overhead_matching/datasets/VIGOR/Seattle v4_202001 /data/overhead_matching/evaluation/paths/Seattle_5k_5km_goal_directed.json cache"
  "Boston /data/overhead_matching/datasets/VIGOR/Boston boston /data/overhead_matching/evaluation/paths/mappilary_equal_length/3k/Boston.json nocache"
  "nightdrive /data/overhead_matching/datasets/VIGOR/nightdrive boston /data/overhead_matching/evaluation/paths/mappilary_equal_length/3k/nightdrive.json nocache"
  "NewYork /data/overhead_matching/datasets/VIGOR/NewYork v4_202001 /data/overhead_matching/evaluation/paths/NewYork_5k_5km_goal_directed.json nocache"
  "Framingham /data/overhead_matching/datasets/VIGOR/mapillary/Framingham Framingham_v1_260101 /data/overhead_matching/evaluation/paths/mappilary_equal_length/3k/Framingham.json nocache"
  "Middletown /data/overhead_matching/datasets/VIGOR/mapillary/Middletown Middletown_v1_250101 /data/overhead_matching/evaluation/paths/mappilary_equal_length/3k/Middletown.json nocache"
  "Norway /data/overhead_matching/datasets/VIGOR/mapillary/Norway Norway_v1_251201 /data/overhead_matching/evaluation/paths/mappilary_equal_length/3k/Norway.json nocache"
  "post_hurricane_ian_sw /data/overhead_matching/datasets/VIGOR/mapillary/post_hurricane_ian_sw post_hurricane_ian_sw_v1_220101 /data/overhead_matching/evaluation/paths/mappilary_equal_length/3k/post_hurricane_ian_sw.json nocache"
  "SanFrancisco_mapillary /data/overhead_matching/datasets/VIGOR/mapillary/SanFrancisco_mapillary SanFrancisco_mapillary_v1_220101 /data/overhead_matching/evaluation/paths/mappilary_equal_length/3k/SanFrancisco_mapillary.json nocache"
)

# === Phase 1: export sim matrices ===
echo "[$(date +%H:%M:%S)] === Phase 1: sim matrix exports ==="
for spec in "${CITIES[@]}"; do
  IFS=' ' read -r CITY DSPATH LMV PATHS CACHE_FLAG <<< "$spec"
  OUT="$DSPATH/similarity_matrices/$NAME.pt"
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] SKIP export $CITY (already exists)"
    continue
  fi
  echo "==========================================="
  echo "[$(date +%H:%M:%S)] export $CITY"
  echo "==========================================="
  EXTRA=""
  [ "$CACHE_FLAG" = "nocache" ] && EXTRA="--disable_safa_cache"
  $EXPORT \
    --model_path "$MODEL" \
    --dataset_path "$DSPATH" \
    --output_path "$OUT" \
    --checkpoint best \
    --landmark_version "$LMV" \
    --fallback_to_config \
    $EXTRA
  echo "[$(date +%H:%M:%S)] $CITY export done"
done

# === Phase 2: calibrate sigma on Seattle ===
echo ""
echo "[$(date +%H:%M:%S)] === Phase 2: calibrate sigma on Seattle ==="
mkdir -p "$SIGMA_OUT"
$SIGMA_BIN \
  --similarity-matrix-path /data/overhead_matching/datasets/VIGOR/Seattle/similarity_matrices/$NAME.pt \
  --dataset-path /data/overhead_matching/datasets/VIGOR/Seattle \
  --landmark-version v4_202001 \
  --output-path "$SIGMA_OUT" \
  --name-prefix "$NAME"
SIGMA=$(python3 -c "import json; print(f\"{json.load(open('$SIGMA_OUT/${NAME}_sigma_calibration.json'))['sigma_mle_per_pair']:.4f}\")")
echo "[$(date +%H:%M:%S)] sigma_MLE = $SIGMA"

# === Phase 3: write configs ===
echo ""
echo "[$(date +%H:%M:%S)] === Phase 3: write per-city aggregator configs (sigma=$SIGMA) ==="
for spec in "${CITIES[@]}"; do
  IFS=' ' read -r CITY DSPATH LMV PATHS CACHE_FLAG <<< "$spec"
  cat > "$CFG_DIR/$CITY.yaml" <<YAML
kind: SingleSimilarityMatrixAggregatorConfig
similarity_matrix_path: $DSPATH/similarity_matrices/$NAME.pt
sigma: $SIGMA
YAML
done

# === Phase 4: path evals ===
echo ""
echo "[$(date +%H:%M:%S)] === Phase 4: path evaluations ==="
for spec in "${CITIES[@]}"; do
  IFS=' ' read -r CITY DSPATH LMV PATHS CACHE_FLAG <<< "$spec"
  OUT="$RESULTS_BASE/$CITY/$NAME"
  if [ -f "$OUT/summary_statistics.json" ]; then
    echo "[$(date +%H:%M:%S)] SKIP eval $CITY (already done)"
    continue
  fi
  mkdir -p "$OUT"
  echo "==========================================="
  echo "[$(date +%H:%M:%S)] eval $CITY"
  echo "==========================================="
  $EVAL_BIN \
    --aggregator-config "$CFG_DIR/$CITY.yaml" \
    --paths-path "$PATHS" \
    --output-path "$OUT" \
    --seed 42 \
    --dataset-path "$DSPATH" \
    --landmark-version "$LMV" \
    --panorama-neighbor-radius-deg 0.0005 \
    --panorama-landmark-radius-px 640 \
    --motion-noise-frac 0.141 \
    --subdivision-factor 4 \
    --convergence-radii "25,50,100" \
    --max-chunk-gib 2.0 \
    --odometry-noise-frac 0.141 \
    --odometry-noise-seed 7919
  echo "[$(date +%H:%M:%S)] $CITY eval done"
done

echo "==========================================="
echo "[$(date +%H:%M:%S)] ALL DONE"
