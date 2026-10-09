#!/bin/bash
# CrossText2Loc baseline: export similarity matrices from scene descriptions + a checkpoint, then run the
# histogram-filter path evaluation with the paper's flags for two rows per environment:
#   ct2l_<tag>           CT2L alone            (SingleSimilarityMatrixAggregatorConfig, sigma)
#   wag_plus_ct2l_<tag>  WAG image + CT2L map  (SafaPlusNormalizedLandmarkAggregatorConfig, raw residual)
#
# usage: ct2l_eval_pipeline.sh <tag> <checkpoint.pth> <satellite_subdir> <sigma> <Env> [Env ...]
#   tag              matrix codename suffix, e.g. ny_sat, chicago_sat, chicago_carto
#   satellite_subdir satellite (text->sat) or satellite_osm_carto (text->OSM)
#   sigma            CT2L likelihood sigma (raw residual); calibrate on Seattle with calibrate_sigma.py
# Idempotent: skips an export if the .pt exists and an eval if summary_statistics.json exists.
# ROWS (env var) restricts which of the two rows run, e.g. ROWS=wag_plus_ct2l_<tag>.
set -u
TAG=$1; CKPT=$2; SUBDIR=$3; SIGMA=$4; shift 4
cd "$(dirname "$0")/../../../.."   # repo root
VIGOR=/data/overhead_matching/datasets/VIGOR
CAPTIONS=/data/overhead_matching/datasets/scene_descriptions
PATHS=/data/overhead_matching/evaluation/paths
RESULTS=${RESULTS_BASE:-/data/overhead_matching/evaluation/results/260928_ct2l}
WAG_SIGMA=0.1809   # paper image sigma (wag_no_hinge, raw residual, Seattle)

declare -A LMV=( [Seattle]=v4_202001 [NewYork]=v4_202001 [Boston]=boston [nightdrive]=boston
  [Framingham]=Framingham_v1_260101 [Middletown]=Middletown_v1_250101 [SanFrancisco_mapillary]=SanFrancisco_mapillary_v1_220101
  [post_hurricane_ian_sw]=post_hurricane_ian_sw_v1_220101 [netherlands_norr]=netherlands_norr_v1_250101 [netherlands_veluwe]=netherlands_veluwe_v1_250101 )
declare -A PATHFILE=( [Seattle]=$PATHS/Seattle_5k_5km_goal_directed.json [NewYork]=$PATHS/NewYork_5k_5km_goal_directed.json )
for e in Boston nightdrive Framingham Middletown SanFrancisco_mapillary post_hurricane_ian_sw netherlands_norr netherlands_veluwe; do
  PATHFILE[$e]=$PATHS/mappilary_equal_length/3k/$e.json
done
# Framingham Mixed-Sat (paper row): Framingham panoramas/captions/paths against the Google leaf-on tiles in
# Framingham/satellite_google (pass satellite_subdir=satellite_google for text->sat); its WAG matrix and the CT2L
# matrices live in google_satellite_related_files/ so they never shadow the MassGIS ones.
GSAT=$VIGOR/Framingham/google_satellite_related_files/similarity_matrices
declare -A DS=( [Framingham_w_leaves]=Framingham )
declare -A MATDIR=( [Framingham_w_leaves]=$GSAT )
declare -A WAGMAT=( [Framingham_w_leaves]=$GSAT/wag_no_hinge.pt )
LMV[Framingham_w_leaves]=${LMV[Framingham]}; PATHFILE[Framingham_w_leaves]=${PATHFILE[Framingham]}

bazel build //experimental/overhead_matching/swag/scripts:export_ct2l_similarity //experimental/overhead_matching/swag/scripts:evaluate_histogram_on_paths >/dev/null 2>&1 || { echo "bazel build failed"; exit 1; }
EXPORT=bazel-bin/experimental/overhead_matching/swag/scripts/export_ct2l_similarity
EVAL=bazel-bin/experimental/overhead_matching/swag/scripts/evaluate_histogram_on_paths

for ENV in "$@"; do
  D=$VIGOR/${DS[$ENV]:-$ENV}
  MAT=${MATDIR[$ENV]:-$D/similarity_matrices}/ct2l_$TAG.pt
  if [ ! -f "$MAT" ]; then
    echo "[$(date +%H:%M:%S)] export $ENV -> $MAT"
    $EXPORT --dataset_path $D --landmark_version ${LMV[$ENV]} --captions_json $CAPTIONS/${DS[$ENV]:-$ENV}/scene_descriptions.json \
      --checkpoint "$CKPT" --satellite_subdir "$SUBDIR" --output_path "$MAT" || { echo "export failed: $ENV"; continue; }
  fi
  CFG=$RESULTS/configs/$ENV; mkdir -p "$CFG"
  cat > $CFG/ct2l_$TAG.yaml <<YAML
kind: SingleSimilarityMatrixAggregatorConfig
similarity_matrix_path: $MAT
sigma: $SIGMA
YAML
  cat > $CFG/wag_plus_ct2l_$TAG.yaml <<YAML
kind: SafaPlusNormalizedLandmarkAggregatorConfig
image_similarity_matrix_path: ${WAGMAT[$ENV]:-$D/similarity_matrices/wag_no_hinge.pt}
landmark_similarity_matrix_path: $MAT
image_sigma: $WAG_SIGMA
landmark_sigma: $SIGMA
landmark_use_raw_residual: true
YAML
  for APPROACH in ${ROWS:-ct2l_$TAG wag_plus_ct2l_$TAG}; do   # ROWS="wag_plus_ct2l_$TAG" runs only the fusion row
    OUT=$RESULTS/$APPROACH/$ENV/$APPROACH
    [ -f "$OUT/summary_statistics.json" ] && { echo "skip $APPROACH/$ENV"; continue; }
    mkdir -p "$OUT"; echo "[$(date +%H:%M:%S)] eval $APPROACH / $ENV"
    $EVAL --aggregator-config $CFG/$APPROACH.yaml --paths-path ${PATHFILE[$ENV]} --output-path "$OUT" --seed 42 \
      --dataset-path $D --landmark-version ${LMV[$ENV]} --panorama-neighbor-radius-deg 0.0005 \
      --panorama-landmark-radius-px 640 --motion-noise-frac 0.141 --subdivision-factor 4 --convergence-radii "25,50,100" \
      --max-chunk-gib 2.0 --odometry-noise-frac 0.141 --odometry-noise-seed 7919 > "$OUT/eval.log" 2>&1 \
      && python3 -c "import json;d=json.load(open('$OUT/summary_statistics.json'));print('  final_err %.0f m  conv@100m %.0f m' % (d['average_final_error'], d['mean_convergence_cost_100m']))" \
      || echo "  eval FAILED (see $OUT/eval.log)"
  done
done
echo "[$(date +%H:%M:%S)] done"
