# Correspondence Similarity Evaluation Pipeline

End-to-end guide for building correspondence-based similarity matrices and evaluating them on the histogram filter path evaluation benchmark.

## Overview

The pipeline has 6 stages:

0. **Generate correspondence data** — Create Gemini-labeled landmark match pairs for training
1. **Text embeddings** — Embed all OSM tag values via Vertex AI
2. **Train classifier** — Train `CorrespondenceClassifier` on Chicago (train) / Seattle (val)
3. **Precompute raw scores** — Compute P(match) for all (pano_landmark, osm_landmark) pairs
4. **Build similarity matrices** — Aggregate raw scores into (num_panos, num_sats) matrices
5. **Evaluate on paths** — Run histogram filter with different fusion configs

Each stage's outputs feed into the next. Stages 0-2 are expensive but run once; stages 3-5 are fast and support iteration.

## Prerequisites

```bash
# Google Cloud auth (for text embedding API)
export GOOGLE_CLOUD_PROJECT=your-project-id
gcloud auth application-default login
```

## Stage 0: Generate Correspondence Data

Create Gemini-labeled landmark correspondence pairs for training the classifier. For each panorama, the script pairs pano_v2 landmarks (Set 1) with nearby OSM database landmarks (Set 2) and asks Gemini to identify matches, assign uniqueness scores (1-5), and provide hard/easy negative examples.

### Create batch requests

```bash
# Generate Gemini batch JSONL for a city
bazel run //experimental/overhead_matching/swag/scripts:landmark_pairing_cli -- \
    --all --city Chicago --with_negatives \
    --generate_batch /tmp/chicago_correspondence_batch.jsonl \
    --thinking_level HIGH
```

This produces a JSONL file where each line pairs one panorama's pano_v2 landmarks against the OSM landmarks on its associated satellite tiles.

### Submit to Gemini

Upload and submit the batch JSONL via Vertex AI:

```bash
# Upload to GCS
gcloud storage cp /tmp/chicago_correspondence_batch.jsonl \
    gs://crossview/correspondence_chicago/requests/

# Submit batch job
bazel run //experimental/overhead_matching/swag/scripts:vertex_batch_manager -- \
    submit-all \
    --input_prefix gs://crossview/correspondence_chicago/requests/ \
    --output_prefix gs://crossview/correspondence_chicago/results/ \
    --model gemini-3-flash-preview

# Download results
gcloud storage cp -r gs://crossview/correspondence_chicago/results/ \
    /data/overhead_matching/datasets/landmark_correspondence/chicago_seattle_neg_v3_full/Chicago/responses/
```

Repeat for Seattle (validation city).

### Expected output structure

```
chicago_seattle_neg_v3_full/
├── Chicago/
│   └── responses/
│       └── prediction-model-{timestamp}/
│           └── predictions.jsonl
├── Seattle/
│   └── responses/
│       └── prediction-model-{timestamp}/
│           └── predictions.jsonl
└── text_embeddings.pkl  (created in Stage 1)
```

Each response contains matches (positive pairs), uniqueness scores, and hard/easy negatives that the training pipeline parses into `CorrespondencePair` objects.

## Stage 1: Text Embeddings

Embed all unique text-type tag values (street names, amenity types, etc.) from OSM landmark feather files and pano_v2 extractions.

```bash
bazel run //experimental/overhead_matching/swag/scripts:precompute_value_embeddings -- \
    --feather_dirs \
        /data/overhead_matching/datasets/VIGOR/Boston \
        /data/overhead_matching/datasets/VIGOR/Chicago \
        /data/overhead_matching/datasets/VIGOR/NewYork \
        /data/overhead_matching/datasets/VIGOR/nightdrive \
        /data/overhead_matching/datasets/VIGOR/SanFrancisco \
        /data/overhead_matching/datasets/VIGOR/Seattle \
        /data/overhead_matching/datasets/VIGOR/mapillary/Framingham \
        /data/overhead_matching/datasets/VIGOR/mapillary/Gap \
        /data/overhead_matching/datasets/VIGOR/mapillary/MiamiBeach \
        /data/overhead_matching/datasets/VIGOR/mapillary/Middletown \
        /data/overhead_matching/datasets/VIGOR/mapillary/Norway \
        /data/overhead_matching/datasets/VIGOR/mapillary/post_hurricane_ian \
        /data/overhead_matching/datasets/VIGOR/mapillary/SanFrancisco_mapillary \
    --pano_v2_base /data/overhead_matching/datasets/semantic_landmark_embeddings/mapillary \
    --base_embeddings /data/overhead_matching/datasets/landmark_correspondence/eval_text_embeddings.pkl \
    --output /data/overhead_matching/datasets/landmark_correspondence/eval_text_embeddings_all_cities.pkl
```

**Output:** `eval_text_embeddings_all_cities.pkl` — 140K entries, ~419 MB

**Stats only** (no API calls): Add `--stats_only` to see how many new values need embedding.

## Stage 2: Train Classifier

Train the `CorrespondenceClassifier` on Gemini-labeled correspondence data from Chicago (train) and Seattle (val).

```bash
# Config file (saved at training output dir)
cat > /tmp/correspondence_train.yaml << 'EOF'
kind: CorrespondenceTrainConfig
data_dir: /data/overhead_matching/datasets/landmark_correspondence/chicago_seattle_neg_v3_full
text_embeddings_path: /data/overhead_matching/datasets/landmark_correspondence/eval_text_embeddings_all_cities.pkl
output_dir: /data/overhead_matching/training_outputs/landmark_correspondence/v5_all_cities
train_city: Chicago
val_city: Seattle
include_difficulties:
- positive
- easy
- hard
batch_size: 512
num_epochs: 20
lr: 0.0003
weight_decay: 0.01
warmup_fraction: 0.05
gradient_clip_norm: 1.0
use_amp: true
num_workers: 4
seed: 42
encoder:
  kind: CorrespondenceEncoderConfig
  text_input_dim: 768
  text_proj_dim: 128
classifier:
  kind: CorrespondenceClassifierMlpConfig
  mlp_hidden_dim: 128
  dropout: 0.1
cosine_schedule: true
EOF

bazel run //experimental/overhead_matching/swag/scripts:train_landmark_correspondence -- \
    --config /tmp/correspondence_train.yaml
```

**Output:** `best_model.pt` (~700KB, 173K parameters), trains in ~5 minutes on GPU.

**Evaluate per-city classification:**
```bash
bazel run //experimental/overhead_matching/swag/scripts:evaluate_correspondence_model -- \
    --model_path /data/overhead_matching/training_outputs/landmark_correspondence/v5_all_cities/best_model.pt \
    --text_embeddings_path /data/overhead_matching/datasets/landmark_correspondence/eval_text_embeddings_all_cities.pkl \
    --data_dirs /data/overhead_matching/datasets/landmark_correspondence/mapillary/*/responses
```

## Stage 3: Precompute Raw Scores

For each city, compute P(match) for every (pano_landmark, osm_landmark) pair and save the full cost matrix. This is the expensive step (~1-10 min per city depending on size).

```bash
MODEL=/data/overhead_matching/training_outputs/landmark_correspondence/v5_all_cities/best_model.pt
TEXT_EMB=/data/overhead_matching/datasets/landmark_correspondence/eval_text_embeddings_all_cities.pkl
PANO_V2_MAPILLARY=/data/overhead_matching/datasets/semantic_landmark_embeddings/mapillary
PANO_V2_EXTRA=/data/overhead_matching/datasets/semantic_landmark_embeddings

# Example: one city
bazel run //experimental/overhead_matching/swag/scripts:export_correspondence_similarity -- \
    --save_raw \
    --model_path $MODEL \
    --text_embeddings_path $TEXT_EMB \
    --dataset_path /data/overhead_matching/datasets/VIGOR/mapillary/Norway \
    --pano_v2_base $PANO_V2_MAPILLARY $PANO_V2_EXTRA \
    --output_path /data/overhead_matching/datasets/VIGOR/mapillary/Norway/correspondence_scores/v5_all_cities_raw.pt
```

**Output:** `v5_all_cities_raw.pt` per city, containing:
- `cost_matrix`: `(total_pano_landmarks, total_osm_landmarks)` float32
- `pano_id_to_lm_rows`: which rows belong to each panorama
- `pano_lm_tags`: tag tuples per pano landmark
- `osm_lm_indices`: column → dataset landmark index mapping
- `osm_lm_tags`: tag dicts per OSM landmark

**Pano_v2 sources:**
- Mapillary cities: `--pano_v2_base .../semantic_landmark_embeddings/mapillary`
- Boston/nightdrive: also pass `.../semantic_landmark_embeddings` (contains `boston_snowy/` and `nightdrive/` subdirs)

**Timing:** MiamiBeach ~1 min, Norway ~10 min, Boston ~15 min (scales with n_pano × n_osm_landmarks).

## Stage 4: Build Similarity Matrices

Convert raw P(match) scores into `(num_panos, num_sats)` similarity matrices using a specific matching/aggregation configuration. This is fast (seconds to minutes per city, no GPU needed).

```bash
# Example: Hungarian matching, sum aggregation, 0.8 threshold, uniqueness-weighted
bazel run //experimental/overhead_matching/swag/scripts:export_correspondence_similarity -- \
    --from_raw /data/overhead_matching/datasets/VIGOR/mapillary/Norway/correspondence_scores/v5_all_cities_raw.pt \
    --dataset_path /data/overhead_matching/datasets/VIGOR/mapillary/Norway \
    --output_path /data/overhead_matching/datasets/VIGOR/mapillary/Norway/similarity_matrices/correspondence_v5_hungarian_0.8.pt \
    --method hungarian \
    --aggregation sum \
    --prob_threshold 0.8 \
    --uniqueness_weighted
```

**Options:**
- `--method`: `hungarian` (optimal 1:1) or `greedy`
- `--aggregation`: `sum`, `max`, or `log_odds`
- `--prob_threshold`: minimum P(match) to include (0.0 - 0.95)
- `--uniqueness_weighted`: weight by `1/log2(1 + n_matches)` per pano landmark

**Batch all cities:**
```bash
BINARY=bazel-bin/experimental/overhead_matching/swag/scripts/export_correspondence_similarity

for CITY_PATH in \
    /data/overhead_matching/datasets/VIGOR/mapillary/Framingham \
    /data/overhead_matching/datasets/VIGOR/mapillary/Gap \
    /data/overhead_matching/datasets/VIGOR/mapillary/MiamiBeach \
    /data/overhead_matching/datasets/VIGOR/mapillary/Middletown \
    /data/overhead_matching/datasets/VIGOR/mapillary/Norway \
    /data/overhead_matching/datasets/VIGOR/mapillary/post_hurricane_ian \
    /data/overhead_matching/datasets/VIGOR/mapillary/SanFrancisco_mapillary \
    /data/overhead_matching/datasets/VIGOR/Boston \
    /data/overhead_matching/datasets/VIGOR/nightdrive; do

    CITY_NAME=$(basename "$CITY_PATH")
    RAW="${CITY_PATH}/correspondence_scores/v5_all_cities_raw.pt"
    OUT="${CITY_PATH}/similarity_matrices/correspondence_v5_hungarian_0.8.pt"

    echo "=== $CITY_NAME ==="
    $BINARY --from_raw "$RAW" --dataset_path "$CITY_PATH" --output_path "$OUT" \
        --method hungarian --aggregation sum --prob_threshold 0.8 --uniqueness_weighted
done
```

## Stage 5: Evaluate on Paths

Run the histogram filter on evaluation paths, comparing different similarity sources and fusion methods.

### Aggregator configs

**SAFA image-only (baseline):**
```yaml
kind: SingleSimilarityMatrixAggregatorConfig
similarity_matrix_path: /data/.../similarity_matrices/safa_dinov3_wag_chicago.pt
sigma: 0.25
```

**Correspondence-only:**
```yaml
kind: SingleSimilarityMatrixAggregatorConfig
similarity_matrix_path: /data/.../similarity_matrices/correspondence_v5_hungarian_0.8.pt
sigma: 0.25
```

**Entropy-adaptive fusion (SAFA + correspondence):**
```yaml
kind: EntropyAdaptiveAggregatorConfig
image_similarity_matrix_path: /data/.../similarity_matrices/safa_dinov3_wag_chicago.pt
landmark_similarity_matrix_path: /data/.../similarity_matrices/correspondence_v5_hungarian_0.8.pt
sigma: 0.25
```

### Run evaluation

```bash
bazel run //experimental/overhead_matching/swag/scripts:evaluate_histogram_on_paths -- \
    --aggregator-config /tmp/config.yaml \
    --paths-path /data/overhead_matching/evaluation/paths/mappilary/Norway.json \
    --output-path /data/overhead_matching/evaluation/results/260325_correspondence_fusion/Norway/ea_safa_corr \
    --dataset-path /data/overhead_matching/datasets/VIGOR/mapillary/Norway \
    --landmark-version Norway_v1_251201 \
    --convergence-radii "25,50,100" \
    --seed 42
```

### Results

Output goes to `--output-path`:
- `summary_statistics.json` — overall metrics (`average_final_error`, `mean_convergence_cost_Xm`)
- `aggregator_config.yaml` — config used
- `NNNNNNN/error.pt` — per-path error trajectories

### Compare results

```bash
python3 -c "
import json
from pathlib import Path
base = Path('/data/overhead_matching/evaluation/results')
for run_dir in ['260310_all_non_vigor_datasets', '260325_correspondence_fusion']:
    for city in sorted((base / run_dir).iterdir()):
        for mode in sorted(city.iterdir()):
            stats = mode / 'summary_statistics.json'
            if stats.exists():
                d = json.load(open(stats))
                err = d['average_final_error']
                c100 = d.get('mean_convergence_cost_100m', 0)
                print(f'{city.name:25s} {mode.name:25s} error={err:7.1f}m  conv100={c100:5.0f}')
"
```

## Interactive Exploration

### Correspondence Explorer (Flask app)

Explore satellite scores interactively with map visualization:

```bash
bazel run //experimental/overhead_matching/swag/analysis:correspondence_explorer -- \
    --precomputed_data /data/.../correspondence_scores/v5_all_cities_raw.pt \
    --dataset_path /data/.../VIGOR/mapillary/MiamiBeach \
    --port 5003
```

### Fusion Explorer (Marimo notebook)

Compare aggregation methods, analyze per-panorama and per-landmark behavior:

```bash
bazel run //common/python:marimo_server -- edit \
    /home/ekf/code/robot-tag-matcher-9000/experimental/overhead_matching/swag/analysis/correspondence_fusion_explorer.py
```

## File Locations

| What | Where |
|------|-------|
| Text embeddings (all cities) | `/data/overhead_matching/datasets/landmark_correspondence/eval_text_embeddings_all_cities.pkl` |
| Trained model (v5) | `/data/overhead_matching/training_outputs/landmark_correspondence/v5_all_cities/best_model.pt` |
| Raw P(match) scores | `/data/overhead_matching/datasets/VIGOR/<city>/correspondence_scores/v5_all_cities_raw.pt` |
| Similarity matrices | `/data/overhead_matching/datasets/VIGOR/<city>/similarity_matrices/correspondence_v5_hungarian_0.8.pt` |
| Evaluation paths | `/data/overhead_matching/evaluation/paths/mappilary/<city>.json` |
| Evaluation results | `/data/overhead_matching/evaluation/results/260325_correspondence_fusion/<city>/<mode>/` |

## City Reference

| City | Dataset path | Landmark version | Pano_v2 base |
|------|-------------|-----------------|--------------|
| Framingham | `VIGOR/mapillary/Framingham` | `Framingham_v1_260101` | mapillary |
| Gap | `VIGOR/mapillary/Gap` | `Gap_v1_250101` | mapillary |
| MiamiBeach | `VIGOR/mapillary/MiamiBeach` | `MiamiBeach_v1_150101` | mapillary |
| Middletown | `VIGOR/mapillary/Middletown` | `Middletown_v1_250101` | mapillary |
| Norway | `VIGOR/mapillary/Norway` | `Norway_v1_251201` | mapillary |
| post_hurricane_ian | `VIGOR/mapillary/post_hurricane_ian` | `post_hurricane_ian_v1_220101` | mapillary |
| SanFrancisco_mapillary | `VIGOR/mapillary/SanFrancisco_mapillary` | `SanFrancisco_mapillary_v1_220101` | mapillary |
| Boston | `VIGOR/Boston` | `boston` | mapillary + semantic_landmark_embeddings |
| nightdrive | `VIGOR/nightdrive` | `boston` | mapillary + semantic_landmark_embeddings |
