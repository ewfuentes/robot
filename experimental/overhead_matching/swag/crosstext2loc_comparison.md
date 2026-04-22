# CVG-Text evaluation

The results directory `/data/overhead_matching/evaluation/results/cvgtext/`
holds the outputs of three retrieval comparisons on the CVG-Text benchmark
([yejy53/CVG-Text](https://github.com/yejy53/CVG-Text), ICCV 2025).

All evaluations use the **full per-city gallery** (16983 Brisbane / 9974
NewYork / 6107 Tokyo satellite tiles, and the same counts for the OSM
gallery) and the **1000 pano test split** per city. The 100-nearest-neighbour
location prior from the CVG-Text paper is not applied.

## Quick reproduction (assumes classifier / artifacts are already trained and staged)

Environment:
```
export GOOGLE_CLOUD_PROJECT=<your-project>
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=True
```

Project setup (one-time, per GCP project):
- Enable `aiplatform.googleapis.com`.
- In Vertex AI → Model Garden, enable `gemini-3-flash-preview` and
  `text-embedding-005` (or your chosen embedding model).
- Create a GCS bucket (any region works — we used `us-central1`):
  `gcloud storage buckets create gs://<bucket> --location=us-central1 --uniform-bucket-level-access`
- `gcloud auth application-default login` and
  `gcloud auth application-default set-quota-project <project>`.

CVG-Text data: the dataset itself is not downloadable by this pipeline.
Pull `LHL3341/CVG-Text_full` from HuggingFace (panos + sat/OSM imagery +
annotations all ship in the HF repo) and place at
`/data/overhead_matching/datasets/cvgtext/`:

```
uvx --from huggingface_hub hf download --repo-type dataset \
  LHL3341/CVG-Text_full \
  --local-dir /data/overhead_matching/datasets/cvgtext
```

Given that, one-shot stage CVG-Text into the VIGOR-shaped directory layout
this pipeline expects:

```
bazel run //experimental/overhead_matching/swag/scripts:stage_cvgtext_for_vigor
```

The script writes `satellite_bbox.json`, populates `satellite/` (full
gallery, VIGOR-format symlinks) and `panorama/` (test-only, VIGOR-format
symlinks) per city under `/data/overhead_matching/datasets/VIGOR/cvgtext_<City>/`.

**Always stage before running Gemini.** Point
`extract_gemini_landmarks_from_panoramas.py --panorama_dir` at
`<vigor_root>/cvgtext_<City>/panorama/` (the staged dir with VIGOR-format
filenames) so the resulting pickle keys line up with VigorDataset's pano_ids
automatically. Running Gemini on the raw CVG-Text dir first produces
pickle keys that silently fail the correspondence pipeline's key lookup
(all pano landmarks drop, recall ~0.1%).

If you're re-running after retraining the correspondence classifier and
nothing else changed, only Step 2.5 below needs to re-run (~3 min total
for all three cities). `--from_raw` makes parameter sweeps near-free.

## Directory layout

```
/data/overhead_matching/evaluation/results/cvgtext/
├── comparison_tables.txt                  ← rendered comparison tables, MRR-first
├── crosstext2loc_train-<X>_test-<Y>_<kind>/
│   ├── similarity.pt                      (num_queries, num_gallery) cosine sim
│   ├── metrics.json                       {recall@1, recall@5, recall@10, mrr}
│   └── config.json                        run metadata
├── expA_test-<City>_260218_093851_all_chicago_dinov3_wag_bs18_v2/
│   ├── similarity.pt
│   ├── metrics.json
│   └── config.json
└── expB_<City>/
    ├── simple_v1_raw.pt                   metadata dict (pano_id→lm_rows, cost_matrix_path, …)
    ├── simple_v1_raw_cost_matrix.npy      (total_pano_lms, total_osm_lms) P(match)
    └── simple_v1_raw_similarity.pt        (num_panos, num_sats) similarity
```

For `crosstext2loc_train-X_test-Y_<kind>`: `X,Y ∈ {Brisbane, NewYork, Tokyo}`,
`kind ∈ {sat, osm}` → 3×3×2 = 18 directories.

## Step 0 — CrossText2Loc baseline (full gallery, all 18 train/test/kind combos)

**Checkpoints**: the 6 `long_model_<City>-mixed_1e-05_128_{sat,osm}_epoch*_*.pth`
files are hosted on the official `CVG-Text/CrossText2Loc` HuggingFace model
repo (not bundled with the `LHL3341/CVG-Text_full` dataset repo). Grab
them into `<cvgtext_root>/models/`:

```
uvx --from huggingface_hub hf download \
  --repo-type model CVG-Text/CrossText2Loc \
  --local-dir /data/overhead_matching/datasets/cvgtext/models
```

`evaluate_cvgtext_crosstext2loc.py` globs that directory with
`long_model_{train_city}-mixed_1e-05_128_{kind}_epoch*_*.pth`, so any
unrelated files in `models/` are ignored but the filename pattern must
be preserved as published.

**Driver**: `experimental/overhead_matching/swag/scripts/evaluate_cvgtext_crosstext2loc.py`.
Takes `--train_city X --test_city Y --gallery_kind {sat,osm}` and writes
`similarity.pt` + `metrics.json` + `config.json` per `{train × test × kind}`
triple (18 directories total).

The upstream `crosstext2loc` package is pinned in
`third_party/python/requirements_3_12.in` as
`crosstext2loc @ git+…@c3b5727…`; the driver calls
`crosstext2loc.load_model(...)` directly. The `long_model_*.pth`
checkpoints are CLIP ViT-L/14@336 with text positional embedding
interpolated from 77 → 300 tokens.

**Vertex AI setup**:
1. `gcloud auth login` + `gcloud auth application-default login`
2. `gcloud config set project <project>`
3. `gcloud storage buckets create gs://<name> --location=us-central1 --uniform-bucket-level-access`
   (any GCS region works; `GOOGLE_CLOUD_LOCATION=global` is the Vertex
   *endpoint* setting and is independent of the bucket region)
4. Ensure `gemini-3-flash-preview` and the relevant text-embedding model are
   enabled in the project's Vertex AI Model Garden.

## Step 1 — WAG pano→satellite (Experiment A)

**Driver**: `experimental/overhead_matching/swag/scripts/evaluate_cvgtext_swag.py`.
Loads pano + sat `WagPatchEmbedding` weights via
`load_models_from_training_output`, runs `evaluate_swag.compute_similarity_matrix`,
writes metrics. Per-city output lives under `expA_test-<City>_<wag_tag>/`.

**WAG checkpoint**:
`/data/overhead_matching/training_outputs/260215_baseline_retraining/260218_093851_all_chicago_dinov3_wag_bs18_v2/`.
DINOv3 backbone, `patch_dims=[320,320]` sat, `[320,640]` pano, trained on
Chicago only (zero-shot on Brisbane / NewYork / Tokyo).

**Dataset adapter**: `CVGTextDataset` exposes `get_pano_view()` /
`get_sat_patch_view()` matching `VigorDataset`'s nested-torch-Dataset
pattern, with a `_MinimalConfig` stub so `vigor_dataset.get_dataloader`'s
`worker_init_fn` runs unchanged.

**Sanity checks in the driver**:
1. **Dataset-init filename GT** — asserts each query's
   `positive_satellite_idxs[0]` points at a gallery row whose filename
   matches the query.
2. **Embedding row-order check** — builds sat and pano embedding matrices,
   asserts `sim[i,i]` is tied for the row max on both sides (stronger
   invariant than strict `argmax==i`, which false-positives on CVG-Text's
   byte-identical duplicate tiles).

CVG-Text has 6 duplicate-sat-tile pairs (Brisbane 2, NewYork 0, Tokyo 4):
byte-identical PNGs with slightly-different filename lat/lon. Doesn't
meaningfully affect retrieval.

## Step 2 — Correspondence-classifier pano→OSM-landmark (Experiment B)

Uses the `simple_v1` text-only correspondence classifier trained on
Chicago + Seattle.

### 2.1 — Gemini landmark extraction on CVG-Text test panos

**Prerequisite**: `stage_cvgtext_for_vigor.py` has run so the staged
`panorama/` dir (with VIGOR-format symlink names) exists per city.

**What to run**, per city:

```
bazel run //experimental/overhead_matching/swag/scripts:extract_gemini_landmarks_from_panoramas -- \
  --name <City> \
  --panorama_dir /data/overhead_matching/datasets/VIGOR/cvgtext_<City>/panorama \
  --output_base <pano_v2_base> \
  --gcs_bucket <bucket> \
  --media_resolution MEDIA_RESOLUTION_ULTRA_HIGH \
  --force
```

`<pano_v2_base>` is the directory under which per-city pickles are written
(e.g. `/data/overhead_matching/datasets/semantic_landmark_embeddings/panov2_tuned_prompt_cvgtext/`).
The orchestrator nests a `<City>/` subdir under `<pano_v2_base>` (driven by
`--name`), matching what `extract_panorama_data_across_cities` walks.

Settings used:

- `--model gemini-3-flash-preview` (default)
- `--prompt_type osm_tags` (default)
- `--media_resolution MEDIA_RESOLUTION_ULTRA_HIGH` (override the default HIGH)
- `--thinking_level HIGH` (default)

`GOOGLE_CLOUD_LOCATION=global` is required — batch prediction for
`gemini-3-flash-preview` is only available at the `global` endpoint;
region-specific endpoints return `MODEL_NOT_SUPPORTED_FOR_BATCH`.

7-stage pipeline: PINHOLE → REQUESTS → UPLOAD → SUBMIT → WAIT → DOWNLOAD → EMBEDDINGS.
Each run takes ~70-80 min end-to-end (mostly Vertex batch queue).

**Artifacts written under** `<pano_v2_base>/<City>/`:

```
├── sentence_requests/panorama_sentence_requests/panorama_request_000.jsonl  (batch input)
├── sentences/results/panorama_request_000/prediction-model-<ts>/predictions.jsonl  (batch output)
├── embeddings/embeddings.pkl  (per-city pickle with 1536-dim Gemini text embeddings)
└── submitted_job_names.txt
```

Pano-landmark counts (post-filter, per city):

| city | mean landmarks/pano | % panos with 0 landmarks |
|---|---|---|
| NewYork | 3.03 | 6.2% |
| Tokyo | 2.75 | 7.7% |
| Brisbane | 1.40 | 38.3% (suburban bias — Gemini prompt correctly rejects generic houses) |

### 2.2 — Historical OSM landmark extraction

Downloaded Geofabrik historical PBF snapshots for the dates bracketing
the pano capture times (panos span 2022-05 through 2023-10). All from
the public `download.geofabrik.de` archive (no auth required):

- Brisbane — `https://download.geofabrik.de/australia-oceania/australia-230101.osm.pbf` (~601 MB)
- NewYork — `https://download.geofabrik.de/north-america/us/new-york-230101.osm.pbf` (~396 MB)
- Tokyo — `https://download.geofabrik.de/asia/japan-230101.osm.pbf` (~1.70 GB)

All placed in `/data/overhead_matching/datasets/osm_dumps/`.

**Driver**: `experimental/overhead_matching/swag/scripts/extract_landmarks_historical.py`,
run per city with `--dataset_path <staging_root>` (reads `satellite_bbox.json`)
and `--pbf_file <geofabrik_pbf>`.

**Pre-requisite**: staging dir with `satellite_bbox.json` computed from the
sat-tile lat/lon extents + 500 m buffer. See §2.4 for the full staging layout.

**Feather counts**:

| city | raw OSM features | unique values after prune | feather size |
|---|---|---|---|
| Brisbane | 243,211 | 29,024 | 40 MB |
| NewYork | 200,081 | 85,880 | 31 MB |
| Tokyo | 127,611 | 45,824 | 17 MB |

Brisbane's huge bbox (~20×30 km) produces 1.2× as many raw features as
NewYork but far fewer *distinct* tag values after `prune_landmark` — a
lot of Brisbane OSM is cookie-cutter residential features with no
distinguishing name/housenumber.

### 2.3 — Text embedding extension

The classifier expects 768-dim `text-embedding-005` embeddings. The base
pickle `/data/overhead_matching/datasets/landmark_correspondence/eval_text_embeddings_panov2_tuned_v3_all.pkl`
has 152,871 entries from Chicago + Seattle + VIGOR mapillary cities and
doesn't cover Tokyo / Brisbane OSM tag values or CVG-Text pano landmarks.

**Driver**: `experimental/overhead_matching/swag/scripts/precompute_value_embeddings.py`
with `--pano_v2_base` (all 3 CVG-Text `cvgtext_pano_v2_base` symlinks),
`--feather_dirs` (the 3 staging dirs), `--base_embeddings` pointing at
the base pickle, and the default `--model text-embedding-005
--output_dimensionality 768`. Runs in ~12 min.

**Output**: 219,712 entries (152,871 base + 66,841 new CVG-Text values)
at `/data/overhead_matching/datasets/landmark_correspondence/eval_text_embeddings_panov2_tuned_v3_all_plus_cvgtext.pkl`
(656 MB).

### 2.4 — VIGOR-compatible staging

Handled by `stage_cvgtext_for_vigor.py` (run as step 1 of the Quick
reproduction block above). Per city:

```
/data/overhead_matching/datasets/VIGOR/cvgtext_<City>/
├── satellite_bbox.json            (computed from sat gallery lat/lon + 500m buffer)
├── satellite/                     (full gallery, VIGOR-format symlinks)
├── panorama/                      (test-only, VIGOR-format symlinks)
└── landmarks/
    └── cvgtext_<City>_v1_230101.feather   (populated by step 2.2)
```

`VigorDataset.load_satellite_metadata` and `load_panorama_metadata` hardcode
specific filename patterns:

- sat: `<anything>_<lat>_<lon>.<ext>`
- pano: `<panoid>,<lat>,<lon>,.<ext>`

CVG-Text's native `<lat>,<lon>_<date>_<panoid>_d<yaw>_z<zoom>.<ext>` fails
both, so the staging script creates a symlink layer with VIGOR-format names.
The panos and pinhole outputs carry those names forward into the pickle
keys — which is why step 2.1 must run on the staged `panorama/` dir.

**Brisbane orphan**: one of the 16983 Brisbane panos
(`-27.39043692,152.95798883_2022-12_1_VLlX4egz-3UZkwiMMWtA_d261_z3.png`)
has no matching sat tile in the gallery — a CVG-Text data quirk.
VigorDataset's `_verify_all_panos_have_satellites` refuses this. Since
it's a train-set pano (not in `annotation/Brisbane/test.json`), the staging
script's restriction of `panorama/` to the 1000 test panos already excludes
the orphan.

### 2.5 — Correspondence similarity

**Command** (per city):

```bash
GOOGLE_CLOUD_PROJECT=<project> GOOGLE_CLOUD_LOCATION=global GOOGLE_GENAI_USE_VERTEXAI=True \
bazel run //experimental/overhead_matching/swag/scripts:export_correspondence_similarity -- \
  --model_path /data/overhead_matching/training_outputs/landmark_correspondence/simple_v1/best_model.pt \
  --text_embeddings_path /data/overhead_matching/datasets/landmark_correspondence/eval_text_embeddings_panov2_tuned_v3_all_plus_cvgtext.pkl \
  --dataset_path /data/overhead_matching/datasets/VIGOR/cvgtext_<City> \
  --pano_v2_base <pano_v2_base> \
  --output_path /data/overhead_matching/evaluation/results/cvgtext/expB_<City>/simple_v1_raw.pt \
  --compute_similarity --prob_threshold 0.8 --uniqueness_weighted
```

`<pano_v2_base>` is the same value passed to Gemini in step 2.1
(`<pano_v2_base>/<City>/embeddings/embeddings.pkl` exists).

Settings for `simple_v1`:
- `--prob_threshold 0.8` (Hungarian matching threshold)
- `--uniqueness_weighted` (each matched pair weighted by `1/log2(1 + n_matches_above_threshold)`)
- `--method hungarian` (default)
- `--aggregation sum` (default)

Each city's run takes ~1 min (1000 panos × 6-17k sats, GPU inference).

Metric computation happens in the same script (`retrieval_metrics.compute_top_k_metrics`
on the Hungarian-aggregated similarity matrix, filename-based single-positive GT).

## Summary of results (MRR, fair comparisons only)

All zero-shot / cross-region numbers — no method trained on the test city.

| test city | best fair method | MRR | Exp B MRR |
|---|---|---|---|
| Brisbane | CT2L text→OSM (NewYork-trained) | 0.081 | 0.072 |
| NewYork | **Exp B pano→OSM-landmark** | **0.231** | 0.231 |
| Tokyo | WAG pano→sat (Chicago-trained) | 0.142 | 0.028 |

See `comparison_tables.txt` in the results directory for the full per-city
ranking including R@1/5/10.

**Headline** (fair comparisons):

- **NewYork** — raw landmarks > rasterised OSM. Exp B (pano → OSM
  landmarks, Chicago+Seattle-trained classifier) beats both CT2L
  text→OSM cross-region runs (16.4, 15.8) at R@1 (18.0) and wins on MRR
  (0.231 vs 0.223, 0.218).

- **Brisbane** — ~tie. Exp B and CT2L cross-region OSM are within noise
  of each other (R@1 5.1 vs 4.9; MRR 0.072 vs 0.081). Both limited by
  Brisbane's 38% empty-landmark rate and sparse OSM unique values.

- **Tokyo** — Exp B loses (MRR 0.028 vs CT2L cross-region 0.073, 0.046).
  Likely explanation: `simple_v1` trained only on English
  Chicago+Seattle OSM, so it can't bridge kanji / romanized-Japanese
  name variants the way CLIP's multilingual pretraining does.

## Gotchas worth knowing before you touch this pipeline

1. **Vertex batch for `gemini-3-flash-preview` only works at
   `locations/global`.** Submitting a batch job under `us-central1` returns
   `MODEL_NOT_SUPPORTED_FOR_BATCH` even though online / `generateContent`
   works there. Set `GOOGLE_CLOUD_LOCATION=global` for Exp B Gemini runs.
2. **Fresh GCP projects have tight text-embedding rate limits.**
   `precompute_value_embeddings.py` retries 10 times with backoff capped
   at 60 s to ride through per-minute quota windows. If you re-run on a
   new project and still hit 429s, bump further.
3. **VigorDataset's filename parsers are format-specific.**
   `load_satellite_metadata` does `_, lat, lon = p.stem.split("_")`;
   `load_panorama_metadata` does `pano_id, lat, lon, _ = p.stem.split(",")`.
   CVG-Text's native `<lat>,<lon>_<date>_<panoid>_d<yaw>_z<zoom>.<ext>`
   fails both. `stage_cvgtext_for_vigor.py` creates a symlink layer with
   VIGOR-format names; don't point `VigorDataset` directly at the CVG-Text
   source dirs.
4. **`extract_tags_from_pano_data` strips `pano_id.split(",")[0]`.**
   This means the pano_v2 pickle keys must be in VIGOR-shape
   (`<panoid>_<date>_d<yaw>,<lat>,<lon>,`) so the first comma-separated
   field matches the VigorDataset pano_id. The fix is to run Gemini
   (step 2.1) on the staged `panorama/` dir — the VIGOR-format filenames
   carry through to the stage-7 pickle keys automatically. Running
   Gemini on the raw CVG-Text dir produces broken pickle keys and the
   correspondence pipeline will silently find 0 pano landmarks.
5. **`VigorDataset._verify_all_panos_have_satellites` rejects any pano
   without a geometric-near sat.** CVG-Text Brisbane has at least one
   train-set pano without a matching sat tile; the staging script already
   restricts `panorama/` to test-split panos, which drops the orphan.
6. **Two pickles, two embedding models.**
   `panov2_tuned_prompt/<City>/embeddings/embeddings.pkl` (from Stage 7
   of the Gemini orchestrator) is **1536-dim `gemini-embedding-001`** and
   is *not* consumed by the correspondence classifier — it's side data
   for other downstream uses. The classifier consumes
   `eval_text_embeddings_panov2_tuned_v3_all*.pkl` which is
   **768-dim `text-embedding-005`** built by `precompute_value_embeddings.py`.
   Easy to waste time chasing a dim mismatch if you don't know which is
   which.
7. **`--from_raw` for cheap parameter sweeps.** Each Exp B run writes
   `<tag>_raw.pt` + `<tag>_raw_cost_matrix.npy` (the P(match) output). To
   sweep `--prob_threshold`, `--aggregation`, or `--uniqueness_weighted`
   without re-running the expensive P(match) inference, pass
   `--from_raw <path>/<tag>_raw.pt --compute_similarity ...` to load the
   cached cost matrix and only re-aggregate. Seconds, no GPU.

## Future work / knobs to sweep

1. **Param sweep on Exp B** for Tokyo: lower `--prob_threshold` (0.8 →
   0.5 / 0.3), try `--aggregation max` / `log_odds`, toggle
   `--uniqueness_weighted`. Unlikely to close the Tokyo gap fully but
   worth a data point.
2. **Multilingual embedding model for Tokyo**: re-run
   `precompute_value_embeddings.py` with `gemini-embedding-001` or a
   multilingual `text-embedding-*` variant and retrain the classifier,
   or use a classifier that cross-embeds Japanese↔romanized pairs.
3. **Bootstrap confidence intervals** on recall@k and MRR (n=1000 test
   queries → σ on R@k around 0.5–1.5 pp). Closes the Brisbane question
   ("tie" vs "slight win").
4. **CrossText2Loc OSM in-distribution**: reproducing their paper's
   100-nearest-prior numbers would give an upper bound for the CT2L OSM
   baseline, not relevant for our fair comparison but useful as a
   sanity anchor.
