# Runbook: running the object-tracking pipeline

Ordered commands to process a leg end to end, or to re-run after a rule change.
Parameter meanings and the reasoning behind every default are in
[`object-tracking-pipeline.md`](object-tracking-pipeline.md); this file is the
sequence and the checkpoints.

All targets abbreviated `//…object_tracking:<name>` =
`//experimental/overhead_matching/swag/landmark_filtering/object_tracking:<name>`.

Set these once per shell:

```bash
export RUN=r004_full_leg1                     # new run name; never reuse
export DS=/data/farfield_matching/boston_harbor_dataset/processed/leg1
export LM=/data/farfield_matching/boston_harbor_dataset/panorama_landmarks/boston_harbor_leg1
export RUNS=/data/farfield_matching/boston_harbor_dataset/object_track_runs/m3_tracks/runs
export RD=$RUNS/$RUN
export FEATHER=/data/farfield_matching/boston_harbor_dataset/landmarks/harbor_osm_enc_v1.feather

export GOOGLE_CLOUD_PROJECT=rrg-dcist
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=True
# gcloud auth application-default login   # if ADC is stale
```

---

## Stage 0 — prerequisites

Per leg you need:

| input | path | notes |
|---|---|---|
| panoramas | `$DS/panorama/f####,<lat>,<lon>,.jpg` | filename carries the GPS fix; the pipeline parses it |
| GPS table | `$DS/frames_gps.csv` | `idx,video_t_s,sensor_elapsed_s,dist_m,latitude,longitude,altitude_m,speed_mps,frame_file`. **No course column** — course is derived |
| VLM detections | `$LM/sentences/results/**/predictions.jsonl` | Gemini batch output from the `osm_tags_farfield` extraction prompt |
| video | `…/videos/<leg>.mp4` | for SAM propagation between keyframes |
| SAM2 checkpoint | `/data/farfield_matching/models/sam2/sam2.1_hiera_large.pt` | |
| map catalog | `$FEATHER` | OSM + ENC, `landmark_type` ∈ {historical, enc} |

Landmark extraction itself (panorama → pinhole → Gemini → predictions) is the
`osm_tags_farfield` prompt in
`swag/model/semantic_landmark_extractor.py`, driven by
`swag/scripts/extract_gemini_landmarks_from_panoramas.py`. That is upstream of
this pipeline and documented with the extraction work.

Sanity check before spending GPU time:

```bash
ls $DS/panorama | wc -l                      # expect one per keyframe
python3 -c "import csv;print(sum(1 for _ in csv.DictReader(open('$DS/frames_gps.csv'))))"
bazel test //…object_tracking/...            # 7 targets, all should pass
```

---

## Stage 1 — geometry spot-check (new rig or new leg only)

Skip for a re-run of a known leg.

```bash
bazel run //…object_tracking:m0_render_boxes -- --dataset_base $DS --landmark_base $LM
bazel run //…object_tracking:m1_heading_windows -- --dataset_base $DS --landmark_base $LM
```

Check M0 boxes land on the objects named in the descriptions, and that M1's
compensated windows track the object rather than staring at open water (a
flipped heading-compensation sign does exactly that).

---

## Stage 2 — track building (M3)

The long pole: GPU, ~1 h for 379 keyframes.

```bash
bazel run //…object_tracking:m3_track_viewer -- \
    --run_name $RUN \
    --range full_leg1 0 378 \
    --notes "what changed in this run and why" \
    --dataset_base $DS --landmark_base $LM --runs_root $RUNS
```

- `--range NAME K_START K_END`, repeatable. Short ranges first when iterating.
- `--skip_existing_ranges` resumes a crashed run.
- `--notes` lands in `run_meta.json` and the diff viewer. Fill it in; it is the
  only record of *why* a run exists.

Writes `tracks_<range>.json`, per-track pages, thumbs, videos, `index.html`.

**Checkpoint.** Read the config actually recorded, not the config you think you
ran:

```bash
python3 -c "
import json;d=json.load(open('$RD/tracks_full_leg1.json'))
c=d['config'];print('tracks',len(d['tracks']),'rejected',len(d.get('rejected_births',[])))
print('context floor',c.get('superset_min_inter_over_box','ABSENT'))
print('weak complement',c.get('weak_min_complement','ABSENT'))
from collections import Counter
print(Counter(s['class'] for t in d['tracks'] for r in t['records'] for s in r.get('supports',[])))"
```

Expect a nonzero `context` count. If those keys are `ABSENT`, the run predates
the rules and its *tracking behaviour* is the old one even though viewers will
display recomputed classes.

Compare against a previous run:

```bash
bazel run //…object_tracking:m3_run_diff -- --run_a $RUNS/r003_full_leg1 --run_b $RD
```

---

## Stage 3 — keyframe viewer (optional, large)

```bash
bazel run //…object_tracking:keyframe_viewer -- --run_dir $RD
```

~380 pages, ~210 MB. Add `--kf_start/--kf_end` to limit. This is the tool for
"why did/didn't this detection do anything" — it shows births, birth
rejections, supports, and refused associations per detection.

---

## Stage 4 — semantic audit (M5)

```bash
bazel run //…object_tracking:m5_build_audit_requests -- \
    --run_dir $RD --dataset_base $DS --landmark_base $LM
```

Prints eligible tracks (≥3 supports) and chips rendered; writes
`requests.jsonl` (~20 MB, images inlined), `audit_meta.json`, `chips/`,
`preview/index.html`.

**Read `preview/index.html` before spending tokens.** It shows the exact prompt
and chips per call.

```bash
bazel run //experimental/overhead_matching/swag/scripts:vertex_batch_manager -- run-online \
    --input $RD/semantic_audit/requests.jsonl \
    --output $RD/semantic_audit/results.jsonl \
    --model gemini-3-flash-preview --parallel 8
```

Cost reference: 84 tracks ≈ 1.0 M tokens, ~4 min. Resumable — rerun the same
command to retry errors only.

```bash
bazel run //…object_tracking:m5_audit_results_viewer -- --run_dir $RD
```

Add `--no_extra_chips` to skip re-rendering strike/secondary chips.

**Checkpoint.** Verdict mix, and that names came back as weighted candidates:

```bash
python3 -c "
import json;from collections import Counter
v=Counter();multi=0;n=0
for l in open('$RD/semantic_audit/results.jsonl'):
    a=json.loads(json.loads(l)['response']['candidates'][0]['content']['parts'][0]['text'])
    v[a['verdict']]+=1;c=a['primary_object'].get('name_candidates',[]);n+=1
    if len(c)>1: multi+=1
print(v,'| tracks',n,'| multi-name',multi)"
```

Then eyeball `semantic_audit/review/index.html`: the **name/alias collision
panel** first (a name claimed by several tracks is either duplicate tracks or a
misidentification), then drops and `keep_partial`s.

---

## Stage 5 — merge (M6)

```bash
bazel run //…object_tracking:m6_merge_tracks -- \
    --run_dir $RD --dataset_base $DS --landmark_base $LM
```

Writes `merged/{landmarks,pair_stats,measurements}.json` and `index.html`.

Prints the pair-verdict histogram. **`merge_conflicts` should be 0**; a nonzero
count means a duplicate chain contradicted a cannot-link and the weakest edge
was dropped — inspect those pairs.

The `ambiguous` list is the adjudication queue: geometry deliberately declines
to decide there.

---

## Stage 6 — mount-offset calibration (per leg, before matching)

Required. The default 214.0° is **specific to boston_harbor leg1**.

Optional context first — renders the vessel sharply and gives a staticness mask:

```bash
bazel run //…object_tracking:bow_calibration -- \
    --dataset_base $DS --out_dir /tmp/bow_cal_$RUN --max_frames 120
```

The calibration proper is a map-free sweep: minimise median triangulation
residual over well-conditioned tracklets. Procedure:

1. Build per-keyframe poses: ENU from `frames_gps.csv` about the leg-mean
   anchor, course from a central difference (gate: >1.0 m of travel).
2. For each candidate offset, for each tracklet with ≥4 fused measurements,
   form world bearings `course + camera − offset` and call
   `bearing_matcher.triangulate`.
3. Keep only well-conditioned results (**condition < 500**) and take the median
   residual.
4. Sweep coarse (5° steps, 180–270) then refine (1°). Take the minimum.

Expect a **smooth unimodal curve**. If it is flat or multimodal, the bearings
or the poses are wrong — do not pick a minimum from noise. For leg1: 5.95° at
180°, 1.33° at 214°, 4.34° at 270°.

Cross-check against an independent hypothesis if you have a confidently named
landmark: `estimate_mount_offset()` reports per-tracklet implied offsets and
their spread. A rigid camera implies one constant, so a large spread across
tracklets means a wrong hypothesis or a drifting heading reference — not a
moving camera.

---

## Stage 7 — pairing labels (M7 → Vertex → M8)

```bash
bazel run //…object_tracking:m7_build_pairing_requests -- \
    --run_dir $RD --dataset_base $DS --feather $FEATHER \
    --mount_offset_deg 214.0 \
    --bearing_slack_deg 8 \
    --min_supports 3 --min_observations 3
```

- **`--mount_offset_deg`** from Stage 6. Wrong by 20° and the correct candidate
  falls outside the wedge entirely.
- **`--bearing_slack_deg 8`** once calibrated. Leave the 25° default only for an
  uncalibrated leg — at 25° a downtown wedge returns ~71 k candidates, i.e. no
  gating at all.
- **`--min_observations 3`** — below 3 the rays cannot be triangulated and the
  wedge is unconstrained; one 2-observation tracklet pulled 66,052 candidates.
- `--max_set2 500` (default) — recall over brevity.
- `--max_tracklets N` for a quick look before committing.

First invocation on a new anchor/feather builds the catalog cache (~227 s);
afterwards ~1.1 s.

**Read `pairing/index.html` before submitting** — wedge maps plus the filled
prompt per tracklet. Check the wedges converge and that plausible candidates
appear near the top of Set 2.

```bash
bazel run //experimental/overhead_matching/swag/scripts:vertex_batch_manager -- run-online \
    --input $RD/pairing/requests.jsonl \
    --output $RD/pairing/results.jsonl \
    --model gemini-3-flash-preview --parallel 8

bazel run //…object_tracking:m8_pairing_results_viewer -- --run_dir $RD
```

Review `pairing/review/index.html`: each match against the Set 2 bundle the
model actually saw, with **instance vs category** and the negatives. A wrong
label here becomes training data, so this is the last cheap place to catch it.

---

## Stage 8 — index and serve

```bash
bazel run //…object_tracking:run_index -- --run_dir $RD --runs_root $RUNS
```

Rewrites `$RD/index.html` as the stage landing page (preserving the m3 board as
`board.html`) and `$RUNS/index.html` as the all-runs index. Re-run it after any
stage so the counts stay current.

Serve the tree:

```bash
cd /data/farfield_matching/boston_harbor_dataset/object_track_runs && \
    python3 -m http.server 8935
```

Then `http://localhost:8935/m3_tracks/runs/index.html`.

---

## Archiving a run

Runs are versioned, so a new `--run_name` already preserves the old one. For an
off-machine snapshot, keep the expensive artifacts and drop the regenerable
images:

```bash
cd $RUNS && tar czf ../archive/${RUN}_artifacts.tgz \
    --exclude='keyframes/img' --exclude='semantic_audit/chips' \
    --exclude='videos' --exclude='_frames' --exclude='thumbs' \
    $RUN
```

Keeps every JSON, both `requests.jsonl`/`results.jsonl` pairs (the token spend)
and all HTML; ~28 MB for a full leg versus ~650 MB raw. Chips, keyframe images
and videos regenerate from the JSON.

---

## Adapting to a new dataset

1. **Paths.** Every module has `DEFAULT_DATASET` / `DEFAULT_LANDMARKS` /
   `DEFAULT_FEATHER` constants pointing at boston_harbor leg1. Pass the flags
   rather than editing the constants.
2. **Re-derive the mount offset** (Stage 6). Never inherit 214.0°.
3. **Range names** — `--range NAME K_START K_END`. The viewers glob
   `tracks_*.json`, and the merged `landmark_id`s embed track ids, so keep one
   range per run unless you want them mixed.
4. **Tag vocabulary.** `harbor_catalog.HARBOR_KEEP_KEYS` /
   `HARBOR_KEEP_PREFIXES` / `HARBOR_DROP_PREFIXES` are harbor-specific. A
   non-maritime environment should extend them (and **bump `CACHE_VERSION`**).
   Survey what is actually populated first:
   ```bash
   python3 -c "
   import pandas as pd
   d=pd.read_feather('$FEATHER'); nn=d.notna().sum()
   print(nn[nn>0].sort_values(ascending=False).head(40).to_string())"
   ```
5. **`position_sigma_m`** — `ENC_POSITION_SIGMA_M` 5 m, `OSM_POSITION_SIGMA_M`
   15 m. Adjust per map provenance; the filter turns this into angular
   uncertainty via `kappa_eff`.
6. **`SALIENT_KEYS` / `TALL_STRUCTURE_VALUES`** in `m7_build_pairing_requests`
   order Set 2. Add classes that are conspicuous in the new environment,
   especially any that lack height tags (this is why cranes needed an entry).
7. **The audit prompt's calibration notes** (`semantic_audit.SYSTEM_PROMPT`)
   cite maritime examples. The structure generalises; the examples should be
   revisited for a different environment.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| Correct map candidate absent from Set 2 | mount offset wrong; the wedge points elsewhere | recalibrate (Stage 6). A **constant** bearing error with small scatter is the signature |
| Wedge returns tens of thousands of candidates | slack too wide, or offset uncalibrated | calibrate, then `--bearing_slack_deg 8` |
| `nan` residual / condition | fewer than 3 fused observations, or degenerate geometry | `--min_observations 3` skips them |
| Catalog changes have no effect | stale cache | bump `harbor_catalog.CACHE_VERSION` |
| Landmark ids look like `osm:('node', 123)` | `id` column holds tuple *reprs* | `_id_text` handles it; bump `CACHE_VERSION` after touching it |
| Audit viewer `KeyError: 'name'` | results predate `name_candidates` | `parse_result_line` upgrades them; use it rather than reading raw JSON |
| Track appears to span several objects | context-class thief, or a containment-shaped thief the drift alarm cannot see | check `context` entries; if the run predates the floor, re-run tracking |
| Vertex stops after 3 errors | quota or auth | `gcloud auth application-default login`; rerun — it resumes |
| Bearings look inconsistent | probably real parallax, not error | use triangulation residual **and** condition number, never bearing spread |
| Track ids do not match an earlier run | ids are not stable across runs | join through `merged/landmarks.json` / `audit_meta.json` |
