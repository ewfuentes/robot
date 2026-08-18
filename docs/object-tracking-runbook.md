# Runbook: running the object-tracking pipeline

Ordered commands to process a leg end to end, or to re-run after a rule change.
Parameter meanings and the reasoning behind every default are in
[`object-tracking-pipeline.md`](object-tracking-pipeline.md); this file is the
sequence and the checkpoints.

## The short version

```bash
bazel run //…object_tracking:run_pipeline -- \
    --dataset boston_harbor_leg2 --run_name r001_full
```

That runs the whole sequence below — extract → boxes → tracks → audit → review →
merge → offset → match → matchview → index — skipping stages whose output already
exists, and leaves every viewer on disk to read afterwards. The reason it does
not block for a human at each stage is that the sequence is **cheap**: measured
on leg1, the entire LLM bill is $26.24 at on-demand list price and about half
that through the Batch API.

| stage | calls | tokens | USD (on-demand list) |
|---|---|---|---|
| extraction | 379 | 6.07 M | $14.53 |
| semantic audit | 105 | 1.17 M | $2.53 |
| matching | 176 | 3.59 M | $9.18 |
| **total** | **660** | **10.83 M** | **$26.24** |

Read from the `usageMetadata` stored in every response, so it is a measurement,
not an estimate — `run_pipeline` prints the same tally for whatever it ran. The
GPU tracking stage (~1 h per 379 keyframes) is the real cost, and it is time.

Two conditions stop the run, because continuing past either yields confident
nonsense rather than an error: an **incomplete extraction** (frames with no VLM
response read downstream as frames with no objects) and a **FLAT or MULTIMODAL
mount-offset curve** (matching would aim its bearings using an offset taken from
noise). Everything else prints and carries on.

`--from` / `--to` / `--only` / `--skip` select a slice, `--force` redoes a
completed stage, `--dry_run` prints the commands. `pipeline_status --dataset X`
reports what exists without running anything.

The rest of this file is the same sequence stage by stage, for when a stage needs
to be run or understood on its own.

All targets abbreviated `//…object_tracking:<name>` =
`//experimental/overhead_matching/swag/landmark_filtering/object_tracking:<name>`.

Every stage resolves its paths from **one flag, `--dataset`**, through
`swag/data/farfield_paths.py`, which encodes the disk layout once. Set these per
shell:

```bash
export DATASET=boston_harbor_leg2             # the only path-ish thing you pick
export RUN=r001_full_leg2                     # new run name; never reuse
export RANGE=full_leg2                        # range name used below

# Only for the shell snippets that read outputs back; the tools resolve these
# themselves and never need them passed.
export RUNS=/data/farfield_matching/artifacts/object_tracks/$DATASET/v1/m3_tracks/runs
export RD=$RUNS/$RUN

export GOOGLE_CLOUD_PROJECT=rrg-dcist
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=True
# gcloud auth application-default login   # if ADC is stale
```

**Model calls default to the Batch API**, which is half the price of on-demand
for identical output; the trade is latency, minutes becoming up to a day. Pass
`--online` (on `run_pipeline`, `m5_build_audit_requests` or
`m9_match_landmarks`) to swap back to synchronous calls when turnaround matters
more than the discount. Both transports write the same records and both are
resumable, so a rerun retries only the failures either way.

Later stages take `--run_dir` and infer the dataset **from the run's own path**,
so they need no `--dataset` at all; passing one that disagrees with the run dir
is an error rather than a silent mismatch. Every resolved path can still be
overridden individually (`--dataset_base`, `--landmark_base`, `--video`,
`--feather`, `--checkpoint`) for ad-hoc work.

To see what a dataset resolves to before running anything, any stage will print
its missing inputs and stop:

```bash
bazel run //…object_tracking:m3_track_viewer -- --dataset $DATASET --run_name probe
```

---

## Stage 0 — prerequisites

Resolved from `--dataset`; nothing here is a path you type:

| input | resolves to | notes |
|---|---|---|
| panoramas | `datasets/<ds>/panorama/f####,<lat>,<lon>,.jpg` | filename carries the GPS fix; the pipeline parses it |
| GPS table | `datasets/<ds>/frames_gps.csv` | `idx,video_t_s,sensor_elapsed_s,dist_m,latitude,longitude,altitude_m,speed_mps,frame_file`. **No course column** — course is derived |
| VLM detections | `artifacts/frame_landmarks/<ds>/v1/sentences/results/**/predictions.jsonl` | see Stage 0b; nothing downstream runs without it |
| pinhole faces | `artifacts/pinhole_images/<ds>/v1/<stem>/yaw_{000,090,180,270}.jpg` | produced by Stage 0b as a first-class artifact |
| video | `video.source_video` in `datasets/<ds>/pipeline_metadata.json` | for SAM propagation between keyframes. The **dataset** states which video is its own; the filename cannot be derived from the dataset name |
| SAM2 checkpoint | `models/sam2/sam2.1_hiera_large.pt` | |
| map catalog | `datasets/<ds>/landmarks/v1_trimmed.feather` | OSM + ENC, `landmark_type` ∈ {historical, enc} |

Sanity check before spending GPU time:

```bash
bazel test //…object_tracking/... //experimental/overhead_matching/swag/data:farfield_paths_test
```

### Stage 0b — landmark extraction (pinholes + Gemini)

Both upstream artifacts come from one command. Pinhole rendering is stage 1 of
that pipeline, not a manual prerequisite: it writes
`artifacts/pinhole_images/<ds>/v1` plus its manifest, verifies an existing render
against the requested resolution instead of re-deriving it, and reuses it when it
matches.

```bash
bazel run //experimental/overhead_matching/swag/scripts:extract_gemini_landmarks_from_panoramas -- \
    --dataset $DATASET \
    --prompt_type osm_tags_farfield \
    --pinhole_resolution 2048 \
    --media_resolution MEDIA_RESOLUTION_ULTRA_HIGH \
    --model gemini-3.1-pro-preview
```

Six stages by default: pinhole → requests → upload → submit → wait → download.
**Stage 7 (embeddings) is opt-in** (`--with_embeddings`) because nothing in the
tracking pipeline reads `embeddings.pkl` — `ingest.py` reads
`predictions.jsonl`; only the older cosine matcher in
`landmark_filtering/semantic_similarity.py` wants it. `--start_stage` /
`--end_stage` resume. Per-part ULTRA_HIGH only applies when the flag is exactly
`MEDIA_RESOLUTION_ULTRA_HIGH`; other values go into `generationConfig` instead.

**Stage 6 verifies coverage and stops on an incomplete artifact.** A Vertex batch
job reports success at the job level while individual requests fail: leg2 lost 23
of 236 frames to transient `TPU device returned error`. Nothing downstream
objects — `ingest` skips a frame with no prediction with a bare `continue`, so
tracking reads those frames as containing no objects and starves tracks crossing
them. Repair the gap rather than accepting it:

```bash
# see the breakdown for an existing extraction
bazel run ...:extract_gemini_landmarks_from_panoramas -- --dataset $DATASET --validate_only

# re-run only the failed requests; results are written as an ADDITIONAL
# predictions file that supersedes the failed attempt (ingest builds a dict over
# a sorted glob), so nothing already on disk is overwritten
bazel run ...:extract_gemini_landmarks_from_panoramas -- --dataset $DATASET \
    --model gemini-3.1-pro-preview --retry_failed
```

`--allow_incomplete` proceeds anyway and records the gap in the manifest
(`complete: false` plus `missing_keys`), so an accepted gap stays visible to
whoever reads the artifact later.

Cost reference: leg1's 379 panoramas took ~8 min on 3.1-pro at 2048 px; leg2's
236 took 24 min of batch wait, and the 23-frame retry 4 min / 344 k tokens.

---

## Stage 1 — geometry spot-check (new rig or new leg only)

Skip for a re-run of a known leg.

```bash
bazel run //…object_tracking:m0_render_boxes -- --dataset $DATASET
bazel run //…object_tracking:m1_heading_windows -- --dataset $DATASET
```

Check M0 boxes land on the objects named in the descriptions, and that M1's
compensated windows track the object rather than staring at open water (a
flipped heading-compensation sign does exactly that).

Both tools carry anchor cases curated on leg1. Observation ids are per-dataset,
so on another leg M1 falls back to anchors chosen from the data itself — ranked
by heading change across the span, because the strip only discriminates where the
boat actually turns. `--auto_cases` forces that even on leg1.

---

## Stage 2 — track building (M3)

The long pole: GPU, ~1 h for 379 keyframes.

```bash
bazel run //…object_tracking:m3_track_viewer -- \
    --dataset $DATASET \
    --run_name $RUN \
    --range $RANGE 0 235 \
    --notes "what changed in this run and why"
```

`--range` end indices are checked against the dataset's actual last keyframe, so
reusing another leg's range numbers fails loudly instead of tracking a truncated
leg. `run_meta.json` records the resolved dataset, landmarks and video, which is
how you answer "which video did this run use" afterwards.

- `--range NAME K_START K_END`, repeatable. Short ranges first when iterating.
- `--skip_existing_ranges` resumes a crashed run.
- `--notes` lands in `run_meta.json` and the diff viewer. Fill it in; it is the
  only record of *why* a run exists.

Writes `tracks_<range>.json`, per-track pages, thumbs, videos, `index.html`.

**Checkpoint.** Read the config actually recorded, not the config you think you
ran:

```bash
python3 -c "
import json;d=json.load(open('$RD/tracks_$RANGE.json'))
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
bazel run //…object_tracking:m3_run_diff -- --run_a $RUNS/<previous_run> --run_b $RD
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
bazel run //…object_tracking:m5_build_audit_requests -- --run_dir $RD
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
bazel run //…object_tracking:m6_merge_tracks -- --run_dir $RD
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
    --dataset $DATASET --out_dir /tmp/bow_cal_$RUN --max_frames 120
```

The calibration proper is now a target, `mount_offset_sweep` — a map-free sweep
that minimises the median triangulation residual over well-conditioned
tracklets. It needs Stage 5's `merged/measurements.json`, which stores raw
camera bearings precisely because the offset has not been applied yet:

```bash
bazel run //…object_tracking:mount_offset_sweep -- --run_dir $RD
```

It sweeps the full circle at 5°, refines ±6° at 1°, prints the curve as a
log-scaled sparkline, judges the curve's shape, and writes
`$RD/mount_offset_sweep.json`. Add `--write_metadata` to record the result in
the dataset's `pipeline_metadata.json` (which also invalidates that dataset's
`checksums.sha256` — regenerate it).

**Three gates, and the third is the subtle one.** Condition number (<500) and
≥4 observations per tracklet are the obvious ones. The third exists because the
condition gate is applied *per candidate offset*, so the number of surviving
tracklets is itself a function of the offset — a badly wrong offset drops nearly
everything, and the few survivors can post an excellent median residual purely
by selection. On leg1 the ungated argmin is **85° at 1.04° over 5 tracklets**,
while the true basin at 210° sits at **1.05° over 37**. A candidate must
therefore retain at least `--min_support_frac` (0.5) of the best-supported
candidate's tracklets. Under-supported candidates print with a `~`.

Expect a **smooth unimodal curve**; the tool says `SMOOTH UNIMODAL`, `FLAT`, or
`MULTIMODAL` and refuses to write a `mount_offset_deg` for the latter two. If it
is flat or multimodal, the bearings or the poses are wrong — do not pick a
minimum from noise.

Reference: on leg1's `r003_full_leg1` the sweep returns **212.0°**, median
residual 0.94° over 37 of 46 tracklets, `SMOOTH UNIMODAL` with 4.0× contrast.
The leg's accepted value is 214°, whose own surveyed-building validation carried
std 2.42°, so the two agree.

The sweep is self-consistent and map-free, which is **not** the same as
externally validated, and the metadata it writes says so
(`accuracy_validated: false`). To cross-check independently, hypothesise a
confidently named landmark and call `estimate_mount_offset()`: it reports
per-tracklet implied offsets and their spread. A rigid camera implies one
constant, so a large spread across tracklets means a wrong hypothesis or a
drifting heading reference — not a moving camera.

---

## Stage 7 — matching (M9 → Vertex → M9 aggregate → M10)

```bash
bazel run //…object_tracking:m9_match_landmarks -- --run_dir $RD --build_only
```

The catalog resolves from the dataset the run dir belongs to, so this needs no
`--feather`.

Builds one request per (tracklet batch × map-signature chunk). Defaults:
`--query_batch 10`, `--chunk_size 500`, `--thinking_level HIGH`, catalog from
the dataset's `landmarks/v1_trimmed.feather`. On leg 1 that is 176 requests over 7,950
signatures for ~102 tracklets.

**No spatial gating.** Set 2 is the whole map. Do not reintroduce a
position-based shortlist — see the removal note in the pipeline doc.

Or in one step — `--submit` executes the requests and carries straight on to
aggregation, through the Batch API by default:

```bash
bazel run //…object_tracking:m9_match_landmarks -- --run_dir $RD --submit
bazel run //…object_tracking:m10_match_viewer -- --run_dir $RD
```

`--online` swaps to synchronous calls (faster, twice the price). To drive the
transport by hand instead, `vertex_batch_manager` has `run-batch` and
`run-online` with the same contract — local requests JSONL in, local results
JSONL out, resumable — then `--aggregate_only` picks the results up:

```bash
bazel run //…swag/scripts:vertex_batch_manager -- run-batch \
    --input $RD/matching/requests.jsonl --output $RD/matching/results.jsonl \
    --model gemini-3-flash-preview --gcs_prefix gs://crossview/batch_stages
bazel run //…object_tracking:m9_match_landmarks -- --run_dir $RD --aggregate_only
```

Costs on leg 1: 176 requests, ~3.6M tokens, ~15 min.

Knobs worth knowing:
- `--instance_max_rows 5` — a signature covering more rows than this cannot be
  an `instance` match by definition, so the label is downgraded in code. Fired
  160 times on leg 1.
- `--confidence_floor 0.05` — matches below this never reach the table.
- Global `no_match_confidence` is **derived** as `1 − best match confidence`,
  not fused from the per-slice values. Per-slice values answer "is it in THIS
  slice", which is trivially yes-it-is-not for ~15 of 16 slices, so fusing them
  cannot recover the global question. Both are recorded.

**Read `matching/review/index.html` before trusting the table** — each
tracklet beside its matches, with expansion width flagged.


## Stage 8 — index and serve

```bash
bazel run //…object_tracking:run_index -- --run_dir $RD --runs_root $RUNS
```

Rewrites `$RD/index.html` as the stage landing page (preserving the m3 board as
`board.html`) and `$RUNS/index.html` as the all-runs index. Re-run it after any
stage so the counts stay current.

Serve the tree:

```bash
cd /data/farfield_matching/artifacts/object_tracks/boston_harbor_leg1/v1 && \
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

1. **Paths.** Pass `--dataset <name>`; everything resolves from the layout via
   `swag/data/farfield_paths.py`. There are no per-module path constants any
   more — they used to default to boston_harbor leg1, so a stage handed one
   leg's `--dataset_base` would still read leg1's *video* and build tracks from
   the wrong imagery without complaining. If a new lane or artifact kind is
   needed, add it to `farfield_paths`, not to a stage.
2. **Extract landmarks first** (Stage 0b) — pinholes and `frame_landmarks` are
   the same command, and nothing downstream runs without them.
3. **Re-derive the mount offset** (Stage 6). Never inherit 214.0°.
4. **Check the catalog actually covers the leg.** m9 does **no spatial
   gating**, so the catalog's own extent is the ceiling on what can ever match.
   Compare its bounds against the trajectory bbox plus a plausible sightline —
   for the Boston legs the shared harbor catalog reaches only ~2.3 km south of
   the Hingham end (leg1 has 7.6 km), so a detection pointing south from there
   has no correct answer available.
5. **Range names** — `--range NAME K_START K_END`. The viewers glob
   `tracks_*.json`, and the merged `landmark_id`s embed track ids, so keep one
   range per run unless you want them mixed.
6. **Tag vocabulary.** `harbor_catalog.HARBOR_KEEP_KEYS` /
   `HARBOR_KEEP_PREFIXES` / `HARBOR_DROP_PREFIXES` are harbor-specific. A
   non-maritime environment should extend them (and **bump `CACHE_VERSION`**).
   Survey what is actually populated first:
   ```bash
   python3 -c "
   import pandas as pd
   d=pd.read_feather('/data/farfield_matching/datasets/$DATASET/landmarks/v1_trimmed.feather')
   nn=d.notna().sum()
   print(nn[nn>0].sort_values(ascending=False).head(40).to_string())"
   ```
7. **`position_sigma_m`** — `ENC_POSITION_SIGMA_M` 5 m, `OSM_POSITION_SIGMA_M`
   15 m. Adjust per map provenance; the filter turns this into angular
   uncertainty via `kappa_eff`.
8. **`instance_max_rows`** in `m9_match_landmarks` (default 5) — the ceiling
   above which an `instance` label is downgraded to `category` in code. The
   model labelled 160 matches `instance` on leg 1 that covered more map rows
   than this; without the downgrade those enter the compatibility table
   claiming to identify one object when they identify a class.


9. **The audit prompt's calibration notes** (`semantic_audit.SYSTEM_PROMPT`)
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
| `error: pass --dataset ...` | a stage no longer defaults to leg1 | pass `--dataset <name>`, or `--run_dir` for the later stages |
| `--dataset X disagrees with <run_dir>` | the run dir belongs to another dataset | drop `--dataset`; the run's path is authoritative |
| `range(s) ... end past this dataset's last keyframe` | another leg's range numbers | use this leg's frame count |
| Pinhole stage re-renders what looks complete | resolution or stem set differs from the request | the printed reason says which; a superset of stems is reused, a resolution mismatch is not |
