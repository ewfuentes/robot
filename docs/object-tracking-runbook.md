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

That runs the whole sequence below — extract → boxes → tracks → keyframes →
audit → review →
merge → offset → match → matchview → index — skipping stages whose output already
exists, and leaves every viewer on disk to read afterwards. The reason it does
not block for a human at each stage is that the sequence is **cheap**: measured
on leg1, the entire LLM bill is $26.24 through the Batch API (the default
transport), i.e. half of the $52.48 on-demand list price.

| stage | calls | tokens | USD (Batch API) |
|---|---|---|---|
| extraction | 379 | 6.07 M | $14.53 |
| semantic audit | 105 | 1.17 M | $2.53 |
| matching | 176 | 3.59 M | $9.18 |
| **total** | **660** | **10.83 M** | **$26.24** |

(These dollars were first published labeled "on-demand" — a 2x under-report.
The 2026-08-17 Vertex bill showed batch billed at $1/M input + $6/M output,
exactly the rates the tally used, so the measured figures are the *batch*
cost and on-demand is twice them. `llm_cost.py` and `run_pipeline.py` now
carry the corrected rates.)

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

**Audit the dataset before tracking a trimmed leg.** `video_t_s` is the address
the tracking stages seek to in the source video, and a wrong address cannot be
detected downstream: the seek lands on a real frame, SAM2 tracks whatever is
there, and the run completes with plausible-looking tracks. `trim_dataset` used
to rebase the column to zero at the new first frame, which discarded the first
kept frame's real video time. No head cut is needed to trigger it:
charles_river_20260727's was a *density* thin that kept frame 0, and it still
lost 510 s, because that dataset's frames were exported starting at video t=510
(`video.export_start_video_t_s`) — putting every tracking window in a different
stretch of the sail. Fixed 2026-08-18 (the column is now
carried through verbatim), and `audit_dataset` now decodes what it points at and
cross-correlates against the panorama:

```bash
bazel run //experimental/overhead_matching/swag/scripts:audit_dataset -- \
    --dataset_path /data/farfield_matching/datasets/$DATASET
#   ok    video_t_s addresses full_sail.mp4 correctly (frame match 0.999-1.000 over 3)
```

Correctly-addressed frames match 0.98-1.00; the broken run scored 0.49-0.74. The
tracking symptom, if this is ever skipped: ~7% strong-evidence tracks instead of
the 23-25% a healthy leg produces, and roughly double the weak-only tracks.

**Tracking speed and how to profile it.** The `tracks` stage was made ~2x faster
on 2026-08-18 (measured 206 s -> 106 s over 20 charles intervals): SAM2 is fed
in-memory clips instead of a temp directory of JPEGs, resize/normalize run on the
GPU, all live tracks advance in lockstep so the image encoder sees one batch per
frame instead of one call per track (SM utilization 17% -> 47%), and the viewer's
filmstrip frames are downscaled on the GPU. Masks from this path are not
bit-comparable with runs built before it.

Profile before optimizing anything here — the loop has surprised every guess so
far:

```bash
TRACK_PROFILE=1 bazel run //…object_tracking:m3_track_viewer -- \
    --dataset $DATASET --frame_landmarks_version v4 \
    --runs_root /tmp/bench --run_name prof --range bench 300 320
```

It prints per-phase seconds, share, and ms/item, synchronizing CUDA on GPU
phases (so the profiled run is slower than the real one — take wall-clock
numbers with profiling off). What remains: the image encoder is ~49% and is
irreducible while each track keeps its own window; decode is ~12% and needs
NVDEC, which this OpenCV build cannot do.

The **serial tails** were the other 25 min of an 84 min leg, and interval-level
profiling never sees them (a short bench range encodes a handful of videos):
per-track mp4 encoding now runs 8 ffmpegs at a time, and `keyframe_viewer`
renders its images in a process pool (`--image_workers`, default 12), taking
charles' keyframes stage from 631 s to 116 s with byte-identical output. Use
processes, not threads, for the image work — PIL holds the GIL, and threads
plateau at 1.9x however many you add.

Benchmarking caveat: check `uptime` first. An unrelated job at load average 46
made a tracking bench look 5x slower and very nearly got blamed on a code
change.

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

**Four gates, and the last two are the subtle ones.** Condition number (<500) and
≥4 observations per tracklet are the obvious ones. The third exists because the
condition gate is applied *per candidate offset*, so the number of surviving
tracklets is itself a function of the offset — a badly wrong offset drops nearly
everything, and the few survivors can post an excellent median residual purely
by selection. On leg1 the ungated argmin is **85° at 1.04° over 5 tracklets**,
while the true basin at 210° sits at **1.05° over 37**. A candidate must
therefore retain at least `--min_support_frac` (0.5) of the best-supported
candidate's tracklets. Under-supported candidates print with a `~`.

The fourth gate is absolute: `--min_tracklets` (5). The other three are all
*relative*, so none of them can catch a sweep that had almost nothing to work
with. On `mount_washington_20260815_leg1` a **single** tracklet survived the
condition gate, and one tracklet's residual is a smooth function of the offset by
construction — the curve came back `SMOOTH UNIMODAL, 0.95° at 211°` while the
same leg with less bearing fusion (7 tracklets) said 23°. That verdict is now
`UNDER-SUPPORTED` and `usable: false`.

Expect a **smooth unimodal curve**; the tool says `SMOOTH UNIMODAL`, `FLAT`, or
`MULTIMODAL` and refuses to write a `mount_offset_deg` for the latter two. If it
is flat or multimodal, the bearings or the poses are wrong — do not pick a
minimum from noise.

Reference: on leg1's `r003_full_leg1` the sweep returns **212.0°**, median
residual 0.94° over 37 of 46 tracklets, `SMOOTH UNIMODAL` with 4.0× contrast.
The leg's accepted value is 214°, whose own surveyed-building validation carried
std 2.42°, so the two agree.

**Measured, 2026-08-18, on the seven video datasets** (v4 runs, epoch=5
fusion). Each leg needs its own number and three of the seven have none:

| run | offset | residual | tracklets | verdict |
|---|---|---|---|---|
| boston_harbor_leg1 `r004_v4_landmarks` | 216° | 1.03° | 30/38 | SMOOTH UNIMODAL |
| boston_harbor_leg2 `r003_v4` | 245° | 1.65° | 19/21 | SMOOTH UNIMODAL |
| boston_harbor_leg3 `r001_v4` | 74° | 0.70° | 61/66 | SMOOTH UNIMODAL |
| charles_river_20260727 `r001_v4` | — | 1.58° | 46/63 | **FLAT** (unusable) |
| mount_washington…leg1 `r001_v4` | — | 0.95° | 1/1 | **UNDER-SUPPORTED** |
| mount_washington…leg2 `r001_v4` | — | 8.84° | 9/14 | **MULTIMODAL** |
| mount_washington…leg3 `r001_v4` | 4° | 8.26° | 19/24 | SMOOTH UNIMODAL |

Two things to read off that table. **The three boston legs disagree by up to
142°** while each is individually well-determined (0.70–1.65° residual, 19–61
tracklets, 4.7–8.2× contrast). Each leg is a separate stitched sequence
(`stitched_from_n_sequences: 1`, one component each), so the likely cause is a
per-sequence yaw datum in the stitch rather than a camera that moved between
legs — which means a per-leg offset is *correct*, and the shared 214° prior in
metadata is wrong for legs 2 and 3. **And the road datasets are much noisier**
(8+° residuals vs 0.7–1.7° on the water): the auto road's hairpins swing the
heading 17–38° within a single 5-keyframe fusion epoch (median; the harbor legs
sit at 2.7–3.3°), so m6 fuses bearings taken at materially different headings.
Lowering `--epoch_keyframes` to 1 buys more well-conditioned tracklets (leg1
1→7, leg3 19→66) without fixing the residuals (8.3°→9.4° on leg3), so fusion is
not the whole story there — and on the harbour legs it barely moves the answer
(leg1 216°→214°, leg2 245°→244°), which is a useful sign the boston numbers are
robust.

**Check it by eye before trusting it** — `heading_check_viewer` draws each
candidate offset as one vertical line on sampled keyframes, plus its 180&deg;
flip (a residual sweep cannot tell the ends of a ray apart), plus a
course/speed/epoch-swing figure:

```bash
bazel run //…object_tracking:heading_check_viewer -- \
    --dataset $DATASET --run_dir $RD --out_dir $RD/heading_check
```

It reads the offsets from the run's sweep and the dataset metadata, so the
question becomes "does the line sit on the bow". **Mind the azimuth zero**:
`pano_geometry` puts azimuth 0 at the **centre** column, which is the convention
`track_merge` stamps into every `bearing_camera_deg`; the dataset metadata's
world-bearing formula keys off column 0 instead. Mixing them puts every marker
180&deg; out, and on a boat it lands convincingly on the wake.

What that check settled on 2026-08-18: all three boston legs' sweep values point
forward along their own deck (216&deg;, 245&deg;, 74&deg; — three different camera
positions, confirmed by the operator), charles's bow sits near its 90&deg; prior
while the FLAT sweep's 257&deg; points abeam at open water, and the
mount_washington_20260815 legs turn out to be **pole/backpack-mounted hiking
footage** — the hiker's head occupies the forward view. A rigid mount offset is
only loosely meaningful there: the camera's yaw relative to travel wobbles with
every step, which is the honest explanation for those legs' 8–10&deg; residuals
and their refused verdicts, and no amount of re-fusing will fix it.

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

**Measured cost per dataset, 2026-08-18** (`--build_only`, then
`llm_cost.estimate_jsonl`; batch prices, and the guard compares the ×1.25
figure against `--cost_limit`, default $50):

| run | catalog | signatures→chunks | requests | prompt tok | $ batch | guard |
|---|---|---|---|---|---|---|
| boston_harbor_leg1 | v1_trimmed 13,210 | 7,950 → 16 | 176 | 2.2M | $10.61 | $13.26 |
| boston_harbor_leg2 | v1_trimmed 13,210 | 7,950 → 16 | 96 | 1.2M | $5.76 | $7.20 |
| boston_harbor_leg3 | v1_trimmed 13,210 | 7,950 → 16 | 256 | 3.1M | $15.43 | $19.29 |
| charles_river_20260727 | v1_trimmed 72,820 | 31,185 → 63 | 882 | 10.0M | **$52.38** | **$65.47** |
| mount_washington…leg1 | v1 5,908 | 583 → 2 | 6 | 0.04M | $0.33 | $0.41 |
| mount_washington…leg2 | v1 5,908 | 583 → 2 | 22 | 0.13M | $1.19 | $1.49 |
| mount_washington…leg3 | v1 5,908 | 583 → 2 | 40 | 0.25M | $2.17 | $2.71 |

Cost is driven by *signatures*, not rows: the alpine catalog's 5,908 rows carry
only 583 distinct tag bundles, so matching it costs cents. Charles is the one
that needs a decision — its trimmed catalog is 5.5× the harbor's and it is the
only run that trips the $50 single-step ceiling, so it needs `--approve_cost`
(or a tighter catalog).

**Catalog size is the price of a large prior, and that is the right trade.**
Extents: leg1 22.9 × 20.9 km = 479 km² at 27.6 rows/km² ($0.022/km²); charles
56.4 × 58.4 km = 3,294 km² at 22.1 rows/km² ($0.016/km²); the alpine legs
29.7 × 26.3 km. Charles is a **bigger box at lower density**, not a bloated one,
so do not clip it to save money — the whole-map prior is the experiment. Take the
saving out of classes instead, and validate against real matches.

**The trim is calibrated for the harbour, and it shows outside one.**
`trim_landmark_feather` + `prune_far_field_tags` were tuned against a harbour
positive set, and the reasoning is explicitly a vessel's: `highway=*` is a
hard-drop key because "a vessel cannot pick out a road, and being named does not
change that". On land that deletes real landmarks — trimming
mount_washington_20260815_leg1's catalog drops 1,672 rows of which **418 carry a
proper name**, including the Cog Railway `Summit` station, `Caps Ridge
Trailhead`, and AMC shelters. It also *keeps* things nothing can see: charles's
trimmed catalog retains 1,870 `tunnel=culvert` streams (underground) and 837
golf bunkers, and the alpine one keeps ~2,800 rows of `natural=fell`/`scree`
cover polygons. For now pass `--catalog v1` for the non-harbour datasets: on the
alpine legs the trim removes only 28% of a 5.9k-row table, so the untrimmed
catalog costs nothing extra and keeps the named summits.

**Tested and rejected: adding the "mountain" tags back** (2026-08-19). `amenity=shelter`, `railway=station`, `amenity=parking` and
`highway=trailhead` do mean different objects in a city and in the mountains, and
name-rescuing all four globally is cheap (+163 signatures on boston_harbor,
+$0.21; +581 on charles, +$0.96). It was implemented and then reverted, because
checking what it actually recovers showed the answer is *nothing useful*: the
visible thing is already in the catalog under a structural tag. The Cog
Railway's `Summit` node sits 34 m from a surviving `building=yes; name=Summit
Stage Office` and 36 m from a `man_made=tower`; `Crawford` station sits 1 m from
`building=train_station; name=Crawford Station`; Pinkham Notch's trailhead
parking sits 76 m from `tourism=hotel; name=Joe Dodge Lodge`. What is *only*
represented by the dropped tag is a lean-to (The Perch, a bare `amenity=shelter`
node), a trail sign and a parking clearing — none of them far-field landmarks,
which is exactly what HARD means.

**The keep side is already environment-neutral, and that is the real answer to
"is this harbour-specific".** It works off `STRUCTURAL_KEYS` (man_made, natural,
tourism, building, place, historic…), so the alpine vocabulary the observer
actually uses is admitted untouched: `man_made=cairn` (observed 700×),
`natural=peak` (677×), `man_made=mast` (486×), `man_made=tower` (533×),
`tourism=alpine_hut` (153×). Every substantial AMC hut survives on
`building=yes; tourism=alpine_hut` — Lakes of the Clouds (454 m²), Madison
Spring (290 m²), Mizpah Spring (213 m²), Gray Knob, Crag Camp, The Log Cabin.
The 40 `generic_small_building` drops on that catalog have a **median footprint
of 48 m²** (a 7 × 7 m shed; one is a 12 m² outhouse) and a median 59 m to the
nearest surviving row, so that rule is right there too.

Relaxing the generic-building rule was rejected on the same evidence, and it is
the more expensive of the two anyway: cheap in batches but it dwarfs the catalog
in **rows**, which are the filter's `particles × landmarks` tensor. At the stock
thresholds `generic_small_building` drops 93,931 rows on the harbour table
(kept: 13,210) and 723,322 on charles (kept: 30,370) — see each catalog's
`v2_trimmed.provenance.json`.

**Two further rules, measured 2026-08-19** — the same two in every
environment, no per-environment keep-list:

1. **identity-only** — drop a row whose surviving tags carry no class key at all.
   On charles that is 6,137 rows, of which 6,020 are bare points whose *raw* tags
   are `shop` (4,946), `office` (970), `craft` (174): tenants that
   `prune_far_field_tags` anonymised by keeping `name`/`brand` while dropping the
   class. leg1 offered 1,472 such signatures and matched **none**.
2. **cover-only** — drop a row whose *only* class tags describe ground cover or
   open recreation area (`natural=wetland/wood/scrub/heath/fell/scree/sand`,
   `landuse=forest/farmland/meadow/grass`, `leisure=park/nature_reserve/garden`).
   The cover-*only* test matters: `leisure=park; name=Peddocks Island;
   natural=coastline` survives on its coastline tag. Zoning
   (`landuse=residential`) is **not** cover — `landuse=residential; name=Harbor
   Towers` is a matched positive.

| catalog | rows | signatures | cost |
|---|---|---|---|
| boston_harbor_leg1 | 13,210 → 10,729 | 7,950 → 6,032 | $10.61 → $8.05 |
| boston_harbor_leg3 | 13,210 → 10,729 | 7,950 → 6,032 | $15.43 → $11.71 |
| charles_river | 72,820 → 52,299 | 31,185 → 23,116 | $52.38 → $38.83 |
| mount_washington (v1) | 5,908 → 2,311 | 583 → 566 | $0.33 → $0.32 |

**Recall guard: 0 of leg1's 120 actually-matched signatures are lost, at any
confidence** — `trim_landmark_feather --matched_from <m9 run dir>` now runs that
check itself and refuses to write when a rule drops something a real run
matched (98/98 survive at the 0.5 confidence floor on leg1's `r003`).
Signatures the table never held (another region, another catalog vintage) are
reported separately and do not block a write. Run that check before touching these rules — it caught two of my
earlier proposals. Three rules that *failed* it and must not be reintroduced:
dropping signatures that cover many rows (leg1 matched `man_made=pier` spanning
375 and 428 rows), dropping by physical extent (matched features include
`place=island` at 957 m and `bridge=yes` at 1,272 m median), and clipping the
catalog's area.

The evidence for rule 2 is the **observed vocabulary** — the `primary_tag` of
every v4 observation, 87 classes over 10,350 observations across the seven video
datasets. Nothing ever observed `leisure=park`, `natural=wetland`,
`waterway=stream`, `power=tower` or `amenity=school`; charles's observer uses just
17 classes, almost all buildings plus `man_made=bridge/chimney/crane/buoy`. Before
using that vocabulary as a filter directly, it needs an **alias table**: the
observer says `man_made=bridge` (510×) where the catalog says `bridge=yes`
(observed 0×), so a naive key=value match would delete 1,377 charles signatures
of the third-most-observed class on the river. Same for `amenity=hospital` vs
`building=hospital`, `tourism=hotel` vs `building=hotel`, `place=islet` vs
`place=island`.

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


### Catalog versions and provenance

`--catalog` selects which table a stage reads; the default is `v1_trimmed`.
As of 2026-08-19 every video dataset also has **`v2_trimmed`**, written by the
current rules with a `v2_trimmed.provenance.json` beside it recording the input
and its sha256, every argument, a fingerprint of the rule sets, the per-rule drop
counts, and a `reproduce` command line. For six of the seven, `v2_trimmed` is
id-for-id identical to `v1_trimmed` — the rules had not changed, and the older
tables reproduce exactly at the stock thresholds (2000 m², 6 levels). The one
real difference:

| dataset | v1_trimmed | v2_trimmed | why |
|---|---|---|---|
| charles_river_20260727 | 72,820 rows / 31,185 sigs / $52.38 | **30,370 rows / 17,348 sigs / $29.15** | `--clip_km 25` centred on the leg's anchor |
| the other six | — | identical ids | provenance only |

So matching charles wants `--catalog v2_trimmed`, which brings it under the $50
single-step ceiling (guard $36.44) without `--approve_cost`. The 25 × 25 km box
is still larger than the 22.9 × 20.9 km harbour prior; charles's sail spans
0.8 × 1.1 km inside it.

Two habits this bakes in: a catalog is **versioned, never overwritten** (the tool
refuses, since every past number was computed against the old file), and any
question of the form "what built this table?" is answered by the sidecar rather
than by re-deriving it — the claim that `v1_trimmed` was stale turned out to be
an analysis passing thresholds by hand, and the fingerprint exists to settle that
in one line.

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
cd /data/farfield_matching/artifacts/object_tracks/boston_harbor_leg1/v1 && \
    python3 -m http.server 8935
```

Then `http://localhost:8935/m3_tracks/runs/index.html`.

---

## Stage 9 — localization (base export → M11 → filter)

The stages above end at *matches*; the pose comes from the bearing-only filter,
which eats an export directory. `m11_base_export` builds the Tier-1 half
(bearings, odometry, truth, catalog) that `m11_localization_export` copies but
cannot produce:

```bash
bazel run //…object_tracking:m11_base_export -- --run_dir $RD           # + --catalog v1 off the water
bazel run //…object_tracking:m11_localization_export -- --run_dir $RD \
    --base_export $RD/localization_export_base \
    --output_dir $RD/localization_export_llm_chunked
bazel run //…bearing_only_localization:run_export -- \
    --export_dir $RD/localization_export_llm_chunked \
    --output_dir /tmp/${RUN}_filter --init uniform --backend torch
```

Run the base export **after** Stage 6: the mount offset is baked into every
bearing (`bearing_body = (bearing_camera − offset) mod 360`), and the tool takes
it from this run's sweep when usable, else the dataset metadata, and prints
which — loudly, when the recorded value was never accuracy-validated.

Skipping M11 is a legitimate run: the base export ships one *uninformative*
table per tracklet (flat log-LR), which is the association-ambiguity floor —
what dead reckoning plus "a bearing to something in the catalog" can do with no
matcher at all. Useful as the baseline every matched run has to beat.

Two operational notes. **M11 must be at least as new as the merge**: it refuses
an export where a measured tracklet has no compatibility table, and a stale
matching run produces exactly that (on leg1's `r003`, 1 of 103 tracklets). The
error now names the count and the cause. **The filter's measurement update is
`particles × landmarks`**, so a 72,820-landmark catalog at 50k particles wants a
14.6 GB tensor — drop `--n_particles` for big maps, and do not run several
`run_export`s on one GPU at once.

Validated 2026-08-18 against the pre-existing leg1 `r003` export, which the
earlier wedge/GPS campaign produced independently: odometry agrees to 9.6e-10 m,
truth positions to 7.3e-10 m, and bearings to 0.41° on shared epochs (the merge
has been re-run since, so the epoch sets differ). Feeding the rebuilt base
through M11 and the filter returns **420 m final / 406 m median** over the whole
23 km harbour map, against the 452 m the campaign recorded.

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
6. **Tag vocabulary.** `harbor_catalog.FAR_FIELD_KEEP_KEYS` /
   `FAR_FIELD_KEEP_PREFIXES` / `FAR_FIELD_DROP_PREFIXES` are harbor-specific. A
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
