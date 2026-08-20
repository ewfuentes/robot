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

**Correction, 2026-08-19: the audit and matching rows above are Pro rates
applied to Flash work.** The token counts are sound — they are read from stored
`usageMetadata`. The *rates* came from that Vertex bill, which priced **Gemini
3.x Pro**, and the tally then applied them to all three stages. Only extraction
ran on Pro (`gemini-3.1-pro-preview`, recorded in the extraction manifest). The
audit runs on `gemini-3-flash-preview` — Stage 4's own command below — and the
Flash tier is priced as one tier with 3.7-flash, $0.375/M in + $1.875/M out, so
that row's real batch cost is about **$0.42, not $2.53**.

The matching row cannot be corrected, because **`r003_full_leg1` did not record
which model it used**: there is no `matching/settings.json` for that run and the
stored responses carry no `modelVersion`. If it was Flash the row is ~$1.5
rather than $9.18. m9 now writes `settings.json` with the model in it, which is
why r004/r005 are answerable and r003 is not.

The lesson is narrower than "the prices were wrong": a rate was carried from an
invoice onto models that invoice never covered, and the artifacts did not record
enough to notice. `llm_cost.MODEL_RATES` is now keyed by model, an unknown model
prices at Pro as an explicitly-labelled **upper bound**, and every rate carries
its provenance in a comment.

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
| map catalog | `datasets/<ds>/landmarks/v2_trimmed.feather` | OSM + ENC, `landmark_type` ∈ {historical, enc}. `v2_trimmed` is the default since 2026-08-19; it carries a `.provenance.json` sidecar and, on charles, a recorded 25 km clip. Row-for-row identical to `v1_trimmed` on the harbour and mountain tables |

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

**Prices are per model, and the spread is bigger than the estimator's error.**
`llm_cost.MODEL_RATES` is keyed by model id (longest-prefix match, so
`-preview` suffixes resolve), and both the pre-flight guard and `run_pipeline`'s
post-hoc tally price each stage at the model that produced it. Anything without
a table entry — `gemini-3-flash-preview` above all, which runs the audit and
matching stages — prices at 3.1-pro rates: conservative, and *not* a verified
price for those models, but it is the rate the tables in this file were computed
at, so the published numbers stay reproducible.

This matters more than a reporting nicety. `pohang_canal_04`'s 1,450-panorama
extraction on `gemini-3.7-flash` estimates **$14.61 batch** ($18.26 after the
1.25x guard margin). At 3.1-pro rates the identical work estimates ~$88 batch,
i.e. ~$110 guarded — over the $50 single-step ceiling, on a step with no
terminal to ask on, so the run would have refused to submit. A model swap
without a rate entry does not mis-report; it blocks.

Two measured notes on Flash from that run: `gemini-3.7-flash` is the real Vertex
model id (`gemini-3.7-flash-preview` 404s), Vertex **Batch** accepts it, and
thinking tokens ran ~4.5x the visible output. Since output bills at 5x input,
thinking dominates a Flash bill even though it never appears in the response.

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

Required, **per leg, and never inherited without a check**. 214.0° is specific to
boston_harbor leg1; legs 2 and 3 carried it as "same physical mount assumed" and
were 27° and 144° wrong.

> **Check the FRAME too, not just the leg.** Camera-frame azimuth 0 is the
> panorama's **centre** column. A dataset's own `azimuth_convention.formula`
> references **column_0** — a correct formula with a different zero — so an offset
> reasoned from it is **exactly 180° out**. `pohang_canal_04` recorded 180° where
> the pipeline needs 358°, and its block said `accuracy_validated: true`, which
> `m11_base_export` ranks *above* this sweep. A validated number is validated for
> one frame and one quantity; it does not travel between them. Only
> `sun_offset_check` can catch this class — the sweep below fits a 180° slip
> perfectly, by construction. Register: [`conventions.md`](conventions.md) §2.

### Read this first: the 180° convention trap

`pano_geometry.pano_px_from_direction` puts **azimuth 0 at the CENTRE column**
(`x = ((az/360 + 0.5) mod 1)·W`), and that is the convention `track_merge`
stamps into `bearing_camera_deg`. An offset reasoned as "azimuth 0 = column 0"
is therefore **exactly 180° out**. Two datasets shipped that way:

| dataset | recorded | operator's reasoning | true value |
|---|---|---|---|
| charles_river | 90° | "bow ¼ frame from the left" | **270°** |
| mount_washington ×3 | 180° | "travel at image centre" | **0°** |

Both were caught by measurement, not by review. Before trusting any prior, render
it: `heading_check_viewer --offset "prior=90"` puts a line on the panorama, and
the direction of travel is a thing you can see (a bow, a wake, the back of the
hiker's head). The same trap once put that viewer's own marker 180° out, where it
landed convincingly on the wake.

### Which estimator to believe

Three of them, and they are **not** equal. Rank by whether the check can see
outside the run's own bearings:

1. **`sun_offset_check` — ABSOLUTE. Use this whenever the sky is clear.**
   `offset = course + az_camera_sun − az_ephemeris_sun`. No map, no tracks, no
   operator. Validated: it returns **215.0° on leg1**, whose 214.0° came from a
   surveyed building over 72 keyframes. It writes metadata with
   `accuracy_validated: true` and refuses to overwrite an existing validated
   value.
2. **`mount_offset_sweep` — RELATIVE. A corroborator.** It finds the angle that
   makes rays to *unknown* objects agree with each other, so it reproduces any
   error the poses and the heading model share — a 180° convention slip fits it
   perfectly. It agreed with the sun to 1.0–3.3° on all four clear-sky datasets,
   which is reassuring and still not the same as checkable.
3. **operator prior** — needs the convention translated first (above).

`m11_base_export` resolves in that order: explicit flag → `accuracy_validated`
metadata → this run's usable sweep → unvalidated metadata (announced loudly).

```bash
bazel run //…object_tracking:sun_offset_check -- \
    --dataset $DATASET --frame_landmarks_version v4 --n_frames 60 --write_metadata
```

**Keep `--elevation_tolerance_deg` tight (default 3°).** A yaw offset cannot
change the sun's elevation, which is what makes ephemeris elevation a legitimate,
non-circular gate — and slack in it admits two impostors that sit at the sun's
own elevation and brightness: the vehicle's sunlit structure, and the antipodal
ghost a dual-fisheye stitch throws ~180° opposite a bright source. Tightening on
charles walks down that list, and R rising monotonically under a yaw-invariant
gate is the signature of closing on the real sun:

| tolerance | median \|elev err\| | offset | R | verdict |
|---|---|---|---|---|
| 12° | 5.4° | 306.7° | 0.165 | FIXED-OBJECT |
| 6° | 2.7° | 282.7° | 0.360 | FIXED-OBJECT |
| 4° | 1.4° | 275.3° | 0.661 | SCATTERED |
| **2.5°** | **0.5°** | **272.4°** | **0.965** | **AGREEING** |

**It needs an absolute clock, and most datasets have not recorded one.**
`log_start_utc` in `pipeline_metadata.json` is the only input the ephemeris
cannot do without. Of 27 datasets on disk, **8 have it** — the four harbour/river
self-collects, the three mount_washington legs, and pohang — and every Mapillary
dataset is MISSING it, so the best offset estimator cannot run on the majority of
the corpus. Mapillary does supply per-image capture times, so this is a gap in
ingest, not in the data.

Where it *is* recorded, check that it is per-leg. The three mount_washington legs
all carry the identical `2026-08-15T17:29:03Z` while each leg's `time_s` restarts
from its own origin, so at most one of them has a correct absolute clock. Nothing
here depends on it — the check abstains on all three for cloud — but do not read
the small elevation residuals as reassurance that the clock is right: the
elevation *gate* selects blobs near the assumed elevation, so a small elevation
error is guaranteed by construction and says nothing about the time.

`--scan_tolerance` prints that table. The tool also masks the rig (temporal
median: the sun sweeps the camera frame as the vehicle turns, bolted-on structure
does not) and fits `az_camera = const` alongside the sun model, reporting
FIXED-OBJECT when the sail explains the frames better than the sun does. It
abstains honestly on overcast — all three mount_washington legs give R = 0.67–0.73
against the 0.95 it needs, so there the sweep is the only estimate.

### When the sweep is the only option

`--min_arc_deg` (default 20) is the gate that makes it work off-boat. It selects
on the tracklet's **bearing span**, which is *offset-invariant* — a candidate
offset subtracts the same constant from every bearing in a tracklet and leaves
their spread unchanged — so the tracklet set is fixed across the sweep and
residuals at different offsets are directly comparable. The condition-number gate
is not offset-invariant, which is the whole reason `--min_support_frac` had to
exist.

It also removes the tracklets that cannot answer the question: rays over a wide
arc can never be rotated into parallelism, while rays over a 2° arc are
near-parallel at *every* offset and therefore always vote for whatever makes them
most parallel. Without it, leg2's argmin is 133° wrong and leg3's 159° wrong,
both with excellent-looking residuals.

| gate | leg1 err | leg2 err | leg3 err | contrast (leg1) |
|---|---|---|---|---|
| none | 6° | 133° | 159° | 3.8 |
| arc ≥ 20 | 0° | 7° | 2° | 6.9 |
| arc ≥ 30 | 0° | 5° | 2° | 14.8 |
| arc ≥ 45 | 0° | 5° | 0° | 26.0 |

When a sweep comes back FLAT, `mount_offset_diagnostics` says why: it reports each
tracklet's arc, baseline, range and `d|residual|/d offset`. charles was FLAT
because its median arc is 12° against the harbour's 45°, its median baseline
200 m against 730 m, and **65% of its tracklets are blind to the offset**
(sensitivity < 0.05°/°) — a river-reach geometry problem, not a data problem.

A leg with no usable geometry and no sun can inherit from a sibling leg of the
same rig, but only with its own visual check: mount_washington legs 1 and 2 take
leg3's 4.0° because `heading_check_viewer` at 4° lands on the back of the hiker's
head in both, which is where a pole-mounted camera riding behind the walker sees
the direction of travel.

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
the dataset's `landmarks/v2_trimmed.feather`. On leg 1 that is 176 requests over 7,950
signatures for ~102 tracklets.

**No spatial gating.** Set 2 is the whole map. Do not reintroduce a
position-based shortlist — see the removal note in the pipeline doc.

Or in one step — `--submit` executes the requests and carries straight on to
aggregation, through the Batch API by default:

```bash
bazel run //…object_tracking:m9_match_landmarks -- --run_dir $RD --submit
bazel run //…object_tracking:m10_match_viewer -- --run_dir $RD
```

M10 writes `matching/review/index.html`: the match list beside a pannable map of
the run — vessel track with heading, the bearing ray for the selected tracklet,
and the map rows it matched. Check the **`ray Δ`** column and the geometry-check
line in the header first; a confident match whose row does not lie along the
bearing it was seen on is wrong regardless of confidence. It needs `--feather`
to resolve (it reads the catalog for positions and the basemap); `--no_map`
skips that.

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
| charles_river_20260727 | v2_trimmed 30,370 | 17,348 → 35 | 490 | 5.6M | **$29.15** | **$36.44** |

Those are *guarded estimates*. What the seven datasets actually billed, read back from stored `usageMetadata`, is **$9.04 total** — 14.1 M prompt, 0.5 M output, 5.1 M thinking tokens at batch rates. `llm_cost`'s per-request output-token assumption is roughly 7x conservative, so treat its number as a ceiling for approval, not a forecast.
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

`--catalog` selects which table a stage reads; the default is `v2_trimmed`
(`farfield_paths.DEFAULT_CATALOG`, changed 2026-08-19).
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

**The base export honours the audit's `drop` verdicts.** M5 returns
`verdict: drop` for tracks that are not a usable distinct object, and
`m9_match_landmarks` never queries those — so they have no compatibility table.
The base export used to include their bearings anyway, which broke two things at
once: M11 died on measurements with no table (blaming a stale matching run, which
was the wrong diagnosis), and any run that got past it fed the filter bearings the
pipeline had already classified as clutter, inflating the clutter rate above the
`pi0` the filter assumes. Five of leg1's 107 tracklets and three of
mount_washington leg3's 199 were in that state. `--keep_dropped_tracklets`
restores the old behaviour for a control.

Two operational notes. **M11 must be at least as new as the merge**: it refuses
an export where a measured tracklet has no compatibility table, and a stale
matching run produces exactly that. The error now names both possible causes and
how to tell them apart. **The filter's measurement update is
`particles × landmarks`**, so a 72,820-landmark catalog at 50k particles wants a
14.6 GB tensor — drop `--n_particles` for big maps, and **do not run several
`run_export`s on one GPU at once**: charles (30,370 landmarks) OOMed at 50k
particles while a second run held 8.6 GB.

Validated 2026-08-18 against the pre-existing leg1 `r003` export, which the
earlier wedge/GPS campaign produced independently: odometry agrees to 9.6e-10 m,
truth positions to 7.3e-10 m, and bearings to 0.41° on shared epochs (the merge
has been re-run since, so the epoch sets differ). Feeding the rebuilt base
through M11 and the filter returns **420 m final / 406 m median** over the whole
23 km harbour map, against the 452 m the campaign recorded.

### Results: 5 of the 7 legs have data that localizes on the whole map

Uniform prior over the catalog's own extent — no prior information at all, which
is the hardest setting the filter offers:

**Read the median over the last 50 keyframes, not the final error.** The final
number is a single keyframe and it can be lucky by an order of magnitude: leg3 run
with a deliberately wrong mount offset finished at 37 m — as good as the correct
offset's 34 m — on a median of **555 m**. The median is the figure of merit below.

With `evidence_gate_selection_charge` on (its default now):

| dataset | median last 50 | seeds | what it needs |
|---|---|---|---|
| **charles_river** | **23 m** | 23, 25 | nothing |
| **boston leg3** | **41 m** | 41, 30 | nothing |
| **boston leg1** | **78 m** *(r003 tracks)* | 78, 79 | its export rebuilt from good tracks |
| **mount_washington leg3** | **113 m** | — | nothing |
| **mount_washington leg2** | **239 m** | — | nothing |
| mount_washington leg1 | 7650 m → **641 m** at a 1 km prior | — | a coarse prior |
| boston leg2 | 11512 m, 9524 m even at 1 km | — | **the one unexplained failure** |
| boston leg1 (r004 export) | 6440 m | 2022–13712 | superseded by the r003-tracks result |

**Five of the seven legs have data that localizes to ≤ 240 m** over a 20–30 km
uniform prior; one more works given a few-km prior; one is unexplained.

charles is the best of the seven, and it had never completed this chain before —
its offset sweep was FLAT and unusable, its recorded offset was 178° wrong, and the
audit-`drop` leak below stopped its export from building at all. Note how
well-behaved a working run is: charles pairs its
25 m with a median bearing residual of **0.25°**, a null share of 0.01, a >80%
single-landmark claim on **387 of 561** measurements, and a reported sigma of 43 m
against a 25 m error.

**How much did the offset correction matter? Less than you would think, and this
is the most surprising measurement of the night.** Rebuilding each export with the
offset it used to carry and changing nothing else:

| dataset | wrong offset | median | correct offset | median | error in the offset |
|---|---|---|---|---|---|
| boston leg3 | 214° | 555 m | 70.5° | 48 m | 144° |
| charles | 90° | 86 m | 272.4° | 25 m | **178°** |

charles still localizes to 86 m with **every bearing reversed**, and it gets there
coherently — 362 of 561 strong single-landmark claims, 0.83° median residual, a
reported sigma of 59 m. Two mechanisms, and both are worth knowing:

- `world_bearing = heading + (camera − offset)`, so a *constant* offset error is
  partly absorbed by the filter's own heading state. Body-frame odometry stops it
  vanishing entirely, since heading is tied to the direction of travel.
- More importantly, the filter **re-associates**. The Charles basin has buildings
  on both banks, so flipping the bearings ~180° can be re-explained by swapping
  which bank each landmark sits on — which puts the boat in a nearly mirrored
  position, and in a 1.3 km-wide basin the mirror is close to the original.

So the offset is much less identifiable than it looks, and a wrong one degrades by
3–11× rather than destroying. That is exactly how one sits in a dataset for weeks
looking merely disappointing — and it is why the offset needs an **absolute** check
rather than a residual that a re-association can quietly satisfy.

Do not reach for those three as a health check on a leg you have not run. They are
**posteriors of the run itself** — the residual is against the filter's own pose,
the null share and the claims come out of its association posterior — so a
diverged filter reports small residuals against the wrong landmarks quite happily.
They also fail to order the outcomes: mount_washington leg3 works on 5% strong
claims and 19.3° of residual while boston leg1 fails on 29% and 4.3°.

**Replicate seeds before comparing anything.** The spread on a failing leg is
wide — boston leg1 over five seeds runs 1984 to 7986 m — so a single run cannot
distinguish a fix from luck. Two of tonight's apparent findings died on that.

## boston leg1: the regression, and re-creating the good run

**The good run is re-created, and improved on.** Copy r003's `merged/` and
`semantic_audit/` into a fresh run dir under the same runs root (paths infer the
dataset from the path), re-match, and rebuild the export on r003's base:

```bash
cp -r $RUNS/r003_full_leg1/{merged,semantic_audit} $RUNS/r005_r003tracks_flashmatch/
bazel run //…object_tracking:m9_match_landmarks -- --run_dir $RUNS/r005_… --submit
bazel run //…object_tracking:m11_localization_export -- --run_dir $RUNS/r005_… \
    --base_export $RUNS/r003_full_leg1/localization_export_llm_chunked \
    --output_dir $RUNS/r005_…/localization_export_llm
```

176 requests, ~$1.6. Medians over the last 50 keyframes:

| export | seeds | median |
|---|---|---|
| **r003 tracks + current matcher** | 78, 79, 83, 77 | **≈79 m** |
| r003's own original export | 653, 442 | ≈550 m |
| r004, the current export | 2022, 4675, 6231, 6440, 13712 | ≈6400 m |
| odometry only | — | 7107 m |

So the leg localizes to ~79 m — better than the campaign's 420 m — on the older
tracking run's tracklets with today's matcher. The run directory is kept at
`runs/260819_final/boston_leg1_r003tracks`.

**What the regression is not.** Ruled out by measurement:

- *the matcher model* — r005 runs today's `gemini-3-flash-preview` on r003's
  tracklets and reproduces r003's statistics (median confidence 0.50, 63 instance
  matches, median `no_match` 0.10), not r004's (0.30, 57, 0.17).
- *bearing quality* — triangulating every tracklet against GPS poses at the known
  214°: median residual **0.99° (r003) vs 0.85° (r004)**, p90 3.16 vs 2.30, none
  above 15°. r004's bearings are if anything slightly better.
- *geometry* — median observation arc 34.9° vs 44.5°, baseline 698 m vs 732 m.
  Comparable, r004 marginally better.
- *top-1 match accuracy* — checked against truth, r005 is the **worst** of the
  three (median claim error 24.3° against r003's 17.1° and r004's 20.5°) and
  localizes best. Whether the single best guess is right does not decide it.

**What it is: recall.** Whether the geometrically-consistent landmark is in the
table *at all* orders the three exports correctly, and it is the only statistic
tested that does:

| export | true landmark present | median error |
|---|---|---|
| r005 | **84%** | 79 m |
| r003 | 79% | ≈550 m |
| r004 | **69%** | ≈6400 m |

The mechanism: the filter marginalises over a table's entries, so a true landmark
present with any reasonable weight can be found, while an absent one cannot be
rescued by confidence on the top entry. r004 carries *more* measurements (381 with
a claim against 308) with the right answer missing from 31% of them.

The drop is concentrated in one class:

| audited extent | r005 recall | r004 recall |
|---|---|---|
| small_extended | 85% | 80% |
| **large_extended** | **84%** | **38%** |
| point_like | 83% | 73% |

Not a centroid artefact, which is the obvious objection since `keep_hulls=False`
reduces an extended object to a point: excusing those failures as centroid offset
would need objects a median of **5.3 km across**, and under 1% could be explained
by anything under 500 m. Boston Harbor islands are a few hundred metres.

**Root cause: uncorroborated names.** The failure is not in the tracking, the
merge, or the matcher — it is in the *name* the audit attached, and the matcher
faithfully honouring it.

Worked example, `LT224_T225` (16 measurements, mean miss 107°):

| stage | what it produced | verdict |
|---|---|---|
| tracking | 32 observations, 107° arc, residual **2.41°**, condition 3.1 | correct — rays converge on one real island at (105, −1254) m |
| merge | two source tracks over identical keyframes 188–262 | correct — two simultaneous detections of one object |
| audit description | "two distinct vegetated hills connected by a lower centre saddle" | **correct** — that is Spectacle Island's two capped drumlins |
| audit name | **"Georges Island" (0.8), "Fort Warren" (0.5)**, basis `reported_by_detections` | **wrong** — Fort Warren is on Georges Island; the detections free-associated a famous name |
| matcher | all three *Georges Island* rows at 0.95, `instance` | correct, given that name |

The island actually at the triangulated point is **Spectacle Island**
(`enc:02263370C5ED15CA`, 60 m away) — and it is in the catalog, in `LT207`'s
table. The two islands' names were transposed: T207 carries "Spectacle Island" at
weight **0.2** while sitting 1.6 km from it, and T224/T225 carry "Georges Island"
at 0.8–0.9 while looking straight at Spectacle.

The aggregate is the regression:

| run | names corroborated (`both`) | uncorroborated (`reported_by_detections`) |
|---|---|---|
| r003 / r005 | **26** | 14 |
| r004 | 19 | **21** |

r003's names were mostly corroborated from the imagery; r004's are mostly
unverified guesses. A name is the strongest signal the matcher has, so a guessed
name converts directly into a confident wrong `instance`. That is also why the
damage concentrates in natural features (recall 74% → 24%, against 88% → 81% for
man-made): an island invites a famous local name, a pier does not.

**The matcher is already told.** `m9_match_landmarks.SYSTEM_PROMPT` explains that
`reported_by_detections` "means it could not corroborate [the name] from the images
it was shown, which does not elimnate the name as a possibility" — and the matcher
returned 0.95 `instance` anyway. So the fix is on one of two levers, both
untested: withhold or hard-cap uncorroborated names before they reach the matcher,
or make the prompt treat an uncorroborated name as weaker than a description.

### The audit prompt hands over the evidence; the average respects it, the tail does not

ekf spotted this in the T224 prompt: `names reported: 'Georges Island' x1, 'Fort
Warren' x1` — out of **64 associated detections** — and the audit emitted those
names at weight **0.8 and 0.5**.

Measured over **all nine audits** (1,101 kept tracks, 858 name rows, 671 endorsed
name candidates). Figure: `runs/260819_final/name_support.png`.

**First: `xN` in the prompt is already a keyframe count.** A track can associate
two detections in one keyframe, so a per-detection vote could in principle
double-count. It barely does — 11 of 827 name rows are affected at all, and the
detection total (3,756) overstates the distinct-keyframe total (3,719) by 1.0%.
Reading `x1` as "one keyframe saw this name" is safe.

**The distribution of names over supporting keyframes is brutally thin:**

| | |
|---|---|
| kept tracks with **no** name reported at all | 675 / 1101 (61%) |
| kept tracks with 1 / 2 / 3+ distinct names | 210 / 117 / 87 |
| median supporting keyframes **per track** | 9 (mean 17.8, p90 42) |
| median keyframes reporting a given **name** | **2** (mean 4.5) |
| names reported by exactly **one** keyframe | **387 / 827 (47%)** |
| median support fraction (name keyframes / track keyframes) | **0.20** |
| endorsed names resting on **zero** detections (invented) | 31 / 671 (4.6%) |
| endorsed names on 1 keyframe | 266 / 671 (40%) |

Half the names the auditor is asked to adjudicate were spoken once, on a track
that was seen nine times.

**The auditor's weight does respond to support — on average.** This corrects an
earlier claim in this runbook that it was uncorrelated; that number came from a
narrower unit of analysis (leg1 only, top-weight name per track):

| scope | unit | n | corr(support fraction, weight) |
|---|---|---|---|
| leg1 r003 | top-weight name per track | 39 | +0.06 |
| leg1 r004 | top-weight name per track | 40 | +0.19 |
| all 9 audits | every name candidate | 640 | **+0.45** |

Mean emitted weight by support fraction, all audits: **0.42** (<5%) → 0.58 → 0.61
→ 0.74 → 0.82 → **0.95** (≥75%). Monotone. The prompt is working.

**What it does not do is bound the tail.** Max weight is 1.00 in *every* support
bucket including <5%, and:

| | |
|---|---|
| endorsed names with ≤2 keyframes **and** weight ≥0.8 | **209 / 671 (31%)** |
| of those, on exactly 1 keyframe | 129 |
| worst cases | 31 invented names, mean weight 0.73; charles T61 "John Hancock Building" at 0.90 and leg1 T248 "Fort Revere Water Tower" at 0.80, both from zero mentions |

Per audit the ≤2-keyframe-and-≥0.8 share runs 22% (charles), 25–36% (leg1),
37% (leg3), 38% (mtw), 41% (pohang). It is not one leg's problem.

**Does low support actually mean wrong?** Directionally yes, but the obvious test
is confounded and the honest answer is weaker than the first pass suggested.
Scoring each endorsed name by what the *matcher* would do with it — is there a
catalog row of that name within 1 km of the tracklet's triangulated position
(`RESOLVES_HERE`), only far away (`RESOLVES_ELSEWHERE`, the dangerous case: a
confident bearing to the wrong place), or nowhere (`UNRESOLVABLE`, inert):

| supporting keyframes | n | resolves at the object |
|---|---|---|
| 0 (invented) | 10 | 50% |
| 1 | 137 | **44%** |
| 2 | 47 | 57% |
| 3–4 | 51 | 71% |
| 5–9 | 54 | 65% |
| 10+ | 66 | **67%** |

**The confound:** "resolves elsewhere" mixes a wrong name with a badly conditioned
triangulation of a genuinely distant object. Applying the sweep's arc gate lifts
the overall resolve rate 57% → 71% (n 365 → 184), which means a large slice of the
apparent name errors were geometry errors. Filtering hard (arc ≥20°, condition ≤10,
residual ≤5°) leaves only n=35, where the trend survives but proves little: 5/13 at
1 keyframe, 8/11 at 10+. Two further false-negative modes to know about: name
variants (leg3 T440 claims "Logan International Airport Control Tower" 14 m from a
row named "**Boston** Air Traffic Control Tower" — right object, unresolvable name)
and range error on skyline objects (charles's "Millennium Tower" scores 12 km off
because a 200 m boat baseline cannot range a 3 km building).

So: **support predicts resolvability, the effect is real but smaller than the leg1
numbers implied, and the strongest statement the data supports is about the tail
rather than the average.**

**Actionable, and untested:** cap the emitted name weight by its keyframe support —
in `semantic_audit.SYSTEM_PROMPT` ("a name reported in 1 of 64 keyframes cannot
carry weight 0.8"), or post-hoc in `m9_match_landmarks.query_bundles` by scaling
each name's weight by its support. By the ablation above that is worth ~44× on
leg1. The cost is real signal: names on a single keyframe still resolve at the
object 44% of the time, so a hard cut throws away roughly as many good names as bad.

*Caveat on the automatic name check.* Comparing an audited name against the
**nearest** named catalog row is not a valid test in the downtown stretch — T0's
"Custom House Tower", reported by 38 of 38 detections at weight 1.00, scores as
wrong against a nearest row of "New England Aquarium". The resolvability test above
avoids that; the island cases stand on a much wider margin (60 m to the Spectacle
rows against 4.9 km to the Georges rows).

### Why v4 regressed: the extraction prompt's naming rules changed

The question "why do we get these bad names now when we didn't before" has a
specific answer, and it is not the audit.

**Ruled out — the audit prompt.** The `systemInstruction` in r003's and r004's
`semantic_audit/requests.jsonl` is byte-identical: 5230 chars, sha1
`939a3da63c26`, same `generationConfig` and response schema. The audit stage did
not change. (Not fully ruled out: the audit *model*. r004's results record
`modelVersion: gemini-3-flash-preview`; r003's predate that field, so its model
cannot be recovered from the artifact.)

**The cause — `osm_tags_farfield` was rewritten between v1 and v4.** Same prompt
*name* in both manifests, but 7220 chars in v1 against 11510 in v4, and the
`<naming_rules>` block was inverted. Diff the actual bytes with
`sentence_requests/panorama_sentence_requests/panorama_request_000.jsonl` in each
version — the prompt is stored per request, so this is always recoverable:

| v1 | v4 |
|---|---|
| "Include a name tag **ONLY if**… If several similar structures are visible, **do NOT assign a famous name** unless you can see the features that distinguish it." | "A name is **the most valuable thing** you can attach to a landmark… **Give one whenever you honestly can.**" |
| "Never guess a name from geographic context alone." | "The test is whether you are confident in THIS feature's identity. **It is NOT whether other things nearby look similar**… treat them as a reason to look for what distinguishes this one, **never as a reason to withhold a name you are sure of**." |
| — | "**Name the peaks, ranges, islands** and well-known structures you genuinely know." |
| — | "**Express your certainty through the `confidence` field rather than by staying silent.**" |

The rewrite was for the mountain/perspective rollout — generalising a
maritime-only prompt to all outdoor settings — and in the process it deleted the
one guard that suppressed famous-name guessing among lookalikes. Boston Harbor is
made of lookalikes: five Back Bay towers within 300 m of each other seen from 3 km
at sea, and a dozen drumlin islands that all look the same.

**It shows up in the data exactly where the prompt predicts.** Same 379 panoramas,
same model family, same resolution:

| | v1 | v4 |
|---|---|---|
| detections | 1403 | 1505 |
| named | 346 (24.7%) | 330 (21.9%) |
| **distinct names** | **33** | **40** |
| named at `medium` confidence | 10 (2.9%) | 28 (8.5%) |
| name rate *within* the medium band | 1.8% | **4.2%** |
| **Back Bay tower cluster** (Hancock + Prudential + 200 Clarendon + Millennium) | **16 detections** | **37** |
| — John Hancock Tower | 2 | **14** |
| — Prudential Tower | 5 | **15** |
| singleton names (uttered once in the leg) | 12 | 14 |
| **of those, landmarks in another city** | **0** | **5** |

Fewer names overall, but more *different* names, 2.3× as many claims on the
tower cluster the v1 guard covered, and a singleton tail that now contains
**Willis Tower, Aon Center, Lake Point Tower, Centennial Wheel** and
**Huntington Bank Pavilion at Northerly Island** — the Chicago lakefront, named in
Boston Harbor. v1's singleton tail is 12 names and every one is a real Boston-area
feature.

**Is it natural extraction variance?** No. The shift is directional and matches
what the prompt asked for (more recognition-based names, doubt expressed through
`confidence` instead of silence), and a coherent five-name Chicago cluster is not
sampling noise — it is one city being recognised as another. What this evidence
cannot separate is prompt from model, since v4 also moved
`gemini-3.1-pro` → `gemini-3.1-pro-preview` with `thinking_level: HIGH`. The
decisive experiment is a v5 extraction of leg1 with the v1 naming guard restored
and everything else held at v4 — 379 panoramas at the measured $0.041/pano is
about $16.

**A structural gap the rewrite opened.** v4's prompt explicitly delegates naming
doubt to the `confidence` field. The audit dossier throws that away: `name` is in
`semantic_audit.NON_IDENTITY_TAG_KEYS`, so it is excluded from the per-confidence
`tag_vote_table`, and `build_dossier`'s `name_votes` counts names with no
confidence attached. Every identity *tag* reaches the auditor split high/medium/low;
the name — the single most consequential piece of evidence — reaches it as a bare
count. Worth fixing on principle, but note it would not have caught the Chicago
names: all five were emitted at `high`.

**Second hazard from the same rewrite: bare numbers as names.** v4 added "a number
or letter painted or mounted on the structure (e.g. a buoy's '8', '13', '1SC') →
give it as `name=<exactly as read>`". On boston_harbor_leg2 that is the *only* kind
of name produced — all six named detections are `4`, `8`, `18`, `16`, `10` — and
leg2's one endorsed audit name is `16` at weight 0.95. A one-character name matched
as a substring hits every catalog row containing that digit. See the name-availability
table below.

### Tuning the extraction prompt on a 22-frame control set

The regression is in `osm_tags_farfield`'s naming rules, so the fix belongs there.
A control set makes the change measurable without re-extracting a dataset.

**Method, and why it is cheap.** Every extraction stores its requests verbatim in
`sentence_requests/panorama_sentence_requests/panorama_request_000.jsonl`, images
base64-inlined. So a prompt variant needs no pinhole rendering and no re-upload:
pull the control frames' request lines, rewrite the `<naming_rules>` block inside
`systemInstruction`, and run them through
`vertex_batch_manager run-online --model gemini-3.1-pro-preview`. 22 frames is
~5 minutes and **~$1.65 a pass** (≈410k tokens, mostly thinking).
Scripts: `build_control.py`, `pull_requests.py`, `make_variant.py`,
`score_control.py`, `aggregate.py`.

**The control set** — 22 frames chosen to cover the damage *and* the gains, because
the v4 rewrite bought the mountain datasets their peak names and a fix must not
give those back:

| group | frames | what it holds |
|---|---|---|
| `leg1_hallucinated_city` | 1 | f106 — the panorama that produced all five Chicago names |
| `leg1_neighbour_of_f106` | 2 | f105, f107 — is the misrecognition local to one frame? |
| `leg1_island_misname` | 2 | f170, f238 — Spectacle / Georges / Fort Warren |
| `leg1_backbay_cluster` | 2 | f90, f119 — the Hancock/Prudential/200 Clarendon lookalikes |
| `leg1_unnamed_control` | 1 | f49 — detections but no names; must stay quiet |
| `leg2_buoy_numbers` | 2 | f118, f119 — the bare-digit buoy names |
| `mtw_peaks` / `mtw_weak_peaks` / `mtw_summit` | 6 | the gains: Mount Monroe, Mount Washington, Lakes of the Clouds Hut, Tip Top House, Sherman Adams |
| `harbor_correct` | 2 | leg3 f272, f363 — harbour names that were right |
| `river_dense` | 2 | charles f35, f192 — the densest correct naming anywhere |
| `canal` | 2 | pohang f7, f9 |

All seven extractions turn out to share one prompt (sha1 `ae35b3db581a`), pohang v5
included, so the control set governs every dataset at once.

**The scoring metric, and its limits.** Write each variant's predictions into a temp
`frame_landmarks` layout, run the real `ingest` so bearings come from the tested
geometry code, then per emitted name ask: is there a catalog row of that name at
that bearing from the camera? World bearing is
`(course + bearing_camera - mount_offset) mod 360`; tolerance is
`20 deg + atan(120 m / range)` so a hut 30 m away is not judged on the same angle as
a tower 12 km away. Verdicts: `AT_BEARING`, `WRONG_BEARING`, `NO_SUCH_ROW`,
`DESIGNATOR` (a bare board number, never resolvable).

Two traps, both hit and fixed: a plain substring test matched "Ellis Island National
Museum of Immigration" to a short unrelated row and `16` to `1600 Beacon Street`, so
matching is token-boundary with a generic-word stoplist; and a fixed 20 deg
tolerance failed every close-range object. Residual artifacts remain — "Boston Gas
Tank" still matches a row named `Boston`, "Omni Mount Washington Resort" matches
`Mount Washington` — so `WRONG_BEARING` counts need reading, not summing.

**Result: the baseline re-run is itself the variance test.** Re-running the *v4
prompt* on f106 produced **no names at all**, not the Chicago five. But f238, which
originally misnamed the islands, came back with **"One World Trade Center" and
"Ellis Island National Museum of Immigration"** — New York. So:

> the *specific* out-of-region names are sampling variance; the *failure mode* —
> recognising a waterfront scene as a famous waterfront somewhere else — reproduces.

That is a correction to the first pass of this diagnosis, which treated the Chicago
cluster as deterministic evidence. It is the mode that is systematic, not the names.

**v2 of the prompt** (`osm_tags_farfield_v2` in `semantic_landmark_extractor.py`)
keeps v4's licence to recognise distant natural features and adds four constraints:

1. **Name the structure, not the scene.** A name must rest on that structure's own
   outline, top, proportions, colour, signage — never on the overall view resembling
   a place you know. "A waterfront with tall towers, and I know a famous waterfront
   with tall towers" identifies a *kind* of scene.
2. **One locality per panorama.** If the candidate names come from different cities
   or regions, that is a resemblance rather than the place: drop them all.
3. **Lookalikes need a differentiator you can see** — and the description must say
   what it was. This is v1's deleted guard, restored but narrowed so it bites on
   rows-of-similar-things rather than on all recognition.
4. **A painted number or letter is `ref=`, never `name=`** — a designator identifies
   a mark only relative to its own channel.

**Measured, 3 passes each (66 frame-samples per config).** Naming is
high-variance — per-pass name counts were 31/34/25 for v4 and 37/23/31 for v2 — so
single-pass comparisons are worthless here.

| per 3 passes | v4 prompt | v2 prompt |
|---|---|---|
| names emitted | 90 | 91 |
| `AT_BEARING` | 59 (66%) | 64 (70%) |
| `WRONG_BEARING` | 7 | **18** |
| `NO_SUCH_ROW` | 18 | 9 |
| **out-of-region names** | **2** | **0** |
| **designator-as-name** | **6** | **0** |
| mountain names | 19 | 23 |
| harbour names (leg3) | 7 | 2 |

The buoy rule works exactly as written: leg2's f118 went from
`{'name': '4', 'colour': 'red', ...}` to `{'ref': '4', 'colour': 'red', ...}`, and
designator-as-name went to zero in all three passes.

**But v2 is NOT a clean win, and the earlier two-pass reading of this experiment was
wrong.** `WRONG_BEARING` more than doubled. Classifying every one of those rows by
hand — a metric artifact is one where the matched row's name is a much shorter
fragment of the emitted name (`John Hancock Tower` -> a row named `John Hancock`,
`Boston Gas Tank` -> a row named `Boston`), marginal is within 8 deg of tolerance:

| per 3 passes | v4 prompt | v2 prompt |
|---|---|---|
| **REAL wrong-bearing** | **2** | **7** |
| marginal | 1 | 2 |
| metric artifact | 4 | 9 |

v2's new real errors are all *in-region* misdirection: `Rowes Wharf` three times at
35-48 deg, `Spectacle Island` at 37 deg, and `Fort Warren` once at 123 deg — the
island misnaming was reduced, not eliminated. That is the **same failure class as
T224**: a name that resolves to a real catalog row in the wrong place. Measured as a
share of names emitted, real misdirection went from 2/90 (2.2%) to 7/91 (7.7%).

So the honest summary of the trade:

| | v4 prompt | v2 prompt |
|---|---|---|
| out-of-region names (inert, but pure noise) | 2 | **0** |
| designator-as-name (a 19 km gamble) | 6 | **0** |
| unresolvable names (inert) | 18 | **9** |
| **in-region misdirection (the T224 mode)** | **2** | **7** |

v2 converts inert failures into resolvable ones. That is good when the resolution is
right (`AT_BEARING` 66% -> 70%) and bad when it is wrong, and the control set cannot
weigh those against each other — only the filter can, because the cost of a wrong
identity is not linear (the ablation above put one at ~44x).

**Hypothesis, tested: clause 2 was the culprit.** "All the names you give for one
panorama must belong to ONE locality" looks protective but plausibly makes the model
*commit* to a locality and then name more things in it. v3 = v2 with clause 2
removed, 3 passes, ~$5. Per pass, 3 passes each, 66 frame-samples per config:

| per pass | v4 (shipped) | v2 (4 clauses) | v3 (no one-locality) |
|---|---|---|---|
| names emitted | 30.0 | 30.3 | 20.3 |
| `AT_BEARING` rate | 66% | 70% | **74%** |
| **REAL wrong-bearing** | **0.7** | 2.3 | **0.7** |
| inert (`NO_SUCH_ROW`) | 6.0 | 3.0 | **2.3** |
| designator-as-name | 2.0 | **0** | **0** |
| out-of-region | 0.7 | **0** | **0** |
| **mountain names** | 6.3 | **7.7** | 3.7 |
| harbour names (leg3) | 2.3 | 0.7 | 1.7 |
| per-pass yield | 31/34/25 | 37/23/31 | 14/29/18 |

Confirmed: removing clause 2 returns real misdirection to v4's level while keeping
out-of-region names and designators at zero, and gives the best precision of the
three (74%).

**But v3 gives the peaks back**, 6.3 -> 3.7 per pass, and the reason is visible in the
prompt: clause 3's own example list names *"a line of summits along a ridge"* as a
lookalike case, so the guard written for harbour drumlins suppresses exactly the
naming that v4's rewrite was for. v2 masked this because clause 2 pushed naming back
up — at the cost of the misdirection.

**The terrain carve-out was tried and refuted.** If clause 3's own example list
("a line of summits along a ridge") were suppressing the peaks, removing that example
and adding *a mountain's profile and its position among the summits you can also see
IS a distinguishing feature* should restore them. It did the opposite — precision
fell to 58% and peak names to 2.7 per pass, the worst of the four. So the peak loss
is **not** caused by that clause's wording and its mechanism is still unknown. Do not
re-try this without a different hypothesis.

Final four-way, per pass (figure: `runs/260819_final/prompt_tuning.png`):

| per pass | v4 shipped | +4 clauses | **KEPT: 3 clauses** | +terrain carve-out |
|---|---|---|---|---|
| names emitted | 30.0 | 30.3 | 20.3 | 18.3 |
| `AT_BEARING` rate | 66% | 70% | **74%** | 58% |
| **REAL misdirection** | 0.7 | 2.3 | **0.7** | 1.0 |
| inert (`NO_SUCH_ROW`) | 6.0 | 3.0 | **2.3** | 4.0 |
| designator-as-name | 2.0 | **0** | **0** | **0** |
| out-of-region | 0.7 | **0** | **0** | **0** |
| mountain names | 6.3 | **7.7** | 3.7 | 2.7 |
| harbour names | 2.3 | 0.7 | 1.7 | 1.0 |

**What shipped, and why.** `osm_tags_farfield_v2` carries the three-clause version:
structure-not-scene, the narrowed lookalike guard, and designator-as-`ref`. It has
misdirection at the shipped prompt's level, the best precision of the four, the
fewest inert names, and zero of both targeted modes. The one-locality clause is
recorded in the code comment as **tried and rejected** so it does not get re-added as
an obvious-looking improvement.

**Its known cost is the peaks:** 6.3 -> 3.7 named per pass on the mountain frames,
mechanism unexplained. That is a real risk to mtw legs 2 and 3, which currently
localize at 239 m and 113 m and whose failure mode was never naming. Weigh it against
the harbour side, where a wrong name is worth ~44x by the ablation above.

**A note on reading these numbers.** Per-pass yields swing by 2x on identical inputs,
so no single pass distinguishes anything and the 3-pass means still carry real
uncertainty on the small buckets (out-of-region is 2 events across 66 samples). Treat
the zeros as "this mode stopped appearing", not as proof it cannot.

**Not yet validated end-to-end, and the control set cannot do it.** These are
naming counts, not position errors, and the cost of a name is not linear in either
direction — one wrong identity was worth ~44x on leg1, while deleting 73 good
measurements cost almost nothing. The decisive experiment is a v5 extraction with
`--prompt_type osm_tags_farfield_v2`, everything else held at v4, then tracks →
audit → match → filter, on **two** legs rather than one: boston_harbor_leg1 (does the
regression go away?) and mount_washington_leg2 or leg3 (does the peak loss cost
anything?). 379 + ~380 panoramas at the measured $0.041/pano is about **$31**, plus
~$3 to re-match. Total spend on this investigation so far is ~$26 of the $300 ceiling.

Also worth knowing before that run: per-pass yields swing 2x on identical inputs
(31/34/25, 37/23/31, 14/29/18, 21/19/15), so end-to-end results on a single
extraction will carry that same variance — a single leg re-extraction that comes out
better or worse by 20% in name count has not proven anything.

### Two supporting fixes landed with the prompt

**The dossier now carries each name's confidence.** `name` is in
`NON_IDENTITY_TAG_KEYS`, so it was excluded from `tag_vote_table` and reached the
auditor as a bare count — while the extraction prompt explicitly delegates naming
doubt to the `confidence` field. `build_dossier` now also returns
`name_confidence` and the prompt line reads
`names reported, with the detector's confidence in each name: 'Georges Island' x1 (1 medium)`.
Note this alone would not have caught the Chicago names: all five were `high`.

**The audit prompt now calibrates weight against support.** `semantic_audit.SYSTEM_PROMPT`
gained an explicit ceiling — weights above 0.7 only for names carried by a
substantial share of detections or corroborated in the images, and below 0.3 for a
name reported once or at `medium` confidence — with the reason stated: a downstream
matcher treats a high-weighted name as near-decisive and will place the object at
that row however far away it is.

**And the manifest now pins the prompt.** `request_sha256` hashes the request file,
images included, so two datasets with an identical prompt get different digests and
the manifest could not answer "did these use the same prompt?". That question is
what cost a hand comparison across 4.4 GB of request JSONL. The manifest now also
records `prompt_sha256` / `prompt_chars`, read back out of the stored request so it
records what went out rather than what the tree says now:
`grep prompt_sha256 artifacts/frame_landmarks/*/*/manifest.json`.

### Which tables actually own the regression (ablation, all free)

`tier1_tables.json` entries can be blanked in a copied export
(`entries: []`, `default_log_lr: 0.0`) and the filter re-run, which isolates a
subset's contribution without touching geometry. Criterion for "uncorroborated":
every `name_candidate` has `basis: reported_by_detections`.

| run | tables blanked | median error | baseline |
|---|---|---|---|
| r004 (v4) | its 19 uncorroborated | **279 / 282 m** | 12243 m |
| r004 (v4) | 19 random others | 8284 / 16042 m | 12243 m |
| **r005 (v1 tracks)** | **its 14 uncorroborated** | **85 / 76 m** | **78 m** |
| r005 (v1 tracks) | 14 random others | 80 / 73 m | 78 m |

**This is the sharpest single fact about the regression.** r003/v1 produces just as
many uncorroborated names, and they are *inert* — blanking them changes nothing,
exactly like blanking random tables. r004/v4's are worth 44×. The regression is not
"more unsupported names", it is that v4's unsupported names point somewhere
specific and wrong.

Candidate deployable rules, tested the same way on r004:

| rule (needs no ground truth) | tables | median error |
|---|---|---|
| A — name is a dataset-wide singleton (this is the Georges Island case) | 2 | 14613 / 13759 m — **no effect** |
| **B — *every* endorsed name rests on ≤2 distinct keyframes** | 20 | **282 / 284 m** |
| C — *any* name on ≤2 kf at weight ≥0.7 | 12 | 9951 / 282 m — unstable |

Rule A settles something: **T224 / "Georges Island" is a good worked example and
individually harmless.** The damage is the aggregate of ~20 thin-support names, not
one spectacular hallucination. Do not go looking for a single culprit.

**Rule B is not safe to ship.** Applying the same rule to the runs that already
work, 2 seeds each:

| run | baseline | with rule B |
|---|---|---|
| boston leg1 r005 | 78 m (5 seeds, 77–83) | 105 / **7516** / 92 / 82 / 101 m |
| boston leg3 | 41 m | 33 / 31 m |
| mount_washington leg2 | 239 m | 307 / 360 m |
| mount_washington leg3 | 113 m | 122 / 140 m |

It rescues r004 and improves leg3, but degrades both mtw legs and turns a leg that
was stable across five seeds (77–83 m) into one that diverges on **1 seed in 5**. A rule that discards thin-support names removes real signal wherever the
names were right — which is the same trade the support analysis predicted. Do not
deploy it on this evidence; the honest state is that the *cause* is identified and
the *fix* is not.

**A null worth recording.** Measuring where each table aims — the distance from a
tracklet's triangulated position to the highest-`log_lr` rows in its own table —
does *not* separate the two runs (r003's well-conditioned tables are 9/11 over 1 km,
r004's 4/9). Most tracklets triangulate with condition numbers in the hundreds or
thousands, so triangulated position is too weak a reference for this comparison.
Use the ablation, which asks the filter directly.

### A latent hazard the same prompt change created: buoy board numbers

v4 added "a number or letter painted or mounted on the structure (e.g. a buoy's
'8', '13', '1SC') → give it as `name=<exactly as read>`". Follow that through the
pipeline on boston_harbor_leg2, where those are the *only* names produced (all six:
`4 8 18 16 10`).

The catalog does carry named buoys — the NOAA ENC import gives 263 of 330 buoy rows
a full name like `Hingham Harbor Channel Buoy 16`. So the matcher resolves the digit
correctly and by name, not by shape. For LT172 (`man_made=buoy`,
`seamark:buoy_lateral:shape=nun`, `colour=red`, audit name `16` at weight 0.95) it
produced a **two-entry table with `default_log_lr: -4.0`**:

```
  +2.9  enc:0226020647E30032  ... name=Hingham Harbor Channel Buoy 16
  -0.6  enc:022602110C66FB5F  ... name=Boston Main Channel Lighted Buoy 16
```

There is a third — `Logan Airport Security Zone Buoy 16` — omitted, so it inherits
−4.0 (≈55:1 against). And the three sit **19.4 km apart**.

**It picked the worst one.** Check each candidate against the tracklet's own
bearings — boat GPS position plus observed world bearing, so this does *not* depend
on triangulation at all:

| candidate | log_lr | max bearing residual | range |
|---|---|---|---|
| **Hingham Harbor Channel Buoy 16** ← the pick | **+2.9** | **115.1°** | 2970 m |
| Boston Main Channel Lighted Buoy 16 | −0.6 | 36.5° | 17237 m |
| Logan Airport Security Zone Buoy 16 | −4.0 | 26.4° | 12548 m |

LT172's three observations (arc 19°, residual 0.01°) put the object roughly *north*
of the boat. The endorsed row is 115° away from that — geometrically impossible, and
the two the matcher down-weighted are both less wrong.

**This is not a matcher bug.** m9's first docstring line is "Match tracklets against
the whole map, **position-free**, via chunked LLM calls" — it has no coordinates by
design, because not knowing where we are is the premise. So "connect the number to
the *nearby* buoy 16" is unavailable at that stage: among three rows differing only
in a channel name, the choice is a text judgement, and one in three is right by luck.

**And unnamed buoys are excluded, which is wrong.** The catalog holds 330 buoy rows
in this export, **60 of them unnamed**. All 60 sit at `default_log_lr` −4.0: the
table lists none of them. The default comes from
`m9_match_landmarks.py:509` —

```python
default_log_lr=to_log_lr(max(1e-4, 1.0 - nm) / max(1, len(sigs)))
```

— the matcher's residual no-match mass spread over all 7950 signatures, which
clips to −4.0 whenever it is confident it found *something*. So every row it did not
enumerate is treated as ≈55:1 impossible. For a number-only identity that is exactly
backwards: **a catalog row having no name is not evidence that it is not buoy 16.**
Bearing-consistent unnamed rows exist here (several `osm:node:*` buoys within 10–13°
of every observation) and all are excluded.

**Why the existing protection misses it.** `--instance_max_rows 5` downgrades an
`instance` match to `category` when one signature covers more than five map rows —
"the tags do not identify one object". Every named ENC buoy has a *unique* signature
covering exactly one row, so the guard never fires. The ambiguity here lives
*across* signatures that share a trailing designator, which nothing checks.

This is structural, not bad luck. **A buoy's painted number is a channel-relative
identifier**: each channel numbers from its own entrance, so the number identifies a
buoy only once you already know which channel you are in — which is the thing being
estimated. Across this catalog:

| | |
|---|---|
| named buoys placed in the catalog | 265 |
| distinct designators | 54 |
| designators used by more than one buoy | 30 |
| **buoys sharing a designator with another** | **241 (91%)** |
| median max separation between same-designator buoys | **14.7 km** (max 19.4) |
| worst case | designator `3` — **22 buoys** over 15.2 km |

**What to do about it.** The number is genuinely useful evidence — "16" narrows 330
buoys to 3 — it is just not an *identity*. The table a designator-only match should
produce has three tiers, not two:

1. every same-designator named buoy at **equal** positive `log_lr` — a disjunction,
   which is the `category` path the run logs already report as *ties — disjunctive
   matches with no unique identity*;
2. every **unnamed** buoy of compatible type at ≈0, neutral rather than excluded,
   because the catalog's name coverage is partial (60 of 330 here);
3. only *differently*-numbered named buoys pushed negative.

Two changes implement it. In extraction, record a read-off number as a `ref`-style
attribute rather than `name=`, so the matcher's name→identity shortcut cannot fire.
In `m9`, extend the `instance_max_rows` downgrade so it also counts rows *across*
signatures sharing a trailing designator, and give a designator-only match a
per-category default instead of the global one.

**Status (2026-08-19): both halves of the extraction side are done; the matcher side
is not.** `osm_tags_farfield_v2` reports a read designator as `ref=` and never as
`name=` (verified on the control set: leg2 f118 went from `{'name': '4', ...}` to
`{'ref': '4', ...}` in all three passes). And the *catalog* side needed the same
move, because ENC only ever mapped `OBJNAM → name`, leaving the board number buried
inside the string — so a detection tagged `ref=16` would have matched nothing at all.
`extract_landmarks_from_enc.designator_from_name` now parses the trailing designator
off `OBJNAM` and publishes it as `ref` ("Hingham Harbor Channel Buoy 16" → `ref=16`,
"Squantum Channel Buoy 1SC" → `ref=1SC`); `ref` was already in `_TAGS_TO_KEEP`, so it
reaches the signature without touching that append-only list.

One correction to tier 1 above: the run logs' *"N are ties — disjunctive matches with
no unique identity"* line is only a **diagnostic count** of tables that happened to
come out with their top `log_lr` shared (`export_ingest.py:173`). It is not a
mechanism and creates nothing — the matcher is an LLM and may still prefer one
same-`ref` row over another. So until the `m9` change lands, the designator is
*inert* rather than misleading: an improvement over a 19 km confident pick, but it
forfeits the constraint instead of using it.

**It is not what killed leg2**, and this was checked: blanking LT172 alone leaves
leg2 at 11101 / 10424 m against a baseline of 11108 m. leg2 dies of having no
identity evidence at all (see below), not of this one table. The hazard is latent —
it will bite on a leg that is otherwise localizing. Elsewhere the exposure is small:
boston leg3 has one designator-style name (`2`), pohang has `B` ×4 and `756`, leg1
v4 has none.

### Name availability is a separate, harder limit — and it explains boston leg2

The same pass measured how many detections carry *any* name, per dataset. This is
an extraction property, upstream of the audit:

| dataset | detections named | median localization error |
|---|---|---|
| charles_river_20260727 | 2079 / 2327 (**89.3%**) | 23 m |
| boston_harbor_leg3 | 773 / 2714 (28.5%) | 41 m |
| boston_harbor_leg1 (v1) | 346 / 1403 (24.7%) | 78 m |
| boston_harbor_leg1 (v4) | 330 / 1505 (21.9%) | 12243 m |
| mount_washington leg2 | 182 / 1034 (17.6%) | 239 m |
| mount_washington leg1 | 28 / 269 (10.4%) | 7650 m |
| mount_washington leg3 | 137 / 1769 (7.7%) | 113 m |
| pohang_canal_04 | 286 / 5073 (5.6%) | not run |
| **boston_harbor_leg2** | **6 / 855 (0.7%)** | **11108 m** |

boston leg2 — the one failure with no explanation — has names on **six**
detections in the whole leg, and its audit endorsed exactly **one** name across 57
kept tracks. There is essentially no identity evidence for the matcher to resolve,
so every table is a disjunctive tag match. That is a much better lead than anything
in the filter, and it is checkable at the extraction stage before any GCP spend.

Note what this table does *not* say: coverage bounds the outcome but does not
explain leg1's regression, where v1 and v4 have the same coverage (24.7% vs 21.9%)
and differ 150× in result. Coverage is a ceiling; name *correctness* is the
regression.

**Reproduce the diagnosis** by triangulating a suspect tracklet from GPS poses and
comparing its own position against what its table claims:
`mount_offset_diagnostics --run_dir $RD --offset_deg 214` for the geometry, then
`matching/matches.json` and `matching/signatures.json` for the claimed rows' names.
A table naming rows kilometres from where the tracklet's own rays converge is the
signature of this fault.

## The three failures are two different problems

Basin of attraction — the same exports started from a Gaussian prior at the first
truth pose instead of a uniform box (`--init truth --prior_sigma_m`):

| leg | uniform | 1 km prior | 3 km prior | reading |
|---|---|---|---|---|
| mount_washington leg1 | 7650 m | **641 m** | 975 m | can hold a pose, cannot find one |
| boston leg1 | 6440 m | 4780 m | 6349 m | bearings actively harmful |
| boston leg2 | 11512 m | 9524 m | 9262 m | bearings actively harmful |

**mount_washington leg1 recovers completely from a coarse prior.** Its 39
measurements are enough to *hold* a pose and not enough to *find* one, which is
global ambiguity and nothing more; give it a few-km prior and it works. Nothing in
the filter needs changing for that leg.

**Boston legs 1 and 2 do not recover even when started within 1 km of the truth** —
they walk out to 4.8 and 9.5 km. That is not insufficient evidence, it is *wrong*
evidence, and it is the signature that sent us looking at the v4 rollout above.
The dead-reckoning floor on leg1 is 7107 m, so its r004 bearings are worth 9%.

### Looking at a run

Three views, all from the same run directory:

```bash
# static pair: catalog + truth + estimate, and error/heading/ESS/null vs keyframe
bazel run //…bearing_only_localization:plot_run -- --run_dir $RUN --animate

# the live one: particle cloud per keyframe, mode ledger, tracklet inspector
bazel run //…bearing_only_localization:viewer -- --run_dir $RUN \
    --output $RUN/viewer.html --basemap_detail 4 \
    --feather $DS/landmarks/v1.feather --sources_dir $RD --satellite $RUN/satellite

# same page, plus full per-keyframe particle sets and live counterfactual replay
bazel run //…bearing_only_localization:viewer_server -- --run_dir $RUN --port 8765
```

The map **zooms** (scroll, drag, double-click; `fit track` / `full extent`), and it
opens fitted to the trajectory because a 25 km box is context you ask for rather
than what you want to read first. The zoom is in the **projection**, not the
`viewBox`, and that choice is load-bearing: glyphs, labels and flag rings are
sized in screen units on purpose, and a `viewBox` zoom turned a 2.2-px landmark
square into a 44-px one that swallowed the map. Anything that *should* scale is
expressed in metres and divided by `mppx()`. Two related fixes came with it — the
projection is now **isotropic** (it used to squash north by 600/780, making 1σ
circles 30% too large north–south) and the map **clips**, because the page's
global `svg{overflow:visible}` otherwise paints panned-away geometry over the rest
of the document.

Two underlays, both optional and both off unless supplied:

- **`--feather` → vector basemap.** Offline OSM/ENC geometry from the dataset's own
  table: land, water, coastline, piers, bridges, buildings. Pass the *full*
  `v1.feather`, not `v2_trimmed` — the trimmed table is class-filtered to point
  landmarks and yields almost nothing. `--basemap_detail 4` raises the vertex
  budgets and tightens simplification; the defaults were sized for reading the
  whole extent, which the zoom invalidated.
- **`--satellite` → ESRI World Imagery.** `satellite_underlay.py` fetches it in
  **two levels** — a coarse mosaic over the catalog extent for context and a sharp
  one over the trajectory, since a single zoom cannot serve a 0.4 km track and an
  18 km one. It picks the highest zoom that fits a tile budget rather than making
  you bisect by hand (mount_washington leg1 gets z18 at 0.43 m/px; boston leg3's
  18 km track caps at z16). `--date YYYY-MM` pins the Wayback release near the
  capture date — construction moves in a harbour, and imagery from three years
  later is a misleading backdrop for a matcher argument.

  **ESRI World Imagery is licensed and not redistributable.** The imagery is
  embedded in the page, so a page built with it is internal-only and must not ship
  with a data release; `satellite.json` records source, release and licence so the
  provenance travels with the file. Verified against ground truth on
  mount_washington leg1: the GPS track lands on the visible trail and terminates
  at the AMC hut. One known distortion — tiles are web mercator, the run frame is
  equirectangular, so a wide mosaic stretches by up to ~0.3% over 25 km (tens of
  metres at the edge, under a metre on a fine layer).

### Do not expect a summary statistic to predict this

`observability` reports a leg's trajectory shape and evidence counts:

```bash
bazel run //experimental/overhead_matching/swag/bearing_only_localization:observability -- \
    --export_dir $RD/localization_export_llm
```

It deliberately issues **no verdict**, because two candidate predictors were
fitted to these datasets and each was refuted by the next dataset to finish:

- **course span** (how far the vehicle turned). Successes then in hand turned
  through 240° and 329° against failures at 100–179°. Refuted by
  mount_washington leg2: 116°, net/path 0.91, the straightest mountain leg, and
  it localizes to 304 m.
- **bearing density** (measurements ÷ catalog landmarks). Better motivated — every
  catalog landmark is another way for bearings to be explained by the wrong pose —
  it ordered all six datasets then measured with a clean gap at 0.033 | 0.055, and
  *correctly predicted mount_washington leg2 out of sample*. Refuted by charles:
  density 0.0185, the second lowest of the seven, and the best result in the table.
  Then refuted again **causally**, by thinning leg3's measurements at random while
  leaving its trajectory and catalog alone:

  | kept | measurements | density | leg3 | the leg at that density |
  |---|---|---|---|---|
  | 100% | 764 | 0.0578 | 34 m | — |
  | 57% | 437 | 0.0331 | 70 m | leg1: 6504 m |
  | 28% | 214 | 0.0162 | 36 m | leg2: 11651 m |

  leg3 at leg1's density is 93× better than leg1; at leg2's density, 320× better
  than leg2. Worth knowing on its own: **a leg that localizes does so on a quarter
  of its bearings**, so the pipeline is not bearing-starved where it works.

A disjunction of the two separates all seven, and is also two thresholds fitted to
seven points after seeing every outcome. Not implemented. Run the filter.

## The one general filter fix that survived: charge selection in the evidence gate

`ProposalConfig.evidence_gate_selection_charge`, **now on by default**. The gate
compares the best of N proposal hypotheses against the incumbent belief, and every
one of those hypotheses was constructed to explain the very window it is scored on.
Best-of-N beats a fixed incumbent by roughly log N nats even when all N are wrong,
so a flat margin gets *easier* to clear the more hypotheses an event generates —
backwards. Charging log N is the same multiple-comparisons correction the gate
already applies over poses *within* a hypothesis, one level up.

This was default-off for most of the night on the strength of one misleading
measurement: tested on leg1's r004 export it made things worse. But r004's bearings
are worth 9% there, so the filter was never tracking and there was no converged
belief for the charge to protect. On data where the filter does track:

| run | charge off | charge on |
|---|---|---|
| leg1, r003 tracks (2 seeds) | median 79 / 86 m, **final 20132 / 24716 m** | median 78 / 79 m, **final 368 / 367 m** |
| boston leg3 (2 seeds) | median 62 / 61 m | median **41 / 30 m** |
| charles (2 seeds) | median 25 / 24 m | median 23 / 25 m |
| mount_washington leg2 | median 239 m | 239 m (identical) |
| mount_washington leg3 | median 94 m | 113 m |

The median barely moves on the leg1 run; what changes is that a marginal
45-hypothesis injection at keyframe 358 — twenty-one keyframes from the end — no
longer destroys a belief that had tracked for 358 keyframes. And it is not merely
conservative: on that same run it still *admits* a 548-hypothesis event at keyframe
368, which carries 6.3 nats of charge and cleared anyway.

Two costs, both bounded and both worth knowing. It makes leg1's already-failed r004
run about twice as bad (6440 → 12243 m; the dead-reckoning floor there is 7107 m,
so nothing was working either way). And kidnapped recovery takes one extra
`refractory_keyframes` cycle, 60 keyframes to 70, because the first post-kidnap
event is refused and the next one carries it. `--no_evidence_gate_selection_charge`
turns it off.

### When a leg fails, what it looks like

On boston leg1, measured. The bearings are **not** the problem: median residual
4.31°, 53% under 5°, and 127 of 437 measurements carry a >80% single-landmark
claim. The filter finds the pose and then loses it — 7367 m → **133 m by kf 16**,
holding 133–376 m to kf 56, ending at 12 km. Between kf 75 and kf 105 it drifts
121 m → 2756 m while its **reported sigma stays at 284 m**, and later as low as
73 m: confidently wrong. GPS-derived odometry cannot drift like that, so it is
the measurement update dragging the pose.

The lock-in is structural. `_belief_window_reference` scores the incumbent
through its **committed** associations while hypotheses are scored under the
mixture, so a confidently mis-associated belief explains its own window better
than any alternative can — event 7 on that run scored the incumbent at **+4.75
nats while it was 5654 m wrong**, and every proposal after it was refused. The
evidence gate was doing its job; the job had become defending the wreckage.
`--association_renewal_rate` (β, default 0.1) is the knob that can release a
wrong commitment.

*Negative result, kept because the reasoning is instructive.* Charging
`log(n_hypotheses)` in the evidence gate for best-of-N selection is
statistically right — the same multiple-comparisons correction already applied
over poses one level down — and per-event it looked decisive, refusing exactly
the two injections that destroyed a converged belief while admitting the
2-hypothesis event that rescued it. **It made leg1 worse, 4594 → 12479 m.**
Per-event counterfactuals do not compose in a sequential filter: refusing two
events yields a different run, not the same run minus two events. It survives as
`ProposalConfig.evidence_gate_selection_charge`, default off.

Two operational cautions for anyone reading numbers off a single run. The
per-run spread on a failing leg is wide (leg1 over five seeds: 1984, 5069, 6504,
7633, 7986 m), so **replicate seeds before comparing anything**. And 50k
particles over a 25 km box is one particle per 107 × 107 m cell carrying *one*
random heading — 10° heading resolution at that spacing would need 1.8 M — so the
uniform prior is a formality and the resection proposal does the real work.

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
3. **Re-derive the mount offset** (Stage 6). Never inherit — not 214.0°, and not
   a sibling leg's value either. Five of the seven video datasets carried a wrong
   offset, two of them by exactly 180°, and boston legs 2 and 3 were wrong
   *because* they inherited leg1's as "same physical mount assumed" (27° and 144°
   out). Record `log_start_utc` in `pipeline_metadata.json` at ingest so
   `sun_offset_check` can run at all; without it only the relative sweep is
   available, and the sweep cannot catch a convention slip.
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
   d=pd.read_feather('/data/farfield_matching/datasets/$DATASET/landmarks/v2_trimmed.feather')
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
