# Object-tracking → landmark → matching pipeline

Reference for `experimental/overhead_matching/swag/landmark_filtering/object_tracking/`.
Turns VLM landmark detections on panoramas into merged landmarks with bearings
and map-candidate compatibility tables, ready for
`swag/bearing_only_localization/`.

- For the ordered command sequence, see [`object-tracking-runbook.md`](object-tracking-runbook.md).
- For matching strategy and open work, see [`farfield-matching-localization-plan.md`](farfield-matching-localization-plan.md).
- For the filter this feeds, see [`localization-design-doc.md`](localization-design-doc.md).

Every bazel target below lives in
`//experimental/overhead_matching/swag/landmark_filtering/object_tracking:<name>`,
abbreviated `//…object_tracking:<name>`.

---

## 1. Conventions you must not get wrong

These have each cost real debugging time. Read before touching geometry.

### 1.1 Panorama ↔ direction

`pano_geometry.py` is the **single source of truth**. Never restate the formula
inline (`track_merge.bearing_series` calls `direction_from_pano_px` for exactly
this reason — a hand-written copy in a test had the convention backwards).

```
az_cw_deg = (x / pano_w - 0.5) * 360 mod 360
el_up_deg = (0.5 - y / pano_h) * 180
```

Consequences that surprise people:

- **pano x = W/2 is azimuth 0**, not 180. x = 0 is azimuth 180.
- The azimuth wrap (359°→0°) therefore sits at the **centre** of the image, not
  at its edges. Wrap-safety tests must straddle x = W/2.
- Pinhole face layout left→right in the panorama is **180 | 90 | 0 | 270**.
- `panorama_to_pinhole.py` uses `col_frac = linspace(1, -1)`, so face yaw_090
  is CCW-**left** of forward.

**Do not use `bearing_geometry.bearing_camera_deg` for pixel work.** It is
mirrored within faces; on faces 90/270 its bearings are ~180° off physically.
The older `landmark_filtering` pipeline absorbed this via a fitted yaw
calibration with a sign parameter. Pixel-space work uses `pano_geometry` only.

### 1.2 Camera → body → world

```
bearing_body_deg  = (bearing_camera_deg - mount_offset_deg) mod 360
bearing_world_deg = (heading_deg + bearing_body_deg) mod 360      # heading ≈ GPS course
```

`mount_offset_deg` is **per rig and per leg**. For boston_harbor leg1 it is
**214.0°** (`m9_match_landmarks.DEFAULT_MOUNT_OFFSET_DEG` if a consumer needs it; the value is recorded in
`docs/farfield-matching-localization-plan.md` §0, which also records how it was calibrated).

How that number was obtained, and why the obvious method failed, is in
[§0 of the plan](farfield-matching-localization-plan.md). Short version: the
bow is **occluded by the deckhouse**, so it cannot be read off an image. The
working method is map-free — sweep the offset and minimise the median
triangulation residual, because a wrong offset rotates every bearing by a
constant and stops a static object's rays from intersecting. For leg1 the
curve is smooth and unimodal: 5.95° at 180°, **1.33° at 214°**, 4.34° at 270°.

**Re-derive this for every new leg.** See the runbook's calibration step.

Assumptions baked in: rigid mount, camera approximately level (roll perturbs
azimuth by ~`roll·tan(elevation)`, second-order for near-horizon landmarks),
bearings relative-only (the filter must never consume `bearing_global_deg`).

### 1.3 Frames and coordinates

- Working frame is **region-anchored local ENU** in metres. Not UTM (grid
  convergence ~1.3° at Boston), not raw lat/lon (cos(lat) bearing errors to
  ~8.5°). `harbor_catalog.enu_from_latlon`.
- The **anchor is the mean of the leg's GPS fixes**. It is recomputed by each
  tool from `frames_gps.csv`, so it is stable as long as the CSV is. It also
  participates in the catalog cache key — moving the anchor invalidates the
  cache.
- `frames_gps.csv` has **no course column**. Course is derived from a central
  difference of consecutive ENU positions and is gated: below 1.0 m of travel
  between neighbours the course is `None` (reporting the direction of GPS
  jitter is worse than reporting nothing). Measurements at those keyframes are
  dropped from the matching stages.

### 1.4 Measuring bearing quality

Use `bearing_matcher.triangulate()`, and use **both** numbers it returns.

- **median residual (deg)** — are these bearings consistent with *some* single
  static point?
- **condition number** — does the observing geometry determine a position at
  all? Bearings over a short arc intersect at a glancing angle, so a tiny
  residual can hide a position uncertain by kilometres along the line of sight.

**Do not use circular std of the world bearing.** An earlier version of this
pipeline did, and it was wrong: a static object 700 m away sweeps ~74° of
*genuine* bearing as the vessel passes it, so spread measures parallax at least
as much as error. It inverted the ranking on real data — a tracklet dismissed
at 50.5° spread has a 0.99° triangulation residual and the best conditioning in
the run, while one praised at 1.6° spread has condition 1259.

---

## 2. Stages

### M0 — `m0_render_boxes`
Reprojects VLM detection boxes onto panoramas. Sanity check that
detection → pano geometry is right before anything downstream. Diagnostic only.

### M1 — `m1_heading_windows`
Heading model + heading-compensated crop windows. Establishes
`az_window = az_anchor - ΔHeading` (confirmed empirically; the flipped sign
stares at open water). GPS-course heading lags rotation-in-place at the dock by
~15° over 42 s, which per-interval re-anchoring absorbs.

### M2 — `m2_sam_tracking`
SAM2 mask propagation over a keyframe interval. Checkpoint
`sam2.1_hiera_large.pt` at `/data/farfield_matching/models/sam2/`; ~43 fps
propagation on a 5090. Backend kept behind the narrow `sam_backend.py`
interface so SAM3 can be swapped in.

### M3 — `m3_track_viewer` (build + board) / `m3_build_tracks` / `m3_run_diff`

Mask-anchored track building. `m3_track_viewer` runs the tracker **and** writes
the board, per-track pages and videos. `m3_run_diff` compares two runs.

Runs are versioned: `<runs_root>/<run_name>/`. `--skip_existing_ranges` resumes
a crashed run.

**Association is purely geometric.** Semantic labels play no part — this is
load-bearing, and the reason is a real failure: at f0000/f0001 the same
'One International Place' label sat on two *different* buildings, and
mask-anchored association correctly refused the match. Semantics are audited
later (M5), never used to associate.

Support classes come from `track_builder.classify_support` on three
mask-vs-box metrics: `iou`, `inter_over_mask` (i/m), `inter_over_box` (i/b).
Use asymmetric containment, not plain IoU — the fort case had iou 0.30 but
i/m 0.95.

| class | meaning | counts as support? |
|---|---|---|
| `continue_clean` | tight agreement, re-anchors the mask | yes |
| `merge_superset` | box contains the mask, coherently | yes |
| `split_child` | box inside a larger mask | yes |
| `weak` | marginal but mutually agreeing | yes |
| `context` | box contains the mask but mask fills <10% of it | **no** |
| `none` | rejected | no |

Support means: votes on semantics, resets patience, extends `end_keyframe`,
**claims** the detection (so it cannot seed another track), and feeds window
sizing. `context` and `none` do none of those.

#### `TrackBuilderConfig` (every parameter)

| parameter | default | meaning / why |
|---|---|---|
| `window_px` | 1024 | base crop side for SAM |
| `window_extent_factor` | 2.0 | window covers this × object extent |
| `window_quantum` | 512 | window sizes quantised to this |
| `window_max_px` | 3072 | cap |
| `clean_iou` | 0.45 | ≥ this ⇒ `continue_clean` (or `weak` if it would truncate the mask) |
| `reanchor_min_inter_over_mask` | 0.75 | i/m needed to re-anchor; every healthy re-anchor measured ≥0.77 |
| `superset_inter_over_mask` | 0.7 | i/m above which a box is a superset of the mask |
| `superset_max_inter_over_box` | 0.4 | i/b below which it is a superset rather than a match |
| **`superset_min_inter_over_box`** | **0.10** | **floor beneath superset: below this the class is `context`.** See §3.1 |
| `child_inter_over_box` | 0.7 | i/b above which the box is a child of the mask |
| `child_max_inter_over_mask` | 0.4 | i/m below which it is a child rather than a match |
| `weak_min_iou` | 0.15 | one route into `weak` |
| `weak_min_containment` | 0.5 | containment route into `weak` |
| **`weak_min_complement`** | **0.10** | **mutual-agreement floor for containment-only weak support.** See §3.2 |
| `birth_min_coverage` | 0.15 | mask∩box / box area at the prompt frame |
| `birth_max_spill` | 0.5 | mask outside box / mask area |
| `birth_min_dominant_cc` | 0.6 | largest connected component / mask area |
| `patience_keyframes` | 15 | unsupported keyframes tolerated on an established track (a crane survived a 5-keyframe detection gap) |
| `patience_unsupported_keyframes` | 3 | for tracks never supported after birth — these are one-shot detections; in one full leg 138/294 tracks had zero post-birth support and each zombie-propagated 15 keyframes |
| `min_mask_area_px` | 20 | below this the mask is dead |
| `drift_gate_px` | 150.0 | near-miss distance for the drift alarm |
| `drift_patience` | 3 | consecutive near-misses before the alarm fires |

Birth gating rejects a seed whose prompt-frame mask is unhealthy; rejections
are recorded in `rejected_births` with a reason (`coverage`, `fragmented`,
`spill`). Mask connected-component count also flags bad masklets — a lattice
tower gave cc=25 because background showed through the lattice and the mask
slid onto containers; demote those to geometry-only.

**Drift-alarm blind spot:** thieves that *contain* the mask register as
support, not as a near-miss, so the alarm cannot see them. The `context` class
removes their life-support, which covers the observed cases, but a
containment-shaped thief holding i/b ≥ 0.10 would still pass. If that appears,
the fix is a track-level check (mask bearing diverging from the supporting
boxes' trend), not another threshold.

**Artifact gotcha:** keep numpy scalars out of records — `np.bool_` from
`mask_health` crashes the JSON writer.

### `keyframe_viewer`
One page per keyframe: the full annotated panorama (detection boxes coloured
stably per (tag, name) identity; every alive track's mask bbox in red labelled
`T##`) plus a table of every detection with its chip, tags, `distance_estimate`,
description, and its relationship to the tracker.

Relationships shown, in priority order: **★ seeds T#** (this detection founded
a track), **birth rejected (reason)**, real supports, then refused associations
labelled `(not support)`. The birth link matters — it is stored only in
`track["birth_obs_id"]`, not in any `supports` list, so a viewer built from
supports alone renders a track's founding detection as "unclaimed".

`--pano_width` (default 3072) sets the rendered panorama width; `--kf_start` /
`--kf_end` limit the range. Output is large: ~380 pages ≈ 210 MB.

### M4 — `m4_level1_review` (superseded, kept for provenance)
Hand-review site for within-track evidence hygiene; 586 supports reviewed by
hand. Its conclusion shaped M5: **no per-instance LLM for evidence hygiene** —
rules plus two small cached tables (tag-pair affinity ~15 entries, name aliases
~5) covered everything seen. Superseded by M5 for routine use.

### M5 — `m5_build_audit_requests` → Vertex → `m5_audit_results_viewer`

Per-track semantic canonicalization by VLM. **One call per track**, O(tracks)
not O(supports).

The request is a *dossier* plus chips. Hard constraints on the dossier, all
enforced by construction and tested:

- **No dataset, run, or location identifiers.** Keyframes appear as relative
  time indices `t0..tN`. A model that knows it is in Boston Harbor will guess
  Boston Harbor names.
- **No hand-written interpretation.** Every line is derivable from the track
  artifact alone. Temporal structure is conveyed by run-length encoding, not by
  narrative like "X dominates early", which cannot be generated procedurally.
- **All tags, primary and additional**, with per-confidence counts — not an
  opaque weighted score.

Chips draw the detection box in green and **the tracked mask's bbox in red**.
This is not cosmetic: a hand review that omitted the mask bbox produced a
conclusion ("giant boxes are often correct") that was exactly wrong.

Chip selection is deterministic: first and last support, then the highest-IoU
support of each primary-tag run, then context boxes **one per distinct tag**
before doubling up on any tag. That last rule exists because a run that picked
context chips purely by lowest fill showed the model two fort boxes and no
tank box, and it then recommended a tank cluster as its own landmark from text
alone — the detection was a misidentified seawall.

#### `AuditConfig`

| parameter | default | meaning |
|---|---|---|
| `min_supports` | 3 | tracks below this are not audited |
| `max_support_chips` | 6 | support chips per request |
| `max_context_chips` | 2 | context chips per request |
| `max_description_samples` | 10 | verbatim descriptions quoted |
| `chip_height_px` | 320 | rendered chip height |
| `thinking_level` | `HIGH` | Gemini thinking level |
| `classifier` | `TrackBuilderConfig()` | classifier used to **recompute** support classes, so old artifacts are audited under current rules |

#### `m5_build_audit_requests` CLI

| flag | default | meaning |
|---|---|---|
| `--run_dir` | required | tracking run to audit |
| `--dataset_base` / `--landmark_base` | boston leg1 | panoramas / VLM predictions |
| `--min_supports` | 3 | overrides `AuditConfig.min_supports` |
| `--max_tracks` | none | cap the number of requests (debugging) |
| `--model` | `gemini-3-flash-preview` | used only with `--submit` |
| `--submit` | off | run the requests through Vertex immediately instead of only writing `requests.jsonl` |
| `--parallel` | 8 | concurrent online requests |

`m5_audit_results_viewer` takes `--run_dir`, `--dataset_base`,
`--landmark_base`, and `--no_extra_chips` (skip re-rendering strike/secondary
chips).

#### Output schema (`semantic_audit.TrackAudit`)

Categorical judgments and matching keys only — **never scalar reliability
scores**, which are computed procedurally downstream.

- `landmark_kind` — fixed_structure / navigation_aid / terrain / vegetation /
  vessel_or_vehicle / transient_phenomenon / mixed_or_unclear
- `single_object`, `valid_segments[{start_t,end_t}]` — the spans belonging to
  the primary object. **Applied before co-visibility in M6.**
- `verdict` — keep / keep_partial / drop; `drop_reason`
- `primary_object` — weighted `tags`; **`name_candidates[{name, weight, basis}]`**
  where basis ∈ read_from_images / reported_by_detections / both;
  `name_aliases`; `description`; `distinctive_features`;
  `extent` ∈ point_like / small_extended / large_extended
- `strike_votes` — contaminant detections (re-routed, not deleted)
- `secondary_objects` — typed `relation` (part_of_primary / contains_primary /
  occluder / adjacent / background) + `worth_own_landmark`
- `confidence`, `unresolved`

**Names are a weighted list, never a single verdict.** See §3.3.

`parse_result_line()` upgrades pre-`name_candidates` payloads on read, so older
`results.jsonl` files still load.

#### Procedural evidence (`semantic_audit.build_evidence`)
Computed in code, never by the model, and travels with every record so
downstream can trust a 68-support track over a 4-support one:

`n_supports`, `n_supported_keyframes`, `lifetime_keyframes`, `support_density`,
`n_context_only`, `n_reanchors`, `drift_alarm`, `close_reason`, `median_iou`,
`median_box_px`, `max_box_px`, `camera_azimuth_span_deg`, `tag_votes`,
`n_distinct_tags`, `tag_top_share`, `confidence_counts`, `name_votes`,
`n_named_supports`, `n_distinct_names`, `name_top_share`, `name_margin`,
`name_contested`.

`name_contested` is a **flag, not a gate** — nothing is filtered on it.

### M6 — `m6_merge_tracks`

Consolidates tracks into landmarks and emits fused bearings.

**Merging is constraint satisfaction on geometry, not similarity search.** Two
tracks alive at the same keyframe are in the same camera frame, so their mask
positions compare directly with no ego-motion, no heading, no GPS. That makes
it the most trustworthy signal available.

| pair verdict | condition | action |
|---|---|---|
| `duplicate` | mask bbox IoU ≥ 0.50 at shared keyframes | **merge** |
| `parent_child` | containment ≥ 0.70 and area ratio ≤ 0.50 | **link**, never merge (fort on island) |
| `ambiguous` | 0.05 ≤ IoU < 0.50 | neither merge nor block; emitted for adjudication |
| `distinct` | IoU < 0.05 | **hard cannot-link**, overrides any name agreement |
| `disjoint` | no shared keyframes | handoff **proposal** only |

Why `ambiguous` exists: measured on leg1, the partial-overlap band genuinely
mixes true duplicates (Tobin Bridge IoU 0.39, Commonwealth Pier 0.25) with
genuinely different adjacent buildings (Boston Harbor Hotel vs One
International Place, 0.23). No threshold on IoU, angular separation, or
separation normalised by object width separates them — normalised separation
puts the different-buildings case at 0.16, between true duplicates at 0.11 and
0.25. So geometry declines to decide, and this is where semantic adjudication
earns its keep.

`cluster()` takes connected components over duplicate edges and **enforces
cannot-links**: a component containing a `distinct` pair is contradictory
(A~B, B~C, but A and C provably differ), so the weakest duplicate edge is
dropped and reported in `merge_conflicts` rather than silently welded.

**Under-merge freely; over-merge never.** Two unmerged tracklets of one object
are just two landmarks that both match the same map feature, which the filter's
data association already handles. A wrong merge fuses two objects into one
landmark with a bimodal bearing — a corrupted measurement nothing downstream
can recover from.

#### `MergeConfig`

| parameter | default | meaning |
|---|---|---|
| `duplicate_min_iou` | 0.50 | merge threshold (scale-free, so it works for towers and islands alike) |
| `child_min_containment` | 0.70 | parent/child containment |
| `child_max_area_frac` | 0.50 | child must be meaningfully smaller |
| `ambiguous_min_iou` | 0.05 | below this a cannot-link is safe to assert |
| `min_covisible_keyframes` | 3 | fewer ⇒ refuse to merge but do **not** assert cannot-link on thin evidence |
| `handoff_max_gap` | 30 | max keyframe gap for a handoff proposal |

#### CLI

| flag | default | meaning |
|---|---|---|
| `--min_supports` | 1 | tracks below this skip consolidation |
| `--epoch_keyframes` | 5 | keyframes fused into one bearing measurement |
| `--bearing_sigma_deg` | 1.0 | per-observation bearing sigma |

#### Bearing fusion
One fused bearing per tracklet per information epoch, matching the filter's
sparse-epoch design. Consecutive bearings on one mask are strongly correlated
(same mask, same tracker), so treating them as independent would overcount
evidence. `kappa` combines the per-observation sigma with a quarter of the
object's angular width and **deliberately does not grow with the number of
fused keyframes** — the conservative choice while the correlation is
unmodelled.

Emitted bearings are **camera-frame** (`bearing_camera_deg`). The mount offset
is applied downstream (M7), so a re-calibration does not require re-running M6.

### `bow_calibration`
Temporal-median image of the panoramas: the vessel is the one thing rigidly
fixed in the camera frame, so it renders sharp while the world washes out.
Produces `median.jpg`, `staticness.png`, `median_with_azimuth.jpg` (azimuth
ruler overlaid), and `calibration.json`.

**It does not detect the bow, and on this rig it cannot** — the deckhouse
occludes it. Use it to *read* structure and to mask the vessel out (the
staticness map) for other estimators. A naive "brightest saturated sky pixel"
sun detector returns ~186° on every frame, which is the white deckhouse roof,
not the sun; the tell is that the sun's camera azimuth must swing as the
vessel turns.

| flag | default | meaning |
|---|---|---|
| `--max_frames` | 150 | panoramas sampled evenly across the leg |
| `--band` | 0.45 0.95 | vertical fraction of the frame holding the vessel |
| `--mount_offset_deg` | none | if given, drawn on the strip to check a candidate reading |

### `harbor_catalog` (library)
Map side. Loads `landmarks/<name>.feather` (OSM + ENC) into region-anchored ENU.
It applies no class filtering of its own — the trimming is a separate
pre-pass (`trim_landmark_feather.py`, §3.4), so which classes are present is
decided by **which feather you point it at**: the full table or the trimmed one.

- `load_catalog_cached()` — decoding 156 k WKB geometries plus convex hulls
  takes ~227 s; cached it is ~1.1 s. The cache key covers the feather's
  identity, the anchor, and **`CACHE_VERSION`** (currently 3). **Bump
  `CACHE_VERSION` whenever `CatalogEntry` fields or the parsing/pruning logic
  change** — without it, a fix silently keeps serving entries built by the old
  code, which happened once.
- `bearing_span_from()` — the angular interval a candidate subtends, from its
  convex hull. Note the widest angle comes from the **near** corners, not the
  centroid range.
- `bearing_span_from()` — the angular interval a candidate subtends, so an
  extended feature is matched as an extent rather than a centroid. Used by
  the mount-offset calibration. (`wedge_candidates()` was removed: see below.)
- `position_sigma_m`: **ENC 5.0 m** (survey grade), **OSM 15.0 m**. The filter
  projects this into the angular domain via `kappa_eff`, so the accuracy
  *class* matters more than the exact position.
- `_id_text()` — the feather stores `id` as the **repr of a tuple**
  (`"('node', 31419650)"`), not a tuple, and all parts must be kept: node 123
  and way 123 are different features.

Tag pruning is `prune_harbor_tags`, **deliberately not**
`semantic_landmark_utils.prune_landmark`. See §3.4.

### M9 — `m9_match_landmarks`
Matches every audited landmark against the **whole map**, with no spatial
information of any kind, and writes the filter's `CompatibilityTable`.

The map is far larger than a prompt, so distinct tag *signatures* (identical
bundles are indistinguishable to a text matcher) are split into chunks and the
tracklets into small batches; every batch is asked about every chunk. A matched
signature expands to every map row carrying it.

Queries are built from the **audit output only** — weighted tags, weighted name
candidates with basis, kind/extent, description, distinctive features and the
auditor's `unresolved` note. There is no fallback to raw detector votes: a track
that was never audited has no canonical semantics and is dropped from the
pipeline rather than matched on un-adjudicated output.

Outputs `matching/{requests,results,signatures,matches,compatibility}.json`.

### M10 — `m10_match_viewer`
Observation beside matched map landmark: confidence, instance/category, how
many map rows the signature expanded to, links back to the track page, the
audit entry and the keyframes.

### Removed: M7/M8 (bearing-wedge pairing)
M7 gated map candidates by a bearing wedge computed from the vessel's GPS
position, and M8 reviewed the resulting labels. Both were deleted: selecting
candidates by where the vehicle was and then asking the filter to recover where
the vehicle was is circular, so any localization result built on it would have
been invalid rather than merely optimistic. Matching is now M9. The rule-based
`TagRuleScorer` went with them — it only ever scored wedge shortlists.


### Vertex submission
Both M5 and M7 emit `requests.jsonl` in `vertex_batch_manager`'s format:
`{"key", "request": {contents, systemInstruction, generationConfig}}`.

```bash
export GOOGLE_CLOUD_PROJECT=… GOOGLE_CLOUD_LOCATION=global GOOGLE_GENAI_USE_VERTEXAI=True
bazel run //experimental/overhead_matching/swag/scripts:vertex_batch_manager -- run-online \
    --input  <dir>/requests.jsonl \
    --output <dir>/results.jsonl \
    --model gemini-3-flash-preview --parallel 8
```

`run-online` **resumes**: it skips keys already present in the output and
retries error records. It stops after 3 consecutive errors (quota/auth
signature). Observed cost: M5 audit 84 tracks ≈ 1.0 M tokens in ~4 min.


### `bearing_matcher` (library) — the localization seam
Produces `CompatibilityTable` dicts whose field names match
`bearing_only_localization.structs.CompatibilityTable` exactly:
`tracklet_id`, `matcher_version`, `entries[{landmark_id, log_lr}]`,
`default_log_lr`, `clip_lo`, `clip_hi`, `status`.

- Scoring is **deliberately uncalibrated** (design doc §6): a scorer returns
  raw scores; `to_compatibility_table` maps them through a tuned affine
  transform + clip. The clips and the filter's null hypothesis carry the safety
  burden. Defaults `DEFAULT_CLIP = 4.0`, `DEFAULT_LOG_LR = -2.0`.
- Entries equal to `default_log_lr` are **omitted** — absent landmarks score
  the default per that struct's contract.
- `TagRuleScorer` is the no-training control (tag weight 1.0, name 3.0, ENC
  bonus 0.5, conspicuous bonus 1.0). Learned scorers plug into the same
  `score_candidates` interface.
- `effective_candidates()` — `exp(entropy)` of the normalised scores: 1.0 when
  one candidate dominates, N when N are indistinguishable. This is the
  "One International Place vs a tower" distinction. It is **not** folded into
  `log_lr`, because the filter already handles a spread match as a mixture and
  discounting it again would penalise the same ambiguity twice.
- `triangulate()` — see §1.4.
- `estimate_mount_offset()` — per-tracklet implied offsets **and their spread**.
  A rigid camera implies one constant, so a small per-tracklet residual with a
  large spread across tracklets means either a hypothesis is wrong or the
  heading reference drifts. Do not average the two cases blindly.

#### `WedgeConfig`

| parameter | default | meaning |
|---|---|---|
| `bearing_slack_deg` | 6.0 | wedge half-width beyond the tracklet's own width |
| `max_range_m` | 20000.0 | visibility horizon |
| `min_range_m` | 50.0 | ignore anything on top of the vessel |
| `min_observation_support` | 0.5 | candidate must sit in the wedge from this fraction of observations — a real landmark stays, clutter drifts out |

### `run_index`
Landing page for a run: walks the directory, reports what each stage produced,
and links all viewers in pipeline order. Pass `--runs_root` to also write an
index of every run. It takes over `index.html` and preserves the m3 board as
`board.html` (first run only).

---

## 3. Decisions, and the evidence for them

### 3.1 The `context` class
A dying island-remnant mask was kept alive 50 keyframes by successive *other*
islands' giant boxes (superset, i/m 1.0, i/b 0.01–0.08); the track spanned
three islands. Those fake supports reset patience, inflated the adaptive window
to 2048 px around an 84 px mask, and let SAM re-attach the remnant to the next
island.

Fix: a coherence floor `superset_min_inter_over_box = 0.10`. Below it the class
is `context` — recorded for merge/occlusion evidence, but **no votes, no
patience reset, no window growth, and no claim**, so the real detections seed
their own tracks. Legitimate granularity supersets (a fort on its island,
i/b 0.16–0.40) are unaffected.

Effect when first active during tracking: tracks 297 → 359, drift alarms
28 → 15, `mask_dead` 118 → 81, `starved` 113 → 225 (tracks now die honestly
instead of being zombie-fed). Cost: audit-eligible tracks 96 → 83 — evidence
spread across more, thinner tracks.

### 3.2 `weak_min_complement`
A foreground island's box covered 0.52 of a *background* island's mask at
iou 0.00 and was absorbed as weak support, polluting votes **and** suppressing
the occluder's own track seed. Association is pure 2-D image space and
occlusion is not modelled, so containment-only weak support now requires mutual
agreement: `weak_min_complement = 0.10`. 15/476 weak supports in a full leg had
this signature, including one giant box "supporting" four different tracks at
once.

### 3.3 Names are a distribution, not a verdict
One track, born from a **correctly identified** 'One International Place'
detection and tracking the same dark glass tower cleanly for 120 keyframes with
no drift alarm, collected eight different names (Custom House Tower ×20, OIP ×7,
Rowes Wharf ×5, Harbor Towers ×5, Millennium ×3, Prudential ×2, …). Modal
voting renamed it Custom House Tower — the correct answer was present at birth
and aggregation destroyed it.

Vote reweighting does **not** fix this. Restricting to boxes ≥40 px gives 13 vs
7; restricting to <2 km gives 9 vs 6. The distribution barely moves, because the
failure is dense-skyline misidentification at *every* range, not a resolution
artifact.

So the schema carries `name_candidates` with weights and a `basis` field, the
prompt forbids ranking a name above one with more supporting detections unless
the images justify it, and resolution is deferred to the matcher — which has a
map the auditor does not.

A share-based gate is also insufficient on its own: one track with **one** stray
name vote has 100% share. `n_named_supports` is what reveals it.

### 3.4 Class filtering of the map (reversed 2026-08-12)
This section previously read "No class filtering of the map". **That is now
reversed: a trimmed catalog is produced and is the intended input for
matching.** The reasoning that argued against it is kept below, because it
defines the constraints the trim has to satisfy — and it did shape the rules.

What did not change: the filter's uniform `log_prior = −log(n)` dilution is
still best solved by **per-particle spatial gating** (design doc §5.3
`cand(x)`), not by a smaller map. Measured: a ±2° wedge returns 579 candidates
in the dense downtown but only 25–36 out in the harbor, so the dilution problem
is local to the city. Gating is **still unimplemented**, which is what tipped
the trade — trimming is an interim, and does not discharge that work.

Two facts that a naive prominence filter would have destroyed, and that the
trim explicitly preserves:
- **80 ENC entries carry `description: "visually conspicuous"`** — a surveyor's
  judgment about what is identifiable from the water, on exactly the classes we
  detect (storage_tank 31, chimney 11, tower 10, lighthouse 6, water_tower 2,
  dome 4). All 883 ENC rows are kept unconditionally.
- Container cranes, the correct answer for one tracklet, carry **no name and no
  height**. `man_made=crane` is structural in the trim, with a test asserting
  it survives.

**Artifacts.** `trim_landmark_feather.py` writes
`harbor_osm_enc_trimmed_v1.feather`; `harbor_osm_enc_v1.feather` is never
modified and stays the source of record. Anything needing completeness —
occlusion masks, compound skyline landmarks, the shoreline signature — loads
the full table. Current trim: **184,805 → 13,210 rows (7.1 %)**.

**Guard.** `landmark_positive_set.py` freezes the pairing run's labels into 58
tag signatures (132 positives: 53 instance, 79 category) and the trim reports
recall on every run; it is 1.0000 today. The guard is thin — 58 signatures from
one leg — and has already missed a real defect once (Bunker Hill Monument, a
67 m obelisk tagged only `name` + `tourism=information`, dropped while recall
read 1.0), so spot-checks against known landmarks stay part of the loop.

Generic entries that survive the trim are still handled by **uniqueness
weighting**, not exclusion: LOCI's labeller assigns a 1–5 uniqueness score and
`export_correspondence_similarity --uniqueness_weighted` applies
`1/log2(1 + N_matches)`.

**Per-particle gating remains a prerequisite, not an optimisation, and is not
yet implemented in the filter.**

#### Harbor tag vocabulary
`prune_harbor_tags` replaces the street-level keep-list because that list was
built for VIGOR panoramas where a shopfront's opening hours and housenumber are
legible. Here the nearest landmark is hundreds of metres away.

- **Added:** the maritime vocabulary the tables actually carry — above all
  `seamark:*` (559 rows) and ENC `object_class` (883 rows), the surveyed
  navigation aids, every one of which the street-level list drops. Plus
  light/beacon subtags and appearance tags that survive distance (height,
  colour, material, roof:shape).
- **Dropped:** `addr:*`, payment, opening hours, surface, lanes, `name:<lang>`,
  massgis ids — unobservable at range, pure bundle dilution.

Distribution shift against the released checkpoint is **accepted deliberately**;
the model is being retrained on this environment.

**Note for retraining:** cross-feature 4 of the released model is
`housenumber_overlap`, which is permanently 0 in this domain
(`addr:housenumber` is dropped). That slot is free for something far-field
relevant — angular-size agreement, or a seamark-category match.

### 3.5 Set 2 size and ordering
`--max_set2` was 60, a carry-over from "a tile's worth of landmarks". It cost
recall: a Conley Terminal container crane, the correct answer for a tracklet
whose bearings triangulated to **21 m** from it, sat outside the cutoff. **A
candidate not in Set 2 can never be labelled**, so every miss is a training
example never obtained. At ~25 tokens per entry, 500 entries is ~12 k tokens per
request — the same order the per-track audit already costs. Measured prompts:
4 k / 6 k / 11 k tokens.

Ordering went through two cliff failures before becoming a score:
1. Tiered as "ENC → named → salient" buried a 46-storey, 183 m tower at
   position 51 of 60 beneath named bus stops.
2. Adding a **binary** tall/not-tall tier let a 6-storey, 19 m building outrank
   that same tower, and then dumped every height-less container crane into the
   bottom bucket with the benches.

A sum degrades gracefully where a tier ordering falls off a cliff. After the
fix: OIP at position 11, cranes at 17–21.

### 3.6 Artifacts can predate the rules that judge them
Run `r002` was built before `superset_min_inter_over_box` and
`weak_min_complement` existed. Every viewer **recomputes** support classes with
the current classifier for display, so `r002` *displays* `context` entries — but
its tracks were built under the old claiming behaviour, where those detections
claimed their observations and suppressed rival seeds.

Check `config` in `tracks_*.json` for the keys before drawing conclusions about
tracking behaviour from an old run. Track ids are **not stable across runs**.

---

## 4. Data contracts

### `<run_dir>/tracks_<range>.json`
`{range, config, tracks[], rejected_births[]}`.

Track: `track_id`, `birth_obs_id`, `birth_keyframe`, `status`, `close_reason`,
`end_keyframe`, `last_keyframe`, `modal_label`, `n_supported_keyframes`,
`records[]`. `end_keyframe` can be `None` for tracks alive at the end.

Record: `keyframe`, `action` (birth / reanchor_clean / continue_mask /
unsupported / mask_dead), `window_origin` [x, y] in pano px, `window_px`,
`mask_area`, `mask_bbox_window` [x0,y0,x1,y1] **in window coordinates**,
`health`, `supports[]`.

Support: `obs_id` (`f####__lm#__box#`), `class` **as recorded at run time**,
`box_window`, `iou`, `inter_over_mask`, `inter_over_box`.

Rejected birth: `obs_id`, `keyframe`, `health{ok, reason, …}`.

### `<run_dir>/semantic_audit/`
`requests.jsonl`, `results.jsonl`, `audit_meta.json` (key → track join info,
support obs ids by `t`, chip paths), `chips/`, `preview/`, `review/`.

### `<run_dir>/merged/`
- `landmarks.json` — `landmark_id` (`L` + member track ids), `track_ids`,
  `n_supports`, `n_supported_keyframes`, `keyframe_span`, `name_votes`,
  `tag_votes`, `name_contested`, `parent_of`, `child_of`,
  `handoff_proposals`, `review_pairs`, `merge_conflicts`
- `pair_stats.json` — every non-disjoint co-visibility verdict
- `measurements.json` — `tracklet_id`, `source_track_id`,
  `anchor_keyframe_idx`, **`bearing_camera_deg`**, `kappa`

### `<run_dir>/pairing/`
`requests.jsonl`, `results.jsonl`, `prompts/*.txt`, `figures/*.png`,
`index.html`, `review/`.

---

## 5. Tests

```bash
bazel test //experimental/overhead_matching/swag/landmark_filtering/object_tracking/...
```

Seven targets: `pano_geometry_test`, `heading_test`, `track_builder_test`,
`semantic_audit_test`, `track_merge_test`, `harbor_catalog_test`,
`bearing_matcher_test`.

Tests that pin non-obvious behaviour, worth keeping:
- geometry wrap-safety around **pano x = W/2** (azimuth 0), not x = 0
- the extended-feature angular span uses **near** corners, not centroid range
- context chips cover distinct tags before repeating (the seawall/tank failure)
- the T4 split-plurality and the single-stray-name shapes
- contradictory merge chains split and report rather than weld
- short-baseline triangulation: tiny residual, huge condition number
- `estimate_mount_offset` recovers a known offset exactly

Gotcha when adding tests: a `if __name__ == "__main__": unittest.main()` block
must be the **last** thing in the file. Classes defined after it never run
under `py_test`.
