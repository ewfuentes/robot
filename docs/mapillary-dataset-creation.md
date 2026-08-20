# Creating a far-field dataset from Mapillary

How to turn a Mapillary app link into a dataset the farfield/landmark-filtering
pipeline can consume. Written so an agent can follow it without rediscovering the
traps; the "Things that will bite you" section is the part that matters most.

Everything is a bazel target in
`//experimental/overhead_matching/swag/mapillary_tools` (moved from
`~/scratch/mappilary` 2026-08-17). The API token is a secret and never lives in
the repo: `MapillaryClient` reads `$MLY_TOKEN`, then `~/.config/mapillary/token`
(chmod 600).

## The short version

```bash
MT=//experimental/overhead_matching/swag/mapillary_tools

# 1. Is this capture usable? GPS consistency + density, metadata only.
bazel run $MT:qc_candidates -- --seeds <pKey>

# 2. Everything, for one trajectory.
bazel run $MT:run_farfield_collection -- --trajectories folkestone_dover

# 3. Or the whole registry, pruning staged originals as it goes.
bazel run $MT:run_farfield_collection -- --trajectories all --prune_raw
```

`--trajectories` is one **comma-separated** value (or `all`/`pilot`/`pano`/`perspective`),
not a space-separated list — argparse rejects the space-separated form outright,
so you find out immediately rather than silently collecting only the first name.
Running several lanes concurrently is how the batch was done:

```bash
for lane in "nagasaki_tometome,tokyo_bay,harima_a" "london_thames,sf_bay_pano"; do
    bazel run $MT:run_farfield_collection -- --prune_raw --trajectories "$lane" \
        > /tmp/lane_${lane%%,*}.log 2>&1 &
done
```

Three lanes is a safe default: stage 5 measures ~3 GB per extraction against its
24 GB cap, and Python block-buffers stdout when redirected, so a lane's log stays
empty for minutes before flushing. Watch the raw staging dir growing rather than
the log to confirm a lane is alive.

Output follows the lifecycle lanes (`docs/farfield-data-organization.md`):
datasets in `/data/farfield_matching/datasets/<name>/`, stage-1 manifests in
`raw_material/mapillary_manifests/`, staged originals in
`raw_material/mapillary_raw/` (removed by `--prune_raw`), pinhole faces as a
versioned artifact `artifacts/pinhole_images/<name>/v1/` with a manifest.json.

## Adding a trajectory

1. Find a capture in the Mapillary web app and copy the `pKey=` value out of the
   URL. That identifies one **image**, which is all you need.

   **Do not trust the URL's `lat`/`lng`** — that is the map viewport centre, not
   the image position. One seed here was labelled "anglesey_menai" from a URL
   reading 53.2273/-4.1111 while the image is actually at 51.0224/2.1732, mid
   Dover Strait, and it turned out to be a second link into a trip already in the
   registry. Resolve the pKey against the API first:

   ```bash
   bazel run $MT:qc_candidates -- --seeds <pKey>   # usability verdict + true km
   ```

   Getting this wrong picks the wrong OSM extract for the whole trajectory —
   verify the *true* position from the stage-1 manifest's bbox (or the QC
   JSON's sequence) before choosing `osm`, and let step 3 confirm it.
2. Add an entry to `swag/mapillary_tools/farfield_trajectories.py`:

```python
"my_harbour": {
    "seed_pkey": "298475668560052",
    "user": "jg360",           # cross-check only
    "pano": True,              # verified against the API, not trusted
    "osm": "europe/united-kingdom-latest.osm.pbf",
    "enc_state": None,         # NOAA state code where US waters
    "note": "what you picked it for",
},
```

3. Find out which OSM extracts the area actually needs, rather than guessing:

```bash
bazel run //experimental/overhead_matching/swag/scripts:pbf_coverage -- \
    --suggest --bbox <west> <south> <east> <north>
```

   Stage 5 refuses to build a partial catalog, so a wrong `osm` list fails loudly
   with the uncovered bounds instead of quietly producing a thin dataset.

4. Download the extracts it names into `~/scratch/osm_downloads/`.

## The stages

| # | stage | what it does |
|---|---|---|
| 1 | RESOLVE | seed pKey → whole-trip stitched manifest |
| 2 | DOWNLOAD | ordered, resume-safe image download |
| 3 | CONVERT | images + metadata → dataset directory |
| 4 | TIMELAPSE | `trajectory.png` + `gps_timelapse.mp4` |
| 5 | OSM | landmark feather (+ NOAA ENC in US waters), merged |
| 6 | PINHOLE | four 90° faces — **equirectangular datasets only** |
| 7 | TRIM | observable-from-water subset (`trim_landmark_feather`) |
| 8 | PLOT | landmark coverage figure + gap checks |
| 9 | AUDIT | dataset contract audit, non-zero exit on failure |

TIMELAPSE moved from 8 to 4 (2026-08-17): it feeds the human triage pass, and
triage is the cheapest gate — 7 of the first 21 trajectories were rejected on
the timelapse for faults no audit sees. **Eyeball the timelapse before running
stage 5+**; a rejected trajectory then costs one mp4, not an Overpass catalog
and a pinhole render.

Run a subset with `--stages 5,8`. Stage 1 skips if a manifest exists (`--force`
to redo).

## What a finished dataset looks like

```
<name>/
├── panorama/                  -> frames (relative symlink; ingest requires this name)
│   └── f0000,51.019702,2.187010,.jpg     {f%04d},{lat:.6f},{lon:.6f},.jpg
├── frames/                    the images as captured
├── frames_gps.csv             idx, video_t_s, sensor_elapsed_s, dist_m,
│                              latitude, longitude, altitude_m, speed_mps, frame_file
├── intrinsics.csv             per frame: projection, w/h, focal_norm, k1, k2,
│                              hfov, vfov, heading_deg, heading_reference
├── extraction_log.csv         full Mapillary provenance per frame
├── pano_id_mapping.csv        pano_id, lat, lon, filename
├── pipeline_metadata.json     projection, azimuth_convention, heading_reliable, ...
├── landmarks/
│   ├── v1.feather             -> the merged catalog
│   ├── v1_trimmed.feather     -> the observable-from-water subset
│   ├── sources/               per-extract feathers before merging
│   ├── PROVENANCE.json        which PBFs/ENC cells, and the bbox
│   └── landmark_coverage.png
├── trajectory.png
└── gps_timelapse.mp4
```

Pinhole faces live at
`/data/farfield_matching/artifacts/pinhole_images/<name>/v1/` with a
`manifest.json`. (The pinhole stage wrote to the old
`/data/overhead_matching/datasets/pinhole_images/` path until 2026-08-17,
when the orchestrator moved into this repo and the default was fixed.)

## Things that will bite you

> **Frames and zeros: [`conventions.md`](conventions.md) is the register, and
> [`azimuth-convention.md`](azimuth-convention.md) is the measured evidence
> behind the panorama formulas.** Mapillary's `computed_compass_angle` is the
> bearing of the **left edge**, not the centre — believing otherwise put north
> at column 0 on six earlier datasets. Restating a frame in your own words is
> how that happens; import the constant instead.

**Images are stored as captured — except world-locked captures, which get
their pixels unwound.** Orientation lives in `intrinsics.csv:heading_deg`, and
`heading_reference` says what that bearing is *of* — `column_0` for
equirectangular frames, `optical_axis` for perspective.

The old blanket rule ("never rotate pixels") was deleted by ekf 2026-08-17
after innsbruck: some 360 apps ship *world-locked* frames — in-camera IMU
stabilization pins yaw to a world reference that drifts (no compass
correction) and re-initializes between recordings, producing a yaw jump at
every recording seam. Those pixels already contain a stabilizer's heading
estimate, and a broken one; "as captured" is not a well-defined frame there.
Re-rolling such frames to true north-at-center (the VIGOR house convention)
is therefore sanctioned; sun-ephemeris yaw is the ground truth for it (see
the sun method in the azimuth-convention notes). No unwind tool is built yet
— innsbruck, the first such capture, was resolved by trimming away the
recording whose yaw reference was unrecoverable and keeping the other.
Ordinary body-fixed captures are still stored exactly as captured. The
azimuth formula is in `pipeline_metadata.json:azimuth_convention`:

```
equirect:    azimuth = (heading_deg + (col/width)*360) mod 360
perspective: azimuth = (heading_deg + degrees(atan((2*col/width - 1)*tan(hfov/2)))) mod 360
```

**The heading-quality field to read differs by projection**, and getting this
backwards makes 14 datasets look broken:

* **equirectangular** → `heading_reliable` (true/false). It scores the heading
  against GPS course. `kurashiki_pano_dense`, `harima_a`, `harima_b_pano` and
  `sf_bay_pano` are `false`; `kurashiki`'s `compass_angle` is exactly 0.0 on every
  frame.
* **perspective** → `heading_sources_disagree` plus
  `heading_sources_median_disagreement_deg`. Here `heading_reliable` is `null`,
  meaning **not applicable, not unknown** — it is deliberately not computed,
  because a perspective camera need not point along the direction of travel, so
  scoring it against GPS course rejects a legitimately side-facing or panning rig
  (`portsmouth_navalbase` spreads 66.9°). The two heading sources are cross-checked
  against each other instead: portsmouth agrees to 8.9°, `tokyo_bay` to 5.4°, and
  `heading_sources_disagree` is `false` for both. `null` is falsy, so do not test
  this field for truthiness across projections.

See
[`azimuth-convention.md`](azimuth-convention.md) for the measurement, and note the
corollary: the six older Mapillary VIGOR datasets *were* rotated, with north at
column 0, which is 180° from the convention the self-collected datasets use.

**Most captures are not 360.** 14 of 22 registry entries are `perspective`,
56–93° HFOV. They get no pinhole faces and need a single-view path through
`ingest.py`; see [`mapillary-perspective-support.md`](mapillary-perspective-support.md) for the full list of
what to change. FOV varies *within* a trajectory (NYC has 23 distinct values), so
it must be read per frame, never from a config scalar.

**Mapillary's per-frame focal length is unreliable, sometimes unphysically so.**
It comes from SfM or EXIF and can be wrong for a long run of frames. Two real
cases, both caught by the audit's plausibility gate and repaired by the converter:

* `tokyo_bay` — 70 contiguous frames report `focal_norm` 5.7999, a **9.85° FOV**,
  against ~0.51 (88°) everywhere else. Visual check: those frames show the same
  broad horizon as their neighbours, so it is metadata, not a zoom.
* `fukuyama_yasunari` — 75% of frames report `focal_norm` ~0.058, a **167° FOV**,
  with `k1 ≈ 0.0005`. A rectilinear 167° lens does not exist, and the images are
  ordinary ~80° photos with straight horizons and no barrel distortion.

Frames outside 25–160° get the trajectory-median focal and are labelled
`intrinsics.csv:focal_source = substituted_implausible`; `api` means measured.
`pipeline_metadata.json:focals_substituted` carries the count. Two traps:

* **The median must be trajectory-wide, not per-sequence.** On `tokyo_bay` the bad
  value is the *majority of its own sequence* (70 of 106), so a per-sequence
  median substitutes the garbage straight back in.
* **A large substituted share is not itself a defect**, so don't gate on it. These
  are fixed single-camera captures, so true FOV is near-constant and a median
  estimates it well — `fukuyama_yasunari` reports 45 distinct focals for one
  4096×3072 camera. What matters is the size of the plausible set behind the
  median; the audit fails below 30 frames and warns otherwise.

Even the surviving `api` rows are noisy — `fukuyama_yasunari`'s plausible set still
spans 69–153°. Anything needing one FOV per trajectory should use the median of
the `api` rows rather than trusting a single frame.

For a dataset built before this check, or whose `_raw/` was pruned so stage 3
cannot re-run, apply it in place (idempotent):

```bash
bazel run //experimental/overhead_matching/swag/mapillary_tools:repair_intrinsics_focals -- --all --dry_run   # inspect
bazel run //experimental/overhead_matching/swag/mapillary_tools:repair_intrinsics_focals -- --all
```

**Every dataset needs a `mount_offset_deg` before the localization filter can use
it.** Design doc §5.2 makes position route through the heading state, so
camera-frame bearings must be de-rotated into the motion body frame
(`bearing_body_deg = bearing_camera_deg − mount_offset_deg`, plan Phase 1). Under
the older world-frame motion contract a constant offset cancelled; it no longer
does, and an uncorrected one rotates the whole dead-reckoned track.

```bash
bazel run //experimental/overhead_matching/swag/scripts:calibrate_mount_offset -- \
    --dataset_path <dataset> [...] --output_json offsets.json
```

It recovers the offset from the **focus of expansion** in image flow — no heading
field, no GPS course, no landmark catalog, so it is independent of everything
these datasets are unreliable about. Results land in
`pipeline_metadata.json:mount_offset`, and only calibrations that pass the gates
carry a number; the rest record why.

**The convention is the direction of travel** (decided 2026-08-14), because
`gps_to_odometry` declares body x to be the travel direction when it sets
`left_m = 0`, and because `bearing_matcher.estimate_mount_offset` already solves
for that same quantity through `bearing_world = course + (bearing_camera −
offset)`. The bow is a different angle, differing by crab; `bow_calibration.py`
measures the bow and says so.

**This estimator is not yet validated for accuracy, and an earlier claim here
that it was is withdrawn.** Checked against `boston_harbor/processed/leg1`, the
only leg with an external reference, it reports an axis of 48°/228° at axis MAD
2.0° — so 228° sits **14° off the reference 214°**, and the direction it selects
is the *other* member of the axis, 48°, chosen by an 85/15 majority. The
reference is much better supported: a unimodal triangulation-residual sweep over
26 tracklets (1.33° at 214° against 5.95° at 180°) plus 72 keyframes against a
surveyed building at mean +0.6°, std 2.42°.

A 2.0° MAD beside a 14° error is the lesson: **these gates measure precision, not
accuracy.** A systematic bias produces exactly this signature, which is why the
metadata block carries `accuracy_validated: false` and calls its gate
`self_consistent` rather than `usable`. Treat the angle as a hypothesis for
triage — "does this leg have a fixed mount at all?" — and get the number itself
from a residual sweep.

Four things learned the hard way, all of which will bite a reimplementation:

* **Pair frames by metres travelled, not frame index.** Far-field parallax over
  one 5 m video step is 0.14°, under a pixel. 200 m works for most; short tracks
  need less (`baltimore_a` is 1.1 km end to end and only works at ≤100 m), which
  the automatic baseline retry handles.
* **Never fit a single-amplitude sinusoid.** A real scene spans a ~100× range of
  depths — on Boston, near structure at R≈500 m gives 23° of flow while the
  open-ocean sector at R≈50 km gives 0.19°. A global amplitude is driven entirely
  by the near sector and, since that sector does not span the circle, biases the
  phase by 20°. Use the scale-free sign identity `sign(Δθ − c) = sign(sin(θ − β))`.
* **A zero-flow sector is not the vehicle occluding itself.** That was a tempting
  read of Boston's profile (76% of features, exactly zero median flow) and it is
  wrong — those bins' *appearance* changes as much as the rest of the scene, and
  masking them moved the answer 100° further off. It is the range effect above.
* **Aggregate onto an axis, not a direction.** A vehicle that reverses genuinely
  inverts its travel direction while the mount does not move, so those pairs land
  at β+180 and are correct. Fold mod 180, take the median, then pick direction by
  majority and report the reversal share separately. A raw circular median is
  unstable on antipodally-bimodal samples — it produced a nonsense 160°/MAD 8° on
  `fukuoka_yumechan_a`, which is really 167° at MAD 1°.

Read two gates, not one: `axis_mad_deg` answers "is the mounting fixed?", and
`reversal_fraction` answers "is the direction resolved?" — near 50% is a coin
flip, so the offset may be 180° out even when the axis is excellent. Current
state: **5 usable, 2 axis-only, 14 uncalibrated** of 21.

**Some captures have genuinely noisy positions, and `dist_m` hides it.** Four
datasets have >10% of consecutive steps above 15 m/s against a boat-speed median:
`harima_b_pano` (27.9%), `fukuoka_yumechan_a` (26.5%), `harima_a` (17.6%),
`miura_sagami` (10.7%). Because `dist_m` is a cumulative sum, every position
outlier is *added* to the track length — `harima_b_pano` reports 93.1 km where its
median speed over the same 92 minutes implies ~37 km. The audit warns with the
implied figure. This is Mapillary's SfM `computed` geometry for every frame, not a
raw/computed mix, so there is no better source to switch to; treat these four as
noisy ground truth rather than trusting `dist_m` or per-frame position. Distinguish
this from the benign case: frames a fraction of a second apart show huge implied
speeds from metre-scale jitter, and the audit says so inline instead of warning.

**Do not derive heading from GPS course on vessel tracks.** On
`folkestone_dover`, heading matches course to 0.33° median, but 8.4% of frames
disagree by ~180° — the ferry is backing out of its berth, tracking north on a
steady south-east heading. `heading.py:heading_model_from_positions` is wrong
exactly during the manoeuvring segments, which are the ones nearest the harbour
landmarks.

**Two links can be the same trip.** Registry entries carry `duplicate_of` where
that has happened — `baltimore_b` (adjacent seed, same 668-image capture) and
`anglesey_menai` (same jg360 Channel crossing). The selectors skip them. Two seeds
resolving to identical image counts and extents is the tell.

**A seed link is a fragment, not a trip.** Mapillary splits captures at 500 or
1000 images. Stitching turns 10,470 seed images into 69,598 across the registry;
Folkestone alone goes 500 → 10,711 (the whole 33.8 km Channel crossing).

**Seam distance cannot be a fixed threshold.** Some captures have GPS quantized
in ~200 m steps, so genuinely consecutive sequences begin 200 m apart with a 0.3 s
gap. A fixed 100 m rejected every seam of an obviously continuous run. The
allowance is `1.5*(mean_speed * time_gap) + 1.5*(one GPS step)`. Judging by
implied speed does not work — 200 m over 0.3 s is 690 m/s.

**Discovery must stay endpoint-local.** The `/images` endpoint rejects a bbox
over 0.010 sq deg *and* separately rejects dense areas on result volume, both as
HTTP 500. Sweeping a trajectory's whole area subdivides exponentially (depth 10 in
SF, Seattle, London) and never finishes.

**`--min_spacing_m` matters more than it looks.** These are video extractions at
1–30 fps and many frames share one GPS fix, so a frame's recorded position often
belongs to its neighbour. The 5 m default takes Folkestone from 10,711 frames to
399 without losing a distinct position.

**Mapillary has no 4096 thumbnail** — 2048 (0.25 MB) or the original (3.75 MB for
a 7680-wide pano) and nothing between. A 4096 cap means fetching originals, so use
`--prune_raw`.

**The landmark buffer must be large on water.** 8 km around the mid-Channel track
gave 16 landmarks; 25 km gave 1,708; and 45 km was needed before the English coast
appeared at all (Dover is 31 km west of that track's western end). Override per
trajectory with `landmark_buffer_km`.

**Prefer consistent OSM snapshot dates** over minimising file size. National
extracts are cheap now (see below), and mixing vintages is a real hazard:
`france-250101` against `nord-pas-de-calais-260812` differs by ~44k features on
the same bbox, purely from 19 months of mapping.

## Memory, and why big extracts are fine now

`extract_landmarks_historical` used to be unable to handle a country-sized PBF.
Two independent causes, both fixed:

* **The tag table.** Landmark feathers stored one column per OSM tag key —
  803,717 rows × 1,716 columns, 0.34% non-null. Tags are now a single JSON
  `tags` column (`swag/data/landmark_schema.py`), which also reads the legacy
  wide layout. That took one extraction from 34.6 GB / 8:39 to 2.9 GB / 0:31.
* **The node index.** libosmium's `FlexMem` held every node in the file.
  `--node_margin_deg` (0.1 by default in the orchestrator) bounds it to
  bbox + margin, taking whole France from 28 GB and climbing to 3.0 GB / 1:37.

The OSM stage also runs inside `systemd-run --scope -p MemoryMax=24G -p
MemorySwapMax=0`, so a surprise dies in its own cgroup instead of taking the
machine down — which it did once, before these fixes.

**Always read landmark feathers through `landmark_schema`** (`tag_dicts`,
`row_dicts`), never by touching tag columns, or you will break on one of the two
layouts.

## Checking your work

```bash
# contract audit: naming, ordering, table agreement, image integrity, staleness
bazel run //experimental/overhead_matching/swag/scripts:audit_dataset -- <dataset> [...]

# landmark coverage figure + quantitative gap checks
bazel run //experimental/overhead_matching/swag/scripts:plot_landmarks -- <dataset>
```

The audit catches the failures that are otherwise silent: raw numeric Mapillary
ids (ingest joins on `int(pano_id[1:])`), absolute `panorama/` symlinks, dot-files
that `vigor_dataset.iterdir()` would ingest as phantom panoramas, and pinhole
faces older than the panoramas they came from — a real hazard, since re-rendering
a panorama does not change its filename, so a name-only check passes on stale
faces.

Watch the plot's per-source line. If one source contributes a few percent of
features with most of them on the bbox rim, the buffer is clipping that landmass
rather than covering it; that is exactly how the missing English coast was found.

Two audit warnings are expected on video-derived tracks: high implied speeds
between frames a fraction of a second apart, which is GPS jitter over a tiny time
base rather than motion. The audit prints the cause inline.

## Triage after collection: what the audit cannot see

A dataset can pass every contract check and still be useless, because the audit
compares the dataset's files against each other and they do agree. What it never
sees is whether the camera was bolted to the vessel — and that, not file
consistency, is what decides usability. Seven of the first twenty-one
trajectories were rejected on review, and all seven passed the audit.

Watch `gps_timelapse.mp4` first. Nothing else finds a camera panning across the
deck, a recording that restarts pointing the other way, or a stretch of open
water with nothing to localise against. Then run the three tools that turn those
impressions into numbers.

```bash
# is there a single mount offset, and is its direction resolved?
bazel run //experimental/overhead_matching/swag/scripts:calibrate_mount_offset -- \
    --dataset_path <dataset> --baseline_m 200 --mask_mode both --write_metadata

# is the camera bolted down, drifting, or is nothing vehicle-fixed in frame?
bazel run //experimental/overhead_matching/swag/scripts:detect_vehicle_anchor -- \
    --dataset_path <dataset> --write_overlay

# which steps span a recording restart and must not carry a measured dyaw?
bazel run //experimental/overhead_matching/swag/scripts:annotate_recording_seams -- \
    --dataset_path <dataset>
```

**Use `--mask_mode both`, and read the difference.** Masking removes every
correspondence that has zero parallax because it is bolted to the camera. An
estimate driven by real scene motion barely notices; one that was really fitting
the boat collapses. That difference is a per-dataset accuracy test needing no
map, no tracklets and no external reference — which matters, because
`boston_harbor_leg1` is the only leg that has one. It is recorded as
`survives_vehicle_mask`.

It is a sharp test:

| dataset | unmasked | masked | reading |
|---|---|---|---|
| dataset | mask | unmasked | masked | axis shift | reading |
|---|---|---|---|---|---|
| `seattle` | 5.7% | MAD 22.7°, 59 pairs | MAD **4.0°**, 59 pairs | 6.7° | **survives** — same angle, scatter cleaned up |
| `kumamoto_yumechan_b` | 22.2% | MAD 35.0°, 37% aligned | MAD **8.0°**, 90% aligned | 1.0° | survives; support 35% |
| `mississippi_rural` | 8.6% | MAD **1.5°**, 100% aligned | MAD 10.5°, 63% aligned | 6.7° | survives; the *tight MAD* was the boat |
| `folkestone_dover` | 21.5% | MAD **0.0°**, 59 pairs | MAD 23.5°, 8 pairs | 6.0° | survives; support 14% |
| `fukuoka_yumechan_a` | 20.5% | MAD **0.0°**, 60 pairs | **0 pairs at any baseline** | — | **fails** — nothing left once the boat is gone |

The takeaway is blunter than "masking helps": **the three tightest MADs in the
whole collection — 0.0°, 0.0° and 1.5° — all belong to datasets with an 8–22%
vehicle anchor, and all three loosen or vanish once it is removed.** Zero-parallax
structure does not just bias a fit, it makes it look *precise*. Never rank
datasets by MAD.

Angle and support fail independently, hence two fields, `mask_axis_shift_deg` and
`mask_pair_retention`. `fukuoka_yumechan_a` has no masked answer at all, so its
0.0° was purely the boat. `folkestone_dover` and `mississippi_rural` keep their
axis to under 7°, so those angles are corroborated — but on 14% and 27% of the
pairs, so the confidence is not. Collapsing that into one boolean would misreport
whichever half the reader cared about.

**A mask that never fires is not a pass.** Seven of the fourteen have no vehicle
structure in frame, so no mask is built and both runs are the same fit.
`survives_vehicle_mask` is `null` there, not `true` — reporting a gate that passed
because it never ran is the exact failure this test exists to catch.

For perspective captures the mechanism is `findEssentialMat` going degenerate:
static structure is consistent with pure rotation, so with enough of it in the
consensus set the reported translation direction is whatever the remaining world
points can drag it to. For equirectangular captures the failure is different —
the horizon band is already the only region used, so masking a vessel that
occupies 20% of the frame can leave too few informative points and the estimator
correctly refuses rather than guessing.

**Read the anchor detector against the offset, not alone.** They answer different
questions and the interesting cases are where they disagree:

| anchor | axis MAD | reading |
|---|---|---|
| `rigid` | low | calibrated; use the offset |
| `rigid` | high | fixed camera the estimator could not solve — a method gap, worth retrying with a mask or a different baseline, *not* a reason to reject |
| `drifting` | high | camera genuinely moved relative to the vessel; no single offset exists, and per-frame alignment against the anchor is the only route |
| `no_anchor` | any | nothing vehicle-fixed is in frame, so the detector has no opinion; the axis MAD is the only evidence |

`no_anchor` is a statement about the imagery, not the mount. A rigidly bolted
camera looking out over open water lands there, and so does a handheld one.

**Two gates, not one.** `axis_mad_deg` asks whether a single offset exists;
`reversal_fraction` asks whether its *direction* is resolved. Near 50% that is a
coin flip even when the axis is perfect — `fukuoka_yumechan_a` has an axis MAD of
0.0° and 47% reversals, so its offset is known only to ±180°. Reversals are also
genuine data: `folkestone_dover` backs off its berth before steaming out, and
during that the travel direction really does invert while the mount does not.

**Seams are not automatically faults.** `annotate_recording_seams` writes every
break in continuity, then leaves the judgement to the consumer via `step_m` and
`implied_speed_mps`. A ferry idle at its berth for three minutes between
recordings has a large `dt_s` and a five-metre `step_m`: pose is continuous and
nothing needs doing. The dangerous seam is the one where the vessel *manoeuvred*
unobserved — `nyc_east_river` turned around during a 235 s gap, which is visible
in the video and invisible in the positions.

## Trimming

Visual review usually condemns part of a trajectory rather than all of it.

```bash
bazel run //experimental/overhead_matching/swag/scripts:trim_dataset -- \
    --dataset_path <dataset> --keep 0:165 --video_fps 15 \
    --reason "..." --dry_run
```

Ranges are original frame indices; `--video_fps` converts a timestamp read off
the timelapse (`frame = seconds x fps`, and check the fps — a track over 1500
frames is subsampled, so `mississippi_rural` runs at two frames per video frame).

Do not hand-edit the CSVs. The audit requires `frames_gps.idx` to be 0..N-1
contiguous *and* `pano_id[1:] == idx`, because that equality is the ingest join
key; cutting from the middle therefore forces a renumber, the renumber forces an
image rename, and the rename has to reach all four tables together. The script
moves dropped images and CSV backups to `trimmed_frames/` inside the dataset, so
a trim costs no extra disk and is reversible.

Two things go stale on every trim and the script says so: the `mount_offset`
block (it is flagged `stale_after_trim` and forced unusable until re-measured),
and, for equirectangular datasets, the pinhole faces, which reference the old
pano_ids and must be regenerated with `panorama_to_pinhole`.

Trims often confirm themselves. `kumamoto_yumechan_b` was cut on the visual
judgement that its first 19 s had no landmarks; that same span turned out to
carry 50.5% of steps above 25 m/s against 5.3% after, so the visual call and the
GPS-quality boundary landed in the same place independently.

## Related

* the collection scripts themselves — now bazel targets in
  `//experimental/overhead_matching/swag/mapillary_tools`, documented in this file
* [`azimuth-convention.md`](azimuth-convention.md) — how the convention was measured
* [`mapillary-perspective-support.md`](mapillary-perspective-support.md) — consuming the non-360 datasets
* `docs/object-tracking-runbook.md` — the M0–M6 tracking pipeline downstream
