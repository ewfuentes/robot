# CrossLocate-Depth: repo grounding and overnight scope

Companion to `dem_baseline_plan.md`. That document is the design; this one maps
it onto this repository and records what is being executed overnight
(2026-08-27). Update statuses here as work lands.

## 1. Where the code goes

New package: `experimental/overhead_matching/swag/farfield/dem_baseline/`.
Follows farfield BUILD style (`_lib` py_library + thin py_binary, visibility
`//experimental/overhead_matching/swag:__subpackages__`), farfield conventions
(`farfield/geometry.py` owns all angle math — camera azimuth CW-positive from
pano centre column; `docs/conventions.md` is the authority), and the farfield
data lanes (`farfield/paths.py`).

Planned modules (plan section → module):

| Plan | Module | Notes |
|---|---|---|
| §5.1 lattice + db | `lattice.py`, `render_db.py` | candidate grid in RegionFrame ENU; descriptor db as npz/torch tensors + JSON manifest |
| §5.2 rendering | `terrain.py`, `depth_render.py` | rasterio/pyproj mosaic → height field; torch GPU ray-march renderer emitting metric float32 depth + sky mask; Earth curvature drop |
| §5.3 queries | `query_crops.py` | pano → ring of 60° FOV crops at reference yaw spacing, via the verified `reproject_pinhole` path (fix the per-channel wrap bug; use `geometry.direction_from_face_px` conventions) |
| §5.4 network | `crosslocate_net.py` | PyTorch VGG16-MAC port; loads the archived name-keyed npz dump of the release TF1 checkpoint (one-time conversion ran in an external TF venv; repo code only reads npz) |
| §5.3 scoring | `panorama_score.py` | circular yaw-shift joint (location, heading) score `S_t(i,k)`; mean over valid crops first |
| §5.5 likelihood | `likelihood.py` | softmax-with-temperature + uniform outlier floor; calibration params live in build config, frozen on validation |
| §5.6 filter | new `measurement_backend`-style seam (see §3 below) | separate PR-sized step after framewise retrieval works |
| eval | `framewise_eval.py` | recall@k/radius vs `frames_gps.csv` truth; reuses `evaluation/retrieval_metrics.py:compute_top_k_metrics` |

Existing pieces reused, not rewritten:

- `farfield/geometry.py` — angles, ENU (`RegionFrame`, `enu_from_latlon`).
- `farfield/extraction/panorama_to_pinhole.py` — `reproject_pinhole` (note:
  column axis reversed ⇒ CCW azimuth; per-channel `mode='wrap' if c==1` is a
  latent bug — dem_baseline uses a corrected copy with all-channel wrap).
- `swag/model/patch_embedding.py` — VGG16 tower template (`load_backbone`,
  interface contract: `model_input_from_batch`, `forward -> (emb, debug)`).
- `swag/scripts/losses.py` (InfoNCE/pairwise) + `distances.py` — only if/when
  CLD-5 disjoint retraining happens; there is **no triplet loss in the repo
  yet**, it would be new code.
- `common/torch/load_and_save_models.py` — checkpoint save/load with commit
  provenance, once we train anything. The zero-shot port stores plain
  state-dict npz under `models/` with SOURCE.md instead.
- `metrics.py` / `run_io.py` / `replay.py` in `farfield/localization` — reused
  as-is for the Bayes variant; none of them are bearing-specific.

## 2. Data lanes (decided)

```
/data/farfield_matching/
  raw_material/crosslocate_release/     # released code clones, meta .npy, sparse tarballs
  models/crosslocate/<checkpoint>/      # released TF1 checkpoints + exact train config + log
  raw_material/usgs_3dep_mount_washington/   # 9x 1/3" tiles (100 km buffer) + 1 m core tiles + TNM records
  artifacts/dem_surfaces/<region>/vN/        # derived mosaics (new artifact kind, manifest.json per farfield rules)
  artifacts/depth_render_db/<region>/vN/     # rendered depth descriptor databases
```

New artifact kinds (`dem_surfaces`, `depth_render_db`) get added to
`paths.ARTIFACT_KINDS` when the derivation code lands — they are keyed by
**region**, not dataset, since one Mt. Washington surface serves all three legs
(this is a deliberate deviation from the per-dataset artifact layout; record it
in the manifest).

### Release findings (CLD-0, verified 2026-08-27)

- Full CrossLocate release is publicly hosted at
  `https://cphoto.fit.vutbr.cz/crosslocate/dataset/`: trained checkpoints
  (`models/AlpsPhotosToDepthCompact_31_2/` = the paper's Alps photos→depth
  model, TF1, epoch 39, 168 MB, **with its exact training config .py and
  output.log**), sparse + uniform datasets, and meta structures (.npy poses/
  splits/positives).
- The GitHub repo (`JanTomesek/CrossLocate`) is code-only, TF 1.14 / Python
  3.6, 22 files; `models/vgg16_weights.npz` is a dead LFS pointer. **No LICENSE
  file** → per plan §7.6: internal use fine, no redistribution of code or a
  weight-bearing port without permission; plan for a clean-room description in
  the paper.
- The release also ships `query_*_result.tar.gz` — released retrieval results.
  **Port verification can compare rankings against these fixed released
  results without ever running TF1**; a TF env is only needed to dump
  checkpoint weights to npz (read-only, `tf.train.load_checkpoint` in a scratch
  venv works on TF1 checkpoints).

## 3. Filter integration seam (CLD-3, grounded)

- The likelihood seam is real but implicit: `run_filter` binds
  `apply_measurement(meas)` / `score_fn(east, north, heading, meas)` closures
  per `FilterConfig.measurement_backend`
  (`farfield/localization/filter.py:1192-1235`). A retrieval likelihood is a
  third binding; motion model, mixture-tracking resample, mode tracker,
  HealthRecord, and the 500 m time-normalized mass metric are all
  observation-agnostic and reused untouched.
- A retrieval observation does **not** fit `TrackletMeasurement` /
  `CompatibilityTable`; it needs a new per-keyframe record (top-k location/yaw
  scores + calibration) and a producer artifact parallel to
  `localization_inputs` (`tier1_*.jsonl`, `message_schema_version 0.7` —
  schema bump required).
- Known bearing-specific couplings to generalize when we get there:
  `filter._validate` (tables required), `runner.execute_localization`'s
  unconditional `bearing_residual_diagnostics` (`runner.py:147`),
  `run_io.validate_manifest`'s `bearings_consumed`/`no_bearings` rule
  (`run_io.py:246`), and `proposal.propose` (bearing resection — first
  retrieval runs go `proposal.enabled=false`, auto-tagged).
- Run classification already enforces plan §8.5: `run_kind == "evaluation"`
  iff uniform init + no ablation tags (`run_export.py:215`); heading is always
  uniform (`filter.py:254`). CrossLocate-Depth + Bayes runs inherit this.

## 4. Dependency facts

Available in `third_party/python/requirements_3_12.txt`: torch 2.7 cu128,
torchvision 0.22, rasterio 1.4.3 (bundled GDAL), pyproj 3.7.2, opencv, scipy,
pillow, lmdb, msgspec. **Not available:** faiss (not needed — 512-D
descriptors × ~1M views fits torch matmul topk on the 5090), laspy/pdal
(needed later for the MassGIS DSM recipe, CLD-4), trimesh/moderngl (avoided by
the height-field ray-march renderer), tensorflow (deliberately kept out;
checkpoint dump runs in a throwaway uv venv).

Database scale check: 100 m grid over a 20×20 km Mt. Washington region =
40k locations × 12 yaws = 480k views. Rendered 500×500 float16 depth ≈ 240 GB
— **do not store dense renders for the full lattice**; store descriptors
(480k × 512 × fp16 ≈ 0.5 GB) plus a small archived render sample for
regression tests. Renders are cheap to regenerate on-GPU from the surface +
manifest.

## 5. Overnight scope — status at 2026-08-27 ~01:30

Data:

- [x] CrossLocate release → `raw_material/crosslocate_release/` +
      `models/crosslocate/AlpsPhotosToDepthCompact_31_2/` (TF1 checkpoint +
      exact train config + full training log; sparse meta structures; uniform
      meta including the checkpoint's own Alps_photos_to_depth_compact
      train/val/test .npy). Sparse tarballs (depth db 6 GB, RGB queries 6 GB,
      depth queries 2 GB) + code clones still downloading in background.
- [x] Depth encoding audited from released EXRs →
      `raw_material/crosslocate_release/DEPTH_ENCODING.md`. Headline: raw
      metric depth, **sky = -1.0**, no clipping transform, no input
      preprocessing anywhere (raw 0-255 RGB / raw meters), input 500×500
      (NOT resized to 224 — plan §5.4 corrected), useFOV=False.
- [x] TF1 checkpoint → `converted_weights.npz` (+ manifest w/ sha256) via
      throwaway uv+tensorflow-cpu env; variables are `conv{i}/kernels|biases`.
- [x] USGS 3DEP Mt. Washington → `raw_material/usgs_3dep_mount_washington/`
      (9 × 1/3" tiles 2.9 GB + 21 × 1 m tiles 4.9 GB + TNM product records).
- [x] Mt. Washington surface v1: 80×80 km @ 10 m UTM 19N mosaic →
      `artifacts/dem_surfaces/mount_washington/v1/` (elev 107–1915 m, zero
      no-data).

Code (`swag/farfield/dem_baseline/`, all tests green):

- [x] `terrain.py` + `build_surface` — mosaic/reproject, HeightField npz+json,
      no-data mask, manifest with source checksums.
- [x] `depth_render.py` — **column-scan** height-field renderer (exact for
      pitch=0: one 1-D march per image column + running-max searchsorted →
      ~500× cheaper than per-pixel marching; 0.09 s per 12-view ring on the
      5090). Metric slant depth + inf sky + per-view source-data coverage;
      Earth curvature with k=0.13 refraction. Analytic tests: flat plane
      h/sin(θ), wall depth/heading, curvature-hides-ground.
- [x] `crosslocate_net.py` — VGG16-MAC port (conv13 no ReLU, pre/post L2,
      ceil_mode pools ≙ TF SAME), raw-input contract, weight loader.
- [x] `query_crops.py` — CW-convention pano→60° crop ring + yaw round-trip
      tests (the extraction/panorama_to_pinhole CCW/wrap quirks not inherited).
- [x] `lattice.py`, `render_db.py` — lattice + descriptor DB builder
      (descriptors-only storage + sample renders; 26 loc/s → 2500-loc DB in
      95 s), `qa_render` surveyed-pose QA CLI.
- [x] `panorama_score.py` — circular joint (location, heading-shift) score;
      shift k ⇒ heading k·30°.
- [x] `framewise_eval.py` — recall@k/radius vs frames_gps truth.
- [x] `port_verification.py` — **PORT VERIFIED (CLD-0 exit criterion met)**:
      on the released 516 Swiss GeoPose3K val queries vs 6192 db depth views,
      the port scores recall@{1,10,100}<400 m = **62.98 / 86.05 / 96.12** vs
      the released log's epoch-39 **62.60 / 85.27 / 93.99** (small positive
      deltas consistent with the released metric also requiring orientation).
      Descriptor level: torch and the actual TF1 graph (run under TF2 compat)
      agree **bit-exactly on CPU** (conv1/conv13 max|diff| = 0.0, descriptor
      1.5e-8); GPU TF32 adds ≤7e-5 on unit descriptors — negligible.
      Report at `models/crosslocate/.../port_verification.json`.

Two gotchas found on the way (recorded so nobody re-hits them):

1. **cv2 reads single-channel EXRs as BGR**: the depth lands in channel 2 and
   channels 0/1 are zeros. Reading `[:, :, 0]` feeds all-zero images and every
   descriptor collapses to one constant vector (below-chance recall was the
   tell). `port_verification.py` documents and handles this.
2. The swiss val dataset's queries dir is `query_original_result/` (processed
   photos), not raw `query_original/` — the DatasetDescriptions entry is the
   authority for which released dirs pair with which meta.

### First diagnostic numbers (NOT protocol-frozen)

Zero-shot Alps checkpoint, provisional 10×10 km declared region around the
summit (UTM 19N 311183–321183 E, 4899497–4909497 N), 100 m lattice (10,201
locations), 30 km render range, `runs/dem_baseline_dev/*_framewise_dev100m`:

| leg | frames | r@1<250m | r@1<500m | r@10<100m | r@10<1000m | top-1 median err |
|---|---|---|---|---|---|---|
| leg1 (ravine→hut) | 134 | 0.06 | 0.13 | 0.04 | 0.71 | — |
| leg2 (hut→ridge) | 265 | **0.40** | **0.63** | **0.68** | 0.89 | — |
| leg3 (summit area) | 398 | 0.00 | 0.01 | 0.00 | 0.20 | 2.7 km |

Chance for r@1<250 m ≈ 0.2%, so leg2's 40% is a real, large zero-shot
signal on open above-treeline terrain; leg1 (partial canopy) is weak; leg3
(summit cone) is near chance — worth qualitative retrieval panels to see
whether buildings/people/near-field rock dominate its crops. This ordering is
exactly the applicability stratification story of plan §8.4.

Deferred (needs decisions or daytime bandwidth): likelihood calibration +
filter integration (CLD-3), disjoint retraining (CLD-5), HORAYZON (separate
task by design).

## 5b. CLD-4 status — completed 2026-08-27, updated 2026-08-31 (baseline-data worktree)

Decisions taken with ekf: candidate regions = the proposed method's uniform
prior (catalog extent + margin_m 1000 from the current
`localization_inputs/*/stage3_17c8031_regen_v8_machine` exports); region
artifacts kept separate (boston_harbor / charles_river / pohang_canal);
Pohang starts coarse (GLO-30) while NGII access is attempted; water = declared
nominal level per region.

- [x] Raw data: USGS 3DEP `MA_CentralEastern_2021_B21` — 495 QL1 LAZ (79 GB)
      + 18 provider 1 m DEM tiles + all 495 per-tile XMLs + TNM records +
      sha256 manifest → `raw_material/usgs_3dep_ma_centraleastern_2021/`.
      NAD83(2011)/UTM19N (EPSG:6348) native, NAVD88 GEOID18, acquired
      2021-03/04. Copernicus GLO-30 4 tiles (hashed, attribution notice) →
      `raw_material/copernicus_glo30/`.
- [x] Code: `lidar_dsm.py` + `build_dsm` (streaming class-filtered per-cell
      max, block-swept >=5-of-8 median hole fill, provider-DEM fallback =
      hydro-flattened water rule, per-cell provenance raster, disjoint-tile
      bbox skip); `build_surface` grew `--surface_kind`/`--note`. laspy 2.6.1
      + lazrs 0.8.2 pinned. All tests green.
- [x] Statistic frozen on validation tile 19TCG313672 (max vs p98 median
      1.4 mm, 146/2.0M cells >1 m → max_per_cell);
      `raw_material/.../statistic_freeze.json`.
- [x] Surfaces: `dem_surfaces/boston_harbor/{v1_dem,v1_dsm}`,
      `dem_surfaces/charles_river/{v1_dem,v1_dsm}` (1 m, EPSG:6348, region
      box integer-aligned), `dem_surfaces/pohang_canal/v1` (GLO-30 30 m DSM,
      region + 30 km buffer). Per-region AUDIT.md covers datum chain, 5.3 y
      (MA) / 6-10 y (Pohang) map-age gaps, provenance histograms, water
      levels (harbor -1.63 m NAVD88 ≈ MLLW, tide ±~3 m sensitivity; basin
      0.00 m; Pohang ~MSL).
- [x] DEM-vs-DSM QA at truth poses: harbor downtown skyline 1.92° (DSM,
      geodetic 1.95°) vs 0.20° (DEM); Prudential from basin 13.89° (DSM,
      expected 13.98°) vs 0.31° (DEM). The plan §7.1 prediction (bare earth
      deletes the urban skyline) is confirmed quantitatively.
- [x] Render buffer, DECIDED with ekf 2026-08-31: **Mt. Washington v2 and
      Pohang v1 carry +30 km beyond the candidate box; the two MA surfaces
      carry none, and that is accepted.** The justification is the curved
      horizon, not the track geometry: from a 3 m eye the sea-level horizon is
      6.6 km (R_eff = R/(1−0.13)), so terrain at range d beyond the map is
      visible only above ~(d−6.6 km)²/(2 R_eff). Measured track clearance to
      the nearest box edge is ≥7.96 km (harbor legs 1/2/3: 8.24/7.96/7.99 km)
      and ≥13.43 km (charles), which puts the visibility threshold at ~128 m
      (harbor) and ~330 m (charles) — and eastern MA has nothing that tall
      within 30 km outside the boxes (Great Blue Hill, 193 m, is INSIDE the
      harbour box; Wachusett, 620 m, is ~50 km out where the threshold is
      ~1.4 km). So no truth-pose query render loses visible geometry.
      The rim-candidate case (a candidate ON the box edge renders false sky
      outward) is NOT covered by that argument, and is what the next item
      fixes.
- [x] **Coarse background surfaces, 2026-08-31 — MA is buffered after all,
      so the item above now only explains why the truncation was survivable
      for query poses.** `TerrainTensor`
      gained an optional `background`: a coarse HeightField consulted wherever
      the fine surface has no source data (outside its box AND in interior
      holes), and `build_lattice` now drops a candidate only when neither
      surface has data. Built `dem_surfaces/{boston_harbor,charles_river}/
      background_glo30_30m` — region box + 30 km at 30 m in EPSG:6348 from 4
      Copernicus GLO-30 tiles (N41/N42 × W071/W072, hashed into
      raw_material/copernicus_glo30), 2945×2911 and 2922×2952 cells, elevation
      −33..193 m / −33..206 m, 0 nodata. Carrying the far field at 1 m was
      never possible: box+30 km is 7.7e9 cells (~31 GB) vs 8.5e6 at 30 m.
      Two things this fixes that the inset did not: (1) all four regions now
      reach 30 km past their candidate box, so the DSM condition is consistent
      across regions; (2) the 16.7% of empty cells in the harbour box —
      including delivery-gap tiles INSIDE the harbour beside the leg3 track —
      were silently deleting candidates (0 of 1,862 truth poses across the
      four datasets landed on one, verified against the provenance rasters,
      but that is luck not design). Declared caveat: GLO-30 is EGM2008 vs the
      1 m foreground's NAVD88/GEOID18, sub-metre, not corrected (<0.06° at
      ≥1 km).
- [x] Per-condition backgrounds, 2026-08-31: the bare-earth v1_dem condition
      uses `background_3dep13_10m` (USGS 3DEP 1/3″, 10 m, NAVD88 — the
      foreground's own datum; 4 tiles n42/n43 × w071/w072, 1.0 GB, hashed to
      raw_material/usgs_3dep_ma_13arcsec) with the GLO-30 background CHAINED
      BEHIND it, because 3DEP has no data over open water: bare earth on
      land, water surface over the sea, and no buildings smuggled into the
      condition whose entire purpose is to lack them.
      `TerrainTensor.chain_from_height_fields` stages a fine→coarse chain and
      `build_lattice` takes the same list. Verified at the 100 m working
      spacing: DSM and
      DEM conditions keep the SAME full candidate set (77,259 boston_harbor,
      78,945 charles_river, 0 dropped) — a per-condition candidate set would
      not be comparable.
- [x] **MA reference databases built 2026-09-01** (~5 h wall, 19 loc/s on the
      5090, 3.49 GB total): `depth_render_db/{boston_harbor,charles_river}/
      {v1_dsm,v1_dem}_100m`, 100 m lattice over the full declared region,
      12 yaws × 60° × 500² per location, curvature k=0.13, 30 km far range,
      sky encoded as −1 to match the release. Observer heights set by ekf per
      platform: **4.0 m boston_harbor** (cruise deck), **1.5 m charles_river**
      (small sailboat) — measured above the map's water plane, so the
      harbour's ±3 m tide moves the true eye height and the geometric horizon
      (7.7→10.1 km); declared sensitivity, not corrected.
      Verified: 77,259 / 78,945 locations, **0 dropped**, source coverage
      **1.0000 on every view** including box corners (the 30 km square buffer
      does cover the 30 km march diagonally), descriptors all finite with
      median L2 norm 1.000, and the DSM/DEM lattices byte-identical per region
      so the ablation is comparable. DSM-vs-DEM descriptor cosine at the same
      location: median 0.648 (boston_harbor), 0.369 (charles_river) — the
      Charles is far more building-dominated, and open-water poses are where
      the two conditions agree.
- Search areas after the background surfaces (both methods search the same declared region; no inset):
      | region | declared region | baseline candidates |
      |---|---|---|
      | boston_harbor | 775 km² | 775 km² (0 dropped, was 12,765 of 77,259 at 100 m) |
      | charles_river | 789 km² | 789 km² (0 dropped, was 3,645 of 78,945) |
      | mount_washington | 6,253 km² | 6,253 km² |
      | pohang_canal | 4,972 km² | 4,972 km² |
- [x] Alignment QA over every keyframe (also closes plan CLD-1's
      "image/render horizon overlay at surveyed poses" deliverable):
      `depth_render.render_cylinder` + `truth_strips` + `export_truth_video`
      render a 360° cylindrical depth strip at each truth pose, roll the photo
      to grid north (pipeline GPS course model, 3 m / 10 s, evaluated in the
      surface's grid frame), overlay the render horizon, and encode one video
      per dataset →
      `runs/260828_dem_baseline_dev/truth_video/<dataset>__{nominal,measured}/`
      (7 videos; charles has no `__measured` because the estimator correctly
      abstained). Washington leg2/leg3 use `mount_washington/v2`; charles uses
      `charles_river/v1_dsm`. Charles's DSM reproduces the Back Bay/downtown
      skyline tower for tower — the strongest visual evidence yet that the
      1 m DSM recipe is right.
- Finding that belongs to the tracking lane, not CLD-4: because the render is a
      full 360° cylinder, heading only rolls the photo, so the photo-vs-render
      azimuth offset is a MEASUREMENT of camera-to-course offset. On the mtw
      legs it says the approved `nominal_forward` 30° is ~28° too large
      (leg2 +28.0°, MAD 8.4° over 258/265 frames, visually confirmed on
      Monroe's cone, and not a course timing lag: intercept +27.9° after
      regressing on course rate). Independent agreement: the 2026-08-19
      arc-gated sweep (4.0°) and the pole geometry (the hiker's back sits at
      the camera centre column ⇒ offset ≈ 0). The sun check does NOT adjudicate
      there — `candidate_concentration` 0.59/0.59/0.74 against its own 0.95
      bar (overcast; 5–19 of 40 frames with a bright blob) — although
      `alignment_diagnostics.json` still publishes its 46/78/80°. On charles
      the sun check IS decisive and confirms the nominal (R 0.972, 270.2°
      vs 270.0°), and the skyline estimator abstains (18/60 frames usable).
- [x] Skyline estimator hardened 2026-08-31 (was: r is profile similarity, not
      azimuth sharpness — leg1 scored r≈0.93 on a broad featureless profile).
      `estimate_shift` now high-passes both profiles (circular moving average,
      45° window) before correlating and returns peak, PROMINENCE (best minus
      the best rival outside the peak's own width) and FWHM, with σ = FWHM/2.355
      for inverse-variance weighting; the gates are r ≥ 0.35, prominence ≥ 0.05,
      FWHM ≤ 60°. **The high-pass was the load-bearing part**: undetrended, the
      correlation peak was ~110° wide on every real frame — strong and weak
      alike — because one slow sky-vs-land lobe dominates the curve, so width
      could not gate anything. Detrended, median FWHM separates the legs
      (leg2 57°, leg1 82°) and leg1 now correctly ABSTAINS (6/60 calibration
      frames pass), while leg2 keeps a +25° median (MAD 9.5) and leg3 +31°
      (MAD 21) — i.e. the ~28° discrepancy against the approved 30° survives
      the harder test on exactly the legs that had the evidence.
- PGC EarthDEM (AWS Open Data, 2 m Maxar-stereo DSM, CC-BY-4.0) checked
      2026-08-31 and RULED OUT: only a Great Lakes/St. Lawrence subset is
      public (collection bbox [-93.46, 39.76, -73.11, 52.02]; strip cells
      n40-n51 x w074-w094; `earthdem/mosaics/` empty), so it covers neither
      Korea nor Massachusetts nor Mt. Washington. Worth re-checking if PGC
      publishes more regions — the license is exactly what a public release
      needs.
- Pohang v2 (NGII 5 m + building extrusion) still blocked on portal access:
      `map.ngii.go.kr` (국토정보플랫폼) serves the nationwide 5 m DEM plus the
      수치지형도 building polygons, needs member registration and a large-file
      transfer helper; `data.go.kr` dataset 15059920 mirrors the DEM download;
      VWorld is API-only (XDO binaries) and reportedly restricted overseas.
      Fallbacks if registration blocks: a Korean-account collaborator, JAXA
      ALOS AW3D30 (30 m, free registration), or extruding OSM building
      footprints — which would make Pohang's DSM condition depend on the same
      OSM layer the proposed method consumes, so it must be disclosed.
- OOM lesson: full-grid morphology on 775M-cell grids must be block-swept.

## 6. Open decisions surfaced by grounding (beyond plan §12)

1. Region-keyed artifacts (`dem_surfaces/<region>/`) vs forcing per-dataset
   layout — going region-keyed; confirm.
2. The three Mt. Washington legs are tiny bboxes (~0.4–1.2 km trajectories
   near the summit cone). The declared candidate region for the whole-map
   claim needs choosing (e.g., Presidential Range box) — an experiment-design
   decision, not code; overnight eval uses an explicit provisional box
   recorded as such.
3. Boston pano datasets have `pipeline_metadata.json` bboxes tight to the
   route; the shared candidate-region polygon for cross-method fairness
   (plan §2) must come from the LOCI/proposed-method configs — align later.
4. Query cadence for CrossLocate-Depth + Bayes should match the keyframe
   schedule in `localization_inputs` exports (3 m distance grid) — default
   position, revisit at calibration time.
