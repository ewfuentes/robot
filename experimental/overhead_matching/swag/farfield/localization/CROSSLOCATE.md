# CrossLocate in the causal grid filter

Use `grid_filter --observation_source crosslocate` or the same arguments as a
`run_grid_batch` job. This reuses LOCI's immediate-per-panorama loop, distance
episode selection, odometry profiles, and `farfield_causal_grid/v1` result
schema. It does not run rendering, retrieval scoring, or delayed track factors.

## Inputs and local storage

- The same completed `localization_inputs` and episode plan used for the paired
  bearing/LOCI evaluation. The full parent catalog remains the prior for every
  episode; it is never cropped to the truth trajectory.
- A schema-0.7 directory containing `retrieval_meta.json` and
  `retrieval_fields.npz` (`lat_deg`, `lon_deg`, `scores`, `keyframe_idx`, `pano_ids`).
- The parent dataset's canonical `frames_gps.csv`. IDs, indices, frame count,
  and positions are checked against the score artifact and localization input.
  Positions are used only to validate identity, not to select score hypotheses.
- An explicit projected render CRS and a separate local cache directory.

Returned artifacts were copied to
`/data/farfield_matching/artifacts/retrieval_observations/<dataset>/<variant>/`.
The six datasets are `boston_harbor_leg1`, `boston_harbor_leg2`,
`boston_harbor_leg3`, `charles_river_20260727`, `franconia_leg1`, and
`pohang_canal_04`; variants are `top6` and `all12`. Harbor leg2's `all12` archive
was invalid in the returned data; use `top6` or obtain a replacement.

The raw files are read-only. The adapter extracts `scores.npy` once into the
specified cache, keyed by the raw archive's SHA-256, and memory-maps it on
subsequent runs. Extraction checks the ZIP CRC and publishes the file atomically.
Only one frame's scores and mapped likelihood are processed at a time. Allow
about 9.6 GB of cache space per Pohang variant. Caches are disposable; do not
edit their contents. Keep both cache and result paths on local drives.

## Coordinates and likelihood

Positions are WGS84 lat/lon, transformed into the filter's region ENU frame.
The render lattice is regular in its projected CRS, not in ENU. Supply the CRS
from the render database manifest when available. The handoff specified:

| Dataset | Render CRS |
| --- | --- |
| Harbor and Charles | EPSG:6348 |
| Franconia | EPSG:26919 |
| Pohang | EPSG:32652 |

These are operator-supplied declarations, not recovered manifest provenance.
The adapter checks that the coordinates lie on the declared metric lattice;
this is a consistency check, not proof of the original datum. Returned metadata
binds the database manifest by hash, but does not include that manifest.

The renderer's yaw is clockwise from projected grid north. Lookup subtracts
both the export's approved nominal-forward camera offset and the node's
meridian convergence from the filter's true-north nominal heading. Position
lookup uses the nearest node within 0.75 lattice spacings; heading lookup
interpolates scores circularly, including across the final/first yaw bins.

`--margin_m 0` is required. The existing filter grid and uniform prior are
preserved exactly, including rounded-up boundary cells whose centers fall
outside the literal catalog bbox. Coverage is determined only by nearest-node
distance on that grid, not by an extra bbox mask. The 30 km terrain halo affects
rendering only and is not a localization prior.

For each frame, mapped scores are temperature-softmaxed over supported filter
states and mixed with a uniform outlier floor over those supported states.
The mixture is rescaled to mean one on supported states; unsupported cells
receive a unit factor (log likelihood zero), neutral relative to that mean.
Missing coverage never removes a cell or assigns it outlier-only evidence.
Flat scores therefore leave the prior unchanged, including boundary cells.
Zero overlap between the retrieval lattice and grid still raises an error.
This normalization is on the filter grid, **not** the original retrieval lattice; calibration therefore
needs validation for this adapter. Defaults (`temperature=0.1`, `epsilon=0.05`)
are provisional and recorded as `calibration_frozen: false` in every result.
Raw nonfinite scores cause an error rather than silently changing support.

## Run

Example, substituting the paired evaluation's input and episode-plan paths:

```bash
bazel run //experimental/overhead_matching/swag/farfield/localization:grid_filter -- \
  --input_dir "$INPUT_DIR" \
  --episode_plan "$EPISODE_PLAN" --episode_index 0 \
  --observation_source crosslocate --availability immediate \
  --retrieval_dir /data/farfield_matching/artifacts/retrieval_observations/boston_harbor_leg1/top6 \
  --retrieval_frames_csv /data/farfield_matching/datasets/boston_harbor_leg1/frames_gps.csv \
  --retrieval_cache_dir /data/farfield_matching/artifacts/retrieval_grid_cache \
  --retrieval_render_crs EPSG:6348 \
  --retrieval_temperature 0.1 --retrieval_outlier_epsilon 0.05 \
  --odometry_profile epson_mg570_calibrated_planar_v1 --odometry_seed 0 \
  --cell_m 100 --n_heading 36 --margin_m 0 \
  --yaw_sigma_scale 1 --heading_rw_deg 1 --diffusion_m 5 --top_modes 0 \
  --device cuda --out "$LOCAL_RESULT_JSON"
```

The output directory must already exist. Use distinct result paths per
dataset, score variant, episode, and calibration setting. Batch jobs enforce
the same explicit paired settings as LOCI and reject existing output paths.
For a quick smoke test add `--output_end 1`; this limits output to the first two
episode frames without changing the episode's initialization or parent mapping.

Results contain the usual per-keyframe masses, map errors, online poses, episode
identity, and runtime. The `crosslocate` section records input digests, scorer,
CRS declaration, heading conversion, interpolation, and calibration settings.
No retrieval-scoring or DSM-rendering data needs to be regenerated to run this.
