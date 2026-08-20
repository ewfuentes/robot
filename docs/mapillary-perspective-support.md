<!-- Moved into the repo 2026-08-19 from ~/scratch/mappilary/PERSPECTIVE_SUPPORT.md.
Frame conventions referenced here are registered in docs/conventions.md. -->

# What must change to consume these datasets

## 0. The panoramas are not north-aligned either

Images are stored **unrotated** for both projections, so the 8 equirectangular
datasets are in the camera frame, exactly like `boston_harbor_dataset`. Anything
that assumes a north-aligned panorama (column 0 = north, or centre = north) is
wrong for them.

Two ways to get world azimuth from an equirect frame:

* **Use the recorded heading** — `intrinsics.csv:heading_deg` is the bearing of
  **column 0**, so `azimuth = (heading_deg + (col/width)*360) mod 360`. Check
  `pipeline_metadata.json:heading_reliable` first: it is the GPS-bearing
  cross-check verdict. `folkestone_dover` is `true`; `kurashiki_pano_dense` is
  `false` (its `compass_angle` is exactly 0.0 on every frame and the SfM bearing
  disagrees with the travel bearing by ~90°).
* **Fit a yaw offset per dataset**, as the boston_harbor pipeline already does
  (`yaw_offset.method: triangulation_sweep`). This is the only option where
  `heading_reliable` is false, and it is the safer default given that these
  headings are third-party metadata rather than a calibrated rig.

The reason not to rotate: rotating bakes a heading estimate into the pixels,
where an error cannot be undone without re-deriving from the originals and cannot
be recalibrated the way a recorded angle can — and `heading_reliable=false` on
one of the two panoramic pilots shows that distrust is warranted.

## The rest: perspective (non-360) datasets

14 of the 22 collected trajectories are limited-FOV captures, not panoramas
(56.6°–93.4° horizontal). They are stored natively — original framing, full
resolution — with a per-frame `intrinsics.csv` instead of four pinhole faces.
This is what the farfield pipeline needs in order to read them.

Contact points below are in
`experimental/overhead_matching/swag/landmark_filtering/` unless noted.

## What the datasets already give you

`intrinsics.csv`, one row per frame:

```
idx,pano_id,projection,width,height,focal_norm,k1,k2,hfov_deg,vfov_deg,heading_deg,heading_reference,heading_source
```

`heading_reference` states what `heading_deg` is the bearing *of* —
`optical_axis` for perspective frames, `column_0` for equirectangular ones. It is
per row precisely because that is the most confusable thing in the file.

`focal_norm` is Mapillary's focal length normalized by `max(width, height)`, so
focal in pixels is `focal_norm * max(width, height)`, and

```
hfov_deg = 2*atan((width/max(w,h)) / (2*focal_norm))
```

`pipeline_metadata.json` carries `projection: "perspective"`,
`is_equirectangular: false`, `north_aligned: false`, and `intrinsics_csv`.
Detect the dataset kind from those rather than from a filename convention.

## 1. `ingest.py` — single-view mode

* `VALID_FACE_YAWS = ("0", "90", "180", "270")` (line 26) and the check at
  line 257 reject any box whose `yaw_angle` is not one of the four faces. A
  single-view frame has no face; allow a sentinel (absent, or `"single"`) and
  carry `face_yaw_deg = 0`.
* `run_ingest` requires `<pinhole_base>/{pano_stem}/yaw_{000,090,180,270}.jpg`.
  For these datasets there is no pinhole dir at all — gate that requirement on
  the projection, and read boxes against the stored image directly.
* `_merge_across_faces` / `_box_edges_unwrapped` seam logic (lines 116–178)
  exists to rejoin boxes split across face boundaries. A single view has no
  seams, so bypass it rather than adapt it. This also sidesteps the documented
  seam-adjacency bug (`ingest.py:147`).

## 2. Field of view must become per-frame

`ingest.py` passes `config.fov_deg` — one scalar for the whole dataset — into
`bg.bearing_camera_deg` and `bg.elevation_deg` (lines 202–208). That is fine for
90° pinhole faces but wrong here: FOV varies **per trajectory** (Seattle 56.6°,
Mississippi 93.4°) and can vary per frame within one. Thread the value from
`intrinsics.csv` keyed by `idx`/`pano_id`. A single wrong FOV scales every
bearing offset in the frame, so this is not a detail that averages out.

## 3. Bearing math — simpler here, and free of the mirror bug

The existing formula already generalizes; only its inputs change:

```python
# bearing_geometry.bearing_camera_deg, with the frame's own values
col_frac    = x_norm / 1000 * 2 - 1
offset_deg  = degrees(atan(col_frac * tan(radians(hfov_deg) / 2)))
bearing_world = (heading_deg + offset_deg) % 360      # heading from intrinsics.csv
```

Two things worth being explicit about:

* **Use the frame's `heading_deg`, not a face yaw.** For perspective frames
  `heading_deg` is the camera's own pointing direction as recorded by Mapillary,
  which is exactly what the offset should be added to.
* **The `bearing_camera_deg` mirroring caveat does not apply.** That warning
  (module docstring, "faces 90/270 pointing ~180 deg away") is a property of the
  equirect→pinhole reprojection in `panorama_to_pinhole.py`. A perspective frame
  is the native camera image, never reprojected, so image-right is genuinely
  increasing clockwise azimuth. These datasets therefore avoid the mirror
  entirely — do not apply a mirror correction to them.

Lens distortion (`k1`, `k2`) is recorded but not applied. For these images the
implied edge displacement is small relative to a landmark bbox, but it is a real
term for wide frames (Mississippi at 93°, `k1 = -0.201`); either undistort `x`
before the `atan`, or state in the config that it is neglected.

## 4. Heading validation differs, deliberately

The converter does **not** apply the equirect GPS-bearing gate to these
datasets. A side-mounted ferry camera legitimately points 90° off travel, so a
large median disagreement is mount geometry, not error. Nor does it gate on the
camera-to-travel offset being *consistent*: a hand-held camera that pans around
the deck is still perfectly usable, because the per-frame heading describes where
the camera actually pointed. `portsmouth_navalbase` is exactly this case — a
66.9° offset spread, reported as `camera_pans_relative_to_travel: true`.

What is checked instead is that the two independent heading sources agree with
each other (`heading_sources_median_disagreement_deg`, warn above 25°), since
that is the only meaningful test when pointing is unconstrained by travel.
Portsmouth's sources agree to 8.9° and NYC's to 10.3°, so both headings are
trustworthy.

Any downstream code that assumes "heading ≈ direction of travel" must not be
applied to these datasets.

## 4b. Do not derive heading from GPS course on vessel trajectories

`object_tracking/heading.py:heading_model_from_positions` fits heading from GPS
positions, which is the right call for boston_harbor. It is **wrong on parts of
these ferry tracks**: on `folkestone_dover`, heading matches GPS course to a
median of 0.33°, but 8.4% of frames (in contiguous runs, e.g. f0073–f0080)
disagree by ~180° because the vessel is *going astern out of the berth* —
tracking north at 3 m/s on a steady south-east heading. Ferry departures and
arrivals do this by design.

So a GPS-derived heading is 180° wrong exactly during the manoeuvring segments,
which are also the segments closest to the harbour landmarks. Prefer the recorded
`intrinsics.csv:heading_deg` where `heading_reliable` is true, and if you must
fall back to course-over-ground, exclude low-speed segments rather than trusting
them.

## 5. No pinhole stage

`panorama_to_pinhole.py` is skipped for these trajectories, and should stay
skipped: a perspective capture already *is* the pinhole image the pipeline wants.

If it is cheaper to make the existing face-indexed code paths work unchanged than
to add a single-view mode, the shim is to present each frame as one face —
write it to `<pinhole_base>/<pano_stem>/yaw_000.jpg` and set the box
`yaw_angle` to `0` — but only if the per-frame FOV from item 2 is threaded
through, since the shim otherwise silently reintroduces the 90° assumption.

## 6. VLM extraction

`extract_gemini_landmarks_from_panoramas.py` renders faces then asks for boxes
tagged with `yaw_angle`. For a single view, stage 1 (PINHOLE) drops out, the
request carries one image, and the response schema's `yaw_angle` must be fixed or
absent — with the matching relaxation of `VALID_FACE_YAWS` in item 1. The
`osm_tags_farfield` prompt text itself needs no change; it is about distant
landmarks from a vessel, which is exactly the content.

## 7. Tracking (M1–M3) needs a video or a gate

`object_tracking/video_frames.py` indexes a source video as
`video_t_s * fps`. These datasets have no video. `frames_gps.csv` does carry a
real `video_t_s` (seconds since first capture), so either synthesize a video
from the frames at a nominal fps, or gate the video-dependent milestones off for
these datasets. Note the frame interval is not uniform — it comes from
`--min_spacing_m` decimation of an irregular capture — so a synthesized
constant-fps video will not have `video_t_s * fps` land on the right frame
unless it is built from the same timestamps.

## 8. Cross-dataset metrics need solid-angle normalization

A 360 frame observes every azimuth; a 66° frame observes 18% of them. So
"landmarks per frame" is not comparable between the two families, and neither is
any recall-style metric that treats a frame as an omnidirectional observation.
Normalize by observed azimuth span (available per frame as `hfov_deg`) before
comparing a perspective dataset against a panoramic one, or the perspective
datasets will look uniformly worse for a purely geometric reason.
