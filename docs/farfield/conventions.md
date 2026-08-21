# Conventions

Every frame, sign, zero point and identifier format this project relies on, in
one place, with the module that **owns** each one. Read this before writing
anything that converts a pixel to a direction, a direction to a bearing, or an
id to a lookup.

---

## The rule

**A convention has exactly one definition, in code, and every consumer imports
it. Restating it in prose, in a second constant, or in a JSON blob is how this
project breaks.**

That is not a style preference. It is the observation that **the same 180°
error happened three separate times**, each time because two places described
the same frame in their own words and drifted:

| when | what | how it was caught |
|---|---|---|
| before 2026-08-12 | loci-release's Mapillary README claimed `computed_compass_angle` is the bearing of the **centre** column. It is the **left edge**. Six datasets were north-aligned with north at column 0. | measured against a ferry **wake**. Recorded 2026-08-19 in the six datasets' `azimuth_convention` (pixels deliberately not re-rolled). |
| 2026-08-05 | the old `bearing_geometry.bearing_camera_deg` carried its own copy of the camera-frame maths; faces 90/270 came out exactly 180° from the physical direction (a per-face constant rotation of `−2·face_yaw`, not a mirror). | reprojecting boxes onto panorama pixels. Fixed 2026-08-19; the farfield tree has **one** definition, in `geometry.py`. |
| 2026-08-19 | `pohang_canal_04`'s recorded `mount_offset_deg` was reasoned in the metadata's **column_0** frame while every consumer reads the **centre** frame. | `mount_offset_sweep` disagreed by ~180° and an independent derivation agreed with the sweep. |

Note what all three have in common: **nothing failed.** No exception, no test,
no assertion. A 180° frame error produces a perfectly well-formed number, and
downstream code consumes it happily. The farfield tree therefore *enforces* the
contract at every seam it can (see "Enforcement points" below) instead of
relying on prose.

### Rules that follow

1. **Import, never restate.** `geometry.CAMERA_FRAME`,
   `geometry.MOUNT_OFFSET_CONVENTION` and `geometry.MOUNT_OFFSET_FRAME` are
   constants so they can be embedded in artifacts without being retyped.
2. **Say where zero is.** "The azimuth in the camera frame" is ambiguous, and
   ambiguity is what caused all three incidents. Write "zero at the centre
   column" or "zero at column 0" every time.
3. **A validated number is validated *for one frame*.** An `accuracy_validated`
   flag must never travel between quantities; a mount-offset record without a
   `frame` field is refused, not interpreted.
4. **Prefer a 180°-asymmetric reference.** A road, a shoreline and a
   triangulation residual are all nearly symmetric under a half turn. A wake,
   a bow, the sun, and a named building are not.
5. **Relative methods cannot catch a shared error.** The sweep finds the angle
   that makes rays to unknown objects agree *with each other*; it fits a 180°
   slip perfectly. Trust it only when something independent agrees.

---

## 1. The camera frame — CANONICAL

**Owner: `farfield/geometry.py`** (the exact, empirically verified inverse of
`extraction/panorama_to_pinhole.py`).

```
az_cw  = (x / pano_w - 0.5) * 360        # clockwise-positive from camera FORWARD
el_up  = (0.5 - y / pano_h) * 180        # up-positive
```

**Zero azimuth is the CENTRE column.** Face layout across the panorama, left
to right: `180 | 90 | 0 | 270`. The face-yaw label is a *render parameter*,
not a camera-frame azimuth: the azimuth of a face centre is `−face_yaw mod
360` (faces 90/270 swap). Elevation has exactly one definition too —
`direction_from_face_px` is off-axis correct, and `bbox_angles` evaluates it
at the bbox centre.

`geometry.azimuth_of_pano_column` is the named helper for "the azimuth of a
column"; `apply_mount_offset` and `body_to_world_bearing_deg` are the named
frame steps. Nothing else converts.

## 2. `mount_offset_deg`

**Owner: `geometry.MOUNT_OFFSET_CONVENTION` / `geometry.MOUNT_OFFSET_FRAME`.**

> the azimuth, IN THE CAMERA FRAME, of the vehicle's **DIRECTION OF TRAVEL** —
> not the bow. Applied as `bearing_body_deg = (bearing_camera_deg −
> mount_offset_deg) mod 360`. Camera-frame azimuth 0 is the **CENTRE** column,
> not column 0; a prior reasoned in the column-0 convention is exactly 180° out.

Two traps in one quantity: direction of travel is not the bow (they differ by
crab/leeway), and zero is at the centre (§1).

| tool | kind | can it catch a 180° slip? |
|---|---|---|
| `calibration:sun_offset_check` | **absolute** — sun vs ephemeris | **yes.** The only one that can. Its `usable` flag is verdict-gated: a FIXED-OBJECT abstention can never publish. |
| `calibration:mount_offset_sweep` | relative — triangulation self-consistency | no. Fits a slip perfectly. |
| operator prior | one look at one frame | no |

Both estimators write **run-dir sidecars only**, stamped with the convention
string and `frame` constant. The one writer of dataset metadata is
`dataset_tools:publish_mount_offset`, which enforces the accuracy-validated
guard and regenerates checksums. `localization:build_export` resolves
best-evidence-first (explicit flag → validated dataset record → sun sidecar →
sweep sidecar → unvalidated record, loudly) and records the source.

## 3. Absolute azimuth from a column (dataset metadata)

**Owner: each dataset's `pipeline_metadata.json:azimuth_convention`**, written
by `dataset_tools:ingest_selfcollect` and `collection:mapillary_to_vigor`.
The equirect formula's zero is **column 0** — a different zero from §1, and
the frame Pohang's bad offset was reasoned in. Both writers stamp a
`mount_offset_frame` note saying exactly that, and `dataset.py` refuses to
consume a `mount_offset` block that does not carry `frame ==
geometry.MOUNT_OFFSET_FRAME` and an explicit `applied_to_heading_deg`.

Images are stored **unrotated** and never north-aligned: rotating bakes a
heading estimate into pixels where an error cannot be recalibrated.
`dataset.require_camera_frame_panoramas` refuses a dataset whose
`north_aligned` is true *or unrecorded*.

## 4. Heading, course, and bearings

| quantity | convention | owner |
|---|---|---|
| heading / course | degrees **clockwise from north**, `atan2(east, north)` | `geometry.compass_bearing_deg` / `compass_bearing_rad` |
| pipeline heading | derived from **GPS course over ground**, never from `intrinsics.csv:heading_deg` | `calibration/heading.py` |
| serialized bearings (`bearing_body_deg` etc.) | stored in **[0, 360)**; compare with the wrap helpers, never by subtraction | `localization/structs.py`, validated at `export_ingest` |
| Mapillary `computed_compass_angle` | bearing of the **LEFT EDGE**, not the centre (incident 1) | recorded per dataset |

## 5. Bounding boxes

Face bbox coords are normalised **0–1000** per face, **y down**
(`geometry.BBOX_NORM_MAX` — the only definition). Pano boxes are represented
unwrapped: `x_max > W` means the box crosses the seam.

## 6. Identifiers

| thing | format | where |
|---|---|---|
| landmark id in matching/export artifacts | `osm:node:257370656` (element kind kept — node 123 ≠ way 123) | `catalog.catalog._id_text` |
| landmark id in the feather | a **tuple repr string** `"('node', 257370656)"` | feather `id` column |
| panorama frame filename | `f####,<lat>,<lon>,.jpg` — comma separated, trailing comma | dataset `panorama/` |
| tracklet id | `T<track_id>` — one per audited track (no merged ids) | `tracking/tracklets.py` |
| observation id | `f{pano}__lm{index}__box{n}` — positional, encodes no content | `dataset.py` |

Positional ids **resolve on any dataset**, so hardcoded fixtures from one leg
never fail loudly on another; gate fixtures on the dataset name, never on
whether ids resolve. And never parse digits out of a pano id to get a frame
index — `dataset.frame_index_by_pano_id` is the one sanctioned join (they
diverge the moment a panorama is missing).

## 7. Time and space

| quantity | convention | owner |
|---|---|---|
| `video_t_s` | seconds into the source video, carried **verbatim** through trims, never rebased (the charles_river 510 s incident) | `dataset_tools:trim_dataset`, audited by `audit_dataset`'s NCC check |
| ENU | region-anchored, anchor = mean frame lat/lon (data-dependent by design; participates in catalog cache keys) | `geometry.RegionFrame`, `dataset.fill_enu` |
| catalog positions | centroid for extended features; `bearing_span_from` returns the angular interval instead | `catalog/catalog.py` |

## Enforcement points

The places where a frame error stops being a well-formed number and becomes an
error, in pipeline order:

1. `dataset.require_camera_frame_panoramas` — north-aligned/unrecorded refused.
2. `dataset.mount_offset_record` — unqualified offset blocks refused.
3. `audit_dataset` — the contract audit, including the video-addressing NCC.
4. Calibration sidecars — stamped with convention + frame by construction.
5. `localization/export_ingest.load` — an export without offset provenance, or
   in the wrong frame, or with out-of-range bearings, is refused.
6. `run_io.write_run` — a run that cannot name its inputs is refused.

## Adding a convention

1. Define it **once**, in the module that owns the quantity, as a constant.
2. Say where zero is and which way is positive.
3. Add a row here, naming the owner.
4. If it can be confused with an existing one, say by how much — "exactly
   180°" is the most useful sentence in this document.
5. Find a 180°-asymmetric way to test it, and write that test.
6. If a consumer can be handed the quantity from outside (metadata, an
   export), add an enforcement point that refuses the unqualified form.
