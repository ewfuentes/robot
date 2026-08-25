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

1. **Import, never restate.** `geometry.CAMERA_FRAME` and
   `nominal_forward.FRAME` are constants so artifacts can embed the contracts
   without retyping them.
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
column"; `nominal_forward.camera_to_forward_cw_deg` and
`geometry.forward_to_world_bearing_cw_deg` are the named frame steps. Nothing
else converts.

## 2. Nominal forward

**Owner: `farfield/nominal_forward.py`.**

Nominal forward is the platform's fixed longitudinal forward axis. The
approved record identifies its azimuth in the camera frame, whose zero is the
panorama centre column. A camera-frame bearing becomes a forward-frame bearing
through `nominal_forward.camera_to_forward_cw_deg`.

Nominal forward is deliberately **not** GPS course over ground and is never
promoted automatically from a sun check, sweep, or localization residual.
Those are diagnostics. Only a human-approved, dataset-bound
`farfield_nominal_forward/v1` record may rotate localization evidence. The
record binds its panorama column/width derivation, mounting identity,
uncertainty, evidence frames, operator, and approval time; the loader rejects
unknown fields or an unapproved/wrong-dataset record.

`calibration:nominal_forward_review` builds a content-bound evidence bundle and
non-authoritative template. Its separate finalize operation is the explicit
approval boundary. Alignment diagnostics can reveal a likely half-turn or
other systematic error, but cannot rewrite the calibration.

## 3. Absolute azimuth from a column (dataset metadata)

**Owner: each dataset's `pipeline_metadata.json:azimuth_convention`**, written
by `dataset_tools:ingest_selfcollect` and `collection:mapillary_to_vigor`.
Source metadata may define an equirectangular absolute azimuth from **column
0** — a different zero from §1, and the frame in which the historical Pohang
error was reasoned. Both writers stamp the source convention and the canonical
camera-frame contract. `dataset.py` refuses rotated/north-aligned pixels or a
missing/mismatched camera-frame declaration.

Images are stored **unrotated** and never north-aligned: rotating bakes a
heading estimate into pixels where an error cannot be recalibrated.
`dataset.require_camera_frame_panoramas` refuses a dataset whose
`north_aligned` is true *or unrecorded*.

## 4. Heading, course, and bearings

| quantity | convention | owner |
|---|---|---|
| heading / course | degrees **clockwise from north**, `atan2(east, north)` | `geometry.compass_bearing_deg` / `compass_bearing_rad` |
| pipeline heading | derived from **GPS course over ground**, never from `intrinsics.csv:heading_deg` | `calibration/heading.py` |
| serialized bearings (`bearing_forward_cw_deg` etc.) | stored in **[0, 360)**; compare with the wrap helpers, never by subtraction | `localization/structs.py`, validated by localization-input ingest |
| Mapillary `computed_compass_angle` | bearing of the **LEFT EDGE**, not the centre (incident 1) | recorded per dataset |

## 5. Bounding boxes

Face bbox coords are normalised **0–1000** per face, **y down**
(`geometry.BBOX_NORM_MAX` — the only definition). Pano boxes are represented
unwrapped: `x_max > W` means the box crosses the seam.

## 6. Identifiers

| thing | format | where |
|---|---|---|
| landmark id in matching/export artifacts | namespaced element identity such as `osm:node:...` (element kind kept) | `catalog.catalog._id_text` |
| landmark id in the feather | a **tuple repr string** `"('node', 257370656)"` | feather `id` column |
| panorama frame filename | `f####,<lat>,<lon>,.jpg` — comma separated, trailing comma | dataset `panorama/` |
| tracklet id | artifact-scoped identity derived from the source track | `tracking/tracklets.py` |
| observation id | `obs-<digest>` derived from dataset/frame/local observation identity | `dataset.py` |
| local observation id | `f{pano}__lm{index}__box{n}` — useful for display, not global identity | `dataset.py` |

Local positional ids are presentation/debug aids. Scientific joins use the
artifact-scoped tracklet identity and content-derived global observation id;
consumers must still validate the owning dataset/artifact rather than trusting
the shape of an id. Never parse digits out of a pano id to get a frame index —
`dataset.frame_index_by_pano_id` is the sanctioned join (they diverge the
moment a panorama is missing).

## 7. Time and space

| quantity | convention | owner |
|---|---|---|
| `video_t_s` | seconds into the source video, carried **verbatim** through trims, never rebased (the charles_river 510 s incident) | `dataset_tools:trim_dataset`, audited by `audit_dataset`'s NCC check |
| ENU | region-anchored, anchor = mean frame lat/lon (data-dependent by design; participates in catalog cache keys) | `geometry.RegionFrame`, `dataset.fill_enu` |
| catalog positions | centroid for extended features; `bearing_span_from` returns the angular interval instead | `catalog/catalog.py` |

## Enforcement points

The places where a frame error stops being a well-formed number and becomes an
error, in pipeline order:

1. `dataset.require_camera_frame_panoramas` — north-aligned, rotated, or
   unrecorded camera-frame imagery is refused.
2. `audit_dataset` — the frozen dataset contract, including video addressing.
3. `nominal_forward.load` — unapproved, wrong-frame, wrong-dataset, or
   internally inconsistent calibration records are refused.
4. `tracking:build_bearing_observations` — only supported observations inside
   canonical audit segments become bearings.
5. `localization:build_export` — exact bearing/match/catalog lineage and the
   approved nominal-forward bytes are required.
6. Typed artifact and run writers — outputs that cannot name and validate
   their exact inputs are refused.

## Adding a convention

1. Define it **once**, in the module that owns the quantity, as a constant.
2. Say where zero is and which way is positive.
3. Add a row here, naming the owner.
4. If it can be confused with an existing one, say by how much — "exactly
   180°" is the most useful sentence in this document.
5. Find a 180°-asymmetric way to test it, and write that test.
6. If a consumer can be handed the quantity from outside (metadata, an
   export), add an enforcement point that refuses the unqualified form.
