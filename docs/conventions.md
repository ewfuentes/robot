# Conventions

Every frame, sign, zero point and identifier format this project relies on, in
one place, with the module that **owns** each one.

Read this before writing anything that converts a pixel to a direction, a
direction to a bearing, or an id to a lookup.

---

## The rule

**A convention has exactly one definition, in code, and every consumer imports
it. Restating it in prose, in a second constant, or in a JSON blob is how this
project breaks.**

That is not a style preference. It is the observation that **the same 180°
error has now happened three separate times**, each time because two places
described the same frame in their own words and drifted:

| when | what | how it was caught |
|---|---|---|
| before 2026-08-12 | loci-release's Mapillary README claimed `computed_compass_angle` is the bearing of the **centre** column. It is the **left edge**. Six datasets (`Middletown`, `Framingham`, `SanFrancisco_mapillary`, `post_hurricane_ian_sw`, `netherlands_norr`, `netherlands_veluwe`) were north-aligned with north at column 0. | measured against a ferry **wake** — see [`azimuth-convention.md`](azimuth-convention.md). **Recorded 2026-08-19** in all six datasets' `azimuth_convention` (pixels deliberately not re-rolled: derived artifacts). |
| 2026-08-05 | `bearing_geometry.bearing_camera_deg` assumed the renderer's unmirrored handedness. Faces 90/270 come out ~180° from the physical direction. | reprojecting boxes onto panorama pixels (`m0_render_boxes`). **Fixed 2026-08-19** — now delegates. |
| 2026-08-19 | `pohang_canal_04`'s `mount_offset_deg: 180` was reasoned in the dataset metadata's **column_0** frame while every consumer reads it in `pano_geometry`'s **centre** frame. | `mount_offset_sweep` said 358°, and an independent derivation from the raw CSVs agreed to 0.5°. **Fixed 2026-08-19** — metadata now 358°, old value under `superseded`. |

Note what all three have in common: **nothing failed.** No exception, no test,
no assertion. A 180° frame error produces a perfectly well-formed number, and
downstream code consumes it happily. Two of the three were caught only because
somebody rendered a picture and looked at it.

And note the multiplier on the third one: `m11_base_export` ranks an
`accuracy_validated` recorded offset **above** the run's own sweep, deliberately,
because a relative sweep "silently reproduces … a 180 deg convention slip, which
it fits perfectly." When the slip is in the validated value instead, that
protection runs backwards and the export bakes in the wrong number **with maximum
confidence**.

### Rules that follow

1. **Import, never restate.** `pano_geometry.CAMERA_FRAME` and
   `pano_geometry.MOUNT_OFFSET_CONVENTION` are strings so they can be embedded
   in artifacts without being retyped. If you need the words, import them.
2. **Say where zero is.** "The azimuth in the camera frame" is ambiguous and
   ambiguity is what caused all three incidents. Write "zero at the centre
   column" or "zero at column 0" every time.
3. **A validated number is validated *for one frame*.** Pohang's sun check is
   genuinely good (0.76° median) and validates the *azimuth formula*. It says
   nothing about a `mount_offset_deg` quoted in a different frame. Do not let
   `accuracy_validated: true` travel between quantities.
4. **Prefer a 180°-asymmetric reference.** A road, a shoreline and a
   triangulation residual are all nearly symmetric under a half turn. A wake, a
   bow, the sun, and a named building are not. Only asymmetric evidence can
   distinguish the two hypotheses that matter.
5. **Relative methods cannot catch a shared error.** `mount_offset_sweep` finds
   the angle that makes rays to unknown objects agree *with each other*; it fits
   a 180° slip perfectly. Trust it only when something independent agrees.

---

## 1. The camera frame — CANONICAL

**Owner: `swag/landmark_filtering/object_tracking/pano_geometry.py`**

```
az_cw  = (x / pano_w - 0.5) * 360        # clockwise-positive from camera FORWARD
el_up  = (0.5 - y / pano_h) * 180        # up-positive
```

**Zero azimuth is the CENTRE column.** Camera forward is the centre of the
image. Elevation is up-positive even though image `y` grows downward.

`pano_geometry` implements the exact inverse of `scripts/panorama_to_pinhole.py`
and is verified empirically: regenerating the stored pinhole faces from the
panorama with this maths reproduces them to JPEG noise, and reprojected landmark
boxes land on their objects. **This is the frame for anything that must land on
panorama pixels.**

Face layout across the panorama, left to right: `180 | 90 | 0 | 270`.

### Deviations from it that exist in the tree

**All resolved 2026-08-19.** There is now one definition and everything imports it.

| module | was | now |
|---|---|---|
| `landmark_filtering/bearing_geometry.py` | own copy of the maths, `face_yaw + atan((2u−1)t)` | **delegates** to `pano_geometry.direction_from_face_px` |
| `landmark_filtering/ingest.py:_is_seam_pair` | required the adjoining face to be `A + 90` | corrected to `A − 90` (i.e. `+270`) |
| `landmark_filtering/yaw_offset.py` | `bearing_sign` absorbed part of the error | kept, but documented as **not** a convention sponge; `+1` is correct for the current converter |
| `scripts/panorama_to_pinhole.py` | renders `col = W*(180 − az)/360` | unchanged — this is the *reference*, and `pano_geometry` is its verified inverse |

**What the old copy actually got wrong**, since the original description was
itself slightly off: it was **not** a mirror within each face. The two formulas
differ by exactly `−2 × face_yaw mod 360`, i.e. **0° on faces 0 and 180, and
exactly 180° on faces 90 and 270**. Both increase bearing image-right. That
matters, because it explains why a fitted per-dataset yaw offset appeared to help:
it could absorb the faces where the error was zero and never the rest. A per-face
rotation is not absorbable by a single global sign or offset.

**Impact of the fix.** Any bearing `bearing_geometry` produced for a box on face
90 or 270 was 180° out; faces 0 and 180 were always exactly right. Seam merging
was pairing faces that are physically 180° apart, so real continuations went
unmerged. Tests that pinned the old behaviour (`bearing_geometry_test`'s
`test_face_center_is_face_yaw`, three `ingest_test` seam fixtures) encoded the
error by reading a face *label* back out as an azimuth — the face yaw is a render
parameter, not a camera-frame bearing, and conflating them is what hid this.

---

## 2. `mount_offset_deg`

**Owner: `pano_geometry.MOUNT_OFFSET_CONVENTION`**, imported by
`mount_offset_sweep.py` and `sun_offset_check.py`.

> the azimuth, IN THE CAMERA FRAME, of the vehicle's **DIRECTION OF TRAVEL** —
> not the bow. Applied as `bearing_body_deg = (bearing_camera_deg −
> mount_offset_deg) mod 360`. Camera-frame azimuth 0 is the **CENTRE** column of
> the panorama, not column 0; a prior reasoned in the column-0 convention is
> exactly 180° out.

Two traps in one quantity:

- **Direction of travel, not the bow** (decided 2026-08-14). They differ by the
  crab/leeway angle — small, but they are not the same thing, and
  `bow_calibration.py` measures the *bow* and is therefore not a calibration of
  this.
- **Zero at the centre**, per §1.

How each estimator relates to it:

| tool | kind | can it catch a 180° slip? |
|---|---|---|
| `sun_offset_check.py` | **absolute** — sun vs ephemeris, no map, no tracks | **yes.** The only one that can. Abstains when `R < 0.8` (weather, not sun). |
| `mount_offset_sweep.py` | relative — minimises triangulation residual | **no.** Fits a slip perfectly. |
| `bearing_matcher.estimate_mount_offset` | relative, needs a landmark hypothesis | no |
| `bow_calibration.py` | measures the bow, not the direction of travel | not a calibration of this quantity at all |
| operator prior | one look at one frame | no |

`m11_base_export.resolve_mount_offset` ranks: explicit flag → `accuracy_validated`
metadata → this run's sweep → unvalidated metadata. **Rule 3 above is what makes
that ordering safe;** without it the top of that list is the most dangerous slot
in the pipeline.

---

## 3. Absolute azimuth from a column (dataset metadata)

**Owner: each dataset's `pipeline_metadata.json:azimuth_convention`**, written by
`scripts/ingest_selfcollect_dataset.py` and
`mapillary_tools/mapillary_to_vigor.py`.

Equirectangular:

```
azimuth_deg = (heading_deg + (col / width) * 360) mod 360     # heading_deg is the bearing of column_0
```

Perspective:

```
azimuth_deg = (heading_deg + degrees(atan((2*col/width − 1) * tan(hfov/2)))) mod 360
                                                    # heading_deg is the bearing of the OPTICAL AXIS
```

**These two branches put zero in different places** — column 0 for equirect,
the optical axis (i.e. the centre) for perspective. Both are internally correct.
The equirect one is the one that differs from §1, and it is the one Pohang was
reasoned in. Both writers now carry a `mount_offset_frame` note saying so.

Images are stored **unrotated** and are *not* north-aligned (decision
2026-08-12): rotating bakes a heading estimate into pixels where an error cannot
be recalibrated. If you ever do north-align, the house convention is

```
azimuth_deg = ((col − W/2) / W) * 360      # zero at the CENTRE — agrees with §1
```

---

## 4. Heading, course, and compensation

| quantity | convention | owner |
|---|---|---|
| heading / course | degrees **clockwise from north**, `atan2(east, north)` | `heading.py`, `bearing_geometry.py` |
| pipeline heading | derived from **GPS course over ground**, not from `intrinsics.csv:heading_deg` | `heading.heading_model_from_positions` |
| heading compensation | `az − dh` (verified on Pohang at dh=40°: the flipped sign loses the landmark) | `m1_heading_windows` |
| `intrinsics.csv:heading_reference` | `column_0` (equirect) / `optical_axis` (perspective) | the two ingest writers |
| Mapillary `computed_compass_angle` | bearing of the **LEFT EDGE**, not the centre — measured, and the source of incident 1 | [`azimuth-convention.md`](azimuth-convention.md) |

Note the second row: the tracking pipeline computes its own course from
positions and **never reads a dataset's `heading_deg`**. On Pohang that column is
attitude-derived and sun-validated, and the pipeline ignores it — which is why
the offset must reconcile *course* to the camera frame, not *heading*.

---

## 5. Bounding boxes

| convention | value | owner |
|---|---|---|
| face bbox coords | normalised **0–1000** per face, **y down** (Gemini convention) | `pano_geometry.BBOX_NORM_MAX`, `bearing_geometry.BBOX_NORM_MAX` |
| pano bbox wrap | represented unwrapped: `x_min ∈ [0, W)`, `x_max ∈ (x_min, x_min + W]`, so `x_max > W` means the box crosses the seam | `pano_geometry` |

---

## 6. Identifiers and serialisation

These are conventions too, and a mismatch here costs an afternoon rather than a
180° error.

| thing | format | where |
|---|---|---|
| landmark id, in matching artifacts | `osm:node:257370656` | `matching/signatures.json`, `matches.json` |
| landmark id, in the catalogue feather | `"('node', 257370656)"` — a **tuple repr string** | `landmarks/*.feather` `id` column |
| panorama frame filename | `f####,<lat:.6f>,<lon:.6f>,.jpg` — **comma separated, trailing comma** | dataset `frames/`, `panorama/` |
| track page filename | `track_<range_name>_T<id>.html`, range name from `tracks_*.json` | `m3_track_viewer.track_key` |
| observation id | `f{frame}__lm{index}__box{n}` — **purely positional, encodes no content** | `ingest` |

The first two cost real time today: joining them needs
`ast.literal_eval` on the feather side. The last one is a trap of its own —
positional ids **resolve on any dataset**, so hardcoded fixtures from one leg
never fail loudly on another; they silently describe the wrong object. That is
what put `custom_house_tower` on a Korean harbour and Boston's anchors on 13
datasets. Gate fixtures on the **dataset name**, never on whether the ids resolve.

---

## 7. Time and space

| quantity | convention | owner |
|---|---|---|
| `video_t_s` | seconds into the dataset's own source video; carried **verbatim** through trims, never rebased | `trim_dataset.py`, audited by `audit_dataset.py` |
| ENU | region-anchored; the anchor participates in the catalogue cache key | `harbor_catalog.enu_from_latlon` |
| catalogue positions | `representative_point()` of the geometry — a **centroid for extended features**, so an island's or a plant's apparent bearing drifts with viewing aspect | `harbor_catalog`; `bearing_span_from` returns the angular interval instead |

`video_t_s` has its own incident: `trim_dataset` used to rebase it to zero,
which moved every tracking window into a different stretch of the sail on
`charles_river_20260727` (510 s off) with no downstream symptom other than a
halved strong-evidence rate. `audit_dataset` now decodes what it points at and
cross-correlates against the panorama.

---

## Adding a convention

1. Define it **once**, in the module that owns the quantity, as a constant.
2. Say where zero is and which way is positive.
3. Add a row here, naming the owner.
4. If it can be confused with an existing one, say by how much — "exactly 180°"
   is the most useful sentence in this document.
5. Find a 180°-asymmetric way to test it, and write that test.
