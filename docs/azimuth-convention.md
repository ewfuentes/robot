<!-- Moved into the repo 2026-08-19 from ~/scratch/mappilary/AZIMUTH_CONVENTION.md,
which mapillary_to_vigor.py has cited by name since the tools moved in on
2026-08-17 while the document itself stayed outside the tree. It is the EVIDENCE
for the panorama azimuth conventions; docs/conventions.md is the register that
indexes it alongside every other convention in the project. -->

# Panorama azimuth convention (measured, not assumed)

> **Decision (2026-08-12): images are stored UNROTATED.** Panoramas are kept as
> captured and are *not* north-aligned. Rotating would bake a heading estimate
> into the pixels, where an error is unfixable without re-deriving from the
> originals and cannot be recalibrated the way a recorded angle can — and the
> heading is not always trustworthy (kurashiki's `compass_angle` is exactly 0.0
> on 100% of frames). Orientation is carried per frame in `intrinsics.csv`
> (`heading_deg`, `heading_reference`), and `pipeline_metadata.json` records the
> column->azimuth formula. `heading_reliable` flags whether the recorded angle
> passed the GPS-bearing cross-check.
>
> For an unrotated Mapillary equirectangular frame:
>
> ```
> azimuth_deg = (heading_deg + (col / width) * 360) mod 360
> ```
>
> because `heading_deg` (= `computed_compass_angle`) is the bearing of
> **column 0**. The rest of this document is the evidence for that left-edge
> fact, which is what makes the formula above correct — and which also explains
> the 180 deg error in the six pre-existing datasets that *were* rotated.


## The house convention

Stated verbatim in the self-collected datasets' own exporter,
`/data/overhead_matching/datasets/VIGOR/fells_midday_run/make_north_aligned_frames.py`:

> the camera's "forward" (direction of walking) sits at FORWARD_FRAC (75%) across
> the frame from the left, and bearing increases left->right (the standard
> equirect convention) … North (b=0) is put at the center …
> After alignment: north at 50%, east at 75%, south at 0%/100%, west at 25%.

So for a north-aligned panorama of width `W`:

```
azimuth_deg = ((col - W/2) / W) * 360   (mod 360)
col         = (W/2 + azimuth_deg/360 * W) mod W
```

This is the convention to use if you ever *do* north-align. The current
converter does not rotate; it records `azimuth_convention` in each dataset's
`pipeline_metadata.json` so consumers can map columns to azimuth themselves.

## What Mapillary actually gives us

`computed_compass_angle` is the bearing of the **left edge (column 0)** of the
equirectangular image, **not** the bearing of the centre column.

This contradicts `data_pipeline/mapillary/README.md` in loci-release, which says
"The center column of a Mapillary equirectangular image points in the
`computed_compass_angle` direction." That claim is wrong, and it made the roll
land north at column 0 — a 180° error against the house convention.

If north-aligning to the house convention were wanted, the correct roll would be:

```python
pixel_shift = int(round(cca / 360.0 * width + width / 2.0)) % width
image = np.roll(image, pixel_shift, axis=1)
#                     ^ W*cca/360 brings north to column 0
#                                    ^ +W/2 moves it on to the centre
```

The converter deliberately does neither — see the decision note at the top.

## How it was measured

On `folkestone_dover` frame `f0200` (a Channel ferry, 2019-08-29 15:11 UTC,
51.0405 N 1.9614 E, heading 261.7°), using two independent absolute references:

| reference | true azimuth | predicted column | result |
|---|---|---|---|
| Sun (ephemeris) | 242.1° (elev 31.9°) | 707 | lands in the solar glare bloom |
| Ferry wake (reciprocal of heading) | 81.7° | 2978 | lands exactly on the wake |

The wake is the decisive one: it is a sharp, unambiguous feature, and unlike a
road it is not 180°-symmetric, so it distinguishes the two hypotheses that
differ by exactly half the image width.

Before the fix, the same two references landed 180° away. The sun alone gives a
residual of about −12° because the glare bloom is clipped by the ship's
superstructure on one side, which biases a brightness centroid; the wake has no
such bias.

Reproduce with `scratchpad/sun_verify.py` and the annotated-overlay snippet, or
directly:

```python
col_of_azimuth = (W/2 + az/360*W) % W     # sun, wake, bow all check out
```

## Related: `panorama_to_pinhole.py` uses the opposite handedness

`experimental/overhead_matching/swag/scripts/panorama_to_pinhole.py` maps

```python
col_frac = (np.pi - azimuth_rad) / (2 * np.pi)     # col = W*(180 - az)/360
```

which is the **mirror** of the house convention (`col = W*(180+az)/360` would
match it). Measured on a synthetic striped panorama, the rendered faces come out
as:

| face | centred on house-convention azimuth |
|---|---|
| `yaw_000` | 180° |
| `yaw_090` | 90° |
| `yaw_180` | 0° (north) |
| `yaw_270` | 270° |

i.e. `face_yaw = (180 − azimuth_house) mod 360`, with x increasing with
`azimuth_house` inside every face.

This is pre-existing behaviour that affects every dataset equally, and it is
consistent with the already-recorded observation that `bearing_geometry` comes
out mirrored on the 90/270 faces. It is **not** changed here — but note that a
mirror cannot be absorbed by a fitted per-dataset yaw offset the way a constant
rotation can, so it is worth resolving deliberately rather than by calibration.

## Consequence for the six pre-existing Mapillary datasets

`Middletown`, `Framingham`, `SanFrancisco_mapillary`, `post_hurricane_ian_sw`,
`netherlands_norr` and `netherlands_veluwe` were all produced by this converter
*before* the `+W/2` term existed, so their panoramas have **north at column 0**
— 180° from the house convention.

Confidence: the roll arithmetic is certain (same code path, and the `+W/2` term
was simply absent), and the left-edge semantics of `computed_compass_angle` were
measured directly on one capture. Those datasets could not be re-verified with
the sun because they are bottom-cropped and overcast. If anything downstream
consumed their absolute azimuths, it inherited the 180°; a pipeline that fits a
per-dataset yaw offset would have silently absorbed it.
