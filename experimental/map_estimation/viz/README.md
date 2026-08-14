# Argoverse 2 viewer

Renders an AV2 log in [rerun](https://rerun.io). Right now that means the **egovehicle and the
path it drove**; sensor streams and model output are the next layers on top.

```bash
V="bazel run //experimental/map_estimation/viz:view_log --"

$V sensor/val                       # list the logs on disk, with what each one has
$V sensor/val --log_id 02678d04-cc9f-3148-9f95-1ba66347dff9
$V tbv --log_id 07YOTz..._Spring_2020 --serve      # stream to a browser
$V sensor/val --log_id 02678d04... --save /tmp/log.rrd
```

The dataset spec, `--log_id` and `--root` mean what they mean in
[`data:argoverse`](../data/README.md), and both tools resolve paths through
`argoverse_layout`, so whatever you downloaded is what this opens. Use `--serve` when the data
sits on a remote box: same logging code, viewer in a browser.

## Layout

| file | role |
|---|---|
| `av2_source.py` | finds a log on disk, reports which streams it has, loads them via `av2`. No rerun import. |
| `av2_scene.py` | turns a source into rerun entities. Owns the entity paths and the vehicle outline. |
| `view_log.py` | CLI. |

The split exists so the loader stays testable without a renderer, and so prediction tooling can
reuse ego poses and calibration without pulling in a viewer.

## The entity tree is the transform hierarchy

A child entity inherits its parent's `Transform3D`, so the path layout *is* the coordinate
story:

```
world                 city frame, right-handed z-up
world/path            the whole drive as one static polyline, in city coordinates
world/ego             city_SE3_egovehicle, one per pose timestamp (~170 Hz)
world/ego/wireframe   the vehicle outline, logged ONCE as static data
```

The wireframe is logged once and moves because its parent moves. Nothing re-logs geometry per
frame. That is the habit to carry into everything added later.

Two consequences worth knowing before extending this:

- **Batch by kind, not by object.** One `LineStrips3D` carrying N strips is one entity with N
  instances. N entities for N objects is slower and makes the entity tree unusable. The vehicle
  is 3 entities (body, wheels, nose), not 20.
- **Do not synchronize streams yourself.** Log each on its own timestamps; the viewer resolves
  "latest value at time T" when scrubbing. Poses run far faster than any sensor, so anything
  logged later lands against a pose within milliseconds of it.

### The view follows the vehicle

The default 3D view is anchored to `world/ego`, so the car holds still and the city sweeps past
it. A view's **origin** is the frame it renders in, and that is the entire mechanism — rerun has
no separate follow toggle.

Its **contents** are `/**` rather than the default `$origin/**`, and that override is load
bearing. `$origin/**` under `world/ego` is the wireframe alone: `world/path` is a *sibling* of
the ego, not a descendant. That placement is deliberate — the path is in city coordinates and
must not inherit the ego transform, or it would drag along with the car instead of staying fixed
to the map — so the view has to reach outside its own origin to draw it. Narrow `contents` back
to `$origin/**` and the path silently disappears.

To switch to a world-fixed view without touching code: select the 3D view, then in the Selection
panel set **Space origin** to `/world` and **Entity path filter** back to `$origin/**`.

**Blueprint edits persist, and they win.** The viewer stores a blueprint per application id
(`argoverse_log`), so once you rearrange anything by hand it sticks across runs and across logs
— and the `default_blueprint` this code sends stops taking effect. If a change here appears to
do nothing, that is why: click **reset blueprint** in the viewer once to pick up the code's
layout again.

### Timelines

`elapsed` (seconds from the log's first pose) is the one you scrub; `timestamp_ns` carries raw
AV2 nanoseconds for cross-referencing against files on disk. Note that `log_time` and `log_tick`
are **reserved** — rerun stamps every `rr.log` call with them, so naming your own timeline
`log_time` makes the SDK complain the timeline changed type.

## Adding streams

Every stream is optional and independently downloaded: a `sensor/val` log often has lidar and
annotations but no imagery, while a `tbv` log has seven cameras and no annotations. Ask
`LogSource.present_items()` rather than assuming. The natural next entities are

```
world/map/...              lane boundaries, crosswalks, drivable areas (static, city frame)
world/ego/lidar            sweeps; already egomotion-compensated into the ego frame
world/ego/annotations      ground-truth cuboids; AV2 stores these in the ego frame too
world/ego/cameras/<name>   ego_SE3_cam + Pinhole (static), EncodedImage per timestamp
```

Log images as `rr.EncodedImage(path=...)` rather than decoding: it stores the jpeg bytes and
lets the viewer decode on demand, which is the difference between 91 MB and gigabytes for one
camera.

## Model output

Predictions go under `av2_scene.PREDICTION_CITY` (`world/prediction`) or
`av2_scene.PREDICTION_EGO` (`world/ego/prediction`) — siblings of the ground-truth paths, under
the same transforms. That placement is what lets them be toggled independently and compared
frame by frame without either code path knowing about the other.

Useful archetypes for that layer, all present in the pinned version: `Boxes3D` for predicted
objects, `LineStrips3D` for trajectories, `Arrows3D` for velocities, `Ellipsoids3D` for
covariance (eigendecompose Σ, `half_sizes = k·√eigenvalues`, quaternion from the eigenvectors),
and `Scalars` for error metrics — scalar plots are time-synced to the 3D view, so clicking a
spike in an error curve jumps the scene to that frame.

## The rerun pin

`rerun-sdk==0.23.1`, in `third_party/python/requirements_3_12.in`.

**It cannot be bumped without moving numpy first.** 0.23.2 and every later release require
`numpy>=2`; this repo pins `numpy==1.26.4`. 0.23.1 is the last version that accepts numpy 1.x.
What that costs is the `GridMap` and `VoxelGridMap` archetypes (added in 0.31 and 0.34), which
would otherwise be the natural fit for occupancy grids — until then, a grid is `Points3D` or a
batched `Boxes3D`.

The wheel also needs `extra_requirement("rerun-sdk", "rerun")` in BUILD files rather than the
usual `requirement("rerun-sdk")`; see `third_party/python/extra_rerun_targets.bzl` for why.
