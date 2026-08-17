# Argoverse 2 viewer

Renders an AV2 log in [rerun](https://rerun.io). Right now that means the **HD map, the
egovehicle, the path it drove, and the lidar**; imagery, annotations, and model output are the
next layers on top.

```bash
V="bazel run //experimental/map_estimation/viz:view_log --"

$V sensor/val                       # list the logs on disk, with what each one has
$V sensor/val --log_id 02678d04-cc9f-3148-9f95-1ba66347dff9
$V tbv --log_id 07YOTz..._Spring_2020 --serve      # stream to a browser
$V sensor/val --log_id 02678d04... --save /tmp/log.rrd
```

The dataset spec and `--root` mean what they mean in [`data:argoverse`](../data/README.md), and
both tools resolve paths through `argoverse_layout`, so whatever you downloaded is what this
opens. `--log_id` is deliberately *not* that CLI's repeatable, fnmatch-able selector: one
invocation produces one recording, so it takes exactly one log id. Use `--serve` when the data
sits on a remote box: same logging code, viewer in a browser.

## Layout

| file | role |
|---|---|
| `av2_source.py` | finds a log on disk, reports which streams it has, loads them via `av2`. No rerun import. |
| `av2_scene.py` | turns a source into rerun entities. Owns the entity paths, the map layers, the lidar coloring, and the vehicle outline. |
| `view_log.py` | CLI. |

The split exists so the loader stays testable without a renderer, and so prediction tooling can
reuse ego poses and calibration without pulling in a viewer.

## The entity tree is the transform hierarchy

A child entity inherits its parent's `Transform3D`, so the path layout *is* the coordinate
story:

```
world                                   city frame, right-handed z-up
world/map/lane_boundaries/<paint>       lane boundaries, bucketed by paint color
world/map/centerlines                   derived lane centers, 10 points per segment
world/map/crosswalks                    pedestrian crossings
world/map/drivable_areas                drivable-area outlines
world/path                              the whole drive as one static polyline
world/lidar                             the returns, one Points3D per sweep (10 Hz)
world/ego               city_SE3_egovehicle, one per pose timestamp (~170 Hz)
world/ego/wireframe     the vehicle outline, logged ONCE as static data
```

Everything above `world/ego` is in city coordinates. The map and the path are also *static* —
no timestamps, valid wherever you scrub; the lidar and the ego transform are the two timestamped
streams. The wireframe is logged once and moves because its parent moves. Nothing re-logs
geometry per frame. That is the habit to carry into everything added later.

**Everything but the wireframe is a sibling of the ego, not a child.** That is not stylistic: a
child inherits its parent's transform, so hanging the map under `world/ego` would make the road
drive along with the car. It would not error — it would just render physics that is quietly
wrong, which is the characteristic failure mode of this data model.

Two consequences worth knowing before extending this:

- **Batch by kind, not by object.** One `LineStrips3D` carrying N strips is one entity with N
  instances. N entities for N objects is slower and makes the entity tree unusable. The vehicle
  is 3 entities (body, wheels, nose), not 20.
- **Do not synchronize streams yourself.** Log each on its own timestamps; the viewer resolves
  "latest value at time T" when scrubbing. Poses run far faster than any sensor, so anything
  logged later lands against a pose within milliseconds of it.

### The map

Each log ships its own local vector map — 92 lane segments for a `sensor/val` log, 150 for the
`tbv` one — loaded with `build_raster=False`. `True` would additionally rasterize every
drivable-area polygon and read the ~1.6 MB ground-height surface off disk, which costs seconds
per log and produces nothing that gets drawn; it is only needed for `get_ground_height_at_xy`.

Lane boundaries are split into **four batched entities keyed by paint color** — yellow, white,
blue, and unmarked — because `LaneMarkType` spells the color into its own name and the split
lets you toggle a centerline separately from a road edge. The **dash pattern is not drawn**:
`LineStrips3D` has no dash support in the pinned version and faking one means chopping every
polyline into alternating segments, so what is encoded is the half of the marking that carries
navigational meaning. Boundary vertices are real city-frame z, so the map sits on the terrain.

Adjacent lane segments each carry their own copy of the boundary they share, so most interior
lines are drawn twice. At ~1500 vertices that is free; it just makes shared boundaries look
slightly brighter.

**Centerlines are derived, not annotated.** AV2 stores a lane as a ladder of two boundaries and
has no centerline field at all; `get_lane_segment_centerline` resamples both boundaries to 10
points and averages them pairwise. It is a faithful approximation — measured against these logs,
the result sits within 0.13 m (median) of equidistant from both boundaries, worst case 0.59 m,
implying lane widths of 3.7–3.9 m median — but the residual is real, because the two boundaries
are resampled by arc length independently, so paired points are not exactly opposite each other
on a curve. Look at them; don't regress against them.

**Lane topology is in the data and is not drawn.** `successors`, `predecessors`,
`left_neighbor_id`, and `right_neighbor_id` are all populated. Two traps if you use them:
`predecessors` is not the mirror of `successors` (113 vs 52 edges in the `sensor/val` log —
invert `successors` yourself if you need a reverse index), and ~13 successor ids per log point
at lanes outside the clipped local map, so traversals have to guard the lookup.

A log with no map, or with a `map/` directory that a partial download left without its archive
JSON, logs a warning and renders the path anyway. Only an empty *pose* stream is fatal.

### Lidar

AV2 stacks two 32-beam Velodynes with overlapping fields of view and ships the sweeps **already
egomotion-compensated into the egovehicle frame** at each sweep's reference timestamp, so
`sweep.xyz` needs no transform to sit on the car and `offset_ns` (which spans the full 106 ms
revolution) has already been applied. Sweep timestamps are a strict subset of pose timestamps —
157/157 exact hits — so placing a sweep in the city frame is a dict lookup with no interpolation.

|  | `sensor/val/02678d04…` | `tbv/07YOTz…_Spring_2020` |
|---|---|---|
| sweeps | 157 @ 10 Hz | 575 |
| points | 14,284,998 | 56,520,865 |
| on disk / as `.rrd` | 145 MB / 192 MB | 396 MB / 754 MB |
| wall clock, whole log | 1.8 s | 3.9 s |
| peak RSS while writing | 543 MB | 638 MB |

Nothing is decimated. What the **viewer** can change afterwards: toggle the entity off, override
radius and color in the Selection panel, and adjust the decay window. What it cannot do is thin
the points — the store already holds all of them — so that is the one lever that would have to
become code.

Two traps, in the order they will cost you time.

**1. The points are in the city frame, not at the sensor.** This inverts the usual rerun advice
(log in the natural frame, let transforms compose), and it is deliberate. A decay window only
draws a correct trail in a frame that does not move — the viewer resolves an entity's transform
chain once at the cursor time, while visible-time-range is a per-visualizer query, so ten sweeps
under a moving `world/ego` would all be drawn at the *current* pose and pile onto the car
instead of laying down road behind it. The second reason is better: accumulated returns that
double a wall or blur the ground are the most direct read on pose quality this viewer offers,
and that only shows up in a fixed frame. Precision is not the tradeoff — float32 at a 6700 m
city coordinate has an ulp of 0.8 mm, some 80× finer than the float16 the returns are stored in.

**2. `Sweep.from_feather` cannot read a tbv log.** TBV sweeps ship without the `offset_ns`
column — theirs are `x, y, z, intensity, laser_number` — and the devkit loader indexes it
unconditionally, so it raises `KeyError` across the whole dataset. `av2_source.lidar_sweeps`
reads the feather itself and substitutes zeros, which also hoists the sensor extrinsics out of
the loop (the devkit re-reads that file once per sweep).

Points are colored by **intensity**, clipped at 110 and run through a five-stop viridis ramp
built in numpy — rerun 0.23.1 ships no colormap, matplotlib is only a transitive dependency of
`av2`, and `av2.rendering.color.create_range_map` does not do what its name says (it colors by
`z`, rounds to integers, and indexes negatively below ground). The clip matters: intensity is
`uint8` but nowhere near uniform over it — median 30, p90 73, p99 108 — so scaling by 255 leaves
the entire cloud in the dark end. At 110 the road surface reads blue-teal and retroreflective
paint and signs land in the yellow.

Registration is checkable without eyes, and worth rerunning if any of this changes: near-ground
returns versus the map's own ground-height surface come out at a **median +0.035 m (sensor) and
+0.002 m (tbv), MAD 0.077 / 0.044 m** across 1.4 M sampled points. Note that proximity to
*painted* boundaries is a much weaker test than it sounds — the sensor log annotates only 39
painted boundaries out of 184, so that statistic mostly measures paint scarcity.

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

**Nothing blocks bumping it.** The pin is inertia, not a constraint. What staying here costs is
the `GridMap` and `VoxelGridMap` archetypes (added in 0.31 and 0.34), which would be the natural
fit for occupancy grids — until someone does the bump, a grid is `Points3D` or a batched
`Boxes3D`.

One thing to know about debugging any of it: rerun degrades the *recording* rather than crashing
the *logger*, so a component it failed to serialize and one it accepted look identical from the
outside — you get an `.rrd`, an exit code of 0, and a missing feature. When a setting appears to
be ignored, check stderr for `RerunWarning` before suspecting your own code.

The wheel also needs `extra_requirement("rerun-sdk", "rerun")` in BUILD files rather than the
usual `requirement("rerun-sdk")`; see `third_party/python/extra_rerun_targets.bzl` for why.
