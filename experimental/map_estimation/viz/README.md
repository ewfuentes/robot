# Argoverse 2 viewer

Renders an AV2 log in [rerun](https://rerun.io). Right now that means the **HD map, the
egovehicle, the path it drove, the lidar, and the camera imagery**; annotations and model output
are the next layers on top.

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
| `av2_scene.py` | turns a source into rerun entities. Owns the entity paths, the map layers, the lidar coloring, the camera grid, and the vehicle outline. |
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
world/ego                  city_SE3_egovehicle, one per pose timestamp (~170 Hz)
world/ego/wireframe        the vehicle outline, logged ONCE as static data
world/ego/cameras/<name>   ego_SE3_cam + Pinhole, static; one EncodedImage per frame (20 Hz)
```

Everything above `world/ego` is in city coordinates. The map and the path are also *static* —
no timestamps, valid wherever you scrub; the lidar, the imagery, and the ego transform are the
timestamped streams. The wireframe is logged once and moves because its parent moves. Nothing
re-logs geometry per frame. That is the habit to carry into everything added later.

**The map, the path, and the lidar are siblings of the ego, not children.** That is not
stylistic: a child inherits its parent's transform, so hanging the map under `world/ego` would
make the road drive along with the car. It would not error — it would just render physics that
is quietly wrong, which is the characteristic failure mode of this data model.

The cameras are the counterexample, and the only place two transforms compose: a static
`ego_SE3_cam` under a per-timestamp ego pose. Nothing in the camera code reads a pose — an image
is placed entirely by its parent.

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

### Cameras

Every camera present is logged; there is no flag and nothing is dropped. A `tbv` log has seven
ring cameras at 20 Hz, a `sensor` log has nine (seven ring plus two stereo), and the `lidar`
dataset declares none, so `log_cameras` returning `(0, 0)` is ordinary and draws no warning.

|  | one camera | all 7 (`tbv`) |
|---|---|---|
| frames | ~1150 @ 20 Hz | 8048 |
| jpegs on disk | 82–95 MB | 611 MB |
| added to the `.rrd` | ~88 MB | 654 MB |
| added wall clock, warm | 0.08 s | **0.8 s** |
| added wall clock, cold cache | 0.8 s | **5.4 s** |
| added peak RSS while writing | — | **none measurable** |

**The imagery is nearly free to log and expensive to hold.** `EncodedImage(path=...)` reads the
jpeg and hands the bytes to Arrow; nothing is ever decoded on the logging side, so seven cameras
add under a second and no measurable memory to a process already peaking at 638 MB for the
lidar. What they do add is 654 MB of recording, taking the `tbv` log to **1408 MB on disk and
~1.9 GB resident** once a viewer holds it. If that ceiling ever bites, a `--cameras` selector is
the obvious lever and was deliberately left out until it does.

For the whole pipeline including cameras: **`tbv` 4.7 s warm / 12.3 s cold, `sensor/val` 1.8 s /
2.6 s**, measured against the binary directly rather than through `bazel run`.

**Why not video.** rerun 0.23.1 has `AssetVideo`, and transcoding the jpegs to H.264 is 4×
smaller (154 MB for all seven) and would play back more smoothly. It is not used because the
native viewer **decodes H.264 by shelling out to an `ffmpeg` binary it looks up on `$PATH`** —
`ffmpeg-sidecar` is linked into the viewer, and there is a `video_decoder_ffmpeg_path` setting
for when the lookup fails. Having already had to work around Bazel not putting the bundled
viewer on `$PATH`, buying a second and heavier `$PATH` dependency in exchange for file size is a
bad trade: without a system ffmpeg the video does not play at all, whereas jpegs always do. If
someone revisits it, one trap is already known — AV2 ring images are **1024×775**, H.264 4:2:0
rejects odd heights, and ffmpeg fails with `height not divisible by 2` unless you pass
`-vf pad=ceil(iw/2)*2:ceil(ih/2)*2`.

**The images are undistorted, so `Pinhole` is exact.** `intrinsics.feather` carries `k1`, `k2`,
`k3` (≈ −0.26, −0.12, 0.20 for the ring cameras) and they are ignored here — because the devkit
ignores them too. Grepping the whole installed `av2` package for `distort|rectif|fisheye`
returns nothing at all; `project_ego_to_img` multiplies by bare `K`. Were the released imagery
actually distorted, those coefficients would put a corner pixel ~15% of the radius out of place,
so this is worth knowing rather than assuming.

Two more facts that cost nothing to record:

- **The AV2 camera frame is RDF** — x right, y down, z forward, same as OpenCV. Readable
  straight off `ego_SE3_cam`, whose rotation columns are ego −y, −z, +x. That is also rerun's
  default, but `camera_xyz` is passed explicitly anyway: left unset, no component is logged and
  the orientation quietly follows whatever the viewer defaults to.
- **Every camera timestamp is a pose timestamp** — 8048/8048 across all seven `tbv` cameras,
  same as the lidar's 157/157.

`tbv` ships six of its seven cameras at **1024×775**, half the `sensor` dataset's resolution,
with `ring_front_center` alone at **1550×2048** portrait. Nothing here depends on that, but a
model consuming these frames does.

### The view follows the vehicle

The default 3D view is anchored to `world/ego`, so the car holds still and the city sweeps past
it. A view's **origin** is the frame it renders in, and that is the entire mechanism — rerun has
no separate follow toggle.

Beside it, when the log has imagery, is a **grid of one 2D view per camera**, ordered front →
side → rear so the layout reads like where the cameras point (sorting the names would file the
rear pair between the front and the side ones). It is three columns wide, which leaves the
seventh ring camera alone on the last row — the price of not being able to leave a hole in the
middle. A log with no cameras gets the bare 3D view rather than an empty pane, which is why
`default_blueprint` takes the source: deciding *which* cameras exist is a directory check on an
already-built `LogSource`, cheap enough to stay ahead of the sink and of any logging.

**The world goes into the cameras, not the cameras into the world.** Each 2D view reaches back
out to `world/lidar` and `world/map/**`, and the pinhole projects them onto the image; the
cameras are excluded from the 3D view entirely, frustums and all.

```python
rrb.Spatial2DView(origin=f"{CAMERAS}/{name}",
                  contents=["$origin/**", LIDAR, f"{MAP}/**"])   # lidar + map, projected
rrb.Spatial3DView(origin=EGO, contents=["/**", f"- {CAMERAS}/**", ...])  # no frustums
```

This is the useful direction, and it is nearly free — **nothing is re-logged and nothing is
projected by hand.** `world/lidar` is the same city-frame cloud the 3D view draws, seen through
`city_SE3_ego @ ego_SE3_cam`. It is also a far better check than a frustum ever was: returns
that miss the objects they came from, or lane paint that floats off the road, is a calibration
or pose error you can see without measuring anything.

The rule that makes it work is worth stating exactly, because it is easy to get backwards:

- **A 2D view anchored at a pinhole projects any 3D entity in its `contents`, including entities
  from a completely different branch of the tree.** Siblings are fine. Verified by rendering.
- **`Points3D` logged as a *child* of the camera renders nothing.** A pinhole splits the tree:
  above it is 3D, at and below it is 2D. Transforming a cloud into camera coordinates and
  logging it under the camera is the obvious-looking move and it silently draws nothing.
- Hand-projecting to `Points2D` works, and is redundant.

The exclusion in the 3D view has to name **both** the subtree and each entity: `- .../cameras/**`
does not cover `.../cameras/ring_front_left` itself, which is where the `Pinhole` and the images
actually live.

`world/path` is deliberately left out of the camera views — it runs through the car, so it would
smear a bright line across the bottom of every frame. The ego frame's axes are 0.5 m, short
enough not to dominate a scene at vehicle scale.

Its **contents** are `/**` rather than the default `$origin/**`, and that override is load
bearing. `$origin/**` under `world/ego` is the wireframe and the cameras: `world/path` is a
*sibling* of the ego, not a descendant. That placement is deliberate — the path is in city coordinates and
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
`LogSource.present_items()` rather than assuming. The natural next entity is

```
world/ego/annotations      ground-truth cuboids; AV2 stores these in the ego frame too
```

which makes it a `Boxes3D` child of `world/ego`, like the cameras and unlike the map.

Nothing on disk exercises the `sensor` dataset's nine-camera layout — the two stereo cameras and
the full 2048×1550 resolution — because none of the `sensor/val` logs here downloaded their
`cameras/` directory. The code paths are shared with the ring cameras, but that is an argument,
not a test.

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
