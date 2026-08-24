# TopoGPT

[TopoGPT](https://github.com/buaa-colalab/TopoGPT) (ECCV 2026, Apache-2.0) generates lane
topology autoregressively — centerline polylines plus their adjacency (successor, predecessor,
left, right) — conditioned on a rasterized BEV lane mask. It was pre-trained on 3.3M scenes from
nuPlan, Argoverse 2, and Waymo, which makes it the natural consumer of the lane topology
[`viz/README.md`](../viz/README.md) notes is in the AV2 data and not yet drawn.

```bash
bazel test //experimental/map_estimation/topogpt:map_prior_smoke_test
```

## Running the prior on a log

`run_prior` rebuilds one pretraining sample from a log's own HD map — the 104 × 64 m ego-frame
crop, centerlines resampled to 20 points, and the 2-channel direction raster that conditions the
model — erases a fraction of the lanes from the *conditioning raster only*, and samples the
autoregressive model. `--drop_p` is how much of the answer is withheld; the model is always scored
against the full crop, so the `ERASED -> pred` line is the one that measures the prior rather than
the copy.

```bash
R="bazel run //experimental/map_estimation/topogpt:run_prior --"

$R tbv --log_id qT9M5446NgGW5izOozHsSM9gLGyGkD1u__Summer_2020 --elapsed_s 30
$R tbv --log_id qT9M... --drop_p 1.0     # empty raster: the unconditional prior
```

It needs `pretrain.ckpt` from the [VectorMapPrior](https://huggingface.co/datasets/MatrixF/VectorMapPrior)
HF dataset (`--ckpt`, default `/data/map_estimation/topogpt_ckpts/pretrain.ckpt`).

Preprocessing calls upstream's own `process_lines` and `Lane2SegMask` rather than reimplementing
them, so a sample built here is the same object the model was trained on. Two consequences worth
knowing: TbV and `sensor` maps ship no `centerline` field, so the devkit derives one per segment
with `compute_midpoint_line`; and `process_lines` consumes only `SUCCESSOR` and `LEFT` adjacency,
so predecessors and right neighbours are dropped as mirror edges.

## Running the fine-tuned model on a log

`run_finetune` is the deployment path: seven ring images in, a lane graph out, with no part of
the map given to the model. The HD map is read only to score the result.

```bash
F="bazel run //experimental/map_estimation/topogpt:run_finetune --"

$F tbv --log_id qT9M5446NgGW5izOozHsSM9gLGyGkD1u__Summer_2020 --elapsed_s 30
$F tbv --log_id qT9M... --every_s 3 --spawn      # predict along the log and watch it
$F tbv --log_id qT9M... --every_s 3 --save out.rrd
```

`--spawn` / `--serve` / `--save` are `view_log`'s three sinks, and mean the same things. The
generated lanes land on `world/prediction` in city coordinates, so they draw in the 3D view
**and project into every camera image** — the 2D views pull world entities through their pinhole,
which makes the overlay a far better check on the result than the bird's-eye view is. Prediction
is what `--every_s` paces; `--cameras` is separate and defaults to every frame, because a
prediction persists until the next one and it is the dense imagery moving underneath it that
shows whether the lanes are holding still against the road.

Two things about the timing, both learned the hard way:

- The recording opens in **`Following`** on the `elapsed` timeline. Predictions are sparse where
  the rest of the scene is dense, and rerun holds the last value at or before the cursor, so
  opening at *t*=0 shows an empty road and reads as "nothing was drawn".
- **Imagery starts about a second after the pose stream.** Instants whose nearest frame is more
  than 50 ms away are skipped and reported rather than silently predicted from stale frames —
  before the guard, the *t*=0 prediction ran on images 1.06 s late and scored worst of the run.

It needs `ckpts/finetune.ckpt` and the log's ring imagery
(`argoverse download tbv --log_id ... --items ring`, ~700 MB per log).

No format conversion is required, because **OpenLane-V2 subset A is the Argoverse 2 `sensor`
dataset re-packaged** — same seven ring cameras, same calibration convention (`extrinsic` is
`ego_SE3_cam`, which is what `LogSource.camera_model()` returns). Three things that are not
obvious:

- **`ring_front_center` must be first.** `CropFrontViewImageForAv2` crops `img[0]` and nothing
  else; it is the one camera stored portrait.
- **TbV ships the six non-front ring cameras at half resolution** — 1024 × 775 against the
  `sensor` dataset's 2048 × 1550 — with intrinsics halved to match (fx 843 vs 1686). The subset A
  config's blanket `ScaleImageMultiViewImage(0.5)` therefore halves them a second time, feeding
  the model 512 × 388 imagery inside a mostly-empty padded canvas. `run_finetune` solves for the
  target width per camera instead, which lands `tbv` and `sensor` logs on the same geometry.
  Getting this wrong is quiet, not loud: it cost 6 of 20 lanes and 3× the position error.
- **The ring is synchronised, and each camera is still resolved separately.** Every camera runs
  at exactly 50.000 ms with no measurable jitter, each at its own fixed trigger phase — the
  phases span ~95 ms of the cycle, so no two cameras share a timestamp. Querying each camera for
  its closest frame is what `SynchronizationDB.get_closest_cam_channel_timestamp` does, so the
  seven images land within half a frame of the query instant.

The ResNet's `ckpt_path` is left `None` rather than the config's `torchvision://resnet50`, which
would fetch ImageNet weights only to overwrite them from the checkpoint a moment later.

`run_prior` also carries a one-line NumPy 2 shim. `datasets/utils/com_vec_feature.py` annotates
`process_lines`'s return with `np.float_`, removed in NumPy 2 and evaluated at import, so the
module will not load at all. It is the package's only NumPy 2 removal; the fix belongs in the
fork alongside the five below.

## The fork

Pinned in `third_party/python/requirements_3_12.in` as
`MapPrior @ git+https://github.com/ewfuentes/TopoGPT.git@<sha>`, following the same pattern as
`dinov2`, `torch_kdtree`, `sdprlayer`, and `chompack`.

**The distribution is named `MapPrior`, not `topogpt`** — pip validates the name on direct-URL
requirements, so the requirement line and `requirement("MapPrior")` must both say `MapPrior`.

Upstream is developed with `rootutils` putting its source tree on `sys.path` and its dependencies
installed by hand, which hides five defects that only appear once it is pip-installed. The fork
carries one commit fixing all of them:

| defect | fix |
|---|---|
| Six directories have no `__init__.py`, so `find_packages` drops them — including `modules/autoregression` (`gpt.py`, `vector_ar.py`) and `modules/bev_encoder`, i.e. every `_target_` in `configs/model/*.yaml` | added them |
| `configs/**.yaml` never ships. `include_package_data` + `MANIFEST.in` does *not* reach it either: `build_py` only hunts for data files inside directories it already knows are packages, and `configs/` subdirectories are not | `package_data` glob, resolved relative to the `MapPrior` directory |
| No `install_requires`, so a pip install produces a package that imports nothing | 19 names, unversioned |
| `load_pretrained_weights` predates torch 2.6 flipping `torch.load`'s `weights_only` to `True` | allowlist, not a disable — see below |
| `tools/eval/graph/topo_metrics.py` does `from geotopo import ...`, which resolves only because `apls.py` appends its own directory to `sys.path` one import earlier | both made package-relative |

### Why the declared dependencies carry no versions

Anything in `install_requires` becomes a hard constraint on a lock shared with every other project
in this repo. Upstream's own pins are not usable as a floor — `torch==2.1.0+cu118` cannot target
any GPU newer than Ada (CUDA 11.8 predates Blackwell, so it cannot run on the 5090 this was
developed against), `numpy==1.23.4` conflicts with most current scientific packages, and `numba`
was pinned but is never imported anywhere in the source. Declaring names only lets
`requirements_3_12.in` stay authoritative while `requirement("MapPrior")` still carries the
import closure transitively — which is why this package's BUILD names four dependencies instead
of sixteen, and why `requirements_3_12.in` gains a single line rather than a dozen. `lightning`,
`pyquaternion` and the rest are locked transitively; `.in` lists direct requirements only, and a
transitive-only package is still addressable as `requirement("<name>")` (this BUILD uses
`omegaconf`, which appears nowhere in `.in`).

Optional groups are extras and are **not** installed here:

- `flow` — **now installed**, as `torchcfm==1.0.7` in `requirements_3_12.in` rather than as the
  unversioned extra. `modules/flow` is reachable only as a `_target_` in
  `configs/model/finetune.yaml`, which hydra resolves at instantiation time, so no import
  analysis finds it. It costs nine lock entries (`torchcfm`, `torchdyn`, `torchsde`, `torchcde`,
  `torchdiffeq`, `pot`, `poethepoet`, `pastel`, `trampoline`) and no version changes to anything
  already pinned. Only `flow_matchers.py`'s module-scope `import torchcfm` makes it mandatory:
  `TorchFlowMatcher.sample` is a plain six-step Euler integrator that never touches the torchcfm
  object, which is used solely by `compute_loss`.
- `eval` — the APLS and topology metrics.
- `train` — `rootutils`, `wandb`.
- `datagen-av2` / `datagen-waymo` / `datagen-nuplan` — the raw-dataset devkits. Mutually
  exclusive in practice: `waymo-open-dataset` pins tensorflow and `nuplan-devkit` pins its own
  torch, so they do not co-resolve with this lock.

### The `weights_only` allowlist

Both released checkpoints store `save_hyperparameters()` output beside the tensors, and here
those are Hydra config objects. Their pickles were inspected directly — a `.ckpt` is a ZIP with
the pickle stored first and uncompressed, so a ranged HTTP request for the first few hundred KB
gets the opcodes without downloading 1.26 GB. Both reference **exactly the same 14 globals**, of
which torch permits four by default. The other ten — the five omegaconf containers,
`collections.defaultdict`, `typing.Any`, and `dict`/`list`/`int` under their Python 2 spellings —
are passed to `torch.serialization.safe_globals` rather than the check being disabled. Each is an
inert container that runs no code when constructed, whereas `weights_only=False` would honour any
`__reduce__` in a file downloaded from a third party.

The allowlist is derived from those two checkpoints. One saved by a different config could
reference something else; the failure mode is a clear `UnpicklingError` naming the missing global.

`resnet_fpn.py` is deliberately left at the default — its result goes straight into
`load_state_dict`, so that path only ever reads a bare tensor dict.

## Bumping the fork

Delete the package's stale line **and its `# via` block** from `requirements_3_12.txt` first. uv
feeds the existing lock back as its own constraint, so a changed git URL fails with
`Requirements contain conflicting URLs for package mapprior`. This is the one legitimate reason
to hand-edit the lock. Then run the update to a fixpoint — it needs several passes and exits 0
even on failure, so judge it by the diff:

```bash
bazel run //third_party/python:requirements_3_12.update   # repeat until the .txt stops changing
```

## Next

- **AV2 → `MapPrior` pkl converter**, written over `data/av2_log.py`'s `LogSource` rather than
  upstream's `datasets/argoverse/av2_map_gen.py`. That script calls the `get_scenario_*` APIs and
  samples ego poses along a trajectory at 5 m intervals, i.e. it targets the **motion-forecasting**
  split, which `av2_log.ensure_supported()` rejects and the viewer cannot render. Every devkit
  method it needs exists on the same `ArgoverseStaticMap` that `LogSource.static_map()` already
  returns, so a converter over `LogSource` works on the `sensor`/`tbv` logs already on disk.
  Target schema, ego-centric, adjacency types `0=successor, 1=predecessor, 2=left, 3=right`:

  ```
  {map_location, ego_pose, map_elem: {CENTERLINE: float[N, 20, 3]}, vec_adj: int[E, 3]}
  ```

  Two traps `viz/README.md` records: AV2's `predecessors` is not the mirror of `successors`
  (113 vs 52 edges in a `sensor/val` log), and ~13 successor ids per log point at lanes outside
  the clipped local map, so traversals must guard the lookup.
- **`np.float_` fix in the fork**, retiring `run_prior`'s shim.
- **Viewer overlay for `run_finetune` output**, under the reserved `PREDICTION_EGO`.
- **Viewer overlay** under `av2_scene.PREDICTION_CITY` (`world/prediction`), already reserved.
- **Training** — needs `rootutils.setup_root` reconciled with Bazel runfiles: it walks up from
  `__file__` looking for a `.project-root` marker, which cannot work from `site-packages`.
