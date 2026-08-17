# TopoGPT

[TopoGPT](https://github.com/buaa-colalab/TopoGPT) (ECCV 2026, Apache-2.0) generates lane
topology autoregressively — centerline polylines plus their adjacency (successor, predecessor,
left, right) — conditioned on a rasterized BEV lane mask. It was pre-trained on 3.3M scenes from
nuPlan, Argoverse 2, and Waymo, which makes it the natural consumer of the lane topology
[`viz/README.md`](../viz/README.md) notes is in the AV2 data and not yet drawn.

**This package is the vendoring step only.** It makes the upstream `MapPrior` package importable
from Bazel and proves the model classes construct. There is no data conversion, no checkpoint
inference, and no viewer overlay yet — see *Next* below.

```bash
bazel test //experimental/map_estimation/topogpt:map_prior_smoke_test
```

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

- `flow` (`torchcfm`) — `modules/flow` is reachable only as a `_target_` in
  `configs/model/finetune.yaml`, which hydra resolves at instantiation time, so no import
  analysis finds it. Left out because `torchcfm` pulls `torchdyn`, `torchsde`, `torchcde`,
  `torchdiffeq`, `pot` and `poethepoet` behind it, none of which the pretrain path touches.
  **Add it if you go the OpenLane-V2 finetune route.**
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

- **AV2 → `MapPrior` pkl converter**, written over `viz/av2_source.py`'s `LogSource` rather than
  upstream's `datasets/argoverse/av2_map_gen.py`. That script calls the `get_scenario_*` APIs and
  samples ego poses along a trajectory at 5 m intervals, i.e. it targets the **motion-forecasting**
  split, which `av2_source.ensure_supported()` rejects and the viewer cannot render. Every devkit
  method it needs exists on the same `ArgoverseStaticMap` that `LogSource.static_map()` already
  returns, so a converter over `LogSource` works on the `sensor`/`tbv` logs already on disk.
  Target schema, ego-centric, adjacency types `0=successor, 1=predecessor, 2=left, 3=right`:

  ```
  {map_location, ego_pose, map_elem: {CENTERLINE: float[N, 20, 3]}, vec_adj: int[E, 3]}
  ```

  Two traps `viz/README.md` records: AV2's `predecessors` is not the mirror of `successors`
  (113 vs 52 edges in a `sensor/val` log), and ~13 successor ids per log point at lanes outside
  the clipped local map, so traversals must guard the lookup.
- **Inference** — `pretrain.ckpt` / `finetune.ckpt` from the
  [VectorMapPrior](https://huggingface.co/datasets/MatrixF/VectorMapPrior) HF dataset. The
  conditioning is a rasterized BEV lane mask, so a mask-rendering step sits between the converter
  and the model.
- **Viewer overlay** under `av2_scene.PREDICTION_CITY` (`world/prediction`), already reserved.
- **Training** — needs `rootutils.setup_root` reconciled with Bazel runfiles: it walks up from
  `__file__` looking for a `.project-root` marker, which cannot work from `site-packages`.
