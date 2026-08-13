# Argoverse 2 data

Tooling to list, inspect, and selectively download the [Argoverse 2](https://argoverse.github.io/user-guide/)
datasets from the public `s3://argoverse` bucket, via the `s5cmd` binary.

Data lands under `/data/map_estimation/datasets/argoverse`, in a tree that **mirrors S3
exactly** — so `<root>/sensor/val` is directly consumable by the upstream `av2` loaders, and the
S3 URI of any local file is a pure string substitution of the root.

## Why selective download matters

One `sensor` log is **1.1 GB**, and 85% of that is camera imagery:

| item | size per log | share |
|---|---|---|
| 9 cameras | 941 MB | 85% |
| one ring camera | 91 MB | 8% |
| lidar | 144 MB | 13% |
| map + calibration + poses + annotations | **2.2 MB** | **0.2%** |

The full sensor dataset is ~1.1 TB; the full `lidar` dataset is ~5.8 TB and does **not** fit on
`/data`. So the default item selection is metadata only, and every command reports what it would
transfer before doing it.

## Setup

`s5cmd` comes from `setup.sh`. Check it is on PATH:

```bash
command -v s5cmd || ./setup.sh
```

The bucket is public, so no AWS credentials are needed — but it lives in `us-east-1` and the
tool pins that region explicitly, because an ambient `AWS_REGION` pointing elsewhere makes every
request fail with `BucketRegionError`.

## The datasets

The four datasets have genuinely different contents, and the tool models that difference rather
than papering over it — each has its own item enum, so a camera cannot be requested from the
`lidar` dataset at all.

| dataset | splits | logs | annotations | lidar | cameras | total size |
|---|---|---|---|---|---|---|
| `sensor` | train/val/test | 700 / 150 / 150 | yes, except test | yes | 9 (7 ring + 2 stereo) | ~1.1 TB |
| `tbv` | none (flat) | 1043 | no | yes | 7 (ring only) | ~900 GB |
| `lidar` | train/val/test | 16000 / 2000 / 2000 | no | yes | none | ~5.8 TB |
| `motion-forecasting` | train/val/test | 199908 / 24988 / 24984 | n/a | no | none | ~60 GB |

`tbv` (Trust But Verify) is the map-change dataset: the same logs revisited across seasons, keyed
`<log_id>__<Season>_<Year>`.

## CLI

```bash
A="bazel run //experimental/map_estimation/data:argoverse --"
```

### Index a dataset once

Nothing about a log — not even which city it is in — is discoverable without listing S3, because
the city appears only inside map filenames. So the first step is building a cached catalog of
per-log, per-item sizes. It lands in `~/.cache/robot/map_estimation/argoverse/` and everything
afterwards is offline.

```bash
$A index sensor/val     # 150 logs, ~3 s
$A index tbv            # 1043 logs, ~33 s
$A index lidar/val      # 2000 logs, ~40 s
```

Motion-forecasting is too large to list exhaustively (250k scenarios), so its catalog records ids
and **extrapolates** sizes from a 24-scenario sample. Output that relies on those numbers says so.

### Explore

```bash
$A list sensor/val --city PIT --limit 5
$A list sensor/val --sort bytes --limit 3        # biggest logs first
$A list sensor/val --local                       # add a LOCAL column (scans the disk)
$A show sensor/val 02678d04-cc9f-3148-9f95-1ba66347dff9
```

```
LOG_ID                               | CITY | SWEEPS | CAMERA_MB | LIDAR_MB | TOTAL_MB
--------------------------------------------------------------------------------------
02678d04-cc9f-3148-9f95-1ba66347dff9 | PIT  |    157 |     941.4 |    144.3 |   1087.9
04994d08-156c-3018-9717-ba0e29be8153 | PIT  |    156 |     767.9 |    153.1 |    923.6
```

`--json` on any subcommand gives machine-readable output.

### Download

`--items` accepts item names and the groups `metadata` / `cameras` / `ring` / `stereo` /
`sensors` / `all`, validated against the dataset in the spec. Default is `metadata`.

```bash
# metadata for the whole val split: 339 MB
$A download sensor/val

# lidar + one camera for a curated list of logs, previewed first
$A download sensor/train --items metadata,lidar,ring_front_center \
    --log_id_file my_logs.txt --dry_run

# everything for one log
$A download sensor/val --log_id 02678d04-cc9f-3148-9f95-1ba66347dff9 --items all

# the exact s5cmd lines, without running them
$A download sensor/val --limit 1 --items lidar --print_commands
```

Downloads are **idempotent** — re-running the same command transfers nothing:

```
plan: sensor/val -> /data/map_estimation/datasets/argoverse/sensor/val
  items: map, calibration, poses, annotations
  0 logs, 0 objects, 0 B in 0 transfers
  skipped: 8 already complete
nothing to download; everything requested is already present.
```

An interrupted download resumes: `status` reports what is partial, and re-running the same
command fetches only the missing files (`s5cmd cp -n` under the hood).

```bash
$A status sensor/val --items metadata,lidar
$A status sensor/val --detail            # one row per log
```

Guardrails: `--confirm_above` (default 10 GB) prompts before large transfers, and a plan that
would not fit in the destination filesystem is refused outright — pass `--ignore_free_space` to
override.

## Library

The library is the primary interface; the CLI is a thin shell over it. `ensure_logs` validates
the request, checks what is already on disk, downloads only the remainder, and returns local
paths:

```python
from experimental.map_estimation.data import argoverse_download as ad
from experimental.map_estimation.data import argoverse_layout as al

logs = ad.ensure_logs(al.SensorRequest(
    split=al.SensorSplit.VAL,
    items=(al.SensorItem.MAP, al.SensorItem.POSES, al.SensorItem.LIDAR),
    log_ids=("02678d04-cc9f-3148-9f95-1ba66347dff9",),
))

sweeps = sorted(logs[0].item_path(al.SensorItem.LIDAR).glob("*.feather"))
```

Calling it again is free — one catalog load plus one directory scan, no network. Pass
`download=False` to turn it into an assertion instead, which raises `MissingDataError` naming
what is absent and the command that would fetch it. Use that in code that must never silently
start a multi-gigabyte transfer.

### Invalid requests are unrepresentable

Each dataset has its own item enum and its own request type, so mistakes fail at the call site
rather than after a long listing:

```python
al.LidarItem.RING_FRONT_CENTER    # AttributeError: the lidar dataset has no cameras
al.TbvRequest(split=...)          # TypeError: TBV has no splits, so no split field exists
al.LidarRequest(split=al.LidarSplit.VAL, items=(al.SensorItem.RING_FRONT_CENTER,))
                                  # UnknownItemError, for the dynamic case
```

The CLI mirrors this when it converts strings to enums:

```
$ argoverse -- download lidar/val --items ring_front_center
error: 'ring_front_center' is not a valid item for the lidar dataset.
       valid items: map, calibration, poses, lidar
```

The one constraint the type system cannot carry is annotations on `sensor/test` (it depends on
the split, not the dataset), so that is a runtime check — and the *default* selection is
split-aware, so a bare `download sensor/test` just works.

## Module layout

| module | role |
|---|---|
| `s5cmd.py` | subprocess wrapper; owns region pinning and batch execution. stdlib + msgspec only |
| `argoverse_layout.py` | per-dataset enums, request types, path builders. Pure, no I/O |
| `argoverse_catalog.py` | builds/loads the cached index. The only module that lists S3 |
| `argoverse_download.py` | `ensure_logs`, plan/execute, local status. Offline except for transfers |
| `argoverse_cli.py` | argparse subcommands (`py_binary` target `argoverse`) |

```bash
bazel test //experimental/map_estimation/data/...   # all offline; no network, no credentials
```

## Known issue: the `av2` package and opencv

`av2==0.3.6` is in `third_party/python/requirements_3_12.in` for downstream dataloading, but
importing `av2.utils.io` (and therefore `av2.datasets.sensor.av2_sensor_dataloader`) currently
fails:

```
ModuleNotFoundError: No module named 'cv2.typing'
```

`av2` requires an unpinned `opencv-python`, while this repo pins `opencv-python==4.7.0.72`;
`cv2.typing` only exists from 4.8 onward. This tooling does **not** depend on `av2` — it needs
only the standard library and msgspec — so downloading is unaffected, and the downloaded files
are readable directly with pyarrow:

```python
import pyarrow.feather
poses = pyarrow.feather.read_table(log.item_path(al.SensorItem.POSES))
sweep = pyarrow.feather.read_table(next(log.item_path(al.SensorItem.LIDAR).glob("*.feather")))
```

Using the `av2` loaders will require resolving that opencv pin first.
