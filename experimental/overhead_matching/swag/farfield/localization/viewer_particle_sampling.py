"""On-demand particle samples for localization viewers.

Static viewers inline a small weighted sample at every checkpoint so scrubbing
never waits on a server.  A user who asks for 10--100% of the complete particle
population needs a different boundary: read only the selected checkpoint and
return that percentage as a deterministic subset without replacement.  This
keeps 10% of a 50k run to 5k distinct points, makes 100% mean every recorded
particle exactly once, and avoids making every generated HTML page carry all
7.3 million checkpoint particles.
"""

from __future__ import annotations

import math
import stat
from pathlib import Path
from typing import Mapping

import numpy as np

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.localization import run_io


PARTICLE_PERCENTAGES = (10, 20, 30, 50, 100)
_REQUIRED_ARRAYS = frozenset({"east_m", "north_m", "mode_id"})


class ParticleSamplingError(ValueError):
    pass


def weighted_sample(log_weight: np.ndarray, count: int,
                    rng: np.random.Generator) -> np.ndarray:
    """A bounded systematic draw from a weighted posterior."""
    n = log_weight.shape[0]
    if n <= count:
        return np.arange(n)
    weights = np.exp(log_weight - log_weight.max())
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        return rng.choice(n, size=count, replace=False)
    positions = (rng.random() + np.arange(count)) / count
    return np.searchsorted(np.cumsum(weights / total), positions)


def _percentage(value: int) -> int:
    if type(value) is not int or value not in PARTICLE_PERCENTAGES:
        raise ParticleSamplingError(
            f"particle percentage must be one of {PARTICLE_PERCENTAGES}")
    return value


def _validated_arrays(arrays: Mapping[str, np.ndarray]
                      ) -> tuple[dict[str, np.ndarray], int]:
    missing = _REQUIRED_ARRAYS - set(arrays)
    if missing:
        raise ParticleSamplingError(
            f"checkpoint lacks particle arrays {sorted(missing)}")
    selected = {name: np.asarray(arrays[name]) for name in _REQUIRED_ARRAYS}
    n = selected["east_m"].shape[0]
    if n <= 0:
        raise ParticleSamplingError("checkpoint has no particles")
    for name, array in selected.items():
        if array.ndim != 1 or array.shape != (n,):
            raise ParticleSamplingError(
                f"checkpoint array {name} must have shape ({n},)")
    for name in ("east_m", "north_m"):
        array = selected[name]
        if array.dtype.kind != "f" or not np.isfinite(array).all():
            raise ParticleSamplingError(
                f"checkpoint array {name} must be finite floating point")
    if selected["mode_id"].dtype.kind not in "iu":
        raise ParticleSamplingError(
            "checkpoint array mode_id must be integer")
    return selected, n


def payload_from_arrays(arrays: Mapping[str, np.ndarray], *,
                        keyframe_idx: int, percent: int) -> dict:
    """Return a deterministic distinct subset sized against the full set."""
    percent = _percentage(percent)
    arrays, total = _validated_arrays(arrays)
    count = max(1, math.ceil(total * percent / 100))

    if count == total:
        index = np.arange(total)
    else:
        # A single seeded permutation makes repeat requests byte-stable and
        # keeps each smaller percentage nested inside every larger choice.
        rng = np.random.default_rng(int(keyframe_idx) & ((1 << 64) - 1))
        index = rng.permutation(total)[:count]

    return {
        "kf": int(keyframe_idx),
        "percent": percent,
        "n": count,
        "total": total,
        "sampling": "all" if count == total else "without_replacement",
        "e": [round(float(value), 1) for value in arrays["east_m"][index]],
        "n_m": [round(float(value), 1)
                for value in arrays["north_m"][index]],
        "mode": [int(value) for value in arrays["mode_id"][index]],
    }


def checkpoint_payload(run_dir: Path, *, keyframe_idx: int,
                       percent: int) -> dict:
    """Read and sample one declared checkpoint without loading the whole run."""
    run_dir = Path(run_dir)
    manifest = artifact.load_manifest(run_dir)
    if manifest.kind != run_io.RUN_KIND:
        raise ParticleSamplingError(
            f"expected {run_io.RUN_KIND}, found {manifest.kind}")
    relative = f"checkpoints/kf_{keyframe_idx:05d}.npz"
    if relative not in manifest.declared_outputs:
        raise FileNotFoundError(
            f"run has no checkpoint at keyframe {keyframe_idx}")
    checkpoint = run_dir / relative
    try:
        metadata = checkpoint.lstat()
    except OSError as error:
        raise FileNotFoundError(
            f"run has no checkpoint at keyframe {keyframe_idx}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ParticleSamplingError(
            f"checkpoint is not a regular file: {checkpoint}")
    try:
        with np.load(checkpoint, allow_pickle=False) as source:
            arrays = {name: source[name] for name in _REQUIRED_ARRAYS}
    except (OSError, ValueError, KeyError) as error:
        raise ParticleSamplingError(
            f"cannot read checkpoint {keyframe_idx}: {error}") from error
    return payload_from_arrays(
        arrays, keyframe_idx=keyframe_idx, percent=percent)
