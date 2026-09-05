"""Write canonical far-field position-mass summaries for a LOCI run."""

import argparse
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.localization import metrics


def _vector(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"{path} must contain a one-dimensional tensor")
    return value.to(torch.float64)


def _canonical_masses(path: Path) -> dict[float, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a radius-to-mass mapping")

    required = metrics.DEFAULT_POSITION_MASS_RADII_M
    found = {}
    for raw_radius, raw_masses in value.items():
        try:
            radius = float(raw_radius)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path} contains non-numeric radius {raw_radius!r}"
            ) from error
        if radius not in required:
            continue
        if radius in found:
            raise ValueError(f"{path} contains duplicate {radius:g} m radii")
        if not isinstance(raw_masses, torch.Tensor) or raw_masses.ndim != 1:
            raise ValueError(
                f"{path} masses at {radius:g} m must be a one-dimensional tensor"
            )
        found[radius] = raw_masses.to(torch.float64)

    missing = [radius for radius in required if radius not in found]
    if missing:
        raise ValueError(
            f"{path} is missing canonical radius/radii: "
            + ", ".join(f"{radius:g} m" for radius in missing)
        )
    return found


def summarize_path(path_dir: Path) -> dict:
    """Score one numeric LOCI path directory and write its metrics.json."""
    distance_path = path_dir / "distance_traveled_m.pt"
    mass_path = path_dir / "prob_mass_by_radius.pt"
    distances = _vector(distance_path)
    masses = _canonical_masses(mass_path)

    if distances.numel() == 0:
        raise ValueError(f"{distance_path} must contain at least one keyframe")
    if not torch.isfinite(distances).all():
        raise ValueError(f"{distance_path} must contain finite distances")
    if distances.numel() > 1 and torch.any(distances[1:] < distances[:-1]):
        raise ValueError(f"{distance_path} must be nondecreasing")
    for radius, series in masses.items():
        if series.numel() != distances.numel() + 1:
            raise ValueError(
                f"{mass_path} masses at {radius:g} m have length {series.numel()}; "
                f"expected {distances.numel() + 1} (initial prior plus keyframes)"
            )
        if (not torch.isfinite(series).all()
                or torch.any(series < 0.0) or torch.any(series > 1.0)):
            raise ValueError(
                f"{mass_path} masses at {radius:g} m must be finite probabilities")
    if torch.any(masses[500.0] + 1e-6 < masses[100.0]):
        raise ValueError(
            f"{mass_path} has 500 m mass below 100 m mass at one or more steps")

    config = metrics.position_mass_metric_config()
    keys = {
        radius: metrics.position_mass_metric_key(config, radius)
        for radius in config.radii_m
    }
    # LOCI stores an initial-prior mass before the per-panorama posteriors, so
    # index + 1 deliberately excludes that prior.  The canonical metric API
    # accepts 2-D truth poses; placing the recorded cumulative distance on one
    # axis preserves its integration coordinate exactly without inventing a
    # second LOCI metric implementation.
    health = [
        SimpleNamespace(
            keyframe_idx=index,
            position_probability_mass={
                keys[radius]: float(masses[radius][index + 1])
                for radius in config.radii_m
            },
        )
        for index in range(distances.numel())
    ]
    origin = float(distances[0])
    truth = [
        SimpleNamespace(
            keyframe_idx=index,
            east_m=float(distance) - origin,
            north_m=0.0,
        )
        for index, distance in enumerate(distances)
    ]

    summary = metrics.position_mass_summary(health, truth, config)
    output_path = path_dir / metrics.POSITION_MASS_SUMMARY_NAME
    artifact.atomic_create_json(output_path, summary)
    return summary


def summarize_run(run_dir: Path) -> list[Path]:
    """Score every numeric path directory directly below ``run_dir``."""
    if (run_dir / artifact.MANIFEST_NAME).exists():
        raise ValueError(f"refusing to modify published artifact {run_dir}")
    path_dirs = sorted(
        (
            path
            for path in run_dir.iterdir()
            if path.is_dir() and len(path.name) == 7 and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )
    if not path_dirs:
        raise ValueError(f"{run_dir} contains no numeric path directories")
    for path_dir in path_dirs:
        summarize_path(path_dir)
    return path_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    for path_dir in summarize_run(args.run_dir.expanduser()):
        print(path_dir / metrics.POSITION_MASS_SUMMARY_NAME)


if __name__ == "__main__":
    main()
