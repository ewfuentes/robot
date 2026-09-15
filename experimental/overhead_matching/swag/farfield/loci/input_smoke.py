#!/usr/bin/env python3
"""Validate and load one published far-field LOCI input bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experimental.overhead_matching.swag.data.vigor_dataset import (
    VigorDataset,
    VigorDatasetConfig,
)
from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.loci import region


SCHEMA = "loci_input_smoke/v1"


def _region_upstream(manifest: artifact.ArtifactManifest,
                     label: str) -> artifact.ArtifactRef:
    regions = [
        upstream for upstream in manifest.upstreams
        if upstream.kind == "loci_regions"
    ]
    if len(regions) != 1:
        raise ValueError(
            f"{label} artifact must have exactly one loci_regions upstream")
    return regions[0]


def _region_identity(reference: artifact.ArtifactRef) -> tuple[str, ...]:
    return (
        reference.kind,
        reference.dataset,
        reference.version,
        reference.manifest_digest,
        reference.content_digest,
    )


def _region_path(satellite_artifact: Path,
                 reference: artifact.ArtifactRef) -> Path:
    """Resolve the region in the live artifact tree, not its recorded path."""
    artifacts_root = Path(satellite_artifact).resolve().parents[2]
    return artifacts_root / reference.kind / reference.dataset / reference.version


def _validate_region_contract(
        satellite_manifest: artifact.ArtifactManifest,
        osm_manifest: artifact.ArtifactManifest,
        region_ref: artifact.ArtifactRef, region_plan: dict) -> dict:
    """Require both downstream artifacts to reproduce the region contract."""
    for label, manifest in (
            ("satellite", satellite_manifest), ("OSM", osm_manifest)):
        configured_digest = manifest.config.get("region_manifest_digest")
        if configured_digest != region_ref.manifest_digest:
            raise ValueError(
                f"{label} artifact config has a different region manifest "
                "digest: "
                f"{configured_digest!r} != {region_ref.manifest_digest!r}")

    region_grid = region_plan.get("grid")
    if not isinstance(region_grid, dict):
        raise ValueError("region plan records no valid grid")
    satellite_grid = satellite_manifest.config.get("grid")
    if satellite_grid != region_grid:
        raise ValueError(
            "satellite artifact grid differs from the region plan grid")

    region_footprint = region_grid.get("footprint_bbox_wsen")
    if osm_manifest.config.get("footprint_bbox_wsen") != region_footprint:
        raise ValueError(
            "OSM artifact footprint differs from the region grid footprint")
    return region_grid


def run(dataset_dir: Path, satellite_artifact: Path,
        osm_artifact: Path) -> dict:
    """Integrity-check both artifacts and construct the real LOCI dataset."""
    satellite_ref = artifact.open_artifact(
        satellite_artifact, expected_kind="loci_satellite")
    osm_ref = artifact.open_artifact(
        osm_artifact, expected_kind="loci_osm_landmarks")
    if satellite_ref.dataset != osm_ref.dataset:
        raise ValueError(
            "satellite and OSM artifacts belong to different datasets: "
            f"{satellite_ref.dataset!r} != {osm_ref.dataset!r}")
    satellite_manifest = artifact.load_manifest(satellite_artifact)
    osm_manifest = artifact.load_manifest(osm_artifact)
    satellite_region = _region_upstream(satellite_manifest, "satellite")
    osm_region = _region_upstream(osm_manifest, "OSM")
    if _region_identity(satellite_region) != _region_identity(osm_region):
        raise ValueError(
            "satellite and OSM artifacts use different loci_regions "
            "upstreams")
    region_ref, region_plan = region.load_region(
        _region_path(satellite_artifact, satellite_region))
    if _region_identity(region_ref) != _region_identity(satellite_region):
        raise ValueError("persisted loci_regions artifact identity changed")
    region_grid = _validate_region_contract(
        satellite_manifest, osm_manifest, region_ref, region_plan)
    dataset_name = Path(dataset_dir).name
    trajectory_datasets = tuple(
        region_plan.get("trajectory", {}).get("datasets", ()))
    if dataset_name not in trajectory_datasets:
        raise ValueError(
            "dataset directory is not one of the region trajectories: "
            f"{dataset_name!r} not in {trajectory_datasets!r}")
    satellite_zoom_level = region_grid.get("zoom")
    if type(satellite_zoom_level) is not int or satellite_zoom_level <= 0:
        raise ValueError("satellite artifact records no valid grid zoom")
    expected_satellites = region_grid.get("n_patches")
    if type(expected_satellites) is not int or expected_satellites <= 0:
        raise ValueError("region grid records no valid patch count")

    satellite_dir = Path(satellite_artifact) / "satellite"
    landmark_path = Path(osm_artifact) / "landmarks.feather"
    loaded = VigorDataset(
        Path(dataset_dir),
        VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            satellite_dir=satellite_dir,
            landmark_path=landmark_path,
            satellite_zoom_level=satellite_zoom_level,
            should_load_images=False,
        ),
    )
    loaded_satellites = len(loaded._satellite_metadata)
    if loaded_satellites != expected_satellites:
        raise ValueError(
            "loaded satellite metadata count differs from the region grid: "
            f"{loaded_satellites} != {expected_satellites}")
    return {
        "schema": SCHEMA,
        "dataset": dataset_name,
        "artifact_scope": satellite_ref.dataset,
        "satellite_artifact": satellite_ref.to_dict(),
        "osm_artifact": osm_ref.to_dict(),
        "region_artifact": satellite_region.to_dict(),
        "satellite_dir": str(satellite_dir.resolve()),
        "landmark_path": str(landmark_path.resolve()),
        "satellite_zoom_level": satellite_zoom_level,
        "n_satellites": loaded_satellites,
        "n_panoramas": len(loaded._panorama_metadata),
        "n_landmarks": len(loaded._landmark_metadata),
        "n_pairs": len(loaded._pairs),
        "all_panoramas_have_satellite_associations": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--satellite_artifact", type=Path, required=True)
    parser.add_argument("--osm_artifact", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.dataset_dir,
        args.satellite_artifact,
        args.osm_artifact,
    ), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
