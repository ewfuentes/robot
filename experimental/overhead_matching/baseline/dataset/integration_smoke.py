"""Smoke check that VigorDataset loads OSM tiles via satellite_subdir.

Not a unit test — talks to the real dataset on disk. Run after baking + rendering
a city. Confirms (1) loader picks up satellite_osm/, (2) tile counts match, (3)
__getitem__ returns the expected shape.
"""
import argparse
import sys
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch  # noqa: F401

from experimental.overhead_matching.swag.data.vigor_dataset import (
    VigorDataset,
    VigorDatasetConfig,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--city-dir", type=Path,
        default=Path("/data/overhead_matching/datasets/VIGOR/Seattle"),
    )
    args = p.parse_args()

    base_cfg = dict(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=True,
        should_load_landmarks=False,
    )

    print(f"Loading {args.city_dir} with satellite_subdir='satellite_osm'...")
    ds_osm = VigorDataset(
        args.city_dir,
        VigorDatasetConfig(satellite_subdir="satellite_osm", **base_cfg),
    )
    print(f"  {len(ds_osm)} samples")

    print(f"Loading {args.city_dir} with satellite_subdir='satellite' (baseline check)...")
    ds_sat = VigorDataset(
        args.city_dir,
        VigorDatasetConfig(satellite_subdir="satellite", **base_cfg),
    )
    print(f"  {len(ds_sat)} samples")

    assert len(ds_osm) == len(ds_sat), (
        f"length mismatch: osm={len(ds_osm)} satellite={len(ds_sat)}"
    )

    item = ds_osm[0]
    sat_tensor = item.satellite
    print(f"satellite tensor shape: {tuple(sat_tensor.shape)} dtype={sat_tensor.dtype}")
    assert sat_tensor.shape[-3] == 3, f"expected 3 channels, got {sat_tensor.shape}"
    assert sat_tensor.shape[-2] == 640 and sat_tensor.shape[-1] == 640, (
        f"expected 640x640, got {sat_tensor.shape}"
    )

    print("OK: VigorDataset loads OSM tiles via satellite_subdir='satellite_osm'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
