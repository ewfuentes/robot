import json
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
from PIL import Image
from shapely.geometry import Point

from experimental.overhead_matching.swag.data import vigor_dataset


def _write_compact_landmarks(path: Path) -> None:
    frame = gpd.GeoDataFrame(
        {
            "id": ["node:1"],
            "landmark_type": ["osm"],
            "tags": [json.dumps({
                "amenity": "school",
                "source": "survey",
            }, sort_keys=True, separators=(",", ":"))],
        },
        geometry=[Point(-71.09, 42.35)],
        crs="EPSG:4326",
    )
    frame.to_feather(path)


class VigorDatasetExternalInputsTest(unittest.TestCase):
    def test_compact_tags_are_decoded_before_loci_pruning(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "landmarks.feather"
            _write_compact_landmarks(path)

            loaded = vigor_dataset.load_landmark_geojson(path, 20)

            self.assertEqual(
                loaded.iloc[0]["pruned_props"],
                frozenset({("amenity", "school")}))
            self.assertIn("geometry_px", loaded.columns)

    def test_compact_catalog_requires_wgs84_coordinates(self):
        for crs, message in ((None, "CRS"), ("EPSG:3857", "EPSG:4326")):
            with self.subTest(crs=crs), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "landmarks.feather"
                frame = gpd.GeoDataFrame(
                    {
                        "id": ["node:1"],
                        "landmark_type": ["osm"],
                        "tags": [json.dumps({"amenity": "school"})],
                    },
                    geometry=[Point(0, 0)],
                    crs=crs,
                )
                frame.to_feather(path)

                with self.assertRaisesRegex(ValueError, message):
                    vigor_dataset.load_landmark_geojson(path, 20)

    def test_external_satellite_and_landmark_paths_override_vigor_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir = root / "dataset"
            panorama_dir = dataset_dir / "panorama"
            satellite_dir = root / "satellite_artifact" / "satellite"
            panorama_dir.mkdir(parents=True)
            satellite_dir.mkdir(parents=True)
            Image.new("RGB", (32, 32)).save(
                panorama_dir / "p0,42.350000,-71.090000,.png")
            Image.new("RGB", (32, 32)).save(
                satellite_dir / "satellite_42.35000000_-71.09000000.png")
            landmark_path = root / "osm_artifact" / "landmarks.feather"
            landmark_path.parent.mkdir()
            _write_compact_landmarks(landmark_path)

            config = vigor_dataset.VigorDatasetConfig(
                satellite_tensor_cache_info=None,
                panorama_tensor_cache_info=None,
                satellite_dir=satellite_dir,
                landmark_path=landmark_path,
                should_load_images=False,
            )
            loaded = vigor_dataset.VigorDataset(dataset_dir, config)

            self.assertEqual(
                loaded._satellite_metadata.iloc[0].path.parent,
                satellite_dir)
            self.assertEqual(
                loaded._landmark_metadata.iloc[0]["pruned_props"],
                frozenset({("amenity", "school")}))
            self.assertEqual(
                loaded._satellite_metadata.iloc[0]["landmark_idxs"], [0])

    def test_external_overrides_reject_ambiguous_multi_dataset_input(self):
        config = vigor_dataset.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            satellite_dir=Path("/external/satellite"),
        )
        with self.assertRaisesRegex(ValueError, "exactly one dataset_path"):
            vigor_dataset.VigorDataset(
                [Path("first"), Path("second")], config)

    def test_external_overrides_reject_unbound_tensor_cache(self):
        config = vigor_dataset.VigorDatasetConfig(
            satellite_tensor_cache_info=object(),
            panorama_tensor_cache_info=None,
            satellite_dir=Path("/external/satellite"),
        )
        with self.assertRaisesRegex(ValueError, "tensor caches to be disabled"):
            vigor_dataset.VigorDataset(Path("dataset"), config)


if __name__ == "__main__":
    unittest.main()
