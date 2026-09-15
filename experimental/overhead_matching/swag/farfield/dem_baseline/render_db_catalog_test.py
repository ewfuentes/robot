"""Tests for catalog-bound database rendering."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experimental.overhead_matching.swag.farfield.dem_baseline import (
    render_db,
    terrain,
)


class RenderDbCatalogTest(unittest.TestCase):

    def test_catalog_support_projects_corners_and_hashes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            payload = b'{"config":{"region_bbox_wsen":[-71.2,42.2,-71.0,42.4]}}\n'
            path.write_bytes(payload)
            support = render_db.catalog_support(path, "EPSG:6348")

        self.assertEqual(support["region_bbox_wsen"], [-71.2, 42.2, -71.0, 42.4])
        self.assertEqual(support["sha256"],
                         "ed16c5821de6cb3d99f901fb2f0d8b481be40e50c72a680f87b076eae7cbf6ba")
        self.assertEqual(support["surface_crs"], "EPSG:6348")
        self.assertLess(support["projected_bounds_xy"][0],
                        support["projected_bounds_xy"][2])
        self.assertLess(support["projected_bounds_xy"][1],
                        support["projected_bounds_xy"][3])

    def test_main_derives_lattice_from_catalog_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            surface = root / "surface"
            terrain.HeightField(
                elevation=np.zeros((4, 4), dtype=np.float32), x0=0.0,
                y0=4.0, res=1.0, crs="EPSG:4326",
                nodata_mask=np.zeros((4, 4), dtype=bool)).save(surface)
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"config": {
                "region_bbox_wsen": [1.0, 1.0, 3.0, 3.0]}}))
            output = root / "database"
            result = {"descriptors": np.zeros((4, 12, 512), np.float16),
                      "coverage": np.ones((4, 12), np.float32),
                      "sample_renders": {}}
            argv = ["render_db", "--height_field", str(surface),
                    "--catalog_manifest", str(catalog), "--weights", "unused",
                    "--spacing_m", "1", "--sky_fill_m", "-1",
                    "--output_dir", str(output), "--device", "cpu"]
            with mock.patch("sys.argv", argv), \
                    mock.patch.object(render_db.crosslocate_net,
                                      "CrossLocateVGG16MAC"), \
                    mock.patch.object(render_db.crosslocate_net,
                                      "load_converted_weights"), \
                    mock.patch.object(render_db, "build_database",
                                      return_value=result) as build:
                render_db.main()

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["catalog"]["path"], str(catalog))
            self.assertEqual(manifest["catalog"]["region_bbox_wsen"],
                             [1.0, 1.0, 3.0, 3.0])
            self.assertEqual(manifest["lattice"]["bounds_xy"],
                             [1.0, 1.0, 3.0, 3.0])
            self.assertEqual(build.call_args.kwargs["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
