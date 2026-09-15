"""Tests for the one buffered-bbox helper (see geometry_helpers.py for why
there is exactly one)."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield.collection import geometry_helpers


class PaddedBboxTest(unittest.TestCase):
    def test_contains_the_points_with_margin(self):
        lats = [42.30, 42.40]
        lons = [-71.10, -71.00]
        west, south, east, north = geometry_helpers.padded_bbox_wsen(
            lats, lons, buffer_km=10.0)
        self.assertLess(west, min(lons))
        self.assertLess(south, min(lats))
        self.assertGreater(east, max(lons))
        self.assertGreater(north, max(lats))

    def test_buffer_is_metric_on_both_axes(self):
        # At 60 N a degree of longitude is half a degree of latitude in
        # metres, so the same km buffer must pad twice as many degrees of
        # longitude — the unscaled version under-reaches at high latitude.
        west, south, east, north = geometry_helpers.padded_bbox_wsen(
            [60.0], [10.0], buffer_km=10.0)
        lat_pad = north - 60.0
        lon_pad = east - 10.0
        self.assertAlmostEqual(lon_pad / lat_pad,
                               1.0 / math.cos(math.radians(60.0)), places=6)
        self.assertAlmostEqual(lat_pad * geometry_helpers.KM_PER_DEG_LAT,
                               10.0, places=6)

    def test_wsen_order(self):
        bbox = geometry_helpers.padded_bbox_wsen([1.0], [2.0], buffer_km=1.0)
        west, south, east, north = bbox
        self.assertLess(west, east)
        self.assertLess(south, north)
        # x (longitude) first: west is a longitude, south a latitude.
        self.assertAlmostEqual((west + east) / 2.0, 2.0, places=9)
        self.assertAlmostEqual((south + north) / 2.0, 1.0, places=9)

    def test_empty_points_raise(self):
        with self.assertRaises(ValueError):
            geometry_helpers.padded_bbox_wsen([], [], buffer_km=1.0)


class BboxFromDatasetTest(unittest.TestCase):
    def test_reads_pano_id_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pano_id_mapping.csv"
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["pano_id", "lat", "lon", "filename"])
                writer.writeheader()
                writer.writerow({"pano_id": "f0000", "lat": 42.0,
                                 "lon": -71.0, "filename": "a.jpg"})
                writer.writerow({"pano_id": "f0001", "lat": 42.1,
                                 "lon": -70.9, "filename": "b.jpg"})
            west, south, east, north = geometry_helpers.bbox_from_dataset(
                Path(tmp), buffer_km=5.0)
        self.assertLess(west, -71.0)
        self.assertGreater(east, -70.9)
        self.assertLess(south, 42.0)
        self.assertGreater(north, 42.1)


if __name__ == "__main__":
    unittest.main()
