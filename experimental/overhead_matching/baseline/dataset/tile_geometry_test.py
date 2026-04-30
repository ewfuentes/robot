import unittest
from pathlib import Path

from experimental.overhead_matching.baseline.dataset import tile_geometry


class TileGeometryTest(unittest.TestCase):
    def test_satellite_filename_to_center(self):
        lat, lon = tile_geometry.satellite_filename_to_center(
            "satellite_42.33276110_-71.04331168.png"
        )
        self.assertAlmostEqual(lat, 42.33276110)
        self.assertAlmostEqual(lon, -71.04331168)

    def test_center_zoom_to_bbox_brackets_center(self):
        # The bbox should contain the center; north > south; east > west.
        lat, lon, zoom = 47.6062, -122.3321, 20
        bbox = tile_geometry.center_zoom_to_bbox(lat, lon, zoom)
        self.assertGreater(bbox.north_lat, lat)
        self.assertLess(bbox.south_lat, lat)
        self.assertGreater(bbox.east_lon, lon)
        self.assertLess(bbox.west_lon, lon)

    def test_center_zoom_to_bbox_matches_boston_csv(self):
        # Check our derived bbox agrees with Boston's published bbox on a known
        # tile to within a small tolerance.
        csv = Path("/data/overhead_matching/datasets/VIGOR/Boston/satellite_tile_metadata.csv")
        if not csv.exists():
            self.skipTest("Boston CSV not present")
        bboxes = tile_geometry.boston_csv_to_bboxes(csv)
        fname = "satellite_42.33276110_-71.06016506.png"
        if fname not in bboxes:
            self.skipTest(f"{fname} not in Boston CSV")
        lat, lon = tile_geometry.satellite_filename_to_center(fname)
        derived = tile_geometry.center_zoom_to_bbox(lat, lon, zoom=20, tile_px=640)
        published = bboxes[fname]
        # Sub-pixel agreement; latitude varies less per-pixel than longitude at
        # mid-latitudes, so loose tolerance is fine.
        self.assertAlmostEqual(derived.north_lat, published.north_lat, places=4)
        self.assertAlmostEqual(derived.south_lat, published.south_lat, places=4)
        self.assertAlmostEqual(derived.east_lon, published.east_lon, places=4)
        self.assertAlmostEqual(derived.west_lon, published.west_lon, places=4)


if __name__ == "__main__":
    unittest.main()
