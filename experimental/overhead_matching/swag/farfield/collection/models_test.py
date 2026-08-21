"""Tests for the collection data model.

The load-bearing assertion: the local haversine that models.py used to define
is GONE, and lengths route through farfield.geometry.haversine_m (the one
haversine, REORG.md rule 1).
"""

import unittest

from experimental.overhead_matching.swag.farfield import geometry
from experimental.overhead_matching.swag.farfield.collection import models


class HaversineOwnershipTest(unittest.TestCase):
    def test_models_no_longer_defines_a_haversine(self):
        # The old module exported `haversine` (km, its own earth radius);
        # every caller now imports geometry.haversine_m instead.
        self.assertFalse(hasattr(models, "haversine"))

    def test_sequence_length_matches_the_one_haversine(self):
        imgs = [
            models.PanoImage(id=str(i), lat=lat, lng=lng, compass_angle=0.0,
                             computed_compass_angle=0.0, captured_at=i,
                             camera_type="spherical", height=2, width=4)
            for i, (lat, lng) in enumerate(
                [(42.35, -71.05), (42.36, -71.04), (42.37, -71.02)])
        ]
        seq = models.PanoSequence(id="s", images=imgs)
        got_km = seq.compute_length()
        want_m = (geometry.haversine_m(42.35, -71.05, 42.36, -71.04)
                  + geometry.haversine_m(42.36, -71.04, 42.37, -71.02))
        self.assertAlmostEqual(got_km, want_m / 1000.0, places=9)
        self.assertEqual(seq.length_km, got_km)

    def test_short_sequence_has_zero_length(self):
        seq = models.PanoSequence(id="s", images=[])
        self.assertEqual(seq.compute_length(), 0.0)


class PanoImageTest(unittest.TestCase):
    def test_from_api_prefers_computed_geometry_and_records_source(self):
        img = models.PanoImage.from_api({
            "id": 123,
            "computed_geometry": {"coordinates": [-71.0, 42.0]},
            "geometry": {"coordinates": [-70.0, 41.0]},
        })
        self.assertEqual((img.lat, img.lng), (42.0, -71.0))
        self.assertEqual(img.geometry_source, "computed")

    def test_from_api_falls_back_to_raw_geometry(self):
        img = models.PanoImage.from_api({
            "id": 123,
            "geometry": {"coordinates": [-70.0, 41.0]},
        })
        self.assertEqual((img.lat, img.lng), (41.0, -70.0))
        self.assertEqual(img.geometry_source, "raw")

    def test_equirect_by_camera_type_both_spellings(self):
        for camera_type in models.EQUIRECT_CAMERA_TYPES:
            img = models.PanoImage.from_api({"id": 1, "camera_type": camera_type})
            self.assertTrue(img.is_equirectangular, camera_type)
        self.assertFalse(models.PanoImage.from_api(
            {"id": 1, "camera_type": "perspective"}).is_equirectangular)

    def test_round_trip_dict(self):
        img = models.PanoImage.from_api({
            "id": 7, "computed_geometry": {"coordinates": [1.0, 2.0]},
            "camera_type": "perspective", "camera_parameters": [0.6, 0.0, 0.0],
            "is_pano": False, "width": 4000, "height": 3000,
            "sequence": "seq9",
        })
        again = models.PanoImage.from_dict(img.to_dict())
        self.assertEqual(again, img)


if __name__ == "__main__":
    unittest.main()
