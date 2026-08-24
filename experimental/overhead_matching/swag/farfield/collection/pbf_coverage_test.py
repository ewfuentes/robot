import tempfile
import unittest
from pathlib import Path
from unittest import mock

from shapely.geometry import box

from experimental.overhead_matching.swag.farfield.collection import (
    pbf_coverage as subject,
)
from experimental.overhead_matching.swag.farfield import artifact


class CoveragePolicyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.cache = Path(self.temporary.name)
        self.want = box(0.0, 0.0, 1.0, 1.0)

    def tearDown(self):
        self.temporary.cleanup()

    def test_ordinary_coverage_rejects_fourteen_percent_land_loss(self):
        chosen = box(0.0, 0.0, 0.857, 1.0)
        with mock.patch.object(
                subject, "_union_of",
                return_value=([("chosen", chosen)], [], chosen)), \
             mock.patch.object(
                 subject, "mapped_area", return_value=self.want):
            ok, message, details = subject.check_coverage(
                ["chosen"], self.want.bounds, self.cache)

        self.assertFalse(ok)
        self.assertIn("14.3%", message)
        self.assertIn({
            "coverage_policy": "mappable_land",
            "tolerance_frac": 0.005,
        }, details)

    def test_half_percent_is_the_only_tolerance(self):
        chosen = box(0.0, 0.0, 0.996, 1.0)
        with mock.patch.object(
                subject, "_union_of",
                return_value=([("chosen", chosen)], [], chosen)), \
             mock.patch.object(
                 subject, "mapped_area", return_value=self.want):
            ok, message, details = subject.check_coverage(
                ["chosen"], self.want.bounds, self.cache)

        self.assertTrue(ok)
        self.assertIn("99.6%", message)
        self.assertIn({
            "coverage_policy": "mappable_land",
            "tolerance_frac": 0.005,
        }, details)

    def test_empty_mapped_target_fails_closed(self):
        with mock.patch.object(
                subject, "_union_of",
                return_value=([("chosen", self.want)], [], self.want)), \
             mock.patch.object(subject, "mapped_area", return_value=None):
            ok, message, _ = subject.check_coverage(
                ["chosen"], self.want.bounds, self.cache)

        self.assertFalse(ok)
        self.assertIn("cannot verify coverage", message)

    def test_actual_pbf_header_and_digest_are_bound(self):
        spec = "north-america/us/massachusetts-latest.osm.pbf"
        pbf = self.cache / "massachusetts-260824.osm.pbf"
        pbf.write_bytes(b"synthetic PBF identity")
        with mock.patch.object(
                subject, "_union_of",
                return_value=([(spec, self.want)], [], self.want)), \
             mock.patch.object(
                 subject, "mapped_area", return_value=self.want), \
             mock.patch.object(
                 subject, "header_bbox", return_value=self.want.bounds):
            ok, _, details = subject.check_coverage(
                [spec], self.want.bounds, self.cache, pbf_paths=[pbf])

        self.assertTrue(ok)
        self.assertEqual(details[0]["pbf"]["sha256"],
                         artifact.sha256_file(pbf))
        self.assertEqual(details[0]["pbf"]["header_bbox"],
                         list(self.want.bounds))

    def test_actual_pbf_with_wrong_header_fails_closed(self):
        spec = "north-america/us/massachusetts-latest.osm.pbf"
        pbf = self.cache / "massachusetts-260824.osm.pbf"
        pbf.write_bytes(b"synthetic PBF identity")
        with mock.patch.object(
                subject, "_union_of",
                return_value=([(spec, self.want)], [], self.want)), \
             mock.patch.object(
                 subject, "mapped_area", return_value=self.want), \
             mock.patch.object(
                 subject, "header_bbox", return_value=(2.0, 2.0, 3.0, 3.0)):
            ok, message, _ = subject.check_coverage(
                [spec], self.want.bounds, self.cache, pbf_paths=[pbf])
        self.assertFalse(ok)
        self.assertIn("HeaderBBox", message)

    def test_poly_cache_key_includes_the_full_spec(self):
        first = subject.poly_cache_path(
            "north-america/us/georgia-latest.osm.pbf", self.cache)
        second = subject.poly_cache_path(
            "europe/georgia-latest.osm.pbf", self.cache)
        self.assertNotEqual(first, second)
        self.assertTrue(first.name.endswith("-georgia.poly"))


if __name__ == "__main__":
    unittest.main()
