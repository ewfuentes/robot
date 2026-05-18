import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.baseline.dataset import city_pbf_map


class CityPbfMapTest(unittest.TestCase):
    def test_cities_covered(self):
        self.assertEqual(
            set(city_pbf_map.cities()),
            {
                # VIGOR original
                "Boston", "Chicago", "NewYork", "Seattle", "SanFrancisco",
                # VIGOR mapillary
                "Framingham", "Middletown", "Gap", "SanFrancisco_mapillary",
                "MiamiBeach", "post_hurricane_ian", "post_hurricane_ian_sw", "Norway",
            },
        )

    def test_resolve_pbf_single_dir_legacy(self):
        # Pre-existing callers pass a single Path; behavior preserved (resolves
        # to that dir even if the file does not exist on disk).
        p = city_pbf_map.resolve_pbf("Seattle", Path("/data/overhead_matching/datasets/osm_dumps"))
        self.assertEqual(p.name, "washington-200101.osm.pbf")
        self.assertEqual(p.parent, Path("/data/overhead_matching/datasets/osm_dumps"))

    def test_resolve_pbf_searches_dirs_in_order(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            d1, d2 = Path(t1), Path(t2)
            (d2 / "norway-251201.osm.pbf").write_text("")
            resolved = city_pbf_map.resolve_pbf("Norway", [d1, d2])
            self.assertEqual(resolved, d2 / "norway-251201.osm.pbf")

    def test_resolve_pbf_falls_back_to_first_dir_when_missing(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            d1, d2 = Path(t1), Path(t2)
            resolved = city_pbf_map.resolve_pbf("Norway", [d1, d2])
            # Neither exists; we return the first dir so the error message
            # points somewhere concrete.
            self.assertEqual(resolved, d1 / "norway-251201.osm.pbf")

    def test_unknown_city(self):
        with self.assertRaises(KeyError):
            city_pbf_map.resolve_pbf("Atlantis", Path("/x"))


if __name__ == "__main__":
    unittest.main()
