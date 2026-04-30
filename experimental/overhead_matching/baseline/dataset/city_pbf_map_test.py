import unittest
from pathlib import Path

from experimental.overhead_matching.baseline.dataset import city_pbf_map


class CityPbfMapTest(unittest.TestCase):
    def test_cities_covered(self):
        self.assertEqual(
            set(city_pbf_map.cities()),
            {"Boston", "Chicago", "NewYork", "Seattle", "SanFrancisco"},
        )

    def test_resolve_pbf(self):
        p = city_pbf_map.resolve_pbf("Seattle", Path("/data/overhead_matching/datasets/osm_dumps"))
        self.assertEqual(p.name, "washington-200101.osm.pbf")

    def test_unknown_city(self):
        with self.assertRaises(KeyError):
            city_pbf_map.resolve_pbf("Atlantis", Path("/x"))


if __name__ == "__main__":
    unittest.main()
