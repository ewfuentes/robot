import tempfile
import unittest
from pathlib import Path

import shapely

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.catalog import lineage, schema
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    add_catalog_source,
    source_publication,
)

DATASET = "test_dataset"
LON, LAT = 129.37, 36.04
DEG_100M_LAT = 100.0 / 111_195.0


def _square(lon, lat, half_deg=0.0005):
    return shapely.box(lon - half_deg, lat - half_deg,
                       lon + half_deg, lat + half_deg)


def publish_full_catalog(root: Path) -> Path:
    rows = [
        ("osm:way:1", {"building": "yes", "name": "롯데백화점 포항점"},
         _square(LON, LAT)),
        ("osm:node:2", {"tourism": "hotel", "name:en": "Commodore Hotel"},
         shapely.Point(LON + 0.01, LAT)),
        ("osm:node:3", {"man_made": "lighthouse", "name": "Far Light"},
         shapely.Point(LON, LAT + 0.05)),
    ]
    frame = schema.build_frame(
        ids=[r[0] for r in rows], geometries=[r[2] for r in rows],
        landmark_types=["osm"] * len(rows), tags=[r[1] for r in rows])
    output_dir = root / "full_v1"
    with artifact.ArtifactDirectoryBuilder(
            output_dir, kind=paths_lib.CATALOGS, dataset=DATASET,
            version="full_v1", generator="test", git_commit="test",
            config={
                "schema": schema.FULL_ARTIFACT_SCHEMA,
                "source_coverage": {
                    "schema": lineage.SOURCE_COVERAGE_SCHEMA,
                    "status": "passed", "message": "test", "details": []},
            },
            declared_outputs=("catalog.feather",)) as builder:
        frame.to_feather(builder.output_path("catalog.feather"))
    return output_dir


def publish_source(root: Path) -> Path:
    rows = [
        # Same name as osm:way:1 after normalisation, 60 m away: duplicate.
        ("overture:a", {"shop": "department_store", "name": "롯데백화점포항점"},
         shapely.Point(LON, LAT + 0.6 * DEG_100M_LAT)),
        # Same name as osm:way:1 but 2 km away: a different branch, kept.
        ("overture:b", {"shop": "department_store", "name": "롯데백화점 포항점"},
         shapely.Point(LON, LAT + 20 * DEG_100M_LAT)),
        # Matches the catalog row's name:en variant, case-insensitively.
        ("overture:c", {"tourism": "hotel", "name": "commodore hotel"},
         shapely.Point(LON + 0.01, LAT + 0.2 * DEG_100M_LAT)),
        # New place, kept; the next row duplicates it inside the source.
        ("overture:d", {"amenity": "restaurant", "name": "영포회타운",
                        "brand": "Yeongpo"},
         shapely.Point(LON - 0.01, LAT)),
        ("overture:e", {"amenity": "restaurant", "brand": "yeongpo"},
         shapely.Point(LON - 0.01, LAT + 0.3 * DEG_100M_LAT)),
        # Far Light exists in OSM 5 km away only: kept.
        ("overture:f", {"man_made": "lighthouse", "name": "Far Light"},
         shapely.Point(LON, LAT)),
    ]
    frame = schema.build_frame(
        ids=[r[0] for r in rows], geometries=[r[2] for r in rows],
        landmark_types=["overture"] * len(rows), tags=[r[1] for r in rows])
    feather, _ = source_publication.publish(
        frame, root / "overture_v1", {"tool": "test"})
    return feather


class AddCatalogSourceTest(unittest.TestCase):

    def test_normalised_names(self):
        self.assertEqual(
            add_catalog_source.normalised_names(
                {"name": "롯데백화점 포항점", "name:en": "Lotte; LOTTE Dept.",
                 "brand": "Lotte", "amenity": "cafe"}),
            {"롯데백화점포항점", "lotte", "lottedept"})

    def test_appends_only_rows_the_catalog_lacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = publish_full_catalog(root)
            feather = publish_source(root)
            output = root / "full_overture_v1"
            merged = add_catalog_source.main(full, feather, output, 150.0)

            self.assertEqual(list(merged["id"]), [
                "osm:way:1", "osm:node:2", "osm:node:3",
                "overture:b", "overture:d", "overture:f"])
            self.assertEqual(
                set(merged["landmark_type"]), {"osm", "overture"})

            manifest = artifact.load_manifest(output)
            config = manifest.config
            self.assertEqual(config["rows_in_catalog"], 3)
            self.assertEqual(config["rows_in_source"], 6)
            self.assertEqual(config["rows_added"], 3)
            self.assertEqual(config["rows_out"], 6)
            self.assertEqual(
                [(p["source_id"], p["duplicate_of"])
                 for p in config["duplicates_of_catalog"]],
                [("overture:a", "osm:way:1"), ("overture:c", "osm:node:2")])
            self.assertEqual(
                [(p["source_id"], p["duplicate_of"])
                 for p in config["duplicates_within_source"]],
                [("overture:e", "overture:d")])
            self.assertLess(config["duplicates_of_catalog"][0]["distance_m"],
                            80.0)
            self.assertEqual(config["source_landmark_types"], ["overture"])
            self.assertEqual(len(manifest.upstreams), 1)

            # The derived catalog still proves coverage through its parent.
            terminal = lineage.require_passed_source_coverage(
                artifact.open_artifact(output))
            self.assertEqual(terminal, artifact.open_artifact(full))
            schema.read_frame(output / "catalog.feather")

            with self.assertRaises(SystemExit):
                add_catalog_source.main(full, feather, output, 150.0)

    def test_radius_zero_keeps_everything_but_exact_overlaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = publish_full_catalog(root)
            feather = publish_source(root)
            merged = add_catalog_source.main(
                full, feather, root / "out", 0.0)
            # overture:f sits inside osm:way:1's footprint but has another
            # name; overture:a is 60 m off, so nothing is within 0 m.
            self.assertEqual(len(merged), 9)


if __name__ == "__main__":
    unittest.main()
