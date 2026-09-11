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


def publish_structure_catalog(root: Path) -> Path:
    """Nameless tall structures, the case name-dedupe cannot handle."""
    x, y = LON + 0.1, LAT + 0.1
    rows = [
        ("osm:node:10", {"man_made": "tower"}, shapely.Point(x, y)),
        # 22 m north of the tower: a different class, never a candidate.
        ("osm:node:12", {"man_made": "chimney"},
         shapely.Point(x, y + 0.22 * DEG_100M_LAT)),
        ("osm:way:11", {"building": "yes", "name": "Mill"},
         _square(x + 0.02, y)),
        # Two towers 40 m apart: anything between them is ambiguous.
        ("osm:node:13", {"man_made": "communications_tower"},
         shapely.Point(x + 0.1, y)),
        ("osm:node:14", {"man_made": "mast"},
         shapely.Point(x + 0.1, y + 0.4 * DEG_100M_LAT)),
    ]
    frame = schema.build_frame(
        ids=[r[0] for r in rows], geometries=[r[2] for r in rows],
        landmark_types=["osm"] * len(rows), tags=[r[1] for r in rows])
    output_dir = root / "structures_v1"
    with artifact.ArtifactDirectoryBuilder(
            output_dir, kind=paths_lib.CATALOGS, dataset=DATASET,
            version="structures_v1", generator="test", git_commit="test",
            config={
                "schema": schema.FULL_ARTIFACT_SCHEMA,
                "source_coverage": {
                    "schema": lineage.SOURCE_COVERAGE_SCHEMA,
                    "status": "passed", "message": "test", "details": []},
            },
            declared_outputs=("catalog.feather",)) as builder:
        frame.to_feather(builder.output_path("catalog.feather"))
    return output_dir


def publish_faa_source(root: Path) -> Path:
    x, y = LON + 0.1, LAT + 0.1
    tol = "faa:position_tolerance_m"
    rows = [
        # 30 m from osm:node:10, tolerance 6 m -> radius 50 m: merged.
        ("1", {"man_made": "tower", "height": "61.0", "faa:oas": "1",
               tol: "6.1"}, shapely.Point(x, y - 0.3 * DEG_100M_LAT)),
        # Second obstacle 40 m from the same, already consumed, tower: added.
        ("6", {"man_made": "tower", "height": "30.0", "faa:oas": "6",
               tol: "6.1"}, shapely.Point(x, y - 0.4 * DEG_100M_LAT)),
        # Tower 2 km away from everything: added.
        ("2", {"man_made": "tower", "height": "90.0", "faa:oas": "2",
               tol: "6.1"}, shapely.Point(x, y - 20 * DEG_100M_LAT)),
        # Building point inside the Mill footprint (distance 0): merged.
        ("3", {"building": "yes", "height": "70.0", "faa:oas": "3",
               tol: "6.1"}, shapely.Point(x + 0.02, y)),
        # Between the two towers, 15 m from node:13 and 25 m from node:14:
        # merged into the nearer one, recorded as contested.
        ("4", {"man_made": "tower", "height": "50.0", "faa:oas": "4",
               tol: "6.1"}, shapely.Point(x + 0.1, y + 0.15 * DEG_100M_LAT)),
        # Chimney 100 m from osm:node:12 with a 76 m tolerance -> radius
        # 152 m: merged; the tower 78 m away is not its class.
        ("5", {"man_made": "chimney", "height": "40.0", "faa:oas": "5",
               tol: "76.2"}, shapely.Point(x, y + 1.22 * DEG_100M_LAT)),
    ]
    frame = schema.build_frame(
        ids=[r[0] for r in rows], geometries=[r[2] for r in rows],
        landmark_types=["faa"] * len(rows), tags=[r[1] for r in rows])
    feather, _ = source_publication.publish(
        frame, root / "faa_v1", {"tool": "test"})
    return feather


class AddCatalogSourceTest(unittest.TestCase):

    def test_structure_class(self):
        cases = {
            ("man_made", "communications_tower"): "tower",
            ("man_made", "storage_tank"): "tank",
            ("power", "tower"): "power_tower",
            ("building", "yes"): "building",
            ("bridge", "yes"): "bridge",
            ("amenity", "cafe"): None,
        }
        for (key, value), expected in cases.items():
            self.assertEqual(
                add_catalog_source.structure_class({key: value}), expected)
        self.assertEqual(add_catalog_source.structure_class(
            {"power": "plant", "plant:source": "solar"}), "solar")
        self.assertEqual(add_catalog_source.structure_class(
            {"man_made": "tower", "building": "yes"}), "tower")

    def test_class_merge_merges_adds_and_flags_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = publish_structure_catalog(root)
            feather = publish_faa_source(root)
            output = root / "structures_faa_v1"
            merged = add_catalog_source.main(
                full, feather, output, 150.0, class_merge_radius_m=50.0,
                class_merge_tolerance_tag="faa:position_tolerance_m")

            self.assertEqual(list(merged["id"]), [
                "osm:node:10", "osm:node:12", "osm:way:11", "osm:node:13",
                "osm:node:14", "6", "2"])
            tags = {row_id: t for row_id, t in
                    zip(merged["id"], schema.tag_dicts(merged))}
            self.assertEqual(tags["osm:node:10"]["faa:oas"], "1")
            self.assertEqual(tags["osm:node:10"]["height"], "61.0")
            self.assertEqual(tags["osm:node:10"]["man_made"], "tower")
            self.assertEqual(tags["osm:way:11"]["height"], "70.0")
            self.assertEqual(tags["osm:way:11"]["name"], "Mill")
            self.assertEqual(tags["osm:node:12"]["faa:oas"], "5")
            self.assertEqual(tags["osm:node:13"]["faa:oas"], "4")
            self.assertNotIn("faa:oas", tags["osm:node:14"])
            self.assertEqual(set(merged["landmark_type"]), {"osm", "faa"})

            config = artifact.load_manifest(output).config
            self.assertEqual(config["rows_added"], 2)
            self.assertEqual(config["rows_merged_into_catalog"], 4)
            self.assertEqual(config["rows_merged_contested"], 1)
            by_source = {m["source_id"]: m
                         for m in config["merged_into_catalog"]}
            self.assertEqual(
                {k: m["merged_into"] for k, m in by_source.items()},
                {"1": "osm:node:10", "3": "osm:way:11", "4": "osm:node:13",
                 "5": "osm:node:12"})
            self.assertEqual(by_source["3"]["distance_m"], 0.0)
            self.assertEqual(by_source["5"]["radius_m"], 152.4)
            self.assertEqual(by_source["1"]["added_tags"],
                             ["faa:oas", "faa:position_tolerance_m",
                              "height"])
            self.assertFalse(by_source["1"]["contested"])
            self.assertTrue(by_source["4"]["contested"])
            self.assertEqual(
                [(a["id"], a["distance_m"]) for a in
                 by_source["4"]["alternatives"]],
                [("osm:node:14", 25.0)])
            schema.read_frame(output / "catalog.feather")

    def test_without_class_merge_nameless_rows_are_all_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merged = add_catalog_source.main(
                publish_structure_catalog(root), publish_faa_source(root),
                root / "out", 150.0)
            self.assertEqual(len(merged), 11)


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
