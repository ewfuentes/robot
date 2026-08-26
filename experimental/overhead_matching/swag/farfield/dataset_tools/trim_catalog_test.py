"""Tests for trim_catalog.

The rules encode judgements about what is visible from a vessel, so the tests
are written as those judgements: a container crane survives without a height
tag, a bench does not, and a bare shed does not while a grain terminal does.
"""

import csv
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    landmark_positive_set as positive_set,
)
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    trim_catalog as tc,
)


def square(lon, lat, side_m):
    """Axis-aligned square footprint of roughly side_m x side_m.

    A degree of longitude is only ~82 km at Boston, so the lon and lat steps
    differ; using one step for both makes a rectangle, not a square.
    """
    import math
    d_lat = side_m / 110574.0
    d_lon = side_m / (111320.0 * math.cos(math.radians(lat)))
    return Polygon([(lon, lat), (lon + d_lon, lat),
                    (lon + d_lon, lat + d_lat), (lon, lat + d_lat)])


def drops(tags: list, areas=None, min_area=2000.0, min_levels=6.0):
    areas = np.zeros(len(tags)) if areas is None else np.asarray(areas)
    masks = tc.evaluate_rules(tags, areas, min_area, min_levels)
    return {name: mask.tolist() for name, mask in masks.items()}


BOSTON_SCOPE = "boston_harbor_20260712"
BOSTON_LEG1 = "boston_harbor_leg1"


def write_active_dataset(root: Path, dataset: str, points):
    dataset_dir = root / "datasets" / dataset
    dataset_dir.mkdir(parents=True)
    mapping = dataset_dir / "pano_id_mapping.csv"
    gps = dataset_dir / "frames_gps.csv"
    frames = dataset_dir / "frames"
    frames.mkdir()
    rows = []
    for index, (lat, lon) in enumerate(points):
        pano_id = f"f{index:06d}"
        filename = f"{pano_id},{lat:.7f},{lon:.7f},.jpg"
        rows.append((pano_id, lat, lon, filename))
        (frames / filename).write_bytes(b"jpeg")
    with mapping.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(tc.active_catalogs.MAPPING_COLUMNS)
        writer.writerows(rows)
    with gps.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("idx", "latitude", "longitude", "dist_m",
                         "video_t_s"))
        for index, (_, lat, lon, _) in enumerate(rows):
            writer.writerow((index, lat, lon, index, index))
    (dataset_dir / "panorama").symlink_to(
        "frames", target_is_directory=True)
    return gps


def reviewed_clip_plan(farfield_root: Path, output_dataset=BOSTON_LEG1,
                       minimum_area_km2=625.0):
    scope = tc.active_catalogs.SCOPE_BY_NAME[BOSTON_SCOPE]
    for index, dataset in enumerate(scope.bbox_datasets):
        write_active_dataset(
            farfield_root, dataset,
            [(42.30 + index * 0.01, -71.05 + index * 0.01)])
    plan = tc.build_clip_plan(
        scope_name=BOSTON_SCOPE,
        output_dataset=output_dataset,
        farfield_root=farfield_root,
        nominal_buffer_km=5.0,
        minimum_buffer_km=1.0,
        minimum_area_km2=minimum_area_km2)
    return plan, tuple(plan["bbox_wsen"])


class RuleTest(unittest.TestCase):
    def test_untagged_row_dropped(self):
        self.assertEqual(drops([{}])["no_far_field_tags"], [True])

    def test_street_furniture_dropped(self):
        for tag in [{"amenity": "bench"}, {"natural": "tree"},
                    {"man_made": "surveillance"}, {"highway": "crossing"},
                    {"highway": "footway"}, {"amenity": "waste_basket"},
                    {"barrier": "fence"}, {"power": "pole"}]:
            with self.subTest(tag=tag):
                self.assertEqual(drops([tag])["unobservable_only"], [True],
                                 f"{tag} should be unobservable from a "
                                 f"vessel")

    def test_navigation_aids_and_structures_kept(self):
        for tag in [{"seamark:type": "buoy_lateral", "name": "Buoy 12"},
                    {"man_made": "lighthouse"},
                    {"man_made": "crane"},      # no height tag: still 50-80 m
                    {"object_class": "BCNLAT"},
                    {"place": "island", "name": "Georges Island"},
                    {"historic": "fort"},
                    {"natural": "cliff"},
                    {"amenity": "ferry_terminal"},
                    {"man_made": "water_tower"}]:
            with self.subTest(tag=tag):
                result = drops([tag])
                self.assertFalse(any(v[0] for v in result.values()),
                                 f"{tag} must survive: {result}")

    def test_name_rescues_a_soft_unobservable_tag(self):
        """Regression: Bunker Hill Monument is a 67 m obelisk whose only OSM
        tags are a name and tourism=information. A name outranks a weak
        tag."""
        result = drops([{"name": "Bunker Hill Monument",
                         "tourism": "information"}])
        self.assertFalse(any(v[0] for v in result.values()), result)

    def test_english_only_name_rescues_a_soft_unobservable_tag(self):
        # 219 rows in pohang's source table are named only in name:en.
        # Before has_name() they read as nameless and were dropped as
        # unobservable.
        self.assertEqual(
            drops([{"tourism": "information",
                    "name:en": "Bunker Hill Monument"}]
                  )["unobservable_only"], [False])

    def test_romanized_name_also_rescues(self):
        self.assertEqual(
            drops([{"tourism": "information", "name:ko-Latn": "Homigot"}]
                  )["unobservable_only"], [False])

    def test_untranslated_name_variant_does_not_rescue(self):
        # name:el on a Korean row is noise, not identity, and must not
        # rescue.
        self.assertEqual(
            drops([{"tourism": "information", "name:el": "noise"}]
                  )["unobservable_only"], [True])

    def test_english_only_name_saves_a_small_building(self):
        self.assertEqual(
            drops([{"building": "yes", "name:en": "Old Custom House"}],
                  areas=[10.0])["generic_small_building"], [False])

    def test_unnamed_information_board_still_dropped(self):
        self.assertEqual(
            drops([{"tourism": "information"}])["unobservable_only"], [True])

    def test_name_does_not_rescue_a_hard_unobservable_tag(self):
        """Regression: named bus stops are real names on invisible objects,
        and admitting them put 11,357 roads and stops into the catalog."""
        for tag in [{"highway": "bus_stop",
                     "name": "Dorchester Ave @ Dix St", "operator": "MBTA"},
                    {"highway": "traffic_signals",
                     "name": "Thomas F. Kennedy Square"},
                    {"amenity": "parking", "name": "Garage A"},
                    {"amenity": "post_box", "name": "Post Box 12"}]:
            with self.subTest(tag=tag):
                self.assertEqual(drops([tag])["unobservable_only"], [True],
                                 f"{tag} must not be rescued by its name")

    def test_tenant_businesses_dropped_even_when_named(self):
        """A restaurant occupies a building rather than being one; the host
        building is its own row, so the tenant adds nothing at range."""
        for tag in [{"amenity": "restaurant", "name": "Legal Sea Foods"},
                    {"amenity": "cafe", "name": "Thinking Cup"},
                    {"amenity": "bank", "name": "Santander"},
                    {"amenity": "fountain", "name": "Rings Fountain"},
                    {"leisure": "pitch", "name": "Field 3"},
                    {"landuse": "grass"}]:
            with self.subTest(tag=tag):
                self.assertEqual(drops([tag])["unobservable_only"], [True],
                                 f"{tag} is not far-field visible")

    def test_civic_and_waterfront_amenities_kept(self):
        """Values that name a whole building or a waterfront structure
        stay."""
        for tag in [{"amenity": "theatre", "name": "Leader Bank Pavilion"},
                    {"amenity": "school", "name": "Hull High School"},
                    {"amenity": "ferry_terminal"},
                    {"amenity": "place_of_worship",
                     "name": "Old North Church"},
                    {"tourism": "hotel", "name": "Boston Harbor Hotel"},
                    {"tourism": "museum", "name": "ICA"},
                    {"leisure": "marina"},
                    {"leisure": "slipway"}]:
            with self.subTest(tag=tag):
                result = drops([tag])
                self.assertFalse(any(v[0] for v in result.values()),
                                 f"{tag} must survive: {result}")

    def test_zoning_class_needs_a_name(self):
        """Regression: `landuse=residential; name=Harbor Towers` is a
        labelled positive, while an unnamed residential polygon is not a
        landmark."""
        self.assertFalse(
            any(v[0] for v in drops([{"landuse": "residential",
                                      "name": "Harbor Towers"}]).values()))
        self.assertEqual(
            drops([{"landuse": "residential"}])["unobservable_only"], [True])

    def test_rail_infrastructure_dropped_even_when_named(self):
        """Track and MBTA subway stops have no silhouette from the water,
        and every one of them is named."""
        for tag in [{"railway": "rail", "name": "Dorchester Branch",
                     "operator": "MBTA"},
                    {"railway": "station", "name": "Mattapan",
                     "operator": "MBTA"},
                    {"railway": "subway", "name": "Red Line"},
                    {"railway": "subway_entrance", "name": "Andrew"},
                    {"railway": "platform"}, {"railway": "abandoned"}]:
            with self.subTest(tag=tag):
                self.assertEqual(drops([tag])["unobservable_only"], [True],
                                 f"{tag} is not far-field visible")

    def test_railyards_and_rail_bridges_kept(self):
        """The two rail things that do read from the water."""
        for tag in [{"landuse": "railway",
                     "name": "Southampton Street Yard",
                     "operator": "Amtrak"},
                    {"landuse": "railway"},               # unnamed yard
                    {"railway": "yard"}, {"railway": "depot"},
                    {"railway": "service_station"},
                    {"railway": "rail", "bridge": "yes",
                     "name": "Dorchester Branch"},
                    {"railway": "rail", "bridge": "yes"}]:
            with self.subTest(tag=tag):
                result = drops([tag])
                self.assertFalse(any(v[0] for v in result.values()),
                                 f"{tag} must survive: {result}")

    def test_bridges_are_kept_named_or_not(self):
        """Bridges survive despite carrying a hard-blocked highway tag."""
        for tag in [{"bridge": "yes", "highway": "footway"},
                    {"bridge": "yes", "highway": "motorway",
                     "name": "Maurice J. Tobin Memorial Bridge"},
                    {"man_made": "bridge", "name": "Tobin Bridge"},
                    {"bridge": "viaduct", "highway": "primary"},
                    {"object_class": "BRIDGE", "man_made": "bridge"}]:
            with self.subTest(tag=tag):
                result = drops([tag])
                self.assertFalse(any(v[0] for v in result.values()),
                                 f"{tag} is a bridge and must survive: "
                                 f"{result}")

    def test_highway_with_structure_survives(self):
        """A bridge carrying a road is still a bridge."""
        result = drops([{"highway": "primary", "man_made": "bridge",
                         "name": "Tobin Bridge"}])
        self.assertFalse(any(v[0] for v in result.values()))

    def test_generic_small_building_dropped(self):
        self.assertEqual(
            drops([{"building": "yes"}],
                  areas=[200.0])["generic_small_building"],
            [True])

    def test_building_survives_on_its_own_merits(self):
        cases = [
            ({"building": "yes", "name": "Custom House Tower"}, 200.0),
            ({"building": "yes", "height": "150"}, 200.0),
            ({"building": "yes", "building:levels": "40"}, 200.0),
            ({"building": "commercial"}, 200.0),          # landmark use
            ({"building": "yes"}, 9000.0),                # big footprint
            ({"building": "yes", "man_made": "chimney"}, 200.0),
        ]
        for tag, area in cases:
            with self.subTest(tag=tag, area=area):
                self.assertEqual(
                    drops([tag], areas=[area])["generic_small_building"],
                    [False], f"{tag} @ {area} m2 should survive")

    def test_numeric_parsing_tolerates_osm_noise(self):
        self.assertEqual(tc._numeric("12 m"), 12.0)
        self.assertEqual(tc._numeric("3;4"), 3.0)
        self.assertEqual(tc._numeric("about ten"), 0.0)
        self.assertEqual(tc._numeric(None), 0.0)

    def test_rules_are_independent(self):
        """Every row gets a verdict from every rule, so counts stay
        auditable."""
        tags = [{}, {"amenity": "bench"}, {"building": "yes"},
                {"man_made": "lighthouse"}]
        masks = tc.evaluate_rules(tags, np.zeros(4), 2000.0, 6.0)
        self.assertEqual(set(masks), {"no_far_field_tags",
                                      "unobservable_only",
                                      "generic_small_building"})
        for mask in masks.values():
            self.assertEqual(len(mask), 4)


class GeometryTest(unittest.TestCase):
    def test_footprint_area_of_polygon(self):
        gdf = gpd.GeoDataFrame(
            {"id": ["a", "b", "c"]},
            geometry=[square(-71.0, 42.3, 100.0), Point(-71.0, 42.3),
                      LineString([(-71.0, 42.3), (-70.99, 42.3)])],
            crs="EPSG:4326")
        areas = tc.footprint_area_m2(gdf)
        self.assertAlmostEqual(areas[0], 100.0 * 100.0, delta=2000.0)
        self.assertEqual(areas[1], 0.0)
        self.assertEqual(areas[2], 0.0)


def catalog(rows, crs="EPSG:4326") -> gpd.GeoDataFrame:
    """Tiny dict-schema catalog: [(id, tags, geometry), ...]."""
    return gpd.GeoDataFrame(
        {"id": [r[0] for r in rows],
         "landmark_type": ["osm"] * len(rows),
         "tags": [json.dumps(r[1]) for r in rows]},
        geometry=[r[2] for r in rows], crs=crs)


DATASET = "test_dataset"


def publish_catalog(root: Path, rows, version: str = "v1",
                    upstreams=(), dataset=DATASET,
                    crs="EPSG:4326") -> Path:
    output_dir = Path(root) / version
    with artifact.ArtifactDirectoryBuilder(
            output_dir,
            kind=paths_lib.CATALOGS,
            dataset=dataset,
            version=version,
            generator="trim_catalog_test",
            git_commit="test",
            upstreams=upstreams,
            declared_outputs=("catalog.feather",)) as builder:
        catalog(rows, crs=crs).to_feather(
            builder.output_path("catalog.feather"))
    return output_dir


def trim_config(output_dir: Path) -> dict:
    return artifact.load_manifest(output_dir).config


def publish_supporting_artifact(
        root: Path, kind: str, version: str,
) -> artifact.ArtifactRef:
    output_dir = root / version
    with artifact.ArtifactDirectoryBuilder(
            output_dir,
            kind=kind,
            dataset=DATASET,
            version=version,
            generator="trim_catalog_test",
            git_commit="test",
            declared_outputs=("payload.json",)) as builder:
        artifact.atomic_write_json(builder.output_path("payload.json"), {})
    return artifact.open_artifact(output_dir)


def signature_tags(display: str) -> dict[str, str]:
    return dict(field.split("=", 1) for field in display.split("; "))


def signature_id_for_display(display: str) -> str:
    return positive_set.signature_id(signature_tags(display))


def signature_entry(display: str, landmark_ids: list[str]) -> dict:
    return {
        "canonical_tags": signature_tags(display),
        "display_label": positive_set.format_signature(signature_tags(display)),
        "landmark_ids": landmark_ids,
    }


def publish_matches(
        root: Path, catalog_ref: artifact.ArtifactRef, entries,
        version: str = "matches_v1",
) -> tuple[Path, artifact.ArtifactRef]:
    """Publish [(tracklet, signature display, confidence)] as LANDMARK_MATCHES."""
    tracks_ref = publish_supporting_artifact(
        root, paths_lib.OBJECT_TRACKS, f"tracks_for_{version}")
    audits_ref = publish_supporting_artifact(
        root, paths_lib.SEMANTIC_AUDITS, f"audits_for_{version}")
    matches = {}
    signatures = {}
    for i, (tracklet, signature, confidence) in enumerate(entries):
        landmark_id = f"osm:way:{i}"
        canonical_id = signature_id_for_display(signature)
        entry = signatures.setdefault(
            canonical_id, signature_entry(signature, []))
        canonical_display = entry["display_label"]
        entry["landmark_ids"].append(landmark_id)
        matches[tracklet] = {
            "n_landmarks": 1,
            "n_signatures": 1,
            "matches": [{
                "landmark_id": landmark_id,
                "per_call_candidate_scores": [confidence],
                "aggregate_confidence": confidence,
                "aggregation_rule": positive_set.CANDIDATE_AGGREGATION_RULE,
                "match_type": "instance",
                "signature_id": canonical_id,
                "signature_display": canonical_display,
            }],
        }
    if not matches:
        matches = {"LT0": {
            "n_landmarks": 0,
            "n_signatures": 0,
            "matches": [],
        }}
        unused = "man_made=unused"
        signatures = {
            signature_id_for_display(unused):
                signature_entry(unused, ["osm:way:unused"]),
        }
    output_dir = root / version
    with artifact.ArtifactDirectoryBuilder(
            output_dir,
            kind=paths_lib.LANDMARK_MATCHES,
            dataset=DATASET,
            version=version,
            generator="trim_catalog_test",
            git_commit="test",
            upstreams=(tracks_ref, audits_ref, catalog_ref),
            config={
                "phase": "canonical_results",
                "coverage": "complete",
                "n_expected": 1,
                "n_successful": 1,
                "n_tracklets_expected": len(matches),
                "n_tracklets_successful": len(matches),
            },
            declared_outputs=positive_set.MATCHING_OUTPUTS) as builder:
        for output in positive_set.MATCHING_OUTPUTS:
            if output == positive_set.MATCHES_PAYLOAD:
                artifact.atomic_write_json(builder.output_path(output), matches)
            elif output == positive_set.SIGNATURES_PAYLOAD:
                artifact.atomic_write_json(
                    builder.output_path(output), signatures)
            else:
                artifact.atomic_write_file(
                    builder.output_path(output), b"{}\n")
    return output_dir, artifact.open_artifact(output_dir)


class EvidenceIsOptionalTest(unittest.TestCase):
    """A new catalog can be trimmed before any matching result exists."""

    def test_publishes_without_matching_or_positive_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = publish_catalog(
                tmp, [("osm:node:1", {"man_made": "lighthouse"},
                       Point(-70.89, 42.32))])
            tc.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False)
            self.assertTrue((tmp / "v2_trimmed" / "manifest.json").exists())
            record = trim_config(tmp / "v2_trimmed")
            self.assertNotIn("recall_guard", record)
            for key in ("matched_from", "positive_set", "confidence_floor",
                        "allow_recall_loss"):
                self.assertNotIn(key, record)

    def test_guarded_and_unguarded_runs_publish_the_same_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [("osm:node:1", {"man_made": "tower"},
                     Point(-71.0, 42.3))]
            source = publish_catalog(root / "source", rows)
            source_ref = artifact.open_artifact(source)
            matching_dir, matching_ref = publish_matches(
                root / "matching", source_ref,
                [("LT0", "man_made=tower", 0.9)])
            positive = root / "positive.json"
            positive_set.main(matching_dir, positive)

            unguarded = root / "unguarded"
            guarded = root / "guarded"
            tc.main(source, unguarded, None, 2000.0, 6.0, False)
            tc.main(source, guarded, positive, 2000.0, 6.0, False,
                    matched_from=[matching_dir], confidence_floor=0.8)
            plain_manifest = artifact.load_manifest(unguarded)
            guarded_manifest = artifact.load_manifest(guarded)
            positive_sha256 = artifact.sha256_file(positive)
            plain_bytes = (unguarded / "catalog.feather").read_bytes()
            guarded_bytes = (guarded / "catalog.feather").read_bytes()

        self.assertEqual(plain_manifest.upstreams, (source_ref,))
        self.assertEqual(
            guarded_manifest.upstreams, (source_ref, matching_ref))
        plain_config = dict(plain_manifest.config)
        guarded_config = dict(guarded_manifest.config)
        plain_guards = plain_config.pop("recall_guards")
        guarded_guards = guarded_config.pop("recall_guards")
        self.assertEqual(guarded_config, plain_config)
        self.assertEqual(plain_guards["matching_artifacts"], [])
        self.assertEqual(
            guarded_guards["matching_artifacts"], [matching_ref.to_dict()])
        self.assertEqual(
            guarded_guards["positive_set_sha256"],
            positive_sha256)
        self.assertEqual(guarded_manifest.declared_outputs,
                         plain_manifest.declared_outputs)
        self.assertEqual(guarded_manifest.content_digest,
                         plain_manifest.content_digest)
        self.assertEqual(guarded_bytes, plain_bytes)


class MatchedRecallGuardTest(unittest.TestCase):
    """The guard that asks a stronger question than the positive set: would a
    rule drop something a matcher already chose on a real run?"""

    def test_reads_matches_and_honours_the_confidence_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = publish_catalog(
                root, [("a", {"man_made": "tower"},
                        Point(-71.0, 42.3))])
            source_ref = artifact.open_artifact(source)
            matching_dir, matching_ref = publish_matches(
                root, source_ref,
                [("LT0", "man_made=lighthouse", 0.9),
                 ("LT1", "amenity=bench", 0.2)])
            found, refs = tc.matched_signatures(
                [matching_dir], 0.5, source_ref)
        expected = signature_id_for_display("man_made=lighthouse")
        self.assertEqual(sorted(found), [expected])
        self.assertEqual(found[expected][0][1], "LT0")
        self.assertEqual(found[expected][0][4], "man_made=lighthouse")
        self.assertEqual(refs, (matching_ref,))

    def test_rejects_a_loose_matches_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = publish_catalog(
                root, [("a", {"man_made": "tower"},
                        Point(-71.0, 42.3))])
            source_ref = artifact.open_artifact(source)
            loose = root / "matches.json"
            loose.write_text("{}")
            with self.assertRaisesRegex(SystemExit, "LANDMARK_MATCHES"):
                tc.matched_signatures([loose], 0.5, source_ref)

    def test_accepts_matches_from_a_prior_trim_of_the_same_full_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [("a", {"man_made": "tower"}, Point(-71.0, 42.3))]
            source = publish_catalog(root / "full", rows)
            source_ref = artifact.open_artifact(source)
            prior_trim = publish_catalog(
                root / "trimmed", rows, version="trimmed_v1",
                upstreams=(source_ref,))
            matching_dir, _ = publish_matches(
                root / "matching", artifact.open_artifact(prior_trim),
                [("LT0", "man_made=tower", 0.9)])
            found, _ = tc.matched_signatures(
                [matching_dir], 0.5, source_ref)
        self.assertEqual(
            set(found), {signature_id_for_display("man_made=tower")})

    def test_display_collision_does_not_satisfy_digest_recall(self):
        one_tag = {"a": "x; b=y"}
        two_tags = {"a": "x", "b": "y"}
        display = positive_set.format_signature(one_tag)
        self.assertEqual(display, positive_set.format_signature(two_tags))
        one_id = positive_set.signature_id(one_tag)
        two_id = positive_set.signature_id(two_tags)
        self.assertNotEqual(one_id, two_id)
        matched = {
            one_id: [("matches_v1", "LT0", 0.9, "instance", display)],
        }

        lost, absent = tc.report_matched_recall(
            matched,
            [two_tags],
            np.array([True]),
            {"none": np.array([False])},
        )

        self.assertEqual(lost, [])
        self.assertEqual(absent, [one_id])

    def test_missing_matching_artifact_is_an_error_not_a_silent_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = publish_catalog(
                root, [("a", {"man_made": "tower"},
                        Point(-71.0, 42.3))])
            with self.assertRaises(SystemExit):
                tc.matched_signatures(
                    [root / "missing"], 0.5,
                    artifact.open_artifact(source))

    def test_matching_bound_to_another_catalog_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = publish_catalog(
                root / "source",
                [("a", {"man_made": "tower"}, Point(-71.0, 42.3))])
            other = publish_catalog(
                root / "other",
                [("b", {"man_made": "pier"}, Point(-70.9, 42.2))])
            matching_dir, _ = publish_matches(
                root / "matching", artifact.open_artifact(other),
                [("LT0", "man_made=pier", 0.9)])
            with self.assertRaisesRegex(SystemExit, "typed descendant"):
                tc.main(
                    source, root / "v2", None, 2000.0, 6.0, False,
                    matched_from=[matching_dir])
            self.assertFalse((root / "v2").exists())

    def test_refuses_to_write_when_a_matched_signature_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = publish_catalog(
                tmp, [("osm:node:1", {"amenity": "bench"},
                       Point(-71.0, 42.3))])
            matches, _ = publish_matches(
                tmp, artifact.open_artifact(source),
                [("LT0", "amenity=bench", 0.9)])
            with self.assertRaises(SystemExit) as caught:
                tc.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False,
                        matched_from=[matches])
            self.assertIn("refusing to write", str(caught.exception))
            self.assertFalse((tmp / "v2_trimmed").exists())

    def test_allow_recall_loss_is_the_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = publish_catalog(
                tmp, [("osm:node:1", {"amenity": "bench"},
                       Point(-71.0, 42.3))])
            source_ref = artifact.open_artifact(source)
            matches, matching_ref = publish_matches(
                tmp, source_ref, [("LT0", "amenity=bench", 0.9)])
            tc.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False,
                    matched_from=[matches], allow_recall_loss=True)
            manifest = artifact.load_manifest(tmp / "v2_trimmed")
        self.assertEqual(manifest.upstreams, (source_ref, matching_ref))
        self.assertTrue(manifest.config["recall_guards"]["allow_recall_loss"])

    def test_more_than_40_percent_absent_requires_a_distinct_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = publish_catalog(
                tmp, [("osm:node:1", {"man_made": "lighthouse"},
                       Point(-70.89, 42.32))])
            source_ref = artifact.open_artifact(source)
            matches, matching_ref = publish_matches(
                tmp, source_ref,
                [("LT0", "natural=peak; name=Mt Adams", 0.9)])
            with self.assertRaisesRegex(SystemExit, "100.0%.*absent"):
                tc.main(source, tmp / "blocked", None, 2000.0, 6.0, False,
                        matched_from=[matches])
            tc.main(
                source, tmp / "v2_trimmed", None, 2000.0, 6.0, False,
                matched_from=[matches], allow_absent_matched_signatures=True)
            manifest = artifact.load_manifest(tmp / "v2_trimmed")
        self.assertEqual(manifest.upstreams, (source_ref, matching_ref))
        self.assertTrue(
            manifest.config["recall_guards"][
                "allow_absent_matched_signatures"])

    def test_exactly_40_percent_absent_does_not_need_the_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                ("a", {"man_made": "tower"}, Point(-71.0, 42.30)),
                ("b", {"man_made": "lighthouse"}, Point(-71.0, 42.31)),
                ("c", {"man_made": "pier"}, Point(-71.0, 42.32)),
            ]
            source = publish_catalog(root, rows)
            matches, _ = publish_matches(
                root, artifact.open_artifact(source), [
                    ("LT0", "man_made=tower", 0.9),
                    ("LT1", "man_made=lighthouse", 0.9),
                    ("LT2", "man_made=pier", 0.9),
                    ("LT3", "natural=peak; name=A", 0.9),
                    ("LT4", "natural=peak; name=B", 0.9),
                ])
            tc.main(source, root / "trimmed", None, 2000.0, 6.0, False,
                    matched_from=[matches])
            self.assertTrue((root / "trimmed" / "manifest.json").is_file())

    def test_passes_when_the_matched_signature_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = publish_catalog(
                tmp, [("osm:node:1", {"man_made": "lighthouse",
                                       "name": "Boston Light"},
                       Point(-70.89, 42.32))])
            matches, _ = publish_matches(
                tmp, artifact.open_artifact(source),
                [("LT0", "man_made=lighthouse; name=Boston Light", 0.9)])
            tc.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False,
                    matched_from=[matches])
            record = trim_config(tmp / "v2_trimmed")
        self.assertEqual(record["rows_out"], 1)


class WriteProtectionTest(unittest.TestCase):
    """A catalog is part of the problem definition, so it is versioned rather
    than overwritten -- every past number was computed against the old one."""

    def build(self, tmp: Path) -> Path:
        return publish_catalog(
            tmp, [("osm:node:1", {"man_made": "lighthouse"},
                   Point(-70.89, 42.32))])

    def test_existing_catalog_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = self.build(tmp)
            tc.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False)
            with self.assertRaises(SystemExit) as caught:
                tc.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False)
        self.assertIn("immutable and versioned", str(caught.exception))

    def test_dry_run_never_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = self.build(tmp)
            tc.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, True)
            self.assertFalse((tmp / "v2_trimmed").exists())

    def test_loose_feather_is_not_accepted_as_an_input_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            loose = tmp / "catalog.feather"
            catalog([("a", {"man_made": "tower"}, Point(-71.0, 42.3))]
                    ).to_feather(loose)
            with self.assertRaises(SystemExit) as caught:
                tc.main(loose, tmp / "v2", None, 2000.0, 6.0, False)
        self.assertIn("invalid input catalog artifact", str(caught.exception))

    def test_non_wgs84_catalog_is_rejected_before_spatial_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = publish_catalog(
                root,
                [("a", {"man_made": "tower"}, Point(0.0, 0.0))],
                crs="EPSG:3857")
            with self.assertRaisesRegex(
                    SystemExit, "exactly WGS84.*EPSG:4326"):
                tc.main(source, root / "v2", None, 2000.0, 6.0, False)
            self.assertFalse((root / "v2").exists())


class CliTest(unittest.TestCase):

    def test_help_renders_percent_literal(self):
        with self.assertRaises(SystemExit) as caught:
            tc.cli(["--help"])
        self.assertEqual(caught.exception.code, 0)


class PositiveSetIdentityTest(unittest.TestCase):

    @staticmethod
    def write_positive(path: Path, matching_dir: Path) -> Path:
        positive_set.main(matching_dir, path)
        return path

    def test_exact_matching_and_catalog_identities_are_validated_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = publish_catalog(
                tmp, [("a", {"man_made": "tower"}, Point(-71.0, 42.3))])
            source_ref = artifact.open_artifact(source)
            matching_dir, matching_ref = publish_matches(
                tmp, source_ref, [])
            positive = self.write_positive(
                tmp / "positive.json", matching_dir)
            output = tmp / "v2"
            tc.main(source, output, positive, 2000.0, 6.0, False)
            manifest = artifact.load_manifest(output)
            positive_sha256 = artifact.sha256_file(positive)
        self.assertEqual(manifest.upstreams, (source_ref, matching_ref))
        self.assertEqual(
            manifest.config["recall_guards"]["positive_set_sha256"],
            positive_sha256)

    def test_positive_set_from_a_prior_trim_of_the_full_catalog_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [("a", {"man_made": "tower"}, Point(-71.0, 42.3))]
            source = publish_catalog(root / "full", rows)
            source_ref = artifact.open_artifact(source)
            prior_trim = publish_catalog(
                root / "trimmed", rows, version="trimmed_v1",
                upstreams=(source_ref,))
            matching_dir, matching_ref = publish_matches(
                root / "matching", artifact.open_artifact(prior_trim),
                [("LT0", "man_made=tower", 0.9)])
            positive = self.write_positive(root / "positive.json", matching_dir)
            output = root / "trimmed_v2"
            tc.main(source, output, positive, 2000.0, 6.0, False)
            manifest = artifact.load_manifest(output)
        self.assertEqual(manifest.upstreams, (source_ref, matching_ref))

    def test_positive_set_from_another_catalog_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = publish_catalog(
                tmp / "source",
                [("a", {"man_made": "tower"}, Point(-71.0, 42.3))])
            other = publish_catalog(
                tmp / "other",
                [("b", {"man_made": "tower"}, Point(-70.0, 41.3))])
            matching_dir, _ = publish_matches(
                tmp / "other_matching", artifact.open_artifact(other), [])
            positive = self.write_positive(
                tmp / "positive.json", matching_dir)
            with self.assertRaises(SystemExit) as caught:
                tc.main(source, tmp / "v2", positive, 2000.0, 6.0, False)
            self.assertFalse((tmp / "v2").exists())
        self.assertIn("typed descendant", str(caught.exception))


class RuleFingerprintTest(unittest.TestCase):

    def test_stable_for_the_same_rules(self):
        self.assertEqual(tc.rule_fingerprint(2000.0, 6.0),
                         tc.rule_fingerprint(2000.0, 6.0))

    def test_changes_when_a_threshold_changes(self):
        self.assertNotEqual(tc.rule_fingerprint(2000.0, 6.0),
                            tc.rule_fingerprint(400.0, 6.0))

    def test_changes_when_a_rule_set_changes(self):
        original = tc.HARD_UNOBSERVABLE_TAGS
        try:
            tc.HARD_UNOBSERVABLE_TAGS = frozenset(
                original | {("amenity", "zzz")})
            changed = tc.rule_fingerprint(2000.0, 6.0)
        finally:
            tc.HARD_UNOBSERVABLE_TAGS = original
        self.assertNotEqual(changed, tc.rule_fingerprint(2000.0, 6.0))

    def test_no_clip_fingerprint_remains_compatible(self):
        self.assertEqual(tc.rule_fingerprint(2000.0, 6.0),
                         "074398ff556ba26a")

    def test_changes_with_exact_spatial_boundary_and_plan(self):
        first = tc.rule_fingerprint(
            2000.0, 6.0, clip_bbox_wsen=(-71.1, 42.2, -70.8, 42.5),
            clip_plan_digest="a" * 64)
        moved = tc.rule_fingerprint(
            2000.0, 6.0, clip_bbox_wsen=(-71.1, 42.2, -70.7, 42.5),
            clip_plan_digest="a" * 64)
        new_plan = tc.rule_fingerprint(
            2000.0, 6.0, clip_bbox_wsen=(-71.1, 42.2, -70.8, 42.5),
            clip_plan_digest="b" * 64)
        self.assertNotEqual(first, moved)
        self.assertNotEqual(first, new_plan)


class ClipBoxTest(unittest.TestCase):

    def test_keeps_inside_and_drops_outside(self):
        inside = Point(-71.08, 42.36)
        outside = Point(-71.08, 42.56)          # ~22 km north
        gdf = catalog([("a", {"man_made": "tower"}, inside),
                       ("b", {"man_made": "tower"}, outside)])
        mask = tc.clip_mask(gdf, 42.36, -71.08, 25.0)
        self.assertEqual(mask.tolist(), [True, False])

    def test_box_is_square_in_metres_not_degrees(self):
        """A degree of longitude is only ~82 km at Boston against ~111 km for
        a degree of latitude, so equal degree offsets are unequal distances:
        0.1 deg is 8.2 km east but 11.1 km north. A box measured in degrees
        would admit or reject the pair together; measured in metres it splits
        them."""
        east = Point(-71.08 + 0.10, 42.36)     # 8.2 km east
        north = Point(-71.08, 42.36 + 0.10)    # 11.1 km north
        gdf = catalog([("e", {"man_made": "tower"}, east),
                       ("n", {"man_made": "tower"}, north)])
        self.assertEqual(tc.clip_mask(gdf, 42.36, -71.08, 25.0).tolist(),
                         [True, True])       # half-box 12.5 km: both inside
        self.assertEqual(tc.clip_mask(gdf, 42.36, -71.08, 20.0).tolist(),
                         [True, False])      # half-box 10 km: east one only

    def test_exact_bbox_is_inclusive_and_uses_representative_points(self):
        crossing = LineString([(-71.2, 42.35), (-71.0, 42.35)])
        gdf = catalog([
            ("west", {"man_made": "tower"}, Point(-71.1, 42.3)),
            ("north", {"man_made": "tower"}, Point(-71.0, 42.4)),
            ("outside", {"man_made": "tower"}, Point(-70.9, 42.3)),
            ("crossing", {"man_made": "pier"}, crossing),
        ])
        mask = tc.clip_bbox_mask(gdf, (-71.1, 42.3, -71.0, 42.4))
        self.assertEqual(mask.tolist(), [True, True, False, True])

    def test_invalid_or_mixed_bbox_configuration_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = publish_catalog(
                tmp, [("a", {"man_made": "tower"},
                       Point(-71.0, 42.3))])
            for bbox in [(-71.0, 42.3, -71.1, 42.4),
                         (-71.1, 42.3, float("nan"), 42.4)]:
                with self.subTest(bbox=bbox), self.assertRaises(SystemExit):
                    tc.main(source, tmp / "v2", None, 2000.0, 6.0, False,
                            clip_bbox_wsen=bbox)
            with self.assertRaises(SystemExit) as caught:
                tc.main(source, tmp / "v2", None, 2000.0, 6.0, False,
                        clip_km=25.0, clip_center_lat=42.35,
                        clip_center_lon=-71.05,
                        clip_bbox_wsen=(-71.1, 42.3, -71.0, 42.4))
        self.assertIn("mutually exclusive", str(caught.exception))

    def test_clip_without_a_centre_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = publish_catalog(
                tmp, [("a", {"man_made": "tower"}, Point(-71.0, 42.3))])
            with self.assertRaises(SystemExit) as caught:
                tc.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False,
                        clip_km=25.0)
        self.assertIn("impossible to reproduce", str(caught.exception))

    def test_clip_is_reported_as_its_own_rule_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = publish_catalog(
                tmp,
                [("a", {"man_made": "tower"}, Point(-71.08, 42.36)),
                 ("b", {"man_made": "tower"}, Point(-71.08, 42.56))])
            tc.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False,
                    clip_km=25.0,
                    clip_center_lat=42.36, clip_center_lon=-71.08)
            record = trim_config(tmp / "v2_trimmed")
        self.assertEqual(record["drops_per_rule"]["outside_clip_box"], 1)
        self.assertEqual(record["rows_out"], 1)
        self.assertEqual(record["clip_km"], 25.0)

    def test_exact_bbox_and_reviewed_plan_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            farfield_root = tmp / "farfield"
            plan, bbox = reviewed_clip_plan(farfield_root)
            digest = artifact.sha256_json(plan)
            plan_path = tmp / "clip_plan.json"
            plan_path.write_text(json.dumps(plan))
            source = publish_catalog(
                tmp / "source",
                [("inside", {"man_made": "tower"}, Point(-71.02, 42.32)),
                 ("outside", {"man_made": "tower"},
                  Point(-70.5, 42.32))], dataset=BOSTON_LEG1)
            tc.main(
                source, tmp / "v2_trimmed", None, 2000.0, 6.0, False,
                clip_bbox_wsen=bbox, clip_plan_path=plan_path,
                expected_clip_plan_digest=digest,
                farfield_root=farfield_root)
            record = trim_config(tmp / "v2_trimmed")
            expected_sources = tc.verify_clip_plan_sources(
                plan, farfield_root)
        self.assertEqual(record["clip_mode"], "bbox_wsen")
        self.assertEqual(record["clip_bbox_wsen"], list(bbox))
        self.assertEqual(record["clip_plan"], plan)
        self.assertEqual(record["clip_plan_digest"], digest)
        self.assertEqual(record["clip_plan_source_verification"],
                         expected_sources)
        self.assertEqual(record["drops_per_rule"]["outside_clip_bbox"], 1)
        self.assertEqual(record["rows_out"], 1)
        self.assertEqual(
            record["rule_fingerprint"],
            tc.rule_fingerprint(
                2000.0, 6.0, clip_bbox_wsen=bbox,
                clip_plan_digest=digest))

    def test_edited_or_mismatched_clip_plan_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            farfield_root = tmp / "farfield"
            plan, bbox = reviewed_clip_plan(farfield_root)
            digest = artifact.sha256_json(plan)
            plan_path = tmp / "clip_plan.json"
            source = publish_catalog(
                tmp / "source",
                [("inside", {"man_made": "tower"},
                  Point(-71.02, 42.32))], dataset=BOSTON_LEG1)
            plan_path.write_text(json.dumps(plan))
            with self.assertRaises(SystemExit) as caught:
                tc.main(
                    source, tmp / "v2", None, 2000.0, 6.0, False,
                    clip_bbox_wsen=bbox, clip_plan_path=plan_path,
                    expected_clip_plan_digest="0" * 64,
                    farfield_root=farfield_root)
            self.assertIn("digest mismatch", str(caught.exception))

            plan["bbox_wsen"][0] += 0.01
            plan_path.write_text(json.dumps(plan))
            with self.assertRaises(SystemExit) as caught:
                tc.main(
                    source, tmp / "v2", None, 2000.0, 6.0, False,
                    clip_bbox_wsen=bbox, clip_plan_path=plan_path,
                    expected_clip_plan_digest=artifact.sha256_json(plan),
                    farfield_root=farfield_root)
            self.assertIn("does not exactly match", str(caught.exception))
            self.assertFalse((tmp / "v2").exists())

    def test_clip_plan_json_rejects_duplicate_keys_and_nonfinite_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plan, bbox = reviewed_clip_plan(tmp / "farfield")
            plan_path = tmp / "clip_plan.json"

            encoded = json.dumps(plan)
            resolved_number = json.dumps(
                plan["policy"]["resolved_buffer_km"])
            resolved_field = f'"resolved_buffer_km": {resolved_number}'
            duplicate = encoded.replace(
                f'"scope": "{BOSTON_SCOPE}"',
                f'"scope": "{BOSTON_SCOPE}", "scope": "shadow"')
            plan_path.write_text(duplicate)
            with self.assertRaises(ValueError) as caught:
                tc.load_clip_plan(
                    plan_path, "0" * 64, bbox,
                    output_dataset=BOSTON_LEG1)
            self.assertIn("duplicate JSON object key", str(caught.exception))

            nonfinite = encoded.replace(
                resolved_field,
                '"resolved_buffer_km": NaN')
            plan_path.write_text(nonfinite)
            with self.assertRaises(ValueError) as caught:
                tc.load_clip_plan(
                    plan_path, "0" * 64, bbox,
                    output_dataset=BOSTON_LEG1)
            self.assertIn("non-finite JSON constant", str(caught.exception))

            overflow = encoded.replace(
                resolved_field,
                '"resolved_buffer_km": 1e9999')
            plan_path.write_text(overflow)
            with self.assertRaises(ValueError) as caught:
                tc.load_clip_plan(
                    plan_path, "0" * 64, bbox,
                    output_dataset=BOSTON_LEG1)
            self.assertIn("non-finite JSON number", str(caught.exception))

    def test_clip_plan_objects_have_exact_key_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, bbox = reviewed_clip_plan(Path(tmp) / "farfield")

            missing_top = dict(plan)
            del missing_top["scope"]
            extra_top = dict(plan, surprise=True)
            missing_policy = dict(plan)
            missing_policy["policy"] = dict(plan["policy"])
            del missing_policy["policy"]["minimum_area_km2"]
            extra_policy = dict(plan)
            extra_policy["policy"] = dict(plan["policy"], surprise=True)

            cases = [
                (missing_top, "top-level fields has missing"),
                (extra_top, "top-level fields has unknown"),
                (missing_policy, "policy fields has missing"),
                (extra_policy, "policy fields has unknown"),
            ]
            for document, message in cases:
                with self.subTest(message=message), self.assertRaises(
                        ValueError) as caught:
                    tc.validate_clip_plan(
                        document, bbox, output_dataset=BOSTON_LEG1)
                self.assertIn(message, str(caught.exception))

    def test_clip_plan_verifies_live_canonical_gps_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            farfield_root = tmp / "farfield"
            plan, bbox = reviewed_clip_plan(farfield_root)
            plan_path = tmp / "clip_plan.json"
            plan_path.write_text(json.dumps(plan))
            gps_path = (farfield_root / "datasets" / BOSTON_LEG1 /
                        "frames_gps.csv")
            gps_path.write_text(gps_path.read_text() +
                                "99,42.3,-71.05,0,0\n")
            with self.assertRaises(ValueError) as caught:
                loaded, _ = tc.load_clip_plan(
                    plan_path, artifact.sha256_json(plan), bbox,
                    output_dataset=BOSTON_LEG1)
                tc.verify_clip_plan_sources(loaded, farfield_root)
            self.assertIn("cannot verify canonical dataset",
                          str(caught.exception))

    def test_clip_plan_is_bound_to_active_scope_and_exact_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, bbox = reviewed_clip_plan(Path(tmp) / "farfield")
            wrong_output = copy.deepcopy(plan)
            wrong_output["output_dataset"] = "boston_harbor_leg2"
            wrong_bbox_set = copy.deepcopy(plan)
            wrong_bbox_set["bbox_datasets"] = [BOSTON_LEG1]
            unknown_scope = copy.deepcopy(plan)
            unknown_scope["scope"] = "made_up_scope"

            cases = [
                (wrong_output, "output_dataset does not exactly match"),
                (wrong_bbox_set, "bbox_datasets does not exactly match"),
                (unknown_scope, "SCOPE_BY_NAME"),
            ]
            for document, message in cases:
                with self.subTest(message=message), self.assertRaises(
                        ValueError) as caught:
                    tc.validate_clip_plan(
                        document, bbox, output_dataset=BOSTON_LEG1)
                self.assertIn(message, str(caught.exception))

    def test_boston_scope_policy_rejects_area_below_625(self):
        with tempfile.TemporaryDirectory() as tmp:
            farfield_root = Path(tmp) / "farfield"
            plan, bbox = reviewed_clip_plan(farfield_root)
            with self.assertRaisesRegex(
                    ValueError, "trusted scope floor of 625"):
                tc.build_clip_plan(
                    scope_name=BOSTON_SCOPE,
                    output_dataset=BOSTON_LEG1,
                    farfield_root=farfield_root,
                    nominal_buffer_km=5.0,
                    minimum_buffer_km=1.0,
                    minimum_area_km2=624.999)

            below = copy.deepcopy(plan)
            below["policy"]["minimum_area_km2"] = 624.999
            with self.assertRaisesRegex(
                    ValueError, "trusted scope floor of 625"):
                tc.validate_clip_plan(
                    below, bbox, output_dataset=BOSTON_LEG1)

    def test_boston_scope_policy_accepts_625_and_larger(self):
        with tempfile.TemporaryDirectory() as tmp:
            farfield_root = Path(tmp) / "farfield"
            at_floor, floor_bbox = reviewed_clip_plan(farfield_root)
            self.assertEqual(at_floor["scope_policy"], {
                "schema": tc.CLIP_SCOPE_POLICY_SCHEMA,
                "minimum_area_floor_km2": 625.0,
                "require_reviewed_bbox_plan": True,
            })
            tc.validate_clip_plan(
                at_floor, floor_bbox, output_dataset=BOSTON_LEG1)

            larger = tc.build_clip_plan(
                scope_name=BOSTON_SCOPE,
                output_dataset=BOSTON_LEG1,
                farfield_root=farfield_root,
                nominal_buffer_km=5.0,
                minimum_buffer_km=1.0,
                minimum_area_km2=900.0)
            tc.validate_clip_plan(
                larger, tuple(larger["bbox_wsen"]),
                output_dataset=BOSTON_LEG1)

    def test_boston_exact_bbox_requires_reviewed_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = publish_catalog(
                root / "source",
                [("a", {"man_made": "tower"}, Point(-71.0, 42.3))],
                dataset=BOSTON_LEG1)
            with self.assertRaisesRegex(
                    SystemExit, "governed dataset requires --clip_plan"):
                tc.main(
                    source, root / "v2", None, 2000.0, 6.0, True,
                    clip_bbox_wsen=(-71.1, 42.2, -70.9, 42.4))
            self.assertFalse((root / "v2").exists())

    def test_non_governed_exact_bbox_remains_available_without_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = publish_catalog(
                root / "source",
                [("a", {"man_made": "tower"}, Point(-71.0, 42.3))])
            result = tc.main(
                source, root / "v2", None, 2000.0, 6.0, True,
                clip_bbox_wsen=(-71.1, 42.2, -70.9, 42.4))
            self.assertEqual(len(result), 1)
            self.assertFalse((root / "v2").exists())

    def test_recorded_area_one_ulp_below_minimum_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, bbox = reviewed_clip_plan(Path(tmp) / "farfield")
            below = copy.deepcopy(plan)
            minimum = below["policy"]["minimum_area_km2"]
            below["policy"]["resolved_area_km2"] = math.nextafter(
                minimum, -math.inf)
            with self.assertRaisesRegex(
                    ValueError, "resolved_area_km2 is below"):
                tc.validate_clip_plan(
                    below, bbox, output_dataset=BOSTON_LEG1)

    def test_plan_table_paths_are_records_not_read_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            farfield_root = Path(tmp) / "farfield"
            plan, bbox = reviewed_clip_plan(farfield_root)
            redirected = copy.deepcopy(plan)
            redirected["dataset_tables"][0]["frames_gps"]["path"] = \
                "/tmp/attacker-selected-frames_gps.csv"
            loaded = tc.validate_clip_plan(
                redirected, bbox, output_dataset=BOSTON_LEG1)
            with self.assertRaisesRegex(
                    ValueError, "tables no longer match"):
                tc.verify_clip_plan_sources(loaded, farfield_root)

    def test_arbitrary_minimum_area_uses_the_exact_positive_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, bbox = reviewed_clip_plan(
                Path(tmp) / "farfield", minimum_area_km2=777.0)
            policy = plan["policy"]
            self.assertAlmostEqual(policy["resolved_area_km2"], 777.0,
                                   places=9)
            self.assertGreaterEqual(policy["resolved_area_km2"], 777.0)
            self.assertGreater(policy["resolved_buffer_km"], 5.0)
            tc.validate_clip_plan(
                plan, bbox, output_dataset=BOSTON_LEG1)

            # A larger buffer satisfies every inequality but is not the
            # uniquely reviewed minimum and therefore must fail closed.
            padded = copy.deepcopy(plan)
            track = tuple(policy["track_bbox_wsen"])
            buffer_km = policy["resolved_buffer_km"] + 1.0
            width, height, km_lon, km_lat, _ = tc._track_metrics(track)
            padded["policy"]["resolved_buffer_km"] = buffer_km
            padded["policy"]["resolved_area_km2"] = \
                (width + 2 * buffer_km) * (height + 2 * buffer_km)
            west, south, east, north = track
            padded_bbox = (
                west - buffer_km / km_lon, south - buffer_km / km_lat,
                east + buffer_km / km_lon, north + buffer_km / km_lat)
            padded["bbox_wsen"] = list(padded_bbox)
            with self.assertRaisesRegex(
                    ValueError, "not the exact policy minimum"):
                tc.validate_clip_plan(
                    padded, padded_bbox, output_dataset=BOSTON_LEG1)

    def test_plan_numbers_require_actual_finite_numeric_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, bbox = reviewed_clip_plan(Path(tmp) / "farfield")
            values = ["5", True, 10 ** 10000]
            for value in values:
                altered = copy.deepcopy(plan)
                altered["policy"]["nominal_buffer_km"] = value
                with self.subTest(value=type(value).__name__), \
                     self.assertRaises(ValueError):
                    tc.validate_clip_plan(
                        altered, bbox, output_dataset=BOSTON_LEG1)

            for value in ["-71", True, 10 ** 10000]:
                with self.subTest(bbox_value=type(value).__name__), \
                     self.assertRaises(ValueError):
                    tc.validate_bbox_wsen((value, 42.0, -70.0, 43.0))

    def test_surrogate_json_and_plan_symlinks_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plan, bbox = reviewed_clip_plan(tmp / "farfield")
            surrogate = copy.deepcopy(plan)
            surrogate["dataset_tables"][0]["frames_gps"]["path"] = "\ud800"
            real_plan = tmp / "real-plan.json"
            real_plan.write_text(json.dumps(surrogate))
            with self.assertRaisesRegex(ValueError, "canonical JSON"):
                tc.load_clip_plan(
                    real_plan, "0" * 64, bbox,
                    output_dataset=BOSTON_LEG1)

            real_plan.write_text(json.dumps(plan))
            linked_plan = tmp / "linked-plan.json"
            linked_plan.symlink_to(real_plan.name)
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                tc.load_clip_plan(
                    linked_plan, artifact.sha256_json(plan), bbox,
                    output_dataset=BOSTON_LEG1)

    def test_sources_are_reverified_inside_publication_transaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            farfield_root = tmp / "farfield"
            plan, bbox = reviewed_clip_plan(farfield_root)
            digest = artifact.sha256_json(plan)
            plan_path = tmp / "clip-plan.json"
            plan_path.write_text(json.dumps(plan))
            source = publish_catalog(
                tmp / "source",
                [("inside", {"man_made": "tower"},
                  Point(-71.02, 42.32))], dataset=BOSTON_LEG1)
            output = tmp / "v2"
            initial = tc.verify_clip_plan_sources(plan, farfield_root)
            with mock.patch.object(
                    tc, "verify_clip_plan_sources",
                    side_effect=[initial, ValueError("changed at publish")]), \
                 self.assertRaisesRegex(
                     SystemExit, "changed before publication"):
                tc.main(
                    source, output, None, 2000.0, 6.0, False,
                    clip_bbox_wsen=bbox, clip_plan_path=plan_path,
                    expected_clip_plan_digest=digest,
                    farfield_root=farfield_root)
            self.assertFalse(output.exists())


class ProvenanceTest(unittest.TestCase):
    """The typed manifest makes the catalog reproducible and immutable."""

    def run_once(self, tmp: Path):
        source = publish_catalog(
            tmp,
            [("a", {"man_made": "lighthouse", "name": "Boston Light"},
              Point(-70.89, 42.32))])
        output = tmp / "v2_trimmed"
        tc.main(source, output, None, 2000.0, 6.0, False,
                clip_km=25.0, clip_center_lat=42.32,
                clip_center_lon=-70.89)
        return artifact.load_manifest(output), artifact.open_artifact(source)

    def test_records_exact_upstream_and_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, source_ref = self.run_once(Path(tmp))
        self.assertEqual(manifest.kind, paths_lib.CATALOGS)
        self.assertEqual(manifest.upstreams, (source_ref,))
        self.assertEqual(manifest.declared_outputs, ("catalog.feather",))
        self.assertEqual(manifest.config["min_building_area_m2"], 2000.0)
        self.assertEqual(manifest.config["min_building_levels"], 6.0)
        self.assertEqual(manifest.config["clip_km"], 25.0)
        self.assertEqual(manifest.config["clip_center_lat"], 42.32)
        self.assertEqual(manifest.config["clip_center_lon"], -70.89)
        self.assertEqual(manifest.config["clip_mode"], "metric_square")
        self.assertIsNone(manifest.config["clip_bbox_wsen"])
        self.assertIsNone(manifest.config["clip_plan_digest"])
        self.assertIsNone(
            manifest.config["clip_plan_source_verification"])
        self.assertEqual(manifest.config["rows_out"], 1)
        self.assertTrue(manifest.config["rule_fingerprint"])
        self.assertTrue(manifest.git_commit)


if __name__ == "__main__":
    unittest.main()
