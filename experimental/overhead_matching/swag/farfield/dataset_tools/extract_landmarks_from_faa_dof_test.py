import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield.catalog import catalog, schema
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    extract_landmarks_from_faa_dof as faa,
)
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    source_publication,
)

HEADER = [
    "  CURRENCY DATE = 08/02/26",
    "                                     LATITUDE    LONGITUDE     OBSTACLE"
    "            AGL   AMSL LT ACC MAR FAA         ACTION",
    "OAS#      V CO ST CITY            DEG MIN SEC   DEG MIN SEC   TYPE"
    "                 HT    HT     H V IND STUDY           JDATE",
    "-" * 127,
]
BBOX = (-70.5, 43.5, -69.8, 44.2)


def line(oas, ver, city, lat, lon, typ, agl, amsl, light="D", hacc="4",
         vacc="D", mark="N", study="2015ANE00210OE", action="C",
         jdate="2017153", qty="1"):
    text = (f"{oas:9s} {ver} US ME {city:16s} {lat} {lon} {typ:18s} {qty} "
            f"{agl:05d} {amsl:05d} {light} {hacc} {vacc} {mark} {study:14s} "
            f"{action} {jdate} ")
    assert len(text) == 128, len(text)
    return text


ROWS = [
    # Verified tower inside the box, accuracy code 4.
    line("23-000127", "O", "GRAY", "43 51 06.46N", "070 19 38.43W",
         "TOWER", 702, 1171, light="N", hacc="1", vacc="A", mark="N",
         study="", action="A", jdate="2014310"),
    # Unverified transmission-line tower inside the box.
    line("23-080577", "U", "WALES", "44 07 00.00N", "070 03 00.00W",
         "T-L TWR", 120, 400),
    # Verified building, no AGL height recorded.
    line("23-000200", "O", "PORTLAND", "43 39 12.00N", "070 16 32.00W",
         "BLDG", 0, 310, hacc="9", vacc="I"),
    # Airport furniture: dropped by type.
    line("23-000300", "O", "PORTLAND", "43 38 20.00N", "070 17 44.00W",
         "POLE", 105, 129, hacc="1", vacc="A"),
    # Unknown type: dropped and counted.
    line("23-000400", "O", "PORTLAND", "43 38 20.00N", "070 17 44.00W",
         "ZEPPELIN MAST", 90, 120),
    # Outside the box (Kittery).
    line("23-021802", "O", "KITTERY", "43 04 47.75N", "070 45 08.72W",
         "BRIDGE", 204, 204),
]


def write_dat(path: Path, rows=ROWS) -> Path:
    path.write_text("\r\n".join(HEADER + rows) + "\r\n", encoding="latin-1")
    return path


class ExtractLandmarksFromFaaDofTest(unittest.TestCase):

    def test_parses_the_readme_layout(self):
        currency, records = faa.read_records(
            write_dat(Path(tempfile.mkdtemp()) / "23-ME.Dat"))
        self.assertEqual(currency, "2026-08-02")
        self.assertEqual(len(records), len(ROWS))
        tower = records[0]
        self.assertEqual(tower["oas"], "23-000127")
        self.assertTrue(tower["verified"])
        self.assertAlmostEqual(tower["lat"], 43 + 51 / 60 + 6.46 / 3600)
        self.assertAlmostEqual(tower["lon"], -(70 + 19 / 60 + 38.43 / 3600))
        self.assertEqual((tower["type"], tower["agl_ft"], tower["amsl_ft"]),
                         ("TOWER", 702, 1171))
        self.assertEqual(tower["action_date"], "2014-11-06")
        self.assertEqual(records[2]["action_date"], "2017-06-02")

    def test_every_mapped_tag_survives_far_field_pruning(self):
        for faa_type, tags in faa.TYPE_TAGS.items():
            kept = catalog.prune_far_field_tags(dict(tags))
            self.assertEqual(kept, tags, faa_type)
        # faa:* facts are provenance, never matcher input.
        self.assertEqual(
            catalog.prune_far_field_tags(
                {"man_made": "tower", "height": "214.0",
                 "faa:oas": "23-000127", "faa:lighting": "red"}),
            {"man_made": "tower", "height": "214.0"})

    def test_extract_maps_filters_and_orders(self):
        currency, records = faa.read_records(
            write_dat(Path(tempfile.mkdtemp()) / "23-ME.Dat"))
        frame, report = faa.extract(records, currency, BBOX,
                                    verified_only=False)
        self.assertEqual(list(frame["id"]),
                         ["23-000127", "23-080577", "23-000200"])
        self.assertEqual(set(frame["landmark_type"]), {"faa"})
        tags = schema.tag_dicts(frame)
        self.assertEqual(tags[0]["man_made"], "tower")
        self.assertEqual(tags[0]["height"], "214.0")
        self.assertEqual(tags[0]["faa:position_tolerance_m"], "6.1")
        self.assertEqual(tags[0]["faa:lighting"], "none")
        self.assertEqual(tags[0]["faa:action_date"], "2014-11-06")
        self.assertNotIn("faa:study", tags[0])
        self.assertEqual(tags[1]["power"], "tower")
        self.assertEqual(tags[1]["faa:verified"], "no")
        self.assertEqual(tags[1]["faa:position_tolerance_m"], "76.2")
        self.assertEqual(tags[2]["building"], "yes")
        self.assertNotIn("height", tags[2])
        self.assertNotIn("faa:position_tolerance_m", tags[2])
        self.assertEqual(report["rows_in"], 6)
        self.assertEqual(report["rows_out"], 3)
        self.assertEqual(report["dropped"],
                         {"dropped_type": 1, "outside_bbox": 1})
        self.assertEqual(report["unmapped_types"], {"ZEPPELIN MAST": 1})
        self.assertEqual(report["by_type"],
                         {"BLDG": 1, "T-L TWR": 1, "TOWER": 1})

        verified, report = faa.extract(records, currency, BBOX,
                                       verified_only=True)
        self.assertEqual(list(verified["id"]), ["23-000127", "23-000200"])
        self.assertEqual(report["dropped"]["unverified"], 1)

    def test_main_publishes_source_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dat = write_dat(root / "23-ME.Dat")
            output = root / "faa_dof_20260802_v1"
            frame = faa.main(dat, BBOX, True, output)
            self.assertEqual(len(frame), 2)
            feather, sidecar, _ = source_publication.output_paths(output)
            schema.read_frame(feather)
            document = json.loads(sidecar.read_text())
            self.assertEqual(document["arguments"]["currency_date"],
                             "2026-08-02")
            self.assertTrue(document["arguments"]["verified_only"])
            self.assertEqual(document["report"]["rows_out"], 2)
            # A second run reuses the exact completed source.
            self.assertEqual(len(faa.main(dat, BBOX, True, output)), 2)

    def test_rejects_file_without_currency_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            dat = Path(tmp) / "bad.Dat"
            dat.write_text("\n".join(ROWS), encoding="latin-1")
            with self.assertRaises(ValueError):
                faa.read_records(dat)


if __name__ == "__main__":
    unittest.main()
