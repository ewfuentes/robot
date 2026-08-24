"""Tests for download_enc_cells (hermetic — no network)."""

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from experimental.overhead_matching.swag.farfield.dataset_tools import (
    download_enc_cells as dec,
)

CATALOG_SNIPPET = """<?xml version="1.0" encoding="UTF-8"?>
<DS_Series xmlns:gml="http://www.opengis.net/gml/3.2">
  <polygon>
    <gml:Polygon gml:id="US5BOSCD_P1">
      <gml:exterior><gml:LinearRing>
        <gml:pos>42.3 -71.0249677</gml:pos>
        <gml:pos>42.3 -70.95</gml:pos>
        <gml:pos>42.375 -70.95</gml:pos>
        <gml:pos>42.375 -71.025</gml:pos>
        <gml:pos>42.3 -71.025</gml:pos>
      </gml:LinearRing></gml:exterior>
    </gml:Polygon>
  </polygon>
  <polygon>
    <gml:Polygon gml:id="US5BOSDD_P1">
      <gml:exterior><gml:LinearRing>
        <gml:pos>42.375 -71.025</gml:pos>
        <gml:pos>42.375 -70.95</gml:pos>
        <gml:pos>42.45 -70.95</gml:pos>
        <gml:pos>42.45 -71.025</gml:pos>
      </gml:LinearRing></gml:exterior>
    </gml:Polygon>
  </polygon>
  <polygon>
    <gml:Polygon gml:id="US4MA1CC_P1">
      <gml:exterior><gml:LinearRing>
        <gml:pos>42.0 -71.2</gml:pos>
        <gml:pos>42.0 -70.6</gml:pos>
        <gml:pos>42.6 -70.6</gml:pos>
        <gml:pos>42.6 -71.2</gml:pos>
      </gml:LinearRing></gml:exterior>
    </gml:Polygon>
  </polygon>
  <polygon>
    <gml:Polygon gml:id="US5MA1KJ_P1">
      <gml:exterior><gml:LinearRing>
        <gml:pos>41.5 -70.7</gml:pos>
        <gml:pos>41.5 -70.6</gml:pos>
        <gml:pos>41.6 -70.6</gml:pos>
        <gml:pos>41.6 -70.7</gml:pos>
      </gml:LinearRing></gml:exterior>
    </gml:Polygon>
  </polygon>
</DS_Series>
"""


def make_cell_zip(cell: str, *, base=b"fake s57 base file",
                  updates=("001",)) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"ENC_ROOT/{cell}/{cell}.000", base)
        for suffix in updates:
            zf.writestr(f"ENC_ROOT/{cell}/{cell}.{suffix}",
                        f"fake update {suffix}".encode())
        zf.writestr("ENC_ROOT/README.TXT", b"shared NOAA readme")
    return buf.getvalue()


class ParseCatalogTest(unittest.TestCase):
    def test_parses_cell_bboxes(self):
        bboxes = dec.parse_catalog_cell_bboxes(CATALOG_SNIPPET)
        self.assertEqual(
            set(bboxes), {"US5BOSCD", "US5BOSDD", "US4MA1CC", "US5MA1KJ"})
        west, south, east, north = bboxes["US5BOSCD"]
        self.assertAlmostEqual(west, -71.025)
        self.assertAlmostEqual(south, 42.3)
        self.assertAlmostEqual(east, -70.95)
        self.assertAlmostEqual(north, 42.375)

    def test_bbox_selection_with_band_filter(self):
        bbox = (-71.05, 42.32, -70.94, 42.37)
        self.assertEqual(
            dec.select_cells_from_catalog(CATALOG_SNIPPET, bbox, band=5),
            ["US5BOSCD"])
        # Without the band filter the overlapping band-4 cell is included
        # too.
        self.assertEqual(
            dec.select_cells_from_catalog(CATALOG_SNIPPET, bbox, band=None),
            ["US4MA1CC", "US5BOSCD"])

    def test_bbox_selection_excludes_disjoint_cells(self):
        bbox = (-70.70, 41.50, -70.60, 41.60)
        self.assertEqual(
            dec.select_cells_from_catalog(CATALOG_SNIPPET, bbox, band=5),
            ["US5MA1KJ"])


class DownloadCellTest(unittest.TestCase):
    def test_download_unzips_and_is_idempotent(self):
        fetched_urls = []

        def fake_fetch(url: str) -> bytes:
            fetched_urls.append(url)
            return make_cell_zip("US5BOSCD")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cell_dir = dec.download_cell("US5BOSCD", out, fetch_fn=fake_fetch)
            self.assertEqual(
                fetched_urls, ["https://charts.noaa.gov/ENCs/US5BOSCD.zip"])
            self.assertTrue((cell_dir / "US5BOSCD.000").exists())
            self.assertTrue((cell_dir / "US5BOSCD.001").exists())
            manifest = dec.validate_cell(cell_dir, "US5BOSCD")
            self.assertEqual(manifest["schema"], dec.CELL_MANIFEST_SCHEMA)
            self.assertEqual(
                {record["path"] for record in manifest["files"]},
                {"US5BOSCD.000", "US5BOSCD.001"})
            self.assertEqual(len(manifest["archive_sha256"]), 64)

            # Second call skips the download entirely.
            dec.download_cell("US5BOSCD", out, fetch_fn=fake_fetch)
            self.assertEqual(len(fetched_urls), 1)

            # force=True re-downloads.
            dec.download_cell("US5BOSCD", out, force=True,
                              fetch_fn=fake_fetch)
            self.assertEqual(len(fetched_urls), 2)

    def test_unexpected_zip_layout_raises(self):
        def fake_fetch(url: str) -> bytes:
            return make_cell_zip("US5WRONGC")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "outside ENC_ROOT"):
                dec.download_cell("US5BOSCD", Path(tmp), fetch_fn=fake_fetch)

    def test_incomplete_existing_directory_is_not_a_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cell = out / "ENC_ROOT" / "US5BOSCD"
            cell.mkdir(parents=True)
            (cell / "US5BOSCD.000").write_bytes(b"unrecorded")
            with self.assertRaisesRegex(ValueError, "manifest"):
                dec.download_cell(
                    "US5BOSCD", out,
                    fetch_fn=lambda _: self.fail("must not fetch"))

    def test_cached_content_digest_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cell = dec.download_cell(
                "US5BOSCD", out,
                fetch_fn=lambda _: make_cell_zip("US5BOSCD"))
            (cell / "US5BOSCD.000").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "content digests"):
                dec.download_cell(
                    "US5BOSCD", out,
                    fetch_fn=lambda _: self.fail("must not fetch"))

    def test_force_replaces_the_whole_cell_without_vintage_mixing(self):
        payloads = iter([
            make_cell_zip("US5BOSCD", base=b"old", updates=("009",)),
            make_cell_zip("US5BOSCD", base=b"new", updates=()),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cell = dec.download_cell(
                "US5BOSCD", out, fetch_fn=lambda _: next(payloads))
            self.assertTrue((cell / "US5BOSCD.009").exists())
            dec.download_cell("US5BOSCD", out, force=True,
                              fetch_fn=lambda _: next(payloads))
            self.assertEqual((cell / "US5BOSCD.000").read_bytes(), b"new")
            self.assertFalse((cell / "US5BOSCD.009").exists())
            dec.validate_cell(cell, "US5BOSCD")

    def test_invalid_refresh_preserves_the_previous_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cell = dec.download_cell(
                "US5BOSCD", out,
                fetch_fn=lambda _: make_cell_zip("US5BOSCD", base=b"old"))
            old_manifest = (cell / dec.CELL_MANIFEST_NAME).read_bytes()
            with self.assertRaises(ValueError):
                dec.download_cell(
                    "US5BOSCD", out, force=True,
                    fetch_fn=lambda _: make_cell_zip("US5WRONGC"))
            self.assertEqual((cell / "US5BOSCD.000").read_bytes(), b"old")
            self.assertEqual((cell / dec.CELL_MANIFEST_NAME).read_bytes(),
                             old_manifest)

    def test_archive_path_escape_is_rejected(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("ENC_ROOT/US5BOSCD/US5BOSCD.000", b"base")
            zf.writestr("ENC_ROOT/US5BOSCD/../../escape", b"bad")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
                dec.download_cell("US5BOSCD", Path(tmp),
                                  fetch_fn=lambda _: buf.getvalue())

    def test_archive_without_exact_base_file_is_rejected(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("ENC_ROOT/US5BOSCD/US5BOSCD.001", b"update only")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "US5BOSCD.000"):
                dec.download_cell("US5BOSCD", Path(tmp),
                                  fetch_fn=lambda _: buf.getvalue())

    def test_malformed_manifest_digest_is_a_validation_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cell = dec.download_cell(
                "US5BOSCD", out,
                fetch_fn=lambda _: make_cell_zip("US5BOSCD"))
            manifest_path = cell / dec.CELL_MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text())
            manifest["archive_sha256"] = 7
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest identity"):
                dec.validate_cell(cell, "US5BOSCD")


class MainTest(unittest.TestCase):
    def test_catalog_driven_main(self):
        def fake_fetch(url: str) -> bytes:
            if url.endswith(".xml"):
                return CATALOG_SNIPPET.encode()
            cell = url.rsplit("/", 1)[1].removesuffix(".zip")
            return make_cell_zip(cell)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            selection = out / "selections" / "test.json"
            cell_dirs = dec.main(
                cells=None, catalog_state="MA",
                bbox=(-71.05, 42.32, -70.94, 42.40), band=5,
                output_dir=out, selection_output=selection,
                force=False, fetch_fn=fake_fetch)
            self.assertEqual(
                [d.name for d in cell_dirs], ["US5BOSCD", "US5BOSDD"])
            self.assertEqual(
                len(list((out / "catalogs").glob(
                    "MA_ENCProdCat_19115-*.xml"))),
                1)
            record = dec.validate_selection(selection, out)
            self.assertEqual(record["cells"], ["US5BOSCD", "US5BOSDD"])
            self.assertEqual(record["bbox"],
                             [-71.05, 42.32, -70.94, 42.40])
            self.assertEqual(record["band"], 5)
            self.assertFalse(record["explicit_cells"])
            self.assertEqual(len(record["catalog"]["sha256"]), 64)

    def test_writes_immutable_selection_with_exact_cell_identity(self):
        def fake_fetch(url: str) -> bytes:
            return make_cell_zip("US5BOSCD")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            selection = out / "selection.json"
            dec.main(cells=["US5BOSCD"], catalog_state=None, bbox=None,
                     band=5, output_dir=out, selection_output=selection,
                     force=False,
                     fetch_fn=fake_fetch)
            manifest = dec.validate_selection(selection, out)
            cell_manifest = json.loads(
                (out / "ENC_ROOT" / "US5BOSCD"
                 / dec.CELL_MANIFEST_NAME).read_text())
        self.assertEqual(manifest["cells"], ["US5BOSCD"])
        self.assertTrue(manifest["explicit_cells"])
        self.assertIsNone(manifest["catalog"])
        self.assertIn("git_commit", manifest)
        self.assertIn("created", manifest)
        self.assertEqual(len(manifest["cell_refs"]), 1)
        self.assertEqual(len(manifest["cell_refs"][0]["content_sha256"]), 64)
        self.assertEqual(cell_manifest["cell"], "US5BOSCD")
        self.assertEqual(len(cell_manifest["files"]), 2)

    def test_selection_exact_reuse_is_no_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            selection = out / "selection.json"
            arguments = dict(
                cells=["US5BOSCD"], catalog_state=None, bbox=None, band=5,
                output_dir=out, selection_output=selection, force=False,
                fetch_fn=lambda _: make_cell_zip("US5BOSCD"))
            dec.main(**arguments)
            before = selection.read_bytes()
            reused = dec.main(**{
                **arguments,
                "fetch_fn": lambda _: self.fail(
                    "exact selection reuse must not fetch"),
            })
            self.assertEqual(selection.read_bytes(), before)
            self.assertEqual(
                reused, [out / "ENC_ROOT" / "US5BOSCD"])
            with self.assertRaisesRegex(ValueError, "different invocation"):
                dec.main(**{**arguments, "band": 4})

    def test_catalog_selection_cells_are_recomputed_on_validation(self):
        def fake_fetch(url: str) -> bytes:
            if url.endswith(".xml"):
                return CATALOG_SNIPPET.encode()
            return make_cell_zip(
                url.rsplit("/", 1)[1].removesuffix(".zip"))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            selection = out / "selection.json"
            dec.main(
                cells=None, catalog_state="MA",
                bbox=(-71.05, 42.32, -70.94, 42.40), band=5,
                output_dir=out, selection_output=selection,
                force=False, fetch_fn=fake_fetch)
            document = json.loads(selection.read_text())
            document["cells"].pop()
            document["cell_refs"].pop()
            selection.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "exactly match"):
                dec.validate_selection(selection, out)


if __name__ == "__main__":
    unittest.main()
