"""Tests for download_enc_cells (hermetic — no network)."""

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from experimental.overhead_matching.swag.farfield import provenance
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


def make_cell_zip(cell: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"ENC_ROOT/{cell}/{cell}.000", b"fake s57 base file")
        zf.writestr(f"ENC_ROOT/{cell}/{cell}.001", b"fake update")
        zf.writestr("ENC_ROOT/README.TXT", b"readme")
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
            with self.assertRaises(FileNotFoundError):
                dec.download_cell("US5BOSCD", Path(tmp), fetch_fn=fake_fetch)


class MainTest(unittest.TestCase):
    def test_catalog_driven_main(self):
        def fake_fetch(url: str) -> bytes:
            if url.endswith(".xml"):
                return CATALOG_SNIPPET.encode()
            cell = url.rsplit("/", 1)[1].removesuffix(".zip")
            return make_cell_zip(cell)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cell_dirs = dec.main(
                cells=None, catalog_state="MA",
                bbox=(-71.05, 42.32, -70.94, 42.40), band=5,
                output_dir=out, force=False, fetch_fn=fake_fetch)
            self.assertEqual(
                [d.name for d in cell_dirs], ["US5BOSCD", "US5BOSDD"])
            self.assertTrue(
                (out / "catalogs" / "MA_ENCProdCat_19115.xml").exists())

    def test_writes_a_provenance_manifest(self):
        """The audit found ENC directories with no record of what was
        fetched or when; every run now leaves one."""
        def fake_fetch(url: str) -> bytes:
            return make_cell_zip("US5BOSCD")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            dec.main(cells=["US5BOSCD"], catalog_state=None, bbox=None,
                     band=5, output_dir=out, force=False,
                     fetch_fn=fake_fetch)
            manifest = json.loads(
                (out / provenance.MANIFEST_NAME).read_text())
        self.assertEqual(manifest["config"]["cells"], ["US5BOSCD"])
        self.assertTrue(manifest["config"]["explicit_cells"])
        self.assertIn("git_commit", manifest)
        self.assertIn("created", manifest)


if __name__ == "__main__":
    unittest.main()
