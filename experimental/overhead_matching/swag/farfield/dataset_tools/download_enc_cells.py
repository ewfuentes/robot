"""Download NOAA ENC (Electronic Navigational Chart) cells for extraction.

NOAA distributes free S-57 vector chart cells, updated every weekday, as one
ZIP per cell at https://charts.noaa.gov/ENCs/<CELL>.zip. Each ZIP unpacks to
ENC_ROOT/<CELL>/<CELL>.000 (base file) plus incremental updates (.001, .002,
...) which GDAL's S-57 driver applies automatically on read.

Cells can be named explicitly, or selected by intersecting a bounding box
against a state's ISO-19115 product catalog
(https://charts.noaa.gov/ENCs/<ST>_ENCProdCat_19115.xml). Cell names encode
the usage band in the third character (1=Overview ... 5=Harbor, 6=Berthing);
band 5 is the right scale for harbor-sized areas.

`--output_dir` is required: three different hardcoded copies of one ENC root
existed on the checkpoint branch, and a default here is how a fourth appears.
Cells are raw material, so the conventional home is
`<farfield root>/raw_material/enc_cells/`. Every run writes a provenance
manifest (`manifest.json`) into the output dir recording what was fetched and
when — ENC cells change weekly, so "which vintage is this?" must be
answerable from the directory.

Example:
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:download_enc_cells -- \\
        --cells US5BOSCC US5BOSCD --output_dir /data/farfield_matching/raw_material/enc_cells

    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:download_enc_cells -- \\
        --catalog_state MA --bbox -71.08 42.24 -70.84 42.43 --band 5 \\
        --output_dir /data/farfield_matching/raw_material/enc_cells
"""

import argparse
import io
import re
import zipfile
from pathlib import Path
from typing import Callable

import requests

from experimental.overhead_matching.swag.farfield import provenance

ENC_BASE_URL = "https://charts.noaa.gov/ENCs"


def fetch_url(url: str) -> bytes:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def parse_catalog_cell_bboxes(catalog_xml: str) -> dict:
    """Extract per-cell bounding boxes from an ISO-19115 ENC product catalog.

    Cell extents appear as GML polygons `<gml:Polygon gml:id="<CELL>_P<n>">`
    whose vertices are `<gml:pos>lat lon</gml:pos>`. A cell may have several
    polygons (coverage panels); the bbox is the union over all of them.

    Returns:
        {cell_name: (west, south, east, north)}
    """
    bboxes: dict = {}
    polygon_re = re.compile(
        r'<gml:Polygon gml:id="(US\w+?)_P\d+">(.*?)</gml:Polygon>', re.S)
    pos_re = re.compile(r"<gml:pos>\s*([-\d.]+)\s+([-\d.]+)\s*</gml:pos>")
    for match in polygon_re.finditer(catalog_xml):
        cell = match.group(1)
        lats, lons = [], []
        for pos in pos_re.finditer(match.group(2)):
            lats.append(float(pos.group(1)))
            lons.append(float(pos.group(2)))
        if not lats:
            continue
        west, south, east, north = min(lons), min(lats), max(lons), max(lats)
        if cell in bboxes:
            ow, os_, oe, on = bboxes[cell]
            west, south = min(west, ow), min(south, os_)
            east, north = max(east, oe), max(north, on)
        bboxes[cell] = (west, south, east, north)
    return bboxes


def select_cells_from_catalog(
    catalog_xml: str,
    bbox: tuple,
    band: int | None,
) -> list:
    """Cell names whose catalog extent intersects bbox (west, south, east,
    north)."""
    west, south, east, north = bbox
    selected = []
    for cell, (cw, cs, ce, cn) in sorted(
            parse_catalog_cell_bboxes(catalog_xml).items()):
        if band is not None and int(cell[2]) != band:
            continue
        if cw <= east and ce >= west and cs <= north and cn >= south:
            selected.append(cell)
    return selected


def download_cell(
    cell: str,
    output_dir: Path,
    force: bool = False,
    fetch_fn: Callable[[str], bytes] = fetch_url,
) -> Path:
    """Download and unzip one ENC cell. Returns the cell directory.

    The ZIP's own layout (ENC_ROOT/<CELL>/...) is preserved under output_dir.
    Existing cell directories are skipped unless force=True.
    """
    cell_dir = output_dir / "ENC_ROOT" / cell
    if cell_dir.exists() and not force:
        print(f"{cell}: already present at {cell_dir}, skipping "
              f"(use --force to refresh)")
        return cell_dir

    url = f"{ENC_BASE_URL}/{cell}.zip"
    print(f"{cell}: downloading {url}")
    payload = fetch_fn(url)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(output_dir)
    if not cell_dir.exists():
        raise FileNotFoundError(
            f"{cell}: ZIP did not contain ENC_ROOT/{cell}/ "
            f"(unexpected layout)")
    print(f"{cell}: extracted to {cell_dir}")
    return cell_dir


def main(
    cells: list | None,
    catalog_state: str | None,
    bbox: tuple | None,
    band: int | None,
    output_dir: Path,
    force: bool,
    fetch_fn: Callable[[str], bytes] = fetch_url,
) -> list:
    requested = cells
    if cells is None:
        if catalog_state is None or bbox is None:
            raise ValueError(
                "Provide either --cells, or --catalog_state with --bbox")
        catalog_url = f"{ENC_BASE_URL}/{catalog_state}_ENCProdCat_19115.xml"
        print(f"Fetching catalog {catalog_url}")
        catalog_xml = fetch_fn(catalog_url).decode("utf-8", errors="replace")
        catalog_dir = output_dir / "catalogs"
        catalog_dir.mkdir(parents=True, exist_ok=True)
        (catalog_dir / f"{catalog_state}_ENCProdCat_19115.xml").write_text(
            catalog_xml)
        cells = select_cells_from_catalog(catalog_xml, bbox, band)
        print(f"Catalog selection (band={band}, bbox={bbox}): "
              f"{len(cells)} cells: {cells}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cell_dirs = [download_cell(cell, output_dir, force=force,
                               fetch_fn=fetch_fn) for cell in cells]
    # ENC cells are updated weekly; the manifest answers "which vintage is
    # this directory?" without a NOAA query. Latest run wins, which is right:
    # the newest fetch describes the on-disk state.
    provenance.write(
        output_dir,
        generator="farfield/dataset_tools/download_enc_cells.py",
        inputs={"source": ENC_BASE_URL,
                "catalog_state": catalog_state or ""},
        config={"cells": list(cells),
                "explicit_cells": requested is not None,
                "bbox": list(bbox) if bbox else None,
                "band": band,
                "force": force},
        notes=f"{len(cell_dirs)} cell(s) present under ENC_ROOT/")
    return cell_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cells", nargs="+",
                       help="Explicit ENC cell names (e.g. US5BOSCD)")
    group.add_argument("--catalog_state",
                       help="Two-letter state code for catalog-based "
                            "selection")
    parser.add_argument("--bbox", nargs=4, type=float,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                        help="Bounding box for --catalog_state selection")
    parser.add_argument("--band", type=int, required=True,
                        help="Usage band filter for catalog selection: 5 is "
                             "Harbour (previously the default, which every "
                             "non-harbour collection then inherited); pass -1 "
                             "to take every band")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="ENC root to populate (holds ENC_ROOT/); "
                             "conventionally "
                             "<farfield root>/raw_material/enc_cells")
    parser.add_argument("--force", action="store_true",
                        help="Re-download existing cells")
    args = parser.parse_args()

    main(
        cells=args.cells,
        catalog_state=args.catalog_state,
        bbox=tuple(args.bbox) if args.bbox else None,
        band=None if args.band == -1 else args.band,
        output_dir=args.output_dir,
        force=args.force,
    )
