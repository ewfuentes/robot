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

`--output_dir` is required. Cells are raw material, so the conventional home is
`<farfield root>/raw_material/enc_cells/`. Each cell carries an exact content
manifest, and each invocation publishes a caller-specified immutable selection
record. ENC cells change weekly, so consumers bind the selection record rather
than infer a vintage from the mutable cache root.

Example:
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:download_enc_cells -- \\
        --cells US5BOSCC US5BOSCD \\
        --output_dir /path/to/farfield/raw_material/enc_cells \\
        --selection_output /path/to/build/enc_selection.json

    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:download_enc_cells -- \\
        --catalog_state MA --bbox -71.08 42.24 -70.84 42.43 --band 5 \\
        --output_dir /path/to/farfield/raw_material/enc_cells \\
        --selection_output /path/to/build/enc_selection.json
"""

import argparse
import datetime
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable

import requests

from experimental.overhead_matching.swag.farfield import artifact, provenance

ENC_BASE_URL = "https://charts.noaa.gov/ENCs"
CELL_MANIFEST_SCHEMA = "farfield.enc_cell.v1"
CELL_MANIFEST_NAME = "manifest.json"
SELECTION_SCHEMA = "farfield.enc_selection.v1"
CELL_RE = re.compile(r"US[1-6][A-Z0-9]{5}\Z")
STATE_RE = re.compile(r"[A-Z]{2}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GENERATOR = "farfield/dataset_tools/download_enc_cells.py"
ENC_ROOT_AUXILIARY_FILES = frozenset({"CATALOG.031", "README.TXT"})
CELL_MANIFEST_KEYS = frozenset({
    "schema", "cell", "source_url", "archive_sha256", "archive_size",
    "files", "generator", "git_commit", "created", "complete",
})
SELECTION_KEYS = frozenset({
    "schema", "generator", "git_commit", "created", "output_dir",
    "catalog_state", "catalog", "bbox", "band", "explicit_cells",
    "cells", "cell_refs", "complete",
})


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path):
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value!r}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


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


def _validate_bbox(bbox: tuple | list) -> tuple[float, float, float, float]:
    if (not isinstance(bbox, (tuple, list)) or len(bbox) != 4
            or any(isinstance(value, bool)
                   or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in bbox)):
        raise ValueError("ENC bbox must contain four finite numbers")
    west, south, east, north = map(float, bbox)
    if (not -180.0 <= west < east <= 180.0
            or not -90.0 <= south < north <= 90.0):
        raise ValueError("ENC bbox must be ordered west,south,east,north")
    return west, south, east, north


def _archive_members(zf: zipfile.ZipFile, cell: str) -> list[zipfile.ZipInfo]:
    """Validate and return the regular files belonging exactly to ``cell``."""
    enc_root = PurePosixPath("ENC_ROOT")
    prefix = enc_root / cell
    members = []
    seen = set()
    for info in zf.infolist():
        raw = info.filename
        if "\\" in raw:
            raise ValueError(f"{cell}: ZIP member uses a backslash: {raw!r}")
        path = PurePosixPath(raw)
        if (path.is_absolute() or any(part in ("", ".", "..")
                                      for part in path.parts)):
            raise ValueError(f"{cell}: unsafe ZIP member path: {raw!r}")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"{cell}: ZIP member is a symlink: {raw!r}")
        if info.is_dir():
            continue
        if (path.parent == enc_root
                and path.name in ENC_ROOT_AUXILIARY_FILES):
            # NOAA includes shared catalog/readme files in some per-cell ZIPs.
            # They are validated as known layout but are not cell content and
            # must never be mixed into a cell's independently hashed vintage.
            continue
        if path.parent != prefix:
            raise ValueError(
                f"{cell}: ZIP file is outside ENC_ROOT/{cell}/: {raw!r}")
        if path.name in seen:
            raise ValueError(f"{cell}: duplicate ZIP member: {path.name}")
        seen.add(path.name)
        members.append(info)
    base = f"{cell}.000"
    if base not in seen:
        raise FileNotFoundError(
            f"{cell}: ZIP lacks required ENC_ROOT/{cell}/{base}")
    return sorted(members, key=lambda info: PurePosixPath(info.filename).name)


def _cell_file_records(cell_dir: Path) -> list[dict]:
    records = []
    for path in sorted(cell_dir.iterdir(), key=lambda item: item.name):
        if path.name == CELL_MANIFEST_NAME:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"ENC cell contains a non-regular file: {path}")
        records.append({
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": artifact.sha256_file(path),
        })
    return records


def validate_cell(cell_dir: Path, cell: str) -> dict:
    """Validate a completed cell and return its per-cell manifest."""
    if cell_dir.is_symlink() or not cell_dir.is_dir():
        raise ValueError(f"ENC cell is not a regular directory: {cell_dir}")
    manifest_path = cell_dir / CELL_MANIFEST_NAME
    try:
        manifest = _load_json(manifest_path)
    except ValueError as exc:
        raise ValueError(
            f"{cell}: invalid or missing per-cell manifest: {exc}") from exc
    if not isinstance(manifest, dict) or set(manifest) != CELL_MANIFEST_KEYS:
        raise ValueError(f"{cell}: invalid per-cell manifest fields")
    archive_sha256 = manifest.get("archive_sha256")
    if (manifest.get("schema") != CELL_MANIFEST_SCHEMA
            or manifest.get("cell") != cell
            or manifest.get("source_url") != f"{ENC_BASE_URL}/{cell}.zip"
            or not isinstance(archive_sha256, str)
            or not SHA256_RE.fullmatch(archive_sha256)
            or type(manifest.get("archive_size")) is not int
            or manifest["archive_size"] <= 0
            or manifest.get("complete") is not True):
        raise ValueError(f"{cell}: invalid per-cell manifest identity")
    expected = manifest.get("files")
    actual = _cell_file_records(cell_dir)
    if expected != actual:
        raise ValueError(f"{cell}: cell files do not match recorded content digests")
    if not any(record["path"] == f"{cell}.000" for record in actual):
        raise ValueError(f"{cell}: required base file is missing")
    return manifest


def _publish_cell(staging: Path, target: Path, replace: bool) -> None:
    """Publish one staged cell, rolling back a forced replacement on error."""
    if not replace:
        artifact.publish_directory_no_clobber(staging, target)
        return
    backup = target.with_name(target.name + ".replaced")
    if backup.exists() or backup.is_symlink():
        raise FileExistsError(f"stale replacement backup blocks refresh: {backup}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_fd = os.open(target.parent, flags)
    moved_old = False
    try:
        fcntl.flock(parent_fd, fcntl.LOCK_EX)
        if not target.exists() or target.is_symlink():
            raise FileNotFoundError(f"cell disappeared during refresh: {target}")
        os.rename(target, backup)
        moved_old = True
        try:
            os.rename(staging, target)
        except BaseException:
            os.rename(backup, target)
            moved_old = False
            raise
        os.fsync(parent_fd)
    finally:
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_UN)
        finally:
            os.close(parent_fd)
    if moved_old:
        shutil.rmtree(backup)


def download_cell(
    cell: str,
    output_dir: Path,
    force: bool = False,
    fetch_fn: Callable[[str], bytes] = fetch_url,
) -> Path:
    """Download, validate, and publish one independently versioned ENC cell."""
    if not CELL_RE.fullmatch(cell):
        raise ValueError(f"invalid NOAA ENC cell identifier: {cell!r}")
    cell_dir = output_dir / "ENC_ROOT" / cell
    staging = cell_dir.with_name(cell + artifact.INCOMPLETE_SUFFIX)
    if cell_dir.exists() and (cell_dir.is_symlink() or not cell_dir.is_dir()):
        raise ValueError(f"refusing non-directory ENC cell path: {cell_dir}")
    if cell_dir.exists() and not force:
        validate_cell(cell_dir, cell)
        print(f"{cell}: already present at {cell_dir}, skipping "
              f"(use --force to refresh)")
        return cell_dir
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(
            f"incomplete ENC cell blocks publication; inspect or remove: {staging}")

    url = f"{ENC_BASE_URL}/{cell}.zip"
    print(f"{cell}: downloading {url}")
    payload = fetch_fn(url)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        members = _archive_members(zf, cell)
        cell_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
        try:
            for info in members:
                destination = staging / PurePosixPath(info.filename).name
                with zf.open(info) as source, destination.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
            files = _cell_file_records(staging)
            manifest = {
                "schema": CELL_MANIFEST_SCHEMA,
                "cell": cell,
                "source_url": url,
                "archive_sha256": hashlib.sha256(payload).hexdigest(),
                "archive_size": len(payload),
                "files": files,
                "generator": GENERATOR,
                "git_commit": provenance.git_commit(),
                "created": datetime.datetime.now(datetime.timezone.utc)
                           .isoformat(timespec="seconds"),
                "complete": True,
            }
            artifact.atomic_write_json(staging / CELL_MANIFEST_NAME, manifest)
            validate_cell(staging, cell)
            _publish_cell(staging, cell_dir, replace=cell_dir.exists())
        except BaseException:
            # Leave a non-consumable .incomplete directory for diagnosis.
            raise
    print(f"{cell}: published to {cell_dir}")
    return cell_dir


def main(
    cells: list | None,
    catalog_state: str | None,
    bbox: tuple | None,
    band: int | None,
    output_dir: Path,
    selection_output: Path,
    force: bool,
    fetch_fn: Callable[[str], bytes] = fetch_url,
) -> list:
    output_dir = Path(output_dir)
    selection_output = Path(selection_output)
    if band is not None and (
            isinstance(band, bool) or not isinstance(band, int)
            or not 1 <= band <= 6):
        raise ValueError("ENC band must be an integer in 1..6 or null")
    if cells is None:
        if (not isinstance(catalog_state, str)
                or not STATE_RE.fullmatch(catalog_state)):
            raise ValueError("catalog_state must be a two-letter uppercase code")
        bbox = _validate_bbox(bbox)
    elif catalog_state is not None or bbox is not None:
        raise ValueError("explicit cells cannot be combined with catalog selection")
    if selection_output.exists() or selection_output.is_symlink():
        if force:
            raise FileExistsError(
                "--force cannot replace an immutable ENC selection; choose "
                f"a new selection output: {selection_output}")
        completed = validate_selection(selection_output, output_dir)
        expected_invocation = {
            "catalog_state": catalog_state,
            "bbox": list(bbox) if bbox else None,
            "band": band,
            "explicit_cells": cells is not None,
        }
        disagreements = {
            key: (completed.get(key), value)
            for key, value in expected_invocation.items()
            if completed.get(key) != value
        }
        if cells is not None and completed.get("cells") != cells:
            disagreements["cells"] = (completed.get("cells"), cells)
        if disagreements:
            raise ValueError(
                "completed ENC selection belongs to a different invocation: "
                f"{disagreements}")
        print(f"Reusing exact completed ENC selection {selection_output}")
        return [output_dir / "ENC_ROOT" / cell
                for cell in completed["cells"]]
    requested = cells
    catalog_record = None
    if cells is None:
        if catalog_state is None or bbox is None:
            raise ValueError(
                "Provide either --cells, or --catalog_state with --bbox")
        catalog_url = f"{ENC_BASE_URL}/{catalog_state}_ENCProdCat_19115.xml"
        print(f"Fetching catalog {catalog_url}")
        catalog_xml = fetch_fn(catalog_url).decode("utf-8")
        catalog_bytes = catalog_xml.encode("utf-8")
        catalog_digest = hashlib.sha256(catalog_bytes).hexdigest()
        catalog_dir = output_dir / "catalogs"
        catalog_dir.mkdir(parents=True, exist_ok=True)
        catalog_path = (
            catalog_dir /
            f"{catalog_state}_ENCProdCat_19115-{catalog_digest[:16]}.xml"
        )
        if catalog_path.exists():
            if (catalog_path.is_symlink()
                    or artifact.sha256_file(catalog_path) != catalog_digest):
                raise ValueError(
                    f"cached ENC catalog identity mismatch: {catalog_path}")
        else:
            artifact.atomic_create_file(catalog_path, catalog_bytes)
        catalog_record = {
            "url": catalog_url,
            "path": str(catalog_path.resolve()),
            "sha256": catalog_digest,
        }
        cells = select_cells_from_catalog(catalog_xml, bbox, band)
        print(f"Catalog selection (band={band}, bbox={bbox}): "
              f"{len(cells)} cells: {cells}")

    if (not isinstance(cells, list) or not cells
            or not all(isinstance(cell, str) and CELL_RE.fullmatch(cell)
                       for cell in cells)
            or len(set(cells)) != len(cells)):
        raise ValueError("ENC selection must contain unique valid cell names")
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_dirs = [download_cell(cell, output_dir, force=force,
                               fetch_fn=fetch_fn) for cell in cells]
    selection = {
        "schema": SELECTION_SCHEMA,
        "generator": GENERATOR,
        "git_commit": provenance.git_commit(),
        "created": datetime.datetime.now(datetime.timezone.utc)
                   .isoformat(timespec="seconds"),
        "output_dir": str(Path(output_dir).resolve()),
        "catalog_state": catalog_state,
        "catalog": catalog_record,
        "bbox": list(bbox) if bbox else None,
        "band": band,
        "explicit_cells": requested is not None,
        "cells": list(cells),
        "cell_refs": [
            {
                "cell": cell,
                "path": str(cell_dir.resolve()),
                "manifest_sha256": artifact.sha256_file(
                    cell_dir / CELL_MANIFEST_NAME),
                "content_sha256": artifact.sha256_directory(cell_dir),
            }
            for cell, cell_dir in zip(cells, cell_dirs, strict=True)
        ],
        "complete": True,
    }
    artifact.atomic_create_json(selection_output, selection)
    validate_selection(selection_output, output_dir)
    return cell_dirs


def validate_selection(path: Path, output_dir: Path) -> dict:
    """Validate one immutable invocation record and every selected cell."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"ENC selection is not a regular file: {path}")
    document = _load_json(path)
    if not isinstance(document, dict) or set(document) != SELECTION_KEYS:
        raise ValueError(f"invalid ENC selection record fields: {path}")
    if (document["schema"] != SELECTION_SCHEMA
            or document["generator"] != GENERATOR
            or document["complete"] is not True
            or document["output_dir"] != str(Path(output_dir).resolve())
            or type(document["explicit_cells"]) is not bool):
        raise ValueError(f"invalid ENC selection identity: {path}")
    for field in ("git_commit", "created"):
        if not isinstance(document[field], str) or not document[field]:
            raise ValueError(f"invalid ENC selection {field}: {path}")
    state = document["catalog_state"]
    catalog = document["catalog"]
    bbox = document["bbox"]
    band = document["band"]
    if state is None:
        if (document["explicit_cells"] is not True
                or catalog is not None or bbox is not None):
            raise ValueError(
                f"explicit ENC selection cannot carry a catalog: {path}")
    else:
        if (document["explicit_cells"] is not False
                or not isinstance(state, str)
                or not STATE_RE.fullmatch(state)):
            raise ValueError(f"invalid ENC catalog state: {path}")
        if (not isinstance(catalog, dict)
                or set(catalog) != {"url", "path", "sha256"}):
            raise ValueError(f"invalid ENC catalog identity: {path}")
        if (not all(isinstance(catalog[field], str) and catalog[field]
                    for field in ("url", "path", "sha256"))):
            raise ValueError(f"invalid ENC catalog identity: {path}")
        catalog_path = Path(catalog["path"])
        expected_url = f"{ENC_BASE_URL}/{state}_ENCProdCat_19115.xml"
        expected_catalog_dir = (Path(output_dir) / "catalogs").resolve()
        if (not isinstance(catalog["sha256"], str)
                or not SHA256_RE.fullmatch(catalog["sha256"])
                or catalog["url"] != expected_url
                or catalog_path.is_symlink()
                or not catalog_path.is_file()
                or catalog_path.resolve().parent != expected_catalog_dir
                or catalog_path.name != (
                    f"{state}_ENCProdCat_19115-"
                    f"{catalog['sha256'][:16]}.xml")
                or artifact.sha256_file(catalog_path) != catalog["sha256"]):
            raise ValueError(f"ENC catalog content changed: {path}")
        try:
            _validate_bbox(bbox)
        except ValueError as error:
            raise ValueError(f"invalid ENC selection bbox: {path}") from error
    if band is not None and (
            isinstance(band, bool) or not isinstance(band, int)
            or not 1 <= band <= 6):
        raise ValueError(f"invalid ENC selection band: {path}")
    cells = document["cells"]
    refs = document["cell_refs"]
    if (not isinstance(cells, list) or not cells
            or not all(isinstance(cell, str) and CELL_RE.fullmatch(cell)
                       for cell in cells)
            or len(set(cells)) != len(cells)
            or not isinstance(refs, list)
            or len(refs) != len(cells)):
        raise ValueError(f"invalid ENC selection cell set: {path}")
    if state is not None:
        catalog_xml = Path(catalog["path"]).read_text(encoding="utf-8")
        recomputed = select_cells_from_catalog(
            catalog_xml, tuple(bbox), band)
        if cells != recomputed:
            raise ValueError(
                f"ENC cells do not exactly match the recorded catalog "
                f"selection: {path}")
    expected_refs = []
    for cell in cells:
        cell_dir = Path(output_dir) / "ENC_ROOT" / cell
        validate_cell(cell_dir, cell)
        expected_refs.append({
            "cell": cell,
            "path": str(cell_dir.resolve()),
            "manifest_sha256": artifact.sha256_file(
                cell_dir / CELL_MANIFEST_NAME),
            "content_sha256": artifact.sha256_directory(cell_dir),
        })
    if refs != expected_refs:
        raise ValueError(
            f"ENC selection cell identities no longer match: {path}")
    return document


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
    parser.add_argument("--band", type=int, default=5,
                        help="Usage band filter for catalog selection "
                             "(default 5=Harbor; pass --band -1 to disable)")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="ENC root to populate (holds ENC_ROOT/); "
                             "conventionally "
                             "<farfield root>/raw_material/enc_cells")
    parser.add_argument(
        "--selection_output",
        type=Path,
        required=True,
        help="new immutable JSON record for the exact cells selected by this "
             "invocation",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download existing cells")
    args = parser.parse_args()

    main(
        cells=args.cells,
        catalog_state=args.catalog_state,
        bbox=tuple(args.bbox) if args.bbox else None,
        band=None if args.band == -1 else args.band,
        output_dir=args.output_dir,
        selection_output=args.selection_output,
        force=args.force,
    )
