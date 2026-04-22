"""Stage the CVG-Text dataset (ICCV 2025, yejy53/CVG-Text) into the VIGOR-shaped
directory layout required by `VigorDataset` / `export_correspondence_similarity.py`.

Accepts either a CVG-Text root that is already unpacked (contains
`reference/<City>-satellite/`, `data/query/<City>-ground/`, `annotation/<City>/test.json`)
or one freshly downloaded from HuggingFace (`LHL3341/CVG-Text_full`, i.e.
`images.zip` + `data/query.zip` + `annotation/`). In the latter case this
script extracts the zips in place, stripping the upstream
`mnt/hwfile/opendatalab/air/linhonglin/CVG-text/` prefix that wraps
`images.zip`. The original zips are left on disk. Then, per city, writes:

    <vigor_root>/cvgtext_<City>/
      satellite_bbox.json              (from sat filenames + buffer, same schema
                                        as Ethan's mapillary cities)
      satellite/                       symlinks with VIGOR-format names
        satellite_<lat>_<lon>.<ext>   → reference/<City>-satellite/<real>.png
      panorama/                        test-only symlinks with VIGOR-format names
        <panoid>_<date>_d<yaw>,<lat>,<lon>,.<ext>
                                      → data/query/<City>-ground/<real>.png
      landmarks/                       empty; populate via extract_landmarks_historical.py

Why the renames: `VigorDataset.load_satellite_metadata` hardcodes
`_, lat, lon = p.stem.split("_")` and `load_panorama_metadata` hardcodes
`pano_id, lat, lon, _ = p.stem.split(",")`. CVG-Text filenames
(`<lat>,<lon>_<date>_<panoid>_d<yaw>_z<zoom>.<ext>`) fail both parsers.

After staging, point every downstream tool at the staged `panorama/` dir —
especially `extract_gemini_landmarks_from_panoramas.py`. Running Gemini on
the staged dir means the stage-7 pickle's keys are the staged VIGOR-format
filenames, which `extract_tags_from_pano_data`'s `pano_id.split(",")[0]`
lookup lines up with `VigorDataset.load_panorama_metadata` automatically.
(If you run Gemini on the raw CVG-Text dir first, the pickle keys are in
the wrong format and the correspondence pipeline silently drops all pano
landmarks. Stage first.)

The script does NOT download CVG-Text itself. If it's not on disk, pull
`LHL3341/CVG-Text_full` from HuggingFace (panoramas + satellite + OSM
imagery all ship in the HF repo; the upstream repo's Google Maps API
fetch scripts are NOT needed).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import zipfile
from pathlib import Path


CITIES_DEFAULT = ("Brisbane", "NewYork", "Tokyo")

# LHL3341/CVG-Text_full's images.zip wraps every entry under this path. The
# query.zip in the same repo has no such prefix (top-level is `query/`).
_IMAGES_ZIP_PREFIX = "mnt/hwfile/opendatalab/air/linhonglin/CVG-text/"

_FILENAME_RE = re.compile(
    r"^(?P<lat>-?[\d.]+),(?P<lon>-?[\d.]+)_"
    r"(?P<date>\d{4}-\d{2})_"
    r"(?P<pano_id>.+)_"
    r"d(?P<yaw>\d+)_"
    r"z(?P<zoom>\d+)"
    r"\.(?P<ext>png|jpg|jpeg)$"
)


def _parse_filename(fn: str) -> dict:
    m = _FILENAME_RE.match(fn)
    if m is None:
        raise ValueError(f"Filename does not match CVG-Text pattern: {fn!r}")
    return m.groupdict()


def _clean_dir(p: Path) -> None:
    if p.is_symlink() or p.is_file():
        p.unlink()
    elif p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)


def _meters_to_deg_lat(m: float) -> float:
    return m / 110574.0


def _meters_to_deg_lon(m: float, lat_deg: float) -> float:
    return m / (111320.0 * math.cos(math.radians(lat_deg)))


def _extract_zip(zip_path: Path, dest: Path, *, strip_prefix: str = "") -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        n_total = len(members)
        n_skipped = 0
        for i, info in enumerate(members):
            if strip_prefix:
                if not info.filename.startswith(strip_prefix):
                    n_skipped += 1
                    continue
                info.filename = info.filename[len(strip_prefix):]
                if not info.filename:
                    continue
            zf.extract(info, dest)
            if (i + 1) % 5000 == 0 or i + 1 == n_total:
                print(f"    ...{i + 1}/{n_total} entries", flush=True)
    if n_skipped:
        print(f"    ({n_skipped} entries skipped — outside '{strip_prefix}')")


def ensure_extracted(cvgtext_root: Path) -> None:
    """Extract HF zips in place if the expected directory layout is missing.

    No-op when `reference/` and `data/query/` already exist. Leaves the
    original zip files on disk.
    """
    images_zip = cvgtext_root / "images.zip"
    reference_dir = cvgtext_root / "reference"
    if not reference_dir.exists():
        if not images_zip.exists():
            raise FileNotFoundError(
                f"Neither extracted {reference_dir} nor {images_zip} exists. "
                "Download LHL3341/CVG-Text_full from HuggingFace first."
            )
        print(f"Extracting {images_zip} -> {reference_dir} (this takes a while)")
        _extract_zip(images_zip, cvgtext_root, strip_prefix=_IMAGES_ZIP_PREFIX)

    query_zip = cvgtext_root / "data" / "query.zip"
    query_dir = cvgtext_root / "data" / "query"
    if not query_dir.exists():
        if not query_zip.exists():
            raise FileNotFoundError(
                f"Neither extracted {query_dir} nor {query_zip} exists. "
                "Download LHL3341/CVG-Text_full from HuggingFace first."
            )
        print(f"Extracting {query_zip} -> {query_dir} (this takes a while)")
        _extract_zip(query_zip, cvgtext_root / "data")


def stage_city(
    *,
    cvgtext_root: Path,
    vigor_root: Path,
    city: str,
    buffer_m: float,
) -> None:
    sat_src = cvgtext_root / "reference" / f"{city}-satellite"
    pano_src = cvgtext_root / "data" / "query" / f"{city}-ground"
    test_json = cvgtext_root / "annotation" / city / "test.json"
    for p, what in [(sat_src, "satellite reference"),
                    (pano_src, "panorama queries"),
                    (test_json, "test annotation")]:
        if not p.exists():
            raise FileNotFoundError(f"{what} missing: {p}")

    stage = vigor_root / f"cvgtext_{city}"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "landmarks").mkdir(exist_ok=True)

    # --- bbox from sat filenames + buffer ---
    lats, lons = [], []
    for p in sorted(sat_src.iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        g = _parse_filename(p.name)
        lats.append(float(g["lat"]))
        lons.append(float(g["lon"]))
    if not lats:
        raise RuntimeError(f"no satellite files parsed in {sat_src}")
    south, north = min(lats), max(lats)
    west, east = min(lons), max(lons)
    mid_lat = (south + north) / 2
    dlat = _meters_to_deg_lat(buffer_m)
    dlon = _meters_to_deg_lon(buffer_m, mid_lat)
    bbox = {
        "south": south - dlat,
        "west": west - dlon,
        "north": north + dlat,
        "east": east + dlon,
        "buffer_m": buffer_m,
        "n_tiles": len(lats),
        "raw_south": south,
        "raw_west": west,
        "raw_north": north,
        "raw_east": east,
    }
    (stage / "satellite_bbox.json").write_text(json.dumps(bbox, indent=2))

    # --- satellite symlinks (full gallery, dedupe by (lat_str, lon_str)) ---
    sat_dst = stage / "satellite"
    _clean_dir(sat_dst)
    seen_latlon: set[tuple[str, str]] = set()
    sat_skipped = 0
    for p in sorted(sat_src.iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        g = _parse_filename(p.name)
        key = (g["lat"], g["lon"])
        if key in seen_latlon:
            sat_skipped += 1
            continue
        seen_latlon.add(key)
        (sat_dst / f"satellite_{g['lat']}_{g['lon']}.{g['ext']}").symlink_to(p.resolve())

    # --- panorama symlinks (test-split only) ---
    pano_dst = stage / "panorama"
    _clean_dir(pano_dst)
    test_filenames = list(json.loads(test_json.read_text()).keys())
    pano_seen: set[str] = set()
    pano_collisions = 0
    pano_orphans = 0
    for fn in test_filenames:
        p = pano_src / fn
        if not p.exists():
            pano_orphans += 1
            continue
        g = _parse_filename(fn)
        disambig_id = f"{g['pano_id']}_{g['date']}_d{g['yaw']}"
        if disambig_id in pano_seen:
            pano_collisions += 1
            continue
        pano_seen.add(disambig_id)
        new_name = f"{disambig_id},{g['lat']},{g['lon']},.{g['ext']}"
        (pano_dst / new_name).symlink_to(p.resolve())

    print(
        f"  {city}: sat={len(seen_latlon)} tiles ({sat_skipped} dupes skipped), "
        f"pano={len(pano_seen)} test panos "
        f"({pano_collisions} collisions, {pano_orphans} orphans)"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--cvgtext_root", type=Path,
        default=Path("/data/overhead_matching/datasets/cvgtext"),
        help="CVG-Text dataset root (contains `reference/`, `data/`, `annotation/`).",
    )
    p.add_argument(
        "--vigor_root", type=Path,
        default=Path("/data/overhead_matching/datasets/VIGOR"),
        help="Parent dir under which `cvgtext_<City>/` staging trees are created.",
    )
    p.add_argument(
        "--cities", nargs="+", default=list(CITIES_DEFAULT),
        help="Cities to stage (default: Brisbane NewYork Tokyo).",
    )
    p.add_argument(
        "--buffer_m", type=float, default=500,
        help="Meters of padding around the sat-gallery bbox in satellite_bbox.json.",
    )
    args = p.parse_args()

    ensure_extracted(args.cvgtext_root)

    print(f"Staging CVG-Text -> VIGOR at {args.vigor_root}")
    for city in args.cities:
        stage_city(
            cvgtext_root=args.cvgtext_root,
            vigor_root=args.vigor_root,
            city=city,
            buffer_m=args.buffer_m,
        )


if __name__ == "__main__":
    main()
