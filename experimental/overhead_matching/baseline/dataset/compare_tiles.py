"""Side-by-side composite of (satellite | OSM) tiles for visual QA.

Usage:
  bazel run //experimental/overhead_matching/baseline/dataset:compare_tiles -- \\
      --city Seattle --n 6

  bazel run //experimental/overhead_matching/baseline/dataset:compare_tiles -- \\
      --city Seattle --filename satellite_47.6_-122.3.png

Writes a single PNG (default /tmp/osm_compare.png) with the satellite tile on
the left and the rendered OSM tile on the right, one row per requested pair.
"""
import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from experimental.overhead_matching.baseline.dataset import (
    city_pbf_map,
    osm_org_tile,
    tile_geometry,
)

LABEL_HEIGHT = 24
GUTTER = 4


def _select_filenames(
    city_dir: Path, osm_subdir: str, filename: str | None, n: int, seed: int
) -> list[str]:
    osm_dir = city_dir / osm_subdir
    sat_dir = city_dir / "satellite"
    if filename is not None:
        if not (sat_dir / filename).exists():
            raise FileNotFoundError(f"satellite tile missing: {sat_dir / filename}")
        if not (osm_dir / filename).exists():
            raise FileNotFoundError(f"osm tile missing: {osm_dir / filename}")
        return [filename]

    available = [
        p.name for p in osm_dir.iterdir()
        if p.suffix == ".png" and (sat_dir / p.name).exists()
    ]
    if not available:
        raise RuntimeError(f"no overlapping tiles in {sat_dir} and {osm_dir}")
    rng = random.Random(seed)
    rng.shuffle(available)
    return available[:n]


def _make_label(text: str, width: int) -> Image.Image:
    img = Image.new("RGB", (width, LABEL_HEIGHT), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 4), text, fill=(220, 220, 220), font=font)
    return img


def build_composite(
    city_dir: Path,
    filenames: list[str],
    osm_subdir: str = "satellite_osm",
    overlay_alpha: float | None = None,
    with_osm_org: bool = False,
) -> Image.Image:
    sat_dir = city_dir / "satellite"
    osm_dir = city_dir / osm_subdir

    # Probe one tile for size; assume all share dims (they do for VIGOR).
    sample = Image.open(sat_dir / filenames[0])
    tile_w, tile_h = sample.size
    sample.close()

    cols = 2
    if overlay_alpha is not None:
        cols += 1
    if with_osm_org:
        cols += 1
    rows = len(filenames)
    width = cols * tile_w + (cols - 1) * GUTTER
    row_height = LABEL_HEIGHT + tile_h
    height = rows * row_height + (rows - 1) * GUTTER

    composite = Image.new("RGB", (width, height), (10, 10, 10))

    for r, name in enumerate(filenames):
        y0 = r * (row_height + GUTTER)
        sat = Image.open(sat_dir / name).convert("RGB")
        osm = Image.open(osm_dir / name).convert("RGB").resize((tile_w, tile_h))

        col = 0
        composite.paste(_make_label(f"{name}  [satellite]", tile_w), (col * (tile_w + GUTTER), y0))
        composite.paste(sat, (col * (tile_w + GUTTER), y0 + LABEL_HEIGHT))
        col += 1
        composite.paste(_make_label(f"{name}  [OSM (ours)]", tile_w), (col * (tile_w + GUTTER), y0))
        composite.paste(osm, (col * (tile_w + GUTTER), y0 + LABEL_HEIGHT))
        col += 1

        if with_osm_org:
            lat, lon = tile_geometry.satellite_filename_to_center(name)
            bbox = tile_geometry.center_zoom_to_bbox(lat, lon, zoom=20, tile_px=tile_w)
            osm_org = osm_org_tile.fetch_for_bbox(bbox, target_px=tile_w)
            composite.paste(_make_label(f"{name}  [osm.org]", tile_w), (col * (tile_w + GUTTER), y0))
            composite.paste(osm_org, (col * (tile_w + GUTTER), y0 + LABEL_HEIGHT))
            col += 1

        if overlay_alpha is not None:
            blended = Image.blend(sat, osm, overlay_alpha)
            composite.paste(_make_label(f"{name}  [overlay α={overlay_alpha:.2f}]", tile_w),
                            (col * (tile_w + GUTTER), y0))
            composite.paste(blended, (col * (tile_w + GUTTER), y0 + LABEL_HEIGHT))
            col += 1

    return composite


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True, choices=city_pbf_map.cities())
    p.add_argument(
        "--vigor-root", type=Path,
        default=Path("/data/overhead_matching/datasets/VIGOR"),
    )
    p.add_argument("--osm-subdir", default="satellite_osm")
    p.add_argument(
        "--filename", default=None,
        help="specific tile filename; overrides --n",
    )
    p.add_argument("--n", type=int, default=6,
                   help="number of random tiles to compare")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=None,
                   help="if set, add a column with satellite blended with OSM at this alpha (0..1)")
    p.add_argument("--with-osm-org", action="store_true",
                   help="add a column showing the corresponding tile from openstreetmap.org")
    p.add_argument("--output", type=Path, default=Path("/tmp/osm_compare.png"))
    args = p.parse_args()

    city_dir = args.vigor_root / args.city
    filenames = _select_filenames(
        city_dir, args.osm_subdir, args.filename, args.n, args.seed
    )

    composite = build_composite(
        city_dir, filenames,
        osm_subdir=args.osm_subdir,
        overlay_alpha=args.alpha,
        with_osm_org=args.with_osm_org,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    composite.save(args.output)
    print(f"wrote {args.output}  ({len(filenames)} pair(s))")
    for fn in filenames:
        print(f"  {fn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
