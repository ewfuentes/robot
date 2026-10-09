"""Verify <env>/satellite_osm_carto/ against <env>/satellite/ and write a [satellite | carto]
contact sheet. Plain python + PIL.

usage: check_carto.py --env Framingham --composite /path/Framingham_carto_check.png
"""
import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--vigor-root", type=Path, default=Path("/data/overhead_matching/datasets/VIGOR"))
    ap.add_argument("--composite", type=Path, help="write a 4-row [satellite | carto x2] PNG here")
    ap.add_argument("--n-open", type=int, default=32, help="random tiles to open with PIL")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    env_dir = a.vigor_root / a.env
    carto_dir = env_dir / "satellite_osm_carto"
    bbox_px = 640
    if (env_dir / "satellite_bbox.json").exists():
        bbox_px = json.loads((env_dir / "satellite_bbox.json").read_text()).get("source_px", 640)
    px = round(bbox_px / 2)

    sats = {p.stem: p for p in (env_dir / "satellite").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    cartos = {p.stem: p for p in carto_dir.iterdir() if p.suffix == ".png"}
    missing = sorted(set(sats) - set(cartos))
    extra = sorted(set(cartos) - set(sats))
    print(f"{a.env}: satellite {len(sats)} carto {len(cartos)} missing {len(missing)} extra {len(extra)} expected px {px}")
    for s in missing[:5]:
        print("  missing", s)
    ok = not missing and not extra

    rng = random.Random(a.seed)
    sample = rng.sample(sorted(cartos), min(a.n_open, len(cartos)))
    for stem in sample:
        im = Image.open(cartos[stem])
        im.load()
        if im.size != (px, px):
            print(f"  bad size {cartos[stem]}: {im.size}")
            ok = False
    print(f"  opened {len(sample)} random tiles, mode {Image.open(cartos[sample[0]]).mode}")

    if a.composite:
        rows = []
        for stem in sample[:4]:
            sat = Image.open(sats[stem]).convert("RGB")
            car = Image.open(cartos[stem]).convert("RGB").resize(sat.size, Image.BICUBIC)
            row = Image.new("RGB", (sat.width * 2 + 6, sat.height), (255, 255, 255))
            row.paste(sat, (0, 0))
            row.paste(car, (sat.width + 6, 0))
            ImageDraw.Draw(row).text((3, 3), f"{a.env} {stem}  | carto {px}px x2", fill=(255, 255, 0))
            rows.append(row)
        sheet = Image.new("RGB", (rows[0].width, sum(r.height + 6 for r in rows)), (255, 255, 255))
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height + 6
        a.composite.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(a.composite)
        print(f"  composite -> {a.composite}")
    print("  OK" if ok else "  FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
