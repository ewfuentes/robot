"""Orientation check figures for a north-aligned VIGOR-format environment.

For a few frames spread over the trajectory (and over travel headings) this writes one figure to
<vigor_dir>/orientation_check/ showing the north-aligned panorama with S/W/N/E column markers
(house convention: north at the centre column, bearing = ((col - W/2) / W) * 360) and a magenta line
where the GPS direction of travel should appear, above a north-up satellite mosaic (and the OSM tile
when present) with the same travel direction drawn as an arrow. If the road ahead does not sit on the
magenta line, the roll is wrong and every image-based artifact will inherit the error.

A human looks at these before any later stage. This exists because heading validation against the GPS
course is circular when the heading field IS the GPS course (Mapillary compass_angle for some uploaders):
the Netherlands environments passed validation with a 0.6 deg residual and were rotated 90 deg.

Usage:
  python orientation_check.py --vigor_dir /data/.../VIGOR/<Env> [--n 6] [--roll_deg 0]
--roll_deg renders an additional row with the panorama rolled by that many degrees, to preview a fix.
"""
import argparse
import csv
import glob
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANO_W, PANO_H = 1600, 800


def travel_bearings(rows, min_step_m=3.0, max_turn_deg=25.0):
    """(frame_idx, bearing) for frames moving >= min_step_m on both sides without turning."""
    lat = np.array([float(r["lat"]) for r in rows]); lng = np.array([float(r["lng"]) for r in rows])
    mx, my = 111320 * math.cos(math.radians(lat.mean())), 110574

    def bearing(i, j):
        dx, dy = (lng[j] - lng[i]) * mx, (lat[j] - lat[i]) * my
        return math.degrees(math.atan2(dx, dy)) % 360, math.hypot(dx, dy)

    out = []
    for i in range(1, len(rows) - 1):
        b1, d1 = bearing(i - 1, i); b2, d2 = bearing(i, i + 1)
        if d1 < min_step_m or d2 < min_step_m or abs((b1 - b2 + 180) % 360 - 180) > max_turn_deg:
            continue
        out.append((i, bearing(i - 1, i + 1)[0]))
    return out


def pick_frames(cands, n):
    """One frame per heading sector where available, then fill by spreading over the sequence."""
    picks = []
    for q in range(n):
        sector = [c for c in cands if q * 360 / n <= c[1] < (q + 1) * 360 / n]
        if sector:
            picks.append(sector[len(sector) // 2])
    for k in range(1, n + 1):
        if len(picks) >= n or not cands:
            break
        c = cands[len(cands) * k // (n + 1)]
        if c not in picks:
            picks.append(c)
    return picks


def mark_pano(im, travel_deg, font):
    W, H = im.size; d = ImageDraw.Draw(im)
    for frac, lab in [(0, "S"), (0.25, "W"), (0.5, "N"), (0.75, "E"), (0.999, "S")]:
        x = int(frac * W)
        d.line([(x, 0), (x, H)], fill=(255, 0, 0), width=3)
        d.rectangle([min(x + 4, W - 40), 4, min(x + 40, W - 4), 26], fill=(0, 0, 0))
        d.text((min(x + 8, W - 36), 8), lab, fill=(255, 60, 60), font=font)
    xf = int(W * ((travel_deg / 360 + 0.5) % 1))
    d.line([(xf, 0), (xf, H)], fill=(255, 0, 255), width=4)
    d.rectangle([max(xf - 60, 0), H - 30, min(xf + 60, W), H - 6], fill=(0, 0, 0))
    d.text((max(xf - 56, 4), H - 26), f"forward {travel_deg:.0f}", fill=(255, 0, 255), font=font)
    return im


def draw_arrow(im, cx, cy, travel_deg, length_px, font, label):
    d = ImageDraw.Draw(im); t = math.radians(travel_deg)
    ex, ey = cx + length_px * math.sin(t), cy - length_px * math.cos(t)
    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(255, 0, 255))
    d.line([(cx, cy), (ex, ey)], fill=(255, 0, 255), width=5)
    for s in (25, -25):
        ts = math.radians(travel_deg + s)
        d.line([(ex, ey), (ex - 14 * math.sin(ts), ey + 14 * math.cos(ts))], fill=(255, 0, 255), width=5)
    d.rectangle([4, 4, 210, 22], fill=(0, 0, 0)); d.text((8, 8), label, fill=(255, 255, 0), font=font)
    S = im.size[0]
    d.text((S // 2 - 4, 26), "N", fill=(255, 60, 60), font=font); d.text((S - 16, S // 2), "E", fill=(255, 60, 60), font=font)


def overhead(vigor_dir, tiles, lat, lng, travel_deg, font, size_m=220, ppm=4):
    """North-up mosaic of the nearest satellite tiles, plus the nearest OSM tile if rendered."""
    mx, my = 111320 * math.cos(math.radians(lat)), 110574
    dx = (tiles["lon"] - lng) * mx; dy = (tiles["lat"] - lat) * my
    near = np.argsort(dx ** 2 + dy ** 2)[:16]
    S = int(size_m * ppm); canvas = Image.new("RGB", (S, S), (40, 40, 40))
    for n in near:
        wm = tiles["w"][n]
        t = Image.open(vigor_dir / "satellite" / tiles["name"][n]).convert("RGB").resize((int(wm * ppm), int(wm * ppm)))
        canvas.paste(t, (int(S / 2 + dx[n] * ppm - t.size[0] / 2), int(S / 2 - dy[n] * ppm - t.size[1] / 2)))
    draw_arrow(canvas, S / 2, S / 2, travel_deg, 30 * ppm, font, f"satellite north-up ~{size_m} m  fwd={travel_deg:.0f}")
    n0 = near[0]
    osm_matches = glob.glob(str(vigor_dir / "satellite_osm" / (Path(tiles["name"][n0]).stem + ".*")))
    osm = None
    if osm_matches:
        wm = tiles["w"][n0]; osm = Image.open(osm_matches[0]).convert("RGB").resize((S, S)); p = S / wm
        draw_arrow(osm, S / 2 - dx[n0] * p, S / 2 + dy[n0] * p, travel_deg, 30 * p, font, f"OSM tile north-up {wm:.0f} m")
    return canvas, osm


def load_tiles(vigor_dir):
    path = vigor_dir / "tile_metadata.csv"
    if path.exists():
        rows = list(csv.DictReader(open(path)))
        return {"name": [r["file_name"] for r in rows], "lat": np.array([float(r["center_lat"]) for r in rows]),
                "lon": np.array([float(r["center_lon"]) for r in rows]), "w": np.array([float(r["width_meters"]) for r in rows])}
    # Fallback: parse satellite filenames; assume the VIGOR footprint (640 px at zoom 20).
    names = sorted(os.path.basename(p) for p in glob.glob(str(vigor_dir / "satellite" / "satellite_*")))
    lat = np.array([float(n.split("_")[1]) for n in names]); lon = np.array([float(Path(n).stem.split("_")[2]) for n in names])
    w = 640 * 156543.03 / (2 ** 20) * np.cos(np.radians(lat))
    return {"name": names, "lat": lat, "lon": lon, "w": w}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vigor_dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=6, help="frames to render")
    ap.add_argument("--roll_deg", type=float, default=0.0, help="also render the panorama rolled by this (preview a fix)")
    ap.add_argument("--out_dir", type=Path, default=None, help="default <vigor_dir>/orientation_check")
    args = ap.parse_args()

    vigor_dir = args.vigor_dir; out_dir = args.out_dir or vigor_dir / "orientation_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(vigor_dir / "extraction_log.csv")))
    picks = pick_frames(travel_bearings(rows), args.n)
    if not picks:
        raise SystemExit("no frames with enough GPS motion to define a travel direction")
    tiles = load_tiles(vigor_dir)
    font = ImageFont.load_default()
    name = vigor_dir.name

    for i, travel in picks:
        r = rows[i]; pano_path = vigor_dir / "panorama" / r["output_filename"]
        a = np.asarray(Image.open(pano_path).convert("RGB")); W = a.shape[1]
        panels = [("AS STORED (markers assume north at centre column)", mark_pano(Image.fromarray(a).resize((PANO_W, PANO_H)), travel, font))]
        if args.roll_deg:
            rolled = np.roll(a, int(round(W * args.roll_deg / 360.0)), axis=1)
            panels.append((f"ROLLED {args.roll_deg:+.0f} deg (preview)", mark_pano(Image.fromarray(rolled).resize((PANO_W, PANO_H)), travel, font)))
        sat, osm = overhead(vigor_dir, tiles, float(r["lat"]), float(r["lng"]), travel, font)
        S = sat.size[0]
        fig = Image.new("RGB", (PANO_W, len(panels) * (PANO_H + 28) + S + 40), (0, 0, 0)); d = ImageDraw.Draw(fig)
        y = 0
        for title, im in panels:
            d.text((8, y + 6), f"{name} frame {i} {r['output_filename']}   {title}", fill=(255, 255, 255), font=font)
            fig.paste(im, (0, y + 24)); y += PANO_H + 28
        d.text((8, y + 6), "the road ahead must sit on the magenta line; the arrow is the same GPS travel direction, north-up", fill=(255, 255, 255), font=font)
        fig.paste(sat, (0, y + 24))
        if osm is not None:
            fig.paste(osm, (S + 20, y + 24))
        out = out_dir / f"orientation_f{i:05d}.jpg"; fig.save(out, quality=88)
        print(f"  wrote {out}  (travel {travel:.0f} deg)")

    (out_dir / "README.md").write_text(
        f"Orientation check for {name}: red = S/W/N/E under the north-at-centre convention, magenta = GPS direction of travel.\n"
        "REVIEW BEFORE RUNNING LATER STAGES: the road ahead must sit on the magenta line in every figure.\n"
        "If it sits a constant angle away, the heading source is not the camera bearing (see mapillary_to_vigor.py) and the\n"
        "panoramas must be re-rolled; nothing downstream (pinholes, Gemini extraction, WAG matrices) is valid until then.\n")
    print(f"REVIEW REQUIRED: {out_dir} ({len(picks)} figures)")


if __name__ == "__main__":
    main()
