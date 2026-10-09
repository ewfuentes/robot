"""Render OSM Carto rasters with python-mapnik. Runs INSIDE overv/openstreetmap-tile-server
(via run_carto_render.sh), where /data/style/mapnik.xml and the PostGIS database live.
Plain python, no repo deps.

usage: render_carto.py <jobs.json> --out-dir DIR [--shard i/n] [--zoom 19] [--buffer 128]
jobs.json: [{"filename": str, "bbox": [xmin, ymin, xmax, ymax] (EPSG:3857 m), "px": int}, ...]

Carto rules are evaluated at --zoom (19 = CVG-Text's tiles) and symbols scaled so that `px`
pixels cover the bbox: mapnik's scale_factor = px / (bbox width in native z19 pixels). For a
320 px render of a 95.5 m z20 bbox that is 1.0 (native z19); consumers upscale x2.
"""
import argparse
import json
import math
import os
import time

import mapnik

MERC_M_PER_PX_Z0 = 2 * math.pi * 6378137.0 / 256.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--xml", default="/data/style/mapnik.xml")
    ap.add_argument("--zoom", type=int, default=19, help="Carto rule zoom to emulate")
    ap.add_argument("--buffer", type=int, default=128, help="label buffer px (renderd uses 128)")
    ap.add_argument("--shard", default="0/1", help="i/n: render every n-th job starting at i")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    jobs = json.load(open(a.jobs))[i::n]

    mapnik.register_fonts("/usr/share/fonts")  # renderd.conf: font_dir=/usr/share/fonts recurse
    m = mapnik.Map(256, 256)
    mapnik.load_map(m, a.xml)
    m.buffer_size = a.buffer
    os.makedirs(a.out_dir, exist_ok=True)

    t0 = time.time()
    done = 0
    for j in jobs:
        out = os.path.join(a.out_dir, j["filename"])
        if os.path.exists(out):
            continue
        px = int(j["px"])
        xmin, ymin, xmax, ymax = j["bbox"]
        sf = px / ((xmax - xmin) / (MERC_M_PER_PX_Z0 / 2 ** a.zoom))
        m.resize(px, px)
        m.zoom_to_box(mapnik.Box2d(xmin, ymin, xmax, ymax))
        im = mapnik.Image(px, px)
        mapnik.render(m, im, sf)
        im.save(out + ".tmp.png", "png")  # write-then-rename: a killed shard leaves no truncated tile
        os.rename(out + ".tmp.png", out)
        done += 1
        if done % 500 == 0:
            print(f"[{a.shard}] {done}/{len(jobs)} {done / (time.time() - t0):.1f} tiles/s", flush=True)
    print(f"[{a.shard}] done {done} of {len(jobs)} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
