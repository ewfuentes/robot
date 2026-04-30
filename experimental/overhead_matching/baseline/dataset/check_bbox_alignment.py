"""Verify center_zoom_to_bbox against Boston's published bboxes to sub-meter
precision, then dump the worst-case row.

Boston ships satellite_tile_metadata.csv with explicit per-tile north/south/
east/west bounds + width/height in meters. We compute the same bbox from the
filename + zoom and report the largest disagreement in meters.
"""
import math
import sys
from pathlib import Path

import pandas as pd

from experimental.overhead_matching.baseline.dataset import tile_geometry


def deg_to_meters(lat_deg: float, dlat: float, dlon: float) -> tuple[float, float]:
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_deg))
    return abs(dlat) * m_per_deg_lat, abs(dlon) * m_per_deg_lon


def main() -> int:
    csv_path = Path("/data/overhead_matching/datasets/VIGOR/Boston/satellite_tile_metadata.csv")
    if not csv_path.exists():
        print(f"missing {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    print(f"checking {len(df)} Boston tiles at zoom=20, tile_px=640...")

    worst = {"row": None, "max_m": 0.0}
    sums = []
    for row in df.itertuples(index=False):
        derived = tile_geometry.center_zoom_to_bbox(
            row.center_lat, row.center_lon, zoom=20, tile_px=640
        )
        d_n, d_s = abs(derived.north_lat - row.north_lat), abs(derived.south_lat - row.south_lat)
        d_e, d_w = abs(derived.east_lon - row.east_lon), abs(derived.west_lon - row.west_lon)
        m_n, _ = deg_to_meters(row.center_lat, d_n, 0.0)
        m_s, _ = deg_to_meters(row.center_lat, d_s, 0.0)
        _, m_e = deg_to_meters(row.center_lat, 0.0, d_e)
        _, m_w = deg_to_meters(row.center_lat, 0.0, d_w)
        max_m = max(m_n, m_s, m_e, m_w)
        sums.append(max_m)
        if max_m > worst["max_m"]:
            worst = {"row": row, "max_m": max_m, "m_n": m_n, "m_s": m_s, "m_e": m_e, "m_w": m_w,
                     "derived": derived}

    sums.sort()
    print(f"\nBbox edge agreement (meters):")
    print(f"  median: {sums[len(sums)//2]:.4f} m")
    print(f"  p95:    {sums[int(len(sums)*0.95)]:.4f} m")
    print(f"  max:    {max(sums):.4f} m")
    print(f"\nWorst-case tile:")
    r = worst["row"]
    print(f"  file:           {r.file_name}")
    print(f"  center:         ({r.center_lat:.7f}, {r.center_lon:.7f})")
    print(f"  csv N/S/E/W:    {r.north_lat:.7f} / {r.south_lat:.7f} / {r.east_lon:.7f} / {r.west_lon:.7f}")
    print(f"  derived N/S/E/W:{worst['derived'].north_lat:.7f} / {worst['derived'].south_lat:.7f} / {worst['derived'].east_lon:.7f} / {worst['derived'].west_lon:.7f}")
    print(f"  errors (m):     N={worst['m_n']:.4f} S={worst['m_s']:.4f} E={worst['m_e']:.4f} W={worst['m_w']:.4f}")
    print(f"  csv tile size:  {r.width_meters:.4f} m wide × {r.height_meters:.4f} m tall")
    return 0


if __name__ == "__main__":
    sys.exit(main())
