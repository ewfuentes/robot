#!/usr/bin/env python3
"""Plot a dataset's landmark catalog so coverage gaps and density are obvious.

Four panels:
  1. map        landmarks by source, the trajectory, the request bbox, and each
                OSM extract's clip boundary -- a missing landmass shows up as a
                blank region with the trajectory running through it
  2. density    log-scaled 2D histogram; empty cells are the thing to look for
  3. range      landmark count vs distance from the track, which is what decides
                whether the bbox buffer was wide enough for far-field work
  4. classes    the most common landmark_type / primary tag values

Also prints (and returns non-zero on) quantitative gap checks, so this is usable
as a pipeline gate and not only as something to eyeball.

    python plot_landmarks.py /data/farfield_matching/mapillary_datasets/folkestone_dover
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Grid used for the emptiness metric. Coarse enough that ordinary sparsity does
# not register, fine enough to localise a missing town.
GRID = 24
# Fail if a run of empty grid columns/rows this wide sits inside the convex span
# of the data -- that is the shape a missing extract makes.
MAX_EMPTY_RUN = 6


def load_trajectory(ds: Path):
    lats, lons = [], []
    with open(ds / "pano_id_mapping.csv") as f:
        for r in csv.DictReader(f):
            lats.append(float(r["lat"]))
            lons.append(float(r["lon"]))
    return np.array(lons), np.array(lats)


def load_landmarks(ds: Path, bbox=None):
    """Landmark representative points + source label, from every source feather.

    Geometries are clipped to the bbox before a representative point is taken.
    Without that, long features that merely *intersect* the request area are
    plotted wherever their midpoint happens to fall: the Dover Strait catalog
    picks up submarine telecom cables (Atlantic Crossing 1, SEA-ME-WE 3, the
    IFA 2000 interconnector) spanning up to 10 degrees, which dragged 94 of 1,708
    UK features outside the bbox and made the gap metrics meaningless.
    """
    lm_dir = ds / "landmarks"
    frames = []
    sources = sorted((lm_dir / "sources").glob("*.feather")) if (lm_dir / "sources").exists() else []
    # Drop the within-sources merge product (osm_<dataset>_v1.feather): it is the
    # union of the per-region feathers sitting beside it, so counting both
    # double-counts every landmark.
    combined = f"osm_{ds.name}_v1"
    if len(sources) > 1:
        sources = [p for p in sources if p.stem != combined] or sources
    if not sources:
        merged = lm_dir / "v1.feather"
        sources = [merged.resolve()] if merged.exists() else []
    if not sources:
        return None

    from shapely import wkb
    from shapely.geometry import box as _box
    clip = _box(*bbox) if bbox else None
    for path in sources:
        df = pd.read_feather(path, columns=None)
        # Tags are one JSON dict column now; older feathers are one column per
        # key. Normalise to a per-row dict either way.
        if "tags" in df.columns:
            import json
            row_tags = [json.loads(t) if isinstance(t, str) else (t or {})
                        for t in df["tags"]]
        else:
            row_tags = None
        pts = []
        for g in df["geometry"]:
            try:
                geom = wkb.loads(bytes(g))
                if clip is not None and not clip.contains(geom):
                    geom = geom.intersection(clip)
                    if geom.is_empty:
                        pts.append((np.nan, np.nan))
                        continue
                p = geom.representative_point()
                pts.append((p.x, p.y))
            except Exception:
                pts.append((np.nan, np.nan))
        sub = pd.DataFrame(pts, columns=["lon", "lat"])
        sub["source"] = path.stem
        sub["landmark_type"] = df.get("landmark_type", pd.Series(["?"] * len(df))).values
        for tag in ("man_made", "natural", "building", "seamark:type", "place", "name"):
            if row_tags is not None:
                sub[tag] = [rt.get(tag) for rt in row_tags]
            else:
                sub[tag] = df[tag].values if tag in df.columns else None
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True).dropna(subset=["lon", "lat"])
    return out


def resolve_bbox(ds: Path, tlon, tlat):
    """The request bbox: from provenance if recorded, else inferred.

    The fallback uses only Point geometries, which are OSM nodes and therefore
    always inside the requested area -- unlike long ways, which can extend far
    beyond it.
    """
    prov = ds / "landmarks" / "PROVENANCE.json"
    if prov.exists():
        b = json.loads(prov.read_text()).get("bbox_wsen")
        if b:
            return tuple(b)
    pts = load_landmarks(ds, bbox=None)
    if pts is not None and len(pts):
        from shapely import wkb  # noqa: F401  (import cost already paid)
        node_like = pts.dropna(subset=["lon", "lat"])
        if len(node_like):
            pad = 0.02
            return (node_like.lon.quantile(0.001) - pad, node_like.lat.quantile(0.001) - pad,
                    node_like.lon.quantile(0.999) + pad, node_like.lat.quantile(0.999) + pad)
    pad = 0.25
    return (tlon.min() - pad, tlat.min() - pad, tlon.max() + pad, tlat.max() + pad)


def clip_boundaries(ds: Path):
    """Geofabrik clip polygons for the extracts this dataset was built from."""
    prov = ds / "landmarks" / "PROVENANCE.json"
    if not prov.exists():
        return []
    try:
        specs = json.loads(prov.read_text()).get("osm_specs") or []
        from experimental.overhead_matching.swag.scripts.pbf_coverage import (
            fetch_poly, parse_poly)
        out = []
        for spec in specs:
            try:
                out.append((spec.rsplit("/", 1)[-1], parse_poly(fetch_poly(spec))))
            except Exception:
                pass
        return out
    except Exception:
        return []


def gap_report(lm, tlon, tlat, bbox):
    """Quantitative emptiness checks over the landmark distribution."""
    west, south, east, north = bbox
    H, xe, ye = np.histogram2d(lm["lon"], lm["lat"], bins=GRID,
                              range=[[west, east], [south, north]])
    occupied = H > 0
    findings = []

    # Empty columns/rows spanning the interior: the signature of a missing extract.
    for axis, label in ((0, "longitude"), (1, "latitude")):
        occ = occupied.any(axis=1 - axis)
        idx = np.where(occ)[0]
        if len(idx) == 0:
            findings.append(("FAIL", "no landmarks at all"))
            continue
        interior = occ[idx[0]:idx[-1] + 1]
        run = best = 0
        for v in interior:
            run = 0 if v else run + 1
            best = max(best, run)
        span = (xe if axis == 0 else ye)
        cell_km = abs(span[1] - span[0]) * 111.0 * (math.cos(math.radians((south + north) / 2))
                                                    if axis == 0 else 1.0)
        if best >= MAX_EMPTY_RUN:
            findings.append(("FAIL", f"{best} consecutive empty {label} bands "
                                     f"(~{best*cell_km:.0f} km) inside the populated "
                                     f"span — looks like a missing extract"))
        else:
            findings.append(("ok", f"largest interior gap along {label}: {best} band(s) "
                                   f"(~{best*cell_km:.0f} km)"))

    frac_empty = 1.0 - occupied.mean()
    findings.append(("ok" if frac_empty < 0.9 else "warn",
                     f"{100*frac_empty:.0f}% of the {GRID}x{GRID} bbox grid has no "
                     f"landmarks (water and open country are legitimately empty)"))

    # A source confined to the rim of the bbox means the buffer only just clipped
    # that landmass -- the Folkestone case, where 25 km reached 1.4007 E and the
    # English coast (Dover 1.31 E) fell outside, leaving 1,699 edge-shaved UK
    # features against 465,569 French ones.
    wspan, hspan = east - west, north - south
    for src, grp in lm.groupby("source"):
        share = len(grp) / len(lm)
        d_edge = np.minimum.reduce([
            (grp.lon.values - west) / wspan, (east - grp.lon.values) / wspan,
            (grp.lat.values - south) / hspan, (north - grp.lat.values) / hspan])
        at_rim = float((d_edge < 0.03).mean())
        if share < 0.2 and at_rim > 0.4:
            near = {"west": (grp.lon.mean() - west) / wspan,
                    "east": (east - grp.lon.mean()) / wspan,
                    "south": (grp.lat.mean() - south) / hspan,
                    "north": (north - grp.lat.mean()) / hspan}
            side = min(near, key=near.get)
            findings.append(("FAIL", f"'{src}' contributes only {100*share:.1f}% of "
                                     f"features and {100*at_rim:.0f}% of them sit on the "
                                     f"bbox rim, mostly at the {side} edge — the buffer "
                                     f"is clipping that landmass rather than covering "
                                     f"it; widen landmark_buffer_km"))
        else:
            findings.append(("ok", f"'{src}': {100*share:.1f}% of features, "
                                   f"{100*at_rim:.0f}% on the bbox rim"))

    # Does the catalog reach out to far-field range, or only hug the track?
    step = max(1, len(tlon) // 400)
    tl = np.stack([tlon[::step], tlat[::step]], axis=1)
    mid = math.radians((south + north) / 2)
    P = np.stack([lm["lon"].values, lm["lat"].values], axis=1)
    sub = P[:: max(1, len(P) // 20000)]
    d = np.sqrt(((sub[:, None, 0] - tl[None, :, 0]) * 111.0 * math.cos(mid)) ** 2
                + ((sub[:, None, 1] - tl[None, :, 1]) * 111.0) ** 2).min(axis=1)
    far = float((d > 5).mean())
    findings.append(("ok" if far > 0.02 else "warn",
                     f"{100*far:.0f}% of landmarks lie >5 km from the track "
                     f"(far-field datasets want a healthy tail here); "
                     f"median {np.median(d):.1f} km, max {d.max():.1f} km"))
    return findings, H, xe, ye, d


def plot(ds: Path, out_path: Path):
    meta = json.loads((ds / "pipeline_metadata.json").read_text())
    tlon, tlat = load_trajectory(ds)

    bbox = resolve_bbox(ds, tlon, tlat)
    lm = load_landmarks(ds, bbox)
    if lm is None or len(lm) == 0:
        print(f"  {ds.name}: no landmark feathers to plot")
        return 1, None
    west, south, east, north = bbox

    findings, H, xe, ye, dists = gap_report(lm, tlon, tlat, bbox)

    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    fig.suptitle(f"{ds.name} — landmark catalog "
                 f"({len(lm):,} features, {meta.get('num_images')} frames, "
                 f"{meta.get('trajectory_km', 0):.1f} km track)", fontsize=14)

    # 1: map
    ax = axes[0][0]
    for src, grp in lm.groupby("source"):
        ax.scatter(grp.lon, grp.lat, s=1.5, alpha=0.25, label=f"{src} ({len(grp):,})")
    ax.plot(tlon, tlat, "-", color="red", lw=2.2, zorder=5)
    ax.plot(tlon[0], tlat[0], "o", color="lime", ms=8, mec="k", zorder=6)
    ax.add_patch(plt.Rectangle((west, south), east - west, north - south,
                               fill=False, ec="k", ls="--", lw=1.6))
    for name, geom in clip_boundaries(ds):
        try:
            for poly in (geom.geoms if hasattr(geom, "geoms") else [geom]):
                x, y = poly.exterior.xy
                ax.plot(x, y, lw=1.0, color="purple", alpha=0.6)
        except Exception:
            pass
    ax.set_xlim(west, east); ax.set_ylim(south, north)
    ax.set_aspect(1 / math.cos(math.radians((south + north) / 2)))
    ax.set_title("landmarks by source + trajectory (purple = extract clip boundary)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=6, color=h.get_facecolor()[0],
                      label=h.get_label())
               for h in ax.collections if h.get_label() and not h.get_label().startswith("_")]
    handles += [Line2D([], [], color="red", lw=2, label="trajectory"),
                Line2D([], [], marker="o", ls="", ms=7, color="lime", mec="k", label="start"),
                Line2D([], [], color="k", ls="--", lw=1.4, label="request bbox"),
                Line2D([], [], color="purple", lw=1.0, label="extract clip boundary")]
    ax.legend(handles=handles, loc="upper right", fontsize=7)

    # 2: density
    ax = axes[0][1]
    im = ax.imshow(np.log10(H.T + 1), origin="lower", aspect="auto",
                   extent=[west, east, south, north], cmap="viridis")
    ax.plot(tlon, tlat, "-", color="red", lw=1.5)
    ax.set_title(f"log10 density on a {GRID}x{GRID} grid (dark = empty)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.colorbar(im, ax=ax, label="log10(count+1)")

    # 3: distance from track
    ax = axes[1][0]
    ax.hist(dists, bins=60, color="steelblue")
    ax.axvline(5, color="r", ls="--", label="5 km")
    ax.set_yscale("log")
    ax.set_title("landmark distance from the trajectory")
    ax.set_xlabel("km from track"); ax.set_ylabel("count (log)")
    ax.legend(fontsize=8)

    # 4: classes
    ax = axes[1][1]
    tag = None
    for cand in ("man_made", "building", "natural", "place"):
        if cand in lm and lm[cand].notna().any():
            tag = cand
            break
    if tag:
        vc = lm[tag].dropna().value_counts().head(14)[::-1]
        ax.barh([str(i) for i in vc.index], vc.values, color="darkorange")
        ax.set_xscale("log")
        ax.set_title(f"most common '{tag}' values")
    else:
        vc = lm["landmark_type"].value_counts().head(14)[::-1]
        ax.barh([str(i) for i in vc.index], vc.values, color="darkorange")
        ax.set_title("landmark_type")
    ax.set_xlabel("count")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    print(f"\n=== {ds.name}: {len(lm):,} landmarks -> {out_path}")
    for kind, msg in findings:
        print(f"  {'FAIL' if kind == 'FAIL' else ('WARN' if kind == 'warn' else 'ok  ')}  {msg}")
    for src, grp in lm.groupby("source"):
        print(f"        {src}: {len(grp):,}")
    return (1 if any(k == "FAIL" for k, _ in findings) else 0), out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="+")
    ap.add_argument("-o", "--output", default=None,
                    help="Output PNG (default: <dataset>/landmarks/landmark_coverage.png)")
    args = ap.parse_args()
    rc = 0
    for d in args.dataset:
        ds = Path(d).resolve()
        out = Path(args.output) if args.output else ds / "landmarks" / "landmark_coverage.png"
        try:
            r, _ = plot(ds, out)
            rc |= r
        except Exception as e:
            print(f"  {ds.name}: ERROR {type(e).__name__}: {e}")
            rc |= 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
