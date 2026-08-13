"""Extract self-contained correspondence data for the seagrant→Boston hand-off bundle.

Reads the raw cost data (`*_raw.pt` + `*_cost_matrix.npy`) produced by
export_correspondence_similarity, resolves each OSM column's lat/lon/name from the Boston
landmark feather, aligns each pano-landmark matrix row back to its source landmark (bbox,
name, description, frame pose), and writes codebase-free data files:

  out_dir/
    correspondence_matrix.npy   (copied; pano_lm x osm_lm, P(match))
    pano_landmarks.csv          (row -> camera, timestamp, frame lat/lon/heading, tags, name, bbox, desc)
    osm_landmarks.csv           (col -> osm_index, tags, name, lat, lon, dist_to_track_m)
    top_matches.json            (per pano landmark: top-K OSM matches w/ score, name, tags, lat/lon, dist)
    bundle_index.json           (provenance + shapes)

A separate (pure-python) packaging step turns these into README + Leaflet viewer + zip.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import geopandas as gpd
from experimental.overhead_matching.swag.data import landmark_schema
import numpy as np
import torch

from experimental.overhead_matching.swag.model.additional_panorama_extractors import load_v2_pickle
from experimental.overhead_matching.swag.model.semantic_landmark_utils import prune_landmark
from experimental.overhead_matching.swag.scripts.landmark_pairing_cli import _should_keep_tag


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_poses(poses_dir: Path):
    """pano_id -> dict(lat, lon, heading_deg, course_deg, cum_dist_m); plus track points."""
    poses = {}
    track = []
    for csv_path, cam in [(poses_dir / "center_poses.csv", "center"),
                          (poses_dir / "right_poses.csv", "right")]:
        for row in csv.DictReader(open(csv_path)):
            pid = f"seagrant_{cam}__{row['timestamp']}"
            lat, lon = float(row["lat"]), float(row["lon"])
            poses[pid] = {
                "lat": lat, "lon": lon,
                "imu_heading_deg": float(row["imu_heading_deg"]) if row["imu_heading_deg"] else None,
                "gps_course_deg": float(row["gps_course_deg"]) if row["gps_course_deg"] else None,
                "cum_dist_m": float(row["cum_dist_m"]),
            }
            track.append((lat, lon))
    return poses, track


def align_pano_rows(pano_v2_pickle: Path):
    """pano_id -> ordered list of source landmark dicts, matching the kept-landmark order
    that extract_tags_from_pano_data produced (the matrix row order)."""
    data = load_v2_pickle(pano_v2_pickle)
    out = {}
    for pano_id, pdata in data["panoramas"].items():
        kept = []
        for lm in pdata["landmarks"]:
            primary = lm.get("primary_tag", {})
            raw = []
            if primary.get("key") and primary.get("value"):
                raw.append((primary["key"], primary["value"]))
            for t in lm.get("additional_tags", []):
                if t.get("key") and t.get("value"):
                    raw.append((t["key"], t["value"]))
            tags = [(k, v) for k, v in raw if _should_keep_tag(k)]
            if not tags:
                continue
            name = next((v for k, v in raw if k == "name"), None)
            kept.append({
                "primary_tag": f"{primary.get('key')}={primary.get('value')}",
                "name": name,
                "confidence": lm.get("confidence", "unknown"),
                "bounding_box": lm.get("bounding_box"),
                "description": lm.get("description", ""),
                "tags": tags,
            })
        out[pano_id.split(",")[0]] = kept
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_pt", type=Path, required=True)
    ap.add_argument("--boston_feather", type=Path, required=True)
    ap.add_argument("--pano_v2_pickle", type=Path, required=True)
    ap.add_argument("--poses_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=25)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw cost data from {args.raw_pt}")
    meta = torch.load(args.raw_pt, weights_only=False)
    cost = np.load(meta["cost_matrix_path"], mmap_mode="r")
    osm_lm_indices = meta["osm_lm_indices"]
    osm_lm_tags = meta["osm_lm_tags"]
    pano_id_to_lm_rows = meta["pano_id_to_lm_rows"]
    pano_lm_tags = meta["pano_lm_tags"]
    n_pano, n_osm = cost.shape
    print(f"  cost matrix {cost.shape}; {n_osm} OSM cols; {len(pano_id_to_lm_rows)} panos")

    # --- OSM geometry/name from feather (col index aligns to feather row index) ---
    print(f"Loading Boston feather {args.boston_feather}")
    feather = gpd.read_feather(args.boston_feather)
    # verify alignment on a sample: pruned feather row == stored osm tags
    for j in (0, n_osm // 2, n_osm - 1):
        idx = osm_lm_indices[j]
        pruned = dict(prune_landmark(landmark_schema.row_dicts(feather.iloc[[idx]])[0]))
        assert pruned == osm_lm_tags[j], f"alignment mismatch at col {j}: {pruned} != {osm_lm_tags[j]}"
    print("  alignment verified (feather row == osm column)")

    cent = feather.geometry.iloc[osm_lm_indices].centroid
    osm_lon = np.asarray(cent.x.values, dtype=np.float64)
    osm_lat = np.asarray(cent.y.values, dtype=np.float64)
    osm_name = [t.get("name", "") for t in osm_lm_tags]

    # --- distance of every OSM landmark to the GPS track ---
    poses, track = load_poses(args.poses_dir)
    tlat = np.array([p[0] for p in track]); tlon = np.array([p[1] for p in track])
    dist_to_track = np.empty(n_osm, dtype=np.float64)
    for j in range(n_osm):
        dist_to_track[j] = haversine_m(osm_lat[j], osm_lon[j], tlat, tlon).min()

    # --- align pano rows to source landmarks ---
    pano_src = align_pano_rows(args.pano_v2_pickle)

    # --- write osm_landmarks.csv ---
    with open(args.out_dir / "osm_landmarks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["col", "osm_index", "name", "tags_json", "lat", "lon", "dist_to_track_m"])
        for j in range(n_osm):
            w.writerow([j, osm_lm_indices[j], osm_name[j], json.dumps(osm_lm_tags[j]),
                        f"{osm_lat[j]:.7f}", f"{osm_lon[j]:.7f}", f"{dist_to_track[j]:.1f}"])

    # --- pano_landmarks.csv + top_matches.json ---
    pano_rows_out = []
    top_matches = []
    for pano_id, rows in pano_id_to_lm_rows.items():
        srcs = pano_src.get(pano_id, [])
        if len(srcs) != len(rows):
            print(f"  WARN {pano_id}: {len(srcs)} src vs {len(rows)} rows; skipping mismatch")
            continue
        cam, ts = pano_id.replace("seagrant_", "").split("__")
        pose = poses.get(pano_id, {})
        for local_i, row in enumerate(rows):
            src = srcs[local_i]
            bb = src["bounding_box"] or {}
            pano_rows_out.append({
                "row": row, "pano_id": pano_id, "camera": cam, "timestamp": ts,
                "frame_lat": pose.get("lat"), "frame_lon": pose.get("lon"),
                "frame_heading_deg": pose.get("imu_heading_deg"),
                "primary_tag": src["primary_tag"], "name": src["name"] or "",
                "confidence": src["confidence"],
                "tags_json": json.dumps(dict(src["tags"])),
                "bbox": json.dumps(bb), "description": src["description"],
            })
            scores = np.asarray(cost[row])
            k = min(args.topk, n_osm)
            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            top_matches.append({
                "row": row, "pano_id": pano_id, "camera": cam, "timestamp": ts,
                "frame_lat": pose.get("lat"), "frame_lon": pose.get("lon"),
                "frame_heading_deg": pose.get("imu_heading_deg"),
                "pano_primary_tag": src["primary_tag"], "pano_name": src["name"] or "",
                "pano_confidence": src["confidence"], "pano_bbox": bb,
                "pano_description": src["description"],
                "matches": [{
                    "col": int(j), "score": float(scores[j]), "osm_name": osm_name[j],
                    "osm_tags": osm_lm_tags[j], "lat": float(osm_lat[j]), "lon": float(osm_lon[j]),
                    "dist_to_track_m": float(dist_to_track[j]),
                } for j in top_idx],
            })

    with open(args.out_dir / "pano_landmarks.csv", "w", newline="") as fh:
        cols = ["row", "pano_id", "camera", "timestamp", "frame_lat", "frame_lon",
                "frame_heading_deg", "primary_tag", "name", "confidence", "tags_json",
                "bbox", "description"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(pano_rows_out)

    with open(args.out_dir / "top_matches.json", "w") as fh:
        json.dump(top_matches, fh)

    # copy the matrix into the bundle
    shutil.copy2(meta["cost_matrix_path"], args.out_dir / "correspondence_matrix.npy")

    index = {
        "n_pano_landmarks": int(n_pano), "n_osm_landmarks": int(n_osm),
        "topk": args.topk, "model_path": meta.get("model_path"),
        "text_embeddings_path": meta.get("text_embeddings_path"),
        "matrix_shape": [int(n_pano), int(n_osm)], "score_semantics": "P(match) in [0,1]",
    }
    with open(args.out_dir / "bundle_index.json", "w") as fh:
        json.dump(index, fh, indent=2)

    print(f"\nWrote bundle data to {args.out_dir}")
    print(f"  pano landmarks: {len(pano_rows_out)}; osm landmarks: {n_osm}")

    # --- spot-check ---
    print("\nSpot-check (top-5 OSM for a few named pano landmarks):")
    shown = 0
    for tm in top_matches:
        if tm["pano_name"] and shown < 6:
            print(f"\n  PANO [{tm['camera']}] {tm['pano_primary_tag']} name={tm['pano_name']!r}")
            for m in tm["matches"][:5]:
                print(f"    {m['score']:.3f}  {m['osm_name']!r:40s} {m['osm_tags']}  "
                      f"({m['dist_to_track_m']:.0f} m)")
            shown += 1


if __name__ == "__main__":
    main()
