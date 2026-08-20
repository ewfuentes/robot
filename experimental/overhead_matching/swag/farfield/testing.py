"""Synthetic dataset fixtures for farfield tests.

Builds a minimal on-disk dataset that satisfies the contract in
`dataset.py` / `audit_dataset.py`: real (tiny) JPEG panoramas with GPS in the
filenames, agreeing CSV tables, convention metadata, and optionally a
`frame_landmarks`-shaped predictions artifact.
"""

import csv
import json
from pathlib import Path

from experimental.overhead_matching.swag.farfield import geometry as geo

ANCHOR_LAT, ANCHOR_LON = 42.35, -71.05


def default_metadata() -> dict:
    return {
        "is_equirectangular": True,
        "north_aligned": False,
        "azimuth_convention": {
            "images_rotated": False,
            "heading_deg_is_bearing_of": "column_0",
            "formula": "azimuth = (heading_deg + 360*x/W) % 360",
            "mount_offset_frame": geo.MOUNT_OFFSET_FRAME,
        },
        "mount_offset": {
            "mount_offset_deg": 214.0,
            "frame": geo.MOUNT_OFFSET_FRAME,
            "applied_to_heading_deg": False,
            "status": "sun_verified",
            "accuracy_validated": True,
        },
    }


def make_dataset(base: Path, n_frames: int = 4, pano_size=(64, 32),
                 metadata: dict | None = None,
                 skip_pano_numbers: tuple = ()) -> Path:
    """A contract-satisfying synthetic dataset under `base`.

    `skip_pano_numbers` omits those panorama files (but keeps their CSV rows)
    to exercise the pano-number-vs-frame-index divergence.
    """
    from PIL import Image

    base = Path(base)
    pano = base / "panorama"
    pano.mkdir(parents=True)

    rows = []
    for i in range(n_frames):
        lat = ANCHOR_LAT + 1e-4 * i
        lon = ANCHOR_LON + 1e-4 * i
        rows.append({
            "idx": i, "latitude": f"{lat:.7f}", "longitude": f"{lon:.7f}",
            "dist_m": f"{10.0 * i:.1f}", "video_t_s": f"{2.0 * i:.2f}",
        })
        if i in skip_pano_numbers:
            continue
        stem = f"f{i:04d},{lat:.7f},{lon:.7f},"
        Image.new("RGB", pano_size, (40 + 10 * i, 80, 120)).save(
            pano / f"{stem}.jpg")

    kept = [r for r in rows if int(r["idx"]) not in skip_pano_numbers]
    _write_csv(base / "frames_gps.csv",
               ["idx", "latitude", "longitude", "dist_m", "video_t_s"], kept)
    # Reindex so idx stays 0..N-1 contiguous, matching the kept panoramas.
    for new_idx, row in enumerate(kept):
        row["idx"] = new_idx
    _write_csv(base / "pano_id_mapping.csv", ["idx", "pano_id"],
               [{"idx": r["idx"], "pano_id": f"f{int(r['idx']):04d}"}
                for r in kept])
    _write_csv(base / "extraction_log.csv",
               ["idx", "mapillary_id", "sequence_id", "sequence_position"],
               [{"idx": r["idx"], "mapillary_id": f"m{int(r['idx']):06d}",
                 "sequence_id": "seq0", "sequence_position": r["idx"]}
                for r in kept])
    _write_csv(base / "intrinsics.csv",
               ["idx", "heading_deg", "heading_reference", "hfov_deg"],
               [{"idx": r["idx"], "heading_deg": "45.0",
                 "heading_reference": "column_0", "hfov_deg": "360.0"}
                for r in kept])
    (base / "pipeline_metadata.json").write_text(
        json.dumps(metadata if metadata is not None else default_metadata(),
                   indent=1))
    return base


def make_predictions(frame_landmarks_dir: Path, per_stem: dict) -> Path:
    """A frame_landmarks-shaped artifact: {pano_stem: [landmark dicts]}."""
    out = (Path(frame_landmarks_dir) / "sentences" / "results" / "batch0"
           / "prediction-model-0")
    out.mkdir(parents=True)
    with open(out / "predictions.jsonl", "w") as f:
        for stem, landmarks in per_stem.items():
            record = {
                "key": stem,
                "response": {"candidates": [{"content": {"parts": [{
                    "text": json.dumps({"location_type": "harbor",
                                        "landmarks": landmarks}),
                }]}}]},
            }
            f.write(json.dumps(record) + "\n")
    return Path(frame_landmarks_dir)


def landmark(name: str, boxes: list, primary=("man_made", "tower"),
             confidence="high") -> dict:
    """One prediction entry; boxes are (yaw, xmin, ymin, xmax, ymax)."""
    return {
        "description": name,
        "confidence": confidence,
        "primary_tag": {"key": primary[0], "value": primary[1]},
        "additional_tags": [{"key": "name", "value": name}],
        "bounding_boxes": [
            {"yaw_angle": yaw, "xmin": xmin, "ymin": ymin,
             "xmax": xmax, "ymax": ymax}
            for yaw, xmin, ymin, xmax, ymax in boxes],
    }


def _write_csv(path: Path, fields: list, rows: list) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
