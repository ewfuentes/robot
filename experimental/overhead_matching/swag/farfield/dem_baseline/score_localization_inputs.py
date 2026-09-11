"""Produce retrieval score fields for a dataset's keyframes (CLD-3 producer).

For every keyframe panorama: extract the query crop ring, embed with the
ported CrossLocate network, and jointly score the depth-render database over
(location, heading). The resulting per-keyframe fields are written in the
``localization/retrieval.py`` artifact contract, ready for
``localization:run_retrieval``.

Keyframe indices are the dataset's own (``dataset.load_frames`` order — the
same indexing build_export gives the odometry/truth in localization_inputs),
so the fields line up with any export built from the same frozen dataset.

    bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:score_localization_inputs -- \
        --dataset mount_washington_20260815_leg2 \
        --db_dir /data/farfield_matching/artifacts/depth_render_db/mount_washington/v1_dev100m \
        --weights .../converted_weights.npz \
        --output_dir /data/farfield_matching/artifacts/retrieval_observations/mount_washington_20260815_leg2/dev100m_v1
"""

import argparse
import hashlib
import time
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.farfield import (
    dataset as dataset_lib,
    paths as paths_lib,
)
from experimental.overhead_matching.swag.farfield.dem_baseline import (
    crosslocate_net,
    panorama_score,
    query_crops,
    render_db,
)
from experimental.overhead_matching.swag.farfield.localization import (
    retrieval,
    structs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    paths_lib.add_arguments(parser)
    parser.add_argument("--db_dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--keyframe_stride", type=int, default=1)
    parser.add_argument("--crop_top_k", type=int, default=None,
                        help="robust aggregation: score each (location, "
                             "shift) by its k best-matching crops instead "
                             "of all of them (occlusion study; recorded in "
                             "the scorer string)")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    paths = paths_lib.resolve(parser, args,
                              require=("panorama_dir", "dataset_base"))

    db = render_db.load_database(args.db_dir, device=args.device)
    manifest = db["manifest"]
    crop_config = query_crops.CropRingConfig(
        n_crops=manifest["render_config"]["n_yaw"],
        fov_deg=manifest["render_config"]["fov_deg"])
    model = crosslocate_net.CrossLocateVGG16MAC().to(args.device).eval()
    crosslocate_net.load_converted_weights(model, args.weights)

    frames = dataset_lib.load_frames(paths.dataset_base)
    selected = frames[::args.keyframe_stride]
    n_nodes = len(db["x_m"])
    n_bins = manifest["render_config"]["n_yaw"]
    scores = np.zeros((len(selected), n_nodes, n_bins), dtype=np.float32)

    started = time.monotonic()
    with torch.inference_mode():
        for i, frame in enumerate(selected):
            pano = np.asarray(Image.open(
                paths.panorama_dir / f"{frame.pano_stem}.jpg").convert("RGB"))
            ring = query_crops.extract_crop_ring(pano, crop_config)
            batch = torch.stack([
                crosslocate_net.rgb_query_tensor(crop) for crop in ring
            ]).to(args.device)
            joint = panorama_score.joint_scores(model(batch),
                                                db["descriptors"],
                                                crop_top_k=args.crop_top_k)
            scores[i] = joint.scores.cpu().numpy()
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.monotonic() - started)
                print(f"{i + 1}/{len(selected)} keyframes ({rate:.2f}/s)",
                      flush=True)

    # Node positions leave the artifact anchor-free (lat/lon).
    import pyproj
    to_wgs84 = pyproj.Transformer.from_crs(
        manifest["lattice"]["crs"], "EPSG:4326", always_xy=True)
    lon_deg, lat_deg = to_wgs84.transform(db["x_m"], db["y_m"])

    db_manifest_sha = hashlib.sha256(
        (args.db_dir / "manifest.json").read_bytes()).hexdigest()
    weights_sha = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    meta = retrieval.RetrievalFieldsMeta(
        schema_version=structs.SCHEMA_VERSION,
        dataset=paths.dataset,
        n_keyframes=len(selected),
        n_nodes=n_nodes,
        n_heading_bins=n_bins,
        node_spacing_m=manifest["lattice"]["spacing_m"],
        db_dir=str(args.db_dir),
        db_manifest_sha256=db_manifest_sha,
        scorer=f"dem_baseline.crosslocate_vgg16_mac@{weights_sha[:12]}"
               + (f" crop_top_k={args.crop_top_k}"
                  if args.crop_top_k is not None else ""))
    retrieval.write_fields(
        args.output_dir, meta, np.asarray(lat_deg), np.asarray(lon_deg),
        scores, np.asarray([f.frame_idx for f in selected]),
        [f.pano_stem for f in selected])
    print(f"score spread (per-field max - median): "
          f"{np.median(scores.max(axis=(1, 2)) - np.median(scores, axis=(1, 2))):.4f}"
          f" median across {len(selected)} keyframes")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
