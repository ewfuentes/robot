"""Export a CrossText2Loc (num_panos, num_sats) similarity matrix for a VIGOR-format environment.

Text side: scene descriptions from `extract_gemini_landmarks_from_panoramas --prompt_type scene_description`
(`<output_base>/<name>/scene_descriptions.json`, keyed by pano_id). Image side: `<dataset>/<satellite_subdir>/`
(the satellite patches, or a rendered OSM Carto subdir). Rows/columns follow VigorDataset order, so the
output drops straight into `similarity_matrices/` for `similarity_matrix_metrics` and the histogram filter.

Example:
  bazel run //experimental/overhead_matching/swag/scripts:export_ct2l_similarity -- \
    --dataset_path /data/overhead_matching/datasets/VIGOR/Seattle \
    --captions_json /data/overhead_matching/datasets/scene_descriptions/Seattle/scene_descriptions.json \
    --checkpoint /data/overhead_matching/datasets/cvgtext/models/long_model_NewYork-mixed_1e-05_128_sat_epoch34_46.25.pth \
    --output_path /data/overhead_matching/datasets/VIGOR/Seattle/similarity_matrices/ct2l_ny_sat.pt
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.data import vigor_dataset as vd
from experimental.overhead_matching.swag.evaluation import retrieval_metrics
from experimental.overhead_matching.swag.model import crosstext2loc_encoder as enc
from experimental.overhead_matching.swag.scripts.export_correspondence_similarity import auto_detect_landmark_version


def load_captions(path: Path) -> dict[str, str]:
    raw = json.loads(Path(path).read_text())
    return {k: (v["description"] if isinstance(v, dict) else v) for k, v in raw.items()}


def load_dataset(dataset_path: Path, landmark_version: str) -> vd.VigorDataset:
    """Geometry (patch grid, positives) always comes from `satellite/`, whatever gallery images are used."""
    return vd.VigorDataset(dataset_path, vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None, panorama_tensor_cache_info=None, should_load_images=False,
        should_load_landmarks=True, landmark_version=landmark_version))


def gallery_image_paths(dataset: vd.VigorDataset, satellite_subdir: str) -> list[str]:
    """Same filenames as `satellite/`, read from `<dataset>/<satellite_subdir>/` (e.g. half-res Carto renders)."""
    paths = []
    for p in dataset._satellite_metadata.path:
        p = Path(p)
        q = p if satellite_subdir == "satellite" else p.parent.parent / satellite_subdir / p.name
        if not q.exists() and q.with_suffix(".png").exists():
            q = q.with_suffix(".png")
        paths.append(str(q))
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} gallery images missing under {satellite_subdir}/, e.g. {missing[0]}")
    return paths


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--landmark_version", default=None, help="Defaults to the environment's single version")
    parser.add_argument("--captions_json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="A released or fine-tuned long_model_*.pth")
    parser.add_argument("--satellite_subdir", default="satellite", help="'satellite' or a rendered map subdir")
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--image_batch_size", type=int, default=32)
    parser.add_argument("--text_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    landmark_version = args.landmark_version or auto_detect_landmark_version(args.dataset_path)
    dataset = load_dataset(args.dataset_path, landmark_version)
    pano_ids = list(dataset._panorama_metadata.pano_id)
    gallery_paths = gallery_image_paths(dataset, args.satellite_subdir)

    captions = load_captions(args.captions_json)
    texts = [captions.get(pid, "") for pid in pano_ids]
    missing = [pid for pid, t in zip(pano_ids, texts) if not t]
    print(f"{len(pano_ids)} panoramas ({len(missing)} without a caption -> empty text), "
          f"{len(gallery_paths)} gallery images from {args.satellite_subdir}/")

    device = torch.device(args.device)
    model, preprocessor = enc.build_model(args.checkpoint, device)
    text_feats = enc.encode_texts(model, preprocessor, texts, device, args.text_batch_size)
    image_feats = enc.encode_images(model, preprocessor, gallery_paths, device, args.image_batch_size, args.num_workers,
                                    desc=args.satellite_subdir)
    similarity = (text_feats @ image_feats.T).float()

    # Metrics over captioned panoramas only, so subset runs report something meaningful.
    has_caption = [i for i, t in enumerate(texts) if t]
    captioned = copy.copy(dataset)
    captioned._panorama_metadata = dataset._panorama_metadata.iloc[has_caption].reset_index(drop=True)
    metrics = retrieval_metrics.compute_top_k_metrics(similarity[has_caption], captioned, ks=[1, 5, 10])
    metrics["num_captioned_panoramas"] = len(has_caption)
    print("Retrieval over captioned panoramas:", json.dumps({k: round(float(v), 4) for k, v in metrics.items()}))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(similarity, args.output_path)
    args.output_path.with_suffix(".json").write_text(json.dumps({
        "timestamp": datetime.now().isoformat(), "method": "crosstext2loc", "tf32": True,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "captions_json": str(args.captions_json), "num_missing_captions": len(missing),
        "missing_caption_pano_ids": missing[:1000], "dataset_path": str(args.dataset_path),
        "satellite_subdir": args.satellite_subdir, "landmark_version": landmark_version,
        "similarity_shape": list(similarity.shape), "retrieval_metrics": {k: float(v) for k, v in metrics.items()},
        "git_commit": _git_sha()}, indent=2))
    print(f"Saved {tuple(similarity.shape)} to {args.output_path}")


if __name__ == "__main__":
    main()
