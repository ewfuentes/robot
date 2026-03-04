import argparse
import datetime
import subprocess
from pathlib import Path

import common.torch.load_torch_deps
import torch
import common.torch.load_and_save_models as lsm
import experimental.overhead_matching.swag.data.vigor_dataset as vd
import experimental.overhead_matching.swag.evaluation.evaluate_swag as es
from experimental.overhead_matching.swag.model import patch_embedding, swag_patch_embedding
import msgspec
import json


def load_model(path, device='cuda', fallback_to_config=False):
    path = Path(path)
    try:
        model = lsm.load_model(path, device=device)
        model.patch_dims
        model.model_input_from_batch
    except Exception as e:
        if not fallback_to_config:
            raise
        print("Failed to load model via pickle, falling back to config+weights:", e)
        config_field = ("sat_model_config" if 'satellite' in path.name
                        else "pano_model_config")
        # Try config.json first, then train_config.yaml
        config_json_path = path.parent / "config.json"
        config_yaml_path = path.parent / "train_config.yaml"
        if config_json_path.exists():
            training_config_json = json.loads(config_json_path.read_text())
            model_config_json = training_config_json[config_field]
            config = msgspec.json.decode(
                json.dumps(model_config_json),
                type=patch_embedding.WagPatchEmbeddingConfig | swag_patch_embedding.SwagPatchEmbeddingConfig)
        elif config_yaml_path.exists():
            import yaml
            train_config_dict = yaml.safe_load(config_yaml_path.read_bytes())
            model_config_json = train_config_dict[config_field]
            config = msgspec.json.decode(
                json.dumps(model_config_json),
                type=patch_embedding.WagPatchEmbeddingConfig | swag_patch_embedding.SwagPatchEmbeddingConfig)
        else:
            raise FileNotFoundError(f"No config.json or train_config.yaml found in {path.parent}")

        model_weights = torch.load(path / 'model_weights.pt', weights_only=True)
        # Strip _orig_mod. prefix from torch.compile-wrapped models
        model_weights = {k.removeprefix("_orig_mod."): v for k, v in model_weights.items()}
        model_type = (patch_embedding.WagPatchEmbedding
                      if isinstance(config, patch_embedding.WagPatchEmbeddingConfig)
                      else swag_patch_embedding.SwagPatchEmbedding)
        model = model_type(config)
        model.load_state_dict(model_weights)
        model = model.to(device)
    return model


def get_latest_checkpoint(p: Path):
    checkpoints = []
    for dir in p.glob("[0-9]*"):
        checkpoints.append(dir.name.split('_')[0])
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint directories matching '[0-9]*' found in {p}. "
            f"Contents: {[x.name for x in p.iterdir()] if p.exists() else '(directory does not exist)'}")
    return sorted(checkpoints)[-1]


def load_models_from_training_output(base_path: Path, device='cuda', checkpoint='latest',
                                     fallback_to_config=False):
    if checkpoint == 'latest':
        checkpoint = get_latest_checkpoint(base_path)
    sat_path = base_path / f"{checkpoint}_satellite"
    pano_path = base_path / f"{checkpoint}_panorama"
    print(f"Loading satellite model from {sat_path}")
    sat_model = load_model(sat_path, device=device, fallback_to_config=fallback_to_config)
    print(f"Loading panorama model from {pano_path}")
    pano_model = load_model(pano_path, device=device, fallback_to_config=fallback_to_config)
    sat_model.eval()
    pano_model.eval()
    return pano_model, sat_model, checkpoint


def get_git_info():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip()
        return {
            "git_commit": commit,
            "git_branch": branch,
            "git_dirty": bool(dirty),
        }
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"WARNING: Could not collect git info: {e}")
        return {"git_commit": "unknown", "git_branch": "unknown", "git_dirty": None}


def main():
    parser = argparse.ArgumentParser(
        description="Export similarity matrix for a model on a VIGOR dataset split.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to training output directory (containing checkpoint dirs)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to VIGOR dataset split (e.g. /data/.../VIGOR/Seattle)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for the similarity matrix .pt file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--factor", type=float, default=1.0,
                        help="Dataset subsampling factor (1.0 = full dataset)")
    parser.add_argument("--landmark_version", type=str, default="v1")
    parser.add_argument("--checkpoint", type=str, default="best",
                        help="Checkpoint prefix to load (e.g. 'best', 'last', '0050', or 'latest' for highest numbered)")
    parser.add_argument("--fallback_to_config", action="store_true",
                        help="If pickle loading fails, fall back to loading from config+weights")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    pano_model, sat_model, checkpoint_idx = load_models_from_training_output(
        model_path, device=args.device, checkpoint=args.checkpoint,
        fallback_to_config=args.fallback_to_config)

    dataset_config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=True,
        should_load_landmarks=False,
        landmark_version=args.landmark_version,
        factor=args.factor,
        satellite_patch_size=sat_model.patch_dims,
        panorama_size=pano_model.patch_dims,
    )
    print(f"Loading dataset from {args.dataset_path}")
    dataset = vd.VigorDataset(config=dataset_config, dataset_path=args.dataset_path)
    num_satellites = len(dataset._satellite_metadata)
    num_panoramas = len(dataset._panorama_metadata)
    print(f"Dataset: {num_satellites} satellites, {num_panoramas} panoramas")

    similarity = es.compute_cached_similarity_matrix(
        dataset=dataset,
        pano_model=pano_model,
        sat_model=sat_model,
        device=args.device,
        use_cached_similarity=False)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(similarity.cpu(), output_path)
    print(f"Saved similarity matrix {tuple(similarity.shape)} to {output_path}")

    # Save metadata alongside the matrix
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_path": str(model_path.resolve()),
        "checkpoint": checkpoint_idx,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "dataset_factor": args.factor,
        "landmark_version": args.landmark_version,
        "num_satellites": num_satellites,
        "num_panoramas": num_panoramas,
        "similarity_shape": list(similarity.shape),
        "device": args.device,
        **get_git_info(),
    }

    # Include training config if available
    config_json_path = model_path / "config.json"
    config_yaml_path = model_path / "train_config.yaml"
    if config_json_path.exists():
        metadata["training_config"] = json.loads(config_json_path.read_text())
    elif config_yaml_path.exists():
        metadata["training_config_path"] = str(config_yaml_path)

    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
