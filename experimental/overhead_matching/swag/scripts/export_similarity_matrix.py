import argparse
import datetime
import hashlib
import inspect
import os
import subprocess
import tempfile
import warnings
from pathlib import Path

import common.torch.load_torch_deps
import torch
import common.torch.load_and_save_models as lsm
import experimental.overhead_matching.swag.data.satellite_embedding_database as sed
import experimental.overhead_matching.swag.data.vigor_dataset as vd
import experimental.overhead_matching.swag.evaluation.evaluate_swag as es
from experimental.overhead_matching.swag.model import patch_embedding, swag_patch_embedding
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    derive_data_requirements_from_model,
)
from experimental.overhead_matching.swag.model.swag_config_types import (
    ExtractorDataRequirement,
)
import msgspec
import json


SATELLITE_EMBEDDINGS_SCHEMA = "swag_satellite_embeddings/v2"
LEGACY_SATELLITE_EMBEDDINGS_SCHEMA = "swag_satellite_embeddings/v1"
_SATELLITE_CACHE_IDENTITY_FIELDS = (
    "satellite_filenames",
    "satellite_files_sha256",
    "satellite_model_sha256",
    "satellite_behavior",
)


def _published_artifact_path(path: Path) -> Path:
    """Return the eventual path for a file written in artifact staging."""
    path = path.expanduser().resolve()
    if path.parent.name.endswith(".incomplete"):
        destination = path.parent.with_name(
            path.parent.name.removesuffix(".incomplete"))
        return destination / path.name
    return path


def _load_training_model_config(
        training_output: Path, config_field: str) -> dict:
    """Load one model config from a training output, independent of its path."""
    config_json_path = training_output / "config.json"
    config_yaml_path = training_output / "train_config.yaml"
    if config_json_path.exists():
        training_config = json.loads(config_json_path.read_text())
    elif config_yaml_path.exists():
        import yaml
        training_config = yaml.safe_load(config_yaml_path.read_bytes())
    else:
        raise FileNotFoundError(
            f"No config.json or train_config.yaml found in {training_output}")
    try:
        return training_config[config_field]
    except KeyError as exc:
        raise KeyError(
            f"{training_output} training config has no {config_field!r}") from exc


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
        model_config_json = _load_training_model_config(
            path.parent, config_field)
        config = msgspec.json.decode(
            json.dumps(model_config_json),
            type=patch_embedding.WagPatchEmbeddingConfig
            | swag_patch_embedding.SwagPatchEmbeddingConfig)

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


def override_tag_text_embeddings(models, pickle_path: Path) -> int:
    """Repoint tag-bundle extractors at `pickle_path` and force a reload.

    Both extractor classes lazy-load their tag-value embedding table from a
    path attribute at first forward, so swapping the attribute before export
    is sufficient. Returns the number of extractors updated.
    """
    from experimental.overhead_matching.swag.model import tag_bundle_extractor as tbe
    pickle_path = Path(pickle_path).expanduser().resolve()
    if not pickle_path.exists():
        raise FileNotFoundError(f"override embeddings pickle not found: {pickle_path}")
    count = 0
    for model in models:
        for module in model.modules():
            if isinstance(module, tbe.OSMTagBundleExtractor):
                module._embedding_path = pickle_path
            elif isinstance(module, tbe.PanoramaTagBundleExtractor):
                module._text_embedding_path = pickle_path
            else:
                continue
            module._text_embeddings = None
            module._files_loaded = False
            count += 1
    return count


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


def _validate_satellite_embeddings(
        embeddings, satellite_filenames: list[str], source: Path) -> None:
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(f"{source} embeddings must be a torch.Tensor")
    if embeddings.ndim not in (2, 3) or (
            embeddings.ndim == 3 and embeddings.shape[1] != 1):
        raise ValueError(
            f"{source} embeddings have unsupported shape {tuple(embeddings.shape)}")
    if embeddings.shape[0] != len(satellite_filenames):
        raise ValueError(
            f"{source} has {embeddings.shape[0]} embedding rows for "
            f"{len(satellite_filenames)} satellite filenames")
    if not torch.is_floating_point(embeddings) or not torch.isfinite(embeddings).all():
        raise ValueError(f"{source} embeddings must be finite floating-point values")


def _hash_ordered_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        name = path.name.encode()
        size = path.stat().st_size
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload, path: Path) -> None:
    """Save in the destination directory, then atomically reveal the cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(temporary_fd)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _satellite_forward_code_sha256(sat_model) -> str:
    """Hash source contents for every Python class in the module graph/MRO."""
    classes = {
        base
        for module in sat_model.modules()
        for base in type(module).__mro__
        if base is not object
    }
    digest = hashlib.sha256(b"swag_satellite_forward_code/v1\0")
    source_hashes = {}
    for cls in sorted(classes, key=lambda value: (
            value.__module__, value.__qualname__)):
        qualified_name = f"{cls.__module__}.{cls.__qualname__}"
        try:
            source_path = inspect.getsourcefile(cls)
        except TypeError:
            source_path = None
        if source_path is not None and Path(source_path).is_file():
            if source_path not in source_hashes:
                source_hashes[source_path] = _sha256_file(Path(source_path))
            source_hash = source_hashes[source_path]
        else:
            try:
                source = inspect.getsource(cls).encode("utf-8")
            except (OSError, TypeError):
                source = b"<python-source-unavailable>"
            source_hash = hashlib.sha256(source).hexdigest()
        for value in (qualified_name, source_hash):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _satellite_behavior_identity(
        sat_model, dataset: vd.VigorDataset,
        satellite_requirements: set[ExtractorDataRequirement],
        cached_extractors: list[str],
        tag_text_embeddings_override: Path | None,
        model_config=None) -> dict:
    """Describe every non-weight input that changes satellite embeddings."""
    config = dataset._config
    tensor_cache_info = config.satellite_tensor_cache_info
    if model_config is None:
        model_config = getattr(sat_model, "_config", None)
    if model_config is None:
        raise ValueError(
            "satellite model has no embedded config; load its model config "
            "from the training output before constructing a cache identity")
    return {
        "schema": "swag_satellite_embedding_behavior/v1",
        "model_class": (
            f"{type(sat_model).__module__}.{type(sat_model).__qualname__}"),
        "model_config_sha256": vd.compute_config_hash(model_config),
        "forward_code_sha256": _satellite_forward_code_sha256(sat_model),
        "model_patch_dims": list(sat_model.patch_dims),
        "runtime_versions": {
            # torch.__version__ is a TorchVersion (a str subclass) in recent
            # releases.  Store builtins so weights_only=True can reload the
            # otherwise data-only cache without allowlisting Python globals.
            "torch": str(torch.__version__),
            "torchvision": str(vd.tv.__version__),
        },
        "cached_extractors": sorted(str(value) for value in cached_extractors),
        "data_requirements": sorted(
            str(value) for value in satellite_requirements),
        "tensor_cache_config_sha256": (
            vd.compute_config_hash(tensor_cache_info)
            if tensor_cache_info is not None else None),
        "dataset_preprocessing": {
            "schema": "vigor_dataset_satellite_preprocessing/v1",
            "load_image_source_sha256": hashlib.sha256(
                inspect.getsource(vd.load_image).encode("utf-8")).hexdigest(),
            "convert_image_dtype": True,
            "resize_shape": list(dataset._satellite_patch_size),
            "should_load_images": config.should_load_images,
            "should_load_landmarks": config.should_load_landmarks,
            "landmark_version": config.landmark_version,
            "panorama_landmark_radius_px": config.panorama_landmark_radius_px,
            "landmark_correspondence_inflation_factor": (
                config.landmark_correspondence_inflation_factor),
        },
        "tag_text_embeddings_override_sha256": (
            _sha256_file(tag_text_embeddings_override)
            if tag_text_embeddings_override is not None else None),
    }


def _satellite_cache_identity(
        satellite_paths: list[Path], sat_model,
        satellite_model_path: Path, satellite_behavior: dict) -> dict:
    return {
        "satellite_filenames": [path.name for path in satellite_paths],
        "satellite_files_sha256": _hash_ordered_files(satellite_paths),
        # Informational only: relocation does not change model behavior.
        "satellite_model_path": str(satellite_model_path.expanduser().resolve()),
        "satellite_model_sha256": es.hash_model(sat_model).hex(),
        "satellite_behavior": satellite_behavior,
    }


def _panorama_identity(
        panorama_paths: list[Path], pano_model,
        panorama_model_path: Path, model_config) -> dict:
    """Bind panorama rows to their image bytes and exact model state/config."""
    return {
        "panorama_filenames": [path.name for path in panorama_paths],
        "panorama_files_sha256": _hash_ordered_files(panorama_paths),
        # Informational only: relocation does not change model behavior.
        "panorama_model_path": str(panorama_model_path.expanduser().resolve()),
        "panorama_model_sha256": es.hash_model(pano_model).hex(),
        "panorama_model_config_sha256": vd.compute_config_hash(model_config),
    }


def load_satellite_embeddings(
        path: Path, expected_identity: dict,
        allow_legacy: bool = False) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Satellite embedding cache {path} is unreadable or incomplete; "
            "remove it and rerun to rebuild it atomically.") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a dictionary")
    schema = payload.get("schema")
    if schema == LEGACY_SATELLITE_EMBEDDINGS_SCHEMA:
        if not allow_legacy:
            raise ValueError(
                f"{path} uses legacy cache schema {schema}, which does not bind "
                "model config, preprocessing, or tag-embedding overrides. Pass "
                "--allow_legacy_satellite_embeddings to reuse it explicitly.")
        warnings.warn(
            f"Reusing legacy satellite embedding cache {path}; model config, "
            "preprocessing, and tag-embedding override identity are unverified.",
            RuntimeWarning,
            stacklevel=2,
        )
        identity_fields = _SATELLITE_CACHE_IDENTITY_FIELDS[:-1]
    elif schema == SATELLITE_EMBEDDINGS_SCHEMA:
        identity_fields = _SATELLITE_CACHE_IDENTITY_FIELDS
    else:
        raise ValueError(f"{path} has an unsupported satellite embedding schema")
    for key in identity_fields:
        expected = expected_identity[key]
        if payload.get(key) != expected:
            raise ValueError(f"{path} {key} does not match the current input")
    embeddings = payload.get("embeddings")
    _validate_satellite_embeddings(
        embeddings, expected_identity["satellite_filenames"], path)
    return embeddings


def load_or_build_satellite_embeddings(
        path: Path, dataset: vd.VigorDataset, sat_model,
        identity: dict,
        device: torch.device,
        allow_legacy: bool = False) -> tuple[torch.Tensor, str]:
    path = path.expanduser()
    if path.exists():
        print(f"Loading satellite embeddings from {path}")
        return load_satellite_embeddings(path, identity, allow_legacy), "reused"

    print(f"Building satellite embeddings for cache {path}")
    loader = vd.get_dataloader(
        dataset.get_sat_patch_view(), batch_size=96, num_workers=8)
    embeddings = sed.build_satellite_db(
        sat_model, loader, device=device).detach().cpu()
    _validate_satellite_embeddings(
        embeddings, identity["satellite_filenames"], path)
    payload = {
        "schema": SATELLITE_EMBEDDINGS_SCHEMA,
        **identity,
        "embeddings": embeddings,
    }
    _atomic_torch_save(payload, path)
    print(f"Saved satellite embeddings {tuple(embeddings.shape)} to {path}")
    return embeddings, "computed"


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
    parser.add_argument("--satellite_subdir", type=str, default="satellite",
                        help="Subdirectory of dataset_path holding satellite tiles "
                             "(e.g. 'satellite' or 'satellite_osm'). Must match what the model was trained on.")
    parser.add_argument("--satellite_dir", type=Path, default=None,
                        help="External directory containing satellite patches. Overrides "
                             "dataset_path/satellite_subdir.")
    parser.add_argument("--satellite_embeddings_path", type=Path, default=None,
                        help="Reusable satellite embedding cache. Existing caches are "
                             "validated against satellite contents/order, model state/code/"
                             "config, preprocessing, and tag overrides; a missing cache is "
                             "computed and saved. Model location is informational only.")
    parser.add_argument("--allow_legacy_satellite_embeddings", action="store_true",
                        help="Explicitly reuse a v1 satellite embedding cache, whose "
                             "model config, preprocessing, and tag override are unbound.")
    parser.add_argument("--panorama_landmark_radius_px", type=float, default=640.0,
                        help="Match training config; only relevant if model uses landmark cache.")
    parser.add_argument("--landmark_correspondence_inflation_factor", type=float, default=1.0,
                        help="Match training config; only relevant if model uses landmark cache.")
    parser.add_argument("--disable_safa_cache", action="store_true",
                        help="Force live SAFA computation (load images, ignore tensor cache). "
                             "Needed for cities that weren't pre-cached during training.")
    parser.add_argument("--tag_text_embeddings_override", type=Path, default=None,
                        help="Repoint every TagBundleExtractor in the loaded models at this "
                             "tag-value text-embeddings pickle before export. Use when exporting "
                             "a city whose tag values postdate the pickle recorded in the model "
                             "config (the override must be a superset of it).")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser()
    pano_model, sat_model, checkpoint_idx = load_models_from_training_output(
        model_path, device=args.device, checkpoint=args.checkpoint,
        fallback_to_config=args.fallback_to_config)

    if args.tag_text_embeddings_override is not None:
        n = override_tag_text_embeddings(
            [pano_model, sat_model], args.tag_text_embeddings_override)
        print(f"Overrode tag text embeddings on {n} extractor(s) -> "
              f"{args.tag_text_embeddings_override}")

    # Determine effective use_cached_extractors. With --disable_safa_cache,
    # treat everything as uncached so SAFA runs live on images.
    if args.disable_safa_cache:
        sat_cached = []
        pano_cached = []
        sat_cache_info = {}
        pano_cache_info = {}
    else:
        sat_cached = getattr(sat_model._config, "use_cached_extractors", [])
        pano_cached = getattr(pano_model._config, "use_cached_extractors", [])
        sat_cache_info = sat_model.cache_info()
        pano_cache_info = pano_model.cache_info()

    sat_req = derive_data_requirements_from_model(
        sat_model, use_cached_extractors=sat_cached)
    pano_req = derive_data_requirements_from_model(
        pano_model, use_cached_extractors=pano_cached)
    all_req = sat_req | pano_req
    should_load_images = ExtractorDataRequirement.IMAGES in all_req
    should_load_landmarks = ExtractorDataRequirement.LANDMARKS in all_req

    dataset_keys = [Path(args.dataset_path).name]
    dataset_config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None if args.disable_safa_cache else vd.TensorCacheInfo(
            dataset_keys=dataset_keys,
            model_type="satellite",
            landmark_version=args.landmark_version,
            panorama_landmark_radius_px=args.panorama_landmark_radius_px,
            landmark_correspondence_inflation_factor=args.landmark_correspondence_inflation_factor,
            extractor_info=sat_cache_info,
        ),
        panorama_tensor_cache_info=None if args.disable_safa_cache else vd.TensorCacheInfo(
            dataset_keys=dataset_keys,
            model_type="panorama",
            landmark_version=args.landmark_version,
            panorama_landmark_radius_px=args.panorama_landmark_radius_px,
            landmark_correspondence_inflation_factor=args.landmark_correspondence_inflation_factor,
            extractor_info=pano_cache_info,
        ),
        should_load_images=should_load_images,
        should_load_landmarks=should_load_landmarks,
        landmark_version=args.landmark_version,
        factor=args.factor,
        satellite_patch_size=sat_model.patch_dims,
        panorama_size=pano_model.patch_dims,
        satellite_subdir=args.satellite_subdir,
        satellite_dir=(args.satellite_dir.expanduser()
                       if args.satellite_dir is not None else None),
    )
    print(f"Data requirements: images={should_load_images}, landmarks={should_load_landmarks}")
    print(f"Loading dataset from {args.dataset_path}")
    dataset = vd.VigorDataset(config=dataset_config, dataset_path=args.dataset_path)
    num_satellites = len(dataset._satellite_metadata)
    num_panoramas = len(dataset._panorama_metadata)
    print(f"Dataset: {num_satellites} satellites, {num_panoramas} panoramas")

    panorama_model_config = getattr(pano_model, "_config", None)
    if panorama_model_config is None:
        panorama_model_config = _load_training_model_config(
            model_path, "pano_model_config")
    panorama_identity = _panorama_identity(
        [Path(path) for path in dataset._panorama_metadata["path"]],
        pano_model,
        model_path / f"{checkpoint_idx}_panorama",
        panorama_model_config,
    )

    satellite_model_config = getattr(sat_model, "_config", None)
    if satellite_model_config is None:
        satellite_model_config = _load_training_model_config(
            model_path, "sat_model_config")
    satellite_behavior = _satellite_behavior_identity(
        sat_model, dataset, sat_req, sat_cached,
        args.tag_text_embeddings_override, satellite_model_config)
    satellite_embeddings_identity = _satellite_cache_identity(
        [Path(path) for path in dataset._satellite_metadata["path"]],
        sat_model,
        model_path / f"{checkpoint_idx}_satellite",
        satellite_behavior,
    )

    satellite_embeddings = None
    satellite_embeddings_source = None
    if args.satellite_embeddings_path is not None:
        satellite_embeddings, satellite_embeddings_source = (
            load_or_build_satellite_embeddings(
                args.satellite_embeddings_path, dataset, sat_model,
                satellite_embeddings_identity, args.device,
                args.allow_legacy_satellite_embeddings))

    similarity = es.compute_cached_similarity_matrix(
        dataset=dataset,
        pano_model=pano_model,
        sat_model=sat_model,
        device=args.device,
        use_cached_similarity=False,
        satellite_embeddings=satellite_embeddings)

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
        **panorama_identity,
        "satellite_subdir": args.satellite_subdir,
        "satellite_dir": (str(args.satellite_dir.expanduser().resolve())
                          if args.satellite_dir is not None else None),
        "satellite_embeddings_path": (
            str(_published_artifact_path(args.satellite_embeddings_path))
            if args.satellite_embeddings_path is not None else None),
        "satellite_embeddings_source": satellite_embeddings_source,
        "satellite_embeddings_shape": (
            list(satellite_embeddings.shape)
            if satellite_embeddings is not None else None),
        "satellite_files_sha256": satellite_embeddings_identity[
            "satellite_files_sha256"],
        "satellite_model_path": str(
            (model_path / f"{checkpoint_idx}_satellite").resolve()),
        "satellite_model_sha256": satellite_embeddings_identity[
            "satellite_model_sha256"],
        "satellite_behavior": satellite_embeddings_identity[
            "satellite_behavior"],
        "landmark_version": args.landmark_version,
        "num_satellites": num_satellites,
        "num_panoramas": num_panoramas,
        "similarity_shape": list(similarity.shape),
        "matrix_identity": vd.similarity_matrix_identity(
            dataset._panorama_metadata, dataset._satellite_metadata),
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
