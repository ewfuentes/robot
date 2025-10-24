
import argparse
from pathlib import Path

import common.torch.load_torch_deps
import torch
import tqdm
import lmdb
import numpy as np
from common.python.serialization import msgspec_enc_hook, msgspec_dec_hook
import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.scripts import train
import experimental.overhead_matching.swag.model.swag_patch_embedding as spe
from experimental.overhead_matching.swag.model.swag_model_input_output import derive_data_requirements_from_model
from experimental.overhead_matching.swag.model.swag_config_types import ExtractorDataRequirement
import msgspec
import hashlib
import io
import dataclasses
import sys


def compute_config_hash(obj):
    yaml_str = msgspec.yaml.encode(obj, enc_hook=msgspec_enc_hook, order='deterministic')
    return yaml_str, hashlib.sha256(yaml_str)


def is_valid_dataset(path: Path) -> bool:
    """Check if a directory is a valid VIGOR dataset."""
    return (path.is_dir() and
            (path / "satellite").exists() and
            (path / "panorama").exists())


def get_dataset_paths(dataset_path: Path) -> list[Path]:
    """
    Get list of dataset paths. If dataset_path is a valid dataset, return [dataset_path].
    If it's a directory containing datasets, return all valid dataset subdirectories.
    """
    if is_valid_dataset(dataset_path):
        return [dataset_path]

    # Check if it's a directory containing multiple datasets
    if dataset_path.is_dir():
        datasets = []
        for subdir in sorted(dataset_path.iterdir()):
            if is_valid_dataset(subdir):
                datasets.append(subdir)
        if datasets:
            return datasets

    raise ValueError(f"{dataset_path} is neither a valid dataset nor a directory containing datasets")


def extract_field_specs_from_config(train_config: train.TrainConfig) -> list[str]:
    """
    Extract field specs from the train config based on use_cached_extractors.
    Returns a list of (field_spec, model_type) tuples.
    """
    field_specs = []

    # Check satellite model config
    if hasattr(train_config.sat_model_config, 'use_cached_extractors'):
        for extractor_name in train_config.sat_model_config.use_cached_extractors:
            field_spec = f"sat_model_config.extractor_config_by_name.{extractor_name}"
            field_specs.append(field_spec)

    # Check panorama model config
    if hasattr(train_config.pano_model_config, 'use_cached_extractors'):
        for extractor_name in train_config.pano_model_config.use_cached_extractors:
            field_spec = f"pano_model_config.extractor_config_by_name.{extractor_name}"
            field_specs.append(field_spec)

    return field_specs


def get_cache_output_path(field_spec: str,
                          landmark_version: str,
                          dataset_path: Path,
                          base_output_path: Path,
                          train_config: train.TrainConfig) -> tuple[Path, str]:
    """
    Determine the output path for a cache without building the dataset.
    Returns (output_path, model_type).
    """
    parts = field_spec.split('.')
    model_config = train_config
    for p in parts:
        if isinstance(model_config, dict):
            model_config = model_config[p]
        else:
            model_config = getattr(model_config, p)
    patch_dims = getattr(train_config, parts[0]).patch_dims
    hash_struct = vd.HashStruct(model_config=model_config, patch_dims=patch_dims, landmark_version=landmark_version)
    _, config_hash = compute_config_hash(hash_struct)

    model_type = 'satellite' if 'sat_model_config' == parts[0] else 'panorama'
    output_path = base_output_path / dataset_path.name / model_type / config_hash.hexdigest()

    return output_path, model_type


def should_skip_cache(output_path: Path) -> bool:
    """Check if a cache already exists and is valid."""
    if output_path.exists():
        # Verify it's a valid LMDB database by checking for data.mdb or lock.mdb
        if (output_path / "data.mdb").exists() or (output_path / "lock.mdb").exists():
            return True
    return False


def build_dataset_for_extractor(field_spec: str,
                                 landmark_version: str,
                                 dataset_path: Path,
                                 train_config: train.TrainConfig):
    """
    Build a dataset for a specific extractor.
    Returns (dataset, model_type, requirements).
    """
    # Get the desired model config
    parts = field_spec.split('.')
    model_config = train_config
    for p in parts:
        if isinstance(model_config, dict):
            model_config = model_config[p]
        else:
            model_config = getattr(model_config, p)
    aux_info = getattr(train_config, parts[0]).auxiliary_info

    # Create the model to determine requirements
    model = spe.create_extractor(model_config, aux_info)

    # Construct the dataset
    requirements = derive_data_requirements_from_model(
        model,
        use_cached_extractors=None)

    should_load_images = ExtractorDataRequirement.IMAGES in requirements
    should_load_landmarks = ExtractorDataRequirement.LANDMARKS in requirements

    dataset = vd.VigorDataset(
        dataset_path,
        vd.VigorDatasetConfig(
            satellite_patch_size=train_config.sat_model_config.patch_dims,
            panorama_size=train_config.pano_model_config.patch_dims,
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            landmark_version=landmark_version,
            should_load_images=should_load_images,
            should_load_landmarks=should_load_landmarks))

    model_type = 'satellite' if 'sat_model_config' == parts[0] else 'panorama'
    dataset = (dataset.get_sat_patch_view() if 'sat_model_config' == parts[0]
               else dataset.get_pano_view())

    return dataset, model_type, requirements


def process_single_cache(field_spec: str,
                         landmark_version: str,
                         dataset_path: Path,
                         base_output_path: Path,
                         batch_size: int,
                         train_config: train.TrainConfig,
                         dataset):
    """
    Process a single cache. Assumes skip_existing check has already been done.
    Dataset must be provided (not optional).
    """
    # Get the desired model config
    parts = field_spec.split('.')
    model_config = train_config
    for p in parts:
        if isinstance(model_config, dict):
            model_config = model_config[p]
        else:
            model_config = getattr(model_config, p)
    aux_info = getattr(train_config, parts[0]).auxiliary_info
    patch_dims = getattr(train_config, parts[0]).patch_dims
    hash_struct = vd.HashStruct(model_config=model_config, patch_dims=patch_dims, landmark_version=landmark_version)
    yaml_str, config_hash = compute_config_hash(hash_struct)

    # Determine output path
    model_type = 'satellite' if 'sat_model_config' == parts[0] else 'panorama'
    output_path = base_output_path / dataset_path.name / model_type / config_hash.hexdigest()

    print('computing cache for: ', hash_struct, 'with hash: ', config_hash.hexdigest())

    # Create the model
    model = spe.create_extractor(model_config, aux_info)
    model = model.cuda()

    # Get a dataloader
    dataloader = vd.get_dataloader(dataset, num_workers=4, batch_size=batch_size)

    # Open a database
    mmap_size = 2**40  # 1 TB
    output_path.mkdir(exist_ok=True, parents=True)
    print('writing to:', output_path)
    num_features = 0
    num_items = 0
    with lmdb.open(str(output_path), map_size=mmap_size) as db:
        with db.begin(write=True) as txn:
            txn.put(b"config_hash", config_hash.digest())
            txn.put(b"config", yaml_str)
            txn.put(b"dataset", dataset_path.name.encode('utf-8'))

        # Process the data
        for batch in tqdm.tqdm(dataloader):
            # Create the model input
            if parts[0] == "sat_model_config":
                model_input = spe.ModelInput(
                    image=batch.satellite,
                    metadata=batch.satellite_metadata,
                    cached_tensors=batch.cached_satellite_tensors).to("cuda")
            elif parts[0] == "pano_model_config":
                model_input = spe.ModelInput(
                    image=batch.panorama,
                    metadata=batch.panorama_metadata,
                    cached_tensors=batch.cached_panorama_tensors).to("cuda")
            else:
                raise NotImplementedError
            # Run it through the model
            out = dataclasses.asdict(model(model_input))
            num_features += (~out['mask']).sum()
            num_items += out['mask'].shape[0]
            # Write the results
            with db.begin(write=True) as txn:
                for batch_idx in range(model_input.image.shape[0]):
                    if 'mask' in out:
                        selector = ~out['mask'][batch_idx].cpu().numpy()
                    else:
                        selector = slice(None)

                    to_write = {}
                    for key, value in out.items():
                        if isinstance(value, torch.Tensor):
                            to_write[key] = value[batch_idx, selector].cpu().numpy()
                        if isinstance(value, dict):
                            for k2, v2 in value.items():
                                to_write[f"{key}.{k2}"] = v2[batch_idx, selector].cpu().numpy()

                    ostream = io.BytesIO()
                    np.savez(ostream, **to_write)
                    key = model_input.metadata[batch_idx]["path"].name.encode('utf-8')
                    txn.put(key, ostream.getvalue())
    print(f"Created a total of {num_features} features for {num_items} items, with an average of {num_features / num_items} features per item")


def main(train_config_path: Path,
         dataset_path: Path,
         landmark_version: str,
         base_output_path: Path,
         batch_size: int,
         field_spec: str | None = None,
         skip_existing: bool = False):
    """
    Main entry point. Handles both single field_spec mode and auto-detection mode.
    Also handles both single dataset and multi-dataset modes.
    """
    # Load the training config once
    with open(train_config_path, 'r') as file_in:
        train_config = msgspec.yaml.decode(
            file_in.read(), type=train.TrainConfig, dec_hook=msgspec_dec_hook)

    # Get dataset paths
    dataset_paths = get_dataset_paths(dataset_path)
    print(f"Found {len(dataset_paths)} dataset(s): {[d.name for d in dataset_paths]}")

    # Get field specs
    if field_spec is not None:
        # Single field spec mode (backward compatibility)
        field_specs = [field_spec]
    else:
        # Auto-detect from config
        field_specs = extract_field_specs_from_config(train_config)
        if not field_specs:
            print("No cached extractors found in config. Check sat_model_config.use_cached_extractors and pano_model_config.use_cached_extractors")
            sys.exit(1)
        print(f"Auto-detected {len(field_specs)} extractor(s) to cache:")
        for fs in field_specs:
            print(f"  - {fs}")

    # Process each combination of dataset and field spec
    total_tasks = len(dataset_paths) * len(field_specs)
    current_task = 0

    # Cache to opportunistically reuse datasets
    previous_dataset_cache = None

    for dataset in dataset_paths:
        # Reset cache when switching datasets
        previous_dataset_cache = None

        for field_spec_str in field_specs:
            current_task += 1
            print(f"\n{'='*80}")
            print(f"Processing task {current_task}/{total_tasks}: {dataset.name} - {field_spec_str}")
            print(f"{'='*80}")

            # Check if we should skip this cache
            output_path, current_model_type = get_cache_output_path(
                field_spec=field_spec_str,
                landmark_version=landmark_version,
                dataset_path=dataset,
                base_output_path=base_output_path,
                train_config=train_config)

            if skip_existing and should_skip_cache(output_path):
                print(f'✓ Skipping existing cache at: {output_path}')
                continue

            # Check if we can reuse the previous dataset
            dataset_to_use = None
            if previous_dataset_cache is not None:
                cached_path, cached_model_type, cached_requirements, cached_dataset = previous_dataset_cache

                # Build requirements for current extractor (without creating the dataset)
                parts = field_spec_str.split('.')
                model_config = train_config
                for p in parts:
                    if isinstance(model_config, dict):
                        model_config = model_config[p]
                    else:
                        model_config = getattr(model_config, p)
                aux_info = getattr(train_config, parts[0]).auxiliary_info
                temp_model = spe.create_extractor(model_config, aux_info)
                current_requirements = derive_data_requirements_from_model(
                    temp_model,
                    use_cached_extractors=None)

                # Check if we can reuse
                if (cached_path == dataset and
                    cached_model_type == current_model_type and
                    cached_requirements == current_requirements):
                    print(f"✓ Reusing previous dataset (model_type={current_model_type}, requirements={current_requirements})")
                    dataset_to_use = cached_dataset
                else:
                    print(f"✗ Cannot reuse dataset - rebuilding (model_type changed: {cached_model_type}→{current_model_type}, requirements changed: {cached_requirements}→{current_requirements})")

            # Build new dataset if we can't reuse
            if dataset_to_use is None:
                dataset_to_use, model_type, requirements = build_dataset_for_extractor(
                    field_spec=field_spec_str,
                    landmark_version=landmark_version,
                    dataset_path=dataset,
                    train_config=train_config)
                previous_dataset_cache = (dataset, model_type, requirements, dataset_to_use)
                print(f"Built new dataset (model_type={model_type}, requirements={requirements})")

            process_single_cache(
                field_spec=field_spec_str,
                landmark_version=landmark_version,
                dataset_path=dataset,
                base_output_path=base_output_path,
                batch_size=batch_size,
                train_config=train_config,
                dataset=dataset_to_use
            )

    print(f"\n{'='*80}")
    print(f"Completed all {total_tasks} tasks!")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate tensor cache for SWAG models.")
    parser.add_argument('--train_config', type=str, required=True,
                        help='Path to training config YAML file')
    parser.add_argument('--field_spec', type=str, default=None,
                        help='Field spec (e.g., sat_model_config.extractor_config_by_name.dinov3_feature_extractor). '
                             'If not provided, auto-detects from use_cached_extractors in config.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset or directory containing multiple datasets')
    parser.add_argument('--landmark_version', type=str, required=True,
                        help='Landmark version (e.g., v3)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument("--skip-existing", action='store_true',
                        help='Skip caches that already exist')

    output_path = Path('~/.cache/robot/overhead_matching/tensor_cache/').expanduser()
    args = parser.parse_args()
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        main(train_config_path=Path(args.train_config),
             dataset_path=Path(args.dataset),
             landmark_version=args.landmark_version,
             base_output_path=output_path,
             batch_size=args.batch_size,
             field_spec=args.field_spec,
             skip_existing=args.skip_existing)
