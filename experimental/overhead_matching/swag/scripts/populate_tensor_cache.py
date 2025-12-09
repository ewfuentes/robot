
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


def extract_unique_datasets_from_config(train_config: train.TrainConfig,
                                       dataset_base_path: Path) -> list[tuple[Path, str, float, float]]:
    """
    Extract unique dataset configurations from training config.
    Returns list of (dataset_path, landmark_version, panorama_landmark_radius_px, landmark_correspondence_inflation_factor) tuples.
    Deduplicates datasets that appear multiple times (e.g., in train and validation with different factors).
    """
    # Collect all dataset configs (train + validation)
    all_dataset_configs = [train_config.dataset_config]
    if train_config.validation_dataset_configs:
        all_dataset_configs.extend(train_config.validation_dataset_configs)

    # Deduplicate by (dataset_path, landmark_version, panorama_landmark_radius_px, landmark_correspondence_inflation_factor)
    # Use dict to preserve order
    unique_datasets = {}
    for dataset_config in all_dataset_configs:
        for path_str in dataset_config.paths:
            # Convert relative path to absolute path
            dataset_path = dataset_base_path / path_str

            key = (
                dataset_path,
                dataset_config.landmark_version,
                dataset_config.panorama_landmark_radius_px,
                dataset_config.landmark_correspondence_inflation_factor
            )

            if key not in unique_datasets:
                unique_datasets[key] = True

    return list(unique_datasets.keys())


def get_cache_output_path(field_spec: str,
                          landmark_version: str,
                          panorama_landmark_radius_px: float,
                          landmark_correspondence_inflation_factor: float,
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
    hash_struct = vd.HashStruct(
        model_config=model_config,
        patch_dims=patch_dims,
        landmark_version=landmark_version,
        panorama_landmark_radius_px=panorama_landmark_radius_px,
        landmark_correspondence_inflation_factor=landmark_correspondence_inflation_factor)
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
                                 panorama_landmark_radius_px: float,
                                 landmark_correspondence_inflation_factor: float,
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
            panorama_landmark_radius_px=panorama_landmark_radius_px,
            landmark_correspondence_inflation_factor=landmark_correspondence_inflation_factor,
            factor=1.0,  # Always use full dataset for caching
            should_load_images=should_load_images,
            should_load_landmarks=should_load_landmarks))

    model_type = 'satellite' if 'sat_model_config' == parts[0] else 'panorama'
    dataset = (dataset.get_sat_patch_view() if 'sat_model_config' == parts[0]
               else dataset.get_pano_view())

    return dataset, model_type, requirements


def process_single_cache(field_spec: str,
                         landmark_version: str,
                         panorama_landmark_radius_px: float,
                         landmark_correspondence_inflation_factor: float,
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
    hash_struct = vd.HashStruct(
        model_config=model_config,
        patch_dims=patch_dims,
        landmark_version=landmark_version,
        panorama_landmark_radius_px=panorama_landmark_radius_px,
        landmark_correspondence_inflation_factor=landmark_correspondence_inflation_factor)
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


def load_train_config(train_config_path: Path) -> train.TrainConfig:
    """Load a single training config."""
    with open(train_config_path, 'r') as file_in:
        return msgspec.yaml.decode(
            file_in.read(), type=train.TrainConfig, dec_hook=msgspec_dec_hook)


def compute_task_requirements(field_spec: str, train_config: train.TrainConfig) -> tuple[str, frozenset]:
    """
    Compute the (model_type, requirements) tuple for a task without building the dataset.
    This is used for sorting tasks to maximize cache reuse.

    Returns: (model_type, requirements) where requirements is a frozenset of ExtractorDataRequirement.
    """
    parts = field_spec.split('.')
    model_config = train_config
    for p in parts:
        if isinstance(model_config, dict):
            model_config = model_config[p]
        else:
            model_config = getattr(model_config, p)
    aux_info = getattr(train_config, parts[0]).auxiliary_info

    # Create temporary model to determine requirements
    temp_model = spe.create_extractor(model_config, aux_info)
    requirements = derive_data_requirements_from_model(
        temp_model,
        use_cached_extractors=None)

    model_type = 'satellite' if parts[0] == 'sat_model_config' else 'panorama'

    # Convert set to frozenset so it can be used as a sort key
    return (model_type, frozenset(requirements))


def build_dataset_to_tasks_explicit_mode(
        train_configs: dict[Path, train.TrainConfig],
        dataset_path: Path,
        landmark_version: str,
        panorama_landmark_radius_px: float,
        landmark_correspondence_inflation_factor: float,
        field_spec: str | None = None) -> dict[tuple[Path, str, float, float], list[tuple[Path, str]]]:
    """
    Build dataset-to-tasks mapping when the dataset is specified explicitly

    Args:
        train_configs: Mapping of config paths to loaded TrainConfig objects
        dataset_path: Path to dataset or directory containing datasets
        landmark_version: Landmark version (e.g., 'v3')
        panorama_landmark_radius_px: Panorama landmark radius in pixels
        landmark_correspondence_inflation_factor: Landmark correspondence inflation factor
        field_spec: Optional specific field spec to cache (if None, auto-detect from config)

    Returns:
        Mapping of (dataset_path, landmark_version, radius, inflation_factor) -> [(config_path, field_spec), ...]
    """
    print("\nMode: Explicit dataset specification")
    if landmark_version is None:
        raise ValueError("--landmark_version is required when using --dataset")

    dataset_paths = get_dataset_paths(dataset_path)
    print(f"Found {len(dataset_paths)} dataset(s):")
    for ds in dataset_paths:
        print(f"  - {ds}")

    dataset_to_tasks = {}  # {(dataset_path, lv, radius, inflation_factor): [(config_path, field_spec), ...]}

    # For each config, get its field specs and map to datasets
    for config_path, train_config in train_configs.items():
        # Get field specs for this config
        if field_spec is not None:
            field_specs = [field_spec]
        else:
            field_specs = extract_field_specs_from_config(train_config)
            if not field_specs:
                print(f"Warning: No cached extractors found in {config_path.name}")
                continue

        # Map each dataset to this config's field specs
        for ds in dataset_paths:
            key = (ds, landmark_version, panorama_landmark_radius_px, landmark_correspondence_inflation_factor)
            if key not in dataset_to_tasks:
                dataset_to_tasks[key] = []
            for fs in field_specs:
                dataset_to_tasks[key].append((config_path, fs))

    return dataset_to_tasks


def build_dataset_to_tasks_config_derived_mode(
        train_configs: dict[Path, train.TrainConfig],
        dataset_base_path: Path,
        field_spec: str | None = None) -> dict[tuple[Path, str, float, float], list[tuple[Path, str]]]:
    """
    Build dataset-to-tasks mapping by deriving datasets from training configs.

    Args:
        train_configs: Mapping of config paths to loaded TrainConfig objects
        dataset_base_path: Base path where datasets are located
        field_spec: Optional specific field spec to cache (if None, auto-detect from config)

    Returns:
        Mapping of (dataset_path, landmark_version, radius, inflation_factor) -> [(config_path, field_spec), ...]
    """
    print("\nMode: Deriving datasets from training configs")
    if dataset_base_path is None:
        raise ValueError("Either --dataset or --dataset_base must be provided")

    dataset_to_tasks = {}  # {(dataset_path, lv, radius, inflation_factor): [(config_path, field_spec), ...]}

    # For each config, extract its datasets and field specs
    for config_path, train_config in train_configs.items():
        # Get field specs for this config
        if field_spec is not None:
            field_specs = [field_spec]
        else:
            field_specs = extract_field_specs_from_config(train_config)
            if not field_specs:
                print(f"Warning: No cached extractors found in {config_path.name}")
                continue

        # Get unique datasets from this config
        dataset_configs = extract_unique_datasets_from_config(train_config, dataset_base_path)

        # Map each dataset to this config's field specs
        for ds_path, lv, radius, inflation_factor in dataset_configs:
            key = (ds_path, lv, radius, inflation_factor)
            if key not in dataset_to_tasks:
                dataset_to_tasks[key] = []
            for fs in field_specs:
                dataset_to_tasks[key].append((config_path, fs))

    return dataset_to_tasks


def main(train_config_paths: list[Path],
         base_output_path: Path,
         batch_size: int,
         field_spec: str | None = None,
         skip_existing: bool = False,
         # Mode 1: Explicit dataset specification
         dataset_path: Path | None = None,
         landmark_version: str | None = None,
         panorama_landmark_radius_px: float | None = None,
         landmark_correspondence_inflation_factor: float = 1.0,
         # Mode 2: Derive from config
         dataset_base_path: Path | None = None):
    """
    Process tensor caching for the given configs.

    Two modes of operation:
    1. Explicit mode: Provide dataset_path, landmark_version, panorama_landmark_radius_px, and landmark_correspondence_inflation_factor
    2. Config-derived mode: Provide dataset_base_path, and datasets are derived from training configs

    When multiple configs are provided, they are processed by dataset first to maximize cache reuse.
    """
    # Load all training configs
    print(f"\n{'#'*80}")
    print(f"Loading {len(train_config_paths)} training config(s)")
    print(f"{'#'*80}")

    train_configs = {}
    for config_path in train_config_paths:
        print(f"  Loading: {config_path}")
        train_configs[config_path] = load_train_config(config_path)

    # Build mapping of (dataset_config) -> list of (train_config_path, field_spec) that need it
    # This allows us to process by dataset first for maximum cache reuse
    if dataset_path is not None:
        # Mode 1: Explicit dataset specification - all configs use same dataset params
        dataset_to_tasks = build_dataset_to_tasks_explicit_mode(
            train_configs=train_configs,
            dataset_path=dataset_path,
            landmark_version=landmark_version,
            panorama_landmark_radius_px=panorama_landmark_radius_px,
            landmark_correspondence_inflation_factor=landmark_correspondence_inflation_factor,
            field_spec=field_spec)
    else:
        # Mode 2: Derive from configs
        dataset_to_tasks = build_dataset_to_tasks_config_derived_mode(
            train_configs=train_configs,
            dataset_base_path=dataset_base_path,
            field_spec=field_spec)

    # Print summary
    print(f"\nFound {len(dataset_to_tasks)} unique dataset configuration(s):")
    for (ds_path, lv, radius, inflation_factor), tasks in dataset_to_tasks.items():
        num_configs = len(set(config_path for config_path, _ in tasks))
        print(f"  - {ds_path.name} (landmark_version={lv}, radius={radius}, inflation={inflation_factor}): {len(tasks)} task(s) from {num_configs} config(s)")

    # Sort tasks within each dataset by (model_type, requirements) to maximize cache reuse
    # Also compute and store requirements to avoid recomputing during processing
    print(f"\nSorting tasks by data requirements to maximize cache usage...")
    for dataset_key in dataset_to_tasks:
        tasks = dataset_to_tasks[dataset_key]
        # Compute requirements for each task and sort by them
        tasks_with_requirements = []
        for config_path, field_spec_str in tasks:
            train_config = train_configs[config_path]
            model_type, requirements = compute_task_requirements(field_spec_str, train_config)
            # Sort by (model_type, sorted requirements) for consistent ordering
            sort_key = (model_type, tuple(sorted(requirements)))
            tasks_with_requirements.append((sort_key, config_path, field_spec_str, model_type, requirements))

        # Sort and store with precomputed requirements
        tasks_with_requirements.sort(key=lambda x: x[0])
        dataset_to_tasks[dataset_key] = [(config_path, field_spec, model_type, requirements)
                                          for _, config_path, field_spec, model_type, requirements in tasks_with_requirements]

    # Process by dataset to maximize cache reuse
    total_tasks = sum(len(tasks) for tasks in dataset_to_tasks.values())
    current_task = 0

    for (ds_path, lv, radius, inflation_factor), tasks in dataset_to_tasks.items():
        print(f"\n{'='*80}")
        print(f"Processing dataset: {ds_path.name}")
        print(f"  landmark_version={lv}, panorama_landmark_radius_px={radius}, inflation_factor={inflation_factor}")
        print(f"  {len(tasks)} task(s) to process")
        print(f"{'='*80}")

        # Cache to opportunistically reuse datasets within this dataset config
        previous_dataset_cache = None

        for config_path, field_spec_str, current_model_type, current_requirements in tasks:
            current_task += 1
            train_config = train_configs[config_path]

            print(f"\n{'-'*80}")
            print(f"Task {current_task}/{total_tasks}: {config_path.name} - {field_spec_str}")
            print(f"  Model type: {current_model_type}, Requirements: {sorted([r.name for r in current_requirements])}")
            print(f"{'-'*80}")

            # Check if we should skip this cache
            output_path, _ = get_cache_output_path(
                field_spec=field_spec_str,
                landmark_version=lv,
                panorama_landmark_radius_px=radius,
                landmark_correspondence_inflation_factor=inflation_factor,
                dataset_path=ds_path,
                base_output_path=base_output_path,
                train_config=train_config)

            if skip_existing and should_skip_cache(output_path):
                print(f'✓ Skipping existing cache at: {output_path}')
                continue

            # Check if we can reuse the previous dataset (already computed requirements above)
            dataset_to_use = None
            if previous_dataset_cache is not None:
                cached_model_type, cached_requirements, cached_dataset = previous_dataset_cache

                # Check if we can reuse (model_type and requirements must match)
                if (cached_model_type == current_model_type and
                    cached_requirements == current_requirements):
                    print(f"✓ Reusing previous dataset (model_type={current_model_type}, requirements={sorted([r.name for r in current_requirements])})")
                    dataset_to_use = cached_dataset
                else:
                    print(f"✗ Cannot reuse dataset - rebuilding (previous: {cached_model_type}, {sorted([r.name for r in cached_requirements])})")

            # Build new dataset if we can't reuse
            if dataset_to_use is None:
                dataset_to_use, model_type, requirements = build_dataset_for_extractor(
                    field_spec=field_spec_str,
                    landmark_version=lv,
                    panorama_landmark_radius_px=radius,
                    landmark_correspondence_inflation_factor=inflation_factor,
                    dataset_path=ds_path,
                    train_config=train_config)
                previous_dataset_cache = (current_model_type, current_requirements, dataset_to_use)
                print(f"Built new dataset (model_type={current_model_type}, requirements={sorted([r.name for r in current_requirements])})")

            process_single_cache(
                field_spec=field_spec_str,
                landmark_version=lv,
                panorama_landmark_radius_px=radius,
                landmark_correspondence_inflation_factor=inflation_factor,
                dataset_path=ds_path,
                base_output_path=base_output_path,
                batch_size=batch_size,
                train_config=train_config,
                dataset=dataset_to_use
            )

    print(f"\n{'='*80}")
    print(f"ALL COMPLETE!")
    print(f"Processed {len(train_configs)} config(s), {len(dataset_to_tasks)} dataset(s), {total_tasks} total task(s)")
    print(f"{'='*80}")
    return total_tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate tensor cache for SWAG models. Supports multiple training configs "
                    "and processes them by dataset for maximum cache reuse.")

    parser.add_argument('--train_config', type=str, nargs='+', required=True,
                        help='Path(s) to training config YAML file(s). Multiple configs can be provided and '
                             'will be processed by dataset first for maximum cache reuse.')
    parser.add_argument('--field_spec', type=str, default=None,
                        help='Field spec (e.g., sat_model_config.extractor_config_by_name.dinov3_feature_extractor). '
                             'If not provided, auto-detects from use_cached_extractors in config.')

    # Mode 1: Explicit dataset specification
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset or directory containing multiple datasets. '
                             'When provided, requires --landmark_version and --panorama_landmark_radius_px. '
                             'If not provided, datasets are derived from training config using --dataset_base.')

    # Mode 2: Config-derived (with default)
    parser.add_argument('--dataset_base', type=str,
                        default='/data/overhead_matching/datasets/VIGOR/',
                        help='Base path where datasets are located. Used when --dataset is not provided. '
                             'Datasets will be derived from training config. (default: /data/overhead_matching/datasets/VIGOR/)')

    parser.add_argument('--landmark_version', type=str, default=None,
                        help='Landmark version (e.g., v3). Required when using --dataset.')
    parser.add_argument('--panorama_landmark_radius_px', type=float, default=None,
                        help='Panorama landmark radius in pixels. Required when using --dataset.')
    parser.add_argument('--landmark_correspondence_inflation_factor', type=float, default=1.0,
                        help='Landmark correspondence inflation factor (default: 1.0)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument("--skip-existing", action='store_true',
                        help='Skip caches that already exist')

    output_path = Path('~/.cache/robot/overhead_matching/tensor_cache/').expanduser()
    args = parser.parse_args()

    # Validate mode-specific requirements
    if args.dataset is not None:
        if args.landmark_version is None:
            parser.error("--landmark_version is required when using --dataset")
        if args.panorama_landmark_radius_px is None:
            parser.error("--panorama_landmark_radius_px is required when using --dataset")

    # Convert train_config paths to Path objects
    train_config_paths = [Path(p) for p in args.train_config]

    import ipdb
    with ipdb.launch_ipdb_on_exception():
        main(train_config_paths=train_config_paths,
             base_output_path=output_path,
             batch_size=args.batch_size,
             field_spec=args.field_spec,
             skip_existing=args.skip_existing,
             # Mode 1 parameters
             dataset_path=Path(args.dataset) if args.dataset else None,
             landmark_version=args.landmark_version,
             panorama_landmark_radius_px=args.panorama_landmark_radius_px,
             landmark_correspondence_inflation_factor=args.landmark_correspondence_inflation_factor,
             # Mode 2 parameter
             dataset_base_path=Path(args.dataset_base))
