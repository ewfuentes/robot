
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
import msgspec
import hashlib
from typing import Any
import io
import dataclasses



def compute_config_hash(obj):
    yaml_str = msgspec.yaml.encode(obj, enc_hook=msgspec_enc_hook, order='deterministic')
    return yaml_str, hashlib.sha256(yaml_str)


def main(train_config_path: Path,
         field_spec: str,
         landmark_version: str, 
         dataset_path: Path,
         base_output_path: Path,
         idx_start: None | int,
         idx_end: None | int,
         batch_size: int):
    # Load the training config
    with open(train_config_path, 'r') as file_in:
        train_config = msgspec.yaml.decode(
            file_in.read(), type=train.TrainConfig, dec_hook=msgspec_dec_hook)

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
    hash_struct = spe.HashStruct(model_config=model_config, patch_dims=patch_dims, landmark_version=landmark_version)
    yaml_str, config_hash = compute_config_hash(hash_struct)
    print('computing cache for: ', hash_struct, 'with hash: ', config_hash.hexdigest())

    # Create the model
    model = spe.create_extractor(model_config, aux_info)
    model = model.cuda()

    # Construct the dataset
    # If the training config specifies that we can ignore images for all extractors
    # then it is safe to ignore images for the specified extractor
    should_load_images = train_config.dataset_config.should_load_images
    dataset = vd.VigorDataset(
        dataset_path,
        vd.VigorDatasetConfig(
            satellite_patch_size=train_config.sat_model_config.patch_dims,
            panorama_size=train_config.pano_model_config.patch_dims,
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            landmark_version=landmark_version,
            should_load_images=should_load_images))
    dataset = (dataset.get_sat_patch_view() if 'sat_model_config' == parts[0]
               else dataset.get_pano_view())
    if idx_start is None:
        idx_start = 0
    if idx_end is None:
        idx_end = len(dataset)
    dataset = torch.utils.data.Subset(dataset, range(idx_start, idx_end))

    # Get a dataloader
    dataloader = vd.get_dataloader(dataset, num_workers=4, batch_size=batch_size)

    # Open a database
    mmap_size = 2**40  # 1 TB
    model_type = 'satellite' if 'sat_model_config' == parts[0] else 'panorama'
    output_path = base_output_path / dataset_path.name / model_type / config_hash.hexdigest()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', type=str, required=True)
    parser.add_argument('--field_spec', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--landmark_version', type=str, required=True)
    parser.add_argument('--idx_start', type=int, default=None)
    parser.add_argument('--idx_end', type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)

    output_path = Path('~/.cache/robot/overhead_matching/tensor_cache/').expanduser()
    args = parser.parse_args()
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        main(train_config_path=Path(args.train_config),
            field_spec=args.field_spec,
            dataset_path=Path(args.dataset),
            landmark_version=args.landmark_version,
            base_output_path=output_path,
            idx_start=args.idx_start,
            idx_end=args.idx_end,
            batch_size=args.batch_size)
