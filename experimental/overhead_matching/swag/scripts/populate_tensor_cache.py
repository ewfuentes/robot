
import argparse
from pathlib import Path

import common.torch.load_torch_deps
import torch
import tqdm
import lmdb
import numpy as np

import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.scripts import train
import experimental.overhead_matching.swag.model.swag_patch_embedding as spe
import msgspec
import hashlib
from typing import Any
import io
import dataclasses


class HashStruct(msgspec.Struct, frozen=True):
    model_config: Any
    patch_dims: tuple[int, int]


def enc_hook(obj):
    if isinstance(obj, Path):
        return str(obj)
    else:
        raise ValueError(f"Unhandled Value: {obj}")


def dec_hook(type, obj):
    if type is Path:
        return Path(obj)
    raise ValueError(f"Unhandled type: {type=} {obj=}")


def compute_config_hash(obj):
    yaml_str = msgspec.yaml.encode(obj, enc_hook=enc_hook, order='deterministic')
    return yaml_str, hashlib.sha256(yaml_str)


def main(train_config_path: Path,
         field_spec: str,
         dataset_path: Path,
         output_path: Path,
         idx_start: None | int,
         idx_end: None | int,
         batch_size: int):
    # Load the training config
    with open(train_config_path, 'r') as file_in:
        train_config = msgspec.yaml.decode(
            file_in.read(), type=train.TrainConfig, dec_hook=dec_hook)

    # Get the desired model config
    parts = field_spec.split('.')
    model_config = train_config
    for p in parts:
        model_config = getattr(model_config, p)
    patch_dims = getattr(train_config, parts[0]).patch_dims
    hash_struct = HashStruct(model_config=model_config, patch_dims=patch_dims)
    yaml_str, config_hash = compute_config_hash(hash_struct)
    print('computing cache for: ', hash_struct, 'with hash: ', config_hash.hexdigest())

    # Create the model
    model = spe.create_extractor(model_config)
    model = model.cuda()

    # Construct the dataset
    dataset = vd.VigorDataset(
        dataset_path,
        vd.VigorDatasetConfig(
            satellite_patch_size=train_config.sat_model_config.patch_dims,
            panorama_size=train_config.pano_model_config.patch_dims,
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None))
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
    output_path.mkdir(exist_ok=True, parents=True)
    with lmdb.open(str(output_path), map_size=mmap_size) as db:
        with db.begin(write=True) as txn:
            txn.put(b"config_hash", config_hash.digest())
            txn.put(b"config", yaml_str)

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
            # Write the results
            with db.begin(write=True) as txn:
                for batch_idx in range(model_input.image.shape[0]):
                    if 'mask' in out:
                        selector = ~out['mask'][batch_idx]
                    else:
                        selector = slice(None)

                    to_write = {}
                    for key, value in out.items():
                        to_write[key] = value[batch_idx, selector].cpu().numpy()

                    ostream = io.BytesIO()
                    np.savez(ostream, **to_write)
                    key = model_input.metadata[batch_idx]["index"].to_bytes(8)
                    txn.put(key, ostream.getvalue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', type=str, required=True)
    parser.add_argument('--field_spec', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--idx_start', type=int, default=None)
    parser.add_argument('--idx_end', type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    main(train_config_path=Path(args.train_config),
         field_spec=args.field_spec,
         dataset_path=Path(args.dataset),
         output_path=Path(args.output),
         idx_start=args.idx_start,
         idx_end=args.idx_end,
         batch_size=args.batch_size)
