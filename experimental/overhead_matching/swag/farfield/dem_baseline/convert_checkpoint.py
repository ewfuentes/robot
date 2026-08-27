"""Dump a TF1 CrossLocate checkpoint's variables to an .npz, name-keyed.

TensorFlow deliberately stays out of this repo's dependency set; this script
runs in a throwaway environment instead of bazel:

    uv run --python 3.12 --with tensorflow-cpu \
        experimental/overhead_matching/swag/farfield/dem_baseline/convert_checkpoint.py \
        --checkpoint_prefix /data/farfield_matching/models/crosslocate/AlpsPhotosToDepthCompact_31_2/models/GeoImRet_..._init-model-39 \
        --output /data/farfield_matching/models/crosslocate/AlpsPhotosToDepthCompact_31_2/converted_weights.npz

`tf.train.load_checkpoint` reads the V2 TensorBundle format directly; no
graph construction, so a TF1-era checkpoint loads fine under TF2. Values are
stored verbatim (TF layouts, TF names); the torch-side permute lives in
`crosslocate_net.load_converted_weights` so this file remains a faithful dump.

The dump also writes `<output>.manifest.json` with names, shapes, dtypes, and
the sha256 of the checkpoint data file, for the render/port manifest chain.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_prefix", required=True,
                        help="TF checkpoint prefix (path without .index/.data)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import tensorflow as tf  # deliberately not a repo dependency

    reader = tf.train.load_checkpoint(args.checkpoint_prefix)
    shape_map = reader.get_variable_to_shape_map()
    arrays = {}
    records = {}
    for name in sorted(shape_map):
        value = reader.get_tensor(name)
        arrays[name] = value
        records[name] = {"shape": list(np.shape(value)),
                         "dtype": str(np.asarray(value).dtype)}
        print(f"{name}: {records[name]['shape']} {records[name]['dtype']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)

    data_file = Path(args.checkpoint_prefix + ".data-00000-of-00001")
    manifest = {
        "checkpoint_prefix": str(args.checkpoint_prefix),
        "checkpoint_data_sha256": hashlib.sha256(
            data_file.read_bytes()).hexdigest() if data_file.exists() else None,
        "variables": records,
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {args.output} and {manifest_path}")


if __name__ == "__main__":
    main()
