
import subprocess
import sys
import logging
from pathlib import Path
import os
import boto3
from urllib.parse import urlparse
import glob


def install_s3_wheel(s3_uri):
    """Download and install wheel from S3"""
    # Parse S3 URI
    parsed = urlparse(s3_uri)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        # Download wheel
        s3_client = boto3.client('s3')
        local_wheel_path = f'/tmp/{key.split("/")[-1]}'
        s3_client.download_file(bucket, key, local_wheel_path)
    else:
        local_wheel_path = parsed.path
    # Install wheel
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', local_wheel_path])

    # This is a hack due to bazel wheels packaging dependencies in another site-packages directory
    to_add = []
    for p in sys.path:
        p = Path(p)
        maybe_to_add = p / 'site-packages'
        if maybe_to_add.exists():
            to_add.append(str(maybe_to_add))
    sys.path.extend(to_add)


wheel_s3_uri = os.environ.get("CUSTOM_WHEEL_S3_URI")
install_s3_wheel(wheel_s3_uri)

import experimental.overhead_matching.swag.scripts.evaluate_model_on_paths as emops

train_config_path = glob.glob("/opt/ml/input/data/config_file/*.yaml")[0]
t.main(dataset_path=Path("/opt/ml/input/data/train"),
       opt_config_path=Path(train_config_path),
       output_dir=Path("/opt/ml/model"),
       tensorboard_output=Path("/opt/ml/output/tensorboard"),
       quiet=True)

