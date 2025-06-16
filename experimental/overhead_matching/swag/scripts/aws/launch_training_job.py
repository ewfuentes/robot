
import boto3
from botocore.exceptions import ClientError
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig
import subprocess
from pathlib import Path
import hashlib
import argparse
import uuid


def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def upload_wheel_with_sha256(wheel_path, bucket, s3_client=None):
    """Upload wheel with SHA256 in the S3 key"""
    if s3_client is None:
        s3_client = boto3.client('s3')
    # Calculate hash
    wheel_hash = calculate_sha256(wheel_path)
    wheel_filename = wheel_path.split('/')[-1]
    # Create S3 key with hash
    s3_key = f'wheels/{wheel_hash}/{wheel_filename}'
    # Check if already exists (optional optimization)
    try:
        s3_client.head_object(Bucket=bucket, Key=s3_key)
        print(f"Wheel already exists in S3: s3://{bucket}/{s3_key}")
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"Uploading wheel to S3: s3://{bucket}/{s3_key}")
            s3_client.upload_file(wheel_path, bucket, s3_key)
        else:
            # Some other error occurred
            raise e
    return f's3://{bucket}/{s3_key}'


def upload_config(config_path, id, bucket):
    s3_client = boto3.client('s3')
    file_stem = config_path.stem
    s3_key = f"train_configs/{file_stem}_{id}.yaml"
    s3_client.upload_file(config_path, bucket, s3_key)
    return f"s3://{bucket}/{s3_key}"


def main(train_config_path):
    session = sagemaker.Session()
    bucket = session.default_bucket()

    # Upload the training config
    id = uuid.uuid1()
    config_s3_uri = upload_config(train_config_path, id, bucket)
    job_name = f"{id}"

    # Build the wheel
    subprocess.check_call(
            "bazel build //experimental/overhead_matching/swag/scripts:train_wheel".split(' '))

    # Get the path to the wheel that was built
    completed_process = subprocess.run(
            'bazel info bazel-bin'.split(' '), check=True, capture_output=True, text=True)
    bazel_bin_dir = Path(completed_process.stdout.strip())
    artifact_dir = bazel_bin_dir / "experimental/overhead_matching/swag/scripts"
    wheel_name_path = artifact_dir / "train_wheel.name"
    wheel_name = wheel_name_path.read_text()
    wheel_path = artifact_dir / wheel_name

    wheel_s3_uri = upload_wheel_with_sha256(str(wheel_path), bucket)

    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path="s3://rrg-overhead-matching/tensorboard_logs/",
    )

    iam = boto3.client("iam")
    role_name = "rrg-sagemaker-access"
    role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
    estimator = PyTorch(
        dependencies=[],
        source_dir='src',
        entry_point="train.py",
        # The wheel already contains the version of pytorch that is used in the robot repo,
        # so this option is just used to select a docker container
        framework_version='2.6.0',
        py_version="py312",
        instance_count=1,
        # instance_type='ml.g4dn.xlarge',
        instance_type='ml.g5.xlarge',
        output_path="s3://rrg-overhead-matching/models",
        code_location=f"s3://{bucket}/source",
        role=role_arn,
        tensorboard_output_config=tensorboard_config,
        tags={
            "config": train_config_path.stem
        },
        environment={
            "CUSTOM_WHEEL_S3_URI": wheel_s3_uri,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
        }
    )

    print("starting training job:", job_name)
    estimator.fit({
        "train": "s3://rrg-overhead-matching/datasets/VIGOR/Chicago",
        "config_file": config_s3_uri},
        job_name=job_name,
        wait=False)

    print("Started Training Job:", estimator.latest_training_job.name)
    print("Model artifacts will be saved to:", estimator.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', required=True)

    args = parser.parse_args()
    main(Path(args.train_config))
