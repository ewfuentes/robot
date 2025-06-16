Launch Training Jobs on AWS
===========================

AWS Setup
---------
1. Get added to the AWS group, ask Erick for details.
2. Sign into the AWS console and on the IAM page, create a new user.
3. Add that user to the `overhead-matching` group.
4. Download the latest version of the aws cli by running:
```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```
5. Create an access key for your user and enter the required details in `aws configure`.
6. If everything works, you should be able to run `aws s3 ls` and see the `rrg-overhead-matching` bucket.

Tooling Setup
-------------
1. Install (uv)[https://docs.astral.sh/uv/#highlights].

Launching a  Job
----------------
To launch a training job, run `uv run launch_training_job.py --train_config <Path to Train Config>`.

The training job will be visible on the Sagemaker AI Dashboard.

Implementation Details
----------------------

When you launch a training job, all sources in the `src` directory are uploaded to the
`s3://sagemaker-us-east-2-390402551597/source/<job_name>` folder on S3. In addition,
the `//experimental/overhead_matching/swag/scripts:train_wheel` target is built and uploaded to
`s3://sagemaker-us-east-2-390402551597/wheels/<sha256 hash>/` folder on S3. Finally, the train config is uploaded to
`s3://sagemaker-us-east-2-390402551597/train_configs/<train_config>_<job_name>.yaml`.

The VIGOR dataset lives at `s3://rrg-overhead-matching/datasets/VIGOR`. The model output for a training job is at
`s3://rrg-overhead-matching/models/<job_name>/output/model.tar.gz`.

