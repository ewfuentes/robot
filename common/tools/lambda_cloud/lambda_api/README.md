# Lambda Cloud API Library

A Python library and CLI for interacting with the Lambda Cloud API to manage GPU instances.

## Setup

Set your Lambda Cloud API key:
```bash
export LAMBDA_API_KEY="your-api-key-here"
```

## CLI Usage

### List Available Instance Types
```bash
bazel run //common/tools/lambda_cloud/lambda_api:cli -- list-types
bazel run //common/tools/lambda_cloud/lambda_api:cli -- list-types --region us-east-3
bazel run //common/tools/lambda_cloud/lambda_api:cli -- list-types --gpu-type gpu_1x_gh200
bazel run //common/tools/lambda_cloud/lambda_api:cli -- list-types --json
```

### List Running Instances (Status)
```bash
bazel run //common/tools/lambda_cloud/lambda_api:cli -- list-instances
bazel run //common/tools/lambda_cloud/lambda_api:cli -- list-instances --json
```

### Launch Instances
```bash
bazel run //common/tools/lambda_cloud/lambda_api:cli -- launch \
  --region us-east-3 \
  --type gpu_1x_gh200 \
  --ssh-keys "ssh key name" \
  --file-systems "vigor" \
  --name "My Instance"
  --wait \
  --max-retries 15
```

### Get SSH Information
```bash
bazel run //common/tools/lambda_cloud/lambda_api:cli -- ssh <instance-id>
bazel run //common/tools/lambda_cloud/lambda_api:cli -- ssh <instance-id> --json
```

### Terminate Instances
```bash
bazel run //common/tools/lambda_cloud/lambda_api:cli -- terminate <instance-id>
```

## Programmatic Usage

```python
from lambda_cloud import LambdaCloudClient

# Initialize client (reads LAMBDA_API_KEY from environment)
client = LambdaCloudClient()

# List available instance types
instance_types = client.list_instance_types()
for it in instance_types:
    print(f"{it.name}: ${it.price_cents_per_hour/100:.2f}/hour")

# List running instances
instances = client.list_instances()
for instance in instances:
    print(f"{instance.name} ({instance.id}): {instance.status.value}")

# Launch an instance with intelligent retry
instance_ids = client.launch_instance_with_retry(
    region_name="us-west-1",
    instance_type_name="gpu_1x_a10",
    ssh_key_names=["my-key"],
    name="Test Instance"
)

# Get SSH info
ssh_info = client.get_ssh_info(instance_ids[0])
print(f"ssh -i <key> {ssh_info.username}@{ssh_info.ip}")

# Terminate instances
terminated = client.terminate_instances(instance_ids)
```

## Features

- **Intelligent Launch**: Automatically retries launches when capacity is unavailable, with helpful feedback about available alternatives
- **Rate Limiting**: Automatically handles API rate limits (1 req/sec general, 1 req/12sec for launches)
- **Error Handling**: Comprehensive error handling with specific exception types for different failure modes
- **SSH Integration**: Provides all information needed to SSH into instances
- **Filtering**: Supports filtering instance types by region and GPU type
- **JSON Output**: All CLI commands support JSON output for scripting
- **Timeout Handling**: Robust timeout and retry logic for network issues

## Error Handling

The library provides specific exception types:
- `LambdaCloudError`: Base exception for all API errors
- `InsufficientCapacityError`: Raised when requested resources aren't available
- `RateLimitError`: Raised when API rate limits are exceeded

The intelligent launch feature automatically handles capacity errors by:
1. Retrying with exponential backoff
2. Providing feedback on available instance types in the region
3. Suggesting alternative regions when the requested type isn't available