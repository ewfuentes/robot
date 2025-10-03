# Lambda Cloud Training Job Launcher

A tool for launching parallel training jobs on Lambda Cloud instances with automatic setup, monitoring, and cleanup.

## Features

- **Parallel Execution**: Launch multiple training jobs simultaneously with configurable parallelism
- **Instance Management**: Automatic instance launching, setup, and termination
- **Branch Support**: Run jobs on different Git branches
- **File Transfer**: Copy datasets, configs, and dependencies to remote instances
- **Remote Setup**: Execute custom setup commands on each instance
- **Training Monitoring**: Monitor training progress and handle timeouts
- **S3 Integration**: Automatically sync training outputs to S3
- **Error Handling**: Comprehensive error reporting and debugging information
- **Cleanup**: Automatic instance termination and resource cleanup

## Directory Structure

```
common/tools/lambda_cloud/
├── lambda_launch/           # Main implementation
│   ├── BUILD               # Bazel build configuration
│   ├── launch_training_jobs.py  # CLI entry point
│   ├── launcher.py         # Main launcher class
│   ├── config.py          # Configuration parsing
│   ├── job_manager.py     # Parallel job execution
│   ├── remote_executor.py # SSH and remote commands
│   ├── remote_monitor.py # Run and monitor the training job on remote
│   └── shutdown_handler.py # S3 sync and cleanup
├── example_machine_config.yaml  # Example machine configuration
├── example_jobs.txt             # Example job file
└── README.md                   # This file
```

### 1. Environment Setup

Set required environment variables:

```bash
export LAMBDA_API_KEY="your_lambda_api_key_here"
```
Make sure your AWS credentials are configured and you can access your target s3 buckets on your host machine.

This tool assumes aws CLI is installed on the remote. 

## Configuration

### Machine Configuration

Create a YAML file specifying your machine setup (see `example_machine_config.yaml`):

```yaml
machine_types:
  - gpu_1x_h100_pcie
  - gpu_1x_a100
ssh_key: your_ssh_key_name
files_to_copy:
  /home/user/.cache/torch/hub/checkpoints: /home/ubuntu/.cache/torch/hub/checkpoints
  ~/.tmux.conf: ~/.tmux.conf
remote_setup_commands:
  - cp -r vigor/Chicago /tmp/
  - cp -r vigor/Seattle /tmp/
  - git clone https://github.com/your-org/robot.git && cd robot && ./setup.sh
max_train_time_hours: 43
```

### Job Configurations

**Option 1: Command Line Arguments**
```bash
bazel run //common/tools/lambda_cloud/lambda_launch:launch_training_jobs -- \\
  --config configs/train1.yaml configs/train2.yaml \\
  --branches main experiment \\
  --machine-config machine_setup.yaml
```

**Option 2: Job File**
Create a CSV file with job configurations (see `example_jobs.txt`):
```
configs/baseline_config.yaml
configs/experiment1_config.yaml,feature/new-loss
configs/experiment2_config.yaml,experiment/augmentation
```

Then run:
```bash
bazel run //common/tools/lambda_cloud/lambda_launch:launch_training_jobs -- \\
  --job-file jobs.txt \\
  --machine-config machine_setup.yaml
```

## Usage Examples

### Basic Usage

Launch a single training job:
```bash
bazel run //common/tools/lambda_cloud/lambda_launch:launch_training_jobs -- \\
  --config configs/train.yaml \\
  --machine-config setup.yaml
```

### Multiple Jobs with Different Branches

Launch multiple jobs on different branches:
```bash
bazel run //common/tools/lambda_cloud/lambda_launch:launch_training_jobs -- \\
  --config config1.yaml config2.yaml config3.yaml \\
  --branches main experiment ablation \\
  --machine-config setup.yaml
```

### Custom Output Directory and Parallelism

Set custom output directory and limit parallelism:
```bash
bazel run //common/tools/lambda_cloud/lambda_launch:launch_training_jobs -- \\
  --config configs/train.yaml \\
  --machine-config setup.yaml \\
  --output-dir ./results \\
  --max-parallel 5
```

### Dry Run Mode

Test configuration without launching instances:
```bash
bazel run //common/tools/lambda_cloud/lambda_launch:launch_training_jobs -- \\
  --config configs/train.yaml \\
  --machine-config setup.yaml \\
  --dry-run
```

### Setup-Only Mode

Launch instances and complete all setup without starting training (useful for debugging or manual training):
```bash
bazel run //common/tools/lambda_cloud/lambda_launch:launch_training_jobs -- \\
  --config configs/train.yaml \\
  --machine-config setup.yaml \\
  --setup-only
```


## Output

The tool creates an output directory containing:
- `job_summary.txt`: Detailed summary of all jobs
- Individual job logs and status information

Training outputs and logs are automatically synced to S3 with paths like:
`s3://rrg-overhead-matching/training_outputs/<job id>/`
