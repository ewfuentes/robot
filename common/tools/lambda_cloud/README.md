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
│   └── shutdown_handler.py # S3 sync and cleanup
├── example_machine_config.yaml  # Example machine configuration
├── example_jobs.txt             # Example job file
└── README.md                   # This file
```

## Setup

### 1. Install Dependencies

Add these dependencies to your `requirements.txt` or pip environment:

```
pyyaml>=6.0
paramiko>=2.11.0
boto3>=1.26.0
lambda_cloud>=1.0.0
```

### 2. Dependencies Already Configured

All required dependencies are already configured:

- ✅ **PyYAML** - Available via `requirement("pyyaml")`  
- ✅ **Paramiko** - Available via `requirement("paramiko")`
- ✅ **Boto3** - Available via `requirement("boto3")`
- ✅ **Lambda Cloud API** - Available via `//common/tools/lambda_cloud/lambda_api:lambda_cloud`

The BUILD file is ready to use with all dependencies properly configured.

### 3. Environment Setup

Set required environment variables:

```bash
export LAMBDA_API_KEY="your_lambda_api_key_here"
export AWS_ACCESS_KEY_ID="your_aws_access_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
export AWS_DEFAULT_REGION="us-west-2"
```

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

### Using Job File

Launch jobs from a file:
```bash
bazel run //common/tools/lambda_cloud/lambda_launch:launch_training_jobs -- \\
  --job-file experiments.txt \\
  --machine-config setup.yaml \\
  --verbose
```

## CLI Options

```
usage: launch_training_jobs [-h] (--config CONFIG [CONFIG ...] | --job-file JOB_FILE)
                           [--branches BRANCHES [BRANCHES ...]] --machine-config MACHINE_CONFIG
                           [--output-dir OUTPUT_DIR] [--max-parallel MAX_PARALLEL] [--dry-run]
                           [--verbose]

Launch training jobs on Lambda Cloud instances

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG [CONFIG ...], -c CONFIG [CONFIG ...]
                        Path(s) to training configuration YAML file(s)
  --job-file JOB_FILE, -j JOB_FILE
                        Path to file containing job configurations (CSV format: config_path[,branch])
  --branches BRANCHES [BRANCHES ...], -b BRANCHES [BRANCHES ...]
                        Git branch(es) for each config. Defaults to 'main'. Must match number of
                        configs or be single branch for all
  --machine-config MACHINE_CONFIG, -m MACHINE_CONFIG
                        Path to machine setup configuration YAML file
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Output directory for logs and results (default: auto-generated in machine
                        config directory)
  --max-parallel MAX_PARALLEL, -p MAX_PARALLEL
                        Maximum number of parallel jobs (default: 10)
  --dry-run             Parse configurations and validate but don't launch instances
  --verbose, -v         Enable verbose output
```

## Workflow

1. **Validation**: Parse and validate all configurations
2. **Instance Launch**: Launch Lambda Cloud instances (trying machine types in order)
3. **Setup**: Connect via SSH, copy files, run setup commands
4. **Training**: Start tmux session and launch training job
5. **Monitoring**: Monitor training progress and handle timeouts
6. **Shutdown**: Sync outputs to S3 and terminate instances
7. **Reporting**: Generate summary report with results and S3 locations

## Output

The tool creates an output directory containing:
- `job_summary.txt`: Detailed summary of all jobs
- Individual job logs and status information
- SSH connection information for debugging failed jobs

Training outputs are automatically synced to S3 with paths like:
`s3://lambda-training-outputs/training_job_001_20240315_143022/`

## Error Handling

If jobs fail, the tool provides:
- Detailed error messages
- SSH connection information for manual debugging
- Instance IDs for manual cleanup if needed
- S3 paths for any partially synced outputs

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Ensure all pip dependencies are installed and BUILD file is updated
2. **SSH Key Issues**: Verify SSH key is registered with Lambda Cloud and accessible
3. **AWS Credentials**: Check AWS environment variables or IAM role permissions
4. **Instance Launch Failures**: Check Lambda Cloud quota and instance availability
5. **Setup Command Failures**: Review machine configuration and ensure commands are valid

### Manual Cleanup

If instances need manual cleanup:
```bash
# Connect to instance
ssh ubuntu@<instance_ip>

# Check running processes
ps aux | grep python

# Check tmux sessions  
tmux list-sessions

# Terminate instance via Lambda Cloud dashboard or API
```

## Development

To extend or modify the tool:

1. **Add new functionality**: Extend the appropriate class in `lambda_launch/`
2. **Test changes**: Use `--dry-run` mode for configuration testing
3. **Update dependencies**: Add new requirements to BUILD file
4. **Test builds**: `bazel build //common/tools/lambda_cloud/lambda_launch:launch_training_jobs`

## Security Notes

- Lambda API keys and AWS credentials are handled securely
- SSH connections use key-based authentication
- Temporary credentials are used on remote instances
- Instance termination is automated to prevent runaway costs