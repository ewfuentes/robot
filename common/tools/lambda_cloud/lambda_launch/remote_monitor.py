#!/usr/bin/env python3
"""Remote monitoring script that runs on Lambda Cloud instances to handle autonomous job lifecycle."""

import os
import sys
import time
import subprocess
import argparse
import json
import psutil
import glob
from datetime import datetime, timedelta
from common.tools.lambda_cloud.lambda_api.client import LambdaCloudClient
from pathlib import Path


class RemoteMonitor:
    """Monitor training job and handle autonomous shutdown."""
    
    def __init__(self, 
                 training_command: str,
                 max_train_hours: float,
                 output_dir: str,
                 s3_bucket: str,
                 s3_key_prefix: str,
                 api_key: str,
                 instance_id: str):
        """Initialize remote monitor.
        
        Args:
            training_command: Command to start training
            max_train_hours: Maximum training time in hours  
            output_dir: Local output directory to sync
            s3_bucket: S3 bucket for output sync
            s3_key_prefix: S3 key prefix
            api_key: Lambda Cloud API key
            instance_id: Lambda Cloud instance ID
        """
        self.training_command = training_command
        self.max_train_hours = max_train_hours
        self.output_dir = output_dir
        self.s3_bucket = s3_bucket
        self.s3_key_prefix = s3_key_prefix
        self.api_key = api_key
        self.instance_id = instance_id
        
        self.start_time = datetime.now()
        self.timeout_time = self.start_time + timedelta(hours=max_train_hours)
        self.training_process = None
        
        # Set up log file paths
        self.training_debug_log = "/tmp/training_debug.log"  # Reliable file-written log
        self.training_output_log = "/tmp/training_output.log"  # Captured stdout (may buffer)
        self.monitor_log_file = "/tmp/monitor.log"
        self.diagnostics_log_file = "/tmp/diagnostics.log"

        # Initialize monitor log file
        self._setup_monitor_logging()
        
    def _setup_monitor_logging(self):
        """Setup monitor log file for tail monitoring."""
        try:
            with open(self.monitor_log_file, 'w') as f:
                f.write(f"=== Training Job Monitor Started ===\n")
                f.write(f"Start time: {self.start_time}\n")
                f.write(f"Timeout: {self.timeout_time}\n")
                f.write(f"Training debug log: {self.training_debug_log}\n")
                f.write(f"Training output log: {self.training_output_log}\n")
                f.write(f"Output dir: {self.output_dir}\n")
                f.write(f"S3 destination: s3://{self.s3_bucket}/{self.s3_key_prefix}/\n")
                f.write(f"=====================================\n\n")
        except Exception as e:
            print(f"Warning: Failed to setup monitor log: {e}")
        
    def log(self, message: str):
        """Log message with timestamp to both stdout and monitor log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        
        # Print to stdout
        print(log_line, flush=True)
        
        # Write to monitor log file
        try:
            with open(self.monitor_log_file, 'a') as f:
                f.write(log_line + '\n')
                f.flush()
        except Exception as e:
            print(f"Warning: Failed to write to monitor log: {e}")
        
    def start_training(self) -> bool:
        """Start the training process with robust logging fixes."""
        try:
            self.log(f"Starting training: {self.training_command}")

            # Create training output log file (captured stdout)
            with open(self.training_output_log, 'w') as f:
                f.write(f"=== Training Started ===\n")
                f.write(f"Command: {self.training_command}\n")
                f.write(f"Start time: {self.start_time}\n")
                f.write(f"========================\n\n")

            # Apply multiple fix strategies for stdout buffering issues

            # Strategy 1: Handle bazel run commands (no python flag needed)
            fixed_command = self.training_command

            # Strategy 2: Create robust wrapper script for bazel commands
            wrapper_script = f'''#!/bin/bash
set -euo pipefail

LOG_FILE="{self.training_output_log}"

echo "🚀 Robust training wrapper starting at $(date)" | tee -a "$LOG_FILE"
echo "Command: {fixed_command}" | tee -a "$LOG_FILE"

# Function to cleanup on exit
cleanup() {{
    echo "⚠️  Wrapper script cleanup at $(date)" | tee -a "$LOG_FILE"
}}
trap cleanup EXIT

# For bazel commands, we need to:
# 1. Force line buffering with stdbuf
# 2. Set PYTHONUNBUFFERED for any Python processes bazel spawns
# 3. Use script to force PTY allocation which often fixes buffering issues

export PYTHONUNBUFFERED=1

# Execute with buffering fixes specific to bazel:
# Option 1: Use stdbuf with explicit line buffering
# Option 2: Use script to create pseudo-TTY which often resolves bazel output issues
if command -v script >/dev/null 2>&1; then
    # Use script for PTY allocation - often works better with bazel
    script -q -c "{fixed_command}" /dev/null 2>&1 | stdbuf -o0 -e0 tee -a "$LOG_FILE"
else
    # Fallback to stdbuf only
    stdbuf -o0 -e0 {fixed_command} 2>&1 | stdbuf -o0 -e0 tee -a "$LOG_FILE"
fi
'''

            wrapper_path = "/tmp/training_wrapper.sh"
            with open(wrapper_path, 'w') as f:
                f.write(wrapper_script)
            os.chmod(wrapper_path, 0o755)

            # Use the wrapper script
            final_command = f"bash {wrapper_path}"

            self.training_process = subprocess.Popen(
                final_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=os.setsid  # Create new process group for easier cleanup
            )

            self.log(f"Training started with PID {self.training_process.pid}")
            self.log(f"Training debug log: tail -f {self.training_debug_log}")
            self.log(f"Training output log: tail -f {self.training_output_log}")
            self.log(f"Monitor logs: tail -f {self.monitor_log_file}")
            self.log(f"Wrapper script: {wrapper_path}")
            self.log("🔧 Applied robust stdout buffering fixes")

            return True

        except Exception as e:
            self.log(f"Failed to start training: {e}")
            return False
    
    def is_training_running(self) -> bool:
        """Check if training process is still running."""
        if not self.training_process:
            return False

        poll_result = self.training_process.poll()
        return poll_result is None

    def is_training_progressing(self) -> bool:
        """Check if training debug log has been modified in the last 15 minutes.

        Returns:
            True if log was modified recently, False otherwise
        """
        if not self.is_training_running():
            return False

        if not os.path.exists(self.training_debug_log):
            self.log(f"Training debug log not found: {self.training_debug_log}")
            return False

        try:
            current_time = datetime.now()
            log_mtime = datetime.fromtimestamp(os.path.getmtime(self.training_debug_log))
            seconds_since_update = (current_time - log_mtime).total_seconds()

            # Consider active if log was modified in last 15 minutes
            is_active = seconds_since_update < 900

            if is_active:
                self.log(f"Training debug log last modified {seconds_since_update:.0f}s ago: {log_mtime}")
            else:
                self.log(f"Training debug log stalled - last modified {seconds_since_update:.0f}s ago: {log_mtime}")

            return is_active
        except Exception as e:
            self.log(f"Error checking training debug log mtime: {e}")
            return False
    
    def is_timeout_reached(self) -> bool:
        """Check if training timeout has been reached."""
        return datetime.now() >= self.timeout_time

    def get_cpu_usage(self) -> float:
        """Get CPU usage for training process with proper warmup."""
        try:
            if not self.training_process:
                return 0.0
            main_process = psutil.Process(self.training_process.pid)
            # Warmup call then measure
            main_process.cpu_percent()
            time.sleep(0.1)
            return main_process.cpu_percent(interval=1)
        except:
            return 0.0

    def log_system_stats(self):
        """Log CPU and GPU usage for monitoring purposes."""
        # Log CPU usage
        cpu_percent = self.get_cpu_usage()
        self.log(f"CPU usage: {cpu_percent:.1f}%")

        # Log GPU usage
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_usage = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip().isdigit():
                        gpu_usage.append(int(line.strip()))

                if gpu_usage:
                    self.log(f"GPU usage: {gpu_usage}")
                else:
                    self.log("GPU usage: no data")
            else:
                self.log("GPU usage: nvidia-smi failed")
        except Exception as e:
            self.log(f"GPU usage: could not query ({e})")

    def sync_to_s3(self) -> bool:
        """Sync output directory and logs to S3."""
        success = True

        try:
            # 1. Sync output directory
            if os.path.exists(self.output_dir):
                self.log(f"Syncing outputs to S3...")
                result = subprocess.run(
                    ["aws", "s3", "sync", self.output_dir,
                     f"s3://{self.s3_bucket}/{self.s3_key_prefix}/outputs/", "--delete"],
                    capture_output=True, text=True, timeout=1800
                )
                if result.returncode != 0:
                    self.log(f"✗ Output sync failed: {result.stderr}")
                    success = False

            # 2. Sync all log files
            log_files = [
                (self.training_debug_log, "training_debug.log"),
                (self.training_output_log, "training_output.log"),
                (self.monitor_log_file, "monitor.log")
            ]

            self.log("Uploading log files to S3...")
            for local_path, s3_name in log_files:
                if os.path.exists(local_path):
                    result = subprocess.run(
                        ["aws", "s3", "cp", local_path,
                         f"s3://{self.s3_bucket}/{self.s3_key_prefix}/logs/{s3_name}"],
                        capture_output=True, text=True, timeout=300
                    )
                    if result.returncode != 0:
                        self.log(f"✗ Failed to upload {s3_name}: {result.stderr}")
                        success = False

            if success:
                self.log(f"✅ All files synced to s3://{self.s3_bucket}/{self.s3_key_prefix}/")
            
            return success
                
        except subprocess.TimeoutExpired:
            self.log("✗ S3 sync timed out after 30 minutes")
            return False
        except Exception as e:
            self.log(f"✗ S3 sync error: {e}")
            return False
    
    
    def terminate_instance(self) -> bool:
        """Terminate this Lambda Cloud instance."""
        try:
            self.log(f"Terminating instance {self.instance_id}...")
            
            # Use bazel to terminate via Lambda Cloud API
            terminate_command = [
                "bazel", "run", 
                "//common/tools/lambda_cloud/lambda_api:cli", 
                "--", "terminate", "--force", self.instance_id
            ]
            
            # Set environment for bazel
            env = os.environ.copy()
            env['LAMBDA_API_KEY'] = self.api_key
            
            result = subprocess.run(
                terminate_command,
                cwd="/home/ubuntu/robot",
                env=env,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0 and "Successfully terminated" in result.stdout:
                self.log(f"✓ Successfully initiated termination of instance {self.instance_id}")
                return True
            else:
                self.log(f"✗ Failed to terminate instance: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"✗ Failed to terminate instance: {e}")
            return False
    
    def cleanup_and_shutdown(self, reason: str):
        """Perform cleanup and shutdown sequence."""
        self.log(f"Starting shutdown sequence: {reason}")
        
        # Stop training if still running
        if self.training_process and self.is_training_running():
            self.log("Stopping training process...")
            try:
                self.training_process.terminate()
                self.training_process.wait(timeout=30)
                self.log("Training process stopped")
            except subprocess.TimeoutExpired:
                self.log("Training process didn't stop gracefully, killing...")
                self.training_process.kill()
        
        self.sync_to_s3()
        
        # Terminate instance
        self.terminate_instance()
        
        self.log("Shutdown sequence completed")

    def monitor_loop(self):
        """Main monitoring loop.

        Exits on one of three conditions:
        1. Training process exits
        2. Timeout is reached
        3. No log file activity for 15 minutes
        """
        self.log(f"Starting monitoring loop (timeout: {self.max_train_hours}h)")
        self.log(f"Training will timeout at: {self.timeout_time}")

        while True:
            # Check 1: Timeout reached
            if self.is_timeout_reached():
                self.log(f"✗ Timeout reached: {self.max_train_hours}h")
                self.cleanup_and_shutdown(f"Timeout reached ({self.max_train_hours}h)")
                break

            # Check 2: Training process exited
            if not self.is_training_running():
                if self.training_process:
                    exit_code = self.training_process.returncode
                    if exit_code == 0:
                        self.log(f"✓ Training process exited successfully (exit code: {exit_code})")
                        self.cleanup_and_shutdown("Training completed successfully")
                    else:
                        self.log(f"✗ Training process failed (exit code: {exit_code})")
                        self.cleanup_and_shutdown(f"Training failed with exit code {exit_code}")
                else:
                    self.log("✗ Training process not found")
                    self.cleanup_and_shutdown("Training process not found")
                break

            # Check 3: No log activity for 15 minutes
            if not self.is_training_progressing():
                self.log("✗ No log file activity for 15 minutes - assuming training stalled")
                self.cleanup_and_shutdown("Training stalled - no log activity for 15 minutes")
                break

            # Log status and system stats
            elapsed = datetime.now() - self.start_time
            remaining = self.timeout_time - datetime.now()
            self.log(f"✓ Training running... Elapsed: {elapsed}, Remaining: {remaining}")
            self.log_system_stats()

            # Sleep before next check
            time.sleep(60)  # Check every minute


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Remote training job monitor")
    parser.add_argument("--training-command", required=True, help="Training command to execute")
    parser.add_argument("--max-train-hours", type=float, required=True, help="Maximum training time in hours")
    parser.add_argument("--output-dir", required=True, help="Output directory to sync to S3")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket for output sync")
    parser.add_argument("--s3-key-prefix", required=True, help="S3 key prefix")
    parser.add_argument("--api-key", required=True, help="Lambda Cloud API key")
    parser.add_argument("--instance-id", required=True, help="Lambda Cloud instance ID")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = RemoteMonitor(
        training_command=args.training_command,
        max_train_hours=args.max_train_hours,
        output_dir=args.output_dir,
        s3_bucket=args.s3_bucket,
        s3_key_prefix=args.s3_key_prefix,
        api_key=args.api_key,
        instance_id=args.instance_id
    )
    
    # Start training
    if not monitor.start_training():
        monitor.log("Failed to start training, exiting")
        sys.exit(1)
    
    # Start monitoring
    try:
        monitor.monitor_loop()
    except KeyboardInterrupt:
        monitor.log("Monitoring interrupted by user")
        monitor.cleanup_and_shutdown("User interrupt")
    except Exception as e:
        monitor.log(f"Monitoring failed: {e}")
        monitor.cleanup_and_shutdown(f"Monitoring error: {e}")
    
    monitor.log("Remote monitor exiting")


if __name__ == "__main__":
    main()