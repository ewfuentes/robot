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
        self.training_log_file = "/tmp/training.log"
        self.monitor_log_file = "/tmp/monitor.log"
        self.diagnostics_log_file = "/tmp/diagnostics.log"

        # Track log monitoring
        self.last_log_size = 0
        self.last_log_activity = datetime.now()
        self.stalled_log_threshold = timedelta(minutes=10)  # Alert if no log activity for 10 min

        # Initialize monitor log file
        self._setup_monitor_logging()
        
    def _setup_monitor_logging(self):
        """Setup monitor log file for tail monitoring."""
        try:
            with open(self.monitor_log_file, 'w') as f:
                f.write(f"=== Training Job Monitor Started ===\n")
                f.write(f"Start time: {self.start_time}\n")
                f.write(f"Timeout: {self.timeout_time}\n")
                f.write(f"Training log: {self.training_log_file}\n")
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

            # Create training log file
            with open(self.training_log_file, 'w') as f:
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

LOG_FILE="{self.training_log_file}"
HEARTBEAT_FILE="/tmp/training_heartbeat.txt"

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
            self.log(f"Training logs: tail -f {self.training_log_file}")
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
        """Check if training is actually making progress (not just running).

        Uses multiple indicators:
        1. Log file modification time
        2. Output directory activity (checkpoints, wandb files)
        3. Training process activity

        Returns:
            True if training appears to be progressing, False if stuck
        """
        if not self.is_training_running():
            return False

        current_time = datetime.now()

        # Check 1: Training log file modification time
        training_log_activity = False
        if os.path.exists(self.training_log_file):
            try:
                log_mtime = datetime.fromtimestamp(os.path.getmtime(self.training_log_file))
                # Consider active if log was modified in last 10 minutes
                if (current_time - log_mtime).total_seconds() < 600:
                    training_log_activity = True
                    self.log(f"Training log last modified: {log_mtime}")
            except Exception as e:
                self.log(f"Error checking training log mtime: {e}")

        # Check 2: Output directory activity (wandb, checkpoints, etc.)
        output_activity = False
        if os.path.exists(self.output_dir):
            try:

                # Check for checkpoint/model files
                if not output_activity:
                    for root, dirs, files in os.walk(self.output_dir):
                        for file in files:
                            if file.endswith(('.ckpt', '.pt', '.pth', '.pkl')):
                                filepath = os.path.join(root, file)
                                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                                if (current_time - file_mtime).total_seconds() < 600:
                                    output_activity = True
                                    self.log(f"Recent model file activity: {filepath} modified at {file_mtime}")
                                    break
                        if output_activity:
                            break

            except Exception as e:
                self.log(f"Error checking output directory activity: {e}")

        # Return True if we see any sign of progress
        is_progressing = training_log_activity or output_activity

        if not is_progressing:
            self.log(f"No training progress detected (log_activity={training_log_activity}, output_activity={output_activity})")

        return is_progressing
    
    def is_timeout_reached(self) -> bool:
        """Check if training timeout has been reached."""
        return datetime.now() >= self.timeout_time

    def check_log_activity(self) -> bool:
        """Check if training log is still being written to."""
        try:
            if not os.path.exists(self.training_log_file):
                return False

            current_size = os.path.getsize(self.training_log_file)
            if current_size > self.last_log_size:
                self.last_log_size = current_size
                self.last_log_activity = datetime.now()
                return True

            # Check if log has been stalled too long
            time_since_activity = datetime.now() - self.last_log_activity
            if time_since_activity > self.stalled_log_threshold:
                self.log(f"⚠️  Training log stalled for {time_since_activity}")
                return False

            return True

        except Exception as e:
            self.log(f"Error checking log activity: {e}")
            return False

    def get_process_tree_info(self) -> dict:
        """Get detailed info about training process tree."""
        try:
            if not self.training_process:
                return {}

            main_process = psutil.Process(self.training_process.pid)
            children = main_process.children(recursive=True)

            info = {
                "main_pid": self.training_process.pid,
                "main_status": main_process.status(),
                "main_cpu_percent": main_process.cpu_percent(),
                "main_memory_mb": main_process.memory_info().rss // (1024*1024),
                "child_count": len(children),
                "children": []
            }

            for child in children:
                try:
                    child_info = {
                        "pid": child.pid,
                        "name": child.name(),
                        "status": child.status(),
                        "cpu_percent": child.cpu_percent(),
                        "cmdline": " ".join(child.cmdline()[:3])  # First 3 args only
                    }
                    info["children"].append(child_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return info

        except Exception as e:
            self.log(f"Error getting process info: {e}")
            return {}

    def check_tensorboard_activity(self) -> dict:
        """Check for recent tensorboard event files to detect if training is progressing."""
        try:
            # Find tensorboard event files
            event_pattern = os.path.join(self.output_dir, "**/events.out.tfevents.*")
            event_files = glob.glob(event_pattern, recursive=True)

            if not event_files:
                return {"status": "no_events", "message": "No tensorboard event files found"}

            # Check most recent modification time
            most_recent_file = max(event_files, key=os.path.getmtime)
            last_modified = datetime.fromtimestamp(os.path.getmtime(most_recent_file))
            time_since_update = datetime.now() - last_modified

            return {
                "status": "found_events",
                "most_recent_file": most_recent_file,
                "last_modified": last_modified,
                "time_since_update": time_since_update,
                "is_recent": time_since_update < timedelta(minutes=5)
            }

        except Exception as e:
            return {"status": "error", "message": f"Error checking tensorboard: {e}"}

    def diagnose_training_state(self):
        """Comprehensive diagnosis of training state."""
        try:
            with open(self.diagnostics_log_file, 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n=== DIAGNOSTIC REPORT [{timestamp}] ===\n")

                # Process info
                process_info = self.get_process_tree_info()
                f.write(f"Process Info: {process_info}\n")

                # Log activity
                log_active = self.check_log_activity()
                f.write(f"Log Activity: {log_active}, Last activity: {self.last_log_activity}\n")

                # Tensorboard activity
                tb_info = self.check_tensorboard_activity()
                f.write(f"Tensorboard Activity: {tb_info}\n")

                # File descriptors
                if self.training_process:
                    try:
                        main_process = psutil.Process(self.training_process.pid)
                        open_files = len(main_process.open_files())
                        f.write(f"Open file descriptors: {open_files}\n")
                    except:
                        f.write("Could not get file descriptor info\n")

                f.write("=== END DIAGNOSTIC ===\n\n")
                f.flush()

        except Exception as e:
            self.log(f"Error in diagnosis: {e}")

    def detect_training_completion_alternative(self) -> bool:
        """Alternative method to detect training completion with multiple verification steps."""
        completion_indicators = 0
        required_indicators = 2  # Require multiple confirmations

        # Method 1: Check if training log shows explicit completion
        try:
            if os.path.exists(self.training_log_file):
                with open(self.training_log_file, 'r') as f:
                    content = f.read().lower()

                    # Look for definitive completion markers
                    completion_phrases = [
                        "training complete", "training finished", "training done",
                        "100% complete", "epoch: 100/100", "final epoch complete",
                        "training successfully completed", "all epochs finished"
                    ]

                    if any(phrase in content for phrase in completion_phrases):
                        completion_indicators += 1
                        self.log("✓ Found explicit completion marker in training log")
        except Exception as e:
            self.log(f"Could not check training log completion: {e}")

        # Method 2: Check process resource usage (low CPU/GPU indicates completion)
        try:
            if self.training_process:
                import psutil
                main_process = psutil.Process(self.training_process.pid)

                # Check CPU usage over brief period
                cpu_percent = main_process.cpu_percent(interval=5)
                if cpu_percent < 2.0:  # Very low CPU usage
                    completion_indicators += 1
                    self.log(f"✓ Very low CPU usage detected: {cpu_percent:.1f}%")
                else:
                    self.log(f"Process still active - CPU: {cpu_percent:.1f}%")
        except Exception as e:
            self.log(f"Could not check process CPU usage: {e}")

        # Method 3: Check GPU utilization (if available)
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_usage = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip().isdigit():
                        gpu_usage.append(int(line.strip()))

                if gpu_usage and max(gpu_usage) < 5:  # Very low GPU usage
                    completion_indicators += 1
                    self.log(f"✓ Very low GPU usage detected: {max(gpu_usage)}%")
                else:
                    self.log(f"GPU still active: {gpu_usage}")
        except Exception as e:
            self.log(f"Could not check GPU usage: {e}")

        # Method 4: Check if output files haven't changed in very long time (30+ minutes)
        try:
            if os.path.exists(self.output_dir):
                latest_mtime = 0
                for root, dirs, files in os.walk(self.output_dir):
                    for file in files:
                        if file.endswith(('.pt', '.pth', '.ckpt', '.log')):  # Model/checkpoint files
                            filepath = os.path.join(root, file)
                            try:
                                mtime = os.path.getmtime(filepath)
                                latest_mtime = max(latest_mtime, mtime)
                            except:
                                continue

                if latest_mtime > 0:
                    minutes_since_change = (datetime.now().timestamp() - latest_mtime) / 60
                    if minutes_since_change > 30:  # No model saves for 30+ minutes
                        completion_indicators += 1
                        self.log(f"✓ Output files stable for {minutes_since_change:.1f} minutes")
                    else:
                        self.log(f"Output files changed {minutes_since_change:.1f} minutes ago")
        except Exception as e:
            self.log(f"Could not check output file activity: {e}")

        is_complete = completion_indicators >= required_indicators
        self.log(f"Completion verification: {completion_indicators}/{required_indicators + 2} indicators met (need {required_indicators})")

        return is_complete
    
    def sync_to_s3(self) -> bool:
        """Sync output directory and logs to S3."""
        success = True
        
        try:
            # 1. Sync output directory
            if os.path.exists(self.output_dir):
                self.log(f"Syncing output directory {self.output_dir} to s3://{self.s3_bucket}/{self.s3_key_prefix}/outputs/")
                
                sync_command = [
                    "aws", "s3", "sync", 
                    self.output_dir, 
                    f"s3://{self.s3_bucket}/{self.s3_key_prefix}/outputs/",
                    "--delete"
                ]
                
                result = subprocess.run(
                    sync_command,
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minute timeout
                )
                
                if result.returncode == 0:
                    self.log("✓ Successfully synced outputs to S3")
                else:
                    self.log(f"✗ Output S3 sync failed: {result.stderr}")
                    success = False
            else:
                self.log(f"⚠ Output directory {self.output_dir} does not exist, skipping output sync")
            
            # 2. Sync training log
            if os.path.exists(self.training_log_file):
                self.log(f"Uploading training log to s3://{self.s3_bucket}/{self.s3_key_prefix}/logs/training.log")
                
                log_upload_command = [
                    "aws", "s3", "cp",
                    self.training_log_file,
                    f"s3://{self.s3_bucket}/{self.s3_key_prefix}/logs/training.log"
                ]
                
                result = subprocess.run(
                    log_upload_command,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    self.log("✓ Successfully uploaded training log to S3")
                else:
                    self.log(f"✗ Training log upload failed: {result.stderr}")
                    success = False
            
            # 3. Sync monitor log
            if os.path.exists(self.monitor_log_file):
                self.log(f"Uploading monitor log to s3://{self.s3_bucket}/{self.s3_key_prefix}/logs/monitor.log")
                
                monitor_upload_command = [
                    "aws", "s3", "cp",
                    self.monitor_log_file,
                    f"s3://{self.s3_bucket}/{self.s3_key_prefix}/logs/monitor.log"
                ]
                
                result = subprocess.run(
                    monitor_upload_command,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    self.log("✓ Successfully uploaded monitor log to S3")
                else:
                    self.log(f"✗ Monitor log upload failed: {result.stderr}")
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
    
    def check_heartbeat_file(self) -> dict:
        """Check heartbeat file for training completion signals."""
        heartbeat_file = "/tmp/training_heartbeat.txt"
        try:
            if not os.path.exists(heartbeat_file):
                return {"status": "no_heartbeat", "message": "No heartbeat file found"}

            with open(heartbeat_file, 'r') as f:
                content = f.read().strip()

            if "TRAINING_COMPLETE" in content:
                return {"status": "completed", "message": content}
            elif "HEARTBEAT" in content:
                # Extract timestamp to check freshness
                return {"status": "active", "message": content}
            else:
                return {"status": "unknown", "message": content}

        except Exception as e:
            return {"status": "error", "message": f"Error reading heartbeat: {e}"}

    def monitor_loop(self):
        """Main monitoring loop with enhanced detection."""
        self.log(f"Starting monitoring loop (timeout: {self.max_train_hours}h)")
        self.log(f"Training will timeout at: {self.timeout_time}")

        diagnostic_counter = 0

        while True:
            # Check if timeout reached
            if self.is_timeout_reached():
                self.cleanup_and_shutdown(f"Timeout reached ({self.max_train_hours}h)")
                break

            # Check traditional process completion
            if not self.is_training_running():
                if self.training_process:
                    exit_code = self.training_process.returncode
                    if exit_code == 0:
                        self.cleanup_and_shutdown("Training completed successfully")
                    else:
                        self.cleanup_and_shutdown(f"Training failed with exit code {exit_code}")
                else:
                    self.cleanup_and_shutdown("Training process not found")
                break

            # Check heartbeat file for completion signal
            heartbeat_status = self.check_heartbeat_file()
            if heartbeat_status["status"] == "completed":
                self.log(f"🎉 Training completion detected via heartbeat: {heartbeat_status['message']}")
                self.cleanup_and_shutdown("Training completed successfully (detected via heartbeat)")
                break

            # Check for training progress to avoid false completion detection
            progress_status = self.is_training_progressing()
            if not progress_status:
                stall_duration = getattr(self, '_stall_start_time', None)
                if stall_duration is None:
                    self._stall_start_time = datetime.now()
                    self.log("⚠️ Training progress stalled - starting monitoring")
                else:
                    stall_minutes = (datetime.now() - self._stall_start_time).total_seconds() / 60
                    if stall_minutes > 45:  # Only check completion after 45 minutes of no progress
                        self.log(f"🔍 No progress for {stall_minutes:.1f} minutes - checking if training completed")
                        if self.detect_training_completion_alternative():
                            self.log("🔍 Training completion confirmed via multiple indicators")
                            self.cleanup_and_shutdown("Training completed successfully (detected via alternative methods)")
                            break
            else:
                # Reset stall timer if progress is detected
                if hasattr(self, '_stall_start_time'):
                    delattr(self, '_stall_start_time')
                    self.log("✅ Training progress resumed")

            # Check log activity
            log_active = self.check_log_activity()
            if not log_active:
                self.log("⚠️  Training log appears stalled")

            # Periodic diagnostics (every 5 minutes)
            diagnostic_counter += 1
            if diagnostic_counter % 5 == 0:
                self.diagnose_training_state()

            # Log progress
            elapsed = datetime.now() - self.start_time
            remaining = self.timeout_time - datetime.now()
            self.log(f"Training running... Elapsed: {elapsed}, Remaining: {remaining}")

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