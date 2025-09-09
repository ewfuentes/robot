#!/usr/bin/env python3
"""Remote monitoring script that runs on Lambda Cloud instances to handle autonomous job lifecycle."""

import os
import sys
import time
import subprocess
import argparse
import json
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
                 instance_ip: str):
        """Initialize remote monitor.
        
        Args:
            training_command: Command to start training
            max_train_hours: Maximum training time in hours  
            output_dir: Local output directory to sync
            s3_bucket: S3 bucket for output sync
            s3_key_prefix: S3 key prefix
            api_key: Lambda Cloud API key
            instance_ip: Instance IP for self-lookup
        """
        self.training_command = training_command
        self.max_train_hours = max_train_hours
        self.output_dir = output_dir
        self.s3_bucket = s3_bucket
        self.s3_key_prefix = s3_key_prefix
        self.api_key = api_key
        self.instance_ip = instance_ip
        
        self.start_time = datetime.now()
        self.timeout_time = self.start_time + timedelta(hours=max_train_hours)
        self.training_process = None
        
        # Set up log file paths
        self.training_log_file = "/tmp/training.log"
        self.monitor_log_file = "/tmp/monitor.log"
        
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
        """Start the training process with proper logging."""
        try:
            self.log(f"Starting training: {self.training_command}")
            
            # Create training log file
            with open(self.training_log_file, 'w') as f:
                f.write(f"=== Training Started ===\n")
                f.write(f"Command: {self.training_command}\n")
                f.write(f"Start time: {self.start_time}\n")
                f.write(f"========================\n\n")
            
            # Start training with output redirected to log file
            log_command = f"({self.training_command}) 2>&1 | tee -a {self.training_log_file}"
            
            self.training_process = subprocess.Popen(
                log_command,
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
    
    def is_timeout_reached(self) -> bool:
        """Check if training timeout has been reached."""
        return datetime.now() >= self.timeout_time
    
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
    
    def get_instance_id(self) -> str:
        """Get instance ID by IP address using Lambda Cloud API."""
        try:
            client = LambdaCloudClient(api_key=self.api_key)
            instances = client.list_instances()
            
            for instance in instances:
                if instance.ip == self.instance_ip:
                    return instance.id
            
            raise RuntimeError(f"No instance found with IP {self.instance_ip}")
            
        except Exception as e:
            self.log(f"Failed to get instance ID: {e}")
            raise
    
    def terminate_instance(self) -> bool:
        """Terminate this Lambda Cloud instance."""
        try:
            instance_id = self.get_instance_id()
            self.log(f"Terminating instance {instance_id}...")
            
            # Use bazel to terminate via Lambda Cloud API
            terminate_command = [
                "bazel", "run", 
                "//common/tools/lambda_cloud/lambda_api:cli", 
                "--", "terminate", "--force", instance_id
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
                self.log(f"✓ Successfully initiated termination of instance {instance_id}")
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
        # self.terminate_instance()
        
        self.log("Shutdown sequence completed")
    
    def monitor_loop(self):
        """Main monitoring loop."""
        self.log(f"Starting monitoring loop (timeout: {self.max_train_hours}h)")
        self.log(f"Training will timeout at: {self.timeout_time}")
        
        while True:
            # Check if timeout reached
            if self.is_timeout_reached():
                self.cleanup_and_shutdown(f"Timeout reached ({self.max_train_hours}h)")
                break
            
            # Check if training completed
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
    parser.add_argument("--instance-ip", required=True, help="Instance IP address")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = RemoteMonitor(
        training_command=args.training_command,
        max_train_hours=args.max_train_hours,
        output_dir=args.output_dir,
        s3_bucket=args.s3_bucket,
        s3_key_prefix=args.s3_key_prefix,
        api_key=args.api_key,
        instance_ip=args.instance_ip
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