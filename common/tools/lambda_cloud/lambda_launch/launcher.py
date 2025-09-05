#!/usr/bin/env python3
"""Main launcher for Lambda Cloud training jobs."""

import os
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from common.tools.lambda_cloud.lambda_api.client import LambdaCloudClient

from common.tools.lambda_cloud.lambda_launch.config import ConfigParser, MachineConfig, JobConfig
from common.tools.lambda_cloud.lambda_launch.job_manager import JobManager, JobResult, JobStatus
from common.tools.lambda_cloud.lambda_launch.shutdown_handler import ShutdownHandler
from common.tools.lambda_cloud.lambda_launch.remote_executor import RemoteExecutor


class LambdaTrainingLauncher:
    """Main launcher for Lambda Cloud training jobs."""
    
    def __init__(self, 
                 machine_config_path: str,
                 output_dir: Optional[str] = None,
                 max_parallel_jobs: int = 10):
        """Initialize the launcher.
        
        Args:
            machine_config_path: Path to machine configuration YAML
            output_dir: Output directory for logs and results
            max_parallel_jobs: Maximum number of parallel jobs
        """
        self.machine_config_path = machine_config_path
        self.max_parallel_jobs = max_parallel_jobs
        
        # Parse machine configuration
        self.machine_config = ConfigParser.parse_machine_config(machine_config_path)
        ConfigParser.validate_config(self.machine_config)
        
        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Create output directory in same location as machine config
            config_dir = Path(machine_config_path).parent
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = config_dir / f"launch_{timestamp}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
        # Initialize Lambda Cloud client
        api_key = os.getenv("LAMBDA_API_KEY")
        if not api_key:
            raise ValueError("LAMBDA_API_KEY environment variable not set")
        
        self.lambda_client = LambdaCloudClient(api_key=api_key)
        self.api_key = api_key
    
    def launch_jobs_from_configs(self, 
                               config_paths: List[str],
                               branches: Optional[List[str]] = None) -> List[JobResult]:
        """Launch training jobs from config paths and branches.
        
        Args:
            config_paths: List of training configuration file paths
            branches: List of branches (optional, defaults to main)
            
        Returns:
            List of job results
        """
        # Parse job configurations
        job_configs = ConfigParser.parse_job_configs(config_paths, branches)
        
        print(f"Parsed {len(job_configs)} job configurations:")
        for i, job_config in enumerate(job_configs):
            print(f"  {i+1}. {job_config.config_path} (branch: {job_config.branch})")
        
        return self._execute_jobs(job_configs)
    
    def launch_jobs_from_file(self, job_file_path: str) -> List[JobResult]:
        """Launch training jobs from a job file.
        
        Args:
            job_file_path: Path to file containing job configurations
            
        Returns:
            List of job results
        """
        # Parse job configurations from file
        job_configs = ConfigParser.parse_job_file(job_file_path)
        
        print(f"Parsed {len(job_configs)} job configurations from {job_file_path}:")
        for i, job_config in enumerate(job_configs):
            print(f"  {i+1}. {job_config.config_path} (branch: {job_config.branch})")
        
        return self._execute_jobs(job_configs)
    
    def _execute_jobs(self, job_configs: List[JobConfig]) -> List[JobResult]:
        """Execute the training jobs.
        
        Args:
            job_configs: List of job configurations
            
        Returns:
            List of job results
        """
        # Create job manager
        job_manager = JobManager(
            machine_config=self.machine_config,
            lambda_client=self.lambda_client,
            max_parallel_jobs=self.max_parallel_jobs
        )
        
        # Run jobs
        print(f"\n🚀 Starting {len(job_configs)} training jobs...")
        results = job_manager.run_jobs(job_configs)
        
        # Handle shutdown for each job
        self._handle_shutdowns(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _handle_shutdowns(self, results: List[JobResult]) -> None:
        """Handle shutdown procedures for all jobs.
        
        Args:
            results: List of job results
        """
        print(f"\n🔧 Handling shutdown for {len(results)} jobs...")
        
        for i, result in enumerate(results):
            job_id = f"job_{i:03d}"
            
            if not result.instance_ip or result.status == JobStatus.FAILED:
                print(f"[{job_id}] Skipping shutdown (no instance or failed)")
                continue
            
            try:
                print(f"[{job_id}] Starting shutdown procedure for {result.instance_ip}")
                
                # Connect to instance
                remote_executor = RemoteExecutor(result.instance_ip)
                if not remote_executor.connect(max_retries=3):
                    print(f"[{job_id}] ✗ Failed to connect for shutdown")
                    continue
                
                try:
                    # Setup shutdown handler
                    shutdown_handler = ShutdownHandler(remote_executor, self.api_key)
                    
                    # Setup AWS credentials
                    if not shutdown_handler.setup_aws_credentials():
                        print(f"[{job_id}] ⚠️  AWS credentials setup failed, manual cleanup required")
                        print(f"[{job_id}]   SSH: ssh ubuntu@{result.instance_ip}")
                        print(f"[{job_id}]   Instance ID: {result.instance_id}")
                        continue
                    
                    # Generate S3 paths
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    s3_bucket = "lambda-training-outputs"  # Default bucket
                    s3_key_prefix = f"training_job_{job_id}_{timestamp}"
                    
                    # Sync output to S3
                    remote_output_dir = f"/tmp/output_{job_id}"
                    if shutdown_handler.sync_output_to_s3(remote_output_dir, s3_bucket, s3_key_prefix):
                        result.output_s3_path = f"s3://{s3_bucket}/{s3_key_prefix}/"
                        print(f"[{job_id}] ✓ Output synced to {result.output_s3_path}")
                    
                    # Terminate instance
                    if result.instance_id:
                        if shutdown_handler.terminate_instance(result.instance_id):
                            print(f"[{job_id}] ✓ Instance termination initiated")
                        else:
                            print(f"[{job_id}] ⚠️  Instance termination failed, manual cleanup required")
                    
                finally:
                    remote_executor.disconnect()
                    
            except Exception as e:
                print(f"[{job_id}] ✗ Shutdown error: {e}")
                print(f"[{job_id}]   SSH: ssh ubuntu@{result.instance_ip}")
                print(f"[{job_id}]   Instance ID: {result.instance_id}")
    
    def _save_results(self, results: List[JobResult]) -> None:
        """Save job results to output directory.
        
        Args:
            results: List of job results
        """
        # Create summary report
        summary_path = self.output_dir / "job_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write(f"Lambda Cloud Training Job Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total jobs: {len(results)}\n")
            
            completed_jobs = [r for r in results if r.status == JobStatus.COMPLETED]
            failed_jobs = [r for r in results if r.status == JobStatus.FAILED]
            
            f.write(f"Completed: {len(completed_jobs)}\n")
            f.write(f"Failed: {len(failed_jobs)}\n")
            f.write(f"\n")
            
            f.write("Job Details:\n")
            f.write("=" * 80 + "\n")
            
            for i, result in enumerate(results):
                job_id = f"job_{i:03d}"
                f.write(f"\n{job_id}:\n")
                f.write(f"  Config: {result.job_config.config_path}\n")
                f.write(f"  Branch: {result.job_config.branch}\n")
                f.write(f"  Status: {result.status.value}\n")
                
                if result.instance_id:
                    f.write(f"  Instance ID: {result.instance_id}\n")
                if result.instance_ip:
                    f.write(f"  Instance IP: {result.instance_ip}\n")
                if result.output_s3_path:
                    f.write(f"  Output: {result.output_s3_path}\n")
                if result.error_message:
                    f.write(f"  Error: {result.error_message}\n")
                
                if result.start_time and result.end_time:
                    duration = result.end_time - result.start_time
                    f.write(f"  Duration: {duration:.1f} seconds\n")
        
        print(f"\n📝 Job summary saved to {summary_path}")
        
        # Print final summary
        completed_count = len([r for r in results if r.status == JobStatus.COMPLETED])
        failed_count = len([r for r in results if r.status == JobStatus.FAILED])
        
        print(f"\n🎯 Final Results:")
        print(f"   ✅ Completed: {completed_count}")
        print(f"   ❌ Failed: {failed_count}")
        print(f"   📁 Output directory: {self.output_dir}")
        
        if failed_count > 0:
            print(f"\n⚠️  Some jobs failed. Check the summary file for details.")
            print(f"   For debugging, you may need to manually clean up failed instances.")
        
        return results