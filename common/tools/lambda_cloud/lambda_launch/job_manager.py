#!/usr/bin/env python3
"""Job management and parallel execution for Lambda Cloud training jobs."""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from common.tools.lambda_cloud.lambda_launch.config import JobConfig, MachineConfig
from common.tools.lambda_cloud.lambda_launch.remote_executor import RemoteExecutor


class JobStatus(Enum):
    """Status of a training job."""
    PENDING = "pending"
    LAUNCHING = "launching"
    SETTING_UP = "setting_up"
    TRAINING = "training"
    SHUTTING_DOWN = "shutting_down"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobResult:
    """Result of a training job execution."""
    job_config: JobConfig
    status: JobStatus
    instance_id: Optional[str] = None
    instance_ip: Optional[str] = None
    error_message: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    output_s3_path: Optional[str] = None


class JobManager:
    """Manage parallel execution of Lambda Cloud training jobs."""
    
    def __init__(self, 
                 machine_config: MachineConfig,
                 lambda_client,
                 max_parallel_jobs: int = 10):
        """Initialize job manager.
        
        Args:
            machine_config: Machine configuration for all jobs
            lambda_client: Lambda Cloud client instance
            max_parallel_jobs: Maximum number of parallel jobs
        """
        self.machine_config = machine_config
        self.lambda_client = lambda_client
        self.max_parallel_jobs = max_parallel_jobs
        self.active_jobs: Dict[str, JobResult] = {}
        self.completed_jobs: List[JobResult] = []
        self._lock = threading.Lock()
    
    def launch_instance(self, machine_type: str, job_id: str) -> Optional[tuple]:
        """Launch a Lambda Cloud instance.
        
        Args:
            machine_type: Type of machine to launch
            job_id: Unique job identifier
            
        Returns:
            (instance_id, instance_ip) if successful, None otherwise
        """
        try:
            print(f"[{job_id}] Launching {machine_type} instance...")
            
            # Launch instance using Lambda Cloud API
            instance = self.lambda_client.launch_instance(
                instance_type_name=machine_type,
                ssh_key_names=[self.machine_config.ssh_key],
                quantity=1
            )
            
            if not instance or len(instance.instance_ids) == 0:
                print(f"[{job_id}] ✗ Failed to launch instance")
                return None
            
            instance_id = instance.instance_ids[0]
            print(f"[{job_id}] ✓ Launched instance {instance_id}")
            
            # Wait for instance to be ready and get IP
            max_wait_time = 300  # 5 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                instances = self.lambda_client.list_instances()
                for inst in instances:
                    if inst.id == instance_id and inst.status.value == "active" and inst.ip:
                        print(f"[{job_id}] ✓ Instance ready at {inst.ip}")
                        return instance_id, inst.ip
                
                time.sleep(10)
            
            print(f"[{job_id}] ✗ Instance failed to become ready within {max_wait_time}s")
            return None
            
        except Exception as e:
            print(f"[{job_id}] ✗ Error launching instance: {e}")
            return None
    
    def setup_instance(self, 
                      remote_executor: RemoteExecutor, 
                      job_config: JobConfig, 
                      job_id: str) -> bool:
        """Setup the remote instance for training.
        
        Args:
            remote_executor: Connected remote executor
            job_config: Job configuration
            job_id: Unique job identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"[{job_id}] Setting up instance...")
            
            # Copy files to remote instance
            print(f"[{job_id}] Copying files...")
            for local_path, remote_path in self.machine_config.files_to_copy.items():
                if not remote_executor.copy_file(local_path, remote_path):
                    return False
            
            # Checkout specified branch
            if job_config.branch != "main":
                print(f"[{job_id}] Checking out branch {job_config.branch}...")
                result = remote_executor.execute_command(f"cd robot && git checkout {job_config.branch}")
                if not result.success:
                    print(f"[{job_id}] ✗ Failed to checkout branch: {result.stderr}")
                    return False
            
            # Run setup commands
            print(f"[{job_id}] Running setup commands...")
            for command in self.machine_config.remote_setup_commands:
                print(f"[{job_id}]   Running: {command}")
                result = remote_executor.execute_command(command, timeout=600)  # 10 minute timeout
                if not result.success:
                    print(f"[{job_id}] ✗ Setup command failed: {command}")
                    print(f"[{job_id}]   Error: {result.stderr}")
                    return False
            
            print(f"[{job_id}] ✓ Instance setup completed")
            return True
            
        except Exception as e:
            print(f"[{job_id}] ✗ Setup error: {e}")
            return False
    
    def start_training(self,
                      remote_executor: RemoteExecutor,
                      job_config: JobConfig,
                      job_id: str) -> bool:
        """Start the training job on the remote instance.
        
        Args:
            remote_executor: Connected remote executor
            job_config: Job configuration  
            job_id: Unique job identifier
            
        Returns:
            True if training started successfully, False otherwise
        """
        try:
            print(f"[{job_id}] Starting training...")
            
            # Start tmux session
            result = remote_executor.start_tmux_session("training")
            if not result.success:
                print(f"[{job_id}] ✗ Failed to start tmux session: {result.stderr}")
                return False
            
            # Copy training config to remote instance
            remote_config_path = f"/tmp/train_config_{job_id}.yaml"
            if not remote_executor.copy_file(job_config.config_path, remote_config_path):
                print(f"[{job_id}] ✗ Failed to copy training config")
                return False
            
            # Start training command
            train_command = (
                f"cd robot && bazel run //experimental/overhead_matching/swag/scripts:train -- "
                f"--dataset_base /tmp/ --output_base /tmp/output_{job_id} "
                f"--train_config {remote_config_path}"
            )
            
            result = remote_executor.run_command_in_tmux(train_command)
            if not result.success:
                print(f"[{job_id}] ✗ Failed to start training command: {result.stderr}")
                return False
            
            print(f"[{job_id}] ✓ Training started")
            return True
            
        except Exception as e:
            print(f"[{job_id}] ✗ Training start error: {e}")
            return False
    
    def monitor_job(self, 
                   remote_executor: RemoteExecutor,
                   job_result: JobResult,
                   job_id: str) -> None:
        """Monitor a running training job.
        
        Args:
            remote_executor: Connected remote executor
            job_result: Job result to update
            job_id: Unique job identifier
        """
        max_train_time = self.machine_config.max_train_time_hours * 3600  # Convert to seconds
        start_time = time.time()
        
        while True:
            elapsed_time = time.time() - start_time
            
            # Check if maximum training time exceeded
            if elapsed_time > max_train_time:
                print(f"[{job_id}] ⏰ Maximum training time reached ({self.machine_config.max_train_time_hours}h)")
                break
            
            # Check tmux session status
            result = remote_executor.execute_command("tmux list-sessions | grep training")
            if not result.success:
                print(f"[{job_id}] ✓ Training completed (tmux session ended)")
                break
            
            # Check if training process is still running
            result = remote_executor.execute_command("pgrep -f 'train.*--train_config'")
            if not result.success:
                print(f"[{job_id}] ✓ Training process completed")
                break
            
            # Sleep before next check
            time.sleep(60)  # Check every minute
        
        # Update job status
        with self._lock:
            job_result.status = JobStatus.SHUTTING_DOWN
    
    def execute_job(self, job_config: JobConfig, job_id: str) -> JobResult:
        """Execute a single training job.
        
        Args:
            job_config: Job configuration
            job_id: Unique job identifier
            
        Returns:
            JobResult with execution status and details
        """
        job_result = JobResult(
            job_config=job_config,
            status=JobStatus.PENDING,
            start_time=time.time()
        )
        
        try:
            # Update status
            job_result.status = JobStatus.LAUNCHING
            
            # Try each machine type in order
            instance_info = None
            for machine_type in self.machine_config.machine_types:
                instance_info = self.launch_instance(machine_type, job_id)
                if instance_info:
                    break
                print(f"[{job_id}] Trying next machine type...")
            
            if not instance_info:
                job_result.status = JobStatus.FAILED
                job_result.error_message = "Failed to launch any instance type"
                return job_result
            
            instance_id, instance_ip = instance_info
            job_result.instance_id = instance_id
            job_result.instance_ip = instance_ip
            
            # Connect to instance
            remote_executor = RemoteExecutor(instance_ip)
            if not remote_executor.connect():
                job_result.status = JobStatus.FAILED
                job_result.error_message = "Failed to connect to instance"
                return job_result
            
            try:
                # Setup instance
                job_result.status = JobStatus.SETTING_UP
                if not self.setup_instance(remote_executor, job_config, job_id):
                    job_result.status = JobStatus.FAILED
                    job_result.error_message = "Instance setup failed"
                    return job_result
                
                # Start training
                job_result.status = JobStatus.TRAINING
                if not self.start_training(remote_executor, job_config, job_id):
                    job_result.status = JobStatus.FAILED
                    job_result.error_message = "Failed to start training"
                    return job_result
                
                # Monitor training
                self.monitor_job(remote_executor, job_result, job_id)
                
                # Job completed successfully
                job_result.status = JobStatus.COMPLETED
                
            finally:
                remote_executor.disconnect()
            
        except Exception as e:
            job_result.status = JobStatus.FAILED
            job_result.error_message = str(e)
        
        finally:
            job_result.end_time = time.time()
        
        return job_result
    
    def run_jobs(self, job_configs: List[JobConfig]) -> List[JobResult]:
        """Run multiple training jobs in parallel.
        
        Args:
            job_configs: List of job configurations
            
        Returns:
            List of job results
        """
        print(f"Starting {len(job_configs)} training jobs with max parallelism {self.max_parallel_jobs}")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_parallel_jobs) as executor:
            # Submit all jobs
            future_to_job = {}
            for i, job_config in enumerate(job_configs):
                job_id = f"job_{i:03d}"
                future = executor.submit(self.execute_job, job_config, job_id)
                future_to_job[future] = job_id
            
            # Collect results as they complete
            for future in as_completed(future_to_job):
                job_id = future_to_job[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.status == JobStatus.COMPLETED:
                        print(f"[{job_id}] ✅ Job completed successfully")
                    else:
                        print(f"[{job_id}] ❌ Job failed: {result.error_message}")
                        
                except Exception as e:
                    print(f"[{job_id}] ❌ Job failed with exception: {e}")
                    results.append(JobResult(
                        job_config=job_configs[int(job_id.split('_')[1])],
                        status=JobStatus.FAILED,
                        error_message=str(e)
                    ))
        
        return results