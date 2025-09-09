#!/usr/bin/env python3
"""Job management and parallel execution for Lambda Cloud training jobs."""

import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from common.tools.lambda_cloud.lambda_api.client import InsufficientCapacityError
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
            # Use first configured region for now (could be enhanced to try multiple regions)
            region_name = self.machine_config.region[0]
            instance_name = f"training-{job_id}-{int(time.time())}"
            
            instance_ids = self.lambda_client.launch_instance(
                region_name=region_name,
                instance_type_name=machine_type,
                ssh_key_names=[self.machine_config.ssh_key],
                name=instance_name,
                file_system_names=self.machine_config.file_systems,
                quantity=1,
                image_family=self.machine_config.image_family
            )
            
            if not instance_ids or len(instance_ids) == 0:
                print(f"[{job_id}] ✗ Failed to launch instance")
                return None
            
            instance_id = instance_ids[0]
            print(f"[{job_id}] ✓ Launched instance {instance_id}")
            
            # Wait for instance to be ready and get IP
            max_wait_time = 20 * 60  # 20 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                instances = self.lambda_client.list_instances()
                for inst in instances:
                    if inst.id == instance_id:
                        # Handle None/empty IP during instance boot
                        ip_status = inst.ip if inst.ip else "None"
                        print(f"[{job_id}] Instance {instance_id} status: {inst.status.value}, IP: {ip_status}")
                        if inst.status.value == "active" and inst.ip:
                            print(f"[{job_id}] ✓ Instance ready at {inst.ip}")
                            return instance_id, inst.ip
                        break
                
                time.sleep(30)
            
            print(f"[{job_id}] ✗ Instance failed to become ready within {max_wait_time}s")
            return None
        except InsufficientCapacityError:  # no nodes avaliable
            print(f"[{job_id}] Failed to launch as out of capacity.")
            return None

        except Exception as e:
            print(f"[{job_id}] ✗ Error launching instance: {e}")
            print(f"[{job_id}] Full traceback:")
            traceback.print_exc()
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
                print(f"[{job_id}]   Copying: {local_path} -> {remote_path}")
                if not remote_executor.copy_file(local_path, remote_path):
                    print(f"[{job_id}] ✗ File copy failed, aborting setup")
                    return False
            
            # Clone repository with specified branch
            print(f"[{job_id}] Cloning repository (branch: {job_config.branch})...")
            if job_config.branch == "main":
                clone_cmd = "git clone https://github.com/ewfuentes/robot.git"
            else:
                clone_cmd = f"git clone -b {job_config.branch} https://github.com/ewfuentes/robot.git"
            
            result = remote_executor.execute_command(clone_cmd, timeout=300)  # 5 minute timeout
            if not result.success:
                print(f"[{job_id}] ✗ Failed to clone repository: {result.stderr}")
                return False
            
            # Run setup.sh to install bazel and dependencies
            print(f"[{job_id}] Running repository setup...")
            result = remote_executor.execute_command("cd /home/ubuntu/robot && ./setup.sh", timeout=600)  # 10 minute timeout
            if not result.success:
                print(f"[{job_id}] ✗ Repository setup failed: {result.stderr}")
                return False
            
            print(result.stderr)
            print(result.stdout)
            
            # Run additional setup commands
            print(f"[{job_id}] Running additional setup commands...")
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
            print(f"[{job_id}] Full traceback:")
            traceback.print_exc()
            return False
    
    def start_autonomous_training(self,
                                 remote_executor: RemoteExecutor,
                                 job_config: JobConfig,
                                 job_id: str,
                                 instance_ip: str,
                                 api_key: str) -> bool:
        """Start autonomous training job with remote monitoring.
        
        Args:
            remote_executor: Connected remote executor
            job_config: Job configuration  
            job_id: Unique job identifier
            instance_ip: Instance IP address
            api_key: Lambda Cloud API key
            
        Returns:
            True if training started successfully, False otherwise
        """
        try:
            print(f"[{job_id}] Starting training...")
            
            # Copy training config to remote instance  
            remote_config_path = f"/tmp/train_config_{job_id}.yaml"
            if not remote_executor.copy_file(job_config.config_path, remote_config_path):
                print(f"[{job_id}] ✗ Failed to copy training config")
                return False
            
            # Prepare training command
            train_command = (
                f"cd /home/ubuntu/robot && bazel run //experimental/overhead_matching/swag/scripts:train -- "
                f"--dataset_base /tmp/ --output_base /tmp/output_{job_id} "
                f"--train_config {remote_config_path} --no-ipdb"
            )
            
            # Prepare S3 configuration  
            s3_bucket = "rrg-overhead-matching"  # TODO: Make this configurable
            s3_key_prefix = f"training_outputs/{job_id}"
            
            # Start autonomous monitoring using bazel run
            monitor_command = (
                f"cd /home/ubuntu/robot && nohup bazel run "
                f"//common/tools/lambda_cloud/lambda_launch:remote_monitor -- "
                f"--training-command '{train_command}' "
                f"--max-train-hours {self.machine_config.max_train_time_hours} "
                f"--output-dir /tmp/output_{job_id} "
                f"--s3-bucket {s3_bucket} "
                f"--s3-key-prefix {s3_key_prefix} "
                f"--api-key {api_key} "
                f"--instance-ip {instance_ip} "
                f"> /tmp/monitor.log 2>&1 &"
            )
            print(monitor_command)
            result = remote_executor.execute_command(monitor_command)
            if not result.success:
                print(f"[{job_id}] ✗ Failed to start remote monitor: {result.stderr}")
                return False
            
            print(f"[{job_id}] ✓ Autonomous training started")
            print(f"[{job_id}] SSH to instance: ssh ubuntu@{instance_ip}")
            print(f"[{job_id}] Monitor logs: tail -f /tmp/monitor.log")
            print(f"[{job_id}] Training logs: tail -f /tmp/training.log")
            print(f"[{job_id}] S3 destination: s3://{s3_bucket}/{s3_key_prefix}/")
            return True
            
        except Exception as e:
            print(f"[{job_id}] ✗ Training start error: {e}")
            return False
    
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
            
            # Try each machine type in order until one works
            instance_info = None
            while True:
                for machine_type in self.machine_config.machine_types:
                    instance_info = self.launch_instance(machine_type, job_id)
                    if instance_info:
                        break
                if instance_info:
                    break
                time.sleep(15)
            
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
                
                # Start autonomous training
                job_result.status = JobStatus.TRAINING
                if not self.start_autonomous_training(remote_executor, job_config, job_id, instance_ip, self.lambda_client.api_key):
                    job_result.status = JobStatus.FAILED
                    job_result.error_message = "Failed to start autonomous training"
                    return job_result
                
                # Job is now autonomous - mark as completed from host perspective
                job_result.status = JobStatus.COMPLETED
                job_result.output_s3_path = f"s3://rrg-overhead-matching/training_outputs/{job_id}/"
                
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
                # Generate job_id: YYMMDD_HHMMSS_nameofconfigfile_uniquesha
                timestamp = time.strftime("%y%m%d_%H%M%S")
                config_name = Path(job_config.config_path).stem
                job_id = f"{timestamp}_{config_name}"
                assert job_id not in future_to_job.values(), f"Created jobs too fast, ended up wit two with same second: {future_to_job}. Tried to add {job_id}"
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