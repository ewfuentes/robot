#!/usr/bin/env python3
"""Job management and parallel execution for Lambda Cloud training jobs."""

import time
import threading
import traceback
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime
from common.tools.lambda_cloud.lambda_api.client import InsufficientCapacityError
from common.tools.lambda_cloud.lambda_launch.config import JobConfig, MachineConfig
from common.tools.lambda_cloud.lambda_launch.remote_executor import RemoteExecutor


class JobLogger:
    """Logger that writes detailed logs to job-specific files while keeping terminal output minimal."""
    
    def __init__(self, job_id: str, log_dir: Path):
        """Initialize job logger.
        
        Args:
            job_id: Unique job identifier
            log_dir: Directory to store log files
        """
        self.job_id = job_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{job_id}.log"
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"=== Training Job {job_id} Log ===\n")
            f.write(f"Started: {datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
    
    def log(self, message: str, level: str = "INFO", terminal: bool = False):
        """Log a message to the job log file and optionally to terminal.
        
        Args:
            message: Message to log
            level: Log level (INFO, ERROR, DEBUG, etc.)
            terminal: If True, also print to terminal
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {level}: {message}"
        
        # Always write to log file
        with open(self.log_file, 'a') as f:
            f.write(log_line + '\n')
            f.flush()
        
        # Optionally print to terminal
        if terminal:
            print(f"[{self.job_id}] {message}")
    
    def log_command(self, command: str, result=None):
        """Log a command and its result.
        
        Args:
            command: Command that was executed
            result: ExecutionResult from the command (optional)
        """
        self.log(f"Executing command: {command}", "CMD")
        
        if result:
            self.log(f"Command exit code: {result.return_code}", "CMD")
            if result.stdout:
                self.log(f"Command stdout:\n{result.stdout}", "STDOUT")
            if result.stderr:
                self.log(f"Command stderr:\n{result.stderr}", "STDERR")
            
            success_msg = "✓" if result.success else "✗"
            self.log(f"Command result: {success_msg}", "CMD")


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
                 max_parallel_jobs: int = 10,
                 log_dir: Optional[Path] = None,
                 setup_only: bool = False):
        """Initialize job manager.

        Args:
            machine_config: Machine configuration for all jobs
            lambda_client: Lambda Cloud client instance
            max_parallel_jobs: Maximum number of parallel jobs
            log_dir: Directory for job log files (optional)
            setup_only: If True, only setup instances without starting training
        """
        self.machine_config = machine_config
        self.lambda_client = lambda_client
        self.max_parallel_jobs = max_parallel_jobs
        self.log_dir = log_dir or Path("/tmp/lambda_jobs")
        self.setup_only = setup_only
        self.active_jobs: Dict[str, JobResult] = {}
        self.completed_jobs: List[JobResult] = []
        self._lock = threading.Lock()
    
    def _generate_safe_instance_name(self, job_id: str) -> str:
        """Generate a safe instance name that fits Lambda Cloud's 64 character limit.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Instance name that is <= 64 characters
        """
        base_name = f"training-{job_id}"
        
        # Lambda Cloud has a 64 character limit for instance names
        max_length = 64
        
        if len(base_name) <= max_length:
            return base_name
        
        # If too long, create a hash-based name
        # Format: training-{trimmed_job_id}-{hash}
        job_id_hash = hashlib.md5(job_id.encode()).hexdigest()[:8]
        
        # Calculate available space for trimmed job_id
        # "training-" (9) + "-" (1) + hash (8) + "-" (1) = 19 chars
        fixed_chars = 9 + 1 + 8 + 1
        available_for_job_id = max_length - fixed_chars
        
        if available_for_job_id > 0:
            trimmed_job_id = job_id[:available_for_job_id]
            return f"training-{trimmed_job_id}-{job_id_hash}"
        else:
            # Extreme case: use only hash and timestamp
            return f"training-{job_id_hash}"
    
    def launch_instance(self, machine_type: str, region_name: str, job_id: str, logger: JobLogger) -> Optional[tuple]:
        """Launch a Lambda Cloud instance.
        
        Args:
            machine_type: Type of machine to launch
            region_name: Region to launch instance in
            job_id: Unique job identifier
            logger: JobLogger instance for this job
            
        Returns:
            (instance_id, instance_ip) if successful, None otherwise
        """
        try:
            logger.log(f"Launching {machine_type} instance in {region_name}...", terminal=True)
            
            # Launch instance using Lambda Cloud API
            instance_name = self._generate_safe_instance_name(job_id)
            logger.log(f"Using instance name: {instance_name} (length: {len(instance_name)})")
            
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
                logger.log("Failed to launch instance", "ERROR", terminal=True)
                return None
            
            instance_id = instance_ids[0]
            logger.log(f"Launched instance {instance_id}", "INFO", terminal=True)
            
            # Wait for instance to be ready and get IP
            max_wait_time = 20 * 60  # 20 minutes
            start_time = time.time()
            
            logger.log(f"Waiting for instance to become ready (max {max_wait_time}s)...")
            
            while time.time() - start_time < max_wait_time:
                instances = self.lambda_client.list_instances()
                for inst in instances:
                    if inst.id == instance_id:
                        # Handle None/empty IP during instance boot
                        ip_status = inst.ip if inst.ip else "None"
                        logger.log(f"Instance {instance_id} status: {inst.status.value}, IP: {ip_status}")
                        if inst.status.value == "active" and inst.ip:
                            logger.log(f"Instance ready at {inst.ip}", "INFO", terminal=True)
                            return instance_id, inst.ip
                        break
                
                time.sleep(30)
            
            logger.log(f"Instance failed to become ready within {max_wait_time}s", "ERROR", terminal=True)
            self.lambda_client.terminate_instances([instance_id])  # attempt to teriminate errored instance
            return None
        except InsufficientCapacityError:
            logger.log("Failed to launch - insufficient capacity", "ERROR", terminal=True)
            return None

        except Exception as e:
            logger.log(f"Error launching instance: {e}", "ERROR", terminal=True)
            logger.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR")
            return None
    
    def setup_instance(self, 
                      remote_executor: RemoteExecutor, 
                      job_config: JobConfig, 
                      job_id: str,
                      logger: JobLogger) -> bool:
        """Setup the remote instance for training.
        
        Args:
            remote_executor: Connected remote executor
            job_config: Job configuration
            job_id: Unique job identifier
            logger: JobLogger instance for this job
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.log("Setting up instance...", "INFO", terminal=True)
            
            # Copy files to remote instance
            logger.log("Copying files...")
            for local_path, remote_path in self.machine_config.files_to_copy.items():
                logger.log(f"Copying: {local_path} -> {remote_path}")
                if not remote_executor.copy_file(local_path, remote_path):
                    logger.log("File copy failed, aborting setup", "ERROR", terminal=True)
                    return False
            
            # Clone repository with specified branch
            logger.log(f"Cloning repository (branch: {job_config.branch})...", terminal=True)
            if job_config.branch == "main":
                clone_cmd = "git clone https://github.com/ewfuentes/robot.git"
            else:
                clone_cmd = f"git clone -b {job_config.branch} https://github.com/ewfuentes/robot.git"
            
            result = remote_executor.execute_command(clone_cmd, timeout=300)  # 5 minute timeout
            logger.log_command(clone_cmd, result)
            if not result.success:
                logger.log(f"Failed to clone repository: {result.stderr}", "ERROR", terminal=True)
                return False
            
            # Run setup.sh to install bazel and dependencies
            logger.log("Running repository setup...")
            setup_cmd = "cd /home/ubuntu/robot && ./setup.sh"
            result = remote_executor.execute_command(setup_cmd, timeout=600)  # 10 minute timeout
            logger.log_command(setup_cmd, result)
            if not result.success:
                logger.log(f"Repository setup failed: {result.stderr}", "ERROR", terminal=True)
                return False
            
            # Run additional setup commands
            logger.log("Running additional setup commands...")
            for command in self.machine_config.remote_setup_commands:
                logger.log(f"Running: {command}")
                result = remote_executor.execute_command(command, timeout=600)  # 10 minute timeout
                logger.log_command(command, result)
                if not result.success:
                    logger.log(f"Setup command failed: {command}", "ERROR", terminal=True)
                    logger.log(f"Error: {result.stderr}", "ERROR")
                    return False
            
            logger.log("Instance setup completed", "INFO", terminal=True)
            return True
            
        except Exception as e:
            logger.log(f"Setup error: {e}", "ERROR", terminal=True)
            logger.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR")
            return False
    
    def start_autonomous_training(self,
                                 remote_executor: RemoteExecutor,
                                 job_config: JobConfig,
                                 job_id: str,
                                 instance_id: str,
                                 instance_ip: str,
                                 api_key: str,
                                 logger: JobLogger) -> bool:
        """Start autonomous training job with remote monitoring.
        
        Args:
            remote_executor: Connected remote executor
            job_config: Job configuration  
            job_id: Unique job identifier
            instance_id: Lambda Cloud instance ID
            instance_ip: Instance IP address
            api_key: Lambda Cloud API key
            logger: JobLogger instance for this job
            
        Returns:
            True if training started successfully, False otherwise
        """
        try:
            logger.log("Starting training...", "INFO", terminal=True)

            # Training config has already been copied to remote instance
            remote_config_path = f"/tmp/train_config_{job_id}.yaml"

            # Prepare training command using bazel
            base_train_args = (
                f"--dataset_base /tmp/ --output_base /tmp/output_{job_id} "
                f"--train_config {remote_config_path} --no_ipdb --quiet"
            )

            # Use bazel to run the train script directly from the repo
            train_command = (
                f"cd /home/ubuntu/robot && bazel run "
                f"//experimental/overhead_matching/swag/scripts:train -- "
                f"{base_train_args}"
            )
            
            # Prepare S3 configuration  
            s3_bucket = "rrg-overhead-matching"  # TODO: Make this configurable
            s3_key_prefix = f"training_outputs/{job_id}"
            
            # Start autonomous monitoring using bazel run
            # stdbuf -oL to force flushing for each line
            # nohup to run even after session disconnects
            monitor_command = (
                f"cd /home/ubuntu/robot && stdbuf -oL nohup bazel run "
                f"//common/tools/lambda_cloud/lambda_launch:remote_monitor -- "
                f"--training-command '{train_command}' "
                f"--max-train-hours {self.machine_config.max_train_time_hours} "
                f"--output-dir /tmp/output_{job_id} "
                f"--s3-bucket {s3_bucket} "
                f"--s3-key-prefix {s3_key_prefix} "
                f"--api-key {api_key} "
                f"--instance-id {instance_id} "
                f"> /tmp/monitor.log 2>&1 &"
            )
            
            # Launch the monitor command with a short timeout to avoid waiting for the background process
            # The command will start the background process and return immediately
            logger.log("Launching monitor command...")
            try:
                result = remote_executor.execute_command(monitor_command, timeout=30)
                logger.log_command(monitor_command, result)
            except Exception as e:
                logger.log(f"Monitor command timed out (expected for background process): {e}")
                # This timeout is expected since the background process keeps the SSH channel open
                
            # Wait a moment for the process to start
            time.sleep(3)
            
            # Verify the monitor process is actually running by checking for the process
            verify_cmd = "ps aux | grep 'remote_monitor' | grep -v grep || echo 'No monitor process found'"
            verify_result = remote_executor.execute_command(verify_cmd, timeout=10)
            logger.log_command(verify_cmd, verify_result)
            
            if verify_result.success and "No monitor process found" not in verify_result.stdout:
                logger.log("Monitor process confirmed running", "INFO")
            else:
                logger.log("Monitor process not found - launch may have failed", "ERROR", terminal=True)
                # Check monitor log for startup errors
                log_check_cmd = "head -20 /tmp/monitor.log 2>/dev/null || echo 'No monitor log yet'"
                log_result = remote_executor.execute_command(log_check_cmd, timeout=10)
                logger.log(f"Monitor log snippet: {log_result.stdout}", "ERROR")
                return False
            
            logger.log("Autonomous training started", "INFO", terminal=True)
            logger.log(f"SSH to instance: ssh ubuntu@{instance_ip}")
            logger.log(f"Monitor logs: tail -f /tmp/monitor.log")
            logger.log(f"Training debug log: tail -f /tmp/training_debug.log")
            logger.log(f"Training output log: tail -f /tmp/training_output.log")
            return True
            
        except Exception as e:
            logger.log(f"Training start error: {e}", "ERROR", terminal=True)
            return False
    
    def execute_job(self, job_config: JobConfig, job_id: str) -> JobResult:
        """Execute a single training job.
        
        Args:
            job_config: Job configuration
            job_id: Unique job identifier
            
        Returns:
            JobResult with execution status and details
        """
        # Create job logger
        logger = JobLogger(job_id, self.log_dir)
        logger.log("Starting training job execution", terminal=True)
        
        job_result = JobResult(
            job_config=job_config,
            status=JobStatus.PENDING,
            start_time=time.time()
        )
        
        try:
            # Update status
            job_result.status = JobStatus.LAUNCHING
            
            # Try each machine type and region combination until one works
            instance_info = None
            while True:
                for machine_type in self.machine_config.get_all_machine_types():
                    regions = self.machine_config.get_regions_for_machine_type(machine_type)
                    for region_name in regions:
                        instance_info = self.launch_instance(machine_type, region_name, job_id, logger)
                        if instance_info:
                            break
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
                if not self.setup_instance(remote_executor, job_config, job_id, logger):
                    job_result.status = JobStatus.FAILED
                    job_result.error_message = "Instance setup failed"
                    return job_result

                # Copy training config to remote instance
                remote_config_path = f"/tmp/train_config_{job_id}.yaml"
                logger.log(f"Copying training config to {remote_config_path}")
                if not remote_executor.copy_file(job_config.config_path, remote_config_path):
                    logger.log("Failed to copy training config", "ERROR", terminal=True)
                    job_result.status = JobStatus.FAILED
                    job_result.error_message = "Failed to copy training config"
                    return job_result

                if self.setup_only:
                    # Setup-only mode: skip training start
                    logger.log("Setup-only mode: skipping training start", "INFO", terminal=True)
                    logger.log(f"Instance ready for manual training", "INFO", terminal=True)
                    logger.log(f"Training config copied to: {remote_config_path}", "INFO", terminal=True)
                    logger.log(f"SSH to instance: ssh ubuntu@{instance_ip}", "INFO", terminal=True)
                    job_result.status = JobStatus.COMPLETED
                    job_result.output_s3_path = None
                else:
                    # Start autonomous training
                    job_result.status = JobStatus.TRAINING
                    if not self.start_autonomous_training(remote_executor, job_config, job_id, instance_id, instance_ip, self.lambda_client.api_key, logger):
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
            for idx, job_config in enumerate(job_configs):
                # Generate job_id: YYMMDD_HHMMSS_config_name_idx
                timestamp = time.strftime("%y%m%d_%H%M%S")
                config_name = Path(job_config.config_path).stem
                job_id = f"{timestamp}_{config_name}_{idx:03d}"
                assert job_id not in future_to_job.values(), f"Created jobs too fast, ended up with two with same second: {future_to_job}. Tried to add {job_id}"
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