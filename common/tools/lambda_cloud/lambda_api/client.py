"""Lambda Cloud API client."""

import os
import time
import logging
import fcntl
import tempfile
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from common.tools.lambda_cloud.lambda_api.models import (
    Instance, InstanceType, InstanceStatus, SSHInfo,
    LambdaCloudError, InsufficientCapacityError, RateLimitError
)


class ProcessSafeRateLimiter:
    """Rate limiter that works across multiple processes using file locks."""
    
    def __init__(self, api_key_hash: str):
        """Initialize rate limiter with API key hash for isolation."""
        self.lockfile = Path(tempfile.gettempdir()) / f"lambda_api_rate_limit_{api_key_hash}.lock"
        self.statefile = Path(tempfile.gettempdir()) / f"lambda_api_rate_limit_{api_key_hash}.json"
    
    def rate_limit(self, is_launch: bool = False):
        """Enforce rate limits across all processes using file locking."""
        # Create lock file if it doesn't exist
        self.lockfile.touch(exist_ok=True)
        
        with open(self.lockfile, 'w') as lock_file:
            try:
                # Acquire exclusive lock with timeout to prevent deadlocks
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # If we can't get the lock immediately, wait a bit and try again
                time.sleep(0.1)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            
            try:
                # Read last request times from state file
                try:
                    with open(self.statefile, 'r') as f:
                        state = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    state = {"last_request": 0, "last_launch": 0}
                
                current_time = time.time()
                min_interval = 12 if is_launch else 1
                last_time = state["last_launch"] if is_launch else state["last_request"]
                
                # Calculate sleep time needed to respect rate limit
                time_since_last = current_time - last_time
                sleep_time = max(0, min_interval - time_since_last)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Update state with current time
                current_time = time.time()
                if is_launch:
                    state["last_launch"] = current_time
                else:
                    state["last_request"] = current_time
                
                # Write updated state back to file
                with open(self.statefile, 'w') as f:
                    json.dump(state, f)
                    
            finally:
                # Lock is automatically released when file is closed
                pass


class LambdaCloudClient:
    """Client for interacting with the Lambda Cloud API."""

    BASE_URL = "https://cloud.lambda.ai/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the client.

        Args:
            api_key: API key for authentication. If None, will try LAMBDA_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("LAMBDA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided via parameter or LAMBDA_API_KEY environment variable")

        # Create process-safe rate limiter using API key hash for isolation
        api_key_hash = hashlib.sha256(self.api_key.encode()).hexdigest()[:16]
        self.rate_limiter = ProcessSafeRateLimiter(api_key_hash)

        self.session = self._setup_session()

        # Set up logging
        self.logger = logging.getLogger(__name__)

    def _setup_session(self) -> requests.Session:
        """Set up requests session with retry strategy and authentication."""
        session = requests.Session()

        # Set up authentication header
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        })

        # Set up retry strategy for network issues
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _rate_limit(self, is_launch: bool = False):
        """Enforce API rate limits across all processes."""
        self.rate_limiter.rate_limit(is_launch)

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a request to the API with rate limiting and error handling."""
        is_launch = "launch" in endpoint
        self._rate_limit(is_launch)

        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = self.session.request(method, url, timeout=30, **kwargs)

            if response.status_code == 401:
                raise LambdaCloudError("unauthorized", "Invalid API key")
            elif response.status_code == 403:
                raise LambdaCloudError("forbidden", "Account inactive or access denied")
            elif response.status_code == 429:
                raise RateLimitError("rate_limit_exceeded", "API rate limit exceeded")

            if not response.ok:
                try:
                    error_data = response.json()
                    error_info = error_data.get("error", {})
                    code = error_info.get("code", f"http_{response.status_code}")
                    message = error_info.get("message", f"HTTP {response.status_code}")
                    suggestion = error_info.get("suggestion")

                    # Handle specific error types
                    if "insufficient-capacity" in code:
                        raise InsufficientCapacityError(code, message, suggestion)
                    else:
                        raise LambdaCloudError(code, message, suggestion)
                except ValueError:
                    # Response is not JSON
                    raise LambdaCloudError(f"http_{response.status_code}", response.text)

            return response.json()

        except requests.exceptions.Timeout:
            raise LambdaCloudError("timeout", "Request timed out")
        except requests.exceptions.ConnectionError:
            raise LambdaCloudError("connection_error", "Failed to connect to API")
        except requests.exceptions.RequestException as e:
            raise LambdaCloudError("request_error", str(e))

    def list_instance_types(self) -> List[InstanceType]:
        """List available instance types with pricing and specifications."""
        response = self._make_request("GET", "/instance-types")

        instance_types = []
        for _, instance_data in response["data"].items():
            instance_type_info = instance_data["instance_type"]
            regions = instance_data.get("regions_with_capacity_available", [])

            # Extract region names from region objects
            region_names = [region["name"] for region in regions]

            instance_type = InstanceType(
                name=instance_type_info["name"],
                description=instance_type_info.get("description", ""),
                price_cents_per_hour=instance_type_info["price_cents_per_hour"],
                vcpus=instance_type_info.get("specs", {}).get("vcpus", 0),
                memory_gib=instance_type_info.get("specs", {}).get("memory_gib", 0),
                storage_gib=instance_type_info.get("specs", {}).get("storage_gib", 0),
                regions=region_names,
                specs=instance_type_info.get("specs", {})
            )
            instance_types.append(instance_type)

        return instance_types

    def list_instances(self) -> List[Instance]:
        """List all running instances."""
        response = self._make_request("GET", "/instances")

        instances = []
        for item in response["data"]:
            instance = Instance(
                id=item["id"],
                name=item.get("name"),
                ip=item.get("ip"),
                private_ip=item.get("private_ip"),
                status=InstanceStatus(item["status"]),
                ssh_key_names=item["ssh_key_names"],
                file_system_names=item["file_system_names"],
                region=item["region"]["name"],
                instance_type=item["instance_type"]["name"],
                hostname=item.get("hostname"),
                jupyter_token=item.get("jupyter_token"),
                jupyter_url=item.get("jupyter_url")
            )
            instances.append(instance)

        return instances

    def get_instance(self, instance_id: str) -> Instance:
        """Get details for a specific instance."""
        response = self._make_request("GET", f"/instances/{instance_id}")
        item = response["data"]

        return Instance(
            id=item["id"],
            name=item.get("name"),
            ip=item.get("ip"),
            private_ip=item.get("private_ip"),
            status=InstanceStatus(item["status"]),
            ssh_key_names=item["ssh_key_names"],
            file_system_names=item["file_system_names"],
            region=item["region"]["name"],
            instance_type=item["instance_type"]["name"],
            hostname=item.get("hostname"),
            jupyter_token=item.get("jupyter_token"),
            jupyter_url=item.get("jupyter_url")
        )

    def launch_instance(self,
                        region_name: str,
                        instance_type_name: str,
                        ssh_key_names: List[str],
                        name: str,
                        file_system_names: Optional[List[str]] = None,
                        quantity: int = 1,
                        image_family: Optional[str] = None) -> List[str]:
        """Launch one or more instances.

        Args:
            region_name: Region to launch in
            instance_type_name: Type of instance to launch
            ssh_key_names: SSH keys to add to the instance
            name: Name for the instance
            file_system_names: File systems to mount (optional)
            quantity: Number of instances to launch
            image_family: Image family name to use (optional)

        Returns:
            List of instance IDs for launched instances
        """
        data = {
            "region_name": region_name,
            "instance_type_name": instance_type_name,
            "ssh_key_names": ssh_key_names,
            "name": name,
            "quantity": quantity,
            "file_system_names": file_system_names or []
        }

        if image_family:
            data["image"] = {"family": image_family}

        response = self._make_request("POST", "/instance-operations/launch", json=data)
        return response["data"]["instance_ids"]

    def launch_instance_with_retry(self,
                                   region_name: str,
                                   instance_type_name: str,
                                   ssh_key_names: List[str],
                                   name: str,
                                   file_system_names: Optional[List[str]] = None,
                                   quantity: int = 1,
                                   max_retries: int = 10,
                                   callback=None,
                                   image_family: Optional[str] = None) -> List[str]:
        """Launch instances with intelligent retry and capacity monitoring.

        Args:
            region_name: Region to launch in
            instance_type_name: Type of instance to launch  
            ssh_key_names: SSH keys to add to the instance
            name: Name for the instance
            file_system_names: File systems to mount (optional)
            quantity: Number of instances to launch
            max_retries: Maximum number of retry attempts
            callback: Optional callback function for status updates
            image_family: Image family name to use (optional)

        Returns:
            List of instance IDs for launched instances
        """
        for attempt in range(max_retries + 1):
            try:
                if callback:
                    callback(
                        f"Attempt {attempt + 1}/{max_retries + 1}: Launching {quantity}x {instance_type_name} in {region_name}")

                return self.launch_instance(
                    region_name, instance_type_name, ssh_key_names,
                    name, file_system_names, quantity, image_family
                )

            except InsufficientCapacityError as e:
                if attempt >= max_retries:
                    raise e

                # Get available instance types to provide helpful feedback
                try:
                    instance_types = self.list_instance_types()
                    available_types = [t for t in instance_types if region_name in t.regions]

                    if callback:
                        if not available_types:
                            callback(
                                f"No capacity in {region_name}. Available regions for {instance_type_name}: checking...")
                            # Find regions with this instance type
                            matching_types = [
                                t for t in instance_types if t.name == instance_type_name]
                            if matching_types:
                                available_regions = matching_types[0].regions
                                callback(
                                    f"Available regions for {instance_type_name}: {', '.join(available_regions)}")
                        else:
                            available_names = [t.name for t in available_types]
                            callback(
                                f"No capacity for {instance_type_name} in {region_name}. Available types: {', '.join(available_names)}")

                except Exception as list_error:
                    if callback:
                        callback(f"Could not fetch available types: {list_error}")

                wait_time = min(30 * (2 ** attempt), 300)  # Exponential backoff, max 5 min
                if callback:
                    callback(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

            except Exception as e:
                if callback:
                    callback(f"Launch failed: {e}")
                raise e

        raise InsufficientCapacityError(
            "max_retries_exceeded", f"Failed to launch after {max_retries + 1} attempts")

    def terminate_instances(self, instance_ids: List[str]) -> List[Instance]:
        """Terminate one or more instances.

        Args:
            instance_ids: List of instance IDs to terminate

        Returns:
            List of terminated instances
        """
        data = {"instance_ids": instance_ids}
        response = self._make_request("POST", "/instance-operations/terminate", json=data)

        terminated = []
        for item in response["data"]["terminated_instances"]:
            instance = Instance(
                id=item["id"],
                name=item.get("name"),
                ip=item["ip"],
                private_ip=item.get("private_ip"),
                status=InstanceStatus(item["status"]),
                ssh_key_names=item["ssh_key_names"],
                file_system_names=item["file_system_names"],
                region=item["region"]["name"],
                instance_type=item["instance_type"]["name"],
                hostname=item.get("hostname"),
                jupyter_token=item.get("jupyter_token"),
                jupyter_url=item.get("jupyter_url")
            )
            terminated.append(instance)

        return terminated

    def get_ssh_info(self, instance_id: str) -> SSHInfo:
        """Get SSH connection information for an instance."""
        instance = self.get_instance(instance_id)

        if not instance.ip:
            raise LambdaCloudError(f"Instance {instance_id} does not have an IP address assigned yet")
        if not instance.hostname:
            raise LambdaCloudError(f"Instance {instance_id} does not have a hostname assigned yet")

        return SSHInfo(
            instance_id=instance.id,
            instance_name=instance.name,
            ip=instance.ip,
            hostname=instance.hostname,
            ssh_key_names=instance.ssh_key_names,
            username="ubuntu"
        )
