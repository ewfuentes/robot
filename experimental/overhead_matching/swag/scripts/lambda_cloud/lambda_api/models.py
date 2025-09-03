"""Data models for Lambda Cloud API responses."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class InstanceStatus(Enum):
    """Instance status values from the API."""
    BOOTING = "booting"
    ACTIVE = "active"
    UNHEALTHY = "unhealthy"
    TERMINATED = "terminated"
    TERMINATING = "terminating"
    PREEMPTED = "preempted"


@dataclass
class InstanceType:
    """Represents an available instance type."""
    name: str
    description: str
    price_cents_per_hour: int
    vcpus: int
    memory_gib: int
    storage_gib: int
    regions: List[str]
    specs: Dict[str, Any]  # GPU specs and other details


@dataclass
class Instance:
    """Represents a running instance."""
    id: str
    name: str
    ip: str
    private_ip: Optional[str]
    status: InstanceStatus
    ssh_key_names: List[str]
    file_system_names: List[str]
    region: str
    instance_type: str
    hostname: str
    jupyter_token: Optional[str] = None
    jupyter_url: Optional[str] = None


@dataclass
class SSHInfo:
    """SSH connection information for an instance."""
    instance_id: str
    instance_name: str
    ip: str
    hostname: str
    ssh_key_names: List[str]
    username: str = "ubuntu"  # Default username for Lambda instances


class LambdaCloudError(Exception):
    """Base exception for Lambda Cloud API errors."""
    
    def __init__(self, code: str, message: str, suggestion: Optional[str] = None):
        self.code = code
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"{code}: {message}")


class InsufficientCapacityError(LambdaCloudError):
    """Raised when there's insufficient capacity for launch request."""
    pass


class RateLimitError(LambdaCloudError):
    """Raised when API rate limit is exceeded."""
    pass