#!/usr/bin/env python3
"""Configuration parsing and validation for Lambda Cloud job launcher."""

import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MachineConfig:
    """Configuration for Lambda Cloud machine setup."""
    machine_types: List[str]
    region: List[str]  # Can be single region or list of regions
    ssh_key: str
    image_family: str  # Image family name (e.g., "lambda-stack-24-04")
    file_systems: List[str]  # File systems to mount (e.g., ["vigor"])
    files_to_copy: Dict[str, str]
    remote_setup_commands: List[str]
    max_train_time_hours: int


@dataclass 
class JobConfig:
    """Configuration for a single training job."""
    config_path: str
    branch: str = "main"


class ConfigParser:
    """Parse and validate configuration files."""
    
    @staticmethod
    def parse_machine_config(config_path: str) -> MachineConfig:
        """Parse machine setup configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Handle region as single value or list
        region_config = config['region']
        if isinstance(region_config, str):
            regions = [region_config]
        else:
            regions = region_config
        
        file_systems = config.get('file_systems')
        if isinstance(file_systems, str):
            file_systems = [file_systems]
        
        return MachineConfig(
            machine_types=config['machine_types'],
            region=regions,
            ssh_key=config['ssh_key'],
            image_family=config.get('image_family', 'lambda-stack-24-04'),
            file_systems=file_systems,
            files_to_copy=config['files_to_copy'],
            remote_setup_commands=config['remote_setup_commands'],
            max_train_time_hours=config['max_train_time_hours']
        )
    
    @staticmethod
    def parse_job_configs(config_paths: List[str], branches: Optional[List[str]] = None) -> List[JobConfig]:
        """Parse job configurations from config paths and branches."""
        jobs = []
        
        if branches is None:
            branches = ["main"] * len(config_paths)
        elif len(branches) == 1:
            branches = branches * len(config_paths)
        elif len(branches) != len(config_paths):
            raise ValueError("Number of branches must match number of config paths or be 1")
        
        for config_path, branch in zip(config_paths, branches):
            # Validate config file exists
            if not Path(config_path).exists():
                raise FileNotFoundError(f"Training config not found: {config_path}")
            
            jobs.append(JobConfig(config_path=config_path, branch=branch))
        
        return jobs
    
    @staticmethod
    def parse_job_file(job_file_path: str) -> List[JobConfig]:
        """Parse job configurations from comma-separated file."""
        jobs = []
        
        with open(job_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(',')
                if len(parts) == 1:
                    config_path = parts[0].strip()
                    branch = "main"
                elif len(parts) == 2:
                    config_path, branch = [p.strip() for p in parts]
                else:
                    raise ValueError(f"Invalid line {line_num} in {job_file_path}: {line}")
                
                # Validate config file exists
                if not Path(config_path).exists():
                    raise FileNotFoundError(f"Training config not found: {config_path}")
                
                jobs.append(JobConfig(config_path=config_path, branch=branch))
        
        return jobs
    
    @staticmethod
    def validate_config(machine_config: MachineConfig) -> None:
        """Validate machine configuration."""
        if not machine_config.machine_types:
            raise ValueError("At least one machine type must be specified")
        
        if machine_config.max_train_time_hours <= 0:
            raise ValueError("max_train_time_hours must be positive")
        
        if not machine_config.ssh_key:
            raise ValueError("ssh_key must be specified")