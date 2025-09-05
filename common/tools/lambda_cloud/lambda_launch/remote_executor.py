#!/usr/bin/env python3
"""Remote command execution and file transfer for Lambda Cloud instances."""

import paramiko
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of a remote command execution."""
    return_code: int
    stdout: str
    stderr: str
    success: bool


class RemoteExecutor:
    """Execute commands and transfer files on remote Lambda Cloud instances."""
    
    def __init__(self, hostname: str, username: str = "ubuntu", port: int = 22):
        """Initialize remote executor for a Lambda Cloud instance.
        
        Args:
            hostname: IP address or hostname of the instance
            username: SSH username (default: ubuntu for Lambda Cloud)
            port: SSH port (default: 22)
        """
        self.hostname = hostname
        self.username = username
        self.port = port
        self.client: Optional[paramiko.SSHClient] = None
        self.sftp: Optional[paramiko.SFTPClient] = None
    
    def connect(self, max_retries: int = 5, retry_delay: int = 10) -> bool:
        """Connect to the remote instance via SSH.
        
        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retry attempts in seconds
            
        Returns:
            True if connection successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                self.client = paramiko.SSHClient()
                self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                # Connect to the instance
                self.client.connect(
                    hostname=self.hostname,
                    username=self.username,
                    port=self.port,
                    timeout=30,
                    banner_timeout=30
                )
                
                # Initialize SFTP client for file transfers
                self.sftp = self.client.open_sftp()
                
                print(f"✓ Connected to {self.hostname}")
                return True
                
            except Exception as e:
                print(f"✗ Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"✗ Failed to connect to {self.hostname} after {max_retries} attempts")
                    return False
        
        return False
    
    def disconnect(self) -> None:
        """Close SSH and SFTP connections."""
        if self.sftp:
            self.sftp.close()
            self.sftp = None
        
        if self.client:
            self.client.close()
            self.client = None
    
    def execute_command(self, command: str, timeout: int = 300) -> ExecutionResult:
        """Execute a command on the remote instance.
        
        Args:
            command: Command to execute
            timeout: Command timeout in seconds
            
        Returns:
            ExecutionResult with command output and status
        """
        if not self.client:
            raise RuntimeError("Not connected to remote instance")
        
        try:
            stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
            
            # Wait for command completion
            return_code = stdout.channel.recv_exit_status()
            
            stdout_text = stdout.read().decode('utf-8', errors='ignore')
            stderr_text = stderr.read().decode('utf-8', errors='ignore')
            
            return ExecutionResult(
                return_code=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
                success=(return_code == 0)
            )
            
        except Exception as e:
            return ExecutionResult(
                return_code=-1,
                stdout="",
                stderr=str(e),
                success=False
            )
    
    def copy_file(self, local_path: str, remote_path: str) -> bool:
        """Copy a file from local to remote instance.
        
        Args:
            local_path: Local file path
            remote_path: Remote destination path
            
        Returns:
            True if successful, False otherwise
        """
        if not self.sftp:
            raise RuntimeError("SFTP not initialized")
        
        try:
            # Expand home directory if needed
            local_path = str(Path(local_path).expanduser())
            
            # Create remote directory if needed
            remote_dir = str(Path(remote_path).parent)
            if remote_dir != '.':
                self.execute_command(f"mkdir -p {remote_dir}")
            
            self.sftp.put(local_path, remote_path)
            print(f"✓ Copied {local_path} -> {remote_path}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to copy {local_path} -> {remote_path}: {e}")
            return False
    
    def copy_directory(self, local_path: str, remote_path: str) -> bool:
        """Copy a directory recursively from local to remote instance.
        
        Args:
            local_path: Local directory path
            remote_path: Remote destination path
            
        Returns:
            True if successful, False otherwise
        """
        if not self.sftp:
            raise RuntimeError("SFTP not initialized")
        
        try:
            local_path = Path(local_path).expanduser()
            
            # Create remote directory
            self.execute_command(f"mkdir -p {remote_path}")
            
            # Recursively copy files
            for local_file in local_path.rglob('*'):
                if local_file.is_file():
                    relative_path = local_file.relative_to(local_path)
                    remote_file_path = f"{remote_path}/{relative_path}"
                    
                    # Create remote subdirectory if needed
                    remote_file_dir = str(Path(remote_file_path).parent)
                    if remote_file_dir != remote_path:
                        self.execute_command(f"mkdir -p {remote_file_dir}")
                    
                    if not self.copy_file(str(local_file), remote_file_path):
                        return False
            
            print(f"✓ Copied directory {local_path} -> {remote_path}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to copy directory {local_path} -> {remote_path}: {e}")
            return False
    
    def start_tmux_session(self, session_name: str = "training") -> ExecutionResult:
        """Start a new tmux session on the remote instance.
        
        Args:
            session_name: Name of the tmux session
            
        Returns:
            ExecutionResult from tmux command
        """
        command = f"tmux new-session -d -s {session_name}"
        return self.execute_command(command)
    
    def run_command_in_tmux(self, command: str, session_name: str = "training") -> ExecutionResult:
        """Run a command inside an existing tmux session.
        
        Args:
            command: Command to run
            session_name: Name of the tmux session
            
        Returns:
            ExecutionResult from tmux send-keys command
        """
        tmux_command = f'tmux send-keys -t {session_name} "{command}" Enter'
        return self.execute_command(tmux_command)
    
    def get_tmux_output(self, session_name: str = "training") -> str:
        """Get the current output from a tmux session.
        
        Args:
            session_name: Name of the tmux session
            
        Returns:
            Current tmux pane content
        """
        result = self.execute_command(f"tmux capture-pane -t {session_name} -p")
        return result.stdout if result.success else ""