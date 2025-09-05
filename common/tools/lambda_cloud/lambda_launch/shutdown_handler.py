#!/usr/bin/env python3
"""Shutdown procedures for Lambda Cloud training jobs."""

import boto3
import json
import time
from pathlib import Path
from typing import Optional
from common.tools.lambda_cloud.lambda_launch.remote_executor import RemoteExecutor, ExecutionResult


class ShutdownHandler:
    """Handle shutdown procedures including data sync and instance termination."""
    
    def __init__(self, remote_executor: RemoteExecutor, lambda_api_key: str):
        """Initialize shutdown handler.
        
        Args:
            remote_executor: Connected RemoteExecutor instance
            lambda_api_key: Lambda Cloud API key for instance termination
        """
        self.remote_executor = remote_executor
        self.lambda_api_key = lambda_api_key
        self.s3_client = None
    
    def setup_aws_credentials(self) -> bool:
        """Setup AWS credentials on the remote instance for S3 sync.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use default AWS credential chain (environment variables, IAM roles, etc.)
            self.s3_client = boto3.client('s3')
            
            # Test AWS credentials by listing S3 buckets
            self.s3_client.list_buckets()
            
            # Setup AWS CLI on remote instance using temporary credentials
            session = boto3.Session()
            credentials = session.get_credentials()
            
            if not credentials:
                print("✗ No AWS credentials found locally")
                return False
            
            # Configure AWS CLI on remote instance
            aws_config_commands = [
                f"aws configure set aws_access_key_id {credentials.access_key}",
                f"aws configure set aws_secret_access_key {credentials.secret_key}",
                "aws configure set region us-west-2",  # Default region
                "aws configure set output json"
            ]
            
            if credentials.token:
                aws_config_commands.append(f"aws configure set aws_session_token {credentials.token}")
            
            for cmd in aws_config_commands:
                result = self.remote_executor.execute_command(cmd)
                if not result.success:
                    print(f"✗ Failed to configure AWS CLI: {result.stderr}")
                    return False
            
            print("✓ AWS credentials configured on remote instance")
            return True
            
        except Exception as e:
            print(f"✗ Failed to setup AWS credentials: {e}")
            return False
    
    def sync_output_to_s3(self, remote_output_dir: str, s3_bucket: str, s3_key_prefix: str) -> bool:
        """Sync training output from remote instance to S3.
        
        Args:
            remote_output_dir: Remote directory containing training output
            s3_bucket: S3 bucket name
            s3_key_prefix: S3 key prefix for uploaded files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create S3 bucket if it doesn't exist
            try:
                self.s3_client.head_bucket(Bucket=s3_bucket)
            except:
                print(f"Creating S3 bucket: {s3_bucket}")
                self.s3_client.create_bucket(
                    Bucket=s3_bucket,
                    CreateBucketConfiguration={'LocationConstraint': 'us-west-2'}
                )
            
            # Sync output directory to S3 using AWS CLI on remote instance
            sync_command = f"aws s3 sync {remote_output_dir} s3://{s3_bucket}/{s3_key_prefix}/ --delete"
            
            print(f"Syncing {remote_output_dir} to s3://{s3_bucket}/{s3_key_prefix}/")
            result = self.remote_executor.execute_command(sync_command, timeout=1800)  # 30 minute timeout
            
            if result.success:
                print("✓ Successfully synced output to S3")
                print(f"  Output available at: s3://{s3_bucket}/{s3_key_prefix}/")
                return True
            else:
                print(f"✗ S3 sync failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"✗ S3 sync error: {e}")
            return False
    
    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate the Lambda Cloud instance.
        
        Args:
            instance_id: Lambda Cloud instance ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import lambda_cloud here to avoid import issues
            from common.tools.lambda_cloud.lambda_api.client import LambdaCloudClient
            
            client = LambdaCloudClient(api_key=self.lambda_api_key)
            client.terminate_instance(instance_id)
            
            print(f"✓ Initiated termination of instance {instance_id}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to terminate instance {instance_id}: {e}")
            return False
    
    def get_instance_id_by_ip(self, instance_ip: str) -> Optional[str]:
        """Get instance ID by IP address.
        
        Args:
            instance_ip: IP address of the instance
            
        Returns:
            Instance ID if found, None otherwise
        """
        try:
            from common.tools.lambda_cloud.lambda_api.client import LambdaCloudClient
            
            client = LambdaCloudClient(api_key=self.lambda_api_key)
            instances = client.list_instances()
            
            for instance in instances:
                if instance.ip == instance_ip:
                    return instance.id
            
            print(f"✗ No instance found with IP {instance_ip}")
            return None
            
        except Exception as e:
            print(f"✗ Failed to find instance by IP: {e}")
            return None
    
    def create_shutdown_script(self, 
                             remote_output_dir: str,
                             s3_bucket: str, 
                             s3_key_prefix: str,
                             instance_ip: str,
                             max_train_time_hours: int) -> str:
        """Create a shutdown script that will run on the remote instance.
        
        Args:
            remote_output_dir: Directory containing training output
            s3_bucket: S3 bucket for output sync  
            s3_key_prefix: S3 key prefix
            instance_ip: Instance IP address
            max_train_time_hours: Maximum training time in hours
            
        Returns:
            Path to the created shutdown script on remote instance
        """
        shutdown_script_content = f'''#!/bin/bash

# Shutdown script for Lambda Cloud training job
set -e

echo "Starting shutdown procedure..."

# Sync output to S3
echo "Syncing output to S3..."
aws s3 sync {remote_output_dir} s3://{s3_bucket}/{s3_key_prefix}/ --delete

# Get instance ID by IP and terminate
echo "Looking up instance ID..."
INSTANCE_ID=$(python3 -c "
from lambda_cloud import LambdaCloudClient
import os

client = LambdaCloudClient(api_key=os.environ['LAMBDA_API_KEY'])
instances = client.list_instances()

for instance in instances:
    if instance.ip == '{instance_ip}':
        print(instance.id)
        break
")

if [ -n "$INSTANCE_ID" ]; then
    echo "Terminating instance $INSTANCE_ID..."
    python3 -c "
from lambda_cloud import LambdaCloudClient
import os

client = LambdaCloudClient(api_key=os.environ['LAMBDA_API_KEY'])  
client.terminate_instance('$INSTANCE_ID')
"
    echo "Instance termination initiated"
else
    echo "Failed to find instance ID for IP {instance_ip}"
fi

echo "Shutdown procedure completed"
'''
        
        # Write script to remote instance
        script_path = "/tmp/shutdown_script.sh"
        
        try:
            # Create the script file using echo commands
            commands = [
                f"cat > {script_path} << 'EOF'",
                shutdown_script_content,
                "EOF",
                f"chmod +x {script_path}"
            ]
            
            for cmd in commands:
                result = self.remote_executor.execute_command(cmd)
                if not result.success and "EOF" not in cmd:
                    print(f"✗ Failed to create shutdown script: {result.stderr}")
                    return ""
            
            print(f"✓ Created shutdown script at {script_path}")
            return script_path
            
        except Exception as e:
            print(f"✗ Failed to create shutdown script: {e}")
            return ""