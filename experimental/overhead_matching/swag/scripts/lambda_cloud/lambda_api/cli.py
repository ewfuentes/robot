#!/usr/bin/env python3
"""Command-line interface for Lambda Cloud API."""

import argparse
import sys
import json
import logging
import time
from typing import List

# Allow running directly as a script
from experimental.overhead_matching.swag.scripts.lambda_cloud.lambda_api.client import LambdaCloudClient
from experimental.overhead_matching.swag.scripts.lambda_cloud.lambda_api.models import LambdaCloudError, InsufficientCapacityError


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(message)s')


def print_json(data):
    """Print data as formatted JSON."""
    print(json.dumps(data, indent=2, default=str))


def print_table(headers: List[str], rows: List[List[str]]):
    """Print data in a simple table format."""
    if not rows:
        return
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Print header
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_row)
    print("-" * len(header_row))
    
    # Print rows
    for row in rows:
        print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))


def cmd_list_types(args, client: LambdaCloudClient):
    """List available instance types."""
    try:
        instance_types = client.list_instance_types()
        
        if args.json:
            data = []
            for it in instance_types:
                data.append({
                    "name": it.name,
                    "description": it.description,
                    "price_cents_per_hour": it.price_cents_per_hour,
                    "price_per_hour": f"${it.price_cents_per_hour / 100:.2f}",
                    "vcpus": it.vcpus,
                    "memory_gib": it.memory_gib,
                    "storage_gib": it.storage_gib,
                    "regions": it.regions,
                    "specs": it.specs
                })
            print_json(data)
        else:
            # Filter by region if specified
            if args.region:
                instance_types = [it for it in instance_types if args.region in it.regions]
            
            # Filter by GPU type if specified
            if args.gpu_type:
                instance_types = [it for it in instance_types
                                  if args.gpu_type.lower() in it.name.lower()]
            
            headers = ["Name", "Price/hour", "vCPUs", "Memory (GiB)", "Storage (GiB)", "Available Regions"]
            rows = []
            for it in instance_types:
                regions_str = ", ".join(it.regions[:3])
                if len(it.regions) > 3:
                    regions_str += f" (+{len(it.regions) - 3} more)"
                
                rows.append([
                    it.name,
                    f"${it.price_cents_per_hour / 100:.2f}",
                    str(it.vcpus),
                    str(it.memory_gib),
                    str(it.storage_gib),
                    regions_str
                ])
            
            print_table(headers, rows)
            
    except LambdaCloudError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list_instances(args, client: LambdaCloudClient):
    """List running instances."""
    try:
        instances = client.list_instances()
        
        if args.json:
            data = []
            for instance in instances:
                data.append({
                    "id": instance.id,
                    "name": instance.name,
                    "ip": instance.ip,
                    "private_ip": instance.private_ip,
                    "status": instance.status.value,
                    "region": instance.region,
                    "instance_type": instance.instance_type,
                    "hostname": instance.hostname,
                    "ssh_key_names": instance.ssh_key_names
                })
            print_json(data)
        else:
            headers = ["ID", "Name", "Status", "Type", "Region", "IP Address"]
            rows = []
            for instance in instances:
                rows.append([
                    instance.id,
                    instance.name,
                    instance.status.value,
                    instance.instance_type,
                    instance.region,
                    instance.ip
                ])
            
            print_table(headers, rows)
            
    except LambdaCloudError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_launch(args, client: LambdaCloudClient):
    """Launch instances."""
    def status_callback(message: str):
        if not args.quiet:
            print(f"[{message}]", file=sys.stderr)
    
    try:
        # Use intelligent retry by default
        instance_ids = client.launch_instance_with_retry(
            region_name=args.region,
            instance_type_name=args.type,
            ssh_key_names=args.ssh_keys,
            name=args.name,
            file_system_names=args.file_systems,
            quantity=args.quantity,
            max_retries=args.max_retries,
            callback=status_callback if not args.quiet else None,
            image_family=args.image_family
        )
        
        if args.json:
            print_json({"instance_ids": instance_ids})
        else:
            print(f"Successfully launched {len(instance_ids)} instance(s):")
            for instance_id in instance_ids:
                print(f"  {instance_id}")
        
        # If user wants to wait for instances to be active
        if args.wait:
            if not args.quiet:
                print("Waiting for instances to become active...", file=sys.stderr)

            active_instances = []
            max_wait_time = 300  # 5 minutes
            start_time = time.time()
            
            while len(active_instances) < len(instance_ids) and time.time() - start_time < max_wait_time:
                for instance_id in instance_ids:
                    if instance_id not in [i.id for i in active_instances]:
                        try:
                            instance = client.get_instance(instance_id)
                            if instance.status.value == "active":
                                active_instances.append(instance)
                                if not args.quiet:
                                    print(f"Instance {instance_id} is now active: {instance.ip}", file=sys.stderr)
                        except Exception:
                            pass
                
                if len(active_instances) < len(instance_ids):
                    time.sleep(10)
            
            if len(active_instances) < len(instance_ids):
                timeout_msg = (f"Warning: Only {len(active_instances)}/"
                               f"{len(instance_ids)} instances became active within timeout")
                print(timeout_msg, file=sys.stderr)

    except InsufficientCapacityError as e:
        print(f"Error: {e.message}", file=sys.stderr)
        if e.suggestion:
            print(f"Suggestion: {e.suggestion}", file=sys.stderr)
        sys.exit(1)
    except LambdaCloudError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_terminate(args, client: LambdaCloudClient):
    """Terminate instances."""
    try:
        if not args.force:
            # Confirm termination
            instance_list = ", ".join(args.instance_ids)
            response = input(f"Are you sure you want to terminate instance(s) {instance_list}? [y/N]: ")
            if response.lower() != 'y':
                print("Termination cancelled.")
                return
        
        terminated = client.terminate_instances(args.instance_ids)
        
        if args.json:
            data = []
            for instance in terminated:
                data.append({
                    "id": instance.id,
                    "name": instance.name,
                    "status": instance.status.value
                })
            print_json(data)
        else:
            print(f"Successfully terminated {len(terminated)} instance(s):")
            for instance in terminated:
                print(f"  {instance.id} ({instance.name})")
                
    except LambdaCloudError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_ssh(args, client: LambdaCloudClient):
    """Get SSH connection information."""
    try:
        ssh_info = client.get_ssh_info(args.instance_id)
        
        if args.json:
            data = {
                "instance_id": ssh_info.instance_id,
                "instance_name": ssh_info.instance_name,
                "ip": ssh_info.ip,
                "hostname": ssh_info.hostname,
                "ssh_key_names": ssh_info.ssh_key_names,
                "username": ssh_info.username,
                "ssh_command": f"ssh -i <private_key_file> {ssh_info.username}@{ssh_info.ip}"
            }
            print_json(data)
        else:
            print(f"SSH Information for instance {ssh_info.instance_id} ({ssh_info.instance_name}):")
            print(f"  IP Address: {ssh_info.ip}")
            print(f"  Hostname: {ssh_info.hostname}")
            print(f"  Username: {ssh_info.username}")
            print(f"  SSH Keys: {', '.join(ssh_info.ssh_key_names)}")
            print()
            print("To connect via SSH:")
            print(f"  ssh -i <private_key_file> {ssh_info.username}@{ssh_info.ip}")
            print()
            print("Replace <private_key_file> with the path to your private key corresponding to one of the SSH keys above.")
            
    except LambdaCloudError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Lambda Cloud API CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # list-types command
    list_types_parser = subparsers.add_parser("list-types", help="List available instance types")
    list_types_parser.add_argument("--region", help="Filter by region")
    list_types_parser.add_argument("--gpu-type", help="Filter by GPU type (e.g., 'a100', 'h100')")
    
    # list-instances command
    subparsers.add_parser("list-instances", help="List running instances")
    
    # launch command
    launch_parser = subparsers.add_parser("launch", help="Launch instances")
    launch_parser.add_argument("--region", "-r", required=True, help="Region to launch in")
    launch_parser.add_argument("--type", "-t", required=True, help="Instance type to launch")
    launch_parser.add_argument("--ssh-keys", nargs="+", required=True, help="SSH key names")
    launch_parser.add_argument("--name", "-n", required=True, help="Name for the instance")
    launch_parser.add_argument("--file-systems", nargs="*", help="File system names to mount")
    launch_parser.add_argument("--quantity", "-q", type=int, default=1, help="Number of instances to launch")
    launch_parser.add_argument("--max-retries", type=int, default=10, help="Maximum retry attempts for capacity issues")
    launch_parser.add_argument("--wait", "-w", action="store_true", help="Wait for instances to become active")
    launch_parser.add_argument("--quiet", action="store_true", help="Suppress status messages")
    launch_parser.add_argument("--image-family", help="Image family name to use")
    
    # terminate command
    terminate_parser = subparsers.add_parser("terminate", help="Terminate instances")
    terminate_parser.add_argument("instance_ids", nargs="+", help="Instance IDs to terminate")
    terminate_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")
    
    # ssh command
    ssh_parser = subparsers.add_parser("ssh", help="Get SSH connection information")
    ssh_parser.add_argument("instance_id", help="Instance ID")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    setup_logging(args.verbose)
    
    try:
        client = LambdaCloudClient()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please set the LAMBDA_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)
    
    # Route to appropriate command handler
    if args.command == "list-types":
        cmd_list_types(args, client)
    elif args.command == "list-instances":
        cmd_list_instances(args, client)
    elif args.command == "launch":
        cmd_launch(args, client)
    elif args.command == "terminate":
        cmd_terminate(args, client)
    elif args.command == "ssh":
        cmd_ssh(args, client)


if __name__ == "__main__":
    main()