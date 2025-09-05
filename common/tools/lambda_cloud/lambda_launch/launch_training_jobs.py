#!/usr/bin/env python3
"""CLI for launching Lambda Cloud training jobs."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from common.tools.lambda_cloud.lambda_launch.launcher import LambdaTrainingLauncher


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch training jobs on Lambda Cloud instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch single job with default branch
  %(prog)s --config configs/train.yaml --machine-config machine_setup.yaml
  
  # Launch multiple jobs with different branches  
  %(prog)s --config config1.yaml config2.yaml --branches main experiment --machine-config setup.yaml
  
  # Launch jobs from a file
  %(prog)s --job-file jobs.txt --machine-config setup.yaml
  
  # Set custom output directory and parallelism
  %(prog)s --config configs/train.yaml --machine-config setup.yaml --output-dir ./results --max-parallel 5

Job file format (CSV):
  config_path
  config_path,branch
  path/to/config1.yaml,main
  path/to/config2.yaml,experiment
  path/to/config3.yaml
        """
    )
    
    # Input configuration
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--config", "-c",
        nargs="+",
        help="Path(s) to training configuration YAML file(s)"
    )
    input_group.add_argument(
        "--job-file", "-j", 
        help="Path to file containing job configurations (CSV format: config_path[,branch])"
    )
    
    # Branch specification (only used with --config)
    parser.add_argument(
        "--branches", "-b",
        nargs="+",
        help="Git branch(es) for each config. Defaults to 'main'. Must match number of configs or be single branch for all"
    )
    
    # Machine configuration
    parser.add_argument(
        "--machine-config", "-m",
        required=True,
        help="Path to machine setup configuration YAML file"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for logs and results (default: auto-generated in machine config directory)"
    )
    
    # Execution configuration
    parser.add_argument(
        "--max-parallel", "-p",
        type=int,
        default=10,
        help="Maximum number of parallel jobs (default: 10)"
    )
    
    # Debugging
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse configurations and validate but don't launch instances"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check machine config file exists
    if not Path(args.machine_config).exists():
        print(f"Error: Machine config file not found: {args.machine_config}")
        sys.exit(1)
    
    # Validate config arguments
    if args.config:
        # Check all config files exist
        for config_path in args.config:
            if not Path(config_path).exists():
                print(f"Error: Training config file not found: {config_path}")
                sys.exit(1)
        
        # Validate branch arguments
        if args.branches:
            if len(args.branches) != 1 and len(args.branches) != len(args.config):
                print(f"Error: Number of branches ({len(args.branches)}) must be 1 or match number of configs ({len(args.config)})")
                sys.exit(1)
    
    # Check job file exists
    if args.job_file and not Path(args.job_file).exists():
        print(f"Error: Job file not found: {args.job_file}")
        sys.exit(1)
    
    # Validate parallel job count
    if args.max_parallel < 1:
        print(f"Error: max-parallel must be at least 1")
        sys.exit(1)
    
    # Check for branches argument with job file (not supported)
    if args.job_file and args.branches:
        print("Error: --branches cannot be used with --job-file (specify branches in the job file)")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_arguments(args)
        
        if args.verbose:
            print("Arguments:")
            for key, value in vars(args).items():
                print(f"  {key}: {value}")
            print()
        
        # Create launcher
        try:
            launcher = LambdaTrainingLauncher(
                machine_config_path=args.machine_config,
                output_dir=args.output_dir,
                max_parallel_jobs=args.max_parallel
            )
        except Exception as e:
            print(f"Error initializing launcher: {e}")
            sys.exit(1)
        
        # Dry run mode
        if args.dry_run:
            print("🧪 DRY RUN MODE - No instances will be launched")
            
            # Parse job configurations to validate
            try:
                if args.config:
                    from common.tools.lambda_cloud.lambda_launch.config import ConfigParser
                    job_configs = ConfigParser.parse_job_configs(args.config, args.branches)
                    print(f"✓ Successfully parsed {len(job_configs)} job configurations")
                elif args.job_file:
                    from common.tools.lambda_cloud.lambda_launch.config import ConfigParser
                    job_configs = ConfigParser.parse_job_file(args.job_file)
                    print(f"✓ Successfully parsed {len(job_configs)} job configurations from file")
                
                print("✓ Machine configuration valid")
                print("✓ All configurations validated successfully")
                
                print("\nJob configurations:")
                for i, job_config in enumerate(job_configs):
                    print(f"  {i+1}. {job_config.config_path} (branch: {job_config.branch})")
                
                return
                
            except Exception as e:
                print(f"✗ Configuration validation failed: {e}")
                sys.exit(1)
        
        # Launch jobs
        try:
            if args.config:
                results = launcher.launch_jobs_from_configs(
                    config_paths=args.config,
                    branches=args.branches
                )
            elif args.job_file:
                results = launcher.launch_jobs_from_file(args.job_file)
            
            # Exit with error code if any jobs failed
            failed_count = len([r for r in results if r.status.value == "failed"])
            if failed_count > 0:
                print(f"\n⚠️  {failed_count} jobs failed")
                sys.exit(1)
            else:
                print(f"\n🎉 All jobs completed successfully!")
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted by user")
            print(f"   Some instances may still be running and need manual cleanup")
            sys.exit(1)
        except Exception as e:
            print(f"Error launching jobs: {e}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()