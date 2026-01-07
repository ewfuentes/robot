#!/usr/bin/env python3
"""Vertex AI Batch Manager - Manage batch jobs using Vertex AI API.

This script manages batch inference jobs using Vertex AI instead of the
Gemini Developer API. It works with files stored in Google Cloud Storage.

Prerequisites:
    1. Set environment variables:
       export GOOGLE_CLOUD_PROJECT=your-project-id
       export GOOGLE_CLOUD_LOCATION=us-central1  # or your preferred location
       export GOOGLE_GENAI_USE_VERTEXAI=True

    2. Authenticate with gcloud:
       gcloud auth application-default login

Example usage:
    # Submit all JSONL files from a list
    # First create a text file with GCS URIs (one per line):
    # gs://bucket/requests/file1.jsonl
    # gs://bucket/requests/file2.jsonl
    bazel run //experimental/overhead_matching/swag/scripts:vertex_batch_manager -- \\
        submit-all \\
        --file_list /path/to/file_list.txt \\
        --output_prefix gs://your-bucket/output-results/ \\
        --model gemini-2.5-flash

    # List all active batch jobs
    bazel run //experimental/overhead_matching/swag/scripts:vertex_batch_manager -- \\
        list --active

    # Get status of a specific job
    bazel run //experimental/overhead_matching/swag/scripts:vertex_batch_manager -- \\
        status --job_name projects/.../batchPredictionJobs/123
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions

try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False


# Job states
ACTIVE_STATES = {
    JobState.JOB_STATE_PENDING,
    JobState.JOB_STATE_RUNNING,
}

COMPLETED_STATES = {
    JobState.JOB_STATE_SUCCEEDED,
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_PAUSED,
}


def check_environment():
    """Check that required environment variables are set."""
    required_vars = [
        'GOOGLE_CLOUD_PROJECT',
        'GOOGLE_CLOUD_LOCATION',
        'GOOGLE_GENAI_USE_VERTEXAI'
    ]

    missing = []
    for var in required_vars:
        if not os.environ.get(var):
            missing.append(var)

    if missing:
        print("Error: Missing required environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nPlease set them with:")
        print("  export GOOGLE_CLOUD_PROJECT=your-project-id")
        print("  export GOOGLE_CLOUD_LOCATION=us-central1")
        print("  export GOOGLE_GENAI_USE_VERTEXAI=True")
        sys.exit(1)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse GCS URI into bucket and prefix.

    Args:
        uri: GCS URI like gs://bucket/path/to/files/

    Returns:
        Tuple of (bucket_name, prefix)
    """
    if not uri.startswith('gs://'):
        raise ValueError(f"Invalid GCS URI: {uri}. Must start with gs://")

    uri = uri[5:]  # Remove gs://
    if '/' in uri:
        bucket, prefix = uri.split('/', 1)
    else:
        bucket = uri
        prefix = ''

    return bucket, prefix


def list_gcs_jsonl_files(bucket_name: str, prefix: str) -> List[str]:
    """List all JSONL files in a GCS bucket/prefix.

    Args:
        bucket_name: GCS bucket name
        prefix: Prefix/directory path

    Returns:
        List of GCS URIs (gs://bucket/path/file.jsonl)
    """
    if not HAS_GCS:
        print("Error: google-cloud-storage is not installed")
        print("Install with: pip install google-cloud-storage")
        sys.exit(1)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Ensure prefix ends with / if it's not empty
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    # List all blobs with the prefix
    blobs = bucket.list_blobs(prefix=prefix)

    jsonl_files = []
    for blob in blobs:
        if blob.name.endswith('.jsonl'):
            jsonl_files.append(f"gs://{bucket_name}/{blob.name}")

    return sorted(jsonl_files)


def get_output_uri(input_uri: str, output_prefix: str) -> str:
    """Generate output URI for a given input file.

    Args:
        input_uri: Input file GCS URI (gs://bucket/input/file.jsonl)
        output_prefix: Output prefix (gs://bucket/output/)

    Returns:
        Output directory URI for this file
    """
    # Extract filename from input
    filename = input_uri.split('/')[-1]
    basename = filename.replace('.jsonl', '')

    # Ensure output_prefix ends with /
    if not output_prefix.endswith('/'):
        output_prefix += '/'

    return f"{output_prefix}{basename}/"


def cmd_submit_all(args):
    """Submit batch jobs for all JSONL files in a GCS directory."""
    check_environment()

    # Parse input prefix
    print(f"Scanning for JSONL files in: {args.input_prefix}")
    try:
        bucket_name, prefix = parse_gcs_uri(args.input_prefix)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # List all JSONL files
    print(f"Listing files in gs://{bucket_name}/{prefix}")
    jsonl_files = list_gcs_jsonl_files(bucket_name, prefix)

    if not jsonl_files:
        print(f"No JSONL files found in {args.input_prefix}")
        return

    print(f"\nFound {len(jsonl_files)} JSONL file(s):")
    for f in jsonl_files:
        print(f"  - {f}")

    if args.dry_run:
        print("\nDRY RUN - No jobs were submitted.")
        return

    # Confirm submission unless --force
    if not args.force:
        response = input(f"\nSubmit {len(jsonl_files)} batch job(s)? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Initialize Vertex AI client
    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    # Submit jobs
    print(f"\nSubmitting batch jobs using model: {args.model}")
    print("=" * 80)

    submitted_jobs = []
    errors = []

    for input_uri in jsonl_files:
        try:
            # Generate output URI
            output_uri = get_output_uri(input_uri, args.output_prefix)

            print(f"\nSubmitting: {input_uri.split('/')[-1]}")
            print(f"  Input:  {input_uri}")
            print(f"  Output: {output_uri}")

            # Create batch job
            job = client.batches.create(
                model=args.model,
                src=input_uri,
                config=CreateBatchJobConfig(dest=output_uri),
            )

            print(f"  ✓ Job created: {job.name}")
            print(f"    State: {job.state}")
            submitted_jobs.append((input_uri, job.name))

            # Brief pause to avoid rate limiting
            if args.delay > 0:
                time.sleep(args.delay)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            errors.append((input_uri, str(e)))

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Submitted: {len(submitted_jobs)}/{len(jsonl_files)} job(s)")
    print(f"Errors: {len(errors)}")

    if submitted_jobs:
        print("\nSubmitted jobs:")
        for input_uri, job_name in submitted_jobs:
            print(f"  {input_uri.split('/')[-1]} -> {job_name}")

    if errors:
        print("\nErrors:")
        for input_uri, error in errors:
            print(f"  {input_uri.split('/')[-1]}: {error}")

    print("\nCheck job status with:")
    print(f"  bazel run //experimental/overhead_matching/swag/scripts:vertex_batch_manager -- list")


def cmd_list(args):
    """List batch jobs."""
    check_environment()

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    print("Fetching batch jobs...")

    # List all jobs
    # Note: The Python SDK's batches.list() returns jobs for the current project/location
    jobs = list(client.batches.list())

    if not jobs:
        print("No batch jobs found.")
        return

    # Apply filters
    filtered = jobs
    if args.active:
        filtered = [j for j in filtered if j.state in ACTIVE_STATES]
    elif args.completed:
        filtered = [j for j in filtered if j.state in COMPLETED_STATES]
    elif args.succeeded:
        filtered = [j for j in filtered if j.state == JobState.JOB_STATE_SUCCEEDED]
    elif args.failed:
        filtered = [j for j in filtered if j.state == JobState.JOB_STATE_FAILED]

    if not filtered:
        print("No jobs match the filter criteria.")
        return

    # Print results
    print(f"\nFound {len(filtered)} batch job(s):")
    print("=" * 120)
    print(f"{'Job Name':<70} {'State':<25} {'Model':<25}")
    print("=" * 120)

    for job in sorted(filtered, key=lambda j: j.create_time or ''):
        job_name = job.name
        state = str(job.state).replace('JobState.', '')
        model = job.model if hasattr(job, 'model') else 'N/A'

        # Truncate long job names
        if len(job_name) > 67:
            job_name = "..." + job_name[-64:]

        print(f"{job_name:<70} {state:<25} {model:<25}")

        if args.verbose:
            print(f"  Created: {job.create_time or 'N/A'}")
            if hasattr(job, 'update_time'):
                print(f"  Updated: {job.update_time or 'N/A'}")
            if hasattr(job, 'input_config') and job.input_config:
                print(f"  Input: {job.input_config}")
            if hasattr(job, 'output_config') and job.output_config:
                print(f"  Output: {job.output_config}")
            print()

    print("=" * 120)
    print(f"Total: {len(filtered)} job(s)")

    # Summary statistics
    if args.stats:
        print("\nStatistics by state:")
        state_counts = {}
        for job in filtered:
            state_str = str(job.state).replace('JobState.', '')
            state_counts[state_str] = state_counts.get(state_str, 0) + 1

        for state, count in sorted(state_counts.items()):
            print(f"  {state}: {count}")


def cmd_status(args):
    """Get status of a specific batch job."""
    check_environment()

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    try:
        job = client.batches.get(name=args.job_name)

        print("\nBatch Job Details:")
        print("=" * 80)
        print(f"Job Name: {job.name}")
        print(f"State: {job.state}")
        print(f"Model: {job.model if hasattr(job, 'model') else 'N/A'}")
        print(f"Created: {job.create_time or 'N/A'}")
        print(f"Updated: {job.update_time or 'N/A'}")

        # Input/output info
        if hasattr(job, 'input_config') and job.input_config:
            print(f"\nInput Configuration:")
            print(f"  {job.input_config}")

        if hasattr(job, 'output_config') and job.output_config:
            print(f"\nOutput Configuration:")
            print(f"  {job.output_config}")

        # Error info
        if job.state == JobState.JOB_STATE_FAILED:
            if hasattr(job, 'error') and job.error:
                print(f"\nError:")
                print(f"  {job.error}")

        print("=" * 80)

        # Suggest next actions
        if job.state == JobState.JOB_STATE_SUCCEEDED:
            print("\nJob completed successfully!")
            if hasattr(job, 'output_config') and job.output_config:
                print(f"Results are available at: {job.output_config}")
        elif job.state in ACTIVE_STATES:
            print(f"\nJob is still {job.state}. Check again later.")

    except Exception as e:
        print(f"Error getting job status: {e}")
        sys.exit(1)


def cmd_cancel(args):
    """Cancel a batch job."""
    check_environment()

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    try:
        # Get current job state
        job = client.batches.get(name=args.job_name)

        if job.state not in ACTIVE_STATES:
            print(f"Job is not active (current state: {job.state})")
            print("Only PENDING or RUNNING jobs can be cancelled.")
            return

        if not args.force:
            response = input(f"Cancel job {args.job_name}? [y/N]: ")
            if response.lower() != 'y':
                print("Cancelled.")
                return

        # Cancel the job
        client.batches.cancel(name=args.job_name)
        print(f"Job cancelled: {args.job_name}")

    except Exception as e:
        print(f"Error cancelling job: {e}")
        sys.exit(1)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vertex AI Batch Manager - Manage batch jobs using Vertex AI API",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # SUBMIT-ALL command
    submit_parser = subparsers.add_parser(
        'submit-all',
        help='Submit batch jobs for all JSONL files in a GCS directory'
    )
    submit_parser.add_argument(
        '--input_prefix',
        type=str,
        required=True,
        help='GCS directory containing JSONL files (e.g., gs://bucket/requests/)'
    )
    submit_parser.add_argument(
        '--output_prefix',
        type=str,
        required=True,
        help='GCS output prefix for results (e.g., gs://bucket/output/)'
    )
    submit_parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.5-flash',
        help='Model name (default: gemini-2.5-flash)'
    )
    submit_parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay in seconds between submissions (default: 1.0)'
    )
    submit_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be submitted without submitting'
    )
    submit_parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt'
    )

    # LIST command
    list_parser = subparsers.add_parser(
        'list',
        help='List batch jobs'
    )
    list_parser.add_argument('--active', action='store_true', help='Show only active jobs')
    list_parser.add_argument('--completed', action='store_true', help='Show only completed jobs')
    list_parser.add_argument('--succeeded', action='store_true', help='Show only succeeded jobs')
    list_parser.add_argument('--failed', action='store_true', help='Show only failed jobs')
    list_parser.add_argument('--verbose', action='store_true', help='Show detailed information')
    list_parser.add_argument('--stats', action='store_true', help='Show summary statistics')

    # STATUS command
    status_parser = subparsers.add_parser(
        'status',
        help='Get status of a specific batch job'
    )
    status_parser.add_argument(
        '--job_name',
        type=str,
        required=True,
        help='Job name (e.g., projects/.../batchPredictionJobs/123)'
    )

    # CANCEL command
    cancel_parser = subparsers.add_parser(
        'cancel',
        help='Cancel a batch job'
    )
    cancel_parser.add_argument(
        '--job_name',
        type=str,
        required=True,
        help='Job name to cancel'
    )
    cancel_parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to command handler
    if args.command == 'submit-all':
        cmd_submit_all(args)
    elif args.command == 'list':
        cmd_list(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'cancel':
        cmd_cancel(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == '__main__':
    main()
