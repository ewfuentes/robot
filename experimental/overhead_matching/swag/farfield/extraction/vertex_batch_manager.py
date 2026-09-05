#!/usr/bin/env python3
"""Vertex execution transport used by immutable farfield LLM stages.

New provider work enters through run_requests, where the stage-owned model,
cost approval, immutable request snapshot, and append-only attempts are
already established. The command-line interface is deliberately limited to
observing or cancelling existing jobs; raw submit/run commands would bypass
those contracts.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import List

from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions

from experimental.overhead_matching.swag.farfield.extraction import (
    llm_cost,
    prompts,
)

try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False


# Job states
ACTIVE_STATES = {
    JobState.JOB_STATE_QUEUED,
    JobState.JOB_STATE_PENDING,
    JobState.JOB_STATE_RUNNING,
    JobState.JOB_STATE_CANCELLING,
}

COMPLETED_STATES = {
    JobState.JOB_STATE_SUCCEEDED,
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_PAUSED,
}

_SUBMISSION_REQUEST_RE = re.compile(r"requests_submit_(\d+)\.jsonl\Z")


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


def cmd_run_online(args):
    """Run batch requests online via live Vertex AI API."""
    check_environment()

    import json
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    # Load requests
    records = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} requests from {args.input}")

    output_path = Path(args.output)
    errors_path = output_path.with_suffix('.errors.jsonl')
    for path in (output_path, errors_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"transport output already exists; request lifecycle must "
                f"allocate a new attempt path: {path}")
    if not records:
        raise ValueError("request transport input is empty")

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    thinking_cfg = records[0]["request"]["generationConfig"].get("thinkingConfig", {})
    thinking_level = thinking_cfg.get("thinkingLevel", "none")
    print(f"Model: {args.model}, thinking: {thinking_level}, parallel: {args.parallel}")
    print(f"Processing {len(records)} requests...")

    MAX_CONSECUTIVE_ERRORS = 3
    total_prompt = 0
    total_output = 0
    total_thinking = 0
    completed = 0
    errors = 0
    consecutive_errors = 0
    stop_early = False
    print_lock = Lock()
    start_time = time.time()
    out_file = open(output_path, 'x')
    errors_file = open(errors_path, 'x')

    def process_one(record):
        req = record["request"]
        try:
            adapted = prompts.online_request_from_batch(record["key"], req)
            response = client.models.generate_content(
                model=args.model,
                contents=adapted["contents"],
                config=adapted["config"],
            )
            return {
                "key": record["key"],
                "response": {
                    "candidates": [{"content": {"parts": [{"text": response.text}], "role": "model"}}],
                    "usageMetadata": {
                        "promptTokenCount": response.usage_metadata.prompt_token_count,
                        "candidatesTokenCount": response.usage_metadata.candidates_token_count,
                        "thoughtsTokenCount": getattr(response.usage_metadata, 'thoughts_token_count', 0) or 0,
                        "totalTokenCount": response.usage_metadata.total_token_count,
                    },
                },
            }
        except Exception as e:
            return {
                "key": record["key"],
                "error": f"{type(e).__name__}: {e}",
            }

    def handle_result(result):
        nonlocal total_prompt, total_output, total_thinking, completed, errors
        nonlocal consecutive_errors, stop_early
        with print_lock:
            if "error" in result:
                errors += 1
                consecutive_errors += 1
                print(f"  ERROR {result['key']}: {result['error']}")
                errors_file.write(json.dumps(result) + '\n')
                errors_file.flush()
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"  STOPPING: {consecutive_errors} consecutive errors, likely quota/auth issue")
                    stop_early = True
            else:
                consecutive_errors = 0
                usage = result["response"]["usageMetadata"]
                total_prompt += usage["promptTokenCount"]
                total_output += usage["candidatesTokenCount"]
                total_thinking += usage.get("thoughtsTokenCount", 0)
                completed += 1
                out_file.write(json.dumps(result) + '\n')
                out_file.flush()

            elapsed = time.time() - start_time
            total_done = completed + errors
            rate = total_done / elapsed if elapsed > 0 else 0
            remaining = (len(records) - total_done) / rate if rate > 0 else 0
            print(f"  [{total_done}/{len(records)}] {result['key']} "
                  f"({rate:.1f}/s, ~{remaining:.0f}s remaining)")

    try:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(process_one, rec): rec["key"] for rec in records}
            for future in as_completed(futures):
                handle_result(future.result())
                if stop_early:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        out_file.close()
        errors_file.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed: {completed}/{len(records)} ({errors} errors) in {elapsed:.1f}s")
    print(f"Token usage:")
    print(f"  Prompt:   {total_prompt:,}")
    print(f"  Output:   {total_output:,}")
    print(f"  Thinking: {total_thinking:,}")
    print(f"  Total:    {total_prompt + total_output + total_thinking:,}")
    print(f"Output: {args.output}")


def _batch_stage_uri(gcs_prefix: str, tag: str,
                     submission_id: str | None = None) -> tuple[str, str]:
    """Unique request/result locations for one batch submission."""
    prefix = gcs_prefix.rstrip('/')
    submission_id = submission_id or uuid.uuid4().hex
    base = f"{prefix}/{tag}/submissions/{submission_id}"
    return f"{base}/requests.jsonl", f"{base}/results/"


def next_submission_paths(work_dir: Path | str) -> tuple[int, Path, Path]:
    """Paths for a fresh stage submission round.

    The immutable request shard is the durable reservation. Counting it,
    rather than completed provider output, ensures a retry gets a new local raw
    path even when a batch is interrupted before it can download any results.
    """
    work_dir = Path(work_dir)
    indices = set()
    if work_dir.exists():
        for entry in work_dir.iterdir():
            match = _SUBMISSION_REQUEST_RE.fullmatch(entry.name)
            if match:
                indices.add(int(match.group(1)))
    round_index = max(indices, default=0) + 1
    return (
        round_index,
        work_dir / f"requests_submit_{round_index:04d}.jsonl",
        work_dir / f"transport_submit_{round_index:04d}.jsonl",
    )


def completed_submission_results(work_dir: Path | str) -> tuple[Path, ...]:
    """Existing main/error transport shards for allocated submission rounds."""
    work_dir = Path(work_dir)
    rounds = set()
    if work_dir.exists():
        for entry in work_dir.iterdir():
            match = _SUBMISSION_REQUEST_RE.fullmatch(entry.name)
            if match:
                rounds.add(int(match.group(1)))
    results = []
    for round_index in sorted(rounds):
        raw_path = work_dir / f"transport_submit_{round_index:04d}.jsonl"
        for path in (raw_path, raw_path.with_suffix(".errors.jsonl")):
            if path.exists() or path.is_symlink():
                if path.is_symlink() or not path.is_file():
                    raise ValueError(
                        f"transport result is not a regular file: {path}")
                results.append(path)
    return tuple(results)


def _upload_to_gcs(local_path: str, uri: str):
    bucket_name, blob_name = parse_gcs_uri(uri)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename(local_path)
    print(f"  uploaded {local_path} -> {uri}")


def _download_batch_results(results_prefix: str) -> List[dict]:
    """Read every predictions JSONL the job wrote under a results prefix."""
    bucket_name, prefix = parse_gcs_uri(results_prefix)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    records = []
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith('.jsonl'):
            continue
        print(f"  reading gs://{bucket_name}/{blob.name}")
        for line in blob.download_as_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _normalize_batch_record(record: dict) -> dict:
    """Batch output -> the shape the aggregators read.

    Batch emits {key, status, response, processed_time} where a non-empty
    `status` means the request failed and `response` is the string "{}". The
    online path already emits exact ``{key,response|error}`` lifecycle records,
    so batch normalization produces the same boundary shape.
    """
    status = record.get('status')
    response = record.get('response')
    out = {'key': record.get('key')}
    failed = bool(status) or not isinstance(response, dict)
    if failed:
        out['error'] = status or f"no response object: {response!r}"
    else:
        out['response'] = response
    return out


def cmd_run_batch(args):
    """Run requests through the Vertex Batch API end to end.

    Upload, submit, poll, download, and normalize one stage-owned request
    shard. Retry selection belongs to the immutable request/attempt lifecycle;
    this transport always writes a fresh raw output.
    """
    check_environment()
    if not HAS_GCS:
        sys.exit("google-cloud-storage is required for batch execution")
    if not args.gcs_prefix:
        sys.exit("--gcs_prefix is required for batch execution (a gs:// "
                 "staging path)")

    records = []
    with open(args.input) as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} requests from {args.input}")

    output_path = Path(args.output)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(
            "transport output already exists; request lifecycle must allocate "
            f"a new attempt path: {output_path}")
    if not records:
        raise ValueError("request transport input is empty")

    tag = args.tag
    requests_uri, results_prefix = _batch_stage_uri(args.gcs_prefix, tag)

    with tempfile.NamedTemporaryFile('w', suffix='.jsonl',
                                     delete=False) as staged:
        for record in records:
            staged.write(json.dumps(record) + "\n")
        staged_path = staged.name
    try:
        _upload_to_gcs(staged_path, requests_uri)
    finally:
        os.unlink(staged_path)

    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    print(f"submitting batch job: model={args.model}")
    job = client.batches.create(
        model=args.model,
        src=requests_uri,
        config=CreateBatchJobConfig(dest=results_prefix),
    )
    print(f"  job: {job.name}")
    print(f"  results: {results_prefix}")

    start = time.time()
    while True:
        job = client.batches.get(name=job.name)
        elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
        if job.state not in ACTIVE_STATES:
            print(f"  {job.state} after {elapsed}")
            break
        print(f"  {job.state} ({elapsed} elapsed); next check in "
              f"{args.poll_interval}s")
        time.sleep(args.poll_interval)

    if job.state != JobState.JOB_STATE_SUCCEEDED:
        detail = getattr(job, 'error', None)
        sys.exit(f"batch job did not succeed: {job.state} {detail or ''}")

    downloaded = _download_batch_results(results_prefix)
    print(f"  {len(downloaded)} result record(s)")

    n_ok = 0
    n_err = 0
    prompt = output = thinking = 0
    with open(output_path, 'x') as handle:
        for record in downloaded:
            normalized = _normalize_batch_record(record)
            if 'error' in normalized:
                n_err += 1
            else:
                n_ok += 1
                usage = (normalized['response'].get('usageMetadata') or {})
                prompt += usage.get('promptTokenCount', 0) or 0
                output += usage.get('candidatesTokenCount', 0) or 0
                thinking += usage.get('thoughtsTokenCount', 0) or 0
            handle.write(json.dumps(normalized) + "\n")

    print(f"\nCompleted: {n_ok} ok, {n_err} error(s) -> {output_path}")
    print(f"Token usage:\n  Prompt:   {prompt:,}\n  Output:   {output:,}\n"
          f"  Thinking: {thinking:,}\n  Total:    {prompt + output + thinking:,}")
    if n_err:
        print(f"\n{n_err} request(s) failed; the request lifecycle will select "
              "them for a new attempt.")


def add_execution_arguments(parser):
    """Flags for a stage that turns a requests JSONL into a results JSONL.

    Batch is the default because it is half the price of on-demand for
    identical output, and the stages that use this are the expensive ones. The
    trade is latency -- minutes becomes up to a day -- so `--online` swaps back
    when a fast turnaround matters more than the discount.

    `--model` has no default: which model to run is a modeling choice recorded
    in the run config, not something a flag block should decide. Likewise there
    is no default staging bucket:
    `--gcs_prefix` names where batch traffic stages, and batch execution
    refuses to run without it.
    """
    group = parser.add_argument_group('model execution')
    group.add_argument('--model', required=True,
                       help='Model id to execute with. Required: the model is '
                            'a modeling choice, recorded in the run config.')
    group.add_argument('--online', action='store_true',
                       help='Use on-demand (synchronous) calls instead of the '
                            'Batch API. Faster to return, twice the price.')
    group.add_argument('--gcs_prefix', default=None,
                       help='Full gs:// staging prefix for batch requests + '
                            'results. Required unless --online; there is no '
                            'default bucket.')
    group.add_argument('--parallel', type=int, default=8,
                       help='Concurrent requests when --online')
    group.add_argument('--poll_interval', type=int, default=120,
                       help='Seconds between batch job state checks')
    group.add_argument('--cost_limit', type=float, default=50.0,
                       help='Refuse a single step estimated to cost more than '
                            'this many USD until a human approves it '
                            '(default: 50)')
    group.add_argument('--approve_cost', action='store_true',
                       help='Approve a step that exceeds --cost_limit')


def run_requests(args, input_path, output_path, *, tag):
    """Execute a requests JSONL through whichever path `args` selects.

    One entry point so a caller does not branch on transport. Both paths write
    the same raw records to a fresh `output_path`; stage-owned lifecycle code
    imports them into immutable attempt shards.
    """
    input_path = str(input_path)
    output_path = Path(output_path)
    errors_path = output_path.with_suffix('.errors.jsonl')
    for path in (output_path, errors_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"transport output already exists; allocate a new attempt "
                f"path: {path}")

    # Before either transport spends anything: estimate, and stop if the step is
    # over the ceiling. Placed here rather than in each caller so no stage can
    # forget it. Priced at the run's model: the price spread across the family
    # is ~5x, and the Pro-rate fallback would refuse a Flash run comfortably
    # inside its ceiling (see llm_cost.MODEL_RATES).
    estimate = llm_cost.estimate_jsonl(input_path, model=args.model)
    llm_cost.enforce_limit(
        estimate, limit_usd=getattr(args, 'cost_limit', 50.0), label=tag,
        online=args.online, approved=getattr(args, 'approve_cost', False))

    if args.online:
        print(f"executing {tag} on-demand (--online)")
        return cmd_run_online(argparse.Namespace(
            input=input_path, output=str(output_path),
            model=args.model, parallel=args.parallel))
    if not args.gcs_prefix:
        sys.exit(f"{tag}: --gcs_prefix is required for batch execution (a "
                 f"gs:// staging path for requests + results); there is no "
                 f"default bucket. Pass --online to run on-demand without "
                 f"GCS staging.")
    print(f"executing {tag} through the Batch API (half price; use --online "
          f"for a faster, dearer run)")
    return cmd_run_batch(argparse.Namespace(
        input=input_path, output=str(output_path), model=args.model,
        gcs_prefix=args.gcs_prefix, tag=tag, poll_interval=args.poll_interval))


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
    if args.command == 'list':
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
