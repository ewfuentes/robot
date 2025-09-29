#!/usr/bin/env python3
"""
Debug training launcher that starts training with enhanced monitoring.
"""

import subprocess
import argparse
import time
import os
import signal
import threading
from pathlib import Path


def launch_training_with_monitoring(train_args: list[str]):
    """Launch training process with debug monitoring."""
    print("🚀 Starting training with debug monitoring...")

    # Import train module directly to avoid nested bazel calls
    import sys
    from pathlib import Path

    # Determine the workspace root and script paths
    # When running via bazel run, we're in a runfiles directory
    current_file = os.path.abspath(__file__)

    # Try to find runfiles directory
    if 'runfiles' in current_file:
        # Extract runfiles root
        runfiles_root = current_file.split('.runfiles')[0] + '.runfiles'
        workspace_name = 'robot'  # your workspace name
        train_script = os.path.join(runfiles_root, workspace_name, 'experimental/overhead_matching/swag/scripts/train')
        monitor_script = os.path.join(runfiles_root, workspace_name, 'experimental/overhead_matching/swag/scripts/debug_hang_monitor')
    else:
        # Fallback: assume we're in workspace and use bazel-bin
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file)))))
        train_script = os.path.join(workspace_root, "bazel-bin/experimental/overhead_matching/swag/scripts/train")
        monitor_script = os.path.join(workspace_root, "bazel-bin/experimental/overhead_matching/swag/scripts/debug_hang_monitor")

    print(f"Training script: {train_script}")
    print(f"Verifying script exists: {os.path.exists(train_script)}")
    print(f"Training args: {' '.join(train_args)}")

    # Clear previous debug logs
    debug_log = "/tmp/training_debug.log"
    heartbeat_log = "/tmp/training_heartbeat.txt"

    for log_file in [debug_log, heartbeat_log]:
        if os.path.exists(log_file):
            os.remove(log_file)

    # Start training process directly (not through bazel)
    train_cmd = [train_script] + train_args
    print(f"Executing: {' '.join(train_cmd)}")

    training_process = subprocess.Popen(
        train_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    print(f"Training started with PID: {training_process.pid}")

    # Wait a moment for training to start and create log files
    time.sleep(5)

    # Start monitoring process directly (not through bazel)
    monitor_cmd = [
        monitor_script,
        "--pid", str(training_process.pid),
        "--debug-log", debug_log,
        "--heartbeat", heartbeat_log
    ]

    print(f"Starting monitor: {' '.join(monitor_cmd)}")
    monitor_process = subprocess.Popen(monitor_cmd)

    # Monitor training output
    def output_reader():
        for line in iter(training_process.stdout.readline, ''):
            print(f"[TRAIN] {line.rstrip()}")

    output_thread = threading.Thread(target=output_reader, daemon=True)
    output_thread.start()

    try:
        # Wait for training to complete
        training_process.wait()
        print(f"Training completed with exit code: {training_process.returncode}")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Stopping processes...")
        training_process.terminate()
        monitor_process.terminate()

    finally:
        # Clean up monitor process
        try:
            monitor_process.terminate()
            monitor_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor_process.kill()

        print("Debug logs available at:")
        print(f"  Debug log: {debug_log}")
        print(f"  Heartbeat: {heartbeat_log}")

    return training_process.returncode


def main():
    parser = argparse.ArgumentParser(description="Launch training with debug monitoring")
    # Use parse_known_args to capture all remaining arguments for train.py
    _, train_args = parser.parse_known_args()

    exit_code = launch_training_with_monitoring(train_args)
    exit(exit_code)


if __name__ == "__main__":
    main()