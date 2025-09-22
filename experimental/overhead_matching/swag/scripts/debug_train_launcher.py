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

    # Build the bazel training command
    bazel_args = ["bazel", "run", "//experimental/overhead_matching/swag/scripts:train", "--"] + train_args
    print(f"Training command: {' '.join(bazel_args)}")

    # Clear previous debug logs
    debug_log = "/tmp/training_debug.log"
    heartbeat_log = "/tmp/training_heartbeat.txt"

    for log_file in [debug_log, heartbeat_log]:
        if os.path.exists(log_file):
            os.remove(log_file)

    # Start training process
    train_cmd = bazel_args
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

    # Start monitoring process using bazel to run from repo
    monitor_cmd = [
        "bazel", "run", "//experimental/overhead_matching/swag/scripts:debug_hang_monitor", "--",
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
    parser.add_argument("train_args", nargs=argparse.REMAINDER,
                       help="Arguments to pass to train.py")

    args = parser.parse_args()

    exit_code = launch_training_with_monitoring(args.train_args)
    exit(exit_code)


if __name__ == "__main__":
    main()