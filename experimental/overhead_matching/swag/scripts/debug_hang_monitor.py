#!/usr/bin/env python3
"""
Debug hang monitor script to help identify where training hangs occur.
Run this in parallel with training to monitor progress.
"""

import argparse
import time
import os
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timedelta
import psutil


class HangMonitor:
    def __init__(self, pid: int, debug_log_path: str = "/tmp/training_debug.log",
                 heartbeat_path: str = "/tmp/training_heartbeat.txt"):
        self.pid = pid
        self.debug_log_path = debug_log_path
        self.heartbeat_path = heartbeat_path
        self.running = threading.Event()
        self.running.set()

        # Track last activity
        self.last_debug_log_mtime = 0
        self.last_heartbeat_mtime = 0
        self.last_debug_line_count = 0

        # Hang detection thresholds
        self.debug_log_hang_threshold = 300  # 5 minutes
        self.heartbeat_hang_threshold = 120  # 2 minutes

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[MONITOR {timestamp}] {message}", flush=True)

    def get_process_info(self):
        """Get detailed process information."""
        try:
            proc = psutil.Process(self.pid)

            # Get CPU and memory usage
            cpu_percent = proc.cpu_percent()
            memory_info = proc.memory_info()

            # Get thread count and status
            threads = proc.threads()

            # Get open files
            try:
                open_files = len(proc.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = "N/A"

            # Get network connections
            try:
                connections = len(proc.connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                connections = "N/A"

            return {
                'cpu_percent': cpu_percent,
                'memory_mb': memory_info.rss / 1024 / 1024,
                'thread_count': len(threads),
                'open_files': open_files,
                'connections': connections,
                'status': proc.status()
            }
        except psutil.NoSuchProcess:
            return None

    def check_gpu_utilization(self):
        """Check GPU utilization using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = []
                for i, line in enumerate(lines):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        gpu_util = parts[0].strip()
                        mem_used = parts[1].strip()
                        mem_total = parts[2].strip()
                        gpu_info.append({
                            'gpu': i,
                            'utilization': f"{gpu_util}%",
                            'memory': f"{mem_used}/{mem_total} MB"
                        })
                return gpu_info
        except Exception as e:
            self.log(f"GPU check failed: {e}")
        return []

    def check_debug_log_activity(self):
        """Check if debug log is being updated."""
        if not os.path.exists(self.debug_log_path):
            return False, "Debug log file does not exist"

        try:
            stat = os.stat(self.debug_log_path)
            current_mtime = stat.st_mtime

            # Count lines in log file
            with open(self.debug_log_path, 'r') as f:
                current_line_count = sum(1 for line in f)

            # Check if file was modified or new lines added
            if (current_mtime > self.last_debug_log_mtime or
                current_line_count > self.last_debug_line_count):

                self.last_debug_log_mtime = current_mtime
                self.last_debug_line_count = current_line_count

                # Get last few lines
                with open(self.debug_log_path, 'r') as f:
                    lines = f.readlines()
                    last_lines = lines[-3:] if len(lines) >= 3 else lines

                return True, f"Active (lines: {current_line_count}). Last: {last_lines[-1].strip() if last_lines else 'Empty'}"
            else:
                time_since_update = time.time() - current_mtime
                return False, f"No activity for {time_since_update:.1f}s (lines: {current_line_count})"

        except Exception as e:
            return False, f"Error checking debug log: {e}"

    def check_heartbeat_activity(self):
        """Check if heartbeat file is being updated."""
        if not os.path.exists(self.heartbeat_path):
            return False, "Heartbeat file does not exist"

        try:
            stat = os.stat(self.heartbeat_path)
            current_mtime = stat.st_mtime

            if current_mtime > self.last_heartbeat_mtime:
                self.last_heartbeat_mtime = current_mtime

                # Read heartbeat content
                with open(self.heartbeat_path, 'r') as f:
                    content = f.read().strip()

                return True, f"Active. Last: {content}"
            else:
                time_since_update = time.time() - current_mtime
                return False, f"No heartbeat for {time_since_update:.1f}s"

        except Exception as e:
            return False, f"Error checking heartbeat: {e}"

    def detect_hanging_indicators(self):
        """Detect potential hanging scenarios."""
        indicators = []

        # Check if process exists
        proc_info = self.get_process_info()
        if proc_info is None:
            indicators.append("CRITICAL: Process not found!")
            return indicators

        # Check CPU usage patterns
        if proc_info['cpu_percent'] < 1.0:
            indicators.append(f"LOW CPU: {proc_info['cpu_percent']:.1f}% (may be hung)")

        # Check thread status
        if proc_info['status'] == 'zombie':
            indicators.append("CRITICAL: Process is zombie")
        elif proc_info['status'] == 'stopped':
            indicators.append("CRITICAL: Process is stopped")

        # Check debug log activity
        debug_active, debug_msg = self.check_debug_log_activity()
        if not debug_active and "No activity for" in debug_msg:
            time_str = debug_msg.split("No activity for ")[1].split("s")[0]
            try:
                inactive_time = float(time_str)
                if inactive_time > self.debug_log_hang_threshold:
                    indicators.append(f"HANG DETECTED: No debug log activity for {inactive_time:.1f}s")
            except ValueError:
                pass

        # Check heartbeat activity
        heartbeat_active, heartbeat_msg = self.check_heartbeat_activity()
        if not heartbeat_active and "No heartbeat for" in heartbeat_msg:
            time_str = heartbeat_msg.split("No heartbeat for ")[1].split("s")[0]
            try:
                inactive_time = float(time_str)
                if inactive_time > self.heartbeat_hang_threshold:
                    indicators.append(f"HANG DETECTED: No heartbeat for {inactive_time:.1f}s")
            except ValueError:
                pass

        return indicators

    def monitor_loop(self):
        """Main monitoring loop."""
        self.log(f"Starting hang monitor for PID {self.pid}")
        self.log(f"Debug log: {self.debug_log_path}")
        self.log(f"Heartbeat: {self.heartbeat_path}")
        self.log("=" * 60)

        while self.running.is_set():
            # Get process info
            proc_info = self.get_process_info()
            if proc_info is None:
                self.log("CRITICAL: Training process not found! Exiting monitor.")
                break

            # Check for hanging indicators
            hang_indicators = self.detect_hanging_indicators()

            # Get GPU info
            gpu_info = self.check_gpu_utilization()

            # Check debug log activity
            debug_active, debug_msg = self.check_debug_log_activity()

            # Check heartbeat activity
            heartbeat_active, heartbeat_msg = self.check_heartbeat_activity()

            # Report status
            status = "🟢 HEALTHY" if not hang_indicators else "🔴 POTENTIAL HANG"
            self.log(f"{status} | CPU: {proc_info['cpu_percent']:.1f}% | "
                    f"RAM: {proc_info['memory_mb']:.1f}MB | "
                    f"Threads: {proc_info['thread_count']} | "
                    f"Status: {proc_info['status']}")

            if gpu_info:
                gpu_str = " | ".join([f"GPU{g['gpu']}: {g['utilization']} util, {g['memory']}"
                                     for g in gpu_info])
                self.log(f"GPU: {gpu_str}")

            self.log(f"Debug Log: {debug_msg}")
            self.log(f"Heartbeat: {heartbeat_msg}")

            if hang_indicators:
                self.log("🚨 HANG INDICATORS:")
                for indicator in hang_indicators:
                    self.log(f"  - {indicator}")

            self.log("-" * 60)

            # Sleep before next check
            time.sleep(30)

    def stop(self):
        self.running.clear()


def main():
    parser = argparse.ArgumentParser(description="Monitor training process for hangs")
    parser.add_argument("--pid", type=int, required=True, help="PID of training process to monitor")
    parser.add_argument("--debug-log", default="/tmp/training_debug.log",
                       help="Path to debug log file")
    parser.add_argument("--heartbeat", default="/tmp/training_heartbeat.txt",
                       help="Path to heartbeat file")

    args = parser.parse_args()

    monitor = HangMonitor(args.pid, args.debug_log, args.heartbeat)

    try:
        monitor.monitor_loop()
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    except Exception as e:
        print(f"Monitor error: {e}")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()