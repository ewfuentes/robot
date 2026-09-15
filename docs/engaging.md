# Running jobs on MIT Engaging

This is a repository-specific quickstart for running independent CPU or GPU
jobs on MIT's Engaging cluster. Consult the linked ORCD documentation for the
current partition limits and policies.

## Connect

The examples assume that `orcd-login` is an SSH target configured for
`orcd-login.mit.edu` as described in the
[ORCD SSH documentation](https://orcd-docs.mit.edu/accessing-orcd/ssh-login/):

```bash
ssh orcd-login
```

For agents and other automation: if this asks for a password or Duo response,
stop and ask the user to run `ssh orcd-login` in a different terminal. Resume
only after the user confirms that login works. Never ask the user to paste a
password, Duo response, private key, or other credential into the automation.

Prefer one persistent SSH session for setup and monitoring instead of many
one-shot SSH calls. ORCD's SSH control-channel instructions can reduce repeat
authentication.

## Put files in the right place

Keep replaceable code and software in home, active job inputs and outputs in
scratch, and inactive large data in pool. Scratch and pool are not backed up.

```bash
REPO="$HOME/code/robot"
RUN_ROOT="$HOME/orcd/scratch/robot"
mkdir -p "$HOME/code" "$RUN_ROOT"/{inputs,logs,runs}
git clone https://github.com/ewfuentes/robot.git "$REPO"
```

Use HTTPS for this public repository; an SSH GitHub URL requires a GitHub SSH
key even when the repository itself is public. Check out and record the exact
branch or commit used for a run:

```bash
git -C "$REPO" switch BRANCH
git -C "$REPO" rev-parse HEAD
```

From the local machine, `rsync` is convenient for a modest input subset:

```bash
rsync -a --info=stats2 path/to/inputs/ \
  orcd-login:orcd/scratch/robot/inputs/
```

Use Globus for large transfers. Transfer only the inputs required by the job,
and confirm that their license and access rules allow them to be placed on
Engaging. Never commit research inputs, credentials, account names, absolute
user paths, run manifests, or logs without reviewing them for non-public data.

See ORCD's [filesystem](https://orcd-docs.mit.edu/filesystems-file-transfer/filesystems/)
and [file-transfer](https://orcd-docs.mit.edu/filesystems-file-transfer/transferring-files/)
guides for current details.

## Install Bazelisk

The repository's `.bazeliskrc` pins the Bazel version. Install Bazelisk in the
user account rather than running the Ubuntu-oriented `setup.sh` on Engaging:

```bash
mkdir -p "$HOME/.local/bin"
curl -fL \
  https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64 \
  -o "$HOME/.local/bin/bazel"
chmod 755 "$HOME/.local/bin/bazel"
"$HOME/.local/bin/bazel" --version
```

Do not copy a workstation-generated `.bazelrc_ubuntu` to Engaging. It names an
Ubuntu toolchain that is not the cluster's Rocky Linux toolchain.

## Choose a partition

- Use `mit_normal` for non-GPU work.
- Use `mit_normal_gpu` for GPU work that must not be preempted. Prefer an L40S
  unless the workload needs another GPU type.
- Use `mit_quicktest` only for short CPU tests.
- Use `mit_preemptable` for restartable or idempotent work that benefits from
  a larger, lower-priority pool. Add `--requeue`; this does not create
  checkpoints for the application.

The older `sched_*` partitions use a different CentOS 7 software stack. Prefer
the modern `mit_*` partitions unless the old stack is intentional. Check live
availability with:

```bash
sinfo
sinfo -p mit_normal,mit_normal_gpu -O Partition,Nodes,CPUs,Memory,Gres -e
```

See the ORCD [scheduler overview](https://orcd-docs.mit.edu/running-jobs/overview/)
and [resource guide](https://orcd-docs.mit.edu/running-jobs/requesting-resources/)
before changing partitions or resource requests.

## Build once on a compute node

Do not build substantial targets on a login node. A minimal `build.sbatch`
looks like this; replace the Bazel target:

```bash
#!/bin/bash
#SBATCH --job-name=robot-build
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/build-%j.log

set -euo pipefail
module purge
module load miniforge

REPO="$HOME/code/robot"
RUN_ROOT="$HOME/orcd/scratch/robot"
export BAZELISK_HOME="$RUN_ROOT/.bazelisk"

cd "$REPO"
"$HOME/.local/bin/bazel" --output_user_root="$RUN_ROOT/.bazel" build \
  //path/to:program
```

Create `logs/` before submission because Slurm opens the log before running
the script. Test the request, then submit it from `RUN_ROOT`:

```bash
cd "$RUN_ROOT"
sbatch --test-only build.sbatch
BUILD_JOB=$(sbatch --parsable build.sbatch)
```

Load `miniforge` in every Python job script. Without it, a Bazel Python
launcher may fail with `/usr/bin/env: python: No such file or directory`.

Build the target once, then invoke its `bazel-bin/...` executable directly
from each array task. Concurrent `bazel run` commands sharing an output base
wait on Bazel's lock and serialize the work.

## Parallelize independent work with an array

Put one input path per line in `inputs.txt` under `RUN_ROOT`. Keep that runtime
manifest in scratch rather than committing it. A GPU array template is:

```bash
#!/bin/bash
#SBATCH --job-name=robot-eval
#SBATCH --partition=mit_normal_gpu
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-15%2
#SBATCH --output=logs/%A_%a.log

set -euo pipefail
module purge
module load miniforge

REPO="$HOME/code/robot"
RUN_ROOT="$HOME/orcd/scratch/robot"
mapfile -t inputs < "$RUN_ROOT/inputs.txt"
input="${inputs[$SLURM_ARRAY_TASK_ID]}"

exec "$REPO/bazel-bin/path/to/program" \
  --input "$input" \
  --output "$RUN_ROOT/runs/$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID"
```

Replace the program and its flags, and set the array's upper bound to one less
than the number of manifest lines. `%2` caps this example at two concurrent
tasks. Match the cap to the selected partition and account limits instead of
submitting as much concurrency as Slurm accepts.

First run one array element after the build succeeds:

```bash
SMOKE_JOB=$(sbatch --parsable --array=0-0 \
  --dependency=afterok:"$BUILD_JOB" array.sbatch)
squeue -j "$SMOKE_JOB"
```

Inspect its log and output before submitting the full array. For preemptable
work, change the partition, add `#SBATCH --requeue`, and ensure rerunning a
task cannot corrupt an existing partial result.

Slurm already packs independent one-GPU jobs onto different GPUs of shared
nodes. Do not request an entire node merely to do that packing yourself. Start
with one process per GPU. If measurements show low sustained GPU utilization
and enough GPU-memory headroom, benchmark two processes per GPU before making
that the default.

See ORCD's [job-array guide](https://orcd-docs.mit.edu/running-jobs/job-arrays/)
for other ways to divide a manifest among fewer Slurm tasks.

## Monitor and right-size

```bash
squeue --me
sacct -j JOB_ID -X \
  -o JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,NodeList
jobstats JOB_ID
```

Use `squeue` for pending/running state, `sacct` for final status and memory,
and `jobstats` for aggregate CPU/GPU efficiency. For a live GPU view, find the
assigned node with `squeue --me`, SSH to that node from the login node, and run
`nvidia-smi -l 10`. A single GPU-utilization sample can miss a bursty workload,
so observe it over time.

After a successful run, copy durable results off scratch:

```bash
rsync -a orcd-login:orcd/scratch/robot/runs/JOB_ID/ ./runs/JOB_ID/
```

ORCD's [job-analysis](https://orcd-docs.mit.edu/running-jobs/application-analysis/)
and [best-practices](https://orcd-docs.mit.edu/running-jobs/best-practices/)
pages describe `jobstats`, live GPU monitoring, and resource right-sizing in
more detail.
