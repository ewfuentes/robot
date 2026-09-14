# Causal cumulative track evidence — experimental handoff

This branch snapshots the current `harel/regression-accuracy` changes and the frozen cumulative track-update experiment. It is for independent testing across **all 13 sequences**, with the same settings for every sequence. It is **not an adopted accuracy fix or a paper-ready end-to-end online result**.

## Requirement for the paper

**Real-time throughput is not required. Future information is forbidden.** An output may be delayed by computation, but its inference must use only inputs available through its declared timestamp. Previously emitted outputs must never be revised. Do not use completed historical tracks or later semantic audits to manufacture earlier evidence.

The new update rule obeys this prefix constraint. The current benchmark has separate unresolved dependencies:

- The supplied Portland camera mounting/nominal-forward calibration is a fixed 272° camera-to-airframe yaw offset. Its approval record lists frames 0,123,246,368,491 and cites aircraft nose/gear appearance plus cruise optical flow from legs 1/2. It is a physical extrinsic, not a per-frame correction or smoother, but its review used later test-sequence imagery. Replace it with independently measured calibration or estimate calibration from an observed prefix before claiming fully online evaluation. Do not silently treat the current calibration as independent.
- Horizon-levelled video has temporal dependencies. The replay delays input availability using the stored dependency bound; historical raw-fit parity and internal video stabilization are not fully certified.
- Per-frame detections are cached, and benchmark motion is derived from GPS. Detector latency is not measured here. These qualifications must remain visible in comparisons.
- Cumulative factor replacement is exact for a static pose but approximate under motion diffusion. This is a max-mixture composite likelihood, not a proof of exact Bayesian filtering.

## Frozen method

A valid birth is first released after at least three supported frames and ten seconds of observed age, or observed closure. At fixed eight-keyframe snapshots, newly supported views update that physical track. Use the ratio `L(new cumulative prefix) / L(previous prefix)`, raised to its fixed first-release exponent. Do not multiply the full cumulative factor again. The first-release exponent is the existing predictive tempering rule; all co-released first factors use the same pre-arrival prior.

Settings: 36 heading bins, 100m cells, max mixture, five-frame measurement epochs, existing soft-name/category matching and guarded reservoir-tag compatibility. The supplied input image dimensions and calibration can differ; inference thresholds do not. No location/frame/track-ID/ground-truth tuning is permitted. Ground truth is used only for scoring; benchmark motion/calibration qualifications above still apply.

## Evidence so far

Portland leg1, intervals 0–180, seeds 0/1/2: cumulative distance-normalized mass within 500m is approximately 0.6235–0.6249, versus 0.2015–0.2027 for the paired one-shot tempered baseline. All five whole-window radii improve over both paired and historical references in this screen. There are still early 50–100m precision regressions. Full-sequence and all13 results are pending; do not extrapolate from this screen.

Tracking over 176 intervals took 1927s for 377s of video (~5.11× playback duration). The modeled batch backlog is not an accuracy measurement with processing delay. Slow execution is not itself a violation of the online-filter requirement.

The files `revision_seed_validation.json`, `portland1_cumulative_tracker_delay176_v1.json`, `overnight_all13_manifest.json`, and `all13_missing_source_inputs.json` preserve the current limited evidence and coverage. The speedup branch remains separate and unchanged.

## Data and environment

No videos, models, raw detections, catalogs, export datasets, credentials, or large result arrays are committed. This code expects the existing dataset tree mounted at `/data/farfield_matching`; do not rewrite provenance hashes to make mismatched inputs pass.

Provide the original accuracy-recovery run directory for the thirteen historical result JSONs and `.run.json` records, catalog report files, and any leveling-readiness files. Their referenced datasets, catalogs, detections and calibration must also exist. The existing smoothing-fix `plan.json` is needed by legacy table-helper imports. Canonical video SHA256 is checked against the original tracking manifest. Nine recorded videos were missing on the originating host; see the missing-input manifest, including the missing Portland2/3 leveling sources. Missing rows must remain unevaluated, never count as passes.

Use the repository's Python3.12/Bazel runfiles environment with Torch2.7/CUDA12.8, or an equivalent tested environment. Build the localization target to materialize dependencies:

```bash
bazel build //experimental/overhead_matching/swag/farfield/localization:grid_filter
```

Local SAM2 weights must already exist at `/data/farfield_matching/models/sam2/sam2.1_hiera_large.pt` with the recorded hash. Local SAM2 Python dependencies are linked from the source run's `tracking_dependencies` directory; install an equivalent offline package there if absent. **No Gemini or other model API calls are allowed.** The generated launcher installs a Python socket-denial hook and sets local model loading to offline mode.

## Prepare an isolated workspace

From this branch's repository root:

```bash
PACKAGE=experimental/overhead_matching/swag/farfield/experiments/causal_track_revisions
python3 "$PACKAGE/prepare_workspace.py" \
  --source-run /data/farfield_matching/runs/260913_accuracy_recovery \
  --output /data/farfield_matching/runs/causal_revision_multigpu
```

Use a new output directory. Preparation copies the small reference files and the frozen source snapshot. It records source relocation separately and overrides inherited cache destinations, preserving the original evidence files. The original active experiment directory is never used for new outputs. The code intentionally leaves dataset paths and historical reference bytes unchanged.

If dependencies are in another already-built runfiles tree, set `FARFIELD_RUNFILES` to that directory. `FARFIELD_PYTHON` can select an equivalent interpreter. `FARFIELD_REPO` defaults to this checkout. **The generated launcher honors `CUDA_VISIBLE_DEVICES`; unlike the old local launcher, it does not force GPU0.**

Check inputs without starting GPU inference:

```bash
python3 "$PACKAGE/run_sequences.py" \
  --workspace /data/farfield_matching/runs/causal_revision_multigpu \
  --gpu 0 --stage preflight
```

A missing source is a failure with a specific log path. The worker continues checking the other sequences and returns nonzero if any fail.

## Run independent GPU shards

Run one worker per GPU, with disjoint sequence shards. For four GPUs:

```bash
for gpu in 0 1 2 3; do
  python3 "$PACKAGE/run_sequences.py" \
    --workspace /data/farfield_matching/runs/causal_revision_multigpu \
    --gpu "$gpu" --shard-index "$gpu" --num-shards 4 \
    --seeds 0 1 2 > "/tmp/causal-revisions-gpu${gpu}.log" 2>&1 &
done
wait
```

Workers take GPU and dataset locks, reject an already-busy GPU, and run tracking, evidence construction, then paired filters sequentially within each sequence. They resume exact compatible tracker checkpoints and reuse checksummed completed outputs. A quality regression does not cancel the other sequences. Failed commands retain their logs and return a nonzero worker status; inspect every `worker.*.json`, not only the shell's final `wait` result.

For rapid iteration, use `--datasets DATASET ... --end 64 --seeds 0`. Omit `--end` for the entire available frame range. `--stage track`, `evidence`, or `filter` can reuse completed earlier stages. Never treat a short diagnostic window as a full-sequence result.

## Required independent checks

1. Run the included focused unit tests, then a short paired smoke run on each dataset before committing long GPU time.
2. Verify the relocated code reproduces an existing Portland prefix numerically before interpreting changed accuracy. The handoff has input and pure-CPU tests; multi-GPU numerical parity is still an independent-agent task.
3. Complete all13 sequences across seeds 0/1/2 with identical settings. Record all five distance-normalized mass metrics (50/100/250/500/1000m), MAP errors, and per-sequence regressions. Historical references stop one keyframe early; compare them only over common coverage, and report the final output separately.
4. Compare short-prefix outputs against the same prefix of the full run. Verify unchanged consumed observations, release clocks, matching tables, exponents, and emitted posterior/MAP outputs. Tracker snapshots occur before EOF finalization; do not flush unreleased tracks at the final frame.
5. Record runtime and dependency/processing delay without making real-time throughput an acceptance criterion. Do not backdate later observations into earlier estimates.
6. Resolve the calibration and preprocessing dependencies before describing end-to-end results as online. Keep the current results explicitly qualified until then.
7. Report an adoption verdict only after the above, with a reproducible command/config manifest and all failures visible. No automatic merge or deployment is performed.

The snapshot includes some legacy helper functions for import compatibility; only the documented worker path implements the frozen candidate. Unrelated helper entry points are not approved alternative policies.
