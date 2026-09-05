# Self-collected 360 data

Self-collected video has two pre-ingest responsibilities that Mapillary data
does not: resolving the video clock onto an independent GPS clock, and removing
identifying imagery. The reusable tools live under `farfield/dataset_tools`:

- `prepare_selfcollect` parses Garmin FIT, Sensor Logger ZIP, or timestamped
  CSV tracks (`timestamp,lat,lon` in Unix seconds), applies
  an explicit sync anchor, and plans distance-spaced output frames;
- `fetch_anonymization_models` downloads content-pinned detector weights;
- `anonymize_video` produces the pinned raw license-plate candidate ledger and
  records the final human-review decision;
- `person_segmentation_preview` evaluates short YOLO person-mask sequences and
  temporal persistence before committing to a whole-video detector pass;
- `person_anonymize_video` resumably scans a complete clip, applies person and
  plate policy, and renders the full and review videos without changing the
  source;
- `ingest_selfcollect` converts blurred source frames into the frozen dataset
  contract and carries the external privacy-review status with them.

The old per-collection scripts under `raw_material/` remain provenance for
their collections. New collections should use the general tools.

## Collection configuration and clock semantics

`prepare_selfcollect` accepts one JSON file with a `gps_sources` object and a
`recordings` list. Paths are relative to the configuration file. Each recording
provides at least:

```json
{
  "dataset_id": "example_leg",
  "video": "videos/example.mp4",
  "capture_fps": 30,
  "output_fps": 3,
  "clip_start_s": 26,
  "clip_end_s": 501,
  "gps_source": "watch",
  "sync": {
    "sensor_elapsed_at_video_start_s": 1234.5,
    "uncertainty_s": 0.5,
    "visual_bracket_half_width_s": 0.016667,
    "evidence": "Timer transition bracketed by original frames 100 and 101."
  },
  "sampling": {"distance_m": 3, "course_radius_m": 5},
  "gps_quality": {"max_gap_s": 25, "fix_near_s": 1.5}
}
```

`capture_fps` is the camera rate at acquisition. It does not change when a
video is later exported at 3 fps. `source_media_fps` is probed from the file and
recorded separately. A transition observed between original 30 fps frames has
a half-bracket of `1/60` second; a transition recoverable only from a 3 fps
derivative has a half-bracket of `1/6` second. Keep both that visual bracket and
the larger absolute sensor-clock uncertainty instead of reporting false
precision.

`clip_start_s` is inclusive and `clip_end_s` is exclusive on the requested
output frame grid. `video_t_s` starts at zero in that clipped output;
`source_video_t_s` retains the corresponding timestamp in the untouched input.
The mapping used by every row is therefore:

```
sensor_elapsed_s = sensor_elapsed_at_video_start_s + source_video_t_s
```

GPS course is retained as diagnostic course over ground. It is not camera
heading and is never used to rotate panoramas.

## Privacy workflow

Fetch the pinned models once into the data root:

```bash
bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:fetch_anonymization_models -- \
  --output /data/farfield_matching/models/anonymization/v1
```

The downloaded detector is the global YOLOv9 license-plate model from Open
Image Models. Its source URL, license, exact weight hash, and upstream revision
are written to `SOURCE.json`. Person segmentation uses the separately retained
YOLO11x-seg weights. The tools perform detection only: there is no face
recognition and no license-plate OCR.

### Person-mask sample gate

When panorama face boxes are noisy, evaluate the older whole-person strategy on
short sequences before scanning a full video. The preview scanner uses
YOLO11x-seg class `person` at `imgsz=1920` and runs both the native panorama and
a half-width horizontal roll. It retains ordinary masks at confidence 0.15 and
weak candidates down to 0.05. It does not modify the video.

```bash
bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:person_segmentation_preview -- \
  scan \
  --source raw.mp4 \
  --weights /data/farfield_matching/models/anonymization/yolo11x_seg_v1/yolo11x-seg.pt \
  --output_dir processed/revisions/example_personseg_scan_v1 \
  --sample opening=135 \
  --sample crowd=1217 \
  --sample seam=1880 \
  --radius_frames 3 \
  --clip_start_s 130 --clip_end_s 2505

bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:person_segmentation_preview -- \
  render \
  --scan_dir processed/revisions/example_personseg_scan_v1 \
  --output_dir processed/revisions/example_personseg_preview_v1
```

Temporal persistence is intentionally conservative. It considers only an
isolated one-frame gap, warps accepted masks from both adjacent frames with
horizontal-wrap/vertical-reflect optical flow, checks forward/backward cycle,
appearance, overlap, area, and scene-cut gates, and prefers a matching weak
middle-frame mask. A synthesized fill is mask-shaped; unchanged rectangles are
never copied between frames. Evidence that fails a gate becomes a review flag,
not an automatic blur. Mostly covered direct masks short-circuit persistence so
ordinary mask-edge jitter does not expand the blur.

The generated `index.html` contains 2x2 frame comparisons and one short video
per sequence. Green is a direct person mask, magenta is an accepted temporal
fill, and orange is uncovered review-only suspicion. Optional controlled
dropouts are clearly labeled ablations: they prove the recovery path without
claiming that the detector actually missed that instance. The manifest binds
the scan tree, model digest, videos, contact sheets, metrics ledgers, and review
HTML. This experiment covers people; license plates still require the separate
plate detector. Do not start the whole-video pass until these samples have been
reviewed.

Plan synchronized frames before the expensive video pass:

```bash
bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:prepare_selfcollect -- \
  plan --config /data/farfield_matching/raw_material/COLLECTION/collection.json
```

### Production person-mask workflow

The production path has three separately bound stages: `scan`, `policy`, and
`render`. Give every stage a new, nonexistent revision directory. A scan,
policy pass, or render interrupted after committing work leaves an
`.incomplete` sibling; run the identical command to validate and resume it. A
changed source, model, clip, threshold, or other bound setting fails closed
instead of silently mixing evidence.

The policy stage needs raw YOLOv9 plate candidates from a compatible
`anonymize_video scan`. Its source hash, clip, 3 fps grid, and frame count must
exactly match the person scan. Only raw detections whose category and source
identify the pinned plate model are imported; face detections, prior temporal
boxes, and prior manual boxes are not reused. If no compatible plate scan
exists, create one with the same source and clip:

```bash
bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:anonymize_video -- \
  scan \
  --source /data/farfield_matching/raw_material/COLLECTION/videos/SOURCE.mp4 \
  --output_dir /data/farfield_matching/raw_material/COLLECTION/processed/revisions/DATASET_NAME_plate_scan_v1 \
  --plate_model /data/farfield_matching/models/anonymization/v1/yolo-v9-t-640-license-plates-end2end.onnx \
  --capture_fps 30 --output_fps 3 \
  --start_s START_SECONDS --end_s END_SECONDS
```

Run the approved YOLO11x-seg person model over the contiguous clip. The scanner
uses a 1920-pixel inference/grid width, native and half-width horizontal-roll
passes, direct person masks at confidence 0.15, and candidates down to 0.05.
It also retains COCO car, motorcycle, bus, and truck masks as independent
context for later plate validation. Per-frame evidence is committed atomically,
so the expensive scan can resume without rerunning valid frames.

```bash
bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:person_anonymize_video -- \
  scan \
  --source /data/farfield_matching/raw_material/COLLECTION/videos/SOURCE.mp4 \
  --weights /data/farfield_matching/models/anonymization/yolo11x_seg_v1/yolo11x-seg.pt \
  --output_dir /data/farfield_matching/raw_material/COLLECTION/processed/revisions/DATASET_NAME_person_scan_v1 \
  --capture_fps 30 --output_fps 3 \
  --start_s START_SECONDS --end_s END_SECONDS \
  --scan_width 1920 --imgsz 1920 \
  --candidate_confidence 0.05 --direct_confidence 0.15 \
  --device auto --workers WORKER_COUNT --torch_threads THREADS_PER_WORKER
```

`--device auto` records CUDA device 0 when CUDA is available to the process and
otherwise records `cpu`; use an explicit device to override it. For a single
large GPU, benchmark one versus two workers before a long scan because multiple
workers create independent model instances.

`--capture_fps 30` records acquisition provenance; it does not assert that the
input file still contains 30 frames per second. When `SOURCE.mp4` is an earlier
3 fps export, the probed `media_fps` remains 3 and `--output_fps 3` processes
that existing grid without inventing intermediate frames. `--start_s` is
inclusive and `--end_s` exclusive on the source media's own clock. Keep the
original capture rate at 30 even when the source media is already 3 fps.

Apply policy in a new directory. The person policy can fill only an isolated
one-frame gap and requires agreement from bidirectional optical flow on both
adjacent frames. A failed gate becomes review-only suspicion, not an applied
blur. Subthreshold weak detector residue is also retained for review, and the
first/last frame uses its sole neighbor as one-sided review evidence without
automatic fill. Every raw plate candidate must pass strict plausible-geometry
checks and overlap independently detected vehicle context before it is blurred.
A missing vehicle match keeps the candidate flagged for review but vetoes
automatic application; geometry-implausible candidates are also review-only.
Plate boxes are never copied to adjacent frames because static propagation can
miss a moving plate while blurring unrelated pixels.

```bash
bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:person_anonymize_video -- \
  policy \
  --scan_dir /data/farfield_matching/raw_material/COLLECTION/processed/revisions/DATASET_NAME_person_scan_v1 \
  --plate_manifest /data/farfield_matching/raw_material/COLLECTION/processed/revisions/DATASET_NAME_plate_scan_v1/anonymization_manifest.json \
  --output_dir /data/farfield_matching/raw_material/COLLECTION/processed/revisions/DATASET_NAME_person_policy_v1 \
  --flow_width 960
```

Render from the untouched source into another revision. The full-resolution
output remains 3 fps. The review video contains the same frame sequence at 5x
playback speed and defaults to 3840 pixels wide. Rendering commits independent
chunks; rerunning the identical command validates completed chunks and resumes
at the first missing one. Neither scan, policy, nor render writes into or
replaces the source video.

```bash
bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:person_anonymize_video -- \
  render \
  --policy_dir /data/farfield_matching/raw_material/COLLECTION/processed/revisions/DATASET_NAME_person_policy_v1 \
  --output_dir /data/farfield_matching/raw_material/COLLECTION/processed/revisions/DATASET_NAME_person_render_v1/anonymization \
  --output_name DATASET_NAME_3fps_personseg_blurred.mp4 \
  --chunk_frames 150 --cfr_fast_seek --encoder nvenc \
  --review_width 3840 --review_speedup 5 \
  --extraction_plan /data/farfield_matching/raw_material/COLLECTION/processed/revisions/DATASET_NAME_person_render_v1/dataset/frames_gps.csv \
  --frames_dir /data/farfield_matching/raw_material/COLLECTION/processed/revisions/DATASET_NAME_person_render_v1/dataset/frames \
  --jpeg_quality 92
```

`--cfr_fast_seek` avoids decoding from frame zero for every resumable chunk. It
is deliberately opt-in: use it only after confirming the source is constant
frame rate and its integer media rate is an exact multiple of the integer output
rate (for example, 3→3 or 30→3 fps). Seeking anchors at whole seconds, preserves
the integer selection phase, and trims the remainder on the output-frame grid.

`--encoder nvenc` uses the NVIDIA video-encoding engine for both the 8K HEVC
result and the accelerated H.264 review copy. It requires an FFmpeg build with
`hevc_nvenc`/`h264_nvenc` and a compatible NVIDIA GPU. Omit it for the portable
software default (`libx265`/`libx264`). The selected backend and its exact
FFmpeg options are recorded in the render specification, so a resumable render
cannot silently mix encoders. NVENC accelerates encoding only: mask replay,
OpenCV blur, decoding, JPEG extraction, and integrity validation remain CPU
work. Also note that the dedicated video engine is reported separately from
CUDA compute utilization by many GPU monitors.

For a self-collected dataset, run `prepare_selfcollect plan` first and point
these two extraction arguments at its `frames_gps.csv` and fresh `frames`
directory. The render manifest binds both paths and every JPEG hash so
`prepare_selfcollect finalize` can prove that ingestion uses the reviewed
blurred frames. Omit them only for a video-only revision that will not be
finalized as a dataset.

The completed render is still `rendered_pending_review`. Publication of files
means only that encoders, hashes, dimensions, frame rate, and frame count
validated; it is not a privacy approval.

## Human review gate

Every production person-mask render produces:

- `review.mp4`, a 3840-pixel-wide, 5x accelerated overview of the blurred output;
- `review.html`, with source-time display, previous/next flagged-frame controls,
  and the full-resolution blurred result at normal 3 fps;
- `detections.jsonl`, the exact person-mask evidence references and applied
  plate/manual regions, plus unresolved person and plate review evidence;
- `anonymization_manifest.json`, binding source, models, ledgers, and outputs by
  SHA-256.

Serve one review bundle without exposing the rest of the data tree or creating
notes inside the audited directory:

```bash
bazel run //experimental/overhead_matching/swag/farfield/viewers:server -- \
  --root /data/farfield_matching/raw_material/COLLECTION/processed/revisions/REVISION/anonymization \
  --port 9876 \
  --read-only
```

Then open `http://localhost:9876/review.html`. The server binds only to IPv4
loopback and supports byte-range video playback; use SSH local forwarding when
reviewing from another machine.

Watch the overview end to end. Green outlines are direct person masks, magenta
outlines are accepted temporal fills, yellow boxes are vehicle-validated
applied plates, and orange marks unresolved person or plate evidence. The
overview is not sufficient on its own: inspect the normal-speed full-resolution
result for crowded, distant, upper/lower, seam, and otherwise ambiguous scenes.
Record the burned-in `SOURCE` time of any identifiable person or plate that is
not blurred.

Do not rerun the expensive person scan for a manual correction. Create a JSON
list of narrow, normalized `[x1, y1, x2, y2]` boxes with time bounds on the
original source-video clock, then run `policy` and `render` again with new,
nonexistent revision directories. For example, a correction file can contain:

```json
[
  {
    "id": "miss_001",
    "category": "manual_privacy",
    "box": [0.10, 0.20, 0.14, 0.31],
    "start_s": 123.0,
    "end_s": 124.0,
    "reason": "identifiable person missed during review"
  }
]
```

Apply it without rerunning either detector:

```bash
bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:person_anonymize_video -- \
  policy \
  --scan_dir /data/farfield_matching/raw_material/COLLECTION/processed/revisions/DATASET_NAME_person_scan_v1 \
  --plate_manifest /data/farfield_matching/raw_material/COLLECTION/processed/revisions/DATASET_NAME_plate_scan_v1/anonymization_manifest.json \
  --output_dir /data/farfield_matching/raw_material/COLLECTION/processed/revisions/DATASET_NAME_person_policy_correction_01 \
  --manual_regions /data/farfield_matching/raw_material/COLLECTION/manual_regions/DATASET_NAME_correction_01.json \
  --flow_width 960
```

Render that corrected policy to `DATASET_NAME_person_render_correction_01` with
the production `render` command above and review both videos again. An approval
bound to an earlier manifest does not carry forward.

An approval or correction decision is additive and never rewrites the evidence
manifest:

```bash
bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:anonymize_video -- \
  mark-review \
  --output_dir /data/farfield_matching/raw_material/COLLECTION/processed/revisions/DATASET_NAME_person_render_v1/anonymization \
  --decision approved --reviewer "reviewer name" --note "watched end to end"
```

This writes `review_decision.json`, bound to the manifest hash. A rendered
dataset with no approved decision is mechanically complete but remains
privacy-review pending and must not be described as privacy-cleared.

## Finalize and ingest

After rendering **and recording an approved human review**, bind the anonymized
video, every extracted JPEG digest, and the ledger to ingest metadata. Finalize
fails closed when `review_decision.json` is absent or requests corrections:

```bash
bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:prepare_selfcollect -- \
  finalize --config /data/farfield_matching/raw_material/<collect>/collection.json

bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:ingest_selfcollect -- \
  --source_dir processed/example_leg \
  --output /data/farfield_matching/datasets/example_leg \
  --dataset_id example_leg --width 7680 --height 3840 \
  --raw_material raw_material/<collect> \
  --log_start_utc 2026-01-01T00:00:00Z \
  --extra_metadata processed/example_leg/extra_metadata.json
```

After approving `nominal_forward.json`, render the standard low-resolution
dataset review. It publishes a trajectory plot, GPS-overlay timelapse, and a
north-aligned timelapse under `_manifests/timelapse/`; the last is omitted when
no approved nominal-forward record exists:

```bash
bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:make_dataset_timelapse -- \
  --dataset_path /data/farfield_matching/datasets/example_leg
```

Finish with `farfield:audit_dataset`. The ingest copies only blurred JPEGs;
the original videos remain under `raw_material/` and the separately blurred
3 fps videos remain derived evidence beside the review artifacts. The
north-aligned video is review-only; canonical panoramas stay unrotated.
