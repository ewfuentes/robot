"""Write the keyframe JPEGs a `prepare_selfcollect plan` asks for, from a video.

The plan's `frames_gps.csv` names each keyframe (`frame_file`) and the index of
the video frame it is (`frame_index`, on the video's own frame grid). The
anonymization render normally does this extraction; a collection that waives
anonymization (aerial imagery) needs it done directly, with the same
accounting: one sequential decode, every JPEG hashed into a manifest.

  extract_plan_frames --video derivative.mp4 --plan processed/<leg>/frames_gps.csv \
      --output_dir processed/<leg>
writes <output_dir>/frames/<frame_file> and <output_dir>/extraction_manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

from experimental.overhead_matching.swag.farfield import artifact, code_provenance

SCHEMA = "farfield_plan_frame_extraction/v1"


def read_plan(plan: Path) -> dict[int, str]:
    wanted: dict[int, str] = {}
    with plan.open(newline="") as fh:
        for row in csv.DictReader(fh):
            index = int(row["frame_index"])
            name = row["frame_file"]
            if Path(name).name != name or not name:
                raise ValueError(f"frame_file must be a bare filename: {name!r}")
            if index in wanted or name in wanted.values():
                raise ValueError(f"duplicate frame_index/frame_file in plan: {index} {name}")
            wanted[index] = name
    if not wanted:
        raise ValueError(f"plan has no rows: {plan}")
    return wanted


def extract(frames, wanted: dict[int, str], frames_dir: Path, jpeg_quality: int) -> dict[str, str]:
    """Iterate (index, bgr) pairs once; write the wanted ones; return name->sha256."""
    frames_dir.mkdir(parents=True, exist_ok=False)
    digests: dict[str, str] = {}
    last = max(wanted)
    for index, bgr in frames:
        name = wanted.get(index)
        if name is not None:
            out = frames_dir / name
            if not cv2.imwrite(str(out), bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                raise RuntimeError(f"failed to write {out}")
            digests[name] = artifact.sha256_file(out)
        if index >= last:
            break
    missing = sorted(set(wanted) - {i for i in wanted if wanted[i] in digests})
    if missing:
        raise RuntimeError(f"video ended before plan frames {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return digests


def video_frames(video: Path):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open {video}")
    try:
        index = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                return
            yield index, bgr
            index += 1
    finally:
        cap.release()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True, help="frames_gps.csv from prepare_selfcollect plan")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--jpeg_quality", type=int, default=92)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.output_dir / "extraction_manifest.json"
    frames_dir = args.output_dir / "frames"
    if manifest_path.exists() or frames_dir.exists():
        raise FileExistsError(f"refusing to replace {frames_dir} / {manifest_path}")
    wanted = read_plan(args.plan)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    digests = extract(video_frames(args.video), wanted, frames_dir, args.jpeg_quality)
    manifest = {
        "schema": SCHEMA,
        "video": {"path": str(args.video.resolve()), "sha256": artifact.sha256_file(args.video),
                  "media_fps": fps},
        "plan": {"path": str(args.plan.resolve()), "sha256": artifact.sha256_file(args.plan),
                 "rows": len(wanted)},
        "jpeg_quality": args.jpeg_quality,
        "frames_dir": frames_dir.name,
        "frame_count": len(digests),
        "frame_sha256": dict(sorted(digests.items())),
        "code_provenance": code_provenance.record(),
    }
    artifact.atomic_write_json(manifest_path, manifest)
    print(json.dumps({k: manifest[k] for k in ("frame_count", "jpeg_quality")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
