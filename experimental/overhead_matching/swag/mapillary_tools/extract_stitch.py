#!/usr/bin/env python3
"""Download one (possibly stitched) sequence from a manifest, in order.

This is the only place that writes `sequence_position` into the sidecar JSON,
and mapillary_to_vigor.py sorts on it. That matters most for stitched
trajectories: each component sequence has its own positions starting at 0, so
they would collide, and the `captured_at` fallback cannot break ties because
Mapillary timestamps repeat within a second.

Resume-safe: existing jpg+json pairs are skipped, and sidecars written by an
older run get `sequence_position` backfilled.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experimental.overhead_matching.swag.mapillary_tools.api import MapillaryClient
from experimental.overhead_matching.swag.mapillary_tools.models import haversine


def decimate_by_spacing(images: list[dict], min_spacing_m: float) -> list[dict]:
    """Keep images at least min_spacing_m apart along the track.

    Applied before download so we never pay bandwidth or disk for frames we
    would discard later. Useful because several of these captures are video
    extractions at 3-30 fps where consecutive frames share a GPS fix and show
    essentially the same view.
    """
    if min_spacing_m <= 0 or not images:
        return images
    kept = [images[0]]
    last = images[0]
    for img in images[1:]:
        d = haversine(last["lat"], last["lng"], img["lat"], img["lng"]) * 1000
        if d >= min_spacing_m:
            kept.append(img)
            last = img
    return kept


def looks_like_complete_jpeg(data: bytes) -> bool:
    """SOI/EOI marker check, so a truncated download fails loudly here.

    Without this a short read is written to disk and only surfaces much later as
    a decode error inside the converter or the VLM stage.
    """
    return len(data) > 1024 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"


def jpeg_dimensions(data: bytes):
    """(width, height) of an in-memory JPEG, or None if it cannot be read."""
    try:
        from io import BytesIO

        from PIL import Image
        with Image.open(BytesIO(data)) as im:
            return im.size
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="Path to a manifest JSON")
    parser.add_argument("--sequence", required=True,
                        help="Sequence id within the manifest (e.g. a trajectory name)")
    parser.add_argument("--output", required=True, help="Output directory for images")
    parser.add_argument("--workers", type=int, default=8, help="Download workers")
    parser.add_argument("--max_width", type=int, default=4096,
                        help="Cap on stored width; fetch the 2048 thumbnail instead of a "
                             "much larger original when that still satisfies the cap "
                             "(default: 4096, 0 disables)")
    parser.add_argument("--min_spacing_m", type=float, default=0.0,
                        help="Drop frames closer together than this along the track, "
                             "before downloading (default: 0, keep all)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would be downloaded and exit")
    args = parser.parse_args()

    with open(args.manifest) as f:
        data = json.load(f)

    seq = next((s for s in data["sequences"] if s["id"] == args.sequence), None)
    if seq is None:
        available = ", ".join(s["id"] for s in data["sequences"][:10])
        print(f"Sequence '{args.sequence}' not found in {args.manifest}")
        print(f"Available: {available}")
        sys.exit(1)

    images = seq["images"]
    print(f"Sequence: {seq['id']}")
    print(f"Images:   {len(images)}")
    print(f"Length:   {seq['length_km']:.2f} km")

    if args.min_spacing_m > 0:
        before = len(images)
        images = decimate_by_spacing(images, args.min_spacing_m)
        print(f"Spacing:  kept {len(images)}/{before} frames at >={args.min_spacing_m}m apart")

    out_dir = Path(args.output)
    max_width = args.max_width or None

    if args.dry_run:
        print(f"[DRY RUN] would download {len(images)} images to {out_dir} "
              f"(max_width={max_width})")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    client = MapillaryClient()
    downloaded = failed = skipped = 0
    start = time.time()
    failures = []

    def download_one(seq_pos, img):
        img_id = img["id"]
        ts = img.get("captured_at", 0) // 1000 if img.get("captured_at") else 0
        lat, lng = img["lat"], img["lng"]
        heading = img.get("computed_compass_angle") or img.get("compass_angle", 0)
        fname = f"{img_id}_lat{lat:.6f}_lng{lng:.6f}_heading{heading:.1f}_ts{ts}"
        jpg_path = out_dir / f"{fname}.jpg"
        json_path = out_dir / f"{fname}.json"

        if jpg_path.exists() and json_path.exists():
            existing = json.loads(json_path.read_text())
            if existing.get("sequence_position") != seq_pos:
                existing["sequence_position"] = seq_pos
                json_path.write_text(json.dumps(existing, indent=2))
            return "skipped"

        url = client.get_image_url(img_id, max_width=max_width)
        if not url:
            raise RuntimeError(f"no download URL for {img_id}")
        blob = client.download_image(url)
        if not looks_like_complete_jpeg(blob):
            # Mapillary's stored *original* is sometimes a truncated object: on
            # nyc_inner_harbor, 66 of 96 2015-era originals came back
            # byte-identical short reads across repeated attempts, so it is
            # server-side corruption rather than a network hiccup. The 2048
            # thumbnail of the same image is intact, so take the resolution loss
            # instead of losing the frame.
            fallback = client.get_image_url(img_id, max_width=2048)
            if fallback and fallback != url:
                candidate = client.download_image(fallback)
                if looks_like_complete_jpeg(candidate):
                    blob, url = candidate, fallback
        if not looks_like_complete_jpeg(blob):
            raise RuntimeError(f"incomplete JPEG for {img_id} ({len(blob)} bytes)")
        jpg_path.write_bytes(blob)
        record = dict(img)
        record["sequence_position"] = seq_pos
        record["source_url_kind"] = "thumb_2048" if "2048" in url else "original"
        # The sidecar must describe the file it sits next to, not what the API
        # said the original was. Downstream reads width/height from here to write
        # intrinsics.csv, so leaving the original's dimensions on a thumbnail
        # would silently overstate the stored image and corrupt focal-in-pixels.
        actual = jpeg_dimensions(blob)
        if actual and (actual != (img.get("width"), img.get("height"))):
            record["width"], record["height"] = actual
            record["api_width"], record["api_height"] = img.get("width"), img.get("height")
        json_path.write_text(json.dumps(record, indent=2))
        return "ok"

    print(f"\nDownloading {len(images)} images with {args.workers} workers "
          f"(max_width={max_width})...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, i, img): img
                   for i, img in enumerate(images)}
        for future in as_completed(futures):
            try:
                if future.result() == "skipped":
                    skipped += 1
                else:
                    downloaded += 1
            except Exception as e:
                failed += 1
                failures.append((futures[future]["id"], str(e)))
            total = downloaded + failed + skipped
            if total % 100 == 0 or total == len(images):
                el = time.time() - start
                rate = (downloaded + skipped) / el if el > 0 else 0
                print(f"  [{total}/{len(images)}] downloaded={downloaded} "
                      f"skipped={skipped} failed={failed} ({rate:.1f}/s)")

    print(f"\nDone in {time.time()-start:.1f}s. {downloaded} downloaded, "
          f"{skipped} skipped, {failed} failed.")
    if failures:
        # A partial trajectory would convert without complaint and be wrong, so
        # make this a non-zero exit rather than a line in the scrollback.
        print(f"\n{len(failures)} failure(s); first 10:")
        for img_id, err in failures[:10]:
            print(f"  {img_id}: {err}")
        print("Re-run to retry (existing files are skipped).")
        sys.exit(1)


if __name__ == "__main__":
    main()
