#!/usr/bin/env python3
"""Scan an area for Mapillary panorama sequences or fetch a single sequence by ID."""

import argparse
import json
import re
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from experimental.overhead_matching.swag.mapillary_tools.api import MapillaryClient
from experimental.overhead_matching.swag.mapillary_tools.geocode import geocode_city
from experimental.overhead_matching.swag.mapillary_tools.models import BBox, PanoImage, PanoSequence, haversine
from experimental.overhead_matching.swag.mapillary_tools.tiling import adaptive_subdivide, generate_tiles


def parse_date_ms(s: str) -> int:
    dt = datetime.strptime(s, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name.split(",")[0].strip())


def parse_resolution(s: str) -> tuple[int, int]:
    """Parse '5760x2880' into (width, height)."""
    parts = s.lower().split("x")
    return int(parts[0]), int(parts[1])


def image_filename(img: PanoImage) -> str:
    ts = img.captured_at // 1000 if img.captured_at else 0
    heading = img.computed_compass_angle or img.compass_angle
    return f"{img.id}_lat{img.lat:.6f}_lng{img.lng:.6f}_heading{heading:.1f}_ts{ts}.jpg"


def download_images(client: MapillaryClient, images: list[PanoImage], out_dir: Path, workers: int):
    """Download images to out_dir with progress."""
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    failed = 0
    start = time.time()

    def download_one(img):
        fname = image_filename(img)
        jpg_path = out_dir / fname
        json_path = out_dir / fname.replace(".jpg", ".json")
        url = client.get_image_url(img.id)
        if not url:
            raise RuntimeError(f"No download URL for {img.id}")
        data = client.download_image(url)
        jpg_path.write_bytes(data)
        json_path.write_text(json.dumps(img.to_dict(), indent=2))
        return img.id

    print(f"\nDownloading {len(images)} images with {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, img): img for img in images}
        for future in as_completed(futures):
            try:
                future.result()
                downloaded += 1
            except Exception as e:
                failed += 1
                print(f"  Failed {futures[future].id}: {e}")
            total_done = downloaded + failed
            if total_done % 50 == 0 or total_done == len(images):
                elapsed = time.time() - start
                rate = downloaded / elapsed if elapsed > 0 else 0
                print(f"  [{total_done}/{len(images)}] downloaded={downloaded} failed={failed} ({rate:.1f} img/s)")

    print(f"Done. {downloaded} downloaded, {failed} failed in {time.time()-start:.1f}s")


def write_sequence_manifest(path: str, sequences: list[PanoSequence], bbox: BBox = None,
                            area_name: str = "", after_ms=None, before_ms=None,
                            min_length_km=0, min_images=2, min_resolution=None,
                            full_sequences=False, extra: dict = None):
    total_images = sum(s.image_count for s in sequences)
    total_length = sum(s.length_km for s in sequences)
    manifest = {
        "metadata": {
            "area_name": area_name,
            "bbox": bbox.to_dict() if bbox else None,
            "total_sequences": len(sequences),
            "total_images": total_images,
            "total_length_km": round(total_length, 2),
            "filters": {
                "after": after_ms,
                "before": before_ms,
                "min_length_km": min_length_km,
                "min_images": min_images,
                "min_resolution": min_resolution,
                "full_sequences": full_sequences,
            },
            "created_at": int(time.time() * 1000),
        },
        "sequences": [s.to_dict() for s in sequences],
    }
    if extra:
        manifest.update(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2))


def format_date(ms):
    if not ms:
        return "N/A"
    return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d")


def print_sequence_summary(seq: PanoSequence):
    print(f"  Sequence: {seq.id}")
    print(f"  Images:   {seq.image_count}")
    print(f"  Length:   {seq.length_km:.2f} km")
    print(f"  Dates:    {format_date(seq.start_time)} — {format_date(seq.end_time)}")
    print(f"  Cameras:  {', '.join(seq.camera_types) or 'N/A'}")
    if seq.images:
        print(f"  Resolution: {seq.min_width}x{seq.min_height}")


def mode_single_sequence(args, client: MapillaryClient):
    """Fetch one or more sequences by ID."""
    seq_ids = args.id
    print(f"Fetching {len(seq_ids)} sequence(s) with {args.workers} workers...\n")

    sequences = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(client.get_full_sequence, sid): sid for sid in seq_ids}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                seq = future.result()
                if seq.images:
                    sequences.append(seq)
                    print_sequence_summary(seq)
                    print()
                else:
                    print(f"  {sid}: no images found\n")
            except Exception as e:
                print(f"  {sid}: error — {e}\n")

    if not sequences:
        print("No valid sequences found.")
        return

    # Sort by the order they were requested
    id_order = {sid: i for i, sid in enumerate(seq_ids)}
    sequences.sort(key=lambda s: id_order.get(s.id, len(seq_ids)))

    total_images = sum(s.image_count for s in sequences)
    total_length = sum(s.length_km for s in sequences)
    print(f"Total: {len(sequences)} sequences, {total_images} images, {total_length:.1f} km")

    # Output
    if len(seq_ids) == 1:
        default_path = f"manifests/seq_{seq_ids[0]}.json"
    else:
        default_path = f"manifests/{len(seq_ids)}_sequences.json"

    if args.output:
        out_path = args.output if not args.download else None
        out_dir = Path(args.output) if args.download else None
    else:
        out_path = default_path
        out_dir = None

    if not args.download:
        out_path = out_path or default_path
        write_sequence_manifest(out_path, sequences, area_name=f"{len(sequences)}_sequences")
        print(f"\nManifest written to {out_path}")
    else:
        out_dir = out_dir or Path(f"images/{'_'.join(seq_ids[:3])}")
        manifest_path = str(out_dir / "manifest.json")
        out_dir.mkdir(parents=True, exist_ok=True)
        write_sequence_manifest(manifest_path, sequences, area_name=f"{len(sequences)}_sequences")
        print(f"Manifest written to {manifest_path}")
        all_imgs = [img for s in sequences for img in s.images]
        download_images(client, all_imgs, out_dir, args.workers)


def find_stitch_groups(sequences: list[PanoSequence], max_time_s: float, max_dist_m: float) -> list[list[dict]]:
    """Find groups of sequences that are likely splits of a longer continuous capture.

    Returns list of groups, where each group is a list of dicts:
      {"sequence": PanoSequence, "time_gap_s": float|None, "dist_gap_m": float|None}
    """
    if not sequences:
        return []

    # Sort by start_time
    seqs = sorted(sequences, key=lambda s: s.start_time)

    # Build chains greedily
    used = set()
    groups = []

    for i, seq in enumerate(seqs):
        if seq.id in used or not seq.images:
            continue
        chain = [{"sequence": seq, "time_gap_s": None, "dist_gap_m": None}]
        used.add(seq.id)
        tail = seq

        # Try to extend chain
        for j in range(i + 1, len(seqs)):
            candidate = seqs[j]
            if candidate.id in used or not candidate.images:
                continue

            # Time gap: candidate starts after tail ends
            time_gap_s = (candidate.start_time - tail.end_time) / 1000.0
            if time_gap_s < 0 or time_gap_s > max_time_s:
                # If candidate starts way after, no point checking further in this chain
                if time_gap_s > max_time_s:
                    continue
                continue

            # Spatial gap: last image of tail to first image of candidate
            tail_last = tail.images[-1]
            cand_first = candidate.images[0]
            dist_gap_km = haversine(tail_last.lat, tail_last.lng, cand_first.lat, cand_first.lng)
            dist_gap_m = dist_gap_km * 1000.0

            if dist_gap_m > max_dist_m:
                continue

            # Resolution must match
            if (tail.min_width != candidate.min_width or
                    tail.min_height != candidate.min_height):
                continue

            chain.append({
                "sequence": candidate,
                "time_gap_s": round(time_gap_s, 1),
                "dist_gap_m": round(dist_gap_m, 1),
            })
            used.add(candidate.id)
            tail = candidate

        if len(chain) > 1:
            groups.append(chain)

    return groups


def mode_user_sequences(args, client: MapillaryClient):
    """Fetch all sequences by a Mapillary username."""
    username = args.username
    print(f"Fetching images by user '{username}'...")

    after_ms = parse_date_ms(args.after) if args.after else None
    before_ms = parse_date_ms(args.before) if args.before else None
    min_res = parse_resolution(args.min_resolution) if args.min_resolution else None

    images = client.search_images_by_user(username, after_ms=after_ms, before_ms=before_ms)
    if not images:
        print("No images found for this user.")
        return

    # Group by sequence
    seq_groups = defaultdict(list)
    no_seq_count = 0
    for img in images:
        if img.sequence_id:
            seq_groups[img.sequence_id].append(img)
        else:
            no_seq_count += 1

    print(f"\nFound {len(images)} images in {len(seq_groups)} sequences ({no_seq_count} without sequence)")

    # Check for duplicate timestamps
    from collections import Counter
    for seq_id, imgs in seq_groups.items():
        ts_counts = Counter(img.captured_at for img in imgs)
        dupes = sum(1 for c in ts_counts.values() if c > 1)
        if dupes:
            total_tied = sum(c for c in ts_counts.values() if c > 1)
            print(f"  Warning: sequence {seq_id[:22]} has {dupes} duplicate timestamps ({total_tied}/{len(imgs)} images)")

    # Fetch correct sequence order from API for each sequence.
    # Cannot sort by captured_at — timestamps have 1s resolution causing ties.
    print(f"\nFetching sequence order for {len(seq_groups)} sequences...")
    sequences = []
    fetch_errors = 0

    def fetch_ordered_seq(seq_id, imgs):
        img_by_id = {img.id: img for img in imgs}
        ordered_ids = client.get_sequence_image_ids(seq_id)
        ordered_imgs = [img_by_id[iid] for iid in ordered_ids if iid in img_by_id]
        # Include any images not in the API response at the end
        seen = set(ordered_ids)
        for img in imgs:
            if img.id not in seen:
                ordered_imgs.append(img)
        seq = PanoSequence(id=seq_id, images=ordered_imgs)
        seq.compute_length()
        return seq

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_ordered_seq, sid, imgs): sid
                   for sid, imgs in seq_groups.items()}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                sequences.append(future.result())
            except Exception as e:
                fetch_errors += 1
                if done <= 3:
                    print(f"  Error ordering {futures[future]}: {e}")
            if done % 50 == 0 or done == len(seq_groups):
                print(f"  [{done}/{len(seq_groups)}] ordered {len(sequences)} sequences, {fetch_errors} errors")

    # Apply filters
    if args.min_images > 0:
        sequences = [s for s in sequences if s.image_count >= args.min_images]
    if args.min_length > 0:
        sequences = [s for s in sequences if s.length_km >= args.min_length]
    if min_res:
        min_w, min_h = min_res
        sequences = [s for s in sequences if s.min_width >= min_w and s.min_height >= min_h]

    # Sort chronologically
    sequences.sort(key=lambda s: s.start_time)

    # Summary
    total_images = sum(s.image_count for s in sequences)
    total_length = sum(s.length_km for s in sequences)
    print(f"\nAfter filtering: {len(sequences)} sequences, {total_images} images, {total_length:.1f} km total")

    if sequences:
        dates = [s.start_time for s in sequences if s.start_time]
        if dates:
            print(f"Date range: {format_date(min(dates))} — {format_date(max(dates))}")

        # Resolution breakdown
        res_counts = defaultdict(int)
        for s in sequences:
            if s.images:
                res_counts[f"{s.min_width}x{s.min_height}"] += 1
        if res_counts:
            print(f"Resolutions: {', '.join(f'{r} ({c})' for r, c in sorted(res_counts.items(), key=lambda x: -x[1]))}")

        # Per-sequence table
        print(f"\n{'Sequence':<25} {'Images':>6} {'Length':>8} {'Start':>12} {'End':>12} {'Resolution':>12}")
        print("-" * 80)
        for s in sequences:
            res = f"{s.min_width}x{s.min_height}" if s.images else "N/A"
            print(f"{s.id:<25} {s.image_count:>6} {s.length_km:>7.2f}km {format_date(s.start_time):>12} {format_date(s.end_time):>12} {res:>12}")

    # Stitch analysis
    stitch_groups = []
    if args.stitch:
        stitch_groups = find_stitch_groups(sequences, args.stitch_time, args.stitch_dist)
        if stitch_groups:
            print(f"\n{'='*80}")
            print(f"STITCH ANALYSIS: {len(stitch_groups)} stitchable group(s) found")
            print(f"  (max time gap: {args.stitch_time}s, max spatial gap: {args.stitch_dist}m)")
            print(f"{'='*80}")
            for gi, group in enumerate(stitch_groups, 1):
                group_imgs = sum(e["sequence"].image_count for e in group)
                group_len = sum(e["sequence"].length_km for e in group)
                print(f"\n  Group {gi}: {len(group)} sequences, {group_imgs} images, {group_len:.2f} km")
                for ei, entry in enumerate(group):
                    s = entry["sequence"]
                    gap_str = ""
                    if entry["time_gap_s"] is not None:
                        gap_str = f"  [gap: {entry['time_gap_s']}s, {entry['dist_gap_m']}m]"
                    print(f"    {ei+1}. {s.id} ({s.image_count} imgs, {s.length_km:.2f}km, {format_date(s.start_time)}){gap_str}")
        else:
            print("\nNo stitchable sequence groups found.")

    # Output manifest
    if args.output:
        out_path = args.output
    else:
        out_path = f"manifests/user_{sanitize_filename(username)}_sequences.json"

    manifest_extra = {}
    if stitch_groups:
        manifest_extra["stitch_groups"] = [
            {
                "sequences": [e["sequence"].id for e in group],
                "total_images": sum(e["sequence"].image_count for e in group),
                "total_length_km": round(sum(e["sequence"].length_km for e in group), 3),
                "gaps": [
                    {"time_gap_s": e["time_gap_s"], "dist_gap_m": e["dist_gap_m"]}
                    for e in group if e["time_gap_s"] is not None
                ],
            }
            for group in stitch_groups
        ]

    write_sequence_manifest(out_path, sequences, area_name=f"user_{username}",
                            after_ms=after_ms, before_ms=before_ms,
                            min_length_km=args.min_length, min_images=args.min_images,
                            min_resolution=args.min_resolution,
                            full_sequences=True,
                            extra=manifest_extra)
    print(f"\nManifest written to {out_path}")

    # Write a merged-stitch manifest: each stitch group becomes one combined sequence
    if stitch_groups:
        stitched_seqs = []
        stitched_ids = set()
        for gi, group in enumerate(stitch_groups, 1):
            all_imgs = []
            for entry in group:
                all_imgs.extend(entry["sequence"].images)
                stitched_ids.add(entry["sequence"].id)
            # Don't re-sort — each sequence's images are already in correct
            # spatial order from the API, and stitch groups are built in time order
            component_ids = [e["sequence"].id for e in group]
            merged = PanoSequence(id=f"stitch_{gi}__{'_'.join(c[:8] for c in component_ids[:4])}", images=all_imgs)
            merged.compute_length()
            stitched_seqs.append(merged)

        # Also include un-stitched sequences as standalone
        for seq in sequences:
            if seq.id not in stitched_ids:
                stitched_seqs.append(seq)

        stitched_seqs.sort(key=lambda s: s.length_km, reverse=True)

        stitch_path = out_path.replace(".json", "_stitched.json")
        write_sequence_manifest(stitch_path, stitched_seqs, area_name=f"user_{username}_stitched",
                                after_ms=after_ms, before_ms=before_ms,
                                min_length_km=args.min_length, min_images=args.min_images,
                                min_resolution=args.min_resolution,
                                full_sequences=True)
        print(f"Stitched manifest written to {stitch_path} ({len(stitch_groups)} groups merged)")


def mode_area_scan(args, client: MapillaryClient):
    """Discover sequences in a geographic area."""
    # Resolve bbox
    if args.city:
        print(f"Geocoding '{args.city}'...")
        bbox, display_name = geocode_city(args.city)
        area_name = display_name
        print(f"  -> {display_name}")
        print(f"  -> bbox: {bbox.to_string()}")
    else:
        bbox = BBox.from_string(args.bbox)
        area_name = f"bbox_{args.bbox}"

    after_ms = parse_date_ms(args.after) if args.after else None
    before_ms = parse_date_ms(args.before) if args.before else None
    min_res = parse_resolution(args.min_resolution) if args.min_resolution else None

    # Tile and scan
    tiles = generate_tiles(bbox)
    print(f"\nScanning {len(tiles)} tiles with {args.workers} workers...")

    all_images = {}
    lock = threading.Lock()
    completed = [0]
    errors = [0]
    start = time.time()

    def scan_tile(tile):
        def query_fn(b):
            return client.search_panos_with_sequences(b, after_ms=after_ms, before_ms=before_ms)
        try:
            return adaptive_subdivide(tile, query_fn)
        except Exception as e:
            return e

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(scan_tile, tile): i for i, tile in enumerate(tiles)}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                completed[0] += 1
                if isinstance(result, Exception):
                    errors[0] += 1
                else:
                    for img in result:
                        all_images[img.id] = img
                if completed[0] % 50 == 0 or completed[0] == len(tiles):
                    elapsed = time.time() - start
                    rate = completed[0] / elapsed if elapsed > 0 else 0
                    eta = (len(tiles) - completed[0]) / rate if rate > 0 else 0
                    print(f"  [{completed[0]}/{len(tiles)}] {len(all_images)} unique panos, {errors[0]} errors ({elapsed:.0f}s, ETA {eta:.0f}s)")

    images = list(all_images.values())

    # Client-side date filtering
    if after_ms is not None:
        images = [img for img in images if img.captured_at >= after_ms]
    if before_ms is not None:
        images = [img for img in images if img.captured_at <= before_ms]

    # Group by sequence
    seq_groups = defaultdict(list)
    no_seq_count = 0
    for img in images:
        if img.sequence_id:
            seq_groups[img.sequence_id].append(img)
        else:
            no_seq_count += 1

    print(f"\nFound {len(images)} panoramas in {len(seq_groups)} sequences ({no_seq_count} without sequence, {errors[0]} tile errors)")

    # If --full, fetch complete sequences
    if args.full:
        print(f"\nFetching full sequences for {len(seq_groups)} unique sequence IDs...")
        sequences = []
        fetch_errors = 0

        def fetch_seq(seq_id):
            return client.get_full_sequence(seq_id)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(fetch_seq, sid): sid for sid in seq_groups}
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    seq = future.result()
                    if seq.images:
                        sequences.append(seq)
                except Exception as e:
                    fetch_errors += 1
                    if done <= 3:
                        print(f"  Error fetching {futures[future]}: {e}")
                if done % 100 == 0 or done == len(seq_groups):
                    print(f"  [{done}/{len(seq_groups)}] fetched {len(sequences)} sequences, {fetch_errors} errors")
    else:
        # Build sequences from bbox fragments — these are partial (bbox-clipped)
        # so we can't get full sequence order. captured_at sort is best-effort
        # but may scramble tied timestamps. Use --full for correct ordering.
        sequences = []
        for seq_id, imgs in seq_groups.items():
            imgs.sort(key=lambda x: x.captured_at)
            seq = PanoSequence(id=seq_id, images=imgs)
            seq.compute_length()
            sequences.append(seq)

    # Apply filters
    if args.min_images > 0:
        sequences = [s for s in sequences if s.image_count >= args.min_images]
    if args.min_length > 0:
        sequences = [s for s in sequences if s.length_km >= args.min_length]
    if min_res:
        min_w, min_h = min_res
        sequences = [s for s in sequences if s.min_width >= min_w and s.min_height >= min_h]

    # Sort by length descending
    sequences.sort(key=lambda s: s.length_km, reverse=True)

    total_images = sum(s.image_count for s in sequences)
    total_length = sum(s.length_km for s in sequences)
    print(f"\nAfter filtering: {len(sequences)} sequences, {total_images} images, {total_length:.1f} km total")

    if sequences:
        print(f"\nTop sequences:")
        for s in sequences[:5]:
            print(f"  {s.id}: {s.image_count} imgs, {s.length_km:.2f} km, {format_date(s.start_time)}")

    # Output
    if args.output:
        out_path = args.output
    else:
        safe_name = sanitize_filename(area_name)
        out_path = f"manifests/{safe_name}_sequences.json"

    if not args.download:
        write_sequence_manifest(out_path, sequences, bbox, area_name,
                                after_ms, before_ms, args.min_length,
                                args.min_images, args.min_resolution, args.full)
        print(f"\nManifest written to {out_path}")
    else:
        # Write manifest and download
        out_dir = Path(args.output) if args.output else Path(f"images/{sanitize_filename(area_name)}_sequences")
        manifest_path = str(out_dir / "manifest.json")
        out_dir.mkdir(parents=True, exist_ok=True)
        write_sequence_manifest(manifest_path, sequences, bbox, area_name,
                                after_ms, before_ms, args.min_length,
                                args.min_images, args.min_resolution, args.full)
        print(f"Manifest written to {manifest_path}")
        all_imgs = [img for s in sequences for img in s.images]
        download_images(client, all_imgs, out_dir, args.workers)


def main():
    parser = argparse.ArgumentParser(description="Scan Mapillary for panorama sequences or fetch by ID")

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--city", help="City name to geocode (area scan)")
    mode_group.add_argument("--bbox", help="Bounding box: west,south,east,north (area scan)")
    mode_group.add_argument("--id", nargs="+", help="Fetch one or more sequences by ID")
    mode_group.add_argument("--username", help="Fetch all sequences by a Mapillary username")

    # Output
    parser.add_argument("-o", "--output", help="Output path (manifest JSON or image dir with --download)")

    # Area scan filters
    parser.add_argument("--after", help="Only images after this date (YYYY-MM-DD)")
    parser.add_argument("--before", help="Only images before this date (YYYY-MM-DD)")
    parser.add_argument("--min-length", type=float, default=0, help="Minimum path length in km (default: 0)")
    parser.add_argument("--min-images", type=int, default=2, help="Minimum images per sequence (default: 2)")
    parser.add_argument("--min-resolution", help="Minimum resolution e.g. 5760x2880")
    parser.add_argument("--full", action="store_true", help="Fetch complete sequences via sequence API")

    # Stitch analysis (for --username mode)
    parser.add_argument("--stitch", action="store_true", help="Detect stitchable sequence groups (implies --full)")
    parser.add_argument("--stitch-time", type=float, default=300, help="Max time gap in seconds for stitching (default: 300)")
    parser.add_argument("--stitch-dist", type=float, default=100, help="Max spatial gap in meters for stitching (default: 100)")

    # Download
    parser.add_argument("--download", action="store_true", help="Download images after fetching")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")

    args = parser.parse_args()

    client = MapillaryClient()

    if args.id:
        mode_single_sequence(args, client)
    elif args.username:
        mode_user_sequences(args, client)
    else:
        mode_area_scan(args, client)


if __name__ == "__main__":
    main()
