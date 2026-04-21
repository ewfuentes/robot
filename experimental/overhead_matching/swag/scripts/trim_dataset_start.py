"""Trim the first N panoramas from a VIGOR dataset.

Moves trimmed panorama images to pretrim/panorama/, backs up and rewrites
CSV files, and updates evaluation path JSON files.

Usage:
    python trim_dataset_start.py \
        --dataset_path /data/overhead_matching/datasets/VIGOR/Boston \
        --num_frames 35 \
        --eval_paths /data/overhead_matching/evaluation/paths/mappilary/Boston.json
"""

import argparse
import csv
import json
import shutil
from pathlib import Path


def read_pano_ids_from_csv(csv_path: Path) -> list[str]:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return [row["pano_id"] for row in reader]


def backup_and_trim_csv(csv_path: Path, pretrim_dir: Path, num_frames: int):
    pretrim_dir.mkdir(parents=True, exist_ok=True)
    backup = pretrim_dir / csv_path.name
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}")

    with open(csv_path) as f:
        lines = f.readlines()

    # lines[0] is header, lines[1:] are data rows
    header = lines[0]
    data = lines[1:]
    print(f"  {csv_path.name}: {len(data)} rows → {len(data) - num_frames} rows")

    shutil.copy2(csv_path, backup)
    with open(csv_path, "w") as f:
        f.write(header)
        f.writelines(data[num_frames:])


def backup_and_trim_extraction_log(log_path: Path, pretrim_dir: Path, trim_ids: set[str]):
    """Trim extraction_log.csv by filtering out rows with trimmed pano_ids (uuid column)."""
    if not log_path.exists():
        return
    if log_path.is_symlink():
        print(f"  {log_path.name}: symlink, skipping (handled by pano_id_mapping.csv)")
        return

    pretrim_dir.mkdir(parents=True, exist_ok=True)
    backup = pretrim_dir / log_path.name
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}")

    with open(log_path) as f:
        lines = f.readlines()

    header = lines[0]
    # Determine the ID column name (uuid or pano_id)
    col_names = header.strip().split(",")
    id_col_idx = None
    for i, name in enumerate(col_names):
        if name in ("uuid", "pano_id"):
            id_col_idx = i
            break
    if id_col_idx is None:
        print(f"  {log_path.name}: no uuid/pano_id column found, skipping")
        return

    kept = []
    removed = 0
    for line in lines[1:]:
        row_id = line.split(",")[id_col_idx]
        if row_id in trim_ids:
            removed += 1
        else:
            kept.append(line)

    print(f"  {log_path.name}: removed {removed} rows, {len(kept)} remaining")
    shutil.copy2(log_path, backup)
    with open(log_path, "w") as f:
        f.write(header)
        f.writelines(kept)


def move_panorama_images(pano_dir: Path, pretrim_dir: Path, trim_ids: set[str]):
    pretrim_pano = pretrim_dir / "panorama"
    pretrim_pano.mkdir(parents=True, exist_ok=True)

    # Resolve symlinks for the panorama directory
    real_pano_dir = pano_dir.resolve()

    moved = 0
    for f in sorted(real_pano_dir.iterdir()):
        pano_id = f.name.split(",")[0]
        if pano_id in trim_ids:
            dest = pretrim_pano / f.name
            if dest.exists():
                raise FileExistsError(f"Already exists: {dest}")
            shutil.move(str(f), str(dest))
            moved += 1

    print(f"  Moved {moved} panorama images to {pretrim_pano}")


def trim_eval_paths(eval_path: Path, pretrim_dir: Path, trim_ids: set[str]):
    if not eval_path.exists():
        print(f"  {eval_path}: not found, skipping")
        return

    pretrim_dir.mkdir(parents=True, exist_ok=True)
    backup = pretrim_dir / eval_path.name
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}")

    with open(eval_path) as f:
        data = json.load(f)

    original_total = sum(len(p) for p in data["paths"])

    new_paths = []
    for path_list in data["paths"]:
        trimmed = [pid for pid in path_list if pid not in trim_ids]
        if trimmed:
            new_paths.append(trimmed)

    new_total = sum(len(p) for p in new_paths)
    print(f"  {eval_path.name}: {len(data['paths'])} paths ({original_total} pano_ids) → {len(new_paths)} paths ({new_total} pano_ids)")

    shutil.copy2(eval_path, backup)
    data["paths"] = new_paths
    data["dataset_hash"] = "stale"
    with open(eval_path, "w") as f:
        json.dump(data, f, indent=2)


def rename_eval_path(eval_path: Path):
    if not eval_path.exists():
        print(f"  {eval_path}: not found, skipping")
        return
    new_name = eval_path.parent / f"old_{eval_path.name}"
    if new_name.exists():
        raise FileExistsError(f"Already exists: {new_name}")
    eval_path.rename(new_name)
    print(f"  Renamed {eval_path.name} → {new_name.name}")


def main():
    parser = argparse.ArgumentParser(description="Trim first N panoramas from a VIGOR dataset")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--eval_paths", type=str, nargs="*", default=[],
                        help="Evaluation path JSON files to trim")
    parser.add_argument("--rename_eval_paths", type=str, nargs="*", default=[],
                        help="Evaluation path JSON files to rename with old_ prefix")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset_path)
    pretrim = dataset / "pretrim"
    num = args.num_frames

    # Step 1: Read pano_ids to trim
    csv_path = dataset / "pano_id_mapping.csv"
    all_ids = read_pano_ids_from_csv(csv_path)
    trim_ids = set(all_ids[:num])
    print(f"Trimming first {num} panoramas from {dataset.name}")
    print(f"  Trim set: {list(trim_ids)[:3]}... ({len(trim_ids)} total)")
    print(f"  New start: {all_ids[num]}")
    print()

    if args.dry_run:
        print("DRY RUN — no changes made")
        return

    # Step 2: Trim CSV files
    print("Trimming CSV files:")
    backup_and_trim_csv(csv_path, pretrim, num)
    extraction_log = dataset / "extraction_log.csv"
    backup_and_trim_extraction_log(extraction_log, pretrim, trim_ids)
    print()

    # Step 3: Move panorama images
    print("Moving panorama images:")
    pano_dir = dataset / "panorama"
    move_panorama_images(pano_dir, pretrim, trim_ids)
    print()

    # Step 4: Trim evaluation paths
    if args.eval_paths or args.rename_eval_paths:
        print("Updating evaluation paths:")
        eval_pretrim = Path("/data/overhead_matching/evaluation/paths/pretrim")
        for ep in args.eval_paths:
            trim_eval_paths(Path(ep), eval_pretrim, trim_ids)
        for ep in args.rename_eval_paths:
            rename_eval_path(Path(ep))
        print()

    print("Done. To undo, restore files from pretrim/ directories.")


if __name__ == "__main__":
    main()
