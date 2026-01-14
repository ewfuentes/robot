"""Extract OSM landmarks for all cities in the city bboxes YAML file.

Reads the city bounding boxes YAML file and extracts landmarks for each city
by calling the extract_landmarks_historical main function.
"""

import argparse
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import yaml

from experimental.overhead_matching.swag.scripts.extract_landmarks_historical import (
    main as extract_landmarks_main,
)


@dataclass
class CityTask:
    """A task to extract landmarks for a single city."""

    city_name: str
    state_id: str
    pbf_path: Path
    bbox: tuple[float, float, float, float]
    output_path: Path


def state_name_to_slug(state_name: str) -> str:
    """Convert state name to the slug used in OSM PBF filenames."""
    return state_name.lower().replace(" ", "-")


def load_cities(
    city_bboxes_yaml: Path,
    states: list[str] | None = None,
    cities: list[str] | None = None,
) -> list[dict]:
    """Load and filter cities from the YAML file."""
    with open(city_bboxes_yaml, "r") as f:
        data = yaml.safe_load(f)

    city_list = data["cities"]
    print(f"Loaded {len(city_list)} cities from {city_bboxes_yaml}")

    if states:
        states_set = set(states)
        city_list = [c for c in city_list if c["state_id"] in states_set]
        print(f"Filtered to {len(city_list)} cities in states: {states}")

    if cities:
        cities_set = set(cities)
        city_list = [c for c in city_list if c["city"] in cities_set]
        print(f"Filtered to {len(city_list)} cities: {cities}")

    return city_list


def find_pbf_for_state(state_name: str, osm_dumps_dir: Path) -> Path | None:
    """Find the PBF file for a given state."""
    state_slug = state_name_to_slug(state_name)
    pbf_pattern = f"{state_slug}-*.osm.pbf"
    pbf_files = list(osm_dumps_dir.glob(pbf_pattern))
    return pbf_files[0] if pbf_files else None


def get_output_path(state_id: str, city_name: str, output_dir: Path) -> Path:
    """Generate the output path for a city's landmarks file."""
    safe_city_name = city_name.lower().replace(" ", "_").replace(".", "")
    output_filename = f"{state_id.lower()}_{safe_city_name}"
    return output_dir / output_filename


def bbox_dict_to_tuple(bbox: dict) -> tuple[float, float, float, float]:
    """Convert bbox dict to tuple (left, bottom, right, top)."""
    return (
        bbox["min_lng"],  # left
        bbox["min_lat"],  # bottom
        bbox["max_lng"],  # right
        bbox["max_lat"],  # top
    )


def process_city(task: CityTask) -> tuple[str, str, bool, str | None]:
    """Process a single city. Returns (city_name, state_id, success, error_msg)."""
    try:
        extract_landmarks_main(
            pbf_path=task.pbf_path,
            dataset_path=None,
            bbox=task.bbox,
            zoom_level=20,
            output_path=task.output_path,
        )
        return (task.city_name, task.state_id, True, None)
    except Exception as e:
        return (task.city_name, task.state_id, False, str(e))


def build_tasks(
    city_list: list[dict],
    osm_dumps_dir: Path,
    output_dir: Path,
    skip_existing: bool,
) -> tuple[list[CityTask], set[str]]:
    """Build list of tasks to process, filtering out missing PBFs and existing outputs."""
    tasks = []
    missing_pbfs: set[str] = set()
    pbf_cache: dict[str, Path | None] = {}

    for city_data in city_list:
        state_name = city_data["state"]
        state_id = city_data["state_id"]
        city_name = city_data["city"]
        bbox = city_data["bbox"]

        output_path = get_output_path(state_id, city_name, output_dir)

        if skip_existing and output_path.with_suffix(".feather").exists():
            continue

        # Cache PBF lookups by state
        if state_name not in pbf_cache:
            pbf_cache[state_name] = find_pbf_for_state(state_name, osm_dumps_dir)

        pbf_path = pbf_cache[state_name]
        if pbf_path is None:
            missing_pbfs.add(state_name)
            continue

        tasks.append(
            CityTask(
                city_name=city_name,
                state_id=state_id,
                pbf_path=pbf_path,
                bbox=bbox_dict_to_tuple(bbox),
                output_path=output_path,
            )
        )

    return tasks, missing_pbfs


def main(
    city_bboxes_yaml: Path,
    osm_dumps_dir: Path,
    output_dir: Path,
    states: list[str] | None = None,
    cities: list[str] | None = None,
    skip_existing: bool = False,
    num_workers: int = 1,
):
    """Extract landmarks for cities in the YAML file.

    Args:
        city_bboxes_yaml: Path to city bounding boxes YAML file.
        osm_dumps_dir: Directory containing OSM PBF files.
        output_dir: Output directory for landmark files.
        states: Only process these states (by state_id). If None, process all.
        cities: Only process these cities (by name). If None, process all.
        skip_existing: Skip cities that already have output files.
        num_workers: Number of parallel workers.
    """
    city_list = load_cities(city_bboxes_yaml, states, cities)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks, missing_pbfs = build_tasks(city_list, osm_dumps_dir, output_dir, skip_existing)

    for state_name in sorted(missing_pbfs):
        print(f"WARNING: No PBF file found for {state_name}")

    print(f"Processing {len(tasks)} cities with {num_workers} workers...")

    if num_workers == 1:
        # Single-threaded for easier debugging
        for i, task in enumerate(tasks):
            print(f"[{i+1}/{len(tasks)}] Processing {task.city_name}, {task.state_id}...")
            city_name, state_id, success, error_msg = process_city(task)
            if not success:
                print(f"ERROR processing {city_name}, {state_id}: {error_msg}")
    else:
        with mp.Pool(num_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(process_city, tasks)):
                city_name, state_id, success, error_msg = result
                status = "done" if success else f"ERROR: {error_msg}"
                print(f"[{i+1}/{len(tasks)}] {city_name}, {state_id}: {status}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract OSM landmarks for all cities in the YAML file"
    )
    parser.add_argument(
        "--city_bboxes_yaml",
        type=Path,
        default=Path("/data/overhead_matching/datasets/us_city_bboxes.yaml"),
        help="Path to city bounding boxes YAML file",
    )
    parser.add_argument(
        "--osm_dumps_dir",
        type=Path,
        default=Path("/data/overhead_matching/datasets/osm_dumps"),
        help="Directory containing OSM PBF files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/data/overhead_matching/datasets/city_landmarks"),
        help="Output directory for landmark files",
    )
    parser.add_argument(
        "--states",
        nargs="*",
        help="Only process these states (by state_id, e.g., CA NY). If not specified, process all.",
    )
    parser.add_argument(
        "--cities",
        nargs="*",
        help="Only process these cities (by name). If not specified, process all.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip cities that already have output files",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count(),
        help=f"Number of parallel workers (default: {mp.cpu_count()})",
    )
    args = parser.parse_args()

    main(
        city_bboxes_yaml=args.city_bboxes_yaml,
        osm_dumps_dir=args.osm_dumps_dir,
        output_dir=args.output_dir,
        states=args.states,
        cities=args.cities,
        skip_existing=args.skip_existing,
        num_workers=args.num_workers,
    )
