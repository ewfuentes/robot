"""Extract OSM landmarks from multiple cities and write directly to SQLite.

This script processes cities in chunks to reduce PBF file reads while still
allowing parallelization. Each chunk extracts multiple city bboxes in a single
PBF pass, then writes results to a normalized SQLite database.
"""

import argparse
import multiprocessing as mp
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely import wkb

from common.openstreetmap import extract_landmarks_python as elm


# --- Data Classes ---


@dataclass
class CityInfo:
    """Information about a city to extract."""
    city_name: str
    state_id: str
    state_name: str
    bbox: tuple[float, float, float, float]  # left, bottom, right, top


@dataclass
class ChunkTask:
    """A chunk of cities to extract from a single PBF file."""
    pbf_path: Path
    state_id: str
    cities: list[CityInfo]


# --- Geometry Utilities ---


def create_shapely_geometry(geom):
    """Convert C++ geometry variant to Shapely geometry."""
    if isinstance(geom, elm.PointGeometry):
        return Point(geom.coord.lon, geom.coord.lat)
    elif isinstance(geom, elm.LineStringGeometry):
        return LineString([(c.lon, c.lat) for c in geom.coords])
    elif isinstance(geom, elm.PolygonGeometry):
        exterior = [(c.lon, c.lat) for c in geom.exterior]
        holes = [[(c.lon, c.lat) for c in hole] for hole in geom.holes]
        return Polygon(exterior, holes if holes else None)
    elif isinstance(geom, elm.MultiPolygonGeometry):
        polygons = []
        for poly in geom.polygons:
            exterior = [(c.lon, c.lat) for c in poly.exterior]
            holes = [[(c.lon, c.lat) for c in hole] for hole in poly.holes]
            polygons.append(Polygon(exterior, holes if holes else None))
        return MultiPolygon(polygons)
    else:
        raise ValueError(f"Unknown geometry type: {type(geom)}")


def get_geometry_type(geom) -> str:
    """Get string name of geometry type."""
    if isinstance(geom, elm.PointGeometry):
        return "Point"
    elif isinstance(geom, elm.LineStringGeometry):
        return "LineString"
    elif isinstance(geom, elm.PolygonGeometry):
        return "Polygon"
    elif isinstance(geom, elm.MultiPolygonGeometry):
        return "MultiPolygon"
    else:
        return "Unknown"


def get_centroid(geom) -> tuple[float, float]:
    """Get centroid (lon, lat) of a geometry."""
    if isinstance(geom, elm.PointGeometry):
        return (geom.coord.lon, geom.coord.lat)
    elif isinstance(geom, elm.LineStringGeometry):
        if geom.coords:
            # Use first coordinate as approximate centroid
            return (geom.coords[0].lon, geom.coords[0].lat)
        return (0.0, 0.0)
    elif isinstance(geom, elm.PolygonGeometry):
        if geom.exterior:
            # Use Shapely to compute centroid
            shapely_geom = create_shapely_geometry(geom)
            centroid = shapely_geom.centroid
            return (centroid.x, centroid.y)
        return (0.0, 0.0)
    elif isinstance(geom, elm.MultiPolygonGeometry):
        if geom.polygons:
            shapely_geom = create_shapely_geometry(geom)
            centroid = shapely_geom.centroid
            return (centroid.x, centroid.y)
        return (0.0, 0.0)
    return (0.0, 0.0)


# --- City Loading and Grouping ---


def load_cities(
    city_bboxes_yaml: Path,
    states: list[str] | None = None,
    cities: list[str] | None = None,
) -> list[CityInfo]:
    """Load and filter cities from the YAML file."""
    with open(city_bboxes_yaml, "r") as f:
        data = yaml.safe_load(f)

    city_list = []
    for c in data["cities"]:
        bbox = c["bbox"]
        city_list.append(
            CityInfo(
                city_name=c["city"],
                state_id=c["state_id"],
                state_name=c["state"],
                bbox=(bbox["min_lng"], bbox["min_lat"], bbox["max_lng"], bbox["max_lat"]),
            )
        )

    print(f"Loaded {len(city_list)} cities from {city_bboxes_yaml}")

    if states:
        states_set = set(states)
        city_list = [c for c in city_list if c.state_id in states_set]
        print(f"Filtered to {len(city_list)} cities in states: {states}")

    if cities:
        cities_set = set(cities)
        city_list = [c for c in city_list if c.city_name in cities_set]
        print(f"Filtered to {len(city_list)} cities: {cities}")

    return city_list


def state_name_to_slug(state_name: str) -> str:
    """Convert state name to the slug used in OSM PBF filenames."""
    return state_name.lower().replace(" ", "-")


def find_pbf_for_state(state_name: str, osm_dumps_dir: Path) -> Path | None:
    """Find the PBF file for a given state."""
    state_slug = state_name_to_slug(state_name)
    pbf_pattern = f"{state_slug}-*.osm.pbf"
    pbf_files = list(osm_dumps_dir.glob(pbf_pattern))
    return pbf_files[0] if pbf_files else None


def group_cities_by_state(
    city_list: list[CityInfo], osm_dumps_dir: Path
) -> dict[str, tuple[Path, list[CityInfo]]]:
    """Group cities by their state PBF file."""
    groups: dict[str, tuple[Path, list[CityInfo]]] = {}
    missing_pbfs: set[str] = set()
    pbf_cache: dict[str, Path | None] = {}

    for city in city_list:
        if city.state_name not in pbf_cache:
            pbf_cache[city.state_name] = find_pbf_for_state(city.state_name, osm_dumps_dir)

        pbf_path = pbf_cache[city.state_name]
        if pbf_path is None:
            missing_pbfs.add(city.state_name)
            continue

        key = city.state_id
        if key not in groups:
            groups[key] = (pbf_path, [])
        groups[key][1].append(city)

    for state_name in sorted(missing_pbfs):
        print(f"WARNING: No PBF file found for {state_name}")

    return groups


def chunk_cities(cities: list[CityInfo], chunk_size: int) -> Iterator[list[CityInfo]]:
    """Yield chunks of cities."""
    for i in range(0, len(cities), chunk_size):
        yield cities[i : i + chunk_size]


def build_chunk_tasks(
    city_list: list[CityInfo], osm_dumps_dir: Path, chunk_size: int
) -> list[ChunkTask]:
    """Build list of chunk tasks to process."""
    groups = group_cities_by_state(city_list, osm_dumps_dir)

    tasks = []
    for state_id, (pbf_path, cities) in groups.items():
        for city_chunk in chunk_cities(cities, chunk_size):
            tasks.append(ChunkTask(pbf_path=pbf_path, state_id=state_id, cities=city_chunk))

    return tasks


# --- SQLite Database ---


def create_database(db_path: Path) -> None:
    """Create the SQLite database with schema.

    Landmarks are deduplicated by their filtered tag signature. Each row represents
    a unique combination of semantically-meaningful tags (excluding metadata like
    tiger:*, gnis:*, source, etc.).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Main landmarks table (deduplicated by tag signature)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS landmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag_signature TEXT NOT NULL UNIQUE,
            count INTEGER NOT NULL DEFAULT 1,
            representative_osm_type TEXT NOT NULL CHECK(representative_osm_type IN ('node', 'way', 'relation')),
            representative_osm_id INTEGER NOT NULL,
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            geometry_type TEXT NOT NULL,
            geometry_wkb BLOB,
            centroid_lon REAL NOT NULL,
            centroid_lat REAL NOT NULL
        )
    """)

    # Tag keys lookup table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tag_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE
        )
    """)

    # Tag values lookup table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tag_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            value TEXT NOT NULL UNIQUE
        )
    """)

    # Tags table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            landmark_id INTEGER NOT NULL,
            key_id INTEGER NOT NULL,
            value_id INTEGER NOT NULL,
            PRIMARY KEY (landmark_id, key_id),
            FOREIGN KEY (landmark_id) REFERENCES landmarks(id) ON DELETE CASCADE,
            FOREIGN KEY (key_id) REFERENCES tag_keys(id),
            FOREIGN KEY (value_id) REFERENCES tag_values(id)
        )
    """)

    conn.commit()
    conn.close()


def create_indexes(db_path: Path) -> None:
    """Create indexes after bulk insert for better performance."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print("Creating indexes...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_landmarks_city_state ON landmarks(state, city)")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_landmarks_centroid ON landmarks(centroid_lon, centroid_lat)"
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_landmarks_geometry_type ON landmarks(geometry_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_landmarks_count ON landmarks(count)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tags_key_id ON tags(key_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tags_value_id ON tags(value_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tags_key_value ON tags(key_id, value_id)")

    conn.commit()
    conn.close()
    print("Indexes created.")


# --- Extraction and Writing ---


# Tag filters matching original script
TAG_FILTERS = {
    "amenity": True,
    "building": True,
    "tourism": True,
    "shop": True,
    "craft": True,
    "emergency": True,
    "geological": True,
    "highway": True,
    "historic": True,
    "landuse": True,
    "leisure": True,
    "man_made": True,
    "military": True,
    "natural": True,
    "office": True,
    "power": True,
    "public_transport": True,
    "railway": True,
}

# Tag prefixes to exclude for deduplication (pure metadata, not visible)
EXCLUDED_TAG_PREFIXES = (
    "tiger:",      # TIGER import metadata
    "gnis:",       # GNIS import metadata
    "source:",     # Source metadata
    "brand:",      # Brand wikidata links (brand name itself is kept)
    "payment:",    # Payment methods
    "contact:",    # Contact info
    "ref:",        # Internal reference IDs (ref:walmart, ref:penndot, etc.)
)

# Specific tags to exclude for deduplication
EXCLUDED_TAGS = frozenset((
    "source",
    "created_by",
    "wikidata",
    "wikipedia",
    "website",
    "phone",
    "fax",
    "email",
    "opening_hours",
    "unsigned_ref",
    "gtfs_id",
    "ntd_id",
    "ele",
    "note",
    "fixme",
    "FIXME",
    "description",
    "is_in",
    "import_uuid",
    "layer",
))


def is_excluded_tag(tag_key: str) -> bool:
    """Check if a tag should be excluded from deduplication."""
    if tag_key in EXCLUDED_TAGS:
        return True
    for prefix in EXCLUDED_TAG_PREFIXES:
        if tag_key.startswith(prefix):
            return True
    return False


def filter_tags(tags: dict[str, str]) -> dict[str, str]:
    """Filter out excluded tags for deduplication."""
    return {k: v for k, v in tags.items() if not is_excluded_tag(k)}


def compute_tag_signature(tags: dict[str, str]) -> str:
    """Compute a deterministic signature for a set of tags."""
    import hashlib
    filtered = filter_tags(tags)
    tag_str = "|".join(f"{k}={v}" for k, v in sorted(filtered.items()))
    return hashlib.sha256(tag_str.encode()).hexdigest()[:16]


OSM_TYPE_MAP = {
    elm.OsmType.NODE: "node",
    elm.OsmType.WAY: "way",
    elm.OsmType.RELATION: "relation",
}


@dataclass
class ChunkResult:
    """Result from processing a chunk of cities."""
    state_id: str
    city_names: list[str]
    city_results: list[tuple[str, str, list]]  # (city_name, state_id, features)
    total_features: int


def process_chunk(task: ChunkTask) -> ChunkResult:
    """Process a chunk of cities. Returns ChunkResult with metadata."""
    # Build bboxes dict
    bboxes = {}
    city_lookup = {}
    for city in task.cities:
        key = f"{city.state_id}_{city.city_name}"
        bboxes[key] = elm.BoundingBox(*city.bbox)
        city_lookup[key] = city

    # Extract all cities in one PBF pass
    results = elm.extract_landmarks(str(task.pbf_path), bboxes, TAG_FILTERS)

    # Group results by city
    city_features: dict[str, list] = {key: [] for key in bboxes}
    for region_id, feature in results:
        city_features[region_id].append(feature)

    # Build output
    city_results = []
    total_features = 0
    for key, features in city_features.items():
        city = city_lookup[key]
        city_results.append((city.city_name, city.state_id, features))
        total_features += len(features)

    return ChunkResult(
        state_id=task.state_id,
        city_names=[c.city_name for c in task.cities],
        city_results=city_results,
        total_features=total_features,
    )


def write_results_to_db(
    db_path: Path,
    chunk_result: ChunkResult,
    store_geometry: bool,
) -> tuple[int, int, int]:
    """Write extraction results to SQLite database with deduplication.

    Landmarks are deduplicated by their filtered tag signature. If a signature
    already exists, its count is incremented. Only filtered (non-metadata) tags
    are stored.

    Returns (landmarks_added, landmarks_deduplicated, tags_added).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Enable WAL mode for better write performance
    cur.execute("PRAGMA journal_mode=WAL")

    # Cache for key/value IDs
    key_cache: dict[str, int] = {}
    value_cache: dict[str, int] = {}

    def get_or_create_key(key: str) -> int:
        if key not in key_cache:
            cur.execute("INSERT OR IGNORE INTO tag_keys (key) VALUES (?)", (key,))
            cur.execute("SELECT id FROM tag_keys WHERE key = ?", (key,))
            key_cache[key] = cur.fetchone()[0]
        return key_cache[key]

    def get_or_create_value(value: str) -> int:
        if value not in value_cache:
            cur.execute("INSERT OR IGNORE INTO tag_values (value) VALUES (?)", (value,))
            cur.execute("SELECT id FROM tag_values WHERE value = ?", (value,))
            value_cache[value] = cur.fetchone()[0]
        return value_cache[value]

    landmarks_added = 0
    landmarks_deduplicated = 0
    tags_added = 0

    cur.execute("BEGIN TRANSACTION")

    for city_name, state_id, features in chunk_result.city_results:
        for feature in features:
            osm_type = OSM_TYPE_MAP[feature.osm_type]
            geometry_type = get_geometry_type(feature.geometry)
            centroid_lon, centroid_lat = get_centroid(feature.geometry)

            # Compute tag signature for deduplication
            tag_sig = compute_tag_signature(feature.tags)
            filtered_tags = filter_tags(feature.tags)

            # Check if this signature already exists
            cur.execute("SELECT id FROM landmarks WHERE tag_signature = ?", (tag_sig,))
            existing = cur.fetchone()

            if existing:
                # Increment count for existing signature
                cur.execute(
                    "UPDATE landmarks SET count = count + 1 WHERE tag_signature = ?",
                    (tag_sig,),
                )
                landmarks_deduplicated += 1
            else:
                # Insert new landmark
                geometry_blob = None
                if store_geometry:
                    shapely_geom = create_shapely_geometry(feature.geometry)
                    geometry_blob = wkb.dumps(shapely_geom)

                cur.execute(
                    """
                    INSERT INTO landmarks
                    (tag_signature, count, representative_osm_type, representative_osm_id,
                     city, state, geometry_type, geometry_wkb, centroid_lon, centroid_lat)
                    VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tag_sig,
                        osm_type,
                        feature.osm_id,
                        city_name,
                        state_id,
                        geometry_type,
                        geometry_blob,
                        centroid_lon,
                        centroid_lat,
                    ),
                )
                landmark_id = cur.lastrowid
                landmarks_added += 1

                # Insert only filtered tags (no metadata)
                for key, value in filtered_tags.items():
                    key_id = get_or_create_key(key)
                    value_id = get_or_create_value(value)
                    cur.execute(
                        "INSERT INTO tags (landmark_id, key_id, value_id) VALUES (?, ?, ?)",
                        (landmark_id, key_id, value_id),
                    )
                    tags_added += 1

    conn.commit()
    conn.close()

    return landmarks_added, landmarks_deduplicated, tags_added


# --- Main ---


def main(
    city_bboxes_yaml: Path,
    osm_dumps_dir: Path,
    output_db: Path,
    states: list[str] | None = None,
    cities: list[str] | None = None,
    chunk_size: int = 20,
    num_workers: int = 1,
    store_geometry: bool = True,
):
    """Extract landmarks for cities and write to SQLite database.

    Args:
        city_bboxes_yaml: Path to city bounding boxes YAML file.
        osm_dumps_dir: Directory containing OSM PBF files.
        output_db: Output SQLite database path.
        states: Only process these states (by state_id). If None, process all.
        cities: Only process these cities (by name). If None, process all.
        chunk_size: Number of cities to process per PBF read.
        num_workers: Number of parallel workers.
        store_geometry: Whether to store full geometry WKB (increases DB size).
    """
    # Load and filter cities
    city_list = load_cities(city_bboxes_yaml, states, cities)

    if not city_list:
        print("No cities to process.")
        return

    # Build chunk tasks
    tasks = build_chunk_tasks(city_list, osm_dumps_dir, chunk_size)

    # Print summary of what will be processed
    states_summary: dict[str, int] = {}
    for task in tasks:
        states_summary[task.state_id] = states_summary.get(task.state_id, 0) + len(task.cities)

    print(f"\nWill process {len(city_list)} cities across {len(states_summary)} states:")
    for state_id, count in sorted(states_summary.items()):
        print(f"  {state_id}: {count} cities")
    print(f"\nCreated {len(tasks)} chunk tasks (chunk_size={chunk_size})\n")

    # Create database
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        print(f"Database already exists: {output_db}")
        print("Appending to existing database...")
    else:
        print(f"Creating database: {output_db}")
    create_database(output_db)

    # Process chunks
    total_landmarks = 0
    total_deduplicated = 0
    total_tags = 0

    def format_city_list(cities: list[str], max_display: int = 3) -> str:
        """Format city list for display, truncating if too long."""
        if len(cities) <= max_display:
            return ", ".join(cities)
        return ", ".join(cities[:max_display]) + f", ... (+{len(cities) - max_display} more)"

    if num_workers == 1:
        for i, task in enumerate(tasks):
            city_str = format_city_list([c.city_name for c in task.cities])
            print(f"[{i+1}/{len(tasks)}] Extracting {task.state_id}: {city_str}...")
            chunk_result = process_chunk(task)
            landmarks, deduped, tags = write_results_to_db(output_db, chunk_result, store_geometry)
            total_landmarks += landmarks
            total_deduplicated += deduped
            total_tags += tags
            print(f"  -> {chunk_result.total_features} features, {landmarks} new, {deduped} deduplicated")
    else:
        with mp.Pool(num_workers) as pool:
            for i, chunk_result in enumerate(pool.imap_unordered(process_chunk, tasks)):
                city_str = format_city_list(chunk_result.city_names)
                landmarks, deduped, tags = write_results_to_db(output_db, chunk_result, store_geometry)
                total_landmarks += landmarks
                total_deduplicated += deduped
                total_tags += tags
                print(
                    f"[{i+1}/{len(tasks)}] {chunk_result.state_id}: {city_str} "
                    f"-> {chunk_result.total_features} features, {landmarks} new, {deduped} deduped"
                )

    # Create indexes after bulk insert
    create_indexes(output_db)

    total_features = total_landmarks + total_deduplicated
    print(f"\nDone!")
    print(f"  Total features processed: {total_features:,}")
    print(f"  Unique landmarks: {total_landmarks:,}")
    print(f"  Deduplicated: {total_deduplicated:,} ({total_deduplicated/total_features*100:.1f}%)")
    print(f"  Tags stored: {total_tags:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract OSM landmarks for cities and write to SQLite database"
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
        "--output_db",
        type=Path,
        default=Path("/data/overhead_matching/datasets/landmarks.db"),
        help="Output SQLite database path",
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
        "--chunk_size",
        type=int,
        default=20,
        help="Number of cities to process per PBF read (default: 20)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, recommended for SQLite)",
    )
    parser.add_argument(
        "--no_geometry",
        action="store_true",
        help="Don't store full geometry WKB (only centroids, reduces DB size)",
    )
    args = parser.parse_args()

    main(
        city_bboxes_yaml=args.city_bboxes_yaml,
        osm_dumps_dir=args.osm_dumps_dir,
        output_db=args.output_db,
        states=args.states,
        cities=args.cities,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        store_geometry=not args.no_geometry,
    )
