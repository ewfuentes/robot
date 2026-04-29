"""HTTP server exposing OSM landmark tags via SQL queries.

Loads one or more feather files into an in-memory SQLite database and serves
read-only SQL queries over HTTP. Designed to be a long-lived companion to
``ollama_osm_extraction.py`` so that the expensive feather→index step only
happens once.

Usage::

    bazel run //experimental/overhead_matching/swag/scripts:osm_tag_server -- \
      --feather /data/.../landmarks.feather

    bazel run //experimental/overhead_matching/swag/scripts:osm_tag_server -- \
      --city Chicago --dataset-base /data/overhead_matching/datasets/VIGOR
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
import shapely

from experimental.overhead_matching.swag.scripts.search_osm_tags import (
    EXCLUDE_KEYS,
    META_COLS,
)

DEFAULT_DATASET_BASE = "/data/overhead_matching/datasets/VIGOR"
DEFAULT_FEATHER_NAME = "v4_202001.feather"
MAX_ROWS = 50


def build_database(feather_paths: list[Path]) -> sqlite3.Connection:
    """Load feather file(s) into an in-memory SpatiaLite database."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")
    conn.execute("SELECT InitSpatialMetaData(1)")
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")

    conn.execute(
        """CREATE TABLE tags (
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (key, value)
        )"""
    )
    conn.execute(
        """CREATE TABLE landmark_tags (
            landmark_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE landmarks (
            landmark_id INTEGER PRIMARY KEY
        )"""
    )
    conn.execute(
        "SELECT AddGeometryColumn('landmarks', 'geom', 4326, 'GEOMETRY', 'XY')"
    )

    landmark_id = 0
    tag_counts: dict[tuple[str, str], int] = {}

    for feather_path in feather_paths:
        print(f"Loading {feather_path}...", file=sys.stderr)
        df = pd.read_feather(feather_path)
        tag_cols = [
            c
            for c in df.columns
            if c not in META_COLS and c not in EXCLUDE_KEYS
        ]
        has_geom = "geometry" in df.columns
        print(
            f"  {len(df)} landmarks, {len(tag_cols)} tag columns"
            f"{', with geometry' if has_geom else ''}",
            file=sys.stderr,
        )

        for _, row in df.iterrows():
            row_tags = []
            for col in tag_cols:
                if pd.notna(row[col]):
                    v = str(row[col])
                    if v.startswith("http") or len(v) > 100:
                        continue
                    row_tags.append((col, v))
                    tag_counts[(col, v)] = tag_counts.get((col, v), 0) + 1

            if row_tags:
                for key, value in row_tags:
                    conn.execute(
                        "INSERT INTO landmark_tags (landmark_id, key, value) VALUES (?, ?, ?)",
                        (landmark_id, key, value),
                    )
                # Insert geometry if available
                if has_geom and pd.notna(row["geometry"]):
                    wkb = row["geometry"]
                    conn.execute(
                        "INSERT INTO landmarks (landmark_id, geom) "
                        "VALUES (?, GeomFromWKB(?, 4326))",
                        (landmark_id, wkb),
                    )
                else:
                    conn.execute(
                        "INSERT INTO landmarks (landmark_id) VALUES (?)",
                        (landmark_id,),
                    )
                landmark_id += 1

    # Populate tags summary table
    for (key, value), count in tag_counts.items():
        conn.execute(
            "INSERT INTO tags (key, value, count) VALUES (?, ?, ?)",
            (key, value, count),
        )

    conn.execute("CREATE INDEX idx_lt_key_value ON landmark_tags(key, value)")
    conn.execute(
        "CREATE INDEX idx_lt_key_value_lid ON landmark_tags(key, value, landmark_id)"
    )
    conn.execute("CREATE INDEX idx_lt_landmark ON landmark_tags(landmark_id)")
    conn.execute(
        "SELECT CreateSpatialIndex('landmarks', 'geom')"
    )

    # Add MBR extent columns for efficient B-tree spatial joins.
    # SpatiaLite's SpatialIndex virtual table doesn't integrate well with
    # SQLite's query planner for correlated joins. Plain REAL columns with
    # B-tree indexes let SQLite use native BETWEEN range scans, which is
    # orders of magnitude faster for spatial joins between filtered sets.
    for col in ("min_x", "max_x", "min_y", "max_y"):
        conn.execute(f"ALTER TABLE landmarks ADD COLUMN {col} REAL")
    conn.execute(
        """UPDATE landmarks SET
            min_x = MbrMinX(geom),
            max_x = MbrMaxX(geom),
            min_y = MbrMinY(geom),
            max_y = MbrMaxY(geom)
        WHERE geom IS NOT NULL"""
    )
    conn.execute("CREATE INDEX idx_lm_minx ON landmarks(min_x)")
    conn.execute("CREATE INDEX idx_lm_maxx ON landmarks(max_x)")
    conn.execute("CREATE INDEX idx_lm_miny ON landmarks(min_y)")
    conn.execute("CREATE INDEX idx_lm_maxy ON landmarks(max_y)")

    # Register meters_to_deg() so queries can use meters instead of degrees.
    # Returns a conservative (overestimated) degree padding for a given radius
    # at a given latitude.
    def _meters_to_deg(meters, lat_deg):
        lat_r = 1.0 / 111320.0
        lon_r = 1.0 / (111320.0 * max(math.cos(math.radians(lat_deg)), 0.01))
        return max(lat_r, lon_r) * meters

    conn.create_function("meters_to_deg", 2, _meters_to_deg)

    conn.commit()

    # Make read-only
    conn.execute("PRAGMA query_only = ON")

    # Print stats
    n_tags = conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
    n_lt = conn.execute("SELECT COUNT(*) FROM landmark_tags").fetchone()[0]
    n_landmarks = conn.execute(
        "SELECT COUNT(DISTINCT landmark_id) FROM landmark_tags"
    ).fetchone()[0]
    n_geom = conn.execute(
        "SELECT COUNT(*) FROM landmarks WHERE geom IS NOT NULL"
    ).fetchone()[0]
    print(
        f"Database ready: {n_tags} unique tags, {n_landmarks} landmarks, "
        f"{n_lt} landmark-tag rows, {n_geom} with geometry",
        file=sys.stderr,
    )

    return conn


def make_handler(conn: sqlite3.Connection):
    """Create a request handler class with access to the database connection."""

    class QueryHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/query":
                self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                request = json.loads(body)
            except json.JSONDecodeError as e:
                self._send_json({"error": f"Invalid JSON: {e}"}, 400)
                return

            sql = request.get("sql")
            if not sql:
                self._send_json({"error": "Missing 'sql' field"}, 400)
                return

            # Block non-SELECT statements
            stripped = sql.strip().upper()
            if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
                self._send_json(
                    {"error": "Only SELECT queries are allowed"}, 400
                )
                return

            try:
                cursor = conn.execute(sql)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchmany(MAX_ROWS)
                self._send_json(
                    {"columns": columns, "rows": [list(r) for r in rows]}
                )
            except sqlite3.Error as e:
                self._send_json({"error": str(e)}, 400)

        def _send_json(self, data: dict, status: int = 200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            print(f"  [{self.address_string()}] {format % args}", file=sys.stderr)

    return QueryHandler


def resolve_feather(
    dataset_base: Path, city: str, feather_name: str = DEFAULT_FEATHER_NAME
) -> Path:
    p = dataset_base / city / "landmarks" / feather_name
    if not p.exists():
        raise FileNotFoundError(f"Feather file not found: {p}")
    return p


def main():
    parser = argparse.ArgumentParser(
        description="HTTP server for OSM tag SQL queries"
    )
    parser.add_argument(
        "--feather",
        type=Path,
        nargs="+",
        default=None,
        help="One or more feather file paths",
    )
    parser.add_argument(
        "--city",
        type=str,
        nargs="+",
        default=None,
        help="City name(s) to auto-resolve feather files",
    )
    parser.add_argument(
        "--dataset-base",
        type=Path,
        default=Path(DEFAULT_DATASET_BASE),
        help=f"Base dataset directory (default: {DEFAULT_DATASET_BASE})",
    )
    parser.add_argument(
        "--feather-name",
        type=str,
        default=DEFAULT_FEATHER_NAME,
        help=f"Feather filename under <city>/landmarks/ (default: {DEFAULT_FEATHER_NAME})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8421,
        help="Server port (default: 8421)",
    )
    args = parser.parse_args()

    # Resolve feather paths
    feather_paths = []
    if args.feather:
        feather_paths.extend(args.feather)
    if args.city:
        for city in args.city:
            feather_paths.append(
                resolve_feather(args.dataset_base, city, args.feather_name)
            )
    if not feather_paths:
        parser.error("Provide --feather or --city")

    conn = build_database(feather_paths)
    handler = make_handler(conn)

    server = HTTPServer(("127.0.0.1", args.port), handler)
    print(f"Serving on http://localhost:{args.port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)
        server.server_close()


if __name__ == "__main__":
    main()
