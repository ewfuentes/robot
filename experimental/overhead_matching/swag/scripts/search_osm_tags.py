"""Fuzzy search tool for OSM tags, intended to be called by an LLM during
landmark extraction from panoramas.

Provides two tools:
  - ``search_tags``: fuzzy-search for individual key=value tags
  - ``get_tag_context``: given a specific tag, show co-occurring tags

Usage as a standalone test::

    python search_osm_tags.py --feather /path/to/landmarks.feather search "tennis"
    python search_osm_tags.py --feather /path/to/landmarks.feather context sport tennis
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from subprocess import PIPE, run

import pandas as pd

META_COLS = {"id", "geometry", "landmark_type"}

# Keys whose values are URLs, free text, or internal metadata — not useful for
# fuzzy tag search.
EXCLUDE_KEYS = {
    "description", "inscription", "note", "comment", "created_by",
    "source", "source:name", "source_ref",
    "image", "facebook", "website", "website:menu",
    "contact:website", "contact:facebook", "contact:twitter",
    "contact:email", "contact:fax", "contact:google_plus", "contact:phone",
    "brand:website", "brand:wikidata", "brand:wikipedia",
    "wikidata", "wikipedia", "artist:wikidata",
    "tiger:upload_uuid", "tiger:cfcc", "tiger:county", "tiger:reviewed",
    "tiger:source", "tiger:separated", "tiger:tlid",
    "tiger:zip_left", "tiger:zip_right",
    "tiger:name_base", "tiger:name_type", "tiger:name_direction_prefix",
    "gnis:feature_id", "gnis:state_id", "gnis:county_id", "gnis:created",
    "gnis:import_uuid", "gnis:county_name", "gnis:reviewed",
    "gnis:Class", "gnis:County", "gnis:County_num",
    "gnis:ST_alpha", "gnis:ST_num",
    "import_uuid", "destination:lanes",
}


def _fzf_filter(choices: list[str], query: str, limit: int = 20) -> list[str]:
    """Run ``fzf --filter`` for non-interactive fuzzy matching."""
    if not choices:
        return []
    input_bytes = "\n".join(choices).encode()
    result = run(
        ["fzf", "--filter", query],
        input=input_bytes,
        stdout=PIPE,
        stderr=PIPE,
    )
    # fzf returns exit code 1 when there are no matches
    if result.returncode not in (0, 1):
        raise RuntimeError(f"fzf failed: {result.stderr.decode()}")
    lines = result.stdout.decode().strip().split("\n")
    return [l for l in lines[:limit] if l]


class TagSearchIndex:
    """Pre-computed search index built from a city's OSM landmark data."""

    def __init__(self, df: pd.DataFrame) -> None:
        tag_cols = [
            c for c in df.columns
            if c not in META_COLS and c not in EXCLUDE_KEYS
        ]

        # Build value -> set of (key, value) for reverse lookup.
        # We search on values only so fzf isn't confused by key prefixes.
        self._value_to_kvs: dict[str, set[tuple[str, str]]] = {}
        for col in tag_cols:
            for v in df[col].dropna().unique():
                v_str = str(v)
                if v_str.startswith("http") or len(v_str) > 100:
                    continue
                self._value_to_kvs.setdefault(v_str, set()).add((col, v_str))
        self._values: list[str] = sorted(self._value_to_kvs.keys())

        # Per-tag (key=value) occurrence count across all landmarks.
        self._tag_counts: Counter[tuple[str, str]] = Counter()
        for _, row in df.iterrows():
            for col in tag_cols:
                if pd.notna(row[col]):
                    v_str = str(row[col])
                    if not v_str.startswith("http") and len(v_str) <= 100:
                        self._tag_counts[(col, v_str)] += 1

        # For each landmark, store its tag set for co-occurrence lookup.
        self._landmark_tags: list[frozenset[tuple[str, str]]] = []
        for _, row in df.iterrows():
            tags = frozenset(
                (col, str(row[col]))
                for col in tag_cols
                if pd.notna(row[col])
                and not str(row[col]).startswith("http")
                and len(str(row[col])) <= 100
            )
            self._landmark_tags.append(tags)

    def search_tags(
        self, query: str, limit: int = 10
    ) -> list[tuple[str, str, int]]:
        """Fuzzy-search for individual tags matching a query.

        Returns (key, value, count) triples sorted by count descending.
        """
        matched_values = _fzf_filter(self._values, query, limit=50)

        # Collect all (key, value) pairs from matched values.
        seen: set[tuple[str, str]] = set()
        for val in matched_values:
            for kv in self._value_to_kvs.get(val, set()):
                seen.add(kv)

        # Sort by occurrence count.
        results = [
            (k, v, self._tag_counts[(k, v)]) for k, v in seen
        ]
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:limit]

    def get_tag_context(
        self, key: str, value: str, limit: int = 10
    ) -> list[tuple[str, str, int]]:
        """Given a specific tag, show which other individual tags co-occur.

        Finds all landmarks with the given tag, then counts how often each
        *other* tag appears alongside it. Returns (key, value, count) triples
        sorted by count descending.
        """
        target = (key, value)

        co_counts: Counter[tuple[str, str]] = Counter()
        total = 0
        for tag_set in self._landmark_tags:
            if target in tag_set:
                total += 1
                for kv in tag_set:
                    if kv != target:
                        co_counts[kv] += 1

        results = [
            (k, v, count) for (k, v), count in co_counts.most_common(limit)
        ]
        return results, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feather", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="Fuzzy search for tags")
    p_search.add_argument("query", type=str)
    p_search.add_argument("--limit", type=int, default=10)

    p_ctx = sub.add_parser("context", help="Show co-occurring tags")
    p_ctx.add_argument("key", type=str)
    p_ctx.add_argument("value", type=str)
    p_ctx.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    df = pd.read_feather(args.feather)
    print(f"Building index from {len(df)} landmarks...")
    index = TagSearchIndex(df)
    print(f"Index ready: {len(index._values)} unique values\n")

    if args.command == "search":
        results = index.search_tags(args.query, limit=args.limit)
        print(f"Tags matching '{args.query}':")
        for key, value, count in results:
            print(f"  {key}={value}  ({count})")

    elif args.command == "context":
        results, total = index.get_tag_context(
            args.key, args.value, limit=args.limit
        )
        print(f"Tags co-occurring with {args.key}={args.value} "
              f"({total} landmarks have this tag):")
        for key, value, count in results:
            print(f"  {key}={value}  ({count}/{total})")


if __name__ == "__main__":
    main()
