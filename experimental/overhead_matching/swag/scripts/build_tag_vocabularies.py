"""Build tag vocabularies from landmarks SQLite database for classification tasks."""

import argparse
import json
import sqlite3
from pathlib import Path


def build_tag_vocabularies_from_db(
    db_path: Path,
    classification_tags: list[str],
    min_count: int = 100,
) -> dict[str, dict[str, int]]:
    """Build class vocabularies directly from database.

    Args:
        db_path: Path to landmarks SQLite database
        classification_tags: List of tag keys to build vocabularies for
        min_count: Minimum count for a value to be included in vocabulary

    Returns:
        Dictionary mapping tag keys to {value: class_index} dictionaries
    """
    conn = sqlite3.connect(db_path)

    vocabularies = {}

    for tag_key in classification_tags:
        print(f"  Building vocabulary for '{tag_key}'...")

        # Count occurrences of each value for this tag key
        query = """
            SELECT tv.value, COUNT(*) as cnt
            FROM tags t
            JOIN tag_keys tk ON t.key_id = tk.id
            JOIN tag_values tv ON t.value_id = tv.id
            WHERE tk.key = ?
            GROUP BY tv.value
            HAVING cnt >= ?
            ORDER BY cnt DESC
        """
        cursor = conn.execute(query, (tag_key, min_count))

        values = []
        total_count = 0
        for row in cursor:
            values.append(row[0])
            total_count += row[1]

        if values:
            vocabularies[tag_key] = {v: i for i, v in enumerate(values)}
            print(f"    {len(values)} classes, {total_count:,} samples")
        else:
            print(f"    No values found with count >= {min_count}")

    conn.close()
    return vocabularies


def get_database_stats(db_path: Path) -> dict:
    """Get basic statistics about the database.

    Args:
        db_path: Path to landmarks database

    Returns:
        Dictionary with database statistics
    """
    conn = sqlite3.connect(db_path)

    stats = {}

    # Total landmarks
    cursor = conn.execute("SELECT COUNT(*) FROM landmarks")
    stats["num_landmarks"] = cursor.fetchone()[0]

    # Total unique tag keys
    cursor = conn.execute("SELECT COUNT(*) FROM tag_keys")
    stats["num_tag_keys"] = cursor.fetchone()[0]

    # Total unique tag values
    cursor = conn.execute("SELECT COUNT(*) FROM tag_values")
    stats["num_tag_values"] = cursor.fetchone()[0]

    # Top tag keys by count
    cursor = conn.execute("""
        SELECT tk.key, COUNT(*) as cnt
        FROM tags t
        JOIN tag_keys tk ON t.key_id = tk.id
        GROUP BY tk.key
        ORDER BY cnt DESC
        LIMIT 20
    """)
    stats["top_tag_keys"] = [(row[0], row[1]) for row in cursor]

    conn.close()
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build tag vocabularies from landmarks SQLite database"
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        required=True,
        help="Path to landmarks SQLite database",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output tag_vocabs.json file",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=100,
        help="Minimum count for a value to be included (default: 100)",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        help="Only build vocabularies for these tags. If not specified, uses defaults.",
    )
    parser.add_argument(
        "--show_stats",
        action="store_true",
        help="Show database statistics before building vocabularies",
    )
    args = parser.parse_args()

    # Default classification tags
    default_tags = [
        "amenity",
        "building",
        "highway",
        "shop",
        "leisure",
        "tourism",
        "landuse",
        "natural",
        "surface",
        "cuisine",
    ]

    classification_tags = args.tags if args.tags else default_tags

    print(f"Database: {args.db_path}")

    if args.show_stats:
        print("\nDatabase statistics:")
        stats = get_database_stats(args.db_path)
        print(f"  Total landmarks: {stats['num_landmarks']:,}")
        print(f"  Unique tag keys: {stats['num_tag_keys']:,}")
        print(f"  Unique tag values: {stats['num_tag_values']:,}")
        print("\n  Top tag keys:")
        for key, count in stats["top_tag_keys"]:
            print(f"    {key}: {count:,}")

    print(f"\nBuilding vocabularies for tags: {classification_tags}")
    print(f"Minimum count: {args.min_count}")

    vocabularies = build_tag_vocabularies_from_db(
        args.db_path,
        classification_tags=classification_tags,
        min_count=args.min_count,
    )

    # Save to JSON
    print(f"\nSaving vocabularies to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(vocabularies, f, indent=2)

    # Print summary
    print("\nVocabulary summary:")
    for tag, vocab in sorted(vocabularies.items()):
        print(f"  {tag}: {len(vocab)} classes")

    total_classes = sum(len(v) for v in vocabularies.values())
    print(f"\nTotal: {len(vocabularies)} tags, {total_classes} classes")


if __name__ == "__main__":
    main()
