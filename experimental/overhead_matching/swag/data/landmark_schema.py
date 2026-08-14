"""Landmark feather schema: a single `tags` dict column, with wide-table compat.

Landmark feathers used to carry one column per distinct OSM tag key. That table
is enormously wide and almost entirely empty -- a region-sized extract came out
803,717 rows x 1,716 columns with 0.34% of cells non-null -- and no consumer ever
wanted it that way: every reader's first move was to convert each row back into a
dict and drop the nulls (`row.dropna().to_dict()`, or
`frame[tag_columns].to_dict(orient="records")`).

Paying columns x rows to store a handful of values per row cost ~35 GB and about
eight minutes of pure Python on a single extraction, and it forced a rare-column
pruning step that could silently drop tags the pipeline needs. Storing the tags
as one dict column removes all of that: cost becomes proportional to the tag
values that actually exist.

Every feather written before this change is wide, including the ones the paper
results depend on, so readers must handle both. Use `tag_dicts` / `row_dicts`
here rather than touching columns directly, and both layouts work.
"""

import json
from pathlib import Path

import pandas as pd

# The dict column. Its presence is what distinguishes the two layouts.
TAGS_COLUMN = "tags"

# Tags are serialized as a JSON object per row rather than as a native Arrow
# type. A column of Python dicts looks like the obvious choice, but pyarrow
# infers a *struct* from it, whose fields are the union of every key seen -- so
# writing 803,717 rows of dicts produced 1,716 struct fields and 1.38 billion
# values on read-back, exactly the density the dict column exists to avoid. A
# native map<string,string> would also do, but geopandas' to_feather gives no way
# to pin a per-column Arrow type without bypassing its geometry/CRS metadata
# handling, and losing that is a worse trade than a json.loads per row.

# Columns that are structure rather than tags, in either layout.
META_COLUMNS = ("id", "geometry", "landmark_type", TAGS_COLUMN)


def is_dict_schema(frame: pd.DataFrame) -> bool:
    """True if tags live in a single dict column rather than one column each."""
    return TAGS_COLUMN in frame.columns


def tag_key_columns(frame: pd.DataFrame) -> list[str]:
    """Tag-bearing columns of a wide frame (empty for the dict layout)."""
    if is_dict_schema(frame):
        return []
    return [c for c in frame.columns if c not in META_COLUMNS]


def _decode(value) -> dict:
    """A stored tags cell -> dict, accepting JSON text or an in-memory dict."""
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value) if value else {}
    if isinstance(value, dict):
        return value
    # pyarrow hands struct columns back as dicts already; anything else is a bug
    # worth surfacing rather than silently treating as empty.
    raise TypeError(f"unsupported tags cell of type {type(value).__name__}")


def _clean(mapping) -> dict:
    """Drop null values, matching the old `.dropna()` behaviour."""
    if mapping is None:
        return {}
    out = {}
    for key, value in dict(mapping).items():
        if value is None:
            continue
        # pd.isna on a list/array returns an array; those are never tag values,
        # so guard the scalar case only.
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        out[key] = value
    return out


def tag_dicts(frame: pd.DataFrame) -> list[dict]:
    """Per-row tag dicts, without the metadata columns, nulls omitted."""
    if is_dict_schema(frame):
        return [_clean(_decode(t)) for t in frame[TAGS_COLUMN]]
    columns = tag_key_columns(frame)
    if not columns:
        return [{} for _ in range(len(frame))]
    return [_clean(record) for record in frame[columns].to_dict(orient="records")]


def row_dicts(frame: pd.DataFrame) -> list[dict]:
    """Per-row dicts of metadata *and* tags flattened together, nulls omitted.

    This is the shape the old `row.dropna().to_dict()` produced, so call sites
    that fed whole rows to `prune_landmark` keep their exact semantics.
    """
    tags = tag_dicts(frame)
    meta_columns = [c for c in frame.columns
                    if c in META_COLUMNS and c != TAGS_COLUMN]
    if not meta_columns:
        return tags
    meta = frame[meta_columns].to_dict(orient="records")
    return [{**_clean(meta[i]), **tags[i]} for i in range(len(frame))]


def row_dicts_with_index(frame: pd.DataFrame, index_key: str = "index") -> list[dict]:
    """`row_dicts` plus the frame index under `index_key`."""
    out = row_dicts(frame)
    for position, index_value in enumerate(frame.index):
        assert index_key not in out[position], (
            f"{index_key!r} collides with a landmark field")
        out[position][index_key] = index_value
    return out


def build_frame(ids, geometries, landmark_types, tags, crs="EPSG:4326"):
    """A GeoDataFrame in the dict schema.

    Kept here so the two writers (OSM and ENC extraction) cannot drift apart.
    """
    import geopandas as gpd

    return gpd.GeoDataFrame(
        {
            "id": list(ids),
            "geometry": list(geometries),
            "landmark_type": list(landmark_types),
            # Sorted so two identical tag sets serialize identically, which
            # keeps any downstream hashing or diffing stable.
            TAGS_COLUMN: [json.dumps(dict(t) if t else {}, sort_keys=True)
                          for t in tags],
        },
        crs=crs,
    )


def widen(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand a dict-schema frame back to one column per tag key.

    Only for tools that genuinely need the old layout (or to write a feather an
    older checkout can read). This reintroduces the columns x rows cost, so it is
    deliberately explicit rather than something a reader does implicitly.
    """
    if not is_dict_schema(frame):
        return frame
    import numpy as np

    tags = tag_dicts(frame)
    out = frame.drop(columns=[TAGS_COLUMN]).copy()
    per_key: dict[str, tuple[list[int], list]] = {}
    for row, mapping in enumerate(tags):
        for key, value in mapping.items():
            entry = per_key.setdefault(key, ([], []))
            entry[0].append(row)
            entry[1].append(value)
    for key, (indices, values) in per_key.items():
        column = np.full(len(frame), None, dtype=object)
        column[indices] = values
        out[key] = column
    return out


def summarize(frame: pd.DataFrame) -> str:
    """One-line description of a landmark frame's layout, for logs."""
    if is_dict_schema(frame):
        counts = sum(len(t) for t in tag_dicts(frame))
        return (f"{len(frame)} landmarks, dict schema, {counts} tag values")
    columns = tag_key_columns(frame)
    return (f"{len(frame)} landmarks, wide schema, {len(columns)} tag columns")


def read_frame(path: Path) -> pd.DataFrame:
    """Read a landmark feather in either layout, preserving geometry."""
    import geopandas as gpd

    return gpd.read_feather(path)
