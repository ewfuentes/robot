"""Authoritative persisted schema for far-field landmark catalogs.

Far-field catalogs require four compact columns: id, geometry, landmark_type,
and tags. Tags is canonical JSON text containing one object whose keys and
values are strings. A small, explicit set of source-provenance columns may be
stored alongside those required columns; all other landmark attributes belong
inside tags. Keeping that distinction here prevents a reader from accepting a
non-compact one-column-per-tag representation.
"""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import geopandas as gpd
import pandas as pd

SCHEMA_VERSION = 1
FULL_ARTIFACT_SCHEMA = "farfield_full_catalog/v1"
TAGS_COLUMN = "tags"
META_COLUMNS = ("id", "geometry", "landmark_type", TAGS_COLUMN)
REQUIRED_COLUMNS = frozenset(META_COLUMNS)
OPTIONAL_STRUCTURAL_COLUMNS = frozenset({"object_class"})
ALLOWED_COLUMNS = REQUIRED_COLUMNS | OPTIONAL_STRUCTURAL_COLUMNS
ALLOWED_LANDMARK_TYPES = frozenset({"osm", "enc", "overture"})


class CatalogSchemaError(ValueError):
    """A landmark frame does not satisfy the far-field catalog contract."""


def _where(context: str | Path | None) -> str:
    return f" in {context}" if context is not None else ""


def _decode_tags(value, row_index,
                 context: str | Path | None = None) -> dict[str, str]:
    """Decode and validate one persisted JSON tags cell."""
    where = _where(context)
    if not isinstance(value, str):
        raise CatalogSchemaError(
            f"row {row_index!r} tags must be JSON object text, got "
            f"{type(value).__name__}{where}")

    def reject_duplicate_keys(pairs):
        decoded = {}
        for key, tag_value in pairs:
            if key in decoded:
                raise CatalogSchemaError(
                    f"row {row_index!r} tags contains duplicate key {key!r}"
                    f"{where}")
            decoded[key] = tag_value
        return decoded

    try:
        decoded = json.loads(value, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise CatalogSchemaError(
            f"row {row_index!r} tags is invalid JSON: {exc.msg}"
            f"{where}") from exc
    if not isinstance(decoded, dict):
        raise CatalogSchemaError(
            f"row {row_index!r} tags must decode to a JSON object, got "
            f"{type(decoded).__name__}{where}")

    for key, tag_value in decoded.items():
        if not isinstance(key, str) or not key:
            raise CatalogSchemaError(
                f"row {row_index!r} tag keys must be non-empty strings"
                f"{where}")
        if key in REQUIRED_COLUMNS:
            raise CatalogSchemaError(
                f"row {row_index!r} tag {key!r} collides with a structural "
                f"catalog field{where}")
        if not isinstance(tag_value, str):
            raise CatalogSchemaError(
                f"row {row_index!r} tag {key!r} must have a string value, "
                f"got {type(tag_value).__name__}{where}")
    return decoded


def _is_null_scalar(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _validate_frame(frame: pd.DataFrame,
                    context: str | Path | None = None) -> list[dict[str, str]]:
    """Validate a compact frame and return its decoded tag objects."""
    columns = set(frame.columns)
    missing = REQUIRED_COLUMNS - columns
    if TAGS_COLUMN in missing:
        raise CatalogSchemaError(
            f"wide landmark schema is not supported{_where(context)}: "
            "the required JSON 'tags' column is missing. Regenerate or "
            "explicitly convert this Feather with catalog.schema.build_frame")
    if missing:
        raise CatalogSchemaError(
            f"catalog is missing required columns {sorted(missing)}"
            f"{_where(context)}")
    unexpected = columns - ALLOWED_COLUMNS
    if unexpected:
        raise CatalogSchemaError(
            f"catalog has unexpected columns {sorted(unexpected)}"
            f"{_where(context)}; compact far-field catalogs store every tag "
            "inside the JSON 'tags' object. Regenerate the Feather")

    if not isinstance(frame, gpd.GeoDataFrame):
        raise CatalogSchemaError(
            f"catalog must be a GeoDataFrame with an active geometry column"
            f"{_where(context)}")
    if frame.crs is None:
        raise CatalogSchemaError(
            f"catalog CRS is missing{_where(context)}; assign the source CRS "
            "before writing the Feather")
    try:
        geometry = frame.geometry
    except (AttributeError, ValueError) as exc:
        raise CatalogSchemaError(
            f"catalog has no active geometry column{_where(context)}") from exc
    if geometry.isna().any():
        rows = frame.index[geometry.isna()].tolist()[:5]
        raise CatalogSchemaError(
            f"catalog geometry is null at rows {rows}{_where(context)}")
    if geometry.is_empty.any():
        rows = frame.index[geometry.is_empty].tolist()[:5]
        raise CatalogSchemaError(
            f"catalog geometry is empty at rows {rows}{_where(context)}")
    if (~geometry.is_valid).any():
        rows = frame.index[~geometry.is_valid].tolist()[:5]
        raise CatalogSchemaError(
            f"catalog geometry is invalid at rows {rows}{_where(context)}")

    ids = frame["id"]
    if ids.isna().any():
        rows = frame.index[ids.isna()].tolist()[:5]
        raise CatalogSchemaError(
            f"catalog id is null at rows {rows}{_where(context)}")
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise CatalogSchemaError(
            f"catalog ids must be non-empty strings{_where(context)}")
    duplicates = ids[ids.duplicated(keep=False)]
    if not duplicates.empty:
        values = list(dict.fromkeys(duplicates.astype(str).tolist()))[:5]
        raise CatalogSchemaError(
            f"catalog ids must be unique; duplicates include {values}"
            f"{_where(context)}")

    invalid_sources = []
    for value in frame["landmark_type"]:
        if (not isinstance(value, str)
                or value not in ALLOWED_LANDMARK_TYPES):
            invalid_sources.append(repr(value))
    invalid_sources = sorted(set(invalid_sources))
    if invalid_sources:
        raise CatalogSchemaError(
            "landmark_type must be one of "
            f"{sorted(ALLOWED_LANDMARK_TYPES)}; found "
            f"{invalid_sources}{_where(context)}")

    decoded_tags = [
        _decode_tags(value, row_index, context=context)
        for row_index, value in frame[TAGS_COLUMN].items()
    ]

    if "object_class" in columns:
        for position, (row_index, value) in enumerate(
                frame["object_class"].items()):
            if not _is_null_scalar(value):
                if not isinstance(value, str) or not value.strip():
                    raise CatalogSchemaError(
                        "object_class values must be non-empty strings or "
                        f"null; row {row_index!r} has {value!r}"
                        f"{_where(context)}")
            mirrored = decoded_tags[position].get("object_class")
            if mirrored is not None and mirrored != value:
                raise CatalogSchemaError(
                    f"row {row_index!r} tag 'object_class' does not match "
                    f"the structural object_class value {value!r}"
                    f"{_where(context)}")

    return decoded_tags


def tag_dicts(frame: pd.DataFrame) -> list[dict[str, str]]:
    """Return validated tag objects from a compact far-field frame."""
    return _validate_frame(frame)


def _as_list(name: str, values: Iterable) -> list:
    try:
        return list(values)
    except TypeError as exc:
        raise CatalogSchemaError(f"{name} must be iterable") from exc


def build_frame(ids, geometries, landmark_types, tags,
                crs="EPSG:4326") -> gpd.GeoDataFrame:
    """Build and validate a compact far-field catalog GeoDataFrame."""
    ids = _as_list("ids", ids)
    geometries = _as_list("geometries", geometries)
    landmark_types = _as_list("landmark_types", landmark_types)
    tags = _as_list("tags", tags)
    lengths = {
        "ids": len(ids),
        "geometries": len(geometries),
        "landmark_types": len(landmark_types),
        "tags": len(tags),
    }
    if len(set(lengths.values())) != 1:
        raise CatalogSchemaError(
            f"catalog columns must have equal lengths, got {lengths}")
    if crs is None:
        raise CatalogSchemaError("catalog CRS must be known")

    encoded_tags = []
    for row_index, mapping in enumerate(tags):
        if not isinstance(mapping, Mapping):
            raise CatalogSchemaError(
                f"row {row_index} tags must be a mapping, got "
                f"{type(mapping).__name__}")
        for key, tag_value in mapping.items():
            if not isinstance(key, str) or not key:
                raise CatalogSchemaError(
                    f"row {row_index} tag keys must be non-empty strings")
            if key in REQUIRED_COLUMNS:
                raise CatalogSchemaError(
                    f"row {row_index} tag {key!r} collides with a "
                    "structural catalog field")
            if not isinstance(tag_value, str):
                raise CatalogSchemaError(
                    f"row {row_index} tag {key!r} must have a string value, "
                    f"got {type(tag_value).__name__}")
        # Round-trip through the same strict decoder used by readers. This
        # catches non-string values before a catalog reaches disk.
        try:
            encoded = json.dumps(dict(mapping), sort_keys=True,
                                 separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise CatalogSchemaError(
                f"row {row_index} tags cannot be encoded as JSON: {exc}"
            ) from exc
        _decode_tags(encoded, row_index)
        encoded_tags.append(encoded)

    try:
        frame = gpd.GeoDataFrame(
            {
                "id": ids,
                "geometry": geometries,
                "landmark_type": landmark_types,
                TAGS_COLUMN: encoded_tags,
            },
            geometry="geometry",
            crs=crs,
        )
    except (TypeError, ValueError) as exc:
        raise CatalogSchemaError(
            f"cannot build catalog GeoDataFrame: {exc}") from exc
    _validate_frame(frame)
    return frame


def summarize(frame: pd.DataFrame) -> str:
    """Return a one-line summary after enforcing the compact schema."""
    tags = _validate_frame(frame)
    count = sum(len(mapping) for mapping in tags)
    return (f"{len(frame)} landmarks, compact JSON-tags schema v"
            f"{SCHEMA_VERSION}, {count} tag values")


def read_frame(path: Path) -> gpd.GeoDataFrame:
    """Read and validate a compact far-field landmark Feather."""
    path = Path(path)
    frame = gpd.read_feather(path)
    _validate_frame(frame, context=path)
    return frame
