"""Landmark feather schema — re-export of the shared owner.

The dict-tags feather schema is shared with the main VIGOR pipeline (its
writers live in `common/openstreetmap` + `swag/scripts`, its readers in
`swag/data/vigor_dataset.py` and here), so the single owner is
`swag/data/landmark_schema.py`. This module re-exports it verbatim to keep
farfield's import surface (`farfield.catalog.schema`) stable — do not add
farfield-only behavior here; it belongs in `catalog.catalog`.
"""

from experimental.overhead_matching.swag.data.landmark_schema import (  # noqa: F401
    META_COLUMNS,
    TAGS_COLUMN,
    build_frame,
    is_dict_schema,
    read_frame,
    row_dicts,
    row_dicts_with_index,
    summarize,
    tag_dicts,
    tag_key_columns,
    widen,
)
