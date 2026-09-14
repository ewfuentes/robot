"""Experimental track-local identity selection from the current spatial belief.

Alternative tables must use the same track-local inputs/availability as the
base tables. This module does not establish upstream provenance or latency.
There is no lookahead, truth input, or persistent reference-trajectory state.
"""
import math
from pathlib import Path
import msgspec
import common.torch.load_torch_deps  # noqa: F401
import torch
from experimental.overhead_matching.swag.farfield.localization import structs


def spatial_mode_mass(belief, cell_east, cell_north, radius_m):
    """Probability in a fixed-radius ball about the current spatial MAP cell.

Marginalize heading first: uncertain orientation is not necessarily uncertain
position. Separated position modes remain uncertain even if each is sharp.
"""
    spatial = belief.sum(dim=0).reshape(-1)
    mode = torch.argmax(spatial)
    distance2 = ((cell_east-cell_east[mode]).square()
                 + (cell_north-cell_north[mode]).square())
    return float((spatial[distance2 <= radius_m*radius_m].sum()
                  / spatial.sum()).item())


def load_alternatives(path: Path, base_tables):
    """Require exact track identities; retain only score-changing alternatives."""
    tables = msgspec.json.decode(path.read_bytes(), type=list[structs.CompatibilityTable])
    by_id = {t.tracklet_id: t for t in tables}
    if len(by_id) != len(tables) or set(by_id) != set(base_tables):
        raise ValueError('adaptive tables must cover the exact base track IDs once')
    changed = {}
    for tid, table in by_id.items():
        numeric = [table.default_log_lr, table.clip_lo, table.clip_hi]
        numeric.extend(e.log_lr for e in table.entries)
        if not all(math.isfinite(v) for v in numeric) or table.clip_lo > table.clip_hi:
            raise ValueError(f'invalid adaptive scores: {tid}')
        if len({e.landmark_id for e in table.entries}) != len(table.entries):
            raise ValueError(f'duplicate adaptive landmark: {tid}')
        base = base_tables[tid]
        if (table.entries, table.default_log_lr, table.clip_lo, table.clip_hi) != (
                base.entries, base.default_log_lr, base.clip_lo, base.clip_hi):
            changed[tid] = table
    return changed
