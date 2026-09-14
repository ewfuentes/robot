"""Choose point or footprint factors from current, unrevised filtering state.

Extent observations must be ready at the current release. This contract does
not establish the provenance or processing delay of the upstream observations.
"""
import json
from pathlib import Path
import common.torch.load_torch_deps  # noqa: F401
import torch


def load_observations(path: Path, track_ids):
    payload = json.loads(path.read_text())
    if payload.get('schema') != 'farfield_observed_extent/v1':
        raise ValueError('Unsupported observed extent schema')
    rows = payload['observations']
    by_id = {}
    for row in rows:
        tid = row['tracklet_id']
        if tid in by_id:
            raise ValueError(f'Duplicate extent observation: {tid}')
        if (type(row['large_extended']) is not bool
                or type(row['available_keyframe']) is not int
                or row['available_keyframe'] < 0):
            raise ValueError(f'Invalid extent observation: {tid}')
        by_id[tid] = row
    if set(by_id) != set(track_ids):
        raise ValueError('Extent observations must cover exact track IDs')
    return by_id


def large_at_release(observations, track_id, keyframe):
    row = observations[track_id]
    if row['available_keyframe'] > keyframe:
        raise ValueError('Extent observation is not ready at this release')
    return row['large_extended']


def choose_factor(prior, point_factor, footprint_factor, cell_east, cell_north,
                  normalize, *, large_extended, mode_mass, radius_m, lock_mass):
    """Evaluate the footprint lazily when a locked large-object update jumps.

Point and footprint hypotheses that agree on a distant correction retain the
point update. No reference trajectory, truth pose, or later state is consulted.
"""
    decision = dict(extent_enabled=False, point_map_jump_m=None, model_comparison=None)
    if not large_extended or mode_mass < lock_mass:
        return point_factor, decision
    prior_index = prior.sum(dim=0).reshape(-1).argmax()
    proposed = normalize(prior * point_factor)
    point_index = proposed.sum(dim=0).reshape(-1).argmax()

    def distance(a, b):
        return float(torch.hypot(cell_east[a]-cell_east[b],
                                 cell_north[a]-cell_north[b]).item())

    point_jump = distance(prior_index, point_index)
    decision['point_map_jump_m'] = point_jump
    if point_jump <= radius_m:
        return point_factor, decision
    footprint = footprint_factor()
    footprint_index = normalize(prior * footprint).sum(dim=0).reshape(-1).argmax()
    disagreement = distance(point_index, footprint_index)
    footprint_jump = distance(prior_index, footprint_index)
    decision['model_comparison'] = dict(
        map_disagreement_m=disagreement, footprint_map_jump_m=footprint_jump,
        point_predictive_evidence=float((prior.double()*point_factor.double()).sum().item()),
        footprint_predictive_evidence=float((prior.double()*footprint.double()).sum().item()))
    decision['extent_enabled'] = disagreement > radius_m and footprint_jump < point_jump
    return (footprint if decision['extent_enabled'] else point_factor), decision
