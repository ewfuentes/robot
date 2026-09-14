"""Average detector hypotheses in likelihood space instead of thresholding votes."""
import math
from audit_only_v2 import table_for_track


def mix_tables(tables, weights):
    if not tables or len(tables) != len(weights):
        raise ValueError('A nonempty weighted table list is required')
    if any(w < 0 for w in weights) or not math.isclose(sum(weights), 1.):
        raise ValueError('Weights must be nonnegative and sum to one')
    first = tables[0]
    for table in tables:
        for key in ['tracklet_id', 'default_log_lr', 'clip_lo', 'clip_hi']:
            if table[key] != first[key]:
                raise ValueError(f'Inconsistent table {key}')
    maps = [{e['landmark_id']:e['log_lr'] for e in t['entries']} for t in tables]
    default = first['default_log_lr']
    entries = []
    for lid in sorted(set().union(*(set(m) for m in maps))):
        lr = math.fsum(w * math.exp(m.get(lid, default)) for m,w in zip(maps,weights))
        score = math.log(lr)
        if score > default:
            entries.append(dict(kind='CompatibilityEntry', landmark_id=lid, log_lr=score))
    return {**first, 'entries':entries, 'matcher_version':'per_supported_frame_likelihood_mixture_v1'}


def mixture_for_track(tid, evidence, observations, signatures):
    provenance = evidence['provenance']
    if provenance.get('excluded_alive_track'):
        return table_for_track(tid, evidence, signatures)
    total = provenance['supported_frame_count_including_birth']
    tables, weights = [], []
    for member in provenance['observations']:
        obs = observations[member['obs_id']]
        tags = {(obs['primary_tag_key'], obs['primary_tag_value'])}
        tags.update(tuple(t) for t in obs.get('additional_tags', []))
        audit = {'primary_object':{'tags':[{'tag':f'{k}={v}','weight':1.}
                 for k,v in sorted(tags) if k and v]}}
        tables.append(table_for_track(tid, audit, signatures))
        weights.append(member['frame_vote_share']/total)
    return mix_tables(tables, weights)
