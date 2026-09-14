"""Soft exact-name evidence from available detector observations only."""
from collections import defaultdict
from audit_name_tables import NAME_KEYS, normalize_name, add_names
from audit_only_v2 import THRESHOLD


def add_prefix_names(event, observations, lookup):
    provenance=event['semantic_evidence']['provenance']
    votes=defaultdict(float)
    supporting=[]
    total=provenance['supported_frame_count_including_birth']
    for member in provenance['observations']:
        obs=observations[member['obs_id']]
        if obs['frame_idx']!=member['frame'] or member['frame']>event['release_keyframe']:
            raise ValueError('Name comes from a future or misjoined observation')
        if obs['confidence']!='high':
            continue
        matched=set()
        names=[]
        for key,value in obs.get('additional_tags',[]):
            if key in NAME_KEYS:
                name=normalize_name(value)
                ids=set(lookup.get(name,())) if name else set()
                if ids:
                    names.append(value)
                    matched.update(ids)
        if matched:
            # Shared existing .8 confidence constant, diluted by the fraction
            # of supporting frames that actually supply this exact name.
            weight=THRESHOLD*member['frame_vote_share']/total
            for lid in matched: votes[lid]+=weight
            supporting.append(dict(obs_id=member['obs_id'],frame=member['frame'],
                                   names=names,landmark_ids=sorted(matched),weight=weight))
    table=add_names(event['table'],dict(votes))
    if supporting:
        table['matcher_version']=event['table']['matcher_version']+'+available_detector_exact_names_v1'
    return table,dict(method='available_high_confidence_exact_names_v1',
        maximum_name_share=THRESHOLD,uses_final_audit=False,observations=supporting,
        supported_landmark_weights=dict(votes))
