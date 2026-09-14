"""Water matching after guarded canonicalization of both input representations."""
import collections
from audit_only_v2 import KIND_KEYS, THRESHOLD, FLOOR, CEILING, to_log_lr


def water_entries(audit, signatures):
    tags = [t for t in audit['primary_object'].get('tags', [])
            if t.get('weight',0)>=THRESHOLD and t['tag'].split('=',1)[0] in KIND_KEYS]
    wanted = collections.defaultdict(set)
    for tag in tags:
        key,value = tag['tag'].split('=',1)
        wanted[key].add(value)
    if wanted.get('natural') != {'water'}:
        return None
    ids = set()
    for signature in signatures.values():
        actual = signature['canonical_tags']
        if all(actual.get(k) in values for k,values in wanted.items()):
            ids.update(signature['landmark_ids'])
    confidence = min((t['weight'] for t in tags),default=0)
    score = to_log_lr(confidence/max(len(ids),1),clip=CEILING,clip_lo=FLOOR)
    return [dict(kind='CompatibilityEntry',landmark_id=lid,log_lr=score)
            for lid in sorted(ids)] if score>FLOOR else []
