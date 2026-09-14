"""Documented reservoir compatibility, applied before lossy map-tag pruning.

https://wiki.openstreetmap.org/wiki/Tag:landuse%3Dreservoir
Original tags are retained separately by the caller. Only the compatibility
representation drops the obsolete key after a valid, conflict-free migration.
"""
import copy

RULE = 'osm_open_reservoir_v2'


def canonical_tags(raw):
    out = dict(raw)
    if out.get('landuse') != 'reservoir':
        return out, []
    if (out.get('covered') == 'yes'
            or out.get('man_made') in {'reservoir_covered','storage_tank','water_tower'}
            or out.get('location') in {'underground','underwater'}
            or out.get('natural') not in {None,'water'}
            or out.get('water') not in {None,'reservoir'}):
        return out, []
    out.pop('landuse')
    out['natural'] = 'water'
    out['water'] = 'reservoir'
    return out, [RULE]


def canonical_observation(audit, threshold):
    """Derive aliases from strong unambiguous tags, without extra confidence."""
    result = copy.deepcopy(audit)
    tags = result['primary_object'].get('tags', [])
    values = {}
    for entry in tags:
        if entry.get('weight',0) >= threshold:
            key,value = entry['tag'].split('=',1)
            values.setdefault(key,set()).add(value)
    if any(len(values.get(k,set()))>1 for k in ['landuse','natural','water','covered','man_made','location']):
        return result, []
    raw = {k:next(iter(v)) for k,v in values.items() if len(v)==1}
    normalized, rules = canonical_tags(raw)
    if not rules:
        return result, []
    confidence = max(e['weight'] for e in tags if e['tag']=='landuse=reservoir')
    tags = [e for e in tags if e['tag']!='landuse=reservoir']
    for key in ['natural','water']:
        value = normalized[key]
        if not any(e['tag']==f'{key}={value}' and e.get('weight',0)>=threshold for e in tags):
            tags.append(dict(tag=f'{key}={value}',weight=confidence))
    result['primary_object']['tags'] = tags
    return result, rules
