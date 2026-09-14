"""Use exact high-confidence names already present in each closed-track audit.

Retain the category candidates as a fallback; no batched matcher response or
geographic truth enters this transformation. Numerals alone are not names.
"""
import json
import math
import re
import unicodedata
from pathlib import Path
from audit_only_v2 import OUT, THRESHOLD, FLOOR, CEILING, to_log_lr
from semantic_alternatives import load_audits

NAME_KEYS = {'name','official_name','alt_name','short_name','name:en'}


def normalize_name(value):
    text = ''.join(c for c in unicodedata.normalize('NFKD',value.casefold()) if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+',' ',text).strip().removeprefix('the ')


def name_evidence(audit, lookup):
    supported = {}
    for candidate in audit['primary_object'].get('name_candidates',[]):
        weight = candidate.get('weight',0)
        name = normalize_name(candidate['name'])
        if weight < THRESHOLD or not re.search('[a-z]',name):
            continue
        for lid in lookup.get(name,[]):
            supported[lid] = max(supported.get(lid,0),weight)
    return supported


def add_names(table, supported):
    if not supported:
        return table
    result = dict(table)
    share = min(max(supported.values()),1.0)
    probabilities = {e['landmark_id']:(1-share)/(1+math.exp(-e['log_lr'])) for e in table['entries']}
    total = sum(supported.values())
    for lid,weight in supported.items():
        probabilities[lid] = probabilities.get(lid,0)+share*weight/total
    result['entries'] = [
        {'kind':'CompatibilityEntry','landmark_id':lid,'log_lr':to_log_lr(p,clip=CEILING,clip_lo=FLOOR)}
        for lid,p in sorted(probabilities.items()) if to_log_lr(p,clip=CEILING,clip_lo=FLOOR)>FLOOR]
    result['matcher_version'] += '+own_audit_exact_names_v1'
    return result


def build(ds):
    base_path = OUT/'tables'/f'{ds}.dedup.audit_only_v2.json'
    report = json.loads(base_path.with_suffix('.report.json').read_text())
    paths = list(report['inputs_sha256'])
    audits = load_audits(Path(next(p for p in paths if '/semantic_audits/' in p)))
    signatures = json.loads(Path(next(p for p in paths if p.endswith('/signatures.json'))).read_text())
    lookup = {}
    for signature in signatures.values():
        for key,value in signature['canonical_tags'].items():
            if key in NAME_KEYS:
                for value in value.split(';'):
                    lookup.setdefault(normalize_name(value),set()).update(signature['landmark_ids'])
    supported = {key:name_evidence(audit,lookup) for key,audit in audits.items()}
    outputs = []
    changed = []
    for label,path in [('base',base_path),('adaptive',OUT/'tables'/f'{ds}.dedup.tower_family_v1.json')]:
        tables = json.loads(path.read_text())
        new = [add_names(t,supported[t['tracklet_id'].split('#')[-1]]) for t in tables]
        if label=='base':
            changed = [{'track':t['tracklet_id'].split('#')[-1],
                        'name_supported_landmarks':supported[t['tracklet_id'].split('#')[-1]]}
                       for t in tables if supported[t['tracklet_id'].split('#')[-1]]]
        target = OUT/'tables'/f'{ds}.dedup.audit_names_v1.{label}.json'
        encoded = json.dumps(new)
        if target.exists() and target.read_text()!=encoded:
            raise RuntimeError(f'Refusing changed experiment input: {target}')
        target.write_text(encoded)
        outputs.append(target)
    outputs[0].with_suffix('.report.json').write_text(json.dumps({
        'source_input_sha256':report['inputs_sha256'],'name_keys':sorted(NAME_KEYS),
        'threshold':THRESHOLD,'changed':changed,'uses_matcher_responses':False,
        'uses_truth':False,'exact_names_only':True},indent=2))
    print(ds,'name-supported tracks',len(changed),flush=True)
    return outputs
