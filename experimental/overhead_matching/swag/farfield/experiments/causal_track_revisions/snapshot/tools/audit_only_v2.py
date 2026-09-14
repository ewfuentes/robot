"""Track-local tag matching with fixed score constants; no matcher score inputs.

The match manifest locates the audit and static catalog signature artifacts.
Measurement IDs select tracks to export; each table depends only on that
track's audit, the static catalog, and the declared constants below.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path
from complete_categories import ROOT, OUT, PLAN, KIND_KEYS, to_log_lr

FLOOR = -12.0
CEILING = 4.0
THRESHOLD = 0.8
VERSION = 'local_own_audit_tag_conjunction_fixed_floor_v2'


def table_for_track(track_id, audit, signatures):
    tags = [x for x in audit['primary_object'].get('tags', [])
            if x.get('weight', 0) >= THRESHOLD
            and x['tag'].split('=', 1)[0] in KIND_KEYS]
    by_key = collections.defaultdict(set)
    for tag in tags:
        key, value = tag['tag'].split('=', 1)
        by_key[key].add(value)
    ids = set()
    if by_key:
        for signature in signatures.values():
            if all(signature['canonical_tags'].get(k) in values
                   for k, values in by_key.items()):
                ids.update(signature['landmark_ids'])
    confidence = min((x['weight'] for x in tags), default=0)
    score = to_log_lr(confidence / max(len(ids), 1), clip=CEILING, clip_lo=FLOOR)
    entries = [{'kind': 'CompatibilityEntry', 'landmark_id': lid, 'log_lr': score}
               for lid in sorted(ids)] if score > FLOOR else []
    return {'kind': 'CompatibilityTable', 'tracklet_id': track_id,
            'matcher_version': VERSION, 'entries': entries,
            'default_log_lr': FLOOR, 'clip_lo': FLOOR, 'clip_hi': CEILING,
            'status': 'fast'}


def build(ds):
    chain = next(c for c in PLAN['chains'] if c['dataset'] == ds and c['method'] == 'dedup')
    matchdir = ROOT / 'artifacts/landmark_matches' / ds / chain['matches_version']
    manifest_path = matchdir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    upstream = next(u for u in manifest['upstreams'] if u['kind'] == 'semantic_audits')
    audit_path = ROOT / 'artifacts/semantic_audits' / ds / upstream['version'] / 'results.jsonl'
    signatures_path = matchdir / 'signatures.json'
    measurement_path = ROOT / 'artifacts/localization_inputs' / ds / chain['inputs_version'] / 'tier1_measurements.jsonl'
    audits = {}
    for line in audit_path.read_text().splitlines():
        response = json.loads(line)
        for candidate in response['response'].get('candidates', []):
            for part in candidate.get('content', {}).get('parts', []):
                if part.get('thought') or 'text' not in part:
                    continue
                try:
                    audit = json.loads(part['text'])
                except json.JSONDecodeError:
                    continue
                if isinstance(audit, dict) and 'primary_object' in audit:
                    audits[response['key']] = audit
    signatures = json.loads(signatures_path.read_text())
    track_ids = sorted({json.loads(line)['tracklet_id'] for line in measurement_path.read_text().splitlines()})
    tables = [table_for_track(tid, audits[tid.split('#')[-1]], signatures) for tid in track_ids]
    target = OUT / 'tables' / f'{ds}.dedup.audit_only_v2.json'
    encoded = json.dumps(tables)
    if target.exists() and target.read_text() != encoded:
        raise RuntimeError(f'Refusing to overwrite a changed experiment input: {target}')
    target.write_text(encoded)
    report = {'uses_matcher_responses_or_table_templates': False,
              'threshold': THRESHOLD, 'default_log_lr': FLOOR, 'clip_lo': FLOOR,
              'clip_hi': CEILING, 'track_count': len(tables),
              'inputs_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [manifest_path, audit_path, signatures_path, measurement_path]},
              'catalog_upstream': next(u for u in manifest['upstreams'] if u['kind'] == 'catalogs')}
    target.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    print(ds, 'v2 tables', len(tables), 'nonempty', sum(bool(t['entries']) for t in tables), flush=True)
    return target


if __name__ == '__main__':
    for dataset in sys.argv[1:]:
        build(dataset)
