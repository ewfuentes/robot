"""Local semantic ablations for hard taxonomy and uncertain peak bearings.

These are experimental, global tag rules, with no truth/frame-selected tracks.
Unchanged tables remain byte-equivalent for exact factor cache reuse.
"""
import json
import sys
from audit_only_v2 import ROOT, OUT, THRESHOLD, FLOOR, CEILING, to_log_lr


def load_audits(path):
    audits = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        for c in r['response'].get('candidates', []):
            for p in c.get('content', {}).get('parts', []):
                if p.get('thought') or 'text' not in p:
                    continue
                try:
                    a = json.loads(p['text'])
                except json.JSONDecodeError:
                    continue
                if isinstance(a, dict) and 'primary_object' in a:
                    audits[r['key']] = a
    return audits


def build(ds, variant):
    from pathlib import Path
    base = OUT / 'tables' / f'{ds}.dedup.audit_only_v2.json'
    provenance = json.loads(base.with_suffix('.report.json').read_text())
    paths = list(provenance['inputs_sha256'])
    audits = load_audits(Path(next(p for p in paths if '/semantic_audits/' in p)))
    signatures = json.loads(Path(next(p for p in paths if p.endswith('/signatures.json'))).read_text())
    tables = json.loads(base.read_text())
    changed = []
    for table in tables:
        tid = table['tracklet_id'].split('#')[-1]
        tags = {t['tag']: t.get('weight', 0) for t in audits[tid]['primary_object'].get('tags', [])}
        family_conf = max(tags.get('man_made=mast', 0), tags.get('man_made=tower', 0))
        before = table['entries']
        expand_family = ('tower_family' in variant and family_conf >= THRESHOLD)
        if variant == 'mast_family':
            # Narrower hypothesis: preserve the detailed tower match when
            # tower is already the preferred audit label. Expand only when
            # the audit's strongest family label is mast.
            expand_family = (tags.get('man_made=mast', 0) >= THRESHOLD
                             and tags.get('man_made=mast', 0) >= tags.get('man_made=tower', 0))
        if expand_family:
            # Tower and mast labels are visually ambiguous. First test their
            # whole catalog family, without requiring complete subtype tags.
            ids = {lid for s in signatures.values()
                   if s['canonical_tags'].get('man_made') in {'tower', 'mast'}
                   for lid in s['landmark_ids']}
            score = to_log_lr(family_conf / max(1, len(ids)), clip=CEILING, clip_lo=FLOOR)
            table['entries'] = [{'kind': 'CompatibilityEntry', 'landmark_id': lid, 'log_lr': score}
                                for lid in sorted(ids)] if score > FLOOR else []
        if 'peak_neutral' in variant and tags.get('natural=peak', 0) >= THRESHOLD:
            # Diagnostic: a broad silhouette's box centre may not measure the
            # mapped summit. Uniform identity is distinct from omitting a factor.
            table['entries'] = []
        if table['entries'] != before:
            table['matcher_version'] = 'local_semantic_ablation_' + variant + '_v1'
            changed.append({'track': tid, 'before_candidates': len(before),
                            'after_candidates': len(table['entries'])})
    target = OUT / 'tables' / f'{ds}.dedup.{variant}_v1.json'
    encoded = json.dumps(tables)
    if target.exists() and target.read_text() != encoded:
        raise RuntimeError(f'Refusing to change existing experiment input: {target}')
    target.write_text(encoded)
    target.with_suffix('.report.json').write_text(json.dumps({
        'base': str(base), 'source_inputs_sha256': provenance['inputs_sha256'],
        'variant': variant, 'changed': changed,
        'uses_matcher_responses': False, 'uses_truth': False,
        'diagnostic_not_adopted': True}, indent=2))
    print(ds, variant, len(changed), 'changed tables', flush=True)
    return target


if __name__ == '__main__':
    for ds in sys.argv[1:]:
        for variant in ['tower_family', 'peak_neutral', 'tower_family_peak_neutral']:
            build(ds, variant)
