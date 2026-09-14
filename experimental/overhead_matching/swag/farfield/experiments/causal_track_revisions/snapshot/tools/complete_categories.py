"""Offline candidate coverage experiment using only the same chain's cached evidence.

If the audited query strongly names a semantic tag AND a category-matched
signature carries that tag, include all catalog signatures carrying it.
This avoids losing the tag because another key won primary-category precedence.
Existing entries/instance weights are preserved. New members share the matched
category confidence; no truth, cross-chain identities, or new model calls.
"""
import collections,json,math,re,sys
from pathlib import Path
ROOT=Path('/data/farfield_matching');OUT=ROOT/'runs/260913_accuracy_recovery';SOURCE=ROOT/'runs/260913_smoothing_fix'
sys.path.insert(0,str(ROOT/'runs/260912_harel_experiments'))
from reaggregate import KIND_KEYS,to_log_lr
PLAN=json.loads((SOURCE/'plan.json').read_text())
def build(ds,method='dedup',conjunction=False):
 chain=next(c for c in PLAN['chains'] if c['dataset']==ds and c['method']==method);base=ROOT/'artifacts/landmark_matches'/ds/chain['matches_version']
 matches=json.loads((base/'matches.json').read_text());sig=json.loads((base/'signatures.json').read_text());queries=json.loads((base/'matching_snapshot.json').read_text())['queries'];tables=json.loads((SOURCE/'tables'/f'{ds}.{method}.divided.json').read_text())
 members=collections.defaultdict(set)
 for v in sig.values():
  for key,value in v['canonical_tags'].items():
   if key in KIND_KEYS:members[f'{key}={value}'].update(v['landmark_ids'])
 changed=[]
 for t in tables:
  tid=t['tracklet_id'];query=queries[tid];first=next((l for l in query.splitlines() if l.startswith('tags: ')),'')
  tags={tag.strip():float(w) for tag,w in re.findall(r'([^;]+?) \(([0-9.]+)\)',first.removeprefix('tags: ')) if float(w)>=.8 and tag.strip().split('=',1)[0] in KIND_KEYS}
  supported={};groups={}
  for m in matches[tid]['matches']:
   if m['match_type']!='category' or m['aggregate_confidence']<.05:continue
   st={f'{key}={value}' for key,value in sig[m['signature_id']]['canonical_tags'].items()}
   shared=tags.keys()&st
   for tag in shared:supported[tag]=max(supported.get(tag,0),m['aggregate_confidence'])
   if shared:
    group=tuple(sorted(shared));groups[group]=max(groups.get(group,0),m['aggregate_confidence'])
  existing={e['landmark_id'] for e in t['entries']};added={}
  expansions=groups.items() if conjunction else [((tag,),c) for tag,c in supported.items()]
  for group,c in expansions:
   ids=set.intersection(*(members[tag] for tag in group));llr=to_log_lr(c/max(len(ids),1),clip_lo=t['clip_lo'])
   if llr<=t['default_log_lr']+1e-12:continue
   for lid in ids-existing:added[lid]=max(added.get(lid,float('-inf')),llr)
  variant='coverage_conjunction' if conjunction else 'coverage'
  if added:
   t['entries'] += [{'kind':'CompatibilityEntry','landmark_id':lid,'log_lr':llr} for lid,llr in sorted(added.items())]
   t['matcher_version'] += ('+audit_tag_conjunction_v1' if conjunction else '+audit_tag_coverage_v1');changed.append({'track':tid.split('#')[-1],'added':len(added),'supported_tags':supported})
 out=OUT/'tables'/f'{ds}.{method}.{variant}.json';out.write_text(json.dumps(tables));(out.with_suffix('.report.json')).write_text(json.dumps(changed,indent=2));print(ds,len(changed),'tracks expanded;',sum(r['added'] for r in changed),'new entries',flush=True)
 return out
if __name__=='__main__':
 for ds in sys.argv[1:]:build(ds)
