"""Causal cumulative revisions from immutable eight-frame tracker snapshots."""
import argparse
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from prefix_release_evidence import build_evidence
from prefix_name_evidence import add_prefix_names
from audit_name_tables import NAME_KEYS, normalize_name
from reservoir_canonicalization import canonical_tags
from experimental.overhead_matching.swag.farfield import dataset
from experimental.overhead_matching.swag.farfield.catalog import catalog, schema

OUT=Path('/data/farfield_matching/runs/260913_accuracy_recovery')
ROOT=OUT.parent.parent
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parser=argparse.ArgumentParser()
parser.add_argument('--end',type=int,required=True)
parser.add_argument('--evidence',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
assert not args.out.exists()
source=args.evidence
base=json.loads(source.read_text())
for path,sha in base['input_sha256'].items():assert digest(path)==sha,path
ds=base['dataset']
manifest_path=next(Path(p) for p in base['input_sha256'] if '/object_tracks/' in p and p.endswith('/manifest.json'))
manifest=json.loads(manifest_path.read_text());config=manifest['config']['resolved']
from all13_inputs import resolve
frame_ref={'path':str(resolve(ds)['frames_path'])}
ingested=dataset.run_ingest(ROOT/'datasets'/ds,Path(frame_ref['path']),dataset.IngestParams(**config['ingest']))
observations={o.obs_id:dataclasses.asdict(o) for o in ingested.observations}
times={f.frame_idx:f.time_s for f in ingested.frames}
sig_path=next(Path(p) for p in base['input_sha256'] if p.endswith('/signatures.json'))
old_signatures=json.loads(sig_path.read_text())
ids={lid for s in old_signatures.values() for lid in s['landmark_ids']}
raw_path=next(Path(p) for p in base['input_sha256'] if p.endswith('/catalog.feather'))
raw=schema.read_frame(raw_path);signatures={}
for i,tags in enumerate(schema.tag_dicts(raw)):
    kind=raw['landmark_type'].iloc[i];lid=catalog._id_text(raw['id'].iloc[i])
    if not lid.startswith(f'{kind}:'):lid=f'{kind}:{lid}'
    if lid in ids:signatures[lid]=dict(canonical_tags=catalog.prune_far_field_tags(canonical_tags(tags)[0]),landmark_ids=[lid])
assert set(signatures)==ids
lookup={}
for s in old_signatures.values():
    for key,value in s['canonical_tags'].items():
        if key in NAME_KEYS:
            for name in value.split(';'):
                name=normalize_name(name)
                if name:lookup.setdefault(name,set()).update(s['landmark_ids'])
initial={e['tracklet_id'].split('#')[-1]:copy.deepcopy(e) for e in base['emissions'] if e['available_time_s']<=times[args.end]}
latest=copy.deepcopy(initial);events=list(initial.values());paths=[source,Path(__file__),OUT/'tools/all13_inputs.py']
for end in range(8,args.end+1,8):
    rp=OUT/f'{ds}.all13_prefix_v1.end{end}.report.json'
    report=json.loads(rp.read_text());paths.append(rp)
    for p,sha in report['output_sha256'].items():assert digest(p)==sha,p
    p=Path(next(iter(report['output_sha256'])));paths.append(p)
    snapshot=json.loads(p.read_text());clock=report['course_availability'][-1]['available_time_s']
    if clock>times[args.end]:continue
    for track in snapshot['tracking']['tracks']:
        physical=f"T{track['track_id']}"
        if physical not in latest:continue
        prev=latest[physical]
        if prev['available_time_s']>=clock:continue
        assert all(r['keyframe']<=end for r in track['records'])
        emission=dict(track=copy.deepcopy(track),release_keyframe=end,available_time_s=clock)
        tid=prev['tracklet_id'].split('#')[0]+f'#{physical}.R{end:04d}'
        new=dict(tracklet_id=tid,release_keyframe=end,available_time_s=clock,
            **build_evidence(emission,observations,times,tracklet_id=tid,
              pano_width=config['tracking']['reference_pano_width'],
              mount_bearing_camera_cw_deg=base['parameters']['mount_bearing_camera_cw_deg'],
              bearing_sigma_deg=base['parameters']['bearing_sigma_deg'],
              epoch_keyframes=base['parameters']['epoch_keyframes'],signatures=signatures))
        if new['last_supported_keyframe']<=prev['last_supported_keyframe']:continue
        new['table'],new['name_evidence']=add_prefix_names(new,observations,lookup)
        new['revision']=dict(physical_track=physical,previous_tracklet_id=prev['tracklet_id'],
            previous_last_supported_keyframe=prev['last_supported_keyframe'],
            policy='cumulative_evidence_at_regular_observed_snapshot')
        events.append(new);latest[physical]=new
base['emissions']=sorted(events,key=lambda e:(e['available_time_s'],e['tracklet_id']))
paths.extend(OUT/'tools'/p for p in ['prefix_release_evidence.py','prefix_name_evidence.py','audit_name_tables.py','reservoir_canonicalization.py'])
base['input_sha256'].update({str(p):digest(p) for p in paths})
base['revision_policy']=dict(snapshot_period_keyframes=8,only_new_supported_frames=True,
    cumulative_revisions=True,no_eof_flush=True,same_parameters_for_all_datasets=True)
base['qualification']+=' Cumulative track revisions use only current snapshot records; first releases remain unchanged. Each revision must replace previous track evidence by a conditional factor ratio, never multiply the cumulative likelihood again. Eight-frame snapshot cadence is fixed, not endpoint-dependent.'
args.out.write_text(json.dumps(base,indent=2))
print('REVISION_EVIDENCE',len(initial),'initial',len(events)-len(initial),'revisions',flush=True)
