import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from prefix_name_evidence import add_prefix_names
from audit_name_tables import NAME_KEYS,normalize_name
from experimental.overhead_matching.swag.farfield import dataset

parser=argparse.ArgumentParser()
parser.add_argument('--evidence',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
assert not args.out.exists()
evidence=json.loads(args.evidence.read_text())
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path,sha in evidence['input_sha256'].items():assert digest(path)==sha,path
root=Path('/data/farfield_matching')
out=root/'runs/260913_accuracy_recovery'
ds=evidence['dataset']
manifest_path=next(Path(p) for p in evidence['input_sha256'] if '/object_tracks/' in p and p.endswith('/manifest.json'))
manifest=json.loads(manifest_path.read_text())
from all13_inputs import resolve
ref={'path':str(resolve(ds)['frames_path'])}
ingested=dataset.run_ingest(root/'datasets'/ds,Path(ref['path']),
                          dataset.IngestParams(**manifest['config']['resolved']['ingest']))
observations={o.obs_id:dataclasses.asdict(o) for o in ingested.observations}
path=next(Path(p) for p in evidence['input_sha256'] if p.endswith('/signatures.json'))
signatures=json.loads(path.read_text())
lookup={}
for signature in signatures.values():
    for key,value in signature['canonical_tags'].items():
        if key in NAME_KEYS:
            for name in value.split(';'):
                normalized=normalize_name(name)
                if normalized:lookup.setdefault(normalized,set()).update(signature['landmark_ids'])
changed=[]
for event in evidence['emissions']:
    table,provenance=add_prefix_names(event,observations,lookup)
    if provenance['observations']:
        changed.append(dict(track=event['tracklet_id'].split('#')[-1],**provenance))
    event['table']=table
    event['name_evidence']=provenance
paths=[out/'tools/all13_inputs.py',args.evidence,Path(__file__),out/'tools/prefix_name_evidence.py',
       out/'tools/audit_name_tables.py',out/'tools/audit_only_v2.py',Path(dataset.__file__)]
evidence['input_sha256'].update({str(p):digest(p) for p in paths})
evidence['name_policy']=dict(changed_tracks=changed,parameters_shared_across_datasets=True,
                            model_api_calls=0,hard_name_pruning=False)
evidence['qualification']+=' Soft exact names from high-confidence available per-frame detections; category fallback retained. Experimental, no claim that detector names are always correct.'
args.out.write_text(json.dumps(evidence,indent=2))
print('PREFIX_NAMES',len(changed),'changed tracks',json.dumps(changed),flush=True)
