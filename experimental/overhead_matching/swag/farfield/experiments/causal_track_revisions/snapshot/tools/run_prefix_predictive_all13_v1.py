"""Run a tool-only live-evidence overlay through the actual fused grid filter.

No original input artifact is edited or represented as a newly valid export.
An explicit overlay identity isolates exact factor caches. Baseline mode is
unmodified and must reproduce the recorded prefix before interpretation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from all13_filter_overlay_v1 import prepare_releases, overlay_data
from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest, odometry_profiles, release_schedule)

OUT=Path('/data/farfield_matching/runs/260913_accuracy_recovery')
import importlib.util
spec=importlib.util.spec_from_file_location('isolated_grid_predictive',OUT/'experimental_predictive_tempering/grid_filter_predictive.py')
g=importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
parser=argparse.ArgumentParser()
parser.add_argument('--evidence',type=Path,required=True)
parser.add_argument('--reference-stem',required=True)
parser.add_argument('--end',type=int,required=True)
parser.add_argument('--seed',type=int,required=True)
parser.add_argument('--mode',choices=['baseline','prefix'],required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--mixture',choices=['sum','max'],required=True)
parser.add_argument('--predictive-tempering',type=int,choices=[0,1],required=True)
args=parser.parse_args()
assert not args.out.exists(), 'Choose a new output name; never overwrite a result'
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
reference_path=OUT/'results'/f'{args.reference_stem}.json'
record_path=OUT/'logs'/f'{args.reference_stem}.run.json'
record=json.loads(record_path.read_text())
assert digest(reference_path)==record['output_sha256']
reference=json.loads(reference_path.read_text())
cfg=reference['config']
assert cfg['smoother']=='none' and cfg['smooth_lag']==0 and cfg['track_joint']
assert cfg['joint_backend']=='fused' and cfg['joint_cap'] is None
assert cfg['init_truth_sigma_m']==cfg['init_heading_sigma_deg']==cfg['heading_meas_sigma_deg']==0
evidence=json.loads(args.evidence.read_text())
for path,sha in evidence['input_sha256'].items():
    assert digest(path)==sha,path
base_path=Path(cfg['input_dir'])
base=export_ingest.load(base_path)
assert evidence['dataset']==base.artifact_ref.dataset
dataset_path=Path('/data/farfield_matching/datasets')/evidence['dataset']
assert evidence['panorama_index_names']==sorted(p.name for p in (dataset_path/'panorama').glob('*.jpg'))
timestamps=odometry_profiles.load_timestamps(base_path,base.n_keyframes)
releases,readiness=prepare_releases(evidence,timestamps,args.end)
files=[args.evidence,reference_path,record_path,Path(__file__),
       OUT/'tools/all13_filter_overlay_v1.py',OUT/'tools/python.sh',
       OUT/'tools/sitecustomize.py',OUT/'tools/factor_cache.py']
files.append(OUT/'tools/predictive_tempering.py')
files.extend(p for p in base_path.iterdir() if p.is_file())
files.extend(Path(g.__file__).parent.glob('*.py'))
source={str(p):digest(p) for p in files}
identity=dict(evidence_sha256=digest(args.evidence),adapter_sha256=digest(Path(__file__)),
              overlay_helper_sha256=digest(OUT/'tools/all13_filter_overlay_v1.py'))
remove={'--odometry_seed','--out','--window_end','--trace_dir','--trace_keyframes','--resume_prior','--mixture'}
if args.mode=='prefix':
    remove.update({'--tables_override','--adaptive_tables','--adaptive_extent_observations','--release_schedule'})
command=record['command'][2:]
assert len(command)%2==0 and all(command[i].startswith('--') for i in range(0,len(command),2))
command=[v for i in range(0,len(command),2) if command[i] not in remove for v in command[i:i+2]]
command.extend(['--odometry_seed',str(args.seed)])
command.extend(['--window_end',str(args.end),'--out',str(args.out),'--mixture',args.mixture])
command.extend(['--predictive_tempering',str(args.predictive_tempering)])
old_load,old_schedule=export_ingest.load,release_schedule.load_sidecar
if args.mode=='prefix':
    adapted=overlay_data(base,releases,identity)
    def load_overlay(path):
        assert path==base_path
        return adapted
    def load_schedule(path,data):
        assert path==args.evidence and data is adapted
        return releases
    export_ingest.load,release_schedule.load_sidecar=load_overlay,load_schedule
    command.extend(['--release_schedule',str(args.evidence)])
    # The ordinary reader checks a schema field solely for output labelling.
    # The evidence document declares its own diagnostic schema explicitly.
    assert evidence['schema']=='farfield_live_prefix_evidence/v1'
torch.set_num_threads(4)
started=time.monotonic()
sys.argv=[g.__file__,*command]
run_record=args.out.with_suffix('.run.json')
provenance=dict(mode=args.mode,source_and_input_sha256=source,filter_argv=sys.argv,
    prefix_readiness=readiness,model_api_calls=0,returncode=None,
    qualification=evidence['qualification'],
    additional_qualification='Experimental predictive tempering may inhibit recovery from a wrong prior; no truth input or adoption claim. Tool-only evidence overlay; original map/prior and GPS-derived benchmark odometry retained. No detector/tracker compute delay budget. One-shot prefixes discard subsequent evidence. No EOF flush or retrospective pose changes.')
run_record.write_text(json.dumps(provenance,indent=2))
try:
    g.main()
finally:
    export_ingest.load,release_schedule.load_sidecar=old_load,old_schedule
result=json.loads(args.out.read_text())
for path,sha in source.items():
    assert digest(path)==sha,path
if args.mode=='baseline' and args.mixture=='sum':
    errors={r:max(abs(a-b) for a,b in zip(curve,reference['mass_by_keyframe'][r][:args.end+1]))
            for r,curve in result['mass_by_keyframe'].items()}
    assert all(len(curve)==args.end+1 for curve in result['mass_by_keyframe'].values())
    assert max(errors.values())==0,errors
    assert result['online_map_state_by_keyframe']['states']==reference['online_map_state_by_keyframe']['states'][:args.end+1]
    provenance['baseline_exact_parity']=errors
elif args.mode=='prefix':
    result['availability']['policy']='immutable_live_prefix_readiness_quantized_forward'
    result['live_prefix_overlay']=dict(identity=identity,readiness=readiness,
        qualification=provenance['additional_qualification'])
    args.out.write_text(json.dumps(result))
provenance['mixture_experiment']=args.mixture
provenance.update(returncode=0,elapsed_seconds=time.monotonic()-started,output_sha256=digest(args.out))
run_record.write_text(json.dumps(provenance,indent=2))
print('LIVE_PREFIX_FILTER_COMPLETE',args.mode,result['summary'],flush=True)
