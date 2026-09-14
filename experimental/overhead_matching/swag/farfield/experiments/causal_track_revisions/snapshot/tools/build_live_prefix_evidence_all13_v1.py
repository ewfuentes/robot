"""Convert completed live-prefix smoke emissions without final audit reuse."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from prefix_release_evidence import build_evidence
from reservoir_canonicalization import canonical_tags
from experimental.overhead_matching.swag.farfield import dataset
from experimental.overhead_matching.swag.farfield.catalog import catalog,schema

ROOT=Path('/data/farfield_matching')
OUT=ROOT/'runs/260913_accuracy_recovery'
from all13_inputs import resolve
parser=argparse.ArgumentParser()
parser.add_argument('--report',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
assert not args.out.exists(), 'Refuse to overwrite evidence'
smoke_report_path=args.report
smoke_report=json.loads(smoke_report_path.read_text())
DS=smoke_report['dataset']
inputs=resolve(DS)
for path,digest in smoke_report['output_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
for path,digest in smoke_report['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
smoke_path=Path(next(iter(smoke_report['output_sha256'])))
smoke=json.loads(smoke_path.read_text())
tracks_manifest_path=inputs['manifest_path']
manifest=json.loads(tracks_manifest_path.read_text())
config=manifest['config']['resolved']
frame_ref={'path':str(inputs['frames_path'])}
ingested=dataset.run_ingest(ROOT/'datasets'/DS,Path(frame_ref['path']),
                            dataset.IngestParams(**config['ingest']))
observations={o.obs_id:dataclasses.asdict(o) for o in ingested.observations}
times={f.frame_idx:f.time_s for f in ingested.frames}
base_input=inputs['base_input']
meta_path=base_input/'export_meta.json'
meta=json.loads(meta_path.read_text())
calibration_path=base_input/'nominal_forward.json'
calibration=json.loads(calibration_path.read_text())
assert hashlib.sha256(calibration_path.read_bytes()).hexdigest()==meta['nominal_forward']['content_sha256']
map_report_path=OUT/'tables'/f'{DS}.dedup.audit_only_v2.report.json'
map_report=json.loads(map_report_path.read_text())
signatures_path=Path(next(p for p in map_report['inputs_sha256'] if p.endswith('/signatures.json')))
assert hashlib.sha256(signatures_path.read_bytes()).hexdigest()==map_report['inputs_sha256'][str(signatures_path)]
ids={lid for s in json.loads(signatures_path.read_text()).values() for lid in s['landmark_ids']}
rawpath=Path(map_report['catalog_upstream']['path'])/'catalog.feather'
rawframe=schema.read_frame(rawpath)
signatures={}
for i,raw in enumerate(schema.tag_dicts(rawframe)):
    source=rawframe['landmark_type'].iloc[i]
    identifier=catalog._id_text(rawframe['id'].iloc[i])
    lid=identifier if identifier.startswith(f'{source}:') else f'{source}:{identifier}'
    if lid in ids:
        signatures[lid]=dict(canonical_tags=catalog.prune_far_field_tags(canonical_tags(raw)[0]),
                             landmark_ids=[lid])
assert set(signatures)==ids
outputs=[]
for event in smoke['emissions']:
    emission=event['release']
    digest=hashlib.sha256(json.dumps(emission,sort_keys=True).encode()).hexdigest()
    assert digest==event['sha256']
    tid=f"object_tracks:{DS}:live_prefix_v1@sha256:{digest}#T{emission['track']['track_id']}"
    evidence=build_evidence(emission,observations,times,tracklet_id=tid,
        pano_width=config['tracking']['reference_pano_width'],
        mount_bearing_camera_cw_deg=calibration['bearing_camera_cw_deg'],
        bearing_sigma_deg=1.,epoch_keyframes=meta['reducer']['epoch_keyframes'],
        signatures=signatures)
    outputs.append(dict(tracklet_id=tid,release_keyframe=emission['release_keyframe'],
                        available_time_s=emission['available_time_s'],**evidence))
paths=[OUT/'tools/all13_inputs.py',smoke_report_path,smoke_path,tracks_manifest_path,meta_path,calibration_path,
       signatures_path,rawpath,Path(frame_ref['path'])/'predictions.jsonl',Path(__file__),
       Path(frame_ref['path'])/'manifest.json',
       ROOT/'datasets'/DS/'frames_gps.csv',ROOT/'datasets'/DS/'pipeline_metadata.json',
       OUT/'tools/prefix_release_evidence.py',OUT/'tools/reservoir_canonicalization.py',
       OUT/'tools/support_likelihood_mixture.py',OUT/'tools/audit_only_v2.py',
       Path(dataset.__file__),Path(catalog.__file__),Path(schema.__file__)]
payload=dict(schema='farfield_live_prefix_evidence/v1',dataset=DS,emissions=outputs,
    panorama_index_names=sorted(p.name for p in (ROOT/'datasets'/DS/'panorama').glob('*.jpg')),
    input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    parameters=dict(bearing_sigma_deg=1.,epoch_keyframes=meta['reducer']['epoch_keyframes'],
                    mount_bearing_camera_cw_deg=calibration['bearing_camera_cw_deg']),
    uses_final_track_audit=False,model_api_calls=0,
    qualification='Measurements/tables built from immutable live emissions and per-frame detector inputs. No full-episode localization result. Uses supplied nominal-forward calibration whose independent provenance is not certified; independent calibration, detector latency, historical leveling and stabilization remain separately qualified.')
target=args.out
target.write_text(json.dumps(payload,indent=2))
print('LIVE_PREFIX_EVIDENCE',len(outputs),'emissions;',sum(len(e['measurements']) for e in outputs),
      'measurements;',sum(bool(e['table']['entries']) for e in outputs),'nonempty tables')
