"""Capture immutable early emissions from a live local causal-course tracker.

Cached per-frame detections and local SAM2 weights only; no model API calls.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import time
import torch
import warnings
from prefix_release_policy_v2 import ValidBirthPrefixRelease
from tracker_checkpoint import write_checkpoint,read_checkpoint,restore_checkpoint
from resume_tracking_range import run_range
from tracking_components import install_cpu_components
install_cpu_components()
warnings.filterwarnings("error", message="(?s).*Skipping the post-processing step.*")
from PIL import Image
from causal_geodetic_course import CausalGeodeticCourse
from experimental.overhead_matching.swag.farfield import dataset,geometry as geo
from experimental.overhead_matching.swag.farfield.tracking import range_runner as rr,track_builder as tb,sam_backend,video_frames

ROOT=Path('/data/farfield_matching')
OUT=ROOT/'runs/260913_accuracy_recovery'
from all13_inputs import resolve
parser=argparse.ArgumentParser()
parser.add_argument('--dataset',required=True)
parser.add_argument('--end',type=int,required=True)
parser.add_argument('--preflight-only',action='store_true')
parser.add_argument('--resume-report',type=Path)
args=parser.parse_args()
END=args.end
DS=args.dataset
inputs=resolve(DS)
assert 8<=END<inputs['row']['n_frames']
tracks_dir=inputs['manifest_path'].parent
manifest=inputs['manifest']
config=inputs['config']
frames_ref={'path':str(inputs['frames_path'])}
base=ROOT/'datasets'/DS
checkpoint=ROOT/'models/sam2/sam2.1_hiera_large.pt'
assert hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()==manifest['config']['source_digests']['sam2_checkpoint']
video=inputs['video']
assert video.is_file(), f'Missing canonical video: {video}'
result=dataset.run_ingest(base,Path(frames_ref['path']),dataset.IngestParams(**config['ingest']))
ordered=sorted(result.frames,key=lambda f:f.frame_idx)
with Image.open(base/'panorama'/f'{ordered[0].pano_stem}.jpg') as im: width,height=im.size
assert width==config['tracking']['reference_pano_width']
obs_by_frame,boxes={},{}
for obs in result.observations:
    obs_by_frame.setdefault(obs.frame_idx,[]).append(obs)
    boxes[obs.obs_id]=geo.pano_bbox_for_observation(obs.boxes,width,height,config['ingest']['fov_deg'])
builder_cfg=tb.TrackBuilderConfig(**{f.name:config['tracking'][f.name] for f in dataclasses.fields(tb.TrackBuilderConfig)})


class SnapshotModel:
    def __init__(self):
        self.course=CausalGeodeticCourse(10.,3.)
        self.cursor=0
        self.snapshot=None
        self.intervals=[]

    def advance(self,start,clock):
        while self.cursor<len(ordered) and ordered[self.cursor].time_s<=clock:
            fix=ordered[self.cursor]
            self.snapshot=self.course.append(fix.time_s,fix.lat,fix.lon)
            self.cursor+=1
        if self.snapshot is None: raise RuntimeError('No observed fix at current interval')
        self.snapshot=dataclasses.replace(self.snapshot,available_time_s=clock)
        self.intervals.append({'interval_start_s':start,'available_time_s':clock,
            'last_consumed_fix_keyframe':ordered[self.cursor-1].frame_idx,
            'last_consumed_fix_time_s':ordered[self.cursor-1].time_s,
            'rotation_compensation_available':self.snapshot.rotation_compensation_available,
            'coordinate_anchor_lat_lon':self.course.anchor})

    def delta_course_cw_deg(self,t,reference_t):
        return self.snapshot.delta_course_cw_deg(t,reference_t)


class ClockedVideo:
    def __init__(self,provider,model): self.provider,self.model=provider,model
    def frames_between(self,start,end):
        # Video addressing rounds to the nearest frame. Its actual timestamp
        # is part of availability; no future camera frame is labelled earlier.
        clock=max(end,self.provider.time_at_index(self.provider.index_at_time(end)))
        last_index=self.provider.index_at_time(end)
        dependencies=leveling['readiness_seconds_by_video_frame'][:last_index+1] if leveling else []
        if any(value is None for value in dependencies):
            raise RuntimeError('Video prefix needs EOF-only leveling evidence')
        clock=max(clock,max(dependencies,default=clock))
        self.model.advance(start,clock)
        return self.provider.frames_between(start,end)


level_path=OUT/f'{DS}.leveling_readiness.json'
metadata=json.loads((base/'pipeline_metadata.json').read_text())
leveling=json.loads(level_path.read_text()) if metadata.get('horizon_leveling') else None
for source,digest in (leveling['source_sha256'] if leveling else {}).items():
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest,source
provider=video_frames.VideoFrameProvider(video)
if leveling: assert abs(provider.fps-leveling['settings']['output_fps'])<1e-6
assert all(0<=provider.index_at_time(f.time_s)<provider.n_frames for f in ordered[:END+1])
model=SnapshotModel()
frames=ClockedVideo(provider,model)
release_policy=ValidBirthPrefixRelease()
emissions=[]
original_builder=tb.TrackBuilder
frame_times={f.frame_idx:f.time_s for f in ordered}
class CapturingBuilder(original_builder):
    def step(self,keyframe,*args,**kwargs):
        # A streaming tracker cannot know the requested experiment endpoint.
        kwargs['allow_new_births']=True
        output=super().step(keyframe,*args,**kwargs)
        current=keyframe+1
        live=rr.track_artifact(self,builder_cfg,'live_prefix',0,current)
        for track in live['tracks']:
            emission=release_policy.consider(track,frame_times,current,model.snapshot.available_time_s)
            if emission is not None:
                encoded=json.dumps(emission,sort_keys=True).encode()
                emissions.append(dict(release=emission,sha256=hashlib.sha256(encoded).hexdigest()))
        if current%8==0 or current==END:
            save_snapshot(current,live,self)
        return output

# Pin both runtime sources and the immutable observation inputs before inference.
files=[Path(__file__),OUT/'tools/prefix_release_policy.py',OUT/'tools/prefix_release_policy_v2.py',
       OUT/'tools/tracker_checkpoint.py',OUT/'tools/resume_tracking_range.py',OUT/'tools/tracking_components.py',
       OUT/'tools/causal_geodetic_course.py',OUT/'tools/causal_crop_course.py',
       OUT/'tools/online_time_course.py',OUT/'tools/tracking_bootstrap.py',OUT/'tools/all13_inputs.py',
       tracks_dir/'manifest.json',Path(frames_ref['path'])/'manifest.json',
       Path(frames_ref['path'])/'predictions.jsonl',base/'frames_gps.csv',base/'pipeline_metadata.json',
       Path(dataset.__file__),Path(rr.__file__),Path(tb.__file__),Path(sam_backend.__file__),
       Path(video_frames.__file__),Path(geo.__file__)]
if leveling:files.append(level_path)
source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
video_stat=video.stat()
video_identity=dict(path=str(video),size=video_stat.st_size,mtime_ns=video_stat.st_mtime_ns,
                    sha256=hashlib.file_digest(video.open('rb'),'sha256').hexdigest())

assert video_identity['sha256']==manifest['config']['source_digests']['video'], 'Canonical video hash differs'
if args.preflight_only:
    print(json.dumps(dict(dataset=DS,end=END,video_identity=video_identity,source_sha256=source_hashes,
        tracking_config=dataclasses.asdict(builder_cfg),first_time_s=ordered[0].time_s,last_time_s=ordered[END].time_s,
        leveling_dependency_bound=bool(leveling),model_api_calls=0)),flush=True)
    provider.close()
    raise SystemExit(0)
backend=sam_backend.Sam2Backend(checkpoint)
elapsed_offset=0.
state_identity=dict(source_sha256=source_hashes,video_identity=video_identity,
                    checkpoint_sha256=manifest['config']['source_digests']['sam2_checkpoint'],
                    tracking_config=dataclasses.asdict(builder_cfg))
(OUT/'tracker_states').mkdir(exist_ok=True)

def save_snapshot(end,payload,live_builder):
    stem=f'{DS}.all13_prefix_v1.end{end}'
    output_path=OUT/f'{stem}.json'
    report_path=OUT/f'{stem}.report.json'
    for f,sha in source_hashes.items():
        assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==sha,f
    current_stat=video.stat()
    assert (current_stat.st_size,current_stat.st_mtime_ns)==(video_identity['size'],video_identity['mtime_ns'])
    contents=json.dumps(dict(tracking=payload,emissions=emissions))
    if output_path.exists():
        assert output_path.read_text()==contents,'Immutable checkpoint would change'
        return
    output_path.write_text(contents)
    elapsed=elapsed_offset+time.monotonic()-started
    state_reference=write_checkpoint(OUT/'tracker_states'/f'{DS}.all13_prefix_v1.f{end:04d}.pkl.gz',
        live_builder,model,release_policy,emissions,frame=end,elapsed_seconds=elapsed,identity=state_identity,
        rng_state=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
    report=dict(dataset=DS,window=[0,end],elapsed_seconds=elapsed,tracker_state=state_reference,
        emission_count=len(emissions),early_emission_count=sum(e['release']['reason']=='age_and_support_prefix' for e in emissions),
        parameters=dict(min_supported_frames=3,min_age_seconds=10.),
        emitted_snapshots_unchanged=True,course_availability=model.intervals,
        output_sha256={str(output_path):hashlib.sha256(output_path.read_bytes()).hexdigest()},
        source_sha256=source_hashes,video_identity=video_identity,
        sam2_checkpoint_sha256=manifest['config']['source_digests']['sam2_checkpoint'],
        panorama_index_names=sorted(f.name for f in (base/'panorama').glob('*.jpg')),
        model_api_calls=0,new_semantic_audits_generated=False,endpoint_does_not_suppress_births=True,
        qualification='Resumable local causal-course tracker with immutable live snapshots and rejected-birth exclusion; endpoint never suppresses births. Inferred leveling readiness honored, but historical raw-fit/stabilization parity, calibration and processing latency remain qualified. Snapshot evidence supports paired one-shot and cumulative evaluations. Cached detectors and supplied calibration are not end-to-end latency/provenance certification.')
    report_path.write_text(json.dumps(report,indent=2))
    print('LIVE_PREFIX_STREAM_V3_CHECKPOINT',end,len(emissions),'emissions',report['elapsed_seconds'],'seconds',flush=True)

tb.TrackBuilder=CapturingBuilder
torch.manual_seed(0)
start_frame=0
existing_builder=None
if args.resume_report:
    previous=json.loads(args.resume_report.read_text())
    for name,sha in previous['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    saved=read_checkpoint(previous['tracker_state'],state_identity)
    start_frame=saved['frame']
    assert 0<start_frame<END
    elapsed_offset=saved['elapsed_seconds']
    existing_builder=CapturingBuilder(backend,builder_cfg,width,height)
    emissions=restore_checkpoint(saved,existing_builder,model,release_policy)
    torch.set_rng_state(saved['rng_state']['cpu'])
    torch.cuda.set_rng_state_all(saved['rng_state']['cuda'])
    print('RESUMED_TRACKER_STATE',start_frame,len(emissions),'emissions',flush=True)
started=time.monotonic()
try:
    builder,payload=run_range('causal_prefix_release',start_frame,END,builder_cfg,backend,
        frames,model,result,obs_by_frame,boxes,width,height,base,
        log=lambda message:print(message,flush=True),existing_builder=existing_builder)
finally:
    provider.close()
    tb.TrackBuilder=original_builder
for event in emissions:
    assert hashlib.sha256(json.dumps(event['release'],sort_keys=True).encode()).hexdigest()==event['sha256']
print('LIVE_PREFIX_STREAM_V3_COMPLETE',END,len(emissions),'emissions',flush=True)
