"""Resolve dataset-specific input paths; never vary inference parameters by dataset."""
import copy
import hashlib
import json
from pathlib import Path
ROOT=Path('/data/farfield_matching')
OUT=ROOT/'runs/260913_accuracy_recovery'
def resolve(dataset):
    rows=json.loads((OUT/'overnight_all13_manifest.json').read_text())['rows']
    row=next(r for r in rows if r['dataset']==dataset)
    path=Path(row['tracking_manifest'])
    manifest=json.loads(path.read_text())
    ref=next(r for r in manifest['upstreams'] if r['kind']=='frame_landmarks')
    frames=Path(ref['path'])
    if not frames.is_dir():
        frames=ROOT/'artifacts/frame_landmarks'/dataset/ref['version']
    fm=json.loads((frames/'manifest.json').read_text())
    assert fm['content_digest']==ref['content_digest'], 'Relocated detections differ'
    config=copy.deepcopy(manifest['config']['resolved'])
    reference=json.loads(Path(rows[0]['tracking_manifest']).read_text())['config']['resolved']
    for section,values in config.items():
        for key,value in values.items():
            if section=='tracking' and key in {'sam2_checkpoint','reference_pano_width'}:continue
            assert value==reference[section][key], (dataset,section,key)
    config['tracking']['sam2_checkpoint']=str(ROOT/'models/sam2/sam2.1_hiera_large.pt')
    historical=OUT/'results'/f"{dataset}.dedup.geometry_integrated_v1_prefix{row['n_frames']-2}.seed0.json"
    baseline=json.loads(historical.read_text())
    return dict(row=row,manifest=manifest,manifest_path=path,config=config,
        frames_path=frames,video=Path(row['video_path']),base_input=Path(baseline['config']['input_dir']),
        reference_stem=historical.stem)
