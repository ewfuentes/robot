"""Build measurements and semantic evidence from an already emitted prefix.

No final audit, eventual track state, or future observation is used. Camera
mount calibration is a caller-supplied input whose provenance remains separate.
"""
import collections
import copy
import dataclasses
import math
from types import SimpleNamespace
from prefix_release_policy import SUPPORT_CLASSES
from support_likelihood_mixture import mix_tables
from reservoir_canonicalization import canonical_observation
from audit_only_v2 import table_for_track, THRESHOLD
from experimental.overhead_matching.swag.farfield.tracking import tracklets


def build_evidence(emission, observations, frame_times, *, tracklet_id,
                   pano_width, mount_bearing_camera_cw_deg,
                   bearing_sigma_deg=1., epoch_keyframes=5, signatures=None):
    track = copy.deepcopy(emission['track'])
    release = emission['release_keyframe']
    clock = emission['available_time_s']
    if (not math.isfinite(mount_bearing_camera_cw_deg)
            or frame_times[release]>clock):
        raise ValueError('Invalid mount calibration or release clock')
    if any(r['keyframe']>release or frame_times[r['keyframe']]>clock for r in track['records']):
        raise ValueError('Future mask record in emission')
    by_frame = collections.defaultdict(set)
    by_frame[track['birth_keyframe']].add(track['birth_obs_id'])
    for record in track['records']:
        for support in record.get('supports',[]):
            if support['class'] in SUPPORT_CLASSES:
                by_frame[record['keyframe']].add(support['obs_id'])
    used, obs_objects = [], {}
    for frame,ids in sorted(by_frame.items()):
        if frame>release or frame_times[frame]>clock:
            raise ValueError('Future support frame in emission')
        for obs_id in sorted(ids):
            obs = observations[obs_id]
            if obs['frame_idx']!=frame:
                raise ValueError('Support observation belongs to another frame')
            used.append(dict(obs_id=obs_id,frame=frame,frame_vote_share=1./len(ids)))
            obs_objects[obs_id] = SimpleNamespace(**obs)
    # A known unsupported tail contributes neither masks nor future validity
    # labels. Earlier propagated masks remain part of this raw prefix screen.
    last_supported = max(by_frame)
    track['records'] = [r for r in track['records'] if r['keyframe']<=last_supported]
    caps = tracklets.range_caps_by_keyframe(track,obs_objects)
    camera = [tracklets.CameraBearingObservation(
        tracklet_id=tracklet_id,keyframe_idx=k,bearing_camera_cw_deg=bearing,
        angular_width_deg=width,sigma_deg=bearing_sigma_deg,
        correlation_group=f'{tracklet_id}/emitted-prefix',range_max_m=caps.get(k))
        for k,bearing,width in tracklets.bearing_series(track,pano_width)]
    params = tracklets.TrackletParams(epoch_keyframes=epoch_keyframes,bearing_sigma_deg=bearing_sigma_deg)
    fused = tracklets.epoch_fused_compat_v1(camera,params)
    measurements = [dict(tracklet_id=m.tracklet_id,anchor_keyframe_idx=m.anchor_keyframe_idx,
        bearing_forward_cw_deg=(m.bearing_camera_cw_deg-mount_bearing_camera_cw_deg)%360.,
        kappa=m.kappa,range_max_m=m.range_max_m) for m in fused]
    evidence = dict(primary_object=dict(tags=[]),provenance=dict(
        method='immutable_prefix_supports_v1',release_keyframe=release,
        available_time_s=clock,supported_frame_count_including_birth=len(by_frame),
        observations=used,reuses_prior_track_audit=False,track_was_closed=track['status']=='closed'))
    table = None
    if signatures is not None:
        tables,weights = [],[]
        for member in used:
            obs = observations[member['obs_id']]
            tags = {(obs['primary_tag_key'],obs['primary_tag_value'])}
            tags.update(tuple(t) for t in obs.get('additional_tags',[]))
            query = dict(primary_object=dict(tags=[dict(tag=f'{k}={v}',weight=1.)
                         for k,v in sorted(tags) if k and v]))
            query,_ = canonical_observation(query,THRESHOLD)
            tables.append(table_for_track(tracklet_id,query,signatures))
            weights.append(member['frame_vote_share']/len(by_frame))
        table = mix_tables(tables,weights)
    return dict(measurements=measurements,camera_observations=[dataclasses.asdict(x) for x in camera],
                semantic_evidence=evidence,table=table,last_supported_keyframe=last_supported,
                qualification='Raw live-prefix masks and per-frame detections only; no final-audit acceptance. Detector compute latency and calibration provenance are not certified by this converter.')
