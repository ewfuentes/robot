"""Timestamp-based online confidence observations and adaptive output policy.

Policy selection reads only timestamps, estimated positions and posterior mass.
Truth-dependent fields are attached by the scoring observer and summarized later.
"""
import csv
import math
from pathlib import Path
import numpy as np
import torch


def read_times(input_dir, n):
    with (Path(input_dir) / 'motion_source.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    times = np.asarray([float(r['video_t_s']) for r in rows])
    if len(times) != n or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('Need one strictly increasing video timestamp per keyframe')
    if [int(r['idx']) for r in rows] != list(range(n)):
        raise ValueError('Motion timestamps must be in keyframe order')
    return times


def posterior_confidence(message, grid, state):
    marginal = message.sum(dim=0)
    ix = int(round((state['east_m']-grid.east_min)/grid.cell_m-.5))
    iy = int(round((state['north_m']-grid.north_min)/grid.cell_m-.5))
    masses = {}
    for radius in (500., 1000.):
        r = math.ceil(radius/grid.cell_m)
        y0,y1=max(0,iy-r),min(grid.n_north,iy+r+1)
        x0,x1=max(0,ix-r),min(grid.n_east,ix+r+1)
        yy=torch.arange(y0,y1,device=message.device)-iy
        xx=torch.arange(x0,x1,device=message.device)-ix
        disk=(yy[:,None]**2+xx[None,:]**2)*grid.cell_m**2 <= radius**2
        masses[radius]=float(marginal[y0:y1,x0:x1][disk].sum())
    hm=message.sum(dim=(1,2))
    entropy=float(-(hm*torch.log(hm.clamp_min(1e-30))).sum())
    return {'mass_within_500m_estimate':masses[500.],
            'mass_outside_1000m_estimate':max(0.,1-masses[1000.]),
            'heading_entropy_nats':entropy,
            'spatial_mass_method':'cell_centers_in_disk'}


def choose_adaptive(records, times, threshold=.95, stability_m=200., maximum=30.):
    """Return emitted record indices and times, with no access to truth fields."""
    selections=[]
    previous_emission=-float('inf')
    for k, row in enumerate(records):
        if times[k]+maximum > times[-1]+1e-8:
            break
        chosen=len(row)-1
        confident=False
        emission=times[k]+maximum
        for i in range(2,len(row)):
            recent=row[i-2:i+1]
            if any(r['confidence']['mass_within_500m_estimate'] < threshold for r in recent):
                continue
            positions=[(r['state']['east_m'],r['state']['north_m']) for r in recent]
            if max(math.dist(positions[a],positions[b]) for a in range(3) for b in range(a)) > stability_m:
                continue
            chosen=i;confident=True;emission=row[i]['asof_time'];break
        emission=max(emission,previous_emission)
        if emission > times[k]+maximum+1e-7:
            raise AssertionError('Monotonic output must respect per-pose deadline')
        previous_emission=emission
        selections.append({'pose':k,'record':chosen,'emitted_time':emission,
                           'delay_seconds':emission-times[k],'confident':confident})
    return selections


def finish(records, times, distances):
    complete=[k for k,t in enumerate(times) if t+30 <= times[-1]+1e-8]
    def summarize(selected):
        samples=[records[s['pose']][s['record']] for s in selected]
        mass=np.asarray([r['observer']['mass500'] for r in samples])
        error=np.asarray([r['observer']['map_error_m'] for r in samples])
        delays=np.asarray([s['delay_seconds'] for s in selected])
        conf=np.asarray([s['confident'] for s in selected],dtype=bool)
        pos=np.asarray([s['pose'] for s in selected],dtype=int)
        dt=np.diff(times[pos]);dd=np.diff(distances[pos])
        return {'n':len(samples),'dn_mass_500':float(np.sum((mass[1:]+mass[:-1])*.5*dd)/max(sum(dd),1e-9)),
                'actual_time_mass_500':float(np.sum((mass[1:]+mass[:-1])*.5*dt)/max(sum(dt),1e-9)),
                'map_error_median':float(np.median(error)), 'map_error_p95':float(np.percentile(error,95)),
                'map_within_500_fraction':float(np.mean(error<=500)),
                'delay_mean':float(np.mean(delays)), 'delay_median':float(np.median(delays)),
                'delay_p95':float(np.percentile(delays,95)),
                'fraction_by_5':float(np.mean(delays<=5+1e-7)), 'fraction_by_10':float(np.mean(delays<=10+1e-7)),
                'fraction_by_30':float(np.mean(delays<=30+1e-7)),
                'uncertain_fraction':float(np.mean(~conf)),
                'false_confident_fraction_of_confident':float(np.mean(error[conf]>500)) if conf.any() else None}
    policies={}
    for delay in (0,5,10,20,30):
        selected=[]
        for k in complete:
            eligible=[i for i,r in enumerate(records[k]) if r['asof_time'] <= times[k]+delay+1e-8]
            i=eligible[-1]
            selected.append({'pose':k,'record':i,'delay_seconds':delay,
                             'confident':records[k][i]['confidence']['mass_within_500m_estimate']>=.95})
        policies[f'fixed_{delay}s']={'summary':summarize(selected),'selection':selected}
    selected=choose_adaptive(records,times)
    policies['adaptive']={'summary':summarize(selected),'selection':selected}
    return {'semantics':'prefix-only joint-release factors; observation-time latency, compute excluded',
            'policy':'95% mass and <=200m change across three consecutive available-frame assessments; ordered emission by 30s',
            'timestamp_source':'motion_source.csv:video_t_s; no position fields used by policy',
            'terminal_incomplete_poses_excluded':len(times)-len(complete),
            'policies':policies,'records':records}
