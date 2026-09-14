"""One sequential worker per GPU; independent sequence shards share frozen settings."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def select_rows(rows,datasets,shard_index,num_shards):
    if num_shards<1 or not 0<=shard_index<num_shards:raise ValueError('Invalid shard')
    if datasets:
        unknown=set(datasets)-{r['dataset'] for r in rows}
        if unknown:raise ValueError(f'Unknown datasets: {sorted(unknown)}')
        rows=[r for r in rows if r['dataset'] in datasets]
    return rows[shard_index::num_shards]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--datasets',nargs='+');p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--stage',choices=['all','preflight','track','evidence','filter'],default='all')
    p.add_argument('--end',type=int,help='Optional shorter diagnostic screen; omit for all frames')
    a=p.parse_args();w=a.workspace.resolve()
    proof=json.loads((w/'workspace_manifest.json').read_text())
    def verify():
        for rel,sha in proof['installed_source_sha256'].items():
            assert digest(w/rel)==sha, f'Workspace source changed: {rel}'
    verify()
    rows=select_rows(json.loads((w/'overnight_all13_manifest.json').read_text())['rows'],a.datasets,a.shard_index,a.num_shards)
    if not rows:raise ValueError('Empty shard')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),FARFIELD_REPO=proof['repo'])
    locks=w/'worker_locks';locks.mkdir(exist_ok=True)
    gpu_handle=None
    if a.stage!='preflight':
        gpu_handle=open(Path('/tmp')/f'farfield-revision-gpu-{os.getuid()}-{a.gpu}.lock','a+')
        fcntl.flock(gpu_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        busy=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if busy:raise RuntimeError(f'GPU {a.gpu} already has compute processes: {busy}')
    report=dict(gpu=a.gpu,stage=a.stage,shard_index=a.shard_index,num_shards=a.num_shards,
        seeds=a.seeds,rows=[],model_api_calls=0,started_unix_s=time.time(),
        qualification='Experimental causal-filter evaluation; supplied calibration and preprocessing are not fully certified online. Runtime is reported, not an acceptance condition. No paper-ready adoption claim.')
    target=w/f'worker.gpu{a.gpu}.shard{a.shard_index}.{time.time_ns()}.json'
    def save():target.write_text(json.dumps(report,indent=2))
    def command(name,script,args,tracking=False):
        verify();cmd=[str(w/'tools/python.sh')]
        if tracking:cmd.append(str(w/'tools/tracking_bootstrap.py'))
        cmd.extend([str(w/'tools'/script),*map(str,args)])
        log=w/'logs'/f'{name}.{time.time_ns()}.log'
        print('START',name,'log',log,flush=True)
        with log.open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        verify();print('DONE',name,flush=True)
    for row in rows:
        ds=row['dataset'];end=a.end if a.end is not None else row['n_frames']-1
        entry=dict(dataset=ds,end=end,status='running');report['rows'].append(entry);save()
        lock=open(locks/(ds+'.lock'),'a+')
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not 8<=end<row['n_frames']:raise ValueError(f'Invalid endpoint for {ds}: {end}')
            stem=f'{ds}.all13_prefix_v1.end{end}';track_report=w/(stem+'.report.json')
            raw=w/f'{ds}.all13.end{end}.evidence.json';named=w/f'{ds}.all13.end{end}.names.json';updated=w/f'{ds}.all13.end{end}.revisions.json'
            if a.stage in {'all','preflight'}:
                command(f'{ds}.preflight','replay_live_prefix_all13_v1.py',['--dataset',ds,'--end',end,'--preflight-only'],tracking=True)
            if a.stage in {'all','track'} and not track_report.exists():
                prior=[]
                for f in w.glob(f'{ds}.all13_prefix_v1.end*.report.json'):
                    k=json.loads(f.read_text())['window'][1]
                    if k<end:prior.append((k,f))
                extra=['--resume-report',max(prior)[1]] if prior else []
                command(stem,'replay_live_prefix_all13_v1.py',['--dataset',ds,'--end',end,*extra],tracking=True)
            if a.stage in {'all','track','evidence','filter'}:
                rp=json.loads(track_report.read_text())
                for group in ['source_sha256','output_sha256']:
                    for path,sha in rp[group].items():assert digest(path)==sha,path
                entry['tracker_elapsed_seconds']=rp['elapsed_seconds']
            if a.stage in {'all','evidence'}:
                for output,script,args in [(raw,'build_live_prefix_evidence_all13_v1.py',['--report',track_report,'--out',raw]),
                    (named,'build_prefix_name_evidence_all13_v1.py',['--evidence',raw,'--out',named]),
                    (updated,'build_track_revision_evidence_all13_v1.py',['--evidence',named,'--end',end,'--out',updated])]:
                    if not output.exists():command(output.stem,script,args)
                    for path,sha in json.loads(output.read_text())['input_sha256'].items():assert digest(path)==sha,path
            if a.stage in {'all','filter'}:
                entry['comparisons']=[]
                for seed in a.seeds:
                    pair={}
                    for kind,evidence,script in [('tempered',named,'run_prefix_predictive_all13_v1.py'),('cumulative',updated,'run_track_revision_all13_v1.py')]:
                        output=w/'results'/f'{ds}.handoff.{kind}.seed{seed}.end{end}.json'
                        if not output.exists():command(output.stem,script,['--evidence',evidence,'--reference-stem',
                            f"{ds}.dedup.geometry_integrated_v1_prefix{row['n_frames']-2}.seed0",'--end',end,'--seed',seed,
                            '--mode','prefix','--mixture','max','--predictive-tempering',1,'--out',output])
                        record=json.loads(output.with_suffix('.run.json').read_text())
                        assert record['returncode']==0 and digest(output)==record['output_sha256']
                        for path,sha in record['source_and_input_sha256'].items():assert digest(path)==sha,path
                        pair[kind]=json.loads(output.read_text())
                        entry.setdefault('outputs',[]).append(str(output))
                    baseline,candidate=pair['tempered'],pair['cumulative']
                    assert all(candidate['config'].get(k)==v for k,v in baseline['config'].items() if k not in {'out','release_schedule'})
                    entry['comparisons'].append(dict(seed=seed,baseline=baseline['summary'],candidate=candidate['summary'],
                        delta={str(r):candidate['summary'][f'dn_mass_{r}']-baseline['summary'][f'dn_mass_{r}'] for r in [50,100,250,500,1000]}));save()
            entry['status']='stage_complete'
        except Exception as error:
            entry.update(status='failed',error=repr(error));print('FAILED',ds,repr(error),flush=True)
        finally:lock.close();save()
    report['finished_unix_s']=time.time();save();print('WORKER_REPORT',target,flush=True)
    if any(r['status']=='failed' for r in report['rows']):raise SystemExit(1)

if __name__=='__main__':main()
