"""Install the frozen snapshot into an isolated experiment output directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

PACKAGE=Path(__file__).resolve().parent
CANONICAL_ROOT=Path('/data/farfield_matching')
ORIGINAL_OUT=CANONICAL_ROOT/'runs/260913_accuracy_recovery'

def prepare(source_run,output,repo):
    source_run=source_run.resolve();output=output.resolve();repo=repo.resolve()
    if output.exists():raise ValueError('Choose a new workspace; existing evidence is immutable')
    manifest=json.loads((PACKAGE/'overnight_all13_manifest.json').read_text())
    # These references remain byte-identical and retain their original hashes.
    # Dataset paths are intentionally not rewritten or re-certified by this tool.
    required=[]
    for row in manifest['rows']:
        stem=f"{row['dataset']}.dedup.geometry_integrated_v1_prefix{row['n_frames']-2}.seed0"
        required.extend([Path('results')/(stem+'.json'),Path('logs')/(stem+'.run.json'),
            Path('tables')/(row['dataset']+'.dedup.audit_only_v2.report.json')])
    missing=[str(source_run/p) for p in required if not (source_run/p).is_file()]
    if missing:raise FileNotFoundError('Missing reference inputs:\n'+'\n'.join(missing))
    snapshot=json.loads((PACKAGE/'snapshot_sources.json').read_text())
    for rel,sha in snapshot['snapshot_sha256'].items():
        assert hashlib.sha256((PACKAGE/'snapshot'/rel).read_bytes()).hexdigest()==sha, rel
    shutil.copytree(PACKAGE/'snapshot',output)
    for directory in ['results','logs','tables','tracker_states','factor_cache']:(output/directory).mkdir(exist_ok=True)
    for rel in required:
        shutil.copy2(source_run/rel,output/rel)
    shutil.copy2(PACKAGE/'overnight_all13_manifest.json',output/'overnight_all13_manifest.json')
    for p in source_run.glob('*.leveling_readiness.json'):shutil.copy2(p,output/p.name)
    dependencies=source_run/'tracking_dependencies'
    if dependencies.exists():(output/'tracking_dependencies').symlink_to(dependencies,target_is_directory=True)
    transformations=[]
    for path in output.rglob('*.py'):
        original=path.read_text();source=original
        source=source.replace(str(ORIGINAL_OUT),str(output))
        source=source.replace("ROOT/'runs/260913_accuracy_recovery'",'Path('+repr(str(output))+')')
        source=source.replace('ROOT=OUT.parent.parent','ROOT=Path('+repr(str(CANONICAL_ROOT))+')')
        # Never write to the inherited reference command's shared factor cache.
        if path.name in {'run_track_revision_all13_v1.py','run_prefix_predictive_all13_v1.py'}:
            old="command.extend(['--odometry_seed',str(args.seed)])"
            assert source.count(old)==1
            source=source.replace(old,old+"\ncommand.extend(['--factor_cache_dir',str(OUT/'factor_cache')])")
        if source!=original:
            path.write_text(source)
            transformations.append(dict(path=str(path.relative_to(output)),
                original_sha256=hashlib.sha256(original.encode()).hexdigest(),installed_sha256=hashlib.sha256(source.encode()).hexdigest()))
    launcher='''#!/usr/bin/env bash
set -euo pipefail
TASK_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_REPO=${FARFIELD_REPO:-REPO_DEFAULT}
TASK_RUNFILES=${FARFIELD_RUNFILES:-$TASK_REPO/bazel-bin/experimental/overhead_matching/swag/farfield/localization/grid_filter.runfiles}
TASK_PYTHONPATH="$TASK_WORKSPACE/tools:$TASK_REPO"
for task_dep in "$TASK_RUNFILES"/pip_3_12_*/site-packages; do
  [[ -d "$task_dep" ]] && TASK_PYTHONPATH="$TASK_PYTHONPATH:$task_dep"
done
export PYTHONPATH="$TASK_PYTHONPATH"
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_COMPILE_THREADS=2
# Honor CUDA_VISIBLE_DEVICES from the worker; never force physical GPU0.
export TORCHINDUCTOR_CACHE_DIR="$TASK_WORKSPACE/inductor_cache/${CUDA_VISIBLE_DEVICES:-default}"
TASK_PYTHON=${FARFIELD_PYTHON:-$TASK_RUNFILES/python_3_12_x86_64-unknown-linux-gnu/bin/python3}
exec "$TASK_PYTHON" "$@"
'''
    import shlex
    launcher=launcher.replace('REPO_DEFAULT',shlex.quote(str(repo)))
    (output/'tools/python.sh').write_text(launcher);(output/'tools/python.sh').chmod(0o755)
    proof=dict(schema='causal_revision_handoff_workspace/v1',repo=str(repo),source_run=str(source_run),
        dataset_root=str(CANONICAL_ROOT),transformations=transformations,
        installed_source_sha256={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.rglob('*') if p.is_file() and p.suffix in {'.py','.sh'}},
        qualification='Only experiment output location, Python launcher, and factor-cache destination are adapted. Input datasets and historical reference bytes are unchanged. This installation is not a new causality or accuracy certification.')
    (output/'workspace_manifest.json').write_text(json.dumps(proof,indent=2))
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,default=ORIGINAL_OUT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repo',type=Path,default=PACKAGE.parents[5])
    a=p.parse_args();print(prepare(a.source_run,a.output,a.repo))
