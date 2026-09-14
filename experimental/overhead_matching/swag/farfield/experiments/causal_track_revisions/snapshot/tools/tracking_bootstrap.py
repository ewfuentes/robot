"""Add isolated local tracking dependencies; preserve the Python network deny hook."""
import os
from pathlib import Path
import runpy
import sys

deps=Path('/data/farfield_matching/runs/260913_accuracy_recovery/tracking_dependencies')
sys.path[:0]=[str(deps/'site-packages'),str(deps/'sam2')]
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
target=sys.argv[1]
sys.argv=sys.argv[1:]
runpy.run_path(target,run_name='__main__')
