"""Original range loop with explicit builder reuse and no duplicate reseeding."""
import inspect
from experimental.overhead_matching.swag.farfield.tracking import range_runner as rr

source=inspect.getsource(rr.run_range)
old='              on_interval=None, log=print):'
assert source.count(old)==1
source=source.replace(old,'              on_interval=None, log=print, existing_builder=None):')
old='    builder = tb.TrackBuilder(backend, builder_cfg, pano_w, pano_h,\n                              on_interval=on_interval)'
assert source.count(old)==1
source=source.replace(old,'    builder = (existing_builder if existing_builder is not None else\n               tb.TrackBuilder(backend, builder_cfg, pano_w, pano_h, on_interval=on_interval))')
assert source.count('    if k_start < k_end:')==1
source=source.replace('    if k_start < k_end:','    if existing_builder is None and k_start < k_end:')
namespace=dict(rr.__dict__)
exec(compile(source,__file__,'exec'),namespace)
run_range=namespace['run_range']
