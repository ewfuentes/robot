"""Content-addressed exact FP32 factors, with a bounded disk LRU cache."""
import hashlib,json,os
from pathlib import Path
import torch
class FactorCache:
 def __init__(self,path,namespace,max_bytes=None):
  self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.namespace=namespace;self.hits=0;self.misses=0
  self.max_bytes=int(float(os.environ.get('FARFIELD_FACTOR_CACHE_MAX_GB','96'))*1024**3) if max_bytes is None else max_bytes
  if self.max_bytes<=0:raise ValueError('factor cache budget must be positive')
  self.total_bytes=sum(p.stat().st_size for p in self.path.glob('*.pt'))
  self._prune()
 def _prune(self):
  if self.total_bytes<=self.max_bytes:return
  files=sorted(self.path.glob('*.pt'),key=lambda p:p.stat().st_mtime_ns)
  for p in files:
   if self.total_bytes<=self.max_bytes:break
   size=p.stat().st_size;p.unlink();self.total_bytes-=size
 def get(self,identity,device,compute):
  key=hashlib.sha256(json.dumps([self.namespace,identity],sort_keys=True,separators=(',',':')).encode()).hexdigest();p=self.path/(key+'.pt')
  if p.exists():
   value=torch.load(p,map_location=device,weights_only=True);p.touch();self.hits+=1;return value
  value=compute();tmp=p.with_suffix(f'.{os.getpid()}.tmp');torch.save(value.detach().cpu(),tmp)
  size=tmp.stat().st_size
  if size<=self.max_bytes:
   tmp.replace(p);self.total_bytes+=size;self._prune()
  else:tmp.unlink()
  self.misses+=1;return value
