"""Exact local tracker state, excluding the stateless per-interval SAM backend."""
import gzip
import hashlib
import pickle
from pathlib import Path


def write_checkpoint(path,builder,model,policy,emissions,*,frame,elapsed_seconds,identity,rng_state=None):
    path=Path(path)
    assert builder.on_interval is None
    state={k:v for k,v in builder.__dict__.items() if k not in {'backend','on_interval'}}
    payload=dict(schema='local_tracker_resume/v1',builder=state,model=model.__dict__,
        policy=policy.__dict__,emissions=emissions,frame=frame,
        elapsed_seconds=elapsed_seconds,identity=identity,rng_state=rng_state)
    encoded=gzip.compress(pickle.dumps(payload,protocol=5),compresslevel=1,mtime=0)
    if path.exists():
        assert path.read_bytes()==encoded,'Refuse to overwrite a different checkpoint'
    else:
        tmp=path.with_suffix(path.suffix+'.tmp')
        tmp.write_bytes(encoded)
        tmp.replace(path)
    return dict(path=str(path),sha256=hashlib.sha256(encoded).hexdigest(),bytes=len(encoded))


def read_checkpoint(reference,identity):
    encoded=Path(reference['path']).read_bytes()
    assert hashlib.sha256(encoded).hexdigest()==reference['sha256']
    # Only locally generated, checksum-bound checkpoints are accepted here.
    payload=pickle.loads(gzip.decompress(encoded))
    assert payload['schema']=='local_tracker_resume/v1' and payload['identity']==identity
    return payload


def restore_checkpoint(payload,builder,model,policy):
    backend=builder.backend
    builder.__dict__.update(payload['builder'])
    builder.backend=backend
    builder.on_interval=None
    model.__dict__.update(payload['model'])
    policy.__dict__.update(payload['policy'])
    return payload['emissions']
