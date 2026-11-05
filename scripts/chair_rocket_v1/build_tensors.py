"""Build 1s ROCKET tensors from explicit private120s window metadata and raw DAT."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import rocket_core as core
from extract import read_raw

def run(a):
    if a.output.exists():raise FileExistsError('Preserve existing tensors')
    meta=pd.read_csv(a.metadata);cases=pd.read_csv(a.case_manifest)
    if not meta.id.is_unique or not cases.seq.is_unique:raise ValueError('Duplicate IDs')
    if set(meta.seq)-set(cases.seq):raise ValueError('Window IDs missing from case manifest')
    np.testing.assert_allclose(meta.end_s-meta.start_s,120.,atol=1e-8,rtol=0)
    x=np.empty((len(meta),len(core.CHANNELS),120),np.float32);audits=[]
    for seq,g in meta.groupby('seq',sort=True):
        case=cases[cases.seq.eq(seq)].iloc[0]
        path=a.c31_root/'chair_export'/case.date/case.chair/f'Chair_log_seq_{int(seq)}.dat'
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        if before!=case.source_sha256:raise ValueError(f'Raw hash mismatch seq{seq}')
        ev,origin,end,audit=read_raw(path)
        if g.start_s.min()<0 or g.end_s.max()>float(case.raw_duration_s)+1e-8:raise ValueError('Window outside raw domain')
        for i,r in g.iterrows():x[i]=core.tensor(ev,float(r.start_s))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before
        audits.append(dict(seq=int(seq),source_sha256=before,windows=len(g)))
        print(json.dumps(audits[-1]),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,**{a.kind:x,a.kind+'_ids':meta.id.to_numpy(str),'channels':np.array(core.CHANNELS)})
    a.output.with_suffix('.audit.json').write_text(json.dumps(dict(requested_cases=int(meta.seq.nunique()),
        actual_cases=len(audits),windows=len(meta),source_unchanged=True,cases=audits),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['c31-root','metadata','case-manifest','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--kind',choices=['train','query'],default='train')
    a=p.parse_args()
    if a.output.suffix!='.npz':p.error('--output must end in .npz')
    run(a)
