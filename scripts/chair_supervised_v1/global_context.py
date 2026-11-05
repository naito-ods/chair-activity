"""Offline whole-case chronology from raw HP states, never annotation labels."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from features import state_segments
from extract import load_case_events

def global_features(events,start,end,segments=None):
    lo=float(events.time_s.min());hi=float(events.time_s.max())+1e-6
    h=state_segments(events,'20039',lo,hi) if segments is None else segments
    mid=(start+end)/2;duration=max(hi-lo,1e-9)
    f={'global_interval_start_progress':(start-lo)/duration,'global_interval_end_progress':(end-lo)/duration}
    for name,a,b in [('before',lo,start),('after',end,hi),('whole',lo,hi)]:
        widths=np.maximum(0,np.minimum(h[:,1],b)-np.maximum(h[:,0],a))
        known=np.isfinite(h[:,2]);total=widths[known].sum();span=max(0,b-a)
        f[f'global_{name}_hp_known_fraction']=total/span if span>0 else np.nan
        for hp in [0,1,2,3]:
            own=widths[h[:,2]==hp].sum()
            f[f'global_{name}_hp{hp}_known_share']=own/total if total>0 else np.nan
            if hp>0:
                f[f'global_{name}_hp{hp}_active_s']=float(own) if total>0 else np.nan
                f[f'global_{name}_hp{hp}_n_runs']=float(((widths>0)&(h[:,2]==hp)).sum()) if total>0 else np.nan
    def unknown_between(a,b):
        widths=np.maximum(0,np.minimum(h[:,1],b)-np.maximum(h[:,0],a));return widths[np.isnan(h[:,2])].sum()>1e-9
    for hp in [1,2,3]:
        before=h[(h[:,2]==hp)&(h[:,1]<=start)];after=h[(h[:,2]==hp)&(h[:,0]>=end)]
        f[f'global_since_last_hp{hp}_s']=float(start-before[-1,1]) if len(before) and not unknown_between(before[-1,1],start) else np.nan
        f[f'global_until_next_hp{hp}_s']=float(after[0,0]-end) if len(after) and not unknown_between(end,after[0,0]) else np.nan
        pos=h[h[:,2]==hp]
        f[f'global_before_first_hp{hp}']=float(mid<pos[0,0]) if len(pos) else np.nan
        f[f'global_after_last_hp{hp}']=float(mid>pos[-1,1]) if len(pos) else np.nan
    return f

def augment(out,case_root):
    p=out/'features_context.csv'
    if p.exists():raise FileExistsError('Preserve previous context export')
    meta=pd.read_csv(out/'annotations_aligned.csv');features=pd.read_csv(out/'features_extended.csv');assert meta.id.tolist()==features.id.tolist()
    rows=[]
    for seq,g in meta.groupby('seq',sort=False):
        ev=load_case_events(case_root/'cases'/f'{seq}.csv.gz');h=state_segments(ev,'20039',float(ev.time_s.min()),float(ev.time_s.max())+1e-6)
        for r in g.itertuples():rows.append({'id':r.id,**global_features(ev,r.start_s,r.end_s,h)})
    f=pd.DataFrame(rows);result=features.merge(f,on='id',how='left',validate='one_to_one');assert result.id.tolist()==meta.id.tolist()
    result.to_csv(p,index=False)
    print(json.dumps({'file':str(p),'rows':len(result),'features':len(result.columns)-1,'future_raw_context':'offline only; no annotation labels in context'}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--case-root',type=Path,required=True);a=ap.parse_args();augment(a.output,a.case_root)
