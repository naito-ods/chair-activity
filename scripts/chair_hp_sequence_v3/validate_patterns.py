"""Reproducible synthetic checks for the sequence rules and missingness guards."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from analyze_patterns import BASE,classify_blocks,make_blocks,map_observations


def sample(values):
    n=len(values)
    block=pd.DataFrame([dict(case_id='synthetic',block_id=1,hp=2,start_s=0.,end_s=float(n),
        active_s=float(n),elapsed_s=float(n),n_runs=1,previous_hp=3,previous_gap_s=10.,
        unknown_gap_s=0.,post_hp3=True,context='POST_HP3')])
    p=pd.DataFrame(dict(case_id='synthetic',block_id=1,run_id=1,rpm=values,
        time_s=np.arange(n)+.5,weight_s=1.,active_clock_s=np.arange(n)+.5,
        active_progress=(np.arange(n)+.5)/n,bin_active_start=np.arange(n,dtype=float),
        bin_active_end=np.arange(1,n+1,dtype=float),run_end_s=float(n)))
    return block,p


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output
    checks={}
    b,p=sample([3000.]*100);r,_=classify_blocks(b,p,BASE);checks['hp3_then_low']=bool(r.post_hp3_low.iloc[0])
    b,p=sample([40000.]*100);r,_=classify_blocks(b,p,BASE);checks['high_observed']=bool(r.high_dominant.iloc[0])
    b,p=sample([40000.]*90+[20000.]*10);r,_=classify_blocks(b,p,{**BASE,'tail_zone':'M'})
    checks['high_then_supported_mid']=bool(r.high_then_short_low.iloc[0])
    checks['ten_second_tail_not_stop_only']=not bool(r.terminal_low_near_stop.iloc[0])
    b,p=sample([40000.]*99+[20000.]);r,_=classify_blocks(b,p,{**BASE,'tail_zone':'M'})
    checks['single_tail_not_confirmed']=not bool(r.high_then_short_low.iloc[0])
    checks['single_tail_stop_flag']=bool(r.terminal_low_near_stop.iloc[0])
    b,p=sample([40000.]*90+[20000.]*3)
    p.loc[p.index>=90,'run_id']=[2,3,4]
    p.loc[p.index>=90,'time_s']=[91.5,93.5,95.5]
    p.loc[p.index>=90,'run_end_s']=[92.,94.,96.]
    p.loc[p.index<90,'run_end_s']=90.
    b.loc[0,'n_runs']=4;b.loc[0,'end_s']=96.;b.loc[0,'elapsed_s']=96.
    r,_=classify_blocks(b,p,{**BASE,'tail_zone':'M'})
    checks['multiple_short_runs_stop_flag']=bool(r.terminal_low_near_stop.iloc[0])
    runs=pd.DataFrame([dict(case_id='x',run_id=1,hp=2,start_s=0.,end_s=5.,active_s=5.),
                       dict(case_id='x',run_id=2,hp=2,start_s=10.,end_s=15.,active_s=5.)])
    intervals=pd.DataFrame(dict(case_id=['x']*3,start_s=[0.,5.,10.],end_s=[5.,10.,15.],known=[True,False,True]))
    blocks,_=make_blocks(runs,intervals,30.);checks['unknown_gap_not_merged']=len(blocks)==2
    intervals['known']=True;blocks,_=make_blocks(runs,intervals,30.)
    checks['known_off_merge_preserves_active']=len(blocks)==1 and blocks.active_s.iloc[0]==10 and blocks.elapsed_s.iloc[0]==15
    assert all(checks.values()),checks
    checks={key:bool(value) for key,value in checks.items()}
    (out/'process/synthetic_validation.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
