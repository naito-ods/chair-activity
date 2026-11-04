"""Positive-time agreement between HP3 and the independent scaler state.

Both-OFF agreement can dominate all-time accuracy. This audit isolates known
positive union/intersection, preserves initial unknown, and changes no labels.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from extract_hp_streams import state_intervals,lookup,sha


def positive_audit(events,duration):
    hp=state_intervals(events,'20039',duration,[0,1,2,3])
    scaler=state_intervals(events,'20774',duration,[0,1])
    breaks=np.unique(np.r_[hp[:,0],hp[:,1],scaler[:,0],scaler[:,1]])
    width=np.diff(breaks);mid=(breaks[:-1]+breaks[1:])/2
    hv,_=lookup(hp,mid);sv,_=lookup(scaler,mid)
    hknown=np.isfinite(hv);sknown=np.isfinite(sv);both=hknown&sknown
    hp3=hv==3;scaler_on=sv==1
    union=both&(hp3|scaler_on);intersection=both&hp3&scaler_on
    hp_only=both&hp3&~scaler_on;scaler_only=both&~hp3&scaler_on
    union_s=float(width[union].sum());intersection_s=float(width[intersection].sum())
    hp_positive_s=float(width[both&hp3].sum());scaler_positive_s=float(width[both&scaler_on].sum())
    values=dict(comparable_known_s=float(width[both].sum()),
        either_unknown_s=float(width[~both].sum()),
        positive_union_s=union_s,positive_intersection_s=intersection_s,
        hp3_only_s=float(width[hp_only].sum()),scaler_only_s=float(width[scaler_only].sum()),
        both_not_scaler_s=float(width[both&~hp3&~scaler_on].sum()),
        hp3_positive_comparable_s=hp_positive_s,scaler_positive_comparable_s=scaler_positive_s,
        hp3_known_positive_s=float(width[hknown&hp3].sum()),
        scaler_known_positive_s=float(width[sknown&scaler_on].sum()),
        hp3_positive_scaler_unknown_s=float(width[hknown&hp3&~sknown].sum()),
        scaler_positive_hp_unknown_s=float(width[sknown&scaler_on&~hknown].sum()),
        positive_jaccard=intersection_s/union_s if union_s>0 else np.nan,
        fraction_hp3_confirmed_by_scaler=intersection_s/hp_positive_s if hp_positive_s>0 else np.nan,
        fraction_scaler_confirmed_by_hp3=intersection_s/scaler_positive_s if scaler_positive_s>0 else np.nan)
    assert np.isclose(values['comparable_known_s']+values['either_unknown_s'],duration)
    assert np.isclose(union_s,intersection_s+values['hp3_only_s']+values['scaler_only_s'])
    assert np.isclose(values['comparable_known_s'],union_s+values['both_not_scaler_s'])
    return values


def case_audit(record):
    source=Path(record['source_path']);actual=sha(source)
    assert actual==record['source_sha256'],record['case_id']
    pieces=[]
    for chunk in pd.read_csv(source,usecols=['time_s','event_code','event_value'],
            dtype={'event_code':str},chunksize=20000,low_memory=False):
        chunk.event_code=chunk.event_code.str.strip()
        keep=chunk[chunk.event_code.isin(['20039','20774','90012'])].copy()
        if not keep.empty:
            keep['time_s']=pd.to_numeric(keep.time_s,errors='coerce')
            keep['numeric_value']=pd.to_numeric(keep.event_value,errors='coerce')
            pieces.append(keep[['time_s','event_code','numeric_value']])
    events=pd.concat(pieces,ignore_index=True)
    events=events[np.isfinite(events.time_s)].sort_values('time_s',kind='stable')
    marker=events[(events.event_code=='90012')&events.numeric_value.isin([0,1])]
    first=marker.loc[marker.numeric_value==1,'time_s'].min()
    v=marker.numeric_value.to_numpy(float)
    final=marker.loc[(v==0)&(np.r_[np.nan,v[:-1]]==1)&(marker.time_s>first),'time_s'].max()
    duration=float(record['duration_s'])
    assert np.isclose(first,0) and np.isclose(final,duration),record['case_id']
    result=positive_audit(events,duration)
    return dict(case_id=record['case_id'],duration_s=duration,**result,
        source_sha256=actual,source_hash_matches=True,boundary_matches=True)


def test_audit():
    frame=pd.DataFrame({'time_s':[1,4,7,9,2,5,8,9],
        'event_code':['20039']*4+['20774']*4,
        'numeric_value':[3,0,3,0,1,0,1,0]})
    a=positive_audit(frame,10)
    assert np.isclose(a['positive_union_s'],5)
    assert np.isclose(a['positive_intersection_s'],3)
    assert np.isclose(a['hp3_only_s'],1) and np.isclose(a['scaler_only_s'],1)
    assert np.isclose(a['positive_jaccard'],.6)
    assert np.isclose(a['hp3_positive_scaler_unknown_s'],1)
    assert np.isclose(a['either_unknown_s'],2)
    empty=pd.DataFrame(columns=['time_s','event_code','numeric_value'])
    b=positive_audit(empty,10)
    assert b['positive_union_s']==0 and np.isnan(b['positive_jaccard'])
    return dict(positive_union_intersection=True,initial_unknown_retained=True,
                disjoint_positive_parts=True,no_positive_denominator_nan=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();out=args.output.resolve()
    dest=out/'hp3_confirmation_audit.csv'
    if dest.exists():parser.error(f'Existing audit preserved: {dest}')
    def log(event,**values):
        row=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),event=event,**values)
        with (out/'process/positive_agreement_log.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    log('synthetic_validation',checks=test_audit())
    manifest=pd.read_csv(out/'case_manifest.csv')
    assert len(manifest)==300 and not manifest.case_id.duplicated().any()
    log('preflight',requested=300,workers=args.workers,
        reason='All-known agreement includes both OFF; audit positive-time concordance separately',
        policy='last same-timestamp record; exact union intervals; first unknown remains unknown')
    start=time.time();results={};failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future={pool.submit(case_audit,r):r for r in manifest.to_dict('records')}
        for n,f in enumerate(as_completed(future),1):
            row=future[f]
            try:results[row['case_id']]=f.result()
            except Exception as e:failures.append(dict(case_id=row['case_id'],error=repr(e)))
            if n%100==0:log('progress',completed=n,failed=len(failures))
    if failures:
        log('failed',failures=failures)
        raise RuntimeError('No partial audit exported')
    audit=pd.DataFrame([results[cid] for cid in manifest.case_id])
    assert len(audit)==300 and not audit.case_id.duplicated().any()
    # Verify the previous all-known comparison without editing its artifact.
    prior=pd.read_csv(out/'hp_status_audit.csv').set_index('case_id')
    indexed=audit.set_index('case_id')
    assert np.allclose(indexed.comparable_known_s,prior.hp3_scaler_comparable_s)
    assert np.allclose(indexed.hp3_only_s+indexed.scaler_only_s,prior.conflict_s)
    audit.to_csv(dest,index=False)
    union=float(audit.positive_union_s.sum());inter=float(audit.positive_intersection_s.sum())
    valid=audit.positive_jaccard.dropna()
    log('complete',requested=300,actual=300,cases_positive_union=int((audit.positive_union_s>0).sum()),
        positive_union_s=union,positive_intersection_s=inter,
        hp3_only_s=float(audit.hp3_only_s.sum()),scaler_only_s=float(audit.scaler_only_s.sum()),
        pooled_positive_jaccard=inter/union if union else None,
        median_case_positive_jaccard=float(valid.median()),q25_case_positive_jaccard=float(valid.quantile(.25)),
        q75_case_positive_jaccard=float(valid.quantile(.75)),min_case_positive_jaccard=float(valid.min()),
        elapsed_seconds=time.time()-start,source_hash_matches_all=True,
        script_sha256=sha(__file__),old_audit_unchanged=True)


if __name__=='__main__':main()
