"""Read-only extraction of exact HP runs and observed numeric samples for v3.

User-confirmed semantics: 20039=1/2/3 means HP1/HP2/HP3(scaler);
20774=1 indicates scaler operation. Unknown remains unknown, not OFF.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

NUMERIC = ['20742','20743','20108','10014','10015']
TARGETS = ['20011','20039','20742','20108','20743','20101','20774','20776','10014','10015']
READ_CODES = set(TARGETS + ['90012'])
INTERVAL_COLS=['case_id','interval_id','start_s','end_s','hp','known','duration_s']
RUN_COLS=['case_id','run_id','interval_id','hp','start_s','end_s','active_s']
OBS_COLS=['case_id','code','time_s','second_bin','value','hp','hp_known','run_id','interval_id',
          'selected_hp','selection_known','selection_mixed','selected_matches_running','raw_original_event_count']
TRANSITION_COLS=['case_id','from_run_id','to_run_id','from_hp','to_hp','gap_s','unknown_in_gap_s']


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def state_intervals(events,code,duration,allowed):
    """Complete [0,duration) interval partition, including initial/invalid unknown."""
    part=events[(events.event_code==code)&(events.time_s>=0)&(events.time_s<duration)]
    part=part.sort_values('time_s',kind='stable').drop_duplicates('time_s',keep='last')
    times=part.time_s.to_numpy(float)
    values=part.numeric_value.to_numpy(float)
    values=np.where(np.isin(values,allowed),values,np.nan)
    starts=np.unique(np.r_[0.,times])
    indices=np.searchsorted(times,starts,side='right')-1
    result=np.full(len(starts),np.nan)
    good=indices>=0
    result[good]=values[indices[good]]
    ends=np.r_[starts[1:],duration]
    intervals=[]
    for start,end,value in zip(starts,ends,result):
        if end<=start:continue
        same=bool(intervals and ((np.isnan(value) and np.isnan(intervals[-1][2])) or value==intervals[-1][2]))
        if same:intervals[-1][1]=float(end)
        else:intervals.append([float(start),float(end),float(value)])
    array=np.asarray(intervals,float).reshape(-1,3)
    assert len(array) and np.isclose((array[:,1]-array[:,0]).sum(),duration)
    return array


def lookup(intervals,times):
    times=np.asarray(times,float)
    indices=np.searchsorted(intervals[:,0],times,side='right')-1
    result=np.full(len(times),np.nan)
    valid=(indices>=0)&(times<intervals[-1,1])
    result[valid]=intervals[indices[valid],2]
    return result,indices


def overlap(intervals,start,end,want_unknown=False):
    widths=np.maximum(0,np.minimum(intervals[:,1],end)-np.maximum(intervals[:,0],start))
    if want_unknown:return float(widths[~np.isfinite(intervals[:,2])].sum())
    return widths


def extract_case(record):
    source=Path(record['source_path']); cid=record['case_id'];duration=float(record['duration_s'])
    actual=sha(source)
    if actual!=record['source_sha256']:raise ValueError(f'Source hash mismatch: {cid}')
    meta=json.loads((source.parent/'meta.json').read_text())
    assert np.isclose(meta['duration_s'],duration),cid
    assert pd.Timestamp(meta['start'])==pd.Timestamp(record['start']),cid
    assert pd.Timestamp(meta['end'])==pd.Timestamp(record['end']),cid
    pieces=[]
    for chunk in pd.read_csv(source,usecols=['time_s','event_code','event_value'],
                             dtype={'event_code':str},chunksize=20000,low_memory=False):
        chunk.event_code=chunk.event_code.str.strip()
        part=chunk[chunk.event_code.isin(READ_CODES)].copy()
        if not part.empty:
            part['time_s']=pd.to_numeric(part.time_s,errors='coerce')
            part['numeric_value']=pd.to_numeric(part.event_value,errors='coerce')
            pieces.append(part[['time_s','event_code','numeric_value']])
    events=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=['time_s','event_code','numeric_value'])
    bad_time=int((~np.isfinite(events.time_s)).sum())
    events=events[np.isfinite(events.time_s)].sort_values('time_s',kind='stable')
    marker=events[(events.event_code=='90012')&events.numeric_value.isin([0,1])]
    first=marker.loc[marker.numeric_value==1,'time_s'].min()
    mv=marker.numeric_value.to_numpy(float)
    final=marker.loc[(mv==0)&(np.r_[np.nan,mv[:-1]]==1)&(marker.time_s>first),'time_s'].max()
    assert np.isclose(first,0) and np.isclose(final,duration),cid
    events=events[(events.time_s>=0)&(events.time_s<duration)].copy()
    hp=state_intervals(events,'20039',duration,[0,1,2,3])
    selection=state_intervals(events,'20011',duration,[0,1,2,3])
    scaler=state_intervals(events,'20774',duration,[0,1])
    intervals=[];runs=[];run_for_interval={};run_id=0
    for i,(start,end,value) in enumerate(hp):
        intervals.append(dict(case_id=cid,interval_id=i,start_s=start,end_s=end,hp=value,
                              known=bool(np.isfinite(value)),duration_s=end-start))
        if np.isfinite(value) and value>0:
            run_id+=1;run_for_interval[i]=run_id
            runs.append(dict(case_id=cid,run_id=run_id,interval_id=i,hp=int(value),
                             start_s=start,end_s=end,active_s=end-start))
    transitions=[]
    for before,after in zip(runs[:-1],runs[1:]):
        gap_start,gap_end=before['end_s'],after['start_s']
        transitions.append(dict(case_id=cid,from_run_id=before['run_id'],to_run_id=after['run_id'],
            from_hp=before['hp'],to_hp=after['hp'],gap_s=gap_end-gap_start,
            unknown_in_gap_s=overlap(hp,gap_start,gap_end,True)))
    observations=[]
    numeric_raw=events[events.event_code.isin(NUMERIC)].copy()
    invalid_numeric=int((~np.isfinite(numeric_raw.numeric_value)).sum())
    numeric_raw=numeric_raw[np.isfinite(numeric_raw.numeric_value)].copy()
    if not numeric_raw.empty:
        h,ix=lookup(hp,numeric_raw.time_s.to_numpy())
        sel,_=lookup(selection,numeric_raw.time_s.to_numpy())
        numeric_raw['hp']=h;numeric_raw['interval_id']=ix
        numeric_raw['selected_hp']=sel
        numeric_raw['second_bin']=np.floor(numeric_raw.time_s).astype(int)
        for (code,interval_id,second),group in numeric_raw.groupby(['event_code','interval_id','second_bin'],sort=True):
            value=float(np.median(group.numeric_value))
            hvalue=float(group.hp.iloc[0])
            states=group.selected_hp.dropna().unique()
            selection_mixed=len(states)>1 or (group.selected_hp.isna().any() and len(states)>0)
            selected=float(states[0]) if len(states)==1 and not selection_mixed else np.nan
            known=bool(np.isfinite(hvalue))
            observations.append(dict(case_id=cid,code=code,time_s=float(np.median(group.time_s)),
                second_bin=int(second),value=value,hp=hvalue,hp_known=known,
                run_id=run_for_interval.get(int(interval_id),np.nan),interval_id=int(interval_id),
                selected_hp=selected,selection_known=bool(np.isfinite(selected)),
                selection_mixed=bool(selection_mixed),
                selected_matches_running=bool(selected==hvalue) if np.isfinite(selected) and known and hvalue>0 else np.nan,
                raw_original_event_count=len(group)))
    # Independent scaler channel and HP3 must agree only where both are known.
    breaks=np.unique(np.r_[hp[:,0],hp[:,1],scaler[:,0],scaler[:,1]])
    widths=np.diff(breaks);mid=(breaks[:-1]+breaks[1:])/2
    hv,_=lookup(hp,mid);sv,_=lookup(scaler,mid)
    comparable=np.isfinite(hv)&np.isfinite(sv)
    agree=comparable&((hv==3)==(sv==1))
    # Selection is expected to match only while a positive HP is running.
    breaks2=np.unique(np.r_[hp[:,0],hp[:,1],selection[:,0],selection[:,1]])
    w2=np.diff(breaks2);m2=(breaks2[:-1]+breaks2[1:])/2
    hv2,_=lookup(hp,m2);select,_=lookup(selection,m2)
    compare2=np.isfinite(hv2)&np.isfinite(select)&(hv2>0)
    agree2=compare2&(hv2==select)
    target=events[events.event_code.isin(TARGETS)]
    widths_hp=hp[:,1]-hp[:,0]
    audit=dict(case_id=cid,duration_s=duration,hp_known_s=float(widths_hp[np.isfinite(hp[:,2])].sum()),
        unknown_s=float(widths_hp[~np.isfinite(hp[:,2])].sum()),run_count=len(runs),
        n_target_events=len(target),n_target_codes=target.event_code.nunique(),source_sha256=actual,
        signal_status=record['signal_status'],boundary_matches=True,
        hp3_scaler_comparable_s=float(widths[comparable].sum()),
        hp3_scaler_agree_s=float(widths[agree].sum()),conflict_s=float(widths[comparable&~agree].sum()),
        selection_comparable_running_s=float(w2[compare2].sum()),selection_agree_running_s=float(w2[agree2].sum()),
        selection_conflict_running_s=float(w2[compare2&~agree2].sum()),
        invalid_timestamp_rows=bad_time,invalid_numeric_rows=invalid_numeric,
        observed_numeric_rows=len(numeric_raw),aggregated_numeric_rows=len(observations))
    for value in [0,1,2,3]:audit[f'hp{value}_s']=float(widths_hp[hp[:,2]==value].sum())
    for code in TARGETS:audit[f'E{code}_n_events']=int((target.event_code==code).sum())
    assert np.isclose(audit['hp_known_s']+audit['unknown_s'],duration),cid
    assert np.isclose(sum(audit[f'hp{value}_s'] for value in [0,1,2,3]),audit['hp_known_s']),cid
    assert sum(o['raw_original_event_count'] for o in observations)==len(numeric_raw),cid
    return intervals,runs,observations,transitions,audit


def synthetic_validation():
    e=pd.DataFrame({'time_s':[2,3,4,6,8],'event_code':['20039']*5,'numeric_value':[2,2,0,3,99]})
    h=state_intervals(e,'20039',10,[0,1,2,3])
    assert np.allclose(h[:,0],[0,2,4,6,8])
    assert np.allclose(h[:,1],[2,4,6,8,10])
    assert np.isnan(h[0,2]) and np.isnan(h[-1,2])
    vals,ix=lookup(h,[1,2,3.9,4,6,9])
    assert np.isnan(vals[0]) and np.isnan(vals[-1])
    assert np.allclose(vals[1:-1],[2,2,0,3])
    assert np.isclose(overlap(h,0,10,True),4)
    # Same integer second straddling a state change must remain two groups.
    obs=pd.DataFrame({'time_s':[2.1,2.2,2.8],'value':[10,30,999]})
    e2=pd.DataFrame({'time_s':[0,2.5],'event_code':['20039']*2,'numeric_value':[2,3]})
    hp=state_intervals(e2,'20039',5,[0,1,2,3]);state,index=lookup(hp,obs.time_s)
    obs['interval']=index;obs['second']=np.floor(obs.time_s).astype(int)
    medians=obs.groupby(['interval','second']).value.median()
    assert len(medians)==2 and np.allclose(medians,[20,999])
    return dict(initial_unknown_preserved=True,invalid_state_unknown=True,same_value_compressed=True,
                exact_duration_partition=True,asof_boundaries_correct=True,
                state_boundary_bin_split=True,observed_only_numeric_aggregation=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    (out/'process').mkdir(parents=True,exist_ok=True)
    def log(event,**info):
        row=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),event=event,**info)
        with (out/'process/extraction_log.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    checks=synthetic_validation();log('synthetic_validation',checks=checks)
    root=Path(__file__).resolve().parents[3]
    source_manifest=root/'Output/chair_numeric_multiview_300_v1_20261104/case_manifest.csv'
    manifest=pd.read_csv(source_manifest)
    assert len(manifest)==300 and not manifest.case_id.duplicated().any()
    records=manifest.to_dict('records')
    if args.smoke:
        records=[records[0],records[1],next(r for r in records if r['signal_status']=='NO_TARGET_SIGNAL')]
        for r in records:
            ans=extract_case(r);log('smoke_case',case_id=r['case_id'],audit=ans[-1])
        return
    for filename in ['case_manifest.csv','hp_intervals.csv','hp_runs.csv','observations.csv','hp_status_audit.csv','transition_audit.csv','extraction_metadata.json']:
        if (out/filename).exists():parser.error(f'Existing extraction output must be preserved: {out/filename}')
    log('preflight',requested_cases=300,source_manifest=str(source_manifest),workers=args.workers,
        numeric_policy='actual timestamp asof to HP state then code x HP interval x 1-second median; no ffill')
    started=time.time();results={};failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future={pool.submit(extract_case,r):r for r in records}
        for i,f in enumerate(as_completed(future),1):
            record=future[f]
            try:results[record['case_id']]=f.result()
            except Exception as exc:
                failures.append(dict(case_id=record['case_id'],error=repr(exc)))
                log('case_failed',**failures[-1])
            if i%25==0:log('progress',completed=i,failures=len(failures))
    if failures:
        (out/'process/extraction_failures.json').write_text(json.dumps(failures,indent=2))
        raise RuntimeError(f'{len(failures)} cases failed; no partial cohort exported')
    intervals=[];runs=[];observations=[];transitions=[];audits=[]
    for r in records:
        a,b,c,d,e=results[r['case_id']];intervals.extend(a);runs.extend(b);observations.extend(c);transitions.extend(d);audits.append(e)
    manifest.to_csv(out/'case_manifest.csv',index=False)
    frames={'hp_intervals.csv':pd.DataFrame(intervals,columns=INTERVAL_COLS),
        'hp_runs.csv':pd.DataFrame(runs,columns=RUN_COLS),
        'observations.csv':pd.DataFrame(observations,columns=OBS_COLS),
        'transition_audit.csv':pd.DataFrame(transitions,columns=TRANSITION_COLS),
        'hp_status_audit.csv':pd.DataFrame(audits)}
    for name,frame in frames.items():frame.to_csv(out/name,index=False)
    assert not frames['hp_intervals.csv'].duplicated(['case_id','interval_id']).any()
    assert not frames['hp_runs.csv'].duplicated(['case_id','run_id']).any()
    assert not frames['observations.csv'].duplicated(['case_id','code','interval_id','second_bin']).any()
    audit=frames['hp_status_audit.csv']
    metadata=dict(version='hp_sequence_v3',requested_cases=300,extracted_cases=len(audit),
        cases_with_hp_runs=int((audit.run_count>0).sum()),cases_no_target_signal=int((audit.n_target_events==0).sum()),
        case_ids_no_hp_runs=audit.loc[audit.run_count==0,'case_id'].tolist(),
        n_hp_intervals=len(intervals),n_hp_runs=len(runs),n_observations=len(observations),n_transitions=len(transitions),
        source_manifest=str(source_manifest),source_manifest_sha256=sha(source_manifest),
        script_source=str(Path(__file__).resolve()),script_sha256=sha(__file__),
        read_columns=['time_s','event_code','event_value'],chunk_rows=20000,workers=args.workers,
        full_wide_tables_loaded=False,missing_numeric_imputed=False,boundary='first90012ON-to-lastOFF; features [0,duration)',
        hp_mapping={'0':'OFF','1':'HP1','2':'HP2','3':'HP3 scaler'},meaning_source='user_confirmed_conversation',
        numeric_codes=NUMERIC,
        state_policy='last observed valid same-code state until next record; initial and invalid states unknown',
        same_timestamp_policy='last recorded state value per same-code timestamp, then asof inclusive',
        numeric_policy='gated by actual timestamp before code x HP interval x integer-second median; raw numeric values retained',
        selection_policy='single unique selected HP in aggregate bin, otherwise NaN+selection_mixed',
        hp3_scaler_comparable_s=float(audit.hp3_scaler_comparable_s.sum()),
        hp3_scaler_conflict_s=float(audit.conflict_s.sum()),
        elapsed_seconds=time.time()-started,python=sys.version,synthetic_validation=checks)
    (out/'extraction_metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    log('complete',**metadata)


if __name__=='__main__':main()
