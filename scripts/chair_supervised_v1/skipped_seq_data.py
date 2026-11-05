"""Read-only export of six annotation-skipped sequences as unknown-only cases.

Never adds training/evaluation targets. Missing lamp starts or preprocessed
metadata do not establish empty/corrupt logs; original full DAT is inspected.
All six requested seq numbers stay in the ledger, including unavailable data.
"""
from pathlib import Path
import argparse,datetime,json,hashlib,shutil
import numpy as np
import pandas as pd
from features import READ_CODES,extract_features
from extract import ROOT
from window_data import local_events,starts,activity_status

REQUESTED=[1449,1456,1459,1463,1473,1474]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read_allow_empty(path):
    pieces=[];lo=None;hi=None;total=0;invalid=0;counts={}
    for q in pd.read_csv(path,skiprows=1,usecols=['occur_date','occur_time','event_code','event_value'],dtype=str,chunksize=20000):
        total+=len(q)
        t=pd.to_datetime(q.occur_date+' '+q.occur_time,format='mixed',errors='coerce')
        valid=t.dropna();invalid+=int(t.isna().sum())
        if len(valid):lo=valid.min() if lo is None else min(lo,valid.min());hi=valid.max() if hi is None else max(hi,valid.max())
        codes=q.event_code.str.strip()
        for c,n in codes.value_counts().items():counts[c]=counts.get(c,0)+int(n)
        keep=codes.isin(READ_CODES)&t.notna()
        if keep.any():pieces.append(pd.DataFrame({'timestamp':t[keep],'code':codes[keep],
            'value':pd.to_numeric(q.loc[keep,'event_value'],errors='coerce')}))
    if lo is None:raise ValueError('No valid timestamps in full raw source')
    if pieces:
        e=pd.concat(pieces,ignore_index=True);e['time_s']=(e.timestamp-lo).dt.total_seconds()
        e=e.sort_values('time_s',kind='stable')[['time_s','code','value']].reset_index(drop=True)
    else:e=pd.DataFrame({'time_s':pd.Series(dtype=float),'code':pd.Series(dtype=str),'value':pd.Series(dtype=float)})
    return e,lo,hi,dict(raw_rows=total,invalid_timestamp_rows=invalid,raw_code_counts=counts,selected_rows=len(e))

def features_for_window(events,start,end,basic_columns,extended_columns):
    if len(events):
        x=extract_features(local_events(events,start,end),start,end,True,3)
    else:
        # No signal evidence is manufactured. Context availability is unknown
        # for this target-code representation, and all predicted labels abstain.
        x=extract_features(events,start,end,False,3)
        for c in extended_columns:
            if c!='id' and c not in x:x[c]=0. if c.endswith('_available_s') else np.nan
    b={c:x[c] for c in basic_columns if c!='id'}
    return b,{c:x[c] for c in extended_columns if c!='id'}

def export(source):
    source=source.resolve();out=source/'windows30/skipped_seq'
    out.mkdir(parents=True,exist_ok=True);(out/'process').mkdir(exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    if (out/'cases_manifest.csv').exists():raise FileExistsError('Preserve prior unknown-only export')
    def log(event,**kw):
        r=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,**kw)
        with (out/'process/export_log.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        print(json.dumps(r,ensure_ascii=False),flush=True)
    summary_path=ROOT/'Output/chair_model_A_results/case_summary.csv'
    original_annotation=source/'annotations_aligned.csv';annotations=pd.read_csv(original_annotation)
    assert not set(REQUESTED)&set(annotations.seq)
    summary=pd.read_csv(summary_path)
    basic_columns=pd.read_csv(source/'features_basic.csv',nrows=0).columns.tolist()
    extended_columns=pd.read_csv(source/'features_extended.csv',nrows=0).columns.tolist()
    source_before={str(p):sha(p) for p in [summary_path,original_annotation,source/'features_basic.csv',source/'features_extended.csv']}
    rows=[];basic=[];extended=[];cases=[];integrity=[];errors=[];equivalence=[]
    log('preflight',requested_cases=6,requested_seqs=REQUESTED,
        label_status='UNANNOTATED_ONLY',new_training_labels=0,window_s=30.,step_s=15.)
    for seq in REQUESTED:
        matches=summary[summary.seq.eq(seq)]
        case=dict(seq=seq,split='UNANNOTATED_ONLY',originalsplit='UNANNOTATED_ONLY',requested=True,
            confirmed=False,treatment_target_valid=False,instrument_target_valid=False)
        try:
            if len(matches)!=1:raise ValueError(f'Ambiguous/missing metadata for seq{seq}')
            row=matches.iloc[0]
            raw=ROOT/'chair_export'/row.date_folder/row.chair_folder/f'Chair_log_seq_{seq}.dat'
            meta=ROOT/'Output/chair_model_A_results/cases'/row.date_folder/row.chair_folder/f'Chair_log_seq_{seq}'/'meta.json'
            before=sha(raw);mh=sha(meta) if meta.exists() else None
            e,lo,hi,audit=read_allow_empty(raw);duration=float((hi-lo).total_seconds())
            cid=f'{row.date_folder}__{row.chair_folder}__Chair_log_seq_{seq}'
            cache=out/'cases'/f'{seq}.csv.gz';e.to_csv(cache,index=False,compression='gzip')
            ss=starts(0.,duration,30.,15.)
            # Test exact pruned representation on actual full source if signals exist.
            if len(e) and duration>0:
                a=min(max(0.,duration/2-15.),max(0.,duration-30.));b=min(duration,a+30.)
                if b>a:
                    full=extract_features(e,a,b,True,3);small=extract_features(local_events(e,a,b),a,b,True,3)
                    np.testing.assert_allclose(list(full.values()),list(small.values()),equal_nan=True,rtol=1e-12,atol=1e-12)
                    equivalence.append(dict(seq=seq,full_vs_pruned_features_equal=True))
            positive_hp_events=int((e.code.eq('20039')&e.value.isin([1,2,3])).sum())
            numeric_events=int((e.code.isin(['20742','20743','20108','10014','10015'])&np.isfinite(e.value)).sum())
            status='NO_TARGET_SIGNAL' if not len(e) else 'NO_ELIGIBLE_30S_WINDOWS' if not ss else 'NO_POSITIVE_HP_EVENT_OBSERVED' if positive_hp_events==0 else 'OBSERVED_TARGET_SIGNALS'
            for index,start in enumerate(ss):
                end=start+30.;wid=f'{seq}__unannotated_only_w{index:04d}'
                bf,xf=features_for_window(e,start,end,basic_columns,extended_columns)
                n_numeric=sum(bf[f'num_{c}_n_obs'] for c in ['20742','20743','20108','10014','10015'])
                rows.append(dict(id=wid,seq=seq,case_id=cid,split='UNANNOTATED_ONLY',originalsplit='UNANNOTATED_ONLY',
                    start_s=start,end_s=end,duration_sec=30.,time_origin_datetime=str(lo),
                    start_datetime=(lo+pd.Timedelta(seconds=start)).isoformat(),end_datetime=(lo+pd.Timedelta(seconds=end)).isoformat(),
                    confirmed=False,treatment_target_valid=False,instrument_target_valid=False,
                    treatment_normalized='',instrument_normalized='',instrument_labels='[]',
                    label_provenance='UNANNOTATED_ONLY_CASE_UNKNOWN',annotation_overlap_s=0.,
                    activity_status=activity_status(bf),case_signal_status=status,
                    target_event_count=bf['quality_target_event_count'],numeric_event_count=n_numeric,
                    hp_known_fraction=bf['state_20039_known_fraction'],cache_path=str(cache),source_path=str(raw)))
                basic.append({'id':wid,**bf});extended.append({'id':wid,**xf})
            after=sha(raw);assert before==after
            if mh is not None:assert sha(meta)==mh
            covered=30.+15.*(len(ss)-1) if len(ss) else 0.
            case.update(case_id=cid,date=row.date_folder,chair=row.chair_folder,source_path=str(raw),cache_path=str(cache),
                source_sha256=before,metadata_path=str(meta),metadata_exists=meta.exists(),metadata_sha256=mh,
                preprocessed_status=row.status,raw_start_datetime=str(lo),raw_end_datetime=str(hi),raw_duration_s=duration,
                selected_rows=len(e),positive_hp_events=positive_hp_events,numeric_events=numeric_events,
                planned_windows=len(ss),actual_windows=len(ss),window_union_duration_s=covered,
                uncovered_tail_s=max(0.,duration-covered),signal_status=status,extraction_status='SUCCESS',
                raw_rows=audit['raw_rows'],invalid_timestamp_rows=audit['invalid_timestamp_rows'])
            integrity.append(dict(seq=seq,source_path=str(raw),source_sha256=before,source_sha256_after=after,
                metadata_path=str(meta),metadata_exists=meta.exists(),metadata_sha256=mh,
                missing_metadata_reason='' if meta.exists() else 'No preprocessed meta; fullraw retained independently of lamp preprocessing',
                source_unchanged=True))
            log('case_complete',seq=seq,signal_status=status,raw_rows=audit['raw_rows'],target_rows=len(e),
                positive_hp_events=positive_hp_events,numeric_events=numeric_events,windows=len(ss),source_unchanged=True)
        except Exception as exc:
            # An actual failure remains explicit and is never replaced by another seq.
            error=dict(seq=seq,error=repr(exc));errors.append(error)
            case.update(extraction_status='FAILED',signal_status='UNKNOWN_EXTRACTION_FAILED',actual_windows=0,error=repr(exc))
            log('case_failed',**error)
        cases.append(case)
    m=pd.DataFrame(rows);b=pd.DataFrame(basic,columns=basic_columns);x=pd.DataFrame(extended,columns=extended_columns)
    if len(m):
        assert m.id.is_unique and m.id.tolist()==b.id.tolist()==x.id.tolist()
        assert not m.confirmed.any() and not m.treatment_target_valid.any() and not m.instrument_target_valid.any()
    assert not np.isinf(b.drop(columns='id').to_numpy(float)).any()
    assert not np.isinf(x.drop(columns='id').to_numpy(float)).any()
    m.to_csv(out/'unannotated_candidates.csv',index=False);b.to_csv(out/'unannotated_features_basic.csv',index=False);x.to_csv(out/'unannotated_features_extended.csv',index=False)
    ledger=pd.DataFrame(cases);assert len(ledger)==6 and ledger.seq.tolist()==REQUESTED
    ledger.to_csv(out/'cases_manifest.csv',index=False);ledger.to_csv(out/'case_ledger.csv',index=False)
    pd.DataFrame(integrity).to_csv(out/'source_integrity.csv',index=False)
    pd.DataFrame(equivalence).to_csv(out/'process/pruning_equivalence.csv',index=False)
    for p,h in source_before.items():assert sha(p)==h
    meta=dict(version='six-skipped-sequences-unknown-only-v1',requested_cases=6,actual_successful_cases=int(ledger.extraction_status.eq('SUCCESS').sum()),
        failed_cases=errors,requested_seqs=REQUESTED,exported_candidate_windows=len(m),
        candidate_cases=int(m.seq.nunique()) if len(m) else 0,new_training_or_evaluation_labels=0,
        window_s=30.,step_s=15.,window_activity_counts=m.activity_status.value_counts().to_dict() if len(m) else {},
        case_signal_status_counts=ledger.signal_status.value_counts().to_dict(),all_requested_cases_in_ledger=True,
        original_35_case_data_unchanged=True,source_hashes=source_before,
        split_policy='All6 UNANNOTATED_ONLY: never included in supervised training/calibration/evaluation',
        label_policy='Unknown only. No lamp start, absent annotation, or no HP evidence is not a negative treatment label.',
        prediction_policy='Only apply validated models provisionally; unavailable/unsupported cases remain UNKNOWN',
        no_silent_case_replacement=True)
    (out/'process/export_summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    (out/'code_snapshot').mkdir(exist_ok=True);shutil.copy2(Path(__file__),out/'code_snapshot'/Path(__file__).name)
    (out/'export_fingerprint.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in [out/'unannotated_candidates.csv',out/'unannotated_features_basic.csv',out/'unannotated_features_extended.csv',out/'cases_manifest.csv',out/'source_integrity.csv']},indent=2))
    log('complete',requested_cases=6,actual_successful_cases=meta['actual_successful_cases'],failed_cases=len(errors),candidate_windows=len(m),all_sources_preserved=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);export(ap.parse_args().source)
