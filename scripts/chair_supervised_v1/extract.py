"""Read-only full-raw extraction of the fixed annotated cohort, no fitting."""
from pathlib import Path
import argparse,hashlib,json,datetime,unicodedata,shutil
import numpy as np
import pandas as pd
from features import READ_CODES,extract_features,synthetic_checks

ROOT=Path(__file__).resolve().parents[3]
FINAL_SEQS={1469,1470,1471,1472,1475,1476,1477,1478}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def normalize(text):
    if pd.isna(text):return ''
    return unicodedata.normalize('NFKC',str(text)).strip()

def instrument_labels(text):
    value=normalize(text)
    if not value:return []
    parts=value.replace(',','、').replace('，','、').split('、')
    return sorted(set('5倍速' if p.strip()=='5倍速コントラ' else p.strip() for p in parts if p.strip()))

def load_case_events(path):
    """Load a narrow exported full-raw cache, preserving NaN numeric values."""
    return pd.read_csv(path,dtype={'code':str})

def read_raw(path):
    chunks=[];lo=None;hi=None;invalid_t=0;total=0;codecounts={}
    for c in pd.read_csv(path,skiprows=1,usecols=['occur_date','occur_time','event_code','event_value'],dtype=str,chunksize=20000):
        total+=len(c);t=pd.to_datetime(c.occur_date+' '+c.occur_time,format='mixed',errors='coerce')
        invalid_t+=int(t.isna().sum());valid=t.dropna()
        if len(valid):lo=valid.min() if lo is None else min(lo,valid.min());hi=valid.max() if hi is None else max(hi,valid.max())
        c['code']=c.event_code.str.strip()
        for code,n in c.code.value_counts().items():codecounts[code]=codecounts.get(code,0)+int(n)
        keep=c.code.isin(READ_CODES)&t.notna()
        if keep.any():chunks.append(pd.DataFrame({'timestamp':t[keep],'code':c.loc[keep,'code'],
            'value':pd.to_numeric(c.loc[keep,'event_value'],errors='coerce')}))
    if lo is None or not chunks:raise ValueError(f'No usable signal timestamps: {path}')
    e=pd.concat(chunks,ignore_index=True);e['time_s']=(e.timestamp-lo).dt.total_seconds()
    e=e.sort_values('time_s',kind='stable')[['time_s','code','value']].reset_index(drop=True)
    return e,lo,hi,dict(raw_rows=total,selected_rows=len(e),invalid_timestamp_rows=invalid_t,raw_code_counts=codecounts)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--annotation',type=Path,default=ROOT/'ChairLog_annotations1438_1479.csv')
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);(out/'process').mkdir(exist_ok=True)
    if (out/'annotations_aligned.csv').exists():raise FileExistsError('Preserve extraction, use new output')
    (out/'cases').mkdir(exist_ok=True)
    def log(event,**kw):
        r=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,**kw)
        with (out/'process/extraction_log.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        print(json.dumps(r,ensure_ascii=False),flush=True)
    checks=synthetic_checks();annotation_sha=sha(a.annotation)
    ann=pd.read_csv(a.annotation,encoding='utf-8-sig');summary=pd.read_csv(ROOT/'Output/chair_model_A_results/case_summary.csv')
    assert len(ann)==110 and ann.seq.nunique()==35 and ann.seq.between(1438,1478).all()
    assert ann.id.is_unique and ann.id.notna().all()
    s=pd.to_datetime(ann.start_datetime,format='mixed');e=pd.to_datetime(ann.end_datetime,format='mixed')
    assert (e>s).all() and np.allclose((e-s).dt.total_seconds(),ann.duration_sec,atol=.002)
    ann['treatment_normalized']=ann.treatment.map(normalize)
    ann['instrument_labels']=ann.instrument.map(lambda x:json.dumps(instrument_labels(x),ensure_ascii=False))
    ann['instrument_normalized']=ann.instrument.map(lambda x:'、'.join(instrument_labels(x)))
    ann['confirmed']=ann.status.eq('確認済み')
    ann['treatment_target_valid']=ann.confirmed&ann.treatment_normalized.ne('')
    ann['instrument_target_valid']=ann.confirmed&ann.instrument_normalized.ne('')
    ann['split']=np.where(ann.seq.isin(FINAL_SEQS),'final','development')
    log('preflight',requested_cases=35,requested_intervals=110,confirmed=int(ann.confirmed.sum()),
        final_cases=8,development_cases=27,source_policy='Full original DAT, annotation timestamps; old lamp boundaries unchanged',synthetic_checks=checks)
    basic=[];extended=[];case_records=[];integrity=[];overlaps=[];outside=[];availability=[]
    for seq,g in ann.groupby('seq',sort=True):
        records=summary[summary.seq.eq(seq)]
        assert len(records)==1,f'Ambiguous/missing seq {seq}'
        r=records.iloc[0]
        raw=ROOT/'chair_export'/r.date_folder/r.chair_folder/f'Chair_log_seq_{seq}.dat'
        meta=ROOT/'Output/chair_model_A_results/cases'/r.date_folder/r.chair_folder/f'Chair_log_seq_{seq}'/'meta.json'
        before=sha(raw);meta_sha=sha(meta)
        known=g.source_sha256.dropna().unique()
        assert len(known)<=1
        if len(known):assert known[0]==before,f'Annotation source hash mismatch {seq}'
        events,lo,hi,audit=read_raw(raw)
        cid=f'{r.date_folder}__{r.chair_folder}__Chair_log_seq_{seq}'
        cache=out/'cases'/f'{seq}.csv.gz';events.to_csv(cache,index=False,compression='gzip')
        ann.loc[g.index,'case_id']=cid
        ann.loc[g.index,'time_origin_datetime']=str(lo)
        ann.loc[g.index,'start_s']=(pd.to_datetime(g.start_datetime,format='mixed')-lo).dt.total_seconds()
        ann.loc[g.index,'end_s']=(pd.to_datetime(g.end_datetime,format='mixed')-lo).dt.total_seconds()
        for row in g.itertuples():
            start=(pd.Timestamp(row.start_datetime)-lo).total_seconds();end=(pd.Timestamp(row.end_datetime)-lo).total_seconds()
            assert start>=0 and pd.Timestamp(row.end_datetime)<=hi, f'Annotation beyond full raw {seq}/{row.id}'
            b=extract_features(events,start,end,False,0);x=extract_features(events,start,end,True,3)
            basic.append({'id':row.id,**b});extended.append({'id':row.id,**x})
            outlamp=pd.Timestamp(row.start_datetime)<pd.Timestamp(r.start) or pd.Timestamp(row.end_datetime)>pd.Timestamp(r.end)
            ann.loc[ann.id.eq(row.id),'outside_existing_lamp_boundary']=outlamp
            if outlamp:outside.append(dict(seq=int(seq),id=row.id,start_datetime=row.start_datetime,end_datetime=row.end_datetime))
            availability.append(dict(id=row.id,seq=int(seq),numeric_event_count=sum(b[f'num_{c}_n_obs'] for c in ['20742','20743','20108','10014','10015']),
                hp_known_fraction=b['state_20039_known_fraction'],target_event_count=b['quality_target_event_count']))
        for i,row in enumerate(g.itertuples()):
            for other in list(g.itertuples())[i+1:]:
                width=(min(pd.Timestamp(row.end_datetime),pd.Timestamp(other.end_datetime))-max(pd.Timestamp(row.start_datetime),pd.Timestamp(other.start_datetime))).total_seconds()
                if width>0:overlaps.append(dict(seq=int(seq),id=row.id,other_id=other.id,overlap_s=width))
        after=sha(raw);assert before==after and sha(meta)==meta_sha
        integrity.append(dict(seq=int(seq),case_id=cid,source_path=str(raw),source_sha256=before,source_sha256_after=after,
            metadata_path=str(meta),metadata_sha256=meta_sha,annotation_source_hash_present=bool(len(known)),
            annotation_source_hash_matches=bool(known[0]==before) if len(known) else np.nan,source_unchanged=True))
        case_records.append(dict(seq=int(seq),case_id=cid,date=r.date_folder,chair=r.chair_folder,source_path=str(raw),
            cache_path=str(cache),raw_start_datetime=str(lo),raw_end_datetime=str(hi),raw_duration_s=(hi-lo).total_seconds(),
            old_lamp_start=r.start,old_lamp_end=r.end,n_annotations=len(g),split='final' if seq in FINAL_SEQS else 'development',
            source_sha256=before,metadata_sha256=meta_sha,raw_rows=audit['raw_rows'],selected_rows=audit['selected_rows'],
            invalid_timestamp_rows=audit['invalid_timestamp_rows']))
        log('case_complete',seq=int(seq),annotations=len(g),selected_rows=len(events),source_unchanged=True)
    b=pd.DataFrame(basic).set_index('id').loc[ann.id].reset_index();x=pd.DataFrame(extended).set_index('id').loc[ann.id].reset_index()
    assert len(b)==len(x)==110 and b.id.tolist()==x.id.tolist()==ann.id.tolist()
    assert not np.isinf(b.drop(columns='id').to_numpy(float)).any()
    assert not np.isinf(x.drop(columns='id').to_numpy(float)).any()
    assert sha(a.annotation)==annotation_sha
    ann.to_csv(out/'annotations_aligned.csv',index=False);b.to_csv(out/'features_basic.csv',index=False);x.to_csv(out/'features_extended.csv',index=False)
    pd.DataFrame(case_records).to_csv(out/'cases_manifest.csv',index=False);pd.DataFrame(integrity).to_csv(out/'source_integrity.csv',index=False)
    pd.DataFrame(overlaps,columns=['seq','id','other_id','overlap_s']).to_csv(out/'annotation_overlaps.csv',index=False)
    pd.DataFrame(outside).to_csv(out/'outside_old_lamp_intervals.csv',index=False)
    pd.DataFrame(availability).to_csv(out/'interval_signal_support.csv',index=False)
    pd.DataFrame({'seq':range(1438,1479),'has_annotation':[s in set(ann.seq) for s in range(1438,1479)]}).to_csv(out/'seq_coverage.csv',index=False)
    meta=dict(version='chair-supervised-v1-fullraw',requested_cases=35,actual_cases=35,requested_intervals=110,actual_intervals=110,
        confirmed_intervals=int(ann.confirmed.sum()),unconfirmed_ids=ann.loc[~ann.confirmed,'id'].tolist(),
        instrument_missing_ids=ann.loc[ann.instrument_normalized.eq(''),'id'].tolist(),
        missing_seq=sorted(set(range(1438,1479))-set(ann.seq)),outside_old_lamp_intervals=len(outside),
        overlapping_annotation_pairs=len(overlaps),all_full_raw_intervals_covered=True,
        basic_features=b.shape[1]-1,extended_features=x.shape[1]-1,
        annotation_path=str(a.annotation.resolve()),annotation_sha256=annotation_sha,annotation_unchanged=True,
        raw_source_sha_verified_where_recorded=True,all_sources_unchanged=True,
        final_case_seqs=sorted(FINAL_SEQS),synthetic_checks=checks,
        state_policy='Last same-code state persists, initial/invalid unknown; all context uses actual raw events only',
        numeric_policy='Observed finite raw numeric event summaries; no filling/interpolation; HP-gated values by exact state lookup',
        temporal_profiles='3 within-interval bins; each reports actual observations',
        context_policy='Offline full-raw +/-60sec with available duration, no annotation content/labels',
        normalization='NFKC/strip; instrument5倍速コントラ alias5倍速; compound instrument list; no clinical treatment merging',
        old_stage2_or_source_modified=False)
    (out/'process/preflight.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    (out/'code_snapshot').mkdir(exist_ok=True)
    for p in [Path(__file__),Path(__file__).parent/'features.py']:shutil.copy2(p,out/'code_snapshot'/p.name)
    (out/'extraction_fingerprint.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in [out/'annotations_aligned.csv',out/'features_basic.csv',out/'features_extended.csv',out/'cases_manifest.csv',out/'source_integrity.csv']},indent=2))
    log('complete',cases=35,intervals=110,basic_features=b.shape[1]-1,extended_features=x.shape[1]-1,
        outside_old_lamp_intervals=len(outside),all_sources_unchanged=True)

if __name__=='__main__':main()
