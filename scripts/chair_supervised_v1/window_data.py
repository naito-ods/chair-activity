"""Export weak within-annotation windows and genuinely unannotated candidates.

No fitting is performed. Window labels are inherited from interval annotations,
not independently verified point labels. Unannotated candidates have NO label;
they are never used as negatives. Original annotations/interval results remain
unchanged. All windows of one case retain the same development/final split.
"""
from pathlib import Path
import argparse,datetime,json,hashlib,shutil
import numpy as np
import pandas as pd
from features import STATE_CODES,extract_features
from extract import load_case_events

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def local_events(events,start,end,context=60.):
    """Exact feature-equivalent view, preserving state history and domain.

    Keep all query/context events, last prior event of every state code, and
    the two actual global endpoint rows. The endpoints preserve the domain
    used by context availability calculations; no artificial events inserted.
    """
    lo=start-context;hi=end+context
    keep=np.flatnonzero(events.time_s.ge(lo)&events.time_s.le(hi)).tolist()
    for code in STATE_CODES:
        ix=np.flatnonzero(events.code.eq(code)&events.time_s.lt(lo))
        if len(ix):keep.append(int(ix[-1]))
    keep.extend([0,len(events)-1])
    return events.iloc[np.unique(keep)]

def overlap_length(start,end,intervals):
    return float(sum(max(0.,min(end,b)-max(start,a)) for a,b in intervals))

def union_intervals(intervals):
    result=[]
    for a,b in sorted(np.asarray(intervals,float).tolist()):
        if not result or a>result[-1][1]:result.append([float(a),float(b)])
        else:result[-1][1]=max(float(b),result[-1][1])
    return result

def complement(intervals,start,end):
    result=[];cursor=float(start)
    for a,b in union_intervals(intervals):
        a=max(float(start),a);b=min(float(end),b)
        if a>cursor:result.append([cursor,a])
        cursor=max(cursor,b)
    if cursor<end:result.append([cursor,float(end)])
    return result

def starts(a,b,width,step):
    if b-a<width-1e-9:return []
    return [float(a+i*step) for i in range(int(np.floor((b-a-width)/step+1e-9))+1)]

def activity_status(features):
    x=np.array([features[f'hp{hp}_active_s'] for hp in [1,2,3]])
    if not np.isfinite(x).any():return 'HP_STATE_UNKNOWN'
    return 'ACTIVE_HP_OBSERVED' if np.nansum(x)>0 else 'NO_ACTIVE_HP_OBSERVED'

def synthetic_checks():
    intervals=[[3,5],[4,7],[10,11]]
    assert union_intervals(intervals)==[[3.,7.],[10.,11.]]
    assert union_intervals(np.asarray(intervals,float))==[[3.,7.],[10.,11.]]
    assert complement(intervals,0,12)==[[0.,3.],[7.,10.],[11.,12.]]
    assert starts(5,35,30,15)==[5.]
    assert starts(5,34,30,15)==[]
    assert overlap_length(7,10,intervals)==0
    assert overlap_length(6,10,[[3,7]])==1
    return dict(complement_includes_all_annotations=True,overlap_rejected=True,
                half_open_boundaries=True,fixed_window_bounds=True)

def export(source,width=30.,step=15.,trim=5.):
    source=source.resolve();out=source/f'windows{int(width)}'
    if (out/'annotations_aligned.csv').exists():raise FileExistsError('Preserve window export; use a new directory/version')
    out.mkdir(parents=True,exist_ok=True);(out/'process').mkdir(exist_ok=True)
    checks=synthetic_checks()
    def log(event,**kw):
        r=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,**kw)
        with (out/'process/export_log.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        print(json.dumps(r,ensure_ascii=False),flush=True)
    paths=[source/'annotations_aligned.csv',source/'cases_manifest.csv',source/'features_basic.csv',source/'features_extended.csv']
    before={str(p):sha(p) for p in paths}
    annotation=pd.read_csv(paths[0]);cases=pd.read_csv(paths[1])
    assert len(annotation)==110 and annotation.seq.nunique()==35 and cases.seq.is_unique
    basic_cols=pd.read_csv(source/'features_basic.csv',nrows=0).columns.tolist()
    extended_cols=pd.read_csv(source/'features_extended.csv',nrows=0).columns.tolist()
    metadata=[];basic=[];extended=[];candidates=[];cbasic=[];cext=[]
    audit=[];rejections=[];cacherecord=[];equivalence=[]
    log('preflight',source_cases=len(cases),source_annotations=len(annotation),width_s=width,
        step_s=step,trim_s=trim,label_provenance='WEAK_INTERVAL_INHERITED',unknown_candidates_are_negatives=False)
    for case in cases.itertuples():
        events=load_case_events(case.cache_path);cachehash=sha(case.cache_path)
        own=annotation[annotation.seq.eq(case.seq)].sort_values('start_s')
        full_intervals=own[['start_s','end_s']].to_numpy(float)
        origin=pd.Timestamp(case.raw_start_datetime)
        if not len(events):raise ValueError(f'Empty cache {case.seq}')
        # Independent full/pruned equivalence, using interval and gap queries.
        queries=[(float(own.iloc[0].start_s),float(own.iloc[0].end_s))]
        qs=max(0.,case.raw_duration_s/2-width/2);queries.append((qs,min(case.raw_duration_s,qs+width)))
        for a,b in queries:
            full=extract_features(events,a,b,True,3);small=extract_features(local_events(events,a,b),a,b,True,3)
            assert full.keys()==small.keys()
            np.testing.assert_allclose(list(full.values()),list(small.values()),rtol=1e-12,atol=1e-12,equal_nan=True)
            equivalence.append(dict(seq=int(case.seq),start_s=a,end_s=b,feature_equivalent=True))
        for row in own.to_dict('records'):
            a=float(row['start_s'])+trim;b=float(row['end_s'])-trim
            duration=b-a
            requests=[]
            if duration>=width-1e-9:requests=[(t,t+width,False) for t in starts(a,b,width,step)]
            elif duration>=10:requests=[(a,b,True)]
            else:
                rejections.append(dict(seq=int(case.seq),annotation_id=row['id'],window_id='',reason='ANNOTATION_TOO_SHORT_AFTER_BOUNDARY_TRIM',start_s=a,end_s=b))
            kept=0;ambiguous=0;no_active=0;unknown_state=0
            others=own.loc[own.id.ne(row['id']),['start_s','end_s']].to_numpy(float)
            for index,(start,end,short) in enumerate(requests):
                wid=f"{row['id']}__w{index:04d}"
                # Any overlap with another annotation makes inherited point labels ambiguous.
                overlap=overlap_length(start,end,others)
                if overlap>1e-8:
                    ambiguous+=1;rejections.append(dict(seq=int(case.seq),annotation_id=row['id'],window_id=wid,
                        reason='OVERLAPS_OTHER_ANNOTATION',start_s=start,end_s=end,other_annotation_overlap_s=overlap))
                    continue
                view=local_events(events,start,end)
                xf=extract_features(view,start,end,True,3);bf={k:xf[k] for k in basic_cols if k!='id'}
                status=activity_status(bf);no_active+=status=='NO_ACTIVE_HP_OBSERVED';unknown_state+=status=='HP_STATE_UNKNOWN'
                item=dict(row);item.update(id=wid,annotation_id=row['id'],annotation_duration_sec=row['duration_sec'],
                    start_s=start,end_s=end,duration_sec=end-start,start_datetime=(origin+pd.Timedelta(seconds=start)).isoformat(),
                    end_datetime=(origin+pd.Timedelta(seconds=end)).isoformat(),label_provenance='WEAK_INTERVAL_INHERITED',
                    short_variable_window=short,activity_status=status,other_annotation_overlap_s=0.,
                    window_width_requested_s=width,window_step_s=step,annotation_boundary_trim_s=trim)
                metadata.append(item);basic.append({'id':wid,**bf});extended.append({'id':wid,**xf});kept+=1
            audit.append(dict(seq=int(case.seq),case_id=case.case_id,annotation_id=row['id'],split=row['split'],confirmed=row['confirmed'],
                n_requested_windows=len(requests),n_exported_windows=kept,n_ambiguous_rejected=ambiguous,
                n_no_active_hp=no_active,n_hp_unknown=unknown_state,
                status='EXPORTED' if kept else 'NO_ELIGIBLE_WINDOWS',
                label_provenance='WEAK_INTERVAL_INHERITED',boundary_trim_s=trim))
        unknown_segments=complement(full_intervals,0.,float(case.raw_duration_s))
        for gap,(a,b) in enumerate(unknown_segments):
            for index,start in enumerate(starts(a,b,width,step)):
                end=start+width;wid=f'{int(case.seq)}__unannotated_g{gap:03d}_w{index:04d}'
                assert overlap_length(start,end,union_intervals(full_intervals))<=1e-8
                xf=extract_features(local_events(events,start,end),start,end,True,3);bf={k:xf[k] for k in basic_cols if k!='id'}
                candidates.append(dict(id=wid,seq=int(case.seq),case_id=case.case_id,split=case.split,start_s=start,end_s=end,
                    duration_sec=end-start,start_datetime=(origin+pd.Timedelta(seconds=start)).isoformat(),
                    end_datetime=(origin+pd.Timedelta(seconds=end)).isoformat(),
                    annotation_overlap_s=0.,label_provenance='UNANNOTATED_UNKNOWN',confirmed=False,
                    treatment_normalized='',instrument_labels='[]',instrument_normalized='',
                    treatment_target_valid=False,instrument_target_valid=False,activity_status=activity_status(bf),
                    source_gap_start_s=a,source_gap_end_s=b,window_width_requested_s=width,window_step_s=step))
                cbasic.append({'id':wid,**bf});cext.append({'id':wid,**xf})
        assert sha(case.cache_path)==cachehash
        cacherecord.append(dict(seq=int(case.seq),cache_path=case.cache_path,sha256=cachehash,unchanged=True))
        log('case_exported',seq=int(case.seq),weak_windows=sum(x['seq']==case.seq for x in metadata),
            unannotated_windows=sum(x['seq']==case.seq for x in candidates),cache_unchanged=True)
    m=pd.DataFrame(metadata);b=pd.DataFrame(basic,columns=basic_cols);x=pd.DataFrame(extended,columns=extended_cols)
    c=pd.DataFrame(candidates);cb=pd.DataFrame(cbasic,columns=basic_cols);cx=pd.DataFrame(cext,columns=extended_cols)
    assert m.id.is_unique and c.id.is_unique and not set(m.id)&set(c.id)
    assert b.id.tolist()==x.id.tolist()==m.id.tolist()
    assert cb.id.tolist()==cx.id.tolist()==c.id.tolist()
    assert len(audit)==110
    assert m.groupby('seq').split.nunique().max()==1
    for k in ['confirmed','treatment_target_valid','instrument_target_valid']:
        assert m[k].notna().all()
    m.to_csv(out/'annotations_aligned.csv',index=False);b.to_csv(out/'features_basic.csv',index=False);x.to_csv(out/'features_extended.csv',index=False)
    c.to_csv(out/'unannotated_candidates.csv',index=False);cb.to_csv(out/'unannotated_features_basic.csv',index=False);cx.to_csv(out/'unannotated_features_extended.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'annotation_window_audit.csv',index=False)
    pd.DataFrame(rejections,columns=['seq','annotation_id','window_id','reason','start_s','end_s','other_annotation_overlap_s']).to_csv(out/'rejected_windows.csv',index=False)
    pd.DataFrame(cacherecord).to_csv(out/'cache_integrity.csv',index=False)
    pd.DataFrame(equivalence).to_csv(out/'process/pruned_feature_equivalence.csv',index=False)
    cases.to_csv(out/'cases_manifest.csv',index=False)
    catalog=[]
    for name in extended_cols:
        if name=='id':continue
        family='context' if name.startswith('context_') else 'temporal_profile' if name.startswith('profile_') else 'quality' if name.startswith(('quality_','count_','rate_')) else 'numeric' if name.startswith(('num_','rpm_')) else 'state_or_structure'
        catalog.append(dict(column=name,role='feature',family=family,feature_eligible=True))
    for name in m.columns:
        catalog.append(dict(column=name,role='target' if name in ['instrument_labels','instrument_normalized','treatment_normalized'] else 'metadata',family='annotation_or_identity',feature_eligible=False))
    pd.DataFrame(catalog).to_csv(out/'metadata_feature_catalog.csv',index=False)
    assert all(sha(p)==h for p,h in before.items())
    meta=dict(version='supervised-weak-windows30-v1',source_annotations=110,source_cases=35,
        weak_windows=len(m),weak_cases=int(m.seq.nunique()),unannotated_candidate_windows=len(c),unannotated_candidate_cases=int(c.seq.nunique()),
        split_counts=m.groupby('split').agg(cases=('seq','nunique'),windows=('id','size')).to_dict('index'),
        source_annotations_with_no_windows=[x['annotation_id'] for x in audit if x['n_exported_windows']==0],
        rejected_window_count=len(rejections),weak_no_active_hp=int(m.activity_status.eq('NO_ACTIVE_HP_OBSERVED').sum()),
        weak_unknown_hp=int(m.activity_status.eq('HP_STATE_UNKNOWN').sum()),
        candidate_activity_counts=c.activity_status.value_counts().to_dict(),
        width_s=width,step_s=step,trim_s=trim,short_window_min_interior_s=10.,
        label_provenance='WEAK_INTERVAL_INHERITED: annotation labels may not apply to every 30sec interior; no independent point verification',
        unlabeled_policy='Complement of UNION of all annotation intervals including unconfirmed; no negative labels inferred',
        ambiguity_policy='Any positive overlap with another source annotation rejects the weak window and is audited',
        split_policy='Original case split inherited; no case/windows crossing. New window learner requires separate dev-only selection and one-shot final.',
        context_policy='Offline +/-60sec may include actual neighboring annotated-operation signals but never their labels; context-free comparator required',
        required_training_weighting='Case-balanced, then annotation-balanced, then windows within annotation; do not treat windows as independent cases',
        feature_schema_matches_interval_exports=True,pruning_equivalence_checks=len(equivalence),synthetic_checks=checks,
        original_sources_unchanged=True,source_hashes=before,source_cache_hashes={x['cache_path']:x['sha256'] for x in cacherecord})
    (out/'process/window_export.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    (out/'code_snapshot').mkdir(exist_ok=True)
    for p in [Path(__file__),Path(__file__).parent/'features.py',Path(__file__).parent/'extract.py']:shutil.copy2(p,out/'code_snapshot'/p.name)
    (out/'export_fingerprint.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in [out/'annotations_aligned.csv',out/'features_basic.csv',out/'features_extended.csv',out/'unannotated_candidates.csv',out/'unannotated_features_basic.csv',out/'unannotated_features_extended.csv']},indent=2))
    log('complete',weak_windows=len(m),unannotated_windows=len(c),source_annotations_with_no_windows=len(meta['source_annotations_with_no_windows']),source_unchanged=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--width',type=float,default=30.)
    ap.add_argument('--step',type=float,default=15.);ap.add_argument('--trim',type=float,default=5.)
    a=ap.parse_args();export(a.source,a.width,a.step,a.trim)
