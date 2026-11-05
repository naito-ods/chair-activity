"""Out-of-case provisional application of separately validated window models.

Never run before BOTH interval and window final evaluations finish. Only class
labels passing both selective gates may be emitted, using the window model's
own frozen threshold. Scores are uncalibrated and unannotated time has no
background ground truth. Additional six skipped cases remain UNKNOWN without
model inference. No frozen learning/extraction source is edited by this file.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,pickle,shutil,traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from learning import Bundle,estimators,target,row_weights,feature_table
from global_context import global_features
from features import state_segments
from extract import load_case_events
from window_data import complement,overlap_length

UNKNOWN='UNKNOWN'
AMBIGUOUS='AMBIGUOUS'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def finite_at(row,name):
    value=row.get(name,np.nan)
    return float(value) if np.isfinite(value) else np.nan

def action_support(row,label):
    if label=='シリンジ':
        picked=finite_at(row,'state_20101_0_share_known')
        return bool(np.isfinite(picked) and picked>=.1 and finite_at(row,'count_20101')>0)
    active=np.array([finite_at(row,f'hp{h}_active_s') for h in [1,2,3]])
    scaler=finite_at(row,'state_20774_1_share_known')
    return bool((np.isfinite(active).any() and np.nansum(active)>=1.) or (np.isfinite(scaler) and scaler>=.1))

def consensus(values):
    """Unanimous supporting windows required; abstention is not a positive."""
    values=list(values)
    if not values:return UNKNOWN,'NO_WINDOW_COVERAGE'
    accepted={v for v in values if v not in [UNKNOWN,AMBIGUOUS,'',None]}
    if len(accepted)>1 or AMBIGUOUS in values:return AMBIGUOUS,'COVERING_WINDOW_DISAGREEMENT'
    if not accepted:return UNKNOWN,'ALL_COVERING_WINDOWS_ABSTAIN'
    if any(v in [UNKNOWN,'',None] for v in values):return UNKNOWN,'INCOMPLETE_COVERING_WINDOW_CONSENSUS'
    return next(iter(accepted)),'UNANIMOUS_PROVISIONAL_WINDOW_CANDIDATE'

def synthetic_checks():
    assert consensus(['A','A'])==('A','UNANIMOUS_PROVISIONAL_WINDOW_CANDIDATE')
    assert consensus(['A',UNKNOWN])[0]==UNKNOWN
    assert consensus(['A','B'])[0]==AMBIGUOUS
    assert consensus([])[0]==UNKNOWN
    assert not action_support({'hp1_active_s':0.,'hp2_active_s':0.,'hp3_active_s':np.nan},'コントラ')
    assert action_support({'hp2_active_s':1.},'コントラ')
    assert not action_support({'state_20101_0_share_known':.5,'count_20101':0},'シリンジ')
    assert action_support({'state_20101_0_share_known':.5,'count_20101':1},'シリンジ')
    return dict(consensus_requires_all_covering_support=True,conflicts_not_smoothed=True,
        no_action_abstains=True,syringe_requires_in_window_record=True)

def passed_labels(interval_root,window_root,task):
    labels=[]
    for root in [interval_root,window_root]:
        f=pd.read_csv(root/f'{task}_final_selective.csv')
        labels.append(set(f.loc[f.passed_for_next_stage.astype(bool),'label']))
    return sorted(labels[0]&labels[1])

def query_matrix(meta,basic,extended,columns,case_manifest,workdir):
    assert basic.id.tolist()==extended.id.tolist()==meta.id.tolist()
    need_global=any(c.startswith('global_') for c in columns)
    context=None
    if need_global:
        rows=[];mapping=case_manifest.set_index('seq')
        for seq,g in meta.groupby('seq',sort=False):
            ev=load_case_events(mapping.loc[seq,'cache_path'])
            if not len(ev):raise ValueError(f'Empty target-code cache in observed cohort seq{seq}')
            lo=float(ev.time_s.min());hi=float(ev.time_s.max())+1e-6
            h=state_segments(ev,'20039',lo,hi)
            for r in g.itertuples():rows.append(dict(id=r.id,**global_features(ev,r.start_s,r.end_s,h)))
        context=pd.DataFrame(rows).set_index('id').loc[meta.id].reset_index()
        context.to_csv(workdir/'candidate_global_features.csv',index=False)
    vectors=[]
    for name in columns:
        if name.startswith('finite__'):
            source=name[len('finite__'):]
            if source not in basic:raise ValueError(f'Missing mask source {source}')
            vectors.append(np.isfinite(basic[source].to_numpy(float)).astype(float))
        elif name in extended:vectors.append(extended[name].to_numpy(float))
        elif name in basic:vectors.append(basic[name].to_numpy(float))
        elif context is not None and name in context:vectors.append(context[name].to_numpy(float))
        else:raise ValueError(f'Frozen model feature missing from query export: {name}')
    result=np.column_stack(vectors)
    assert not np.isinf(result).any()
    return result

def deployment_scores(root,meta,basic,extended,cases,task,frozen,modeldir):
    """Use the exact development GroupKFold exclusion for each query case."""
    spec=frozen['dependencies'][task]
    train_meta,x,columns=feature_table(root,spec['view']);assert columns==spec['features']
    y,mask,classes=target(train_meta,task);assert classes==spec['classes']
    dev=np.flatnonzero(mask&train_meta.split.eq('development').to_numpy())
    groups=train_meta.seq.to_numpy()[dev]
    query=query_matrix(meta,basic,extended,columns,cases,modeldir)
    scores=np.full((len(meta),len(classes)),np.nan)
    keys=np.empty(len(meta),dtype=object);train_lists=np.empty(len(meta),dtype=object)
    manifests=[];templates=estimators()
    dev_case_set=set(groups)
    final_or_untrained=np.flatnonzero(~meta.seq.isin(dev_case_set))
    if len(final_or_untrained):
        path=root/'models'/f'{task}_development.pkl'
        with path.open('rb') as f:model=pickle.load(f)
        assert model.features==columns and model.classes==classes
        scores[final_or_untrained]=model.predict_scores(query[final_or_untrained])
        key=f'{task}__saved_development';seqs=sorted(int(v) for v in dev_case_set)
        for i in final_or_untrained:keys[i]=key;train_lists[i]=json.dumps(seqs)
        qs=sorted(int(v) for v in meta.seq.iloc[final_or_untrained].unique())
        assert not set(qs)&set(seqs)
        manifests.append(dict(task=task,model_key=key,path=str(path.resolve()),source='SAVED_FROZEN_WINDOW_DEVELOPMENT_MODEL',
            train_seqs=json.dumps(seqs),query_seqs=json.dumps(qs),query_case_excluded=True,sha256=sha(path)))
    for fold,(ti,vi) in enumerate(GroupKFold(5).split(x[dev],groups=groups)):
        tr=dev[ti];holdout=sorted(int(v) for v in train_meta.seq.iloc[dev[vi]].unique())
        qi=np.flatnonzero(meta.seq.isin(holdout))
        if not len(qi):continue
        trainseq=sorted(int(v) for v in train_meta.seq.iloc[tr].unique())
        assert not set(trainseq)&set(holdout)
        model=Bundle(task,templates[spec['algorithm']],columns,classes).fit(x[tr],y[tr],train_meta.seq.iloc[tr].to_numpy(),row_weights(train_meta,tr))
        path=modeldir/f'{task}_out_of_case_fold{fold}.pkl';model.save(path)
        scores[qi]=model.predict_scores(query[qi])
        key=f'{task}__out_of_case_fold{fold}'
        for i in qi:keys[i]=key;train_lists[i]=json.dumps(trainseq)
        qs=sorted(int(v) for v in meta.seq.iloc[qi].unique())
        assert not set(qs)&set(trainseq)
        manifests.append(dict(task=task,model_key=key,path=str(path.resolve()),source='REFIT_SELECTED_SPEC_EXCLUDING_QUERY_CASE_FOLD',
            train_seqs=json.dumps(trainseq),query_seqs=json.dumps(qs),query_case_excluded=True,sha256=sha(path)))
    assert np.isfinite(scores).all() and all(isinstance(k,str) for k in keys)
    for i,seq in enumerate(meta.seq):assert int(seq) not in json.loads(train_lists[i])
    return scores,keys,train_lists,manifests

def instrument_compatibility(interval_meta):
    d=interval_meta[interval_meta.split.eq('development')&interval_meta.confirmed.astype(bool)]
    allowed={}
    for r in d.itertuples():
        labs=json.loads(r.instrument_labels) if isinstance(r.instrument_labels,str) else []
        if labs:allowed.setdefault(str(r.treatment_normalized),set()).update(labs)
    return {k:sorted(v) for k,v in allowed.items()}

def atomic_segments(predictions,gaps,annotations):
    segments=[];coverage=[];segment_id=0
    for (seq,cid,originalsplit),casegaps in gaps.groupby(['seq','case_id','split'],sort=True):
        q=predictions[predictions.seq.eq(seq)]
        protected=annotations.loc[annotations.seq.eq(seq),['start_s','end_s']].to_numpy(float)
        caserows=[]
        for gap in casegaps.itertuples():
            a=float(gap.gap_start_s);b=float(gap.gap_end_s)
            relevant=q[q.start_s.ge(a-1e-8)&q.end_s.le(b+1e-8)]
            cuts=np.unique(np.r_[a,b,relevant.start_s.to_numpy(float),relevant.end_s.to_numpy(float)])
            for start,end in zip(cuts[:-1],cuts[1:]):
                if end-start<=1e-9:continue
                assert start>=a-1e-8 and end<=b+1e-8
                assert overlap_length(start,end,protected)<=1e-7
                mid=(start+end)/2
                covering=relevant[relevant.start_s.le(mid)&relevant.end_s.gt(mid)]
                inst,ireason=consensus(covering.instrument_candidate)
                treat,treason=consensus(covering.treatment_candidate)
                raw_treatment_consensus=treat
                if inst in [UNKNOWN,AMBIGUOUS] and treat not in [UNKNOWN,AMBIGUOUS]:
                    treat=UNKNOWN;treason='NO_SEGMENT_INSTRUMENT_CONSENSUS'
                hasinst=inst not in [UNKNOWN,AMBIGUOUS];hastreat=treat not in [UNKNOWN,AMBIGUOUS]
                status='PROVISIONAL_CANDIDATE' if hasinst or hastreat else AMBIGUOUS if AMBIGUOUS in [inst,treat] else UNKNOWN
                segment_id+=1
                row=dict(segment_id=f'segment_{segment_id:06d}',seq=int(seq),case_id=cid,originalsplit=originalsplit,
                    gap_index=int(gap.gap_index),start_s=float(start),end_s=float(end),duration_s=float(end-start),
                    n_covering_windows=len(covering),covering_window_ids=json.dumps(covering.id.tolist()),
                    instrument_candidate=inst,treatment_candidate=treat,raw_treatment_window_consensus=raw_treatment_consensus,
                    instrument_score=float(covering.instrument_score.min()) if hasinst else np.nan,
                    treatment_score=float(covering.treatment_score.min()) if hastreat else np.nan,
                    status=status,instrument_reason=ireason,treatment_reason=treason,
                    annotation_overlap_s=0.,clinical_truth=False)
                segments.append(row);caserows.append(row)
        z=pd.DataFrame(caserows)
        total=float(casegaps.gap_duration_s.sum())
        assert np.isclose(z.duration_s.sum(),total,rtol=1e-10,atol=1e-6)
        coverage.append(dict(seq=int(seq),case_id=cid,originalsplit=originalsplit,n_unannotated_gaps=len(casegaps),
            unannotated_s=total,window_covered_s=float(z.loc[z.n_covering_windows.gt(0),'duration_s'].sum()),
            instrument_candidate_s=float(z.loc[~z.instrument_candidate.isin([UNKNOWN,AMBIGUOUS]),'duration_s'].sum()),
            treatment_candidate_s=float(z.loc[~z.treatment_candidate.isin([UNKNOWN,AMBIGUOUS]),'duration_s'].sum()),
            conflict_s=float(z.loc[z.instrument_candidate.eq(AMBIGUOUS)|z.treatment_candidate.eq(AMBIGUOUS),'duration_s'].sum()),
            unscanned_s=float(z.loc[z.n_covering_windows.eq(0),'duration_s'].sum()),
            both_heads_unknown_s=float(z.loc[z.instrument_candidate.eq(UNKNOWN)&z.treatment_candidate.eq(UNKNOWN),'duration_s'].sum()),
            candidate_windows=len(q),clinical_ground_truth_available=False))
    return pd.DataFrame(segments),pd.DataFrame(coverage)

def apply(interval_root):
    interval_root=interval_root.resolve();out=interval_root/'windows30';skipped=out/'skipped_seq'
    if not (interval_root/'final_summary.json').exists() or not (out/'final_summary.json').exists():
        raise RuntimeError('Both interval and separate window final evaluations must finish BEFORE application')
    if (out/'candidate_predictions.csv').exists():raise FileExistsError('Preserve previous application outputs')
    modeldir=out/'deployment_models';modeldir.mkdir(exist_ok=True)
    checks=synthetic_checks()
    frozen=json.loads((out/'selection_freeze.json').read_text())
    for path,h in {**frozen['input_sha256'],**frozen['code_sha256']}.items():assert sha(path)==h,path
    interval_frozen=json.loads((interval_root/'selection_freeze.json').read_text())
    for path,h in {**interval_frozen['input_sha256'],**interval_frozen['code_sha256']}.items():assert sha(path)==h,path
    meta=pd.read_csv(out/'unannotated_candidates.csv');basic=pd.read_csv(out/'unannotated_features_basic.csv');extended=pd.read_csv(out/'unannotated_features_extended.csv')
    extra=pd.read_csv(skipped/'unannotated_candidates.csv')
    cases=pd.read_csv(interval_root/'cases_manifest.csv');extra_cases=pd.read_csv(skipped/'cases_manifest.csv')
    annotations=pd.read_csv(interval_root/'annotations_aligned.csv')
    assert len(meta)==2479 and len(extra)==251 and len(cases)==35 and len(extra_cases)==6
    assert meta.id.tolist()==basic.id.tolist()==extended.id.tolist()
    assert not set(meta.id)&set(extra.id)
    assert not meta.treatment_target_valid.any() and not extra.treatment_target_valid.any()
    assert not meta.instrument_target_valid.any() and not extra.instrument_target_valid.any()
    input_paths=[interval_root/'annotations_aligned.csv',interval_root/'final_summary.json',interval_root/'selection_freeze.json',
        out/'final_summary.json',out/'selection_freeze.json',out/'unannotated_candidates.csv',out/'unannotated_features_basic.csv',out/'unannotated_features_extended.csv',
        skipped/'unannotated_candidates.csv',skipped/'unannotated_features_basic.csv',skipped/'unannotated_features_extended.csv',out/'unannotated_gap_audit.csv']
    for root in [interval_root,out]:
        input_paths.extend(root/f'{task}_final_selective.csv' for task in ['instrument','treatment'])
    for r in pd.concat([cases,extra_cases],ignore_index=True).itertuples():
        input_paths.extend([Path(r.cache_path),Path(r.source_path)])
    input_paths=list(dict.fromkeys(input_paths));before={str(p):sha(p) for p in input_paths}
    instruments=passed_labels(interval_root,out,'instrument');treatments=passed_labels(interval_root,out,'treatment')
    thresholds={task:{r['label']:float(r['threshold']) for r in frozen['thresholds'][task]} for task in ['instrument','treatment']}
    compatibility=instrument_compatibility(annotations)
    score={};keys={};trainlists={};deployment=[]
    for task in ['instrument','treatment']:
        score[task],keys[task],trainlists[task],records=deployment_scores(out,meta,basic,extended,cases,task,frozen,modeldir)
        deployment.extend(records)
    rows=[];b=basic.set_index('id')
    ic=frozen['dependencies']['instrument']['classes'];tc=frozen['dependencies']['treatment']['classes']
    for i,r in enumerate(meta.itertuples()):
        rawi=int(score['instrument'][i].argmax());rawt=int(score['treatment'][i].argmax())
        accepted=[];qualified_inst=[]
        for j,c in enumerate(ic):
            if c in instruments and score['instrument'][i,j]>=thresholds['instrument'][c]:
                qualified_inst.append(c)
                if action_support(b.loc[r.id],c):accepted.append(c)
        inst='|'.join(sorted(accepted)) if accepted else UNKNOWN
        ireason='SELECTIVE_GATES_THRESHOLD_AND_ACTION_SUPPORTED' if accepted else 'NO_CLASS_PASSED_BOTH_VALIDATION_GATES' if not instruments else 'NO_IN_WINDOW_ACTION_SUPPORT' if qualified_inst else 'BELOW_FROZEN_WINDOW_THRESHOLD'
        tlabel=tc[rawt];tvalue=float(score['treatment'][i,rawt]);treat=UNKNOWN
        if tlabel not in treatments:treason='TOP_CLASS_NOT_SUPPORTED_BY_BOTH_VALIDATION_GATES'
        elif tvalue<thresholds['treatment'][tlabel]:treason='BELOW_FROZEN_WINDOW_THRESHOLD'
        elif not accepted:treason='NO_ACCEPTED_INSTRUMENT'
        elif not set(accepted)&set(compatibility.get(tlabel,[])):treason='NO_DEV_LABEL_INSTRUMENT_COMPATIBILITY'
        else:treat=tlabel;treason='SELECTIVE_GATES_THRESHOLD_AND_INSTRUMENT_COMPATIBLE'
        iscore=max(float(score['instrument'][i,ic.index(c)]) for c in accepted) if accepted else float(score['instrument'][i,rawi])
        row=dict(id=r.id,seq=int(r.seq),case_id=r.case_id,originalsplit=r.split,start_s=r.start_s,end_s=r.end_s,duration_sec=r.duration_sec,
            instrument_candidate=inst,treatment_candidate=treat,instrument_score=iscore,treatment_score=tvalue,
            status='PROVISIONAL_CANDIDATE' if accepted or treat!=UNKNOWN else UNKNOWN,
            score=max(iscore,tvalue),reason=ireason+';'+treason,instrument_reason=ireason,treatment_reason=treason,
            raw_top_instrument=ic[rawi],raw_top_instrument_score=float(score['instrument'][i,rawi]),raw_top_treatment=tlabel,raw_top_treatment_score=tvalue,
            instrument_model_provenance=keys['instrument'][i],treatment_model_provenance=keys['treatment'][i],
            instrument_training_seqs=trainlists['instrument'][i],treatment_training_seqs=trainlists['treatment'][i],
            annotation_overlap_s=0.,score_calibrated=False,clinical_truth=False,source_group='ANNOTATED_CASE_UNANNOTATED_GAP')
        rows.append(row)
    for r in extra.itertuples():
        rows.append(dict(id=r.id,seq=int(r.seq),case_id=r.case_id,originalsplit='UNANNOTATED_ONLY',start_s=r.start_s,end_s=r.end_s,duration_sec=r.duration_sec,
            instrument_candidate=UNKNOWN,treatment_candidate=UNKNOWN,instrument_score=np.nan,treatment_score=np.nan,status=UNKNOWN,score=np.nan,
            reason='ADDITIONAL_SEQ_UNOBSERVABLE_NO_HP_OR_NUMERIC_EVIDENCE',instrument_reason='MODEL_NOT_APPLIED_UNOBSERVABLE_CASE',treatment_reason='MODEL_NOT_APPLIED_UNOBSERVABLE_CASE',
            raw_top_instrument='',raw_top_instrument_score=np.nan,raw_top_treatment='',raw_top_treatment_score=np.nan,
            instrument_model_provenance='NOT_APPLIED',treatment_model_provenance='NOT_APPLIED',instrument_training_seqs='[]',treatment_training_seqs='[]',
            annotation_overlap_s=0.,score_calibrated=False,clinical_truth=False,source_group='ADDITIONAL_UNANNOTATED_ONLY_CASE'))
    predictions=pd.DataFrame(rows);assert len(predictions)==2730 and predictions.id.is_unique
    # Exact annotation exclusion uses ALL110 source rows, including unconfirmed.
    for seq,q in predictions.groupby('seq'):
        protected=annotations.loc[annotations.seq.eq(seq),['start_s','end_s']].to_numpy(float)
        assert all(overlap_length(r.start_s,r.end_s,protected)<=1e-7 for r in q.itertuples())
    gaps=pd.read_csv(out/'unannotated_gap_audit.csv')[['seq','case_id','split','gap_index','gap_start_s','gap_end_s','gap_duration_s']]
    additional=pd.DataFrame([dict(seq=int(r.seq),case_id=r.case_id,split='UNANNOTATED_ONLY',gap_index=0,
        gap_start_s=0.,gap_end_s=float(r.raw_duration_s),gap_duration_s=float(r.raw_duration_s)) for r in extra_cases.itertuples()])
    allgaps=pd.concat([gaps,additional],ignore_index=True)
    segments,coverage=atomic_segments(predictions,allgaps,annotations)
    assert coverage.seq.nunique()==41 and coverage.seq.is_unique
    assert np.isclose(coverage.unannotated_s.sum(),allgaps.gap_duration_s.sum(),atol=1e-6)
    for seq,q in segments.groupby('seq'):
        order=q.sort_values('start_s')
        assert np.all(order.start_s.to_numpy()[1:]>=order.end_s.to_numpy()[:-1]-1e-7)
    predictions.to_csv(out/'candidate_predictions.csv',index=False)
    segments.to_csv(out/'candidate_segments.csv',index=False);coverage.to_csv(out/'candidate_case_coverage.csv',index=False)
    coverage[coverage.originalsplit.ne('UNANNOTATED_ONLY')].to_csv(out/'candidate_35_case_coverage.csv',index=False)
    coverage[coverage.originalsplit.eq('UNANNOTATED_ONLY')].to_csv(out/'candidate_skipped_6_case_coverage.csv',index=False)
    pd.DataFrame(deployment).to_csv(out/'deployment_model_provenance.csv',index=False)
    for task in ['instrument','treatment']:
        allscores=np.full((len(predictions),score[task].shape[1]),np.nan);allscores[:len(meta)]=score[task]
        np.savez_compressed(out/f'candidate_{task}_uncalibrated_scores.npz',ids=predictions.id.to_numpy(str),scores=allscores,
            classes=np.array(frozen['dependencies'][task]['classes']))
    after={str(p):sha(p) for p in input_paths};assert before==after
    result=dict(version='supervised-window-provisional-application-v1',candidate_windows=2730,observed_case_gap_windows=2479,
        skipped_case_unknown_windows=251,total_audited_cases=41,allowed_instrument_labels=instruments,allowed_treatment_labels=treatments,
        compatibility_from_development_annotations=compatibility,candidate_window_status_counts=predictions.status.value_counts().to_dict(),
        candidate_instrument_counts=predictions.instrument_candidate.value_counts().to_dict(),candidate_treatment_counts=predictions.treatment_candidate.value_counts().to_dict(),
        atomic_segments=len(segments),unknown_gap_count=len(allgaps),unannotated_s=float(coverage.unannotated_s.sum()),
        window_covered_s=float(coverage.window_covered_s.sum()),unscanned_s=float(coverage.unscanned_s.sum()),
        provisional_instrument_candidate_s=float(coverage.instrument_candidate_s.sum()),provisional_treatment_candidate_s=float(coverage.treatment_candidate_s.sum()),
        conflict_s=float(coverage.conflict_s.sum()),out_of_case_model_provenance=True,
        clinical_or_background_detection_accuracy_established=False,probability_scores_calibrated=False,
        interpretation='Selective uncalibrated hypotheses only. No negative/background labels exist. UNKNOWN/AMBIGUOUS and uncovered short gaps remain explicit; time is atomized union, never summed overlapping windows.')
    (out/'application_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    audit=dict(input_hashes_before=before,input_hashes_after=after,all_input_files_and_raw_caches_unchanged=True,
        annotation_overlap_s=0.,all110_annotations_protected=True,all_query_cases_excluded_from_model_training=True,
        additional6cases_never_fitted_or_predicted=True,skipped_candidate_count=251,all_skipped_candidates_unknown=True,
        feature_selection_uses_frozen_window_schema_only=True,unannotated_targets_never_assumed_negative=True,
        coverage_atomized_and_nonoverlapping=True,source_code_sha256=sha(__file__),synthetic_checks=checks)
    (out/'process/application_checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    shutil.copy2(Path(__file__),modeldir/'apply_windows.py')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);a=ap.parse_args()
    try:apply(a.source)
    except Exception as exc:
        out=a.source/'windows30';(out/'process').mkdir(parents=True,exist_ok=True)
        (out/'process/application_error.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),ensure_ascii=False,indent=2));raise
