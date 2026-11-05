"""Decompose window-context candidate spans into exact observed state support.

Interpretation only: no classifications, model scores, gates or candidate-band
boundaries are changed. HP3/scaler ON states are not clinical treatment proof.
An accepted 30sec-window context may contain OFF/unknown time at its edges.
"""
from pathlib import Path
import argparse,datetime,json,hashlib,shutil
import numpy as np
import pandas as pd
from features import state_segments
from extract import load_case_events
from window_data import overlap_length

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def measure(partition,start,end,value):
    widths=np.maximum(0.,np.minimum(partition[:,1],end)-np.maximum(partition[:,0],start))
    choose=np.isnan(partition[:,2]) if value=='UNKNOWN' else partition[:,2]==value
    return float(widths[choose].sum())

def overlay(source):
    source=source.resolve();out=source/'windows30'
    if (out/'candidate_raw_support_summary.json').exists():raise FileExistsError('Preserve previous raw-support interpretation')
    p=out/'candidate_segments.csv';old=pd.read_csv(p,float_precision='round_trip');segments=old.copy(deep=True)
    metadata=pd.concat([pd.read_csv(source/'cases_manifest.csv'),pd.read_csv(out/'skipped_seq/cases_manifest.csv')],ignore_index=True)
    assert metadata.seq.is_unique and len(metadata)==41
    annotations=pd.read_csv(source/'annotations_aligned.csv')
    protected={int(seq):g[['start_s','end_s']].to_numpy(float) for seq,g in annotations.groupby('seq')}
    backup=out/'process/raw_support_overlay';backup.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,backup/'candidate_segments_before_overlay.csv')
    pred_hash=sha(out/'candidate_predictions.csv')
    modelpaths=list((out/'models').glob('*.pkl'))+list((out/'deployment_models').glob('*.pkl'))
    modelhash={str(p):sha(p) for p in modelpaths}
    sourcehash={str(p):sha(p) for p in [source/'annotations_aligned.csv',source/'selection_freeze.json',out/'selection_freeze.json']}
    fields=['hp3_observed_active_s','scaler20774_on_s','hp_known_off_s','hp_unknown_s','otherhp_active_s']
    for f in fields:segments[f]=np.nan
    segments['time_basis']='WINDOW_CONTEXT_SPAN'
    segments['clinical_label_confirmed']=False
    segments['raw_support_flag']=''
    exact=[];state_maps={};cachehash={};origins={}
    for case in metadata.itertuples():
        events=load_case_events(case.cache_path);cachehash[case.cache_path]=sha(case.cache_path)
        hp=state_segments(events,'20039',0.,float(case.raw_duration_s))
        scaler=state_segments(events,'20774',0.,float(case.raw_duration_s))
        state_maps[int(case.seq)]=hp;origins[int(case.seq)]=pd.Timestamp(case.raw_start_datetime)
        q=segments[segments.seq.eq(case.seq)]
        for index,row in q.iterrows():
            start=float(row['start_s']);end=float(row['end_s']);duration=end-start
            h3=measure(hp,start,end,3.);on=measure(scaler,start,end,1.)
            off=measure(hp,start,end,0.);unknown=measure(hp,start,end,'UNKNOWN')
            other=measure(hp,start,end,1.)+measure(hp,start,end,2.)
            assert np.isclose(h3+off+unknown+other,duration,atol=1e-8)
            segments.loc[index,fields]=[h3,on,off,unknown,other]
            segments.loc[index,'raw_support_flag']='NO_IN_ATOM_SCALER_OPERATION' if h3<=1e-9 and on<=1e-9 else 'SCALER_OPERATION_STATE_OBSERVED'
            if row['status']!='PROVISIONAL_CANDIDATE':continue
            for a,b,value in hp:
                if value!=3:continue
                a=max(float(a),start);b=min(float(b),end)
                if b-a<=1e-9:continue
                assert overlap_length(a,b,protected.get(int(case.seq),[]))<=1e-7
                exact.append(dict(seq=int(case.seq),case_id=case.case_id,originalsplit=row['originalsplit'],
                    start_s=a,end_s=b,source_candidate_segment_ids=[row['segment_id']]))
    # Merge only a physically contiguous HP3 state through adjacent accepted atoms.
    merged=[]
    for row in sorted(exact,key=lambda x:(x['seq'],x['start_s'],x['end_s'])):
        same=bool(merged and row['seq']==merged[-1]['seq'] and abs(row['start_s']-merged[-1]['end_s'])<=1e-8)
        if same:
            previous=merged[-1];start=previous['start_s'];end=row['end_s']
            support=measure(state_maps[row['seq']],start,end,3.)
            if np.isclose(support,end-start,atol=1e-8,rtol=0):
                previous['end_s']=end
                previous['source_candidate_segment_ids']+=row['source_candidate_segment_ids']
                continue
        merged.append(dict(row))
    for i,row in enumerate(merged):
        row['hp3_run_id']=f'candidate_hp3_{i+1:04d}'
        row['duration_s']=row['end_s']-row['start_s']
        row['start_datetime']=(origins[row['seq']]+pd.Timedelta(seconds=row['start_s'])).isoformat()
        row['end_datetime']=(origins[row['seq']]+pd.Timedelta(seconds=row['end_s'])).isoformat()
        row['source_candidate_segment_ids']=json.dumps(sorted(set(row['source_candidate_segment_ids'])))
        row['support_basis']='EXACT_CLIPPED_HP3_STATE_INTERVAL'
        row['display_label']='HP3作動状態（用途未判別、臨床ラベル未確認）'
        row['clinical_label_confirmed']=False
        assert overlap_length(row['start_s'],row['end_s'],protected.get(row['seq'],[]))<=1e-7
        assert np.isclose(measure(state_maps[row['seq']],row['start_s'],row['end_s'],3.),row['duration_s'],atol=1e-8,rtol=0)
    runs=pd.DataFrame(merged)
    segments.to_csv(p,index=False)
    reread=pd.read_csv(p,float_precision='round_trip')
    pd.testing.assert_frame_equal(old,reread[old.columns])
    runs.to_csv(out/'candidate_observed_hp3_runs.csv',index=False)
    accepted=segments[segments.status.eq('PROVISIONAL_CANDIDATE')]
    assert np.isclose(accepted.duration_s.sum(),120.,atol=1e-8)
    assert np.isclose(runs.duration_s.sum(),accepted.hp3_observed_active_s.sum(),atol=1e-8)
    for path,h in {**sourcehash,**cachehash,**modelhash}.items():assert sha(path)==h,path
    assert sha(out/'candidate_predictions.csv')==pred_hash
    summary=dict(version='candidate-window-context-raw-support-overlay-v1',interpretation_only=True,
        candidate_time_basis='WINDOW_CONTEXT_SPAN',candidate_context_s=float(accepted.duration_s.sum()),
        candidate_context_atoms=len(accepted),candidate_hp3_observed_active_s=float(accepted.hp3_observed_active_s.sum()),
        candidate_scaler20774_on_s=float(accepted.scaler20774_on_s.sum()),
        candidate_hp_known_off_s=float(accepted.hp_known_off_s.sum()),candidate_hp_unknown_s=float(accepted.hp_unknown_s.sum()),
        candidate_otherhp_active_s=float(accepted.otherhp_active_s.sum()),
        candidate_no_in_atom_scaler_operation_s=float(accepted.loc[accepted.raw_support_flag.eq('NO_IN_ATOM_SCALER_OPERATION'),'duration_s'].sum()),
        candidate_no_in_atom_scaler_operation_atoms=int(accepted.raw_support_flag.eq('NO_IN_ATOM_SCALER_OPERATION').sum()),
        hp3_exact_clipped_runs=len(runs),hp3_run_duration_s=float(runs.duration_s.sum()),
        candidate_context_by_case=accepted.groupby('seq').duration_s.sum().to_dict(),
        hp3_observed_active_by_case=accepted.groupby('seq').hp3_observed_active_s.sum().to_dict(),
        hp_known_off_by_case=accepted.groupby('seq').hp_known_off_s.sum().to_dict(),
        signals_are_not_additive=True,
        decomposition_identity='HP3 + otherHP + knownOFF + unknown = candidate context span. 20774ON is a separate possibly-overlapping state channel and MUST NOT be added to HP3 seconds.',
        raw_state_policy='Observed same-code HP/scaler states persist until next recorded state; no numeric interpolation or OFF/unknown-gap filling.',
        clinical_label_confirmed=False,classification_or_confidence_changed=False,
        classification_accuracy_improvement_claimed=False,
        display_note='候補帯は窓の合意範囲で、実作動時間ではありません。正確なHP3作動状態を別に表示し、OFF・未知・他HPを分けてください。',
        state_observation_is_not_clinical_treatment_proof=True)
    (out/'candidate_raw_support_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    verify=dict(original_segments_sha256=sha(backup/'candidate_segments_before_overlay.csv'),
        original_segment_columns_exactly_unchanged=True,prediction_csv_unchanged_sha256=pred_hash,
        model_sha256_unchanged=modelhash,source_and_cache_sha256_unchanged={**sourcehash,**cachehash},
        clinical_confirmation_never_added=True,all_clipped_runs_zero_annotation_overlap=True,
        no_off_or_unknown_interval_interpolated=True,source_code_sha256=sha(__file__))
    (backup/'verification.json').write_text(json.dumps(verify,ensure_ascii=False,indent=2)+'\n')
    shutil.copy2(Path(__file__),backup/'raw_support_overlay.py')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);overlay(ap.parse_args().source)
