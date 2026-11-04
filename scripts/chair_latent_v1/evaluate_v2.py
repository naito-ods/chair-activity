"""One-shot evaluation on the metadata-selected fresh v2 cohort."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from semantic_baseline import semantic_features
from study_v2 import transformed
from experiment import record_arrays,case_errors,within_case_prediction
from final_evaluation import load_linear,load_ridge
from prepare_data import sha,CHANNELS

def semantic_points(out):
    o=pd.read_csv(out/'observations.csv',dtype={'code':str})
    o=o[o.hp.eq(2)&o.code.isin(['20742','20743','10015'])]
    keys=['case_id','run_id','second_bin'];names={'20742':'rpm','20743':'torque','10015':'pedal2'}
    values=o.pivot(index=keys,columns='code',values='value').rename(columns=names)
    times=o.pivot(index=keys,columns='code',values='time_s').rename(columns={k:v+'_time_s' for k,v in names.items()})
    for n in names.values():
        if n not in values:values[n]=np.nan
        if n+'_time_s' not in times:times[n+'_time_s']=np.nan
    p=values.join(times).reset_index();p=p[p.rpm.notna()].copy();p['time_s']=p.rpm_time_s
    runs=pd.read_csv(out/'hp_runs.csv')[['case_id','run_id','start_s','end_s']].rename(columns={'start_s':'run_start_s','end_s':'run_end_s'})
    p=p.merge(runs,on=['case_id','run_id'],how='left',validate='many_to_one')
    p['weight_s']=np.maximum(0,np.minimum(p.second_bin+1,p.run_end_s)-np.maximum(p.second_bin,p.run_start_s))
    return p

def evaluate(out):
    if (out/'test_summary.json').exists():raise FileExistsError('Fresh test already evaluated')
    freeze=json.loads((out/'selection_freeze.json').read_text())
    execution=json.loads((out/'execution_freeze.json').read_text())
    for path,h in execution['code_sha256'].items():assert sha(path)==h,path
    assert sha(out/'requested_manifest.csv')==freeze['fresh_manifest_sha256']
    for path,h in freeze['source_model_sha256'].items():assert sha(path)==h,path
    data=np.load(out/'data.npz');ledger=pd.read_csv(out/'case_ledger.csv');manifest=pd.read_csv(out/'case_manifest.csv')
    assert len(ledger)==len(manifest)==200 and ledger.case_id.is_unique
    requested=pd.read_csv(out/'requested_manifest.csv')
    assert ledger.case_id.tolist()==manifest.case_id.tolist()==requested.case_id.tolist()==data['case_ids'].tolist()==freeze['fresh_ids']
    ids=np.flatnonzero(ledger.eligible)
    scale=np.load(out/'scaling.npz');x=(transformed(data['raw'])-scale['center'])/scale['scale']
    xx,vis,tar=record_arrays(x[ids],data['numeric_visible'][ids]);rows=(ids[:,None]*4+np.arange(4)).ravel()
    mean=np.load(out/'models/population.npy');weights=np.sqrt(np.tile([.25,.25,.25,.25,1,1,1],48))
    p={}
    for name,width in [('local6',6.),('local12',12.)]:p[name]=within_case_prediction(xx,vis,mean,width)
    p['linear4']=load_linear(out/'models/linear4.npz').predict(xx*weights,vis)/weights
    intervals=pd.read_csv(out/'hp_intervals.csv');runs=pd.read_csv(out/'hp_runs.csv');points=semantic_points(out)
    fs=[];names=None
    for rep in range(4):
        f,nn=semantic_features(manifest,intervals,runs,points,data['numeric_visible'][:,rep],bins=48)
        if names is not None:assert names==nn
        names=nn;fs.append(f)
    f=np.stack(fs,axis=1).reshape(200*4,-1)[rows]
    p['semantic_kernel']=load_ridge(out/'models/semantic_kernel.npz',True).predict(f)
    mf=np.concatenate([np.where(vis&np.tile(np.arange(7)<4,48),xx,np.nan),vis.astype(float)],axis=1)
    p['mask_HP']=load_ridge(out/'models/mask_HP.npz').predict(mf)
    baseline=np.full_like(xx,np.nan).reshape(-1,48,7);candidate=baseline.copy()
    for j,c in enumerate(CHANNELS[4:],4):
        choice=freeze['candidate_by_channel'][c];b=p[choice['baseline']].reshape(-1,48,7)[:,:,j]
        baseline[:,:,j]=b
        if choice['type']=='baseline':candidate[:,:,j]=b
        else:candidate[:,:,j]=choice['mix']*p['linear4'].reshape(-1,48,7)[:,:,j]+(1-choice['mix'])*p['local6'].reshape(-1,48,7)[:,:,j]
    p['baseline_channelwise']=baseline.reshape(xx.shape);p['signal_specific_latent_local']=candidate.reshape(xx.shape)
    assert np.allclose(candidate[:,:,4],baseline[:,:,4]),'RPM exact fallback invariant'
    (out/'test_predictions').mkdir(exist_ok=True)
    allrows=[];cases=[];err={};ch={}
    for name,pred in p.items():
        e,cc=case_errors(pred,xx,tar,len(ids));err[name]=e;ch[name]=cc
        np.savez_compressed(out/'test_predictions'/f'{name}.npz',prediction=pred,case_indices=ids,case_errors=e,channel_errors=cc)
        allrows.append(dict(model=name,error=float(np.mean(e)),n_cases=len(ids),**{c+'_error':float(np.nanmean(cc[:,j])) for j,c in enumerate(CHANNELS[4:])}))
        cases.extend(dict(model=name,case_id=ledger.case_id.iloc[k],error=ee,**{c+'_error':cc[ii,j] for j,c in enumerate(CHANNELS[4:])}) for ii,(k,ee) in enumerate(zip(ids,e)))
    pd.DataFrame(allrows).to_csv(out/'test_comparison.csv',index=False);pd.DataFrame(cases).to_csv(out/'test_case_errors.csv',index=False)
    rng=np.random.default_rng(20261105+900);boot=rng.integers(0,len(ids),size=(5000,len(ids)));a=err['signal_specific_latent_local'];comp=[]
    for name in ['baseline_channelwise','semantic_kernel','linear4','local6','mask_HP']:
        b=err[name];bg=np.mean((b-a)[boot],axis=1)/np.mean(b[boot],axis=1)
        comp.append(dict(comparator=name,relative_gain=float(1-a.mean()/b.mean()),ci_low=float(np.quantile(bg,.025)),ci_high=float(np.quantile(bg,.975)),fraction_cases_improved=float((a<b).mean()),n_cases=len(ids)))
    pd.DataFrame(comp).to_csv(out/'paired_case_bootstrap.csv',index=False)
    ac=np.nanmean(ch['signal_specific_latent_local'],axis=0);bc=np.nanmean(ch['baseline_channelwise'],axis=0);g=(bc-ac)/bc
    gate=bool(comp[0]['relative_gain']>=.05 and comp[0]['ci_low']>0 and g.min()>=-.1 and comp[-1]['ci_low']>0)
    high=(np.isfinite(data['raw'][ids,:,4:]).sum(1)>=6).all(1)
    sens=dict(definition='>=6 supported bins each numeric channel',n_cases=int(high.sum()),excluded_ids=ledger.case_id.iloc[ids[~high]].tolist())
    if high.any():sens['relative_gain']=float(1-a[high].mean()/err['baseline_channelwise'][high].mean())
    result=dict(version='chair-latent-v2',requested_fresh_cases=200,actual_extracted_cases=200,eligible_cases=len(ids),
        excluded_count=int((~ledger.eligible).sum()),excluded=ledger.loc[~ledger.eligible,['case_id','eligibility_reason']].to_dict('records'),
        primary_comparison=comp[0],numeric_channel_relative_gains=dict(zip(CHANNELS[4:],g.tolist())),
        predictive_gate_passed=gate,rpm_identical_to_baseline=True,
        scientific_status='PREDICTIVE_GAIN_ONLY' if gate else 'SUPERIORITY_NOT_ESTABLISHED',
        new_operation_pattern_validated=False,external_validation=False,clinical_validation=False,
        support_sensitivity=sens,heldout_target_input_overlap=int((vis&tar).sum()),
        hidden_empty_views=int((~tar.any(1)).sum()),all_eligible_cases_have_targets=bool(np.isfinite(a).all()),
        interpretation='Fresh case holdout from same-chair selected duration cohort. Prediction improvement, if supported, does not establish a new treatment class or a new recurring raw operation motif.',
        v1_failed_verdict_preserved=True,v1_test_retired_into_development=True)
    (out/'test_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);evaluate(ap.parse_args().output)
