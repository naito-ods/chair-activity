"""Freeze development choices, then evaluate a single untouched case holdout.

Run freeze before evaluate. Both preserve immutable audit files. No model is
refitted or selected from test scores. Test is same-chair internal case holdout,
not independent-patient/site clinical validation.
"""
from pathlib import Path
import argparse,json,datetime,shutil
import numpy as np
import pandas as pd
from experiment import (SEED,coordinate_scale,record_arrays,case_errors,log,
    RidgeObserved,MaskedFactors,within_case_prediction,prepare_features)
from improve import KernelObserved
from latent_model import MaskedVAE
from prepare_data import sha,CHANNELS,SOURCE

def context(out):
    data=np.load(out/'data.npz');ledger=pd.read_csv(out/'case_ledger.csv')
    train=np.flatnonzero(ledger.split.eq('train'));x,_,_=coordinate_scale(data['raw'],train)
    xx,visible,target=record_arrays(x,data['numeric_visible'])
    f,names=prepare_features(out,ledger,data['numeric_visible'])
    rows=pd.read_csv(out/'development_comparison.csv').set_index('model').to_dict('index')
    return data,ledger,train,x,xx,visible,target,f,rows

def load_ridge(p,kernel=False):
    d=np.load(p);m=KernelObserved(float(d['alpha']),float(d['width'])) if kernel else RidgeObserved(float(d['alpha']))
    for name in ['center','scale','coef','target_mean']:setattr(m,name,d[name])
    if kernel:m.a=d['a']
    return m

def load_linear(p):
    d=np.load(p);m=MaskedFactors(int(d['k']),regularization=float(d['reg']));m.mean=d['mean'];m.loading=d['loading'];return m

def predict(out,name,xx,visible,f,rows):
    r=rows[name];family=r['family'];mean=np.load(out/'models/population.npy')
    if family=='population':return np.broadcast_to(mean,xx.shape).copy()
    if family=='within_case':return within_case_prediction(xx,visible,mean,None if name=='case_median' else float(r['bandwidth']))
    if family in ['semantic','semantic_kernel']:return load_ridge(out/'models'/f'{name}.npz',family=='semantic_kernel').predict(f)
    if family in ['raw_ridge','mask_HP','shuffle_ridge']:
        features=np.where(visible,xx,np.nan)
        if family=='mask_HP':features=np.concatenate([np.where(visible&np.tile(np.arange(7)<4,48),xx,np.nan),visible.astype(float)],axis=1)
        if family=='shuffle_ridge':features=shuffle_features(xx,visible)
        return load_ridge(out/'models'/f'{name}.npz').predict(features)
    if family=='linear':
        weights=np.sqrt(np.tile([.25,.25,.25,.25,1,1,1],48))
        return load_linear(out/'models'/f'{name}.npz').predict(xx*weights,visible)/weights
    if family in ['vae','ae']:return MaskedVAE.load(out/'models'/f'{name}.npz').predict(xx,visible)
    if family in ['conditional_linear','conditional_vae']:
        numeric=np.tile(np.arange(7)>=4,48)
        bp=within_case_prediction(xx,visible,mean,None if r['base']=='median' else 6.)
        residual=xx.copy();residual[:,numeric]-=bp[:,numeric]
        if family=='conditional_vae':pp=MaskedVAE.load(out/'models'/f'{name}.npz').predict(residual,visible)
        else:
            w=np.sqrt(np.tile([.025,.025,.025,.025,1,1,1],48))
            pp=load_linear(out/'models'/f'{name}.npz').predict(residual*w,visible)/w
        pp[:,numeric]+=bp[:,numeric];return pp
    if family in ['hybrid','linear_local']:
        lp=predict(out,r['latent_model'],xx,visible,f,rows)
        width=float(r['bandwidth']) if family=='linear_local' else (None if r['base']=='median' else 6.)
        bp=within_case_prediction(xx,visible,mean,width)
        return float(r['mix'])*lp+(1-float(r['mix']))*bp
    raise ValueError(family)

def shuffle_features(xx,visible):
    f=np.where(visible,xx,np.nan).reshape(-1,48,7).copy()
    # Values shuffled only among visible positions, masks unchanged. HP fixed.
    # Per-view RNG independent of case identity/order; same algorithm all splits.
    for i in range(len(f)):
        rng=np.random.default_rng(SEED+700)
        for c in range(4,7):
            ix=np.flatnonzero(np.isfinite(f[i,:,c]));f[i,ix,c]=rng.permutation(f[i,ix,c])
    return f.reshape(xx.shape)

def controls(out):
    _,ledger,train,_,xx,vis,tar,f,rows=context(out)
    if (out/'selection_freeze.json').exists():raise RuntimeError('Development is frozen')
    tr=(train[:,None]*4+np.arange(4)).ravel();val=np.flatnonzero(ledger.split.eq('validation'));vr=(val[:,None]*4+np.arange(4)).ravel()
    table=pd.read_csv(out/'development_comparison.csv');features=shuffle_features(xx,vis)
    from experiment import evaluate_candidate
    entries=table.to_dict('records')
    for alpha in [10.,100.]:
        name=f'shuffle_ridge_a{alpha:g}'
        if name in rows:continue
        m=RidgeObserved(alpha).fit(features[tr],xx[tr],tar[tr]);m.save(out/'models'/f'{name}.npz')
        evaluate_candidate(out,name,m.predict(features[vr]),xx[vr],tar[vr],val,entries,dict(family='shuffle_ridge',alpha=alpha,revision=3))

def freeze(out):
    if (out/'selection_freeze.json').exists():raise FileExistsError('Selection already frozen')
    _,ledger,_,_,_,_,_,_,_=context(out)
    table=pd.read_csv(out/'development_comparison.csv')
    def best(families):return table[table.family.isin(families)].sort_values(['validation_error','model']).iloc[0].model
    basefamilies=['semantic','semantic_kernel','linear','within_case','raw_ridge']
    choices=dict(candidate=best(['vae','conditional_vae','hybrid','linear_local']),
        semantic=best(['semantic','semantic_kernel']),linear=best(['linear']),
        local=best(['within_case']),best_single_baseline=best(basefamilies),
        joint_vae=best(['vae']),conditional_vae=best(['conditional_vae']),
        mask_HP=best(['mask_HP']),temporal_shuffle=best(['shuffle_ridge']))
    basetable=table[table.family.isin(basefamilies)]
    channelwise={c:basetable.sort_values(c+'_error').iloc[0].model for c in CHANNELS[4:]}
    source=Path(__file__).parent;(out/'code_snapshot').mkdir(exist_ok=True)
    for p in source.glob('*.py'):shutil.copy2(p,out/'code_snapshot'/p.name)
    checks={str(p.relative_to(out)):sha(p) for p in [out/'data.npz',out/'case_ledger.csv',out/'semantic_inputs.npz',out/'development_comparison.csv',out/'models/population.npy',*sorted((out/'models').glob('*.npz'))]}
    freeze=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=SEED,choices=choices,
        channelwise_baseline=channelwise,selection='Validation error only. No test scores inspected.',
        primary_comparator='baseline_channelwise (per-channel constituent selected on validation only)',
        test_case_ids=ledger.loc[ledger.split.eq('test'),'case_id'].tolist(),
        bootstrap_case_repeats=5000,practical_gain=.05,ci_lower_must_exceed_zero=True,
        per_channel_max_relative_degradation=.10,source_and_model_sha256=checks,
        predictive_gain_is_not_new_operation_pattern=True,clinical_or_external_validation=False,
        controls='mask+HP trained control; retrained shuffled-visible-values Ridge; query temporal perturbation sensitivity of candidate',
        no_further_test_based_tuning=True,
        executable_source_sha256={str((source/name).resolve()):sha(source/name) for name in
            ['prepare_data.py','experiment.py','improve.py','latent_model.py','semantic_baseline.py','final_evaluation.py']})
    (out/'selection_freeze.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n')
    log(out,'selection_frozen',choices=choices,channelwise_baseline=channelwise)

def evaluate(out):
    if (out/'test_summary.json').exists():raise FileExistsError('Final test already scored; do not tune/repeat')
    freeze=json.loads((out/'selection_freeze.json').read_text())
    for rel,expected in freeze['source_and_model_sha256'].items():assert sha(out/rel)==expected,rel
    for path,expected in freeze['executable_source_sha256'].items():assert sha(path)==expected,path
    data,ledger,train,x,xx,vis,tar,f,rows=context(out)
    test=np.flatnonzero(ledger.split.eq('test'));ix=(test[:,None]*4+np.arange(4)).ravel()
    (out/'test_predictions').mkdir(exist_ok=True)
    names=list(dict.fromkeys([*freeze['choices'].values(),*freeze['channelwise_baseline'].values()]))
    predictions={name:predict(out,name,xx[ix],vis[ix],f[ix],rows) for name in names}
    combined=np.full_like(xx[ix],np.nan).reshape(len(ix),48,7)
    for c,name in freeze['channelwise_baseline'].items():
        j=CHANNELS.index(c);combined[:,:,j]=predictions[name].reshape(len(ix),48,7)[:,:,j]
    predictions['baseline_channelwise']=combined.reshape(xx[ix].shape)
    # Perturbation diagnostic: frozen predictor, same masks/HP/hidden targets.
    shuffled=shuffle_features(xx[ix],vis[ix]);shufvalues=xx[ix].copy();numeric=np.tile(np.arange(7)>=4,48)
    shufvalues[:,numeric]=np.where(vis[ix][:,numeric],shuffled[:,numeric],xx[ix][:,numeric])
    candidate=freeze['choices']['candidate']
    predictions['candidate_shuffled_query']=predict(out,candidate,shufvalues,vis[ix],f[ix],rows)
    summary=[];percase=[];errors={};channel_errors={}
    for name,p in predictions.items():
        ee,cc=case_errors(p,xx[ix],tar[ix],len(test));errors[name]=ee;channel_errors[name]=cc
        np.savez_compressed(out/'test_predictions'/f'{name}.npz',prediction=p,case_indices=test,case_errors=ee,channel_errors=cc)
        row=dict(model=name,test_error=float(np.nanmean(ee)),n_cases=len(test))
        for j,c in enumerate(CHANNELS[4:]):row[c+'_error']=float(np.nanmean(cc[:,j]));row[c+'_supported_cases']=int(np.isfinite(cc[:,j]).sum())
        summary.append(row)
        percase.extend(dict(model=name,case_id=ledger.case_id.iloc[i],error=float(e),RPM_error=cc[k,0],torque_error=cc[k,1],pedal2_error=cc[k,2]) for k,(i,e) in enumerate(zip(test,ee)))
    pd.DataFrame(summary).to_csv(out/'test_comparison.csv',index=False);pd.DataFrame(percase).to_csv(out/'test_case_errors.csv',index=False)
    rng=np.random.default_rng(SEED+9000);boot=rng.integers(0,len(test),size=(freeze['bootstrap_case_repeats'],len(test)))
    comparisons=[]
    for other in ['baseline_channelwise',freeze['choices']['semantic'],freeze['choices']['linear'],freeze['choices']['local'],freeze['choices']['mask_HP']]:
        a=errors[candidate];b=errors[other];delta=b-a
        bootgain=np.mean(delta[boot],axis=1)/np.mean(b[boot],axis=1)
        comparisons.append(dict(candidate=candidate,comparator=other,relative_gain=float(delta.mean()/b.mean()),
            ci_low=float(np.quantile(bootgain,.025)),ci_high=float(np.quantile(bootgain,.975)),
            fraction_cases_improved=float((delta>0).mean()),n_cases=len(test)))
    pd.DataFrame(comparisons).to_csv(out/'paired_case_bootstrap.csv',index=False)
    primary=comparisons[0];a=np.nanmean(channel_errors[candidate],axis=0);b=np.nanmean(channel_errors['baseline_channelwise'],axis=0)
    channel_gain=(b-a)/b
    mask_comparison=comparisons[-1]
    mask_gate=bool(mask_comparison['relative_gain']>=.05 and mask_comparison['ci_low']>0)
    gate=bool(primary['relative_gain']>=.05 and primary['ci_low']>0 and channel_gain.min()>=-.10 and mask_gate)
    viewaudit=[]
    for s in ['train','validation','test']:
        ids=np.flatnonzero(ledger.split.eq(s));rs=(ids[:,None]*4+np.arange(4)).ravel()
        for j,r in enumerate(rs):viewaudit.append(dict(case_id=ledger.case_id.iloc[r//4],split=s,mask_replicate=r%4,
            hidden_numeric_target_count=int(tar[r].sum()),view_has_target=bool(tar[r].any())))
    pd.DataFrame(viewaudit).to_csv(out/'hidden_target_support.csv',index=False)
    checks=dict(case_ids_disjoint=True,unique_case_count=300,hp2_evaluable=249,train=149,validation=50,test=50,
        test_cases_with_any_hidden_target=int(np.isfinite(errors[candidate]).sum()),
        empty_target_views_by_split=pd.DataFrame(viewaudit).groupby('split').view_has_target.apply(lambda s:int((~s).sum())).to_dict(),
        empty_target_policy='No case excluded. Empty individual views have no scoring terms; allcase50 aggregated over complementary4 layouts. VAE earlystop evaluates nonempty views only, explicit audit.',
        observed_target_input_overlap=int((vis&tar).sum()),test_selection_frozen=True,
        no_annotation_targets=True,signal_transform='RPM/40000; log1p torque/pedal; traincase median/SD scales; original raw values retained')
    # Fixed support sensitivity; no refitting or selection from this result.
    supportcases=(np.isfinite(data['raw'][test,:,4:]).sum(1)>=6).all(1)
    support=dict(definition='>=6 observed macro bins for EACH of RPM/torque/pedal2',n_cases=int(supportcases.sum()),
        case_ids=ledger.case_id.iloc[test[supportcases]].tolist(),excluded_ids=ledger.case_id.iloc[test[~supportcases]].tolist())
    if supportcases.any():support['relative_gain']=float(1-errors[candidate][supportcases].mean()/errors['baseline_channelwise'][supportcases].mean())
    result=dict(version='chair-latent-v1',candidate=candidate,primary_comparison=primary,
        numeric_channel_relative_gains=dict(zip(CHANNELS[4:],channel_gain.tolist())),predictive_gate_passed=gate,mask_control_gate_passed=mask_gate,
        scientific_status='PREDICTIVE_GAIN_ONLY_PENDING_RECURRENCE_REVIEW' if gate else 'SUPERIORITY_NOT_ESTABLISHED',
        new_operation_pattern_validated=False,clinical_validation=False,external_validation=False,
        interpretation='Continuous latent hybrid may add predictive value; this does not establish a new treatment or natural class.',
        quality_checks=checks,support_sensitivity=support,selected_models=freeze['choices'],
        cohort='Same-chair selected300 cohort; case IDs do not prove independent patients; test is internal heldout case evaluation.')
    (out/'test_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    log(out,'final_test_complete',candidate=candidate,primary_comparison=primary,predictive_gate_passed=gate)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('stage',choices=['controls','freeze','evaluate'])
    a=ap.parse_args();globals()[a.stage](a.output)
