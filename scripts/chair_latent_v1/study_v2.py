"""Second, separately frozen study: signal-specific latent/local blends.

V1's test is retired into development. Model choice uses case cross-fitting on
those original249 eligible cases only. Fresh cases are selected from metadata
and cannot be extracted numerically before this version's freeze. RPM falls
back EXACTLY to its strongest development baseline: no claimed new RPM gain.
"""
from pathlib import Path
import argparse,json,datetime,shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from experiment import coordinate_scale,record_arrays,case_errors,MaskedFactors,within_case_prediction,RidgeObserved
from improve import KernelObserved
from final_evaluation import load_linear,load_ridge
from prepare_data import sha,CHANNELS

SEED=20261105

def transformed(raw):
    x=raw.copy();x[:,:,4]/=40000
    x[:,:,5:]=np.log1p(x[:,:,5:]);return x

def point_mean(x):return np.array([np.nanmean(v) if np.isfinite(v).any() else 0 for v in x.T])

def scalar_predict(fold_pred,center,scale,gcenter,gscale):
    a=fold_pred.reshape(-1,48,7)*scale+center
    return ((a-gcenter)/gscale).reshape(fold_pred.shape)

def develop(old,out):
    out.mkdir(exist_ok=True,parents=True);(out/'process').mkdir(exist_ok=True);(out/'models').mkdir(exist_ok=True)
    if (out/'selection_freeze.json').exists():raise FileExistsError('V2 already frozen')
    data=np.load(old/'data.npz');ledger=pd.read_csv(old/'case_ledger.csv');ids=np.flatnonzero(ledger.eligible)
    assert len(ids)==249
    gx,gc,gs=coordinate_scale(data['raw'],ids);gxx,_,gt=record_arrays(gx[ids],data['numeric_visible'][ids])
    f=np.load(old/'semantic_inputs.npz')['f'];allrows=(ids[:,None]*4+np.arange(4)).ravel();f=f[allrows]
    basenames=['local6','local12','linear4','semantic_kernel']
    predictions={n:np.full_like(gxx,np.nan) for n in basenames+['mask_HP']}
    foldrows=[];weights=np.sqrt(np.tile([.25,.25,.25,.25,1,1,1],48))
    for fold,(ti,vi) in enumerate(KFold(3,shuffle=True,random_state=SEED).split(ids)):
        tx,c,s=coordinate_scale(data['raw'],ids[ti]);xx,vis,tar=record_arrays(tx[ids],data['numeric_visible'][ids])
        tr=(ti[:,None]*4+np.arange(4)).ravel();vr=(vi[:,None]*4+np.arange(4)).ravel()
        mean=point_mean(tx[ids[ti]].reshape(len(ti),-1))
        for n,width in [('local6',6.),('local12',12.)]:predictions[n][vr]=scalar_predict(within_case_prediction(xx[vr],vis[vr],mean,width),c,s,gc,gs)
        model=MaskedFactors(4).fit(tx[ids[ti]].reshape(len(ti),-1)*weights)
        predictions['linear4'][vr]=scalar_predict(model.predict(xx[vr]*weights,vis[vr])/weights,c,s,gc,gs)
        km=KernelObserved(.1,2.).fit(f[tr],xx[tr],tar[tr]);predictions['semantic_kernel'][vr]=scalar_predict(km.predict(f[vr]),c,s,gc,gs)
        mf=np.concatenate([np.where(vis&np.tile(np.arange(7)<4,48),xx,np.nan),vis.astype(float)],axis=1)
        rm=RidgeObserved(100).fit(mf[tr],xx[tr],tar[tr]);predictions['mask_HP'][vr]=scalar_predict(rm.predict(mf[vr]),c,s,gc,gs)
        foldrows.extend(dict(case_id=ledger.case_id.iloc[ids[i]],fold=fold,original_split=ledger.split.iloc[ids[i]]) for i in vi)
        print(json.dumps(dict(stage='v2_crossfit',fold=fold,train_cases=len(ti),validation_cases=len(vi))),flush=True)
    pd.DataFrame(foldrows).to_csv(out/'development_folds.csv',index=False)
    errors={};table=[]
    for name,p in predictions.items():
        e,cc=case_errors(p,gxx,gt,len(ids));errors[name]=(e,cc)
        table.append(dict(model=name,error=float(e.mean()),**{c+'_error':float(np.nanmean(cc[:,j])) for j,c in enumerate(CHANNELS[4:])}))
    baseline={c:min(basenames,key=lambda n:np.nanmean(errors[n][1][:,j])) for j,c in enumerate(CHANNELS[4:])}
    # Select global/local mixture only where cross-fitted improvement is supported.
    channel_choice={};selection=[];rng=np.random.default_rng(SEED+100);boot=rng.integers(0,len(ids),size=(3000,len(ids)))
    for j,c in enumerate(CHANNELS[4:]):
        b=errors[baseline[c]][1][:,j];candidates=[]
        for mix in [.25,.5,.75]:
            pp=mix*predictions['linear4']+(1-mix)*predictions['local6'];_,cc=case_errors(pp,gxx,gt,len(ids))
            e=cc[:,j];good=np.isfinite(e)&np.isfinite(b)
            ix=np.flatnonzero(good);bb=np.random.default_rng(SEED+200+j).integers(0,len(ix),size=(3000,len(ix)))
            gains=np.mean((b[good]-e[good])[bb],axis=1)/np.mean(b[good][bb],axis=1)
            row=dict(channel=c,mix=mix,error=float(np.nanmean(e)),baseline=baseline[c],baseline_error=float(np.nanmean(b)),relative_gain=float(1-np.nanmean(e)/np.nanmean(b)),ci_low=float(np.quantile(gains,.025)),ci_high=float(np.quantile(gains,.975)))
            candidates.append(row);selection.append(row)
        best=min(candidates,key=lambda r:r['error'])
        use=bool(c!='RPM' and best['relative_gain']>=.05 and best['ci_low']>0)
        channel_choice[c]=dict(type='blend' if use else 'baseline',baseline=baseline[c],mix=best['mix'] if use else 0.,
            reason='Cross-fit supported incremental numeric prediction' if use else 'Exact baseline fallback; RPM deliberately preserved after V1 tradeoff')
    pd.DataFrame(table).to_csv(out/'crossfit_comparison.csv',index=False);pd.DataFrame(selection).to_csv(out/'crossfit_blends.csv',index=False)
    # Fit final immutable parameters using all original eligible cases, no fresh values.
    xx,vis,tar=record_arrays(gx[ids],data['numeric_visible'][ids]);mean=point_mean(gx[ids].reshape(len(ids),-1))
    np.save(out/'models/population.npy',mean);np.savez(out/'scaling.npz',center=gc,scale=gs)
    MaskedFactors(4).fit(gx[ids].reshape(len(ids),-1)*weights).save(out/'models/linear4.npz')
    KernelObserved(.1,2.).fit(f,xx,tar).save(out/'models/semantic_kernel.npz')
    mf=np.concatenate([np.where(vis&np.tile(np.arange(7)<4,48),xx,np.nan),vis.astype(float)],axis=1)
    RidgeObserved(100).fit(mf,xx,tar).save(out/'models/mask_HP.npz')
    ledger.to_csv(out/'retired_v1_development_ledger.csv',index=False)
    sources={str(p.resolve()):sha(p) for p in [old/'data.npz',old/'semantic_inputs.npz',old/'test_summary.json',out/'scaling.npz',*sorted((out/'models').glob('*'))]}
    plan=out/'requested_manifest.csv'
    if not plan.exists():raise RuntimeError('Metadata-selected fresh case manifest must precede freeze')
    fresh=pd.read_csv(plan);assert len(fresh)==200 and not set(fresh.case_id)&set(ledger.case_id)
    assert not set(fresh.source_sha256)&set(pd.read_csv(old/'case_manifest.csv').source_sha256)
    selectionfreeze=dict(version='chair-latent-v2',seed=SEED,frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original300_role='All original300 now development; v1 finaltest retired.249 eligible used in crossfit/finalfit.',
        training_cases=249,case_crossfit_folds=3,linear_latent_dim=4,no_annotation_targets=True,
        baseline_by_channel=baseline,candidate_by_channel=channel_choice,
        primary='same hidden observed numeric targets; case/channel-balanced standardized squared error',
        practical_gain=.05,paired_bootstrap_repeats=5000,ci_low_gt_zero=True,max_channel_degradation=.10,
        rpm='Exact selected baseline prediction. No new RPM signal information claimed.',
        fresh_requested_cases=200,fresh_manifest_sha256=sha(plan),fresh_ids=fresh.case_id.tolist(),
        cohort_eligibility='>=1 supported RPM macrobin (>=3 actual observations/bin) AND positive HP2run; explicitly differs from v1 v3-pattern evaluable249 definition',
        source_model_sha256=sources,method='Signal-specific blend of conditional linear latent reconstruction and local observed-value predictor. No neural superiority claim.',
        discovery='Predictive gain alone is insufficient for new operation-pattern claim; no fixed class count',
        final_test_once=True,external_patient_or_clinical_validation=False)
    (out/'selection_freeze.json').write_text(json.dumps(selectionfreeze,ensure_ascii=False,indent=2)+'\n')
    (out/'process/development_decision.json').write_text(json.dumps(dict(v1_verdict_preserved=True,v1_tradeoff_not_reclassified=True,
        new_test_numerics_seen=False,thresholds_unchanged=True,choices=channel_choice),indent=2)+'\n')
    print(json.dumps(dict(stage='v2_frozen',baseline=baseline,candidate=channel_choice)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--old',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();develop(a.old,a.output)
