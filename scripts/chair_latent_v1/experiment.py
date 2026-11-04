"""Frozen case splits, self-supervised numeric hiding, explicit baselines.

No annotation or pre-existing cluster is a learning target. Regression targets
are observed sensor summaries only. Model-selection uses validation cases;
final test access is a separate command after candidate choices are frozen.
"""
from pathlib import Path
import argparse, json, time, hashlib
import numpy as np
import pandas as pd
from scipy.linalg import svd
from latent_model import MaskedVAE
from prepare_data import SOURCE,CHANNELS,sha
from semantic_baseline import semantic_features

SEED=20261104

def log(out,event,**kw):
    row=dict(time=time.time(),event=event,**kw)
    with (out/'process/experiment_log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def coordinate_scale(raw,train):
    # Fixed invertible transforms, train-only numeric channel scales.
    x=raw.copy();x[:,:,4]/=40000
    for c in [5,6]:
        if np.nanmin(x[:,:,c])<0:raise ValueError('Unexpected negative torque/pedal')
        x[:,:,c]=np.log1p(x[:,:,c])
    center=np.zeros(7);scale=np.ones(7)
    for c in range(4,7):
        # Each case has equal influence on scale; no high-record case weighting.
        med=np.array([np.nanmedian(x[i,:,c]) if np.isfinite(x[i,:,c]).any() else np.nan for i in train])
        center[c]=np.nanmedian(med)
        scale[c]=max(.15,float(np.nanstd(med)))
    x=(x-center)/scale
    return x,center,scale

def record_arrays(x,numeric_visible):
    n,b,c=x.shape
    xx=np.repeat(x,4,axis=0).reshape(n*4,b*c)
    visible=np.isfinite(xx).reshape(n*4,b,c)
    visible[:,:,4:]&=numeric_visible.reshape(n*4,b,1)
    target=np.isfinite(xx).reshape(n*4,b,c)&~visible
    assert not (visible&target).any()
    return xx,visible.reshape(n*4,-1),target.reshape(n*4,-1)

def case_errors(pred,x,target,ncases):
    e=np.where(target,(pred-x)**2,np.nan).reshape(ncases,4,48,7)
    channels=np.full((ncases,3),np.nan)
    for i in range(ncases):
        for c in range(3):
            v=e[i,:,:,c+4];good=np.isfinite(v)
            if good.any():channels[i,c]=v[good].mean()
    return np.nanmean(channels,axis=1),channels

class RidgeObserved:
    """Independent target-coordinate ridge with observed-only training targets.

    Missing predictor coordinates are explicit training-mean model tokens plus
    mask indicators, not data imputation. Targets remain NaN/masked throughout.
    """
    def __init__(self,alpha=10):self.alpha=alpha
    def fit(self,f,y,target):
        self.center=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0 for v in f.T])
        self.scale=np.array([max(.1,np.std(v[np.isfinite(v)])) if np.isfinite(v).any() else 1 for v in f.T])
        a=self.design(f);self.coef=np.zeros((a.shape[1],y.shape[1]));self.support=target.sum(0)
        self.target_mean=np.array([np.mean(y[target[:,j],j]) if target[:,j].any() else 0 for j in range(y.shape[1])])
        for j in np.flatnonzero(target.any(0)):
            ix=target[:,j];aa=a[ix];yy=y[ix,j]-self.target_mean[j]
            if len(aa)<a.shape[1]:
                self.coef[:,j]=aa.T@np.linalg.solve(aa@aa.T+self.alpha*np.eye(len(aa)),yy)
            else:self.coef[:,j]=np.linalg.solve(aa.T@aa+self.alpha*np.eye(aa.shape[1]),aa.T@yy)
        return self
    def design(self,f):
        m=np.isfinite(f)
        return np.concatenate([np.where(m,(f-self.center)/self.scale,0),m.astype(float)],axis=1)
    def predict(self,f):return self.design(f)@self.coef+self.target_mean
    def save(self,p):np.savez_compressed(p,center=self.center,scale=self.scale,coef=self.coef,target_mean=self.target_mean,support=self.support,alpha=self.alpha)

class MaskedFactors:
    """Low-rank Gaussian-style conditional reconstruction, linear comparator.

    Missing training coordinates are iteratively MODEL-predicted; observed
    values and masks are never overwritten in exports. Conditional latent
    inference uses only coordinates visible in each query.
    """
    def __init__(self,k=4,iterations=35,regularization=.1):self.k=k;self.iterations=iterations;self.reg=regularization
    def fit(self,x):
        m=np.isfinite(x)
        self.mean=np.array([np.mean(v[np.isfinite(v)]) if np.isfinite(v).any() else 0 for v in x.T])
        estimate=np.where(m,x,self.mean)
        for it in range(self.iterations):
            u,s,v=svd(estimate-self.mean,full_matrices=False,check_finite=False)
            recon=(u[:,:self.k]*s[:self.k])@v[:self.k]+self.mean
            estimate=np.where(m,x,recon)
        self.loading=v[:self.k].T
        return self
    def transform(self,x,mask):
        z=[]
        for a,m in zip(x,mask):
            l=self.loading[m]
            z.append(np.linalg.solve(l.T@l+self.reg*np.eye(self.k),l.T@(a[m]-self.mean[m])))
        return np.array(z)
    def predict(self,x,mask):return self.transform(x,mask)@self.loading.T+self.mean
    def save(self,p):np.savez_compressed(p,mean=self.mean,loading=self.loading,k=self.k,reg=self.reg)

def within_case_prediction(xx,visible,train_mean,bandwidth=None,shrink=0.):
    """Transparent observed-value predictor; output is model estimate only.

    Conditional sensor levels are an essential strong non-neural control.
    No numerical observation is fabricated or exported into a source table.
    """
    prediction=np.broadcast_to(train_mean,xx.shape).copy().reshape(-1,48,7)
    a=xx.reshape(-1,48,7);m=visible.reshape(-1,48,7)
    for i in range(len(a)):
        for c in range(4,7):
            ix=np.flatnonzero(m[i,:,c])
            if not len(ix):continue
            for t in range(48):
                if bandwidth is None:v=np.median(a[i,ix,c])
                else:
                    w=np.exp(-.5*((ix-t)/bandwidth)**2)
                    if w.sum()<1e-10:v=np.median(a[i,ix,c])
                    else:v=np.sum(w*a[i,ix,c])/w.sum()
                prediction[i,t,c]=(1-shrink)*v+shrink*prediction[i,t,c]
    return prediction.reshape(xx.shape)

def prepare_features(out,ledger,visible):
    cache=out/'semantic_inputs.npz'
    fingerprint=hashlib.sha256((''.join(sha(p) for p in [out/'data.npz',SOURCE/'observations.csv',
        SOURCE/'hp2_points.csv',SOURCE/'hp_intervals.csv',SOURCE/'hp_runs.csv',out/'case_manifest.csv',
        Path(__file__).parent/'semantic_baseline.py'])).encode()).hexdigest()
    if cache.exists():
        d=np.load(cache)
        if str(d.get('version',''))!='per_signal_actual_timestamps_v2':raise ValueError('Incompatible semantic cache version')
        if str(d.get('fingerprint',''))!=fingerprint:raise ValueError('Semantic cache input/code fingerprint mismatch; archive and regenerate explicitly')
        return d['f'],d['feature_names'].tolist()
    manifest=pd.read_csv(out/'case_manifest.csv')
    intervals=pd.read_csv(SOURCE/'hp_intervals.csv');runs=pd.read_csv(SOURCE/'hp_runs.csv')
    points=pd.read_csv(SOURCE/'hp2_points.csv',low_memory=False)
    timestamps=pd.read_csv(SOURCE/'observations.csv',usecols=['case_id','code','time_s','run_id','second_bin','hp'],dtype={'code':str})
    timestamps=timestamps[timestamps.hp.eq(2)&timestamps.code.isin(['20742','20743','10015'])]
    keys=['case_id','run_id','second_bin']
    for code,signal in [('20742','rpm'),('20743','torque'),('10015','pedal2')]:
        t=timestamps[timestamps.code.eq(code)][keys+['time_s']].rename(columns={'time_s':signal+'_time_s'})
        assert not t.duplicated(keys).any()
        points=points.merge(t,on=keys,how='left',validate='one_to_one')
    fs=[];names=None
    for rep in range(4):
        f,nn=semantic_features(manifest,intervals,runs,points,visible[:,rep],bins=48)
        if names is not None:assert names==nn
        names=nn;fs.append(f)
    f=np.stack(fs,axis=1).reshape(len(ledger)*4,-1)
    np.savez_compressed(cache,f=f,feature_names=np.array(names),version='per_signal_actual_timestamps_v2',fingerprint=fingerprint)
    (out/'semantic_feature_catalog.json').write_text(json.dumps(names,indent=2)+'\n')
    return f,names

def evaluate_candidate(out,name,pred,xx,tt,case_ix,rows,config):
    ee,cc=case_errors(pred,xx,tt,len(case_ix))
    r=dict(model=name,validation_error=float(np.nanmean(ee)),n_cases=len(case_ix),**config)
    for j,c in enumerate(CHANNELS[4:]):r[c+'_error']=float(np.nanmean(cc[:,j]));r[c+'_supported_cases']=int(np.isfinite(cc[:,j]).sum())
    rows.append(r)
    np.savez_compressed(out/'development_predictions'/f'{name}.npz',prediction=pred,case_errors=ee,channel_errors=cc,case_indices=case_ix)
    log(out,'candidate_evaluated',**r)
    pd.DataFrame(rows).to_csv(out/'development_comparison.csv',index=False)

def develop(out,revision):
    d=np.load(out/'data.npz');ledger=pd.read_csv(out/'case_ledger.csv')
    train=np.flatnonzero(ledger.split.eq('train'));val=np.flatnonzero(ledger.split.eq('validation'))
    test=np.flatnonzero(ledger.split.eq('test'))
    assert not set(train)&set(val) and not set(train)&set(test) and not set(val)&set(test)
    x,center,scale=coordinate_scale(d['raw'],train);np.savez(out/'scaling.npz',center=center,scale=scale)
    xx,vis,tar=record_arrays(x,d['numeric_visible']);trows=(train[:,None]*4+np.arange(4)).ravel();vrows=(val[:,None]*4+np.arange(4)).ravel()
    f,names=prepare_features(out,ledger,d['numeric_visible'])
    (out/'models').mkdir(exist_ok=True);(out/'development_predictions').mkdir(exist_ok=True)
    rows=pd.read_csv(out/'development_comparison.csv').to_dict('records') if (out/'development_comparison.csv').exists() else []
    done={r['model'] for r in rows}
    def record(name,pred,**cfg):evaluate_candidate(out,name,pred,xx[vrows],tar[vrows],val,rows,cfg)
    # Base training profile, only a calibration control.
    if 'population' not in done:
        mean=np.array([np.nanmean(v) if np.isfinite(v).any() else 0 for v in x[train].reshape(len(train),-1).T])
        np.save(out/'models/population.npy',mean);record('population',np.broadcast_to(mean,xx[vrows].shape),family='population',revision=1)
    mean=np.load(out/'models/population.npy')
    for width in [None,2.,6.,12.]:
        name='case_median' if width is None else f'local_kernel_w{width:g}'
        if name not in done:record(name,within_case_prediction(xx[vrows],vis[vrows],mean,width),family='within_case',bandwidth=width,revision=1)
    # Interpretable exact HP/order/RPM summaries; include known support covariates.
    for family,features in [('semantic',f),('raw_ridge',np.where(vis,xx,np.nan)),('mask_HP',np.where(vis&np.tile(np.arange(7)<4,48),xx,np.nan))]:
        if family=='mask_HP':features=np.concatenate([features,vis.astype(float)],axis=1)
        for alpha in [1.,10.,100.]:
            name=f'{family}_a{alpha:g}'
            if name in done:continue
            model=RidgeObserved(alpha).fit(features[trows],xx[trows],tar[trows])
            model.save(out/'models'/f'{name}.npz');record(name,model.predict(features[vrows]),family=family,alpha=alpha,revision=1)
    # Weighted per-channel scales already train-fitted; HP family weighted 1/4.
    weights=np.tile([.25,.25,.25,.25,1,1,1],48)
    sqrt=np.sqrt(weights)
    for k in [2,4,8,16]:
        name=f'linear_k{k}'
        if name in done:continue
        model=MaskedFactors(k).fit(x[train].reshape(len(train),-1)*sqrt)
        model.save(out/'models'/f'{name}.npz')
        record(name,model.predict(xx[vrows]*sqrt,vis[vrows])/sqrt,family='linear',latent_dim=k,revision=1)
    combos=[(k,b,h) for k in [2,4,8] for b in [.001,.01] for h in [32]]
    if revision>=2:combos += [(4,0.,32),(8,0.,32),(4,.05,24),(8,.001,64)]
    valid=tar[vrows].any(1)
    for k,b,h in combos:
        name=f'{"vae" if b else "ae"}_k{k}_b{b:g}_h{h}'
        if name in done:continue
        model=MaskedVAE(latent_dim=k,hidden_dim=h,beta=b,seed=SEED,epochs=650,lr=.002,patience=90,
            feature_weights=weights,protected_features=np.tile(np.arange(7)<4,48),input_dropout=.1)
        # Four train views per case; full observed targets, incomplete encoder.
        model.fit(xx[trows],np.isfinite(xx[trows]),xx[vrows][valid],tar[vrows][valid],
            input_mask=vis[trows],input_mask_val=vis[vrows][valid])
        model.save(out/'models'/f'{name}.npz');pd.DataFrame(model.history).to_csv(out/'models'/f'{name}_history.csv',index=False)
        record(name,model.predict(xx[vrows],vis[vrows]),family='vae' if b else 'ae',latent_dim=k,beta=b,hidden_dim=h,best_epoch=model.best_epoch,revision=1 if (k,b,h) in combos[:6] else 2)
    log(out,'development_complete',revision=revision,candidates=len(rows),test_cases_not_scored=len(test))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--revision',type=int,default=1)
    a=ap.parse_args();develop(a.output,a.revision)
if __name__=='__main__':main()
