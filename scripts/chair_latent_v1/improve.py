"""Recorded development revision: condition on levels, learn residual shape.

The initial common latent failed to beat the linear comparator. This revision
retains the same data/splits/thresholds and separates visible within-case
operating level from a learned low-dimensional variation of temporal shape.
No test case score is computed here.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from experiment import (SEED,coordinate_scale,record_arrays,case_errors,log,RidgeObserved,
    MaskedFactors,within_case_prediction,evaluate_candidate)
from latent_model import MaskedVAE

class KernelObserved(RidgeObserved):
    def __init__(self,alpha=.1,width=1.):super().__init__(alpha);self.width=width
    def kernel(self,a,b):return np.exp(-cdist(a,b,'sqeuclidean')/a.shape[1]/(2*self.width**2))
    def fit(self,f,y,target):
        self.center=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0 for v in f.T])
        self.scale=np.array([max(.1,np.std(v[np.isfinite(v)])) if np.isfinite(v).any() else 1 for v in f.T])
        self.a=self.design(f);k=self.kernel(self.a,self.a)
        self.coef=np.zeros((len(f),y.shape[1]))
        self.target_mean=np.array([np.mean(y[target[:,j],j]) if target[:,j].any() else 0 for j in range(y.shape[1])])
        for j in np.flatnonzero(target.any(0)):
            ix=np.flatnonzero(target[:,j]);self.coef[ix,j]=np.linalg.solve(k[np.ix_(ix,ix)]+self.alpha*np.eye(len(ix)),y[ix,j]-self.target_mean[j])
        return self
    def predict(self,f):return self.kernel(self.design(f),self.a)@self.coef+self.target_mean
    def save(self,p):np.savez_compressed(p,center=self.center,scale=self.scale,a=self.a,coef=self.coef,target_mean=self.target_mean,alpha=self.alpha,width=self.width)

def run(out):
    d=np.load(out/'data.npz');ledger=pd.read_csv(out/'case_ledger.csv')
    train=np.flatnonzero(ledger.split.eq('train'));val=np.flatnonzero(ledger.split.eq('validation'))
    x,_,_=coordinate_scale(d['raw'],train);xx,vis,tar=record_arrays(x,d['numeric_visible'])
    tr=(train[:,None]*4+np.arange(4)).ravel();vr=(val[:,None]*4+np.arange(4)).ravel()
    f=np.load(out/'semantic_inputs.npz')['f'];rows=pd.read_csv(out/'development_comparison.csv').to_dict('records');done={r['model'] for r in rows}
    def record(name,pred,**cfg):evaluate_candidate(out,name,pred,xx[vr],tar[vr],val,rows,cfg)
    mean=np.load(out/'models/population.npy')
    for width in [.5,1.,2.]:
        for alpha in [.1,1.]:
            name=f'semantic_kernel_w{width:g}_a{alpha:g}'
            if name in done:continue
            model=KernelObserved(alpha,width).fit(f[tr],xx[tr],tar[tr]);model.save(out/'models'/f'{name}.npz')
            record(name,model.predict(f[vr]),family='semantic_kernel',width=width,alpha=alpha,revision=3)
    valid=tar[vr].any(1);vi=vr[valid]
    # Exact case/channel validation metric, unaffected by how many bins a case has.
    def score(pred):
        full=np.full_like(xx[vr],np.nan);full[valid]=pred
        return float(np.nanmean(case_errors(full,xx[vr],tar[vr],len(val))[0]))
    weights=np.tile([.025,.025,.025,.025,1,1,1],48)
    for base,width in [('median',None),('local6',6.)]:
        bp=within_case_prediction(xx,vis,mean,width)
        residual=xx.copy();numeric=np.tile(np.arange(7)>=4,48)
        residual[:,numeric]-=bp[:,numeric]
        for k in [2,4,8]:
            name=f'conditional_linear_{base}_k{k}'
            if name not in done:
                w=np.sqrt(weights);model=MaskedFactors(k).fit(residual[tr]*w)
                model.save(out/'models'/f'{name}.npz')
                rpred=model.predict(residual[vr]*w,vis[vr])/w
                pred=rpred.copy();pred[:,numeric]+=bp[vr][:,numeric]
                record(name,pred,family='conditional_linear',base=base,latent_dim=k,revision=3)
            for beta in [.01,.05]:
                name=f'conditional_vae_{base}_k{k}_b{beta:g}'
                if name in done:continue
                def residual_score(pred):
                    final=pred.copy();final[:,numeric]+=bp[vi][:,numeric]
                    return score(final)
                model=MaskedVAE(latent_dim=k,hidden_dim=24,beta=beta,seed=SEED,epochs=800,lr=.002,patience=110,
                    feature_weights=weights,protected_features=~numeric,input_dropout=.05)
                model.fit(residual[tr],np.isfinite(residual[tr]),residual[vi],tar[vi],
                    input_mask=vis[tr],input_mask_val=vis[vi],validation_score_fn=residual_score)
                model.save(out/'models'/f'{name}.npz');pd.DataFrame(model.history).to_csv(out/'models'/f'{name}_history.csv',index=False)
                rpred=model.predict(residual[vr],vis[vr]);pred=rpred.copy();pred[:,numeric]+=bp[vr][:,numeric]
                record(name,pred,family='conditional_vae',base=base,latent_dim=k,beta=beta,best_epoch=model.best_epoch,revision=3)
    # Prespecified shrinkage blend: a hybrid, never claimed as a pure VAE win.
    tab=pd.DataFrame(rows)
    best_cond=tab[tab.family.eq('conditional_vae')].sort_values('validation_error').iloc[0]
    cv=np.load(out/'development_predictions'/f'{best_cond.model}.npz')['prediction']
    for base,width in [('local6',6.),('median',None)]:
        bp=within_case_prediction(xx[vr],vis[vr],mean,width)
        for mix in [.25,.5,.75]:
            name=f'hybrid_{base}_mix{mix:g}'
            if name in done:continue
            record(name,mix*cv+(1-mix)*bp,family='hybrid',base=base,mix=mix,latent_model=best_cond.model,revision=3)
    # Global low-rank and local evidence can carry complementary information.
    # This declared hybrid is evaluated against BOTH constituent baselines.
    best_linear=tab[tab.family.eq('linear')].sort_values('validation_error').iloc[0]
    lp=np.load(out/'development_predictions'/f'{best_linear.model}.npz')['prediction']
    for width in [6.,12.]:
        bp=within_case_prediction(xx[vr],vis[vr],mean,width)
        for mix in [.25,.5,.75]:
            name=f'linear_local_w{width:g}_mix{mix:g}'
            if name in done:continue
            record(name,mix*lp+(1-mix)*bp,family='linear_local',bandwidth=width,mix=mix,latent_model=best_linear.model,revision=3)
    log(out,'revision3_complete',n_candidates=len(rows),test_not_scored=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);run(ap.parse_args().output)
