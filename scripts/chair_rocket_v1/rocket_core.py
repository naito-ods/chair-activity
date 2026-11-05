"""Multivariate ROCKET with explicit missingness and case-weighted Ridge.

No patient data, selected case IDs, or local reports are embedded in this module.
Numeric input tensors preserve NaN; model encoding uses a sentinel and mask.
"""
import os
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/chair_rocket_numba')
os.environ.setdefault('NUMBA_NUM_THREADS','4')
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
from sktime.transformations.rocket import Rocket
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'chair_supervised_v1'))
from features import state_segments
from learning import row_weights


SEED=20261105

KERNELS=10000

STATE={'20039':[0,1,2,3],'20011':[0,1,2,3],'20101':[0,1],'20774':[0,1]}

NUM=['20742','20743','20108','10014','10015']

CHANNELS=[f'{c}_state_{v}' for c,vs in STATE.items() for v in vs+['unknown']]

CHANNELS += [f'{c}_observed_median' for c in NUM]+[f'{c}_observed_mask' for c in NUM]

NS=sum(len(v)+1 for v in STATE.values())

def tensor(events,start):
    """120 one-second bins; state duration fractions, numeric observed medians."""
    arr=np.full((len(CHANNELS),120),np.nan,dtype=np.float32)
    offset=0
    edges=start+np.arange(121,dtype=float)
    for code,values in STATE.items():
        state=state_segments(events,code,start,start+120)
        occupancy=np.zeros((len(values)+1,120),np.float64)
        for a,b,v in state:
            widths=np.maximum(0,np.minimum(edges[1:],b)-np.maximum(edges[:-1],a))
            index=values.index(int(v)) if np.isfinite(v) else len(values)
            occupancy[index]+=widths
        np.testing.assert_allclose(occupancy.sum(axis=0),1,atol=1e-8)
        arr[offset:offset+len(values)+1]=occupancy
        offset+=len(values)+1
    for j,code in enumerate(NUM):
        q=events[events.code.eq(code)&events.time_s.ge(start)&events.time_s.lt(start+120)&events.value.notna()]
        bins=np.floor(q.time_s.to_numpy(float)-start).astype(int)
        if len(q):
            med=pd.Series(q.value.to_numpy(float)).groupby(bins).median()
            arr[NS+j,med.index.to_numpy(int)]=med.to_numpy(float)
        arr[NS+len(NUM)+j]=np.isfinite(arr[NS+j]).astype(np.float32)
    return arr

class InputEncoding:
    """Fit numeric centering/scaling only on training folds; retain missing mask."""
    def fit(self,x):
        self.center=[];self.scale=[];self.all_missing=[]
        for j in range(len(NUM)):
            v=x[:,NS+j,:];v=v[np.isfinite(v)]
            self.all_missing.append(not len(v))
            center=float(np.median(v)) if len(v) else 0.
            scale=float(np.quantile(v,.75)-np.quantile(v,.25)) if len(v) else 1.
            if scale<=1e-12:scale=float(np.ptp(v)) if len(v) else 1.
            if scale<=1e-12:scale=1.
            self.center.append(center);self.scale.append(scale)
        return self
    def transform(self,x,variant):
        if variant=='mask_only':
            rows=[];i=0
            for values in STATE.values():
                rows.append(1-x[:,i+len(values),:]);i+=len(values)+1
            rows.extend(x[:,NS+len(NUM)+j,:] for j in range(len(NUM)))
            ans=np.stack(rows,axis=1)
        else:
            ans=x.copy()
            for j in range(len(NUM)):
                v=x[:,NS+j,:]
                # -3 is an explicit model-input missing token, never observed OFF/0.
                ans[:,NS+j,:]=np.where(np.isfinite(v),(v-self.center[j])/self.scale[j],-3.)
        assert np.isfinite(ans).all()
        return np.ascontiguousarray(ans,dtype=np.float32)

class WeightedRocket:
    """Official random temporal kernels + standardization + weighted ridge."""
    def fit(self,x,meta,variant):
        self.variant=variant
        self.encoder=InputEncoding().fit(x)
        xx=self.encoder.transform(x,variant)
        self.transformer=Rocket(num_kernels=KERNELS,normalise=False,n_jobs=4,random_state=SEED)
        z=np.asarray(self.transformer.fit_transform(xx),dtype=np.float64)
        assert z.shape==(len(meta),2*KERNELS) and np.isfinite(z).all()
        self.scaler=StandardScaler(with_mean=False).fit(z)
        z=self.scaler.transform(z)
        self.models={};self.classes={};self.train_cases={}
        for task in ['instrument','treatment']:
            valid=meta.confirmed.astype(bool).to_numpy()
            if task=='instrument':
                labs=[json.loads(v) for v in meta.instrument_labels]
                valid &= np.array([bool(a) for a in labs])
                classes=sorted({v for i,a in enumerate(labs) if valid[i] for v in a})
                yy=np.array([[int(c in a) for c in classes] for a in labs])
            else:
                yy=meta.treatment_normalized.fillna('').astype(str).to_numpy()
                valid &= yy!='';classes=sorted(set(yy[valid]))
            ix=np.flatnonzero(valid);weights=row_weights(meta,ix)
            self.classes[task]=classes
            self.train_cases[task]=sorted(map(int,meta.seq.iloc[ix].unique()))
            fitted=[]
            targets=[yy[ix,j] for j in range(len(classes))] if task=='instrument' else [yy[ix]]
            for target in targets:
                unique=np.unique(target)
                if len(unique)==1:fitted.append(('constant',unique[0]))
                else:
                    m=RidgeClassifier(alpha=1.,class_weight='balanced',solver='auto',tol=1e-4)
                    m.fit(z[ix],target,sample_weight=weights)
                    fitted.append(('ridge',m))
            self.models[task]=fitted
        return self
    def predict(self,x):
        z=np.asarray(self.transformer.transform(self.encoder.transform(x,self.variant)),dtype=np.float64)
        z=self.scaler.transform(z)
        result={}
        for task,models in self.models.items():
            if task=='instrument':
                decisions=[]
                for kind,m in models:
                    decisions.append(np.full(len(x),1. if m==1 else -1.) if kind=='constant' else m.decision_function(z))
                scores=np.column_stack(decisions)
                predicted=['|'.join(c for c,s in zip(self.classes[task],row) if s>=0) for row in scores]
            else:
                kind,m=models[0]
                if kind=='constant':
                    scores=np.ones((len(x),1));predicted=[str(m)]*len(x)
                else:
                    scores=m.decision_function(z)
                    if scores.ndim==1:scores=np.column_stack([-scores,scores])
                    assert m.classes_.tolist()==self.classes[task]
                    predicted=np.array(self.classes[task])[scores.argmax(axis=1)].tolist()
            result[task]=(predicted,scores)
        return result

def score_predictions(meta,pred,task):
    good=meta.confirmed.astype(bool).to_numpy()
    if task=='instrument':
        truth=['|'.join(sorted(json.loads(v))) for v in meta.instrument_labels]
        good &= np.array([bool(json.loads(v)) for v in meta.instrument_labels])
    else:
        truth=meta.treatment_normalized.fillna('').astype(str).tolist();good &= np.array(truth)!=''
    ix=np.flatnonzero(good);w=row_weights(meta,ix)
    actual=[truth[i] for i in ix];guessed=[pred[i] for i in ix]
    exact=np.array([set(a.split('|'))==set(b.split('|')) for a,b in zip(actual,guessed)])
    if task=='instrument':
        labels=sorted({v for a in actual for v in a.split('|') if v})
        a=np.array([[int(c in v.split('|')) for c in labels] for v in actual]);b=np.array([[int(c in v.split('|')) for c in labels] for v in guessed])
        f1=f1_score(a,b,average='macro',sample_weight=w,zero_division=0)
    else:f1=f1_score(actual,guessed,labels=sorted(set(actual)),average='macro',sample_weight=w,zero_division=0)
    return dict(accuracy=float(np.average(exact,weights=w)),macro_f1=float(f1),n_cases=int(meta.iloc[ix].seq.nunique()),n_windows=len(ix))
