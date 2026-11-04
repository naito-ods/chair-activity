"""Unit tests with synthetic numeric sequences; no clinical validation implied."""
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

import chair_two_stage as mod


def rec(x,code='20742',duration=30.):
    d={}
    for c in ['20742','20743','20108']:
        raw=np.asarray(x,dtype=float) if c==code else np.array([])
        d[c]={'raw':raw,'norm':raw/10.,'t':np.arange(len(raw)),
              'coverage':len(raw)/30.,'n':len(raw)}
    return {'by_code':d,'duration':duration}


def main():
    assert np.isclose(mod.dtw_1d([1,2,3],[1,2,3],.25),0.)
    assert mod.dtw_1d([0,0,0],[3,3,3],.25)>0.
    a=rec([100,110,120,100]);b=rec([103,111,119,98]);c=rec([250,270,290,240])
    D,pairs=mod.calculate_numeric_matrix([a,b,c],['20742'],64,.25)
    assert D.shape==(3,3) and np.isfinite(D).all()
    assert np.allclose(D,D.T) and np.allclose(np.diag(D),0)
    assert D[0,1]<D[0,2],(D,pairs)
    class Args:
        min_subcluster_size=2
        min_subcluster_cases=2
        min_silhouette=.1
        subsample_repeats=3
        subsample_fraction=.8
        seed=42
    ids=np.array(['case'+str(i//2) for i in range(20)])
    rng=np.random.default_rng(8)
    X=np.r_[rng.normal(0,.01,(10,2)),rng.normal(1,.01,(10,2))]
    Ds=np.linalg.norm(X[:,None,:]-X[None,:,:],axis=2)
    np.fill_diagonal(Ds,0)
    df=pd.DataFrame({'case_id':ids})
    _,diag,k,labels=mod.score_partitions(Ds,df,4,Args)
    assert k is not None and labels is not None
    stability=mod.case_stability(Ds,ids,labels,k,Args)
    assert len(stability)>0
    assert (stability.ari_to_reference>=-.5).all()
    print('PASS: observed-only DTW, raw amplitude, symmetry, clustering, case subsampling')


if __name__=='__main__':main()
