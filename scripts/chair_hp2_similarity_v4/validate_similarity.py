"""Synthetic checks for case weighting, missingness and distinct-case retrieval."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from compare_similarity import distance,fit,nearest,quantile


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    raw=pd.DataFrame({'a':np.arange(8,dtype=float),'b':[0.,1.,1.,2.,3.,5.,7.,9.]})
    cat=pd.DataFrame({'feature':['a','b'],'view':['structure']*2,'block':['a','b'],'transform':['log1p','identity']})
    cases=np.repeat(['A','B','C','D'],2);d,_=distance(raw,cat,cases,'structure')
    extended=pd.concat([raw]+[raw.iloc[:2]]*4,ignore_index=True);ecases=np.r_[cases,['A']*8]
    other,_=distance(extended,cat,ecases,'structure')
    checks=dict(case_weight_duplicate_invariant=bool(np.allclose(d,other[:8,:8],atol=1e-12)),
        symmetric=bool(np.allclose(d,d.T)),diagonal_zero=bool(np.allclose(np.diag(d),0)),
        finite=bool(np.isfinite(d).all()),neighbor_cases_distinct=len({cases[i] for i in nearest(d,cases,0)})==3,
        neighbor_own_case_excluded=all(cases[i]!='A' for i in nearest(d,cases,0)),
        cdf_boundary_precision=quantile(np.arange(10,dtype=float),np.full(10,.1),.5)==4.)
    x=np.r_[np.arange(6)*.01,10+np.arange(6)*.01];matrix=np.abs(x[:,None]-x[None,:]);truth=np.repeat([0,1],6)
    for method in ['kmedoids','average']:
        labels=fit(matrix,np.array([str(i) for i in range(12)]),method,2,42)
        checks['separated_'+method]=bool(adjusted_rand_score(truth,labels)==1)
    missing=raw.copy();missing.loc[0,'a']=np.nan;original=missing.copy();md,_=distance(missing,cat,cases,'structure')
    checks['missing_not_imputed']=missing.equals(original) and np.isnan(missing.loc[0,'a']) and np.isfinite(md).all()
    checks={key:bool(value) for key,value in checks.items()};assert all(checks.values()),checks
    (out/'process/synthetic_validation.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
