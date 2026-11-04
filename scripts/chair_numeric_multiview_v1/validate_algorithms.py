"""Small meaningful checks for the custom observed-only distance and medoid fit."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from cluster_observed_v2 import fit, observed_distance


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    x=np.r_[np.arange(6)*.01,10+np.arange(6)*.01]
    distance=np.abs(x[:,None]-x[None,:])
    truth=np.repeat([0,1],6)
    checks={}
    for method in ['kmedoids','average']:
        labels=fit(distance,method,2,42)
        checks['separated_'+method]=bool(adjusted_rand_score(truth,labels)==1)
    raw=np.column_stack([np.linspace(0,1,12),np.linspace(1,0,12),np.full(12,np.nan)])
    catalog=pd.DataFrame({'feature':['hp1','hp2','missing_numeric'],
        'view':['operations','operations','levels'],'code':['20011','20039','20742']})
    distance,shared,numeric,_=observed_distance(raw,catalog)
    checks.update(distance_symmetric=bool(np.allclose(distance,distance.T)),
        distance_zero_diagonal=bool(np.allclose(np.diag(distance),0)),
        distance_finite=bool(np.isfinite(distance).all()),
        unknown_numeric_not_fabricated=bool(np.all(numeric==0)),
        shared_support_count=bool(np.all(shared==2)))
    assert all(checks.values()),checks
    destination=args.output/'process/custom_algorithm_validation.json'
    destination.write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
