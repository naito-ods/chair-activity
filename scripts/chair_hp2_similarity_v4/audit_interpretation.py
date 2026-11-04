"""Single-code dependence and equal-opportunity neighbor sensitivity."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from compare_similarity import distance,fit,nearest,SEED,log


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output.resolve()
    f=pd.read_csv(out/'features.csv').set_index('unit_id');u=pd.read_csv(out/'units.csv').set_index('unit_id')
    c=pd.read_csv(out/'feature_catalog.csv');cf=pd.read_csv(out/'case_features.csv').set_index('case_id')
    cc=pd.read_csv(out/'case_feature_catalog.csv');m=pd.read_csv(out/'models.csv');a=pd.read_csv(out/'assignments.csv')
    dependence=[]
    for level,raw,cat in [('block',f,c),('case',cf,cc)]:
        for model in m[(m.level==level)&(m.view=='numeric')].itertuples():
            reference=a[(a.model_id==model.model_id)&a.cluster.notna()].set_index('unit_id').cluster
            ids=reference.index;casearray=u.loc[ids,'case_id'].to_numpy(str) if level=='block' else ids.to_numpy(str)
            for signal in ['torque','pedal2']:
                cat1=cat[(cat.view=='numeric')&(cat.block==signal)]
                if level=='case':cat1=cat1[cat1.feature.str.endswith('__median')]
                d,_=distance(raw.loc[ids],cat1,casearray,'numeric')
                labels=fit(d,casearray,model.method,int(model.k),SEED)
                dependence.append(dict(model_id=model.model_id,level=level,signal_only=signal,n_units=len(ids),
                    n_cases=len(set(casearray)),ari_to_two_signal_partition=float(adjusted_rand_score(reference,labels))))
    pd.DataFrame(dependence).to_csv(out/'numeric_signal_dependence.csv',index=False)
    # A random one-block-per-case candidate pool removes more-blocks-more-chances.
    # Distances are fixed here, isolating candidate opportunity from scale refitting.
    results=[]
    for view in ['structure','numeric','combined']:
        frame=pd.read_csv(out/f'distances/block_{view}.csv',index_col=0);ids=frame.index.to_numpy();d=frame.to_numpy()
        cases=u.loc[ids,'case_id'].to_numpy(str);groups={case:np.flatnonzero(cases==case) for case in sorted(set(cases))}
        rng=np.random.default_rng(SEED+4000);scores={i:[] for i in range(len(ids))}
        baseline=[{cases[j] for j in nearest(d,cases,i)} for i in range(len(ids))]
        for iteration in range(100):
            chosen=np.array([rng.choice(group) for group in groups.values()],int)
            for i in range(len(ids)):
                valid=chosen[cases[chosen]!=cases[i]]
                top=valid[np.argsort(d[i,valid],kind='stable')[:3]]
                current={cases[j] for j in top};union=baseline[i]|current
                scores[i].append(len(baseline[i]&current)/len(union))
        for i,values in scores.items():
            results.append(dict(view=view,unit_id=ids[i],case_id=cases[i],repeats=100,
                mean_top3_jaccard=float(np.mean(values)),meaning='candidate one-per-case; frozen distance; different sensitivity from scale-refit neighbor stability'))
    pd.DataFrame(results).to_csv(out/'equal_opportunity_neighbor_sensitivity.csv',index=False)
    log(out,'interpretation_audit_complete',signal_dependence=dependence,
        opportunity_case_mean=pd.DataFrame(results).groupby(['view','case_id']).mean_top3_jaccard.mean().groupby('view').mean().to_dict())


if __name__=='__main__':main()
