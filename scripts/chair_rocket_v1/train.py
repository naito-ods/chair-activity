"""Fit ROCKET on private fixed-width tensors using case-grouped validation."""
import argparse,json,pickle,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import rocket_core as core

def run(a):
    out=a.output.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(a.metadata)
    required={'id','seq','annotation_id','confirmed','instrument_labels','treatment_normalized'}
    if required-set(meta.columns):raise ValueError('Missing required metadata fields')
    if not meta.id.is_unique:raise ValueError('Duplicate window IDs')
    if 'split' in meta and not meta.split.eq('development').all():raise ValueError('Training metadata contains non-development cases')
    arrays=np.load(a.tensors,allow_pickle=False)
    x=arrays['train']
    assert x.shape==(len(meta),len(core.CHANNELS),120)
    assert arrays['train_ids'].tolist()==meta.id.tolist()
    assert arrays['channels'].tolist()==core.CHANNELS
    core.SEED=a.seed;core.KERNELS=a.num_kernels
    variants=['time_series','mask_only'];models=[];folds=[];oofrows=[]
    fingerprint={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.metadata,a.tensors]}
    (out/'plan.json').write_text(json.dumps(dict(seed=a.seed,num_kernels=a.num_kernels,
        alpha=1.,cv='GroupKFold by case; no tuning',cv_folds=a.folds,
        actual_cases=int(meta.seq.nunique()),actual_windows=len(meta),
        source_fingerprint=fingerprint,variants=variants),indent=2)+'\n')
    for variant in variants:
        oof={task:['']*len(meta) for task in ['instrument','treatment']}
        for fold,(ti,vi) in enumerate(GroupKFold(a.folds).split(x,groups=meta.seq)):
            assert not set(meta.seq.iloc[ti])&set(meta.seq.iloc[vi])
            model=core.WeightedRocket().fit(x[ti],meta.iloc[ti].reset_index(drop=True),variant)
            for task,(labels,margins) in model.predict(x[vi]).items():
                for i,label in zip(vi,labels):oof[task][i]=label
            folds.extend(dict(variant=variant,fold=fold,id=meta.id.iloc[i],seq=int(meta.seq.iloc[i]),
                train_cases=json.dumps(sorted(map(int,meta.seq.iloc[ti].unique())))) for i in vi)
            print(json.dumps(dict(variant=variant,fold=fold,validation_cases=int(meta.seq.iloc[vi].nunique()))),flush=True)
        for task,pred in oof.items():
            models.append(dict(variant=variant,task=task,**core.score_predictions(meta,pred,task)))
            oofrows.extend(dict(variant=variant,task=task,id=r.id,seq=int(r.seq),predicted=p) for r,p in zip(meta.itertuples(),pred))
        final=core.WeightedRocket().fit(x,meta,variant)
        with (out/f'{variant}_model.pkl').open('wb') as f:pickle.dump(final,f)
    pd.DataFrame(models).to_csv(out/'development_oof_metrics.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'development_folds.csv',index=False)
    pd.DataFrame(oofrows).to_csv(out/'development_oof_predictions.csv',index=False)
    meta.to_csv(out/'training_windows.csv',index=False)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in fingerprint.items())
    print(json.dumps(models,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--tensors',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261105)
    p.add_argument('--num-kernels',type=int,default=10000)
    p.add_argument('--folds',type=int,default=5)
    with threadpool_limits(limits=4):run(p.parse_args())
