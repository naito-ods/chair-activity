"""Small, separately validated weak-window classifiers; no background labels.

Window scores are inherited-label agreement, not point-verified detection.
All windows from a case stay in one fold; weights equalize annotation parents.
Freeze this task BEFORE reading the shared chronological final case results.
"""
from pathlib import Path
import argparse,json,traceback,warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from learning import (estimators,feature_table,target,supports,Bundle,metrics,row_weights,
    log,freeze,evaluate)

def develop(out,revision):
    if (out/'selection_freeze.json').exists():raise RuntimeError('Window development frozen')
    (out/'models').mkdir(exist_ok=True);(out/'development_predictions').mkdir(exist_ok=True)
    models=estimators();rows=pd.read_csv(out/'development_comparison.csv').to_dict('records') if (out/'development_comparison.csv').exists() else []
    done={r['model'] for r in rows}
    plan={1:{'duration_only':['top1_prior','logistic_C01'],
             'mask_only':['logistic_C01','extra_leaf2'],
             'basic':['logistic_C01','logistic_C1','extra_leaf2','extra_nobalance_leaf3']},
          2:{'extended':['logistic_C01','logistic_C1','extra_leaf2','extra_nobalance_leaf3']},
          3:{'extended_no_quality':['logistic_C01','logistic_C1'],
             'global_context':['logistic_C01','logistic_C1']}}
    for task in ['instrument','treatment']:
        for view,algorithms in plan[revision].items():
            meta,x,cols=feature_table(out,view);y,mask,classes=target(meta,task)
            ix=np.flatnonzero(mask&meta.split.eq('development').to_numpy());groups=meta.seq.to_numpy()[ix]
            support=supports(meta,y,mask&meta.split.eq('development').to_numpy(),task,classes)
            pd.DataFrame(support).to_csv(out/f'{task}_development_label_support.csv',index=False)
            supported=[r['label'] for r in support if r['n_cases']>=3]
            for algorithm in algorithms:
                name=f'{task}__{view}__{algorithm}'
                if name in done:continue
                prediction=np.full((len(ix),len(classes)),np.nan);audit=[]
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    for fold,(ti,vi) in enumerate(GroupKFold(5).split(x[ix],groups=groups)):
                        tr=ix[ti];va=ix[vi];assert not set(meta.seq.iloc[tr])&set(meta.seq.iloc[va])
                        model=Bundle(task,models[algorithm],cols,classes).fit(x[tr],y[tr],meta.seq.to_numpy()[tr],row_weights(meta,tr))
                        prediction[vi]=model.predict_scores(x[va])
                        audit.extend(dict(id=meta.id.iloc[r],annotation_id=meta.annotation_id.iloc[r],seq=int(meta.seq.iloc[r]),fold=fold) for r in va)
                assert np.isfinite(prediction).all()
                result=dict(model=name,task=task,view=view,algorithm=algorithm,revision=revision,n_features=len(cols),
                    **metrics(y[ix],prediction,groups,task,classes,supported,row_weights(meta,ix)))
                rows.append(result);pd.DataFrame(rows).to_csv(out/'development_comparison.csv',index=False)
                np.savez_compressed(out/'development_predictions'/f'{name}.npz',scores=prediction,indices=ix,classes=np.array(classes),y=y[ix],groups=groups)
                pd.DataFrame(audit).to_csv(out/'development_predictions'/f'{name}_folds.csv',index=False)
                if captured:(out/'process'/f'{name}_warnings.json').write_text(json.dumps([str(w.message) for w in captured],ensure_ascii=False,indent=2))
                log(out,'window_development_model',**result)
    log(out,'window_development_revision_complete',revision=revision,n_models=len(rows),final_cases_scored=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['develop','freeze','evaluate']);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--revision',type=int,default=1);a=ap.parse_args()
    try:
        if a.stage=='develop':develop(a.output,a.revision)
        else:globals()[a.stage](a.output)
    except Exception as exc:
        (a.output/'process/latest_error.json').write_text(json.dumps({'stage':a.stage,'error':repr(exc),'traceback':traceback.format_exc()},ensure_ascii=False,indent=2));raise
