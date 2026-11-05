"""Small classifiers, group-wise validation and an immutable final holdout.

Only log-derived numeric features enter models. Human instrument labels are
targets, never treatment predictors. Unannotated time is not background data.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,pickle,traceback,warnings,sys
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score,precision_recall_fscore_support,accuracy_score
from scipy.special import softmax

SEED=20261105
sys.modules.setdefault('learning',sys.modules[__name__])

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def log(out,event,**kwargs):
    r=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,**kwargs)
    with (out/'process/learning_log.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps(r,ensure_ascii=False),flush=True)

def case_weights(groups):
    _,inverse,count=np.unique(groups,return_inverse=True,return_counts=True)
    w=1/count[inverse];return w/w.mean()

def row_weights(meta,indices):
    """Equal cases; within a case equal annotated segments; then equal windows."""
    q=meta.iloc[np.asarray(indices)].copy()
    if 'annotation_id' not in q:return case_weights(q.seq.to_numpy())
    per_annotation=q.groupby(['seq','annotation_id']).id.transform('size').to_numpy(float)
    parents=q.groupby('seq').annotation_id.transform('nunique').to_numpy(float)
    w=1/(per_annotation*parents);return w/w.mean()

def estimators():
    # Fixed small search; there are no learned parameters shared across folds.
    im=lambda:SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)
    return {
      'dummy':make_pipeline(im(),DummyClassifier(strategy='prior')),
      'top1_prior':make_pipeline(im(),DummyClassifier(strategy='most_frequent')),
      'logistic_C01':make_pipeline(im(),StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,random_state=SEED)),
      'logistic_C1':make_pipeline(im(),StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=SEED)),
      'logistic_C10':make_pipeline(im(),StandardScaler(),LogisticRegression(C=10.,class_weight='balanced',max_iter=2000,random_state=SEED)),
      'logistic_nobalance_C01':make_pipeline(im(),StandardScaler(),LogisticRegression(C=.1,class_weight=None,max_iter=2000,random_state=SEED)),
      'logistic_nobalance_C1':make_pipeline(im(),StandardScaler(),LogisticRegression(C=1.,class_weight=None,max_iter=2000,random_state=SEED)),
      'extra_nobalance_leaf3':make_pipeline(im(),ExtraTreesClassifier(n_estimators=160,min_samples_leaf=3,max_features=.7,class_weight=None,n_jobs=4,random_state=SEED)),
      'gaussian_smooth01':make_pipeline(im(),StandardScaler(),GaussianNB(var_smoothing=.1)),
      'hist_small':make_pipeline(im(),HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=5,min_samples_leaf=5,l2_regularization=3.,random_state=SEED)),
      'extra_leaf2':make_pipeline(im(),ExtraTreesClassifier(n_estimators=160,min_samples_leaf=2,max_features=.7,class_weight='balanced',n_jobs=4,random_state=SEED)),
      'forest_depth4':make_pipeline(im(),RandomForestClassifier(n_estimators=160,max_depth=4,min_samples_leaf=2,class_weight='balanced',n_jobs=4,random_state=SEED)),
      'pca8_logistic':make_pipeline(im(),StandardScaler(),PCA(n_components=8,random_state=SEED),LogisticRegression(C=.3,class_weight='balanced',max_iter=2000,random_state=SEED))}

def fit_one(template,x,y,groups,sample_weight=None):
    if len(np.unique(y))==1:
        m=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),DummyClassifier(strategy='constant',constant=y[0]))
    else:m=clone(template)
    key=m.steps[-1][0]+'__sample_weight'
    m.fit(x,y,**{key:case_weights(groups) if sample_weight is None else sample_weight})
    return m

def probabilities(model,x,classes):
    p=np.zeros((len(x),len(classes)));q=model.predict_proba(x)
    actual=model.steps[-1][1].classes_
    for i,c in enumerate(actual):
        if c in classes:p[:,classes.index(c)]=q[:,i]
    return p

class Bundle:
    """Saved model(s), with a fixed feature schema and target label order."""
    def __init__(self,task,template,features,classes):self.task=task;self.template=template;self.features=features;self.classes=classes
    def fit(self,x,y,groups,sample_weight=None):
        self.top1_prior=None
        last=self.template.steps[-1][1]
        if self.task=='instrument' and isinstance(last,DummyClassifier) and last.strategy=='most_frequent':
            weights=case_weights(groups) if sample_weight is None else sample_weight
            self.top1_prior=int(np.argmax(np.sum(y*weights[:,None],axis=0)))
        if self.task=='treatment':self.models=[fit_one(self.template,x,y,groups,sample_weight)]
        else:self.models=[fit_one(self.template,x,y[:,j],groups,sample_weight) for j in range(y.shape[1])]
        return self
    def predict_scores(self,x):
        if self.top1_prior is not None:
            p=np.zeros((len(x),len(self.classes)));p[:,self.top1_prior]=1.;return p
        if self.task=='treatment':return probabilities(self.models[0],x,self.classes)
        return np.column_stack([probabilities(m,x,[0,1])[:,1] for m in self.models])
    def save(self,p):
        with Path(p).open('wb') as f:pickle.dump(self,f)

Bundle.__module__='learning'

def target(meta,task):
    if task=='treatment':
        labels=meta.treatment_normalized.fillna('').astype(str).to_numpy();mask=meta.confirmed.astype(bool).to_numpy()&(labels!='')
        classes=sorted(set(labels[mask]));y=labels
    else:
        lab=[json.loads(v) if isinstance(v,str) and v else [] for v in meta.instrument_labels]
        classes=sorted({s for a in lab for s in a});y=np.array([[int(c in a) for c in classes] for a in lab],int)
        mask=meta.confirmed.astype(bool).to_numpy()&np.array([bool(a) for a in lab])
    return y,mask,classes

def metrics(y,p,groups,task,classes,supported,sample_weight=None):
    w=case_weights(groups) if sample_weight is None else sample_weight
    if task=='treatment':
        pred=np.array(classes)[p.argmax(1)];present=sorted(set(y));sl=[c for c in supported if c in set(y)]
        return dict(accuracy=float(accuracy_score(y,pred,sample_weight=w)),
            macro_f1=float(f1_score(y,pred,labels=present,average='macro',sample_weight=w,zero_division=0)),
            supported_macro_f1=float(f1_score(y,pred,labels=sl,average='macro',sample_weight=w,zero_division=0)) if sl else np.nan,
            micro_f1=float(f1_score(y,pred,average='micro',sample_weight=w,zero_division=0)),n_intervals=len(y),n_cases=len(set(groups)))
    pred=p>=.5;present=np.flatnonzero(y.sum(0)>0);sl=[classes.index(c) for c in supported if classes.index(c) in present]
    return dict(accuracy=float(accuracy_score(y,pred,sample_weight=w)),
        macro_f1=float(f1_score(y[:,present],pred[:,present],average='macro',sample_weight=w,zero_division=0)),
        supported_macro_f1=float(f1_score(y[:,sl],pred[:,sl],average='macro',sample_weight=w,zero_division=0)) if sl else np.nan,
        micro_f1=float(f1_score(y,pred,average='micro',sample_weight=w,zero_division=0)),n_intervals=len(y),n_cases=len(set(groups)))

def supports(meta,y,mask,task,classes):
    rows=[]
    for j,c in enumerate(classes):
        own=mask&((y==c) if task=='treatment' else (y[:,j]==1))
        rows.append(dict(label=c,n_intervals=int(own.sum()),n_cases=int(meta.loc[own,'seq'].nunique())))
    return rows

def feature_table(out,view):
    if view=='global_context':table=pd.read_csv(out/'features_context.csv')
    elif view in ['basic','duration_only','mask_only']:table=pd.read_csv(out/'features_basic.csv')
    else:table=pd.read_csv(out/'features_extended.csv')
    meta=pd.read_csv(out/'annotations_aligned.csv')
    assert table.id.tolist()==meta.id.tolist() and meta.id.is_unique
    # Export contract: the ONLY non-feature column is id.
    cols=[c for c in table if c!='id']
    x=table[cols].to_numpy(float)
    if view=='duration_only':
        cols=[c for c in cols if c in ['duration_s','duration_sec']]
        if not cols:raise ValueError('Missing duration feature')
        x=table[cols].to_numpy(float)
    if view=='mask_only':x=np.isfinite(x).astype(float);cols=['finite__'+c for c in cols]
    if view=='extended_no_quality':
        keep=[i for i,c in enumerate(cols) if not (c.startswith(('count_','rate_','quality_')) or
            any(t in c for t in ['event_count','n_obs','n_events','known_fraction','unknown_fraction','missing','coverage','observed_second_count']) or
            (c.startswith('num_') and c.endswith('_n')))]
        x=x[:,keep];cols=[cols[i] for i in keep]
    if view=='instrument_core':
        # Fixed sensor families, not human instrument/treatment labels.
        keep=[i for i,c in enumerate(cols) if any(t in c for t in ['20039','20011','20101','20774','20776','20742','hp','rpm','duration'])]
        x=x[:,keep];cols=[cols[i] for i in keep]
    assert np.isfinite(x).any() and not np.isinf(x).any()
    return meta,x,cols

def develop(out,revision):
    if (out/'selection_freeze.json').exists():raise RuntimeError('Interval final evaluation already frozen; use a separate version')
    (out/'models').mkdir(exist_ok=True);(out/'development_predictions').mkdir(exist_ok=True)
    summary=pd.read_csv(out/'development_comparison.csv').to_dict('records') if (out/'development_comparison.csv').exists() else []
    done={r['model'] for r in summary};templates=estimators()
    views=['global_context'] if revision==6 else (['duration_only'] if revision==0 else (['duration_only','mask_only','basic'] if revision==1 else (['extended'] if revision==2 else (['extended_no_quality','instrument_core'] if revision==3 else ['basic','extended']))))
    for task in ['instrument','treatment']:
        for view in views:
            meta,x,cols=feature_table(out,view);y,mask,classes=target(meta,task)
            dev=mask&meta.split.eq('development').to_numpy();indices=np.flatnonzero(dev);groups=meta.seq.to_numpy()[dev]
            support=supports(meta,y,dev,task,classes);supported=[r['label'] for r in support if r['n_cases']>=3]
            pd.DataFrame(support).to_csv(out/f'{task}_development_label_support.csv',index=False)
            chosen=['dummy','top1_prior','logistic_C01','extra_leaf2'] if view=='duration_only' else (['extra_leaf2','logistic_C01'] if view=='mask_only' else [k for k in templates if k!='top1_prior' and (revision>=4 or 'nobalance' not in k) and (revision>=5 or k not in ['gaussian_smooth01','hist_small'])])
            for algorithm in chosen:
                name=f'{task}__{view}__{algorithm}'
                if name in done:continue
                prediction=np.full((len(indices),len(classes)),np.nan);fold_audit=[]
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    for fold,(ti,vi) in enumerate(GroupKFold(5).split(x[dev],groups=groups)):
                        tr=indices[ti];va=indices[vi];assert not set(meta.seq.iloc[tr])&set(meta.seq.iloc[va])
                        model=Bundle(task,templates[algorithm],cols,classes).fit(x[tr],y[tr],meta.seq.iloc[tr].to_numpy(),row_weights(meta,tr))
                        prediction[vi]=model.predict_scores(x[va])
                        fold_audit.extend(dict(id=meta.id.iloc[v],seq=int(meta.seq.iloc[v]),fold=fold,
                            training_label_available=bool(y[v] in set(y[tr])) if task=='treatment' else True) for v in va)
                assert np.isfinite(prediction).all()
                row=dict(model=name,task=task,view=view,algorithm=algorithm,revision=revision,n_features=len(cols),
                    **metrics(y[dev],prediction,groups,task,classes,supported,row_weights(meta,indices)))
                summary.append(row);pd.DataFrame(summary).to_csv(out/'development_comparison.csv',index=False)
                np.savez_compressed(out/'development_predictions'/f'{name}.npz',scores=prediction,indices=indices,classes=np.array(classes),y=y[dev],groups=groups)
                pd.DataFrame(fold_audit).to_csv(out/'development_predictions'/f'{name}_folds.csv',index=False)
                if captured:(out/'process'/f'{name}_warnings.json').write_text(json.dumps([str(w.message) for w in captured],ensure_ascii=False,indent=2))
                log(out,'development_model',**row)
    log(out,'development_revision_complete',revision=revision,candidate_count=len(summary),final_cases_scored=False)

def selective_thresholds(meta,task,classes,y,p,groups,development_support,sample_weight=None):
    result=[];supportmap={r['label']:r for r in development_support};weights=case_weights(groups) if sample_weight is None else sample_weight
    for j,c in enumerate(classes):
        positive=(y==c) if task=='treatment' else y[:,j].astype(bool)
        options=[]
        for threshold in [.5,.6,.7,.8,.9]:
            choose=p[:,j]>=threshold
            if task=='treatment':choose&=p.argmax(1)==j
            n=int(choose.sum());nc=len(set(groups[choose]));precision=float(np.average(positive[choose],weights=weights[choose])) if n else np.nan
            recall=float(weights[choose&positive].sum()/weights[positive].sum()) if positive.any() else np.nan
            options.append(dict(label=c,threshold=threshold,precision=precision,recall=recall,n_predictions=n,n_prediction_cases=nc))
        valid=[r for r in options if r['n_prediction_cases']>=5 and r['precision']>=.85 and r['recall']>=.4 and supportmap[c]['n_cases']>=3]
        best=min(valid,key=lambda r:r['threshold']) if valid else options[-1]
        result.append({**best,'eligible_for_window_validation':bool(valid),'true_development_cases':supportmap[c]['n_cases'],
            'status':'OOF_SUPPORTED_PENDING_FINAL_AND_WINDOW_VALIDATION' if valid else 'INSUFFICIENT_SELECTIVE_EVIDENCE'})
    return result

def freeze(out):
    if (out/'selection_freeze.json').exists():raise FileExistsError('Final selection exists')
    comparison=pd.read_csv(out/'development_comparison.csv');choices={};thresholds={};dependencies={};controls={}
    for task in ['instrument','treatment']:
        allowed=comparison[(comparison.task==task)&comparison.view.ne('duration_only')&~comparison.algorithm.isin(['dummy','top1_prior'])]
        best=allowed.sort_values(['supported_macro_f1','macro_f1','accuracy','model'],ascending=[False,False,False,True]).iloc[0]
        choices[task]=best.model
        meta,x,cols=feature_table(out,best['view']);y,mask,classes=target(meta,task);dev=mask&meta.split.eq('development').to_numpy()
        support=supports(meta,y,dev,task,classes)
        oof=np.load(out/'development_predictions'/f'{best.model}.npz');assert oof['indices'].tolist()==np.flatnonzero(dev).tolist()
        thresholds[task]=selective_thresholds(meta,task,classes,y[dev],oof['scores'],meta.seq.to_numpy()[dev],support,row_weights(meta,np.flatnonzero(dev)))
        model=Bundle(task,estimators()[best.algorithm],cols,classes).fit(x[dev],y[dev],meta.seq.to_numpy()[dev],row_weights(meta,np.flatnonzero(dev)));model.save(out/'models'/f'{task}_development.pkl')
        dependencies[task]=dict(view=best['view'],algorithm=best.algorithm,features=cols,classes=classes,supported_classes=[r['label'] for r in support if r['n_cases']>=3])
        controls[task]={}
        for control,view in [('duration','duration_only'),('mask','mask_only')]:
            candidates=comparison[(comparison.task==task)&comparison.view.eq(view)]
            if not len(candidates):continue
            cb=candidates.sort_values(['supported_macro_f1','macro_f1'],ascending=False).iloc[0]
            cm,cx,cc=feature_table(out,view);cy,cmask,cclasses=target(cm,task);cd=cmask&cm.split.eq('development').to_numpy()
            Bundle(task,estimators()[cb.algorithm],cc,cclasses).fit(cx[cd],cy[cd],cm.seq.to_numpy()[cd],row_weights(cm,np.flatnonzero(cd))).save(out/'models'/f'{task}_control_{control}.pkl')
            controls[task][control]=dict(model=cb.model,view=view,algorithm=cb.algorithm)
    input_paths=[out/n for n in ['annotations_aligned.csv','features_basic.csv','features_extended.csv','features_context.csv','cases_manifest.csv','development_comparison.csv'] if (out/n).exists()]
    input_paths+=sorted((out/'models').glob('*.pkl'))
    if (out/'extraction_fingerprint.json').exists():input_paths.append(out/'extraction_fingerprint.json')
    source=Path(__file__).parent
    frozen=dict(version='supervised-interval-v1',seed=SEED,created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        choices=choices,dependencies=dependencies,thresholds=thresholds,controls=controls,
        final_case_ids=[1469,1470,1471,1472,1475,1476,1477,1478],
        evaluation='Oracle annotation boundaries; case-balanced metrics; final chronological8cases once. Class probability scores are uncalibrated. Mask-only encodes structural absence as well as missing observations, not pure recording-quality control.',
        missing_instrument='Missing target rows excluded for instrument only, all rows retained in audit',
        unit='weak_interval_inherited_window' if 'annotation_id' in meta else 'human_annotated_interval',
        sample_weights='Equal cases, equal annotation parents within case, equal windows within parent if applicable.',
        unmatched_labels='Unseen treatment labels and novel instrument combinations remain in final evaluation',
        learnability='Report target and class-specific evidence; no all18-label claim. Window application requires separate validation and never treats gaps as negatives.',
        input_sha256={str(p.resolve()):sha(p) for p in input_paths},
        code_sha256={str(p.resolve()):sha(p) for p in source.glob('*.py') if p.name in ['learning.py','features.py','extract.py','global_context.py','window_learning.py','window_data.py']})
    (out/'selection_freeze.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n')
    log(out,'interval_selection_frozen',choices=choices,eligible_window_labels={t:[r['label'] for r in thresholds[t] if r['eligible_for_window_validation']] for t in choices})

def per_label(y,p,groups,task,classes,sample_weight=None):
    records=[];w=case_weights(groups) if sample_weight is None else sample_weight
    for j,c in enumerate(classes):
        true=(y==c) if task=='treatment' else y[:,j].astype(bool)
        pred=(p.argmax(1)==j) if task=='treatment' else p[:,j]>=.5
        pr,re,f1,_=precision_recall_fscore_support(true,pred,average='binary',sample_weight=w,zero_division=0)
        records.append(dict(label=c,precision=float(pr),recall=float(re),f1=float(f1),n_true_intervals=int(true.sum()),n_true_cases=len(set(groups[true])),n_predicted=int(pred.sum())))
    return records

def selective_rows(y,p,groups,task,classes,thresholds,sample_weight=None):
    w=case_weights(groups) if sample_weight is None else sample_weight;records=[]
    for r in thresholds:
        j=classes.index(r['label']);true=(y==r['label']) if task=='treatment' else y[:,j].astype(bool)
        choose=p[:,j]>=r['threshold']
        if task=='treatment':choose&=p.argmax(1)==j
        if not r['eligible_for_window_validation']:choose[:]=False
        precision=float(np.average(true[choose],weights=w[choose])) if choose.any() else np.nan
        recall=float(w[choose&true].sum()/w[true].sum()) if true.any() else np.nan
        truecases=len(set(groups[true]));predcases=len(set(groups[choose]));n=int(choose.sum())
        passed=bool(r['eligible_for_window_validation'] and n>=3 and predcases>=2 and truecases>=2 and precision>=.8)
        records.append(dict(label=r['label'],threshold=r['threshold'],development_eligible=r['eligible_for_window_validation'],
            n_predictions=n,n_prediction_cases=predcases,n_true_cases=truecases,precision=precision,recall=recall,
            weighted_coverage=float(w[choose].sum()/w.sum()),passed_for_next_stage=passed,
            interpretation='Conditional on supplied annotated intervals; not unannotated-background specificity.'))
    return records

def evaluate(out):
    if (out/'final_summary.json').exists():raise FileExistsError('Final interval test already used')
    frozen=json.loads((out/'selection_freeze.json').read_text())
    for path,h in {**frozen['input_sha256'],**frozen['code_sha256']}.items():assert sha(path)==h,path
    (out/'final_predictions').mkdir(exist_ok=True);summary={};rows=[]
    for task in ['instrument','treatment']:
        spec=frozen['dependencies'][task];meta,x,cols=feature_table(out,spec['view']);assert cols==spec['features']
        y,mask,classes=target(meta,task);ix=np.flatnonzero(mask&meta.split.eq('final').to_numpy())
        with (out/'models'/f'{task}_development.pkl').open('rb') as f:model=pickle.load(f)
        p=model.predict_scores(x[ix]);groups=meta.seq.to_numpy()[ix]
        weights=row_weights(meta,ix)
        result=metrics(y[ix],p,groups,task,classes,spec['supported_classes'],weights)
        labels=per_label(y[ix],p,groups,task,classes,weights);pd.DataFrame(labels).to_csv(out/f'{task}_final_per_label.csv',index=False)
        # Chronological baseline fitted only to the same development target rows.
        dev=mask&meta.split.eq('development').to_numpy();baseline=Bundle(task,estimators()['top1_prior' if task=='instrument' else 'dummy'],cols,classes).fit(x[dev],y[dev],meta.seq.to_numpy()[dev],row_weights(meta,np.flatnonzero(dev)))
        bp=baseline.predict_scores(x[ix]);bresult=metrics(y[ix],bp,groups,task,classes,spec['supported_classes'],weights)
        np.savez_compressed(out/'final_predictions'/f'{task}.npz',scores=p,baseline_scores=bp,y=y[ix],indices=ix,classes=np.array(classes),groups=groups)
        selective=selective_rows(y[ix],p,groups,task,classes,frozen['thresholds'][task],weights);pd.DataFrame(selective).to_csv(out/f'{task}_final_selective.csv',index=False)
        control_results={}
        for control,cspec in frozen['controls'][task].items():
            cm,cx,cc=feature_table(out,cspec['view'])
            with (out/'models'/f'{task}_control_{control}.pkl').open('rb') as ff:cmodel=pickle.load(ff)
            cp=cmodel.predict_scores(cx[ix]);control_results[control]=metrics(y[ix],cp,groups,task,classes,spec['supported_classes'],weights)
            np.savez_compressed(out/'final_predictions'/f'{task}_control_{control}.npz',scores=cp,y=y[ix],indices=ix,classes=np.array(classes),groups=groups)
        labelaudit=[];train_combinations={tuple(v.astype(int)) for v in y[dev]} if task=='instrument' else None
        for j,r in enumerate(ix):
            if task=='treatment':actual=str(y[r]);pred=classes[p[j].argmax()];available=actual in set(y[dev])
            else:actual='|'.join(np.array(classes)[y[r].astype(bool)]);pred='|'.join(np.array(classes)[p[j]>=.5]);available=True
            rows.append(dict(task=task,id=meta.id.iloc[r],seq=int(meta.seq.iloc[r]),actual=actual,predicted=pred,training_label_available=available,score=float(p[j].max()),
                training_instrument_combination_available=bool(tuple(y[r].astype(int)) in train_combinations) if task=='instrument' else None))
            labelaudit.append(dict(id=meta.id.iloc[r],seq=int(meta.seq.iloc[r]),training_label_available=available))
        summary[task]=dict(model=frozen['choices'][task],metrics=result,baseline=bresult,controls=control_results,per_label=labels,selective=selective,
            unseen_treatment_rows=sum(not a['training_label_available'] for a in labelaudit) if task=='treatment' else None,
            interval_classification_only=True,probability_calibrated=False)
        log(out,'final_interval_evaluated',task=task,metrics=result,baseline=bresult)
    pd.DataFrame(rows).to_csv(out/'final_predictions.csv',index=False)
    result=dict(status='CLASS_SPECIFIC_ASSESSMENT_REQUIRED',targets=summary,requested_cases=int(meta.seq.nunique()),requested_intervals=len(meta),unit=frozen['unit'],
        final_case_count=8,final_evaluation_reused_for_tuning=False,unannotated_time_has_ground_truth=False,
        treatment_names_are_user_annotations=True,clinical_or_external_patient_validation=False)
    (out/'final_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['develop','freeze','evaluate']);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--revision',type=int,default=1);a=ap.parse_args()
    try:
        if a.stage=='develop':develop(a.output,a.revision)
        else:globals()[a.stage](a.output)
    except Exception as exc:
        p=a.output/'process';p.mkdir(exist_ok=True,parents=True)
        (p/'latest_error.json').write_text(json.dumps(dict(stage=a.stage,error=repr(exc),traceback=traceback.format_exc()),ensure_ascii=False,indent=2))
        raise

if __name__=='__main__':main()
