"""Semantics-corrected, observed-only, non-DTW comparison (separate version).

No median imputation. Pairwise distances use jointly observed code blocks and
are not assumed Euclidean; only average linkage and k-medoids are applied.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

from cluster_compare import eta_squared, transform, fit_labels

NUMERIC = ['20742','20108','20743','20776','10014','10015']
STATES = ['20011','20039','20101','20774']
SEED=42


def prepare(features, catalog, audit, numeric_min=10, known_min=.2):
    cat=catalog[catalog.view.isin(['levels','operations'])].copy().reset_index(drop=True)
    # Avoid redundant HP OFF/nonzero complements; retain specific nonzero states.
    cat=cat[~cat.feature.str.contains(r'__state0_fraction$|__active_fraction$') |
            cat.code.isin(['20774'])].reset_index(drop=True)
    raw=features[cat.feature].copy()
    counts=audit.pivot(index='case_id',columns='code',values='n_observed_bins')
    for code in NUMERIC:
        if code=='20776':
            continue  # change-only setting: event count is not information coverage
        low=counts[code].reindex(raw.index)<numeric_min
        fields=cat[(cat.code==code)&(cat.view=='levels')].feature
        raw.loc[low,fields]=np.nan
    for code in STATES:
        low=features[f'E{code}__known_fraction']<known_min
        fields=cat[(cat.code==code)&(cat.view=='operations')].feature
        raw.loc[low,fields]=np.nan
    joint=[f for f in raw if 'selected_match_given_running' in f]
    low=(features['E20011__known_fraction']<known_min)|(features['E20039__known_fraction']<known_min)
    raw.loc[low,joint]=np.nan
    numeric_fields=cat[cat.view=='levels'].feature
    # A conservative supplemental cohort: both HP states have known support and
    # at least one actual numeric setting/measurement. Full 300-case audit retained.
    usable=((features['E20011__known_fraction']>=known_min)&
            (features['E20039__known_fraction']>=known_min)&raw[numeric_fields].notna().any(axis=1))
    return raw,cat,usable


def observed_distance(raw, catalog):
    """Robust code-balanced L1 distance with NaN-preserving common support."""
    count=np.isfinite(raw).sum(axis=0)
    keep=count>=max(5,int(.10*len(raw)))
    cat=catalog.loc[keep].reset_index(drop=True)
    x=np.asarray(raw[:,keep],float)
    med=np.nanmedian(x,axis=0)
    lo,q1,q3,hi=np.nanquantile(x,[.1,.25,.75,.9],axis=0)
    scale=np.maximum(q3-q1,.25*(hi-lo))
    sd=np.nanstd(x,axis=0)
    scale=np.where(scale>1e-9,scale,sd)
    variable=scale>1e-9
    cat=cat.loc[variable].reset_index(drop=True)
    x=x[:,variable]; med=med[variable];scale=scale[variable]
    z=np.clip((x-med)/scale,-5,5)  # NaNs intentionally remain NaN.
    n=len(x)
    family_sum={family:np.zeros((n,n)) for family in ['levels','operations']}
    family_n={family:np.zeros((n,n),int) for family in ['levels','operations']}
    for (family,code),fields in cat.groupby(['view','code']):
        ix=fields.index.to_numpy()
        block=z[:,ix]
        valid=np.isfinite(block[:,None,:])&np.isfinite(block[None,:,:])
        diff=np.minimum(np.abs(block[:,None,:]-block[None,:,:])/3,1)
        weight=valid.sum(axis=2)
        distance=np.divide(np.nansum(np.where(valid,diff,np.nan),axis=2),weight,
                           out=np.zeros((n,n)),where=weight>0)
        family_sum[family]+=np.where(weight>0,distance,0)
        family_n[family]+=(weight>0)
    total=np.zeros((n,n));n_families=np.zeros((n,n),int)
    for family in ['levels','operations']:
        good=family_n[family]>0
        total+=np.divide(family_sum[family],family_n[family],out=np.zeros((n,n)),where=good)
        n_families+=good
    assert np.all(n_families>0), 'Undefined distance from no common features'
    matrix=total/n_families
    np.fill_diagonal(matrix,0)
    shared=family_n['levels']+family_n['operations']
    assert np.all(shared[np.triu_indices(n,1)]>=2), 'Insufficient shared signal blocks'
    return matrix,shared,family_n['levels'],cat


def fit(matrix,method,k,seed):
    if method=='average':
        lab=fcluster(linkage(squareform(matrix,checks=False),method='average'),k,criterion='maxclust')
        return lab
    # Alternate assignment/update k-medoids; multi-start, with observed medoids.
    rng=np.random.default_rng(seed)
    best=None
    for init in range(8):
        centers=[int(rng.integers(len(matrix)))]
        while len(centers)<k:
            nearest=np.min(matrix[:,centers],axis=1)
            nearest[centers]=0
            remaining=np.setdiff1d(np.arange(len(matrix)),centers)
            if nearest.sum()>1e-12:
                nxt=int(rng.choice(len(matrix),p=nearest/nearest.sum()))
            else:
                nxt=int(rng.choice(remaining))
            centers.append(nxt)
        for iteration in range(100):
            labels=np.argmin(matrix[:,centers],axis=1)
            updated=[]
            for cl in range(k):
                members=np.flatnonzero(labels==cl)
                if len(members):
                    updated.append(int(members[np.argmin(matrix[np.ix_(members,members)].sum(axis=1))]))
                else:
                    nearest=np.min(matrix[:,centers],axis=1)
                    nearest[updated+centers]=-1
                    updated.append(int(np.argmax(nearest)))
            if updated==centers:
                break
            centers=updated
        labels=np.argmin(matrix[:,centers],axis=1)
        objective=float(np.min(matrix[:,centers],axis=1).sum())
        if best is None or objective<best[0]:
            best=objective,labels
    return best[1]+1


def subsamples(raw,cat,labels,method,k,repeats=60):
    rng=np.random.default_rng(SEED+2000)
    rows=[]
    for iteration in range(repeats):
        ix=np.sort(rng.choice(len(raw),int(round(.8*len(raw))),replace=False))
        matrix,_,_,_=observed_distance(raw[ix],cat)
        candidate=fit(matrix,method,k,SEED+iteration+2001)
        rows.append(dict(iteration=iteration,n_cases=len(ix),ari=float(adjusted_rand_score(labels[ix],candidate)),
                         k_actual=len(set(candidate)),valid_repeat=len(set(candidate))==k))
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args(); out=args.output.resolve()
    if (out/'selected_models.csv').exists():
        parser.error('Model outputs already exist. Use a fresh output directory; saved labels must be preserved.')
    (out/'process').mkdir(exist_ok=True)
    def log(event,**info):
        row=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),event=event,**info)
        with (out/'process/modeling_log.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    f=pd.read_csv(out/'features.csv').set_index('case_id')
    c=pd.read_csv(out/'feature_catalog.csv',dtype={'code':str})
    m=pd.read_csv(out/'case_manifest.csv')
    audit=pd.read_csv(out.parent/'signal_audit.csv',dtype={'code':str})
    raw,cat,usable=prepare(f,c,audit)
    raw.to_csv(out/'analysis_features.csv',index_label='case_id')
    cat.to_csv(out/'analysis_feature_catalog.csv',index=False)
    ids=raw.index[usable]
    x=raw.loc[ids].to_numpy(float)
    plan=dict(version='recording_semantics_v2',boundary='first90012ON-lastOFF',seed=SEED,
        numeric_min_bins=10,state_min_known_fraction=.2,
        scaler_output='20776 setting held until next record, duration-weighted within 20774=1',
        syringe='20101=0 picked up; 1 stored. Pickup does not establish spraying.',
        imputation=False,code_distance='robust observed-only L1, equally weighted code blocks within each modality',
        modality_weight='equal numeric/operation weights when both jointly observed; one modality if other has no shared codes',
        mandatory_support='HP20011 and HP20039 known>=20% and at least one supported numeric code',
        minimum_shared_codes=2,metric_warning='Different pairs can share different codes; not a Euclidean metric',
        methods=['kmedoids','average'],k_range=list(range(2,9)),min_cluster_n=10,max_largest_fraction=.9,
        screening_repeats=12,final_repeats=60,selection='within method: valid size/balance then silhouette + .35*screen median ARI',
        sensitivity_conditions=[['numeric5_known20',5,.2],['numeric20_known20',20,.2],['numeric10_known50',10,.5]],
        scientific_status='Supplemental limited-support cohort; no external/clinical validation')
    (out/'model_plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False))
    scope=m[['case_id','date','duration_s','signal_status']].copy().set_index('case_id')
    scope['v2_status']=np.where(usable.reindex(scope.index),'CLASSIFIED',
        np.where(scope.signal_status=='NO_TARGET_SIGNAL','NO_TARGET_SIGNAL','INSUFFICIENT_COMPARABLE_SUPPORT'))
    scope['hp_selection_known_fraction']=f['E20011__known_fraction'].reindex(scope.index)
    scope['hp_running_known_fraction']=f['E20039__known_fraction'].reindex(scope.index)
    scope['supported_numeric_codes']=[';'.join(code for code in NUMERIC
        if raw.loc[case,cat[(cat.code==code)&(cat.view=='levels')].feature].notna().any())
        for case in scope.index]
    scope['support_failure_reason']=[('ALL_TEN_TARGET_CODES_UNRECORDED' if row.signal_status=='NO_TARGET_SIGNAL'
        else ';'.join([reason for failed,reason in [
            (row.hp_selection_known_fraction<.2,'HP_SELECTION_KNOWN_LT20PCT'),
            (row.hp_running_known_fraction<.2,'HP_RUNNING_KNOWN_LT20PCT'),
            (not row.supported_numeric_codes,'NO_SUPPORTED_NUMERIC_CODE')] if failed]))
        for row in scope.itertuples()]
    scope.reset_index().to_csv(out/'cohort_scope.csv',index=False)
    matrix,shared,numeric_shared,used=observed_distance(x,cat)
    pd.DataFrame(matrix,index=ids,columns=ids).to_csv(out/'observed_distances.csv',index_label='case_id')
    pd.DataFrame(shared,index=ids,columns=ids).to_csv(out/'shared_code_counts.csv',index_label='case_id')
    pd.DataFrame(numeric_shared,index=ids,columns=ids).to_csv(out/'shared_numeric_code_counts.csv',index_label='case_id')
    used.to_csv(out/'used_features.csv',index=False)
    off=np.triu_indices(len(ids),1)
    log('preflight',requested=300,classifiable=len(ids),status_counts=scope.v2_status.value_counts().to_dict(),
        shared_code_min=int(shared[off].min()),shared_code_median=float(np.median(shared[off])),
        pairs_with_no_shared_numeric=int(np.sum(numeric_shared[off]==0)),n_pairs=len(off[0]))
    candidates=[]; labels_lookup={}
    for method in ['kmedoids','average']:
        for k in range(2,9):
            labels=fit(matrix,method,k,SEED)
            sizes=pd.Series(labels).value_counts()
            screen=subsamples(x,cat,labels,method,k,12)
            vals=[r['ari'] for r in screen if r['valid_repeat']]
            sil=float(silhouette_score(matrix,labels,metric='precomputed')) if len(sizes)>1 else np.nan
            valid=(len(sizes)==k and sizes.min()>=10 and sizes.max()/len(ids)<=.9 and sil>0)
            mid=f'observed__{method}__k{k}'
            candidates.append(dict(model_id=mid,view='observed',method=method,k=k,n_cases=len(ids),
                silhouette=sil,min_cluster_n=int(sizes.min()),largest_fraction=float(sizes.max()/len(ids)),
                stability_median=float(np.median(vals)) if vals else np.nan,valid=valid,
                selection_score=float(sil+.35*np.median(vals)) if vals else -np.inf))
            labels_lookup[mid]=labels
    candidates=pd.DataFrame(candidates)
    candidates.to_csv(out/'model_candidates.csv',index=False)
    selected=[]
    for method,g in candidates.groupby('method'):
        pool=g[g.valid]
        if pool.empty:pool=g
        selected.append(pool.sort_values(['selection_score','k'],ascending=[False,True]).iloc[0].to_dict())
    rows=Parallel(n_jobs=args.workers)(delayed(subsamples)(x,cat,labels_lookup[r['model_id']],r['method'],int(r['k']),60)
                                          for r in selected)
    assignments=[];reps=[];stability=[];sizes=[]
    for model, trials in zip(selected,rows):
        mid=model['model_id'];labels=labels_lookup[mid]
        vals=[r['ari'] for r in trials if r['valid_repeat']]
        model.update(stability_median=float(np.median(vals)),stability_q25=float(np.quantile(vals,.25)),
                     stability_q75=float(np.quantile(vals,.75)),stability_min=float(np.min(vals)),
                     stability_valid_repeats=len(vals),stability_failed_repeats=60-len(vals))
        quality_cat=c[c.view=='quality'].reset_index(drop=True)
        quality_matrix,_,_=transform(f.loc[ids,quality_cat.feature].to_numpy(float),quality_cat)
        q1,_=fit_labels(quality_matrix,'kmeans',int(model['k']),SEED)
        q2,_=fit_labels(quality_matrix,'ward',int(model['k']),SEED)
        indexed_manifest=m.set_index('case_id').loc[ids]
        model['quality_ari']=float(max(adjusted_rand_score(labels,q1),adjusted_rand_score(labels,q2)))
        model['date_nmi']=float(normalized_mutual_info_score(indexed_manifest.date.str[:4],labels))
        model['duration_eta2']=eta_squared(indexed_manifest.duration_s,labels)
        for row in trials:stability.append(dict(model_id=mid,**row))
        labmap=dict(zip(ids,labels))
        for case in m.case_id:
            assignments.append(dict(model_id=mid,case_id=case,cluster=labmap.get(case,np.nan),
                status=scope.loc[case,'v2_status'],view='observed',method=model['method'],k=model['k']))
        for cl in sorted(set(labels)):
            ix=np.flatnonzero(labels==cl)
            sizes.append(dict(model_id=mid,cluster=int(cl),n_cases=len(ix)))
            med=int(ix[np.argmin(matrix[np.ix_(ix,ix)].sum(axis=1))])
            ordered=[med]+[j for j in ix[np.argsort(matrix[med,ix],kind='stable')] if j!=med][:4]
            for rank,j in enumerate(ordered):
                reps.append(dict(model_id=mid,cluster=int(cl),rank=rank,case_id=ids[j],
                    is_medoid=j==med,distance_to_center=float(matrix[med,j])))
    pd.DataFrame(selected).to_csv(out/'selected_models.csv',index=False)
    pd.DataFrame(assignments).to_csv(out/'assignments_long.csv',index=False)
    pd.DataFrame(stability).to_csv(out/'stability_repeats.csv',index=False)
    pd.DataFrame(reps).to_csv(out/'representatives.csv',index=False)
    pd.DataFrame(sizes).to_csv(out/'cluster_sizes.csv',index=False)
    sensitivity=[]
    for scenario,minimum,known in plan['sensitivity_conditions']:
        rx,rc,ok=prepare(f,c,audit,minimum,known)
        common=ids.intersection(rx.index[ok],sort=False)
        dm,_,_,_=observed_distance(rx.loc[common].to_numpy(float),rc)
        for model in selected:
            lab=fit(dm,model['method'],int(model['k']),SEED)
            original=pd.Series(labels_lookup[model['model_id']],index=ids).loc[common]
            sensitivity.append(dict(model_id=model['model_id'],scenario=scenario,n_common_cases=len(common),
                n_primary_cases=len(ids),ari=float(adjusted_rand_score(original,lab)),k_actual=len(set(lab))))
    pd.DataFrame(sensitivity).to_csv(out/'support_sensitivity.csv',index=False)
    # All primary-v1 labels compared on exactly this supplemental cohort.
    old=pd.read_csv(out.parent/'assignments_long.csv')
    comparison=[]
    for model in selected:
        current=pd.Series(labels_lookup[model['model_id']],index=ids)
        for om,group in old[old.cluster.notna()].groupby('model_id'):
            previous=group.set_index('case_id').cluster
            common=current.index.intersection(previous.index)
            comparison.append(dict(v2_model=model['model_id'],v1_model=om,n_common_cases=len(common),
                ari=float(adjusted_rand_score(current.loc[common],previous.loc[common]))))
    pd.DataFrame(comparison).to_csv(out/'comparison_with_v1.csv',index=False)
    with threadpool_limits(limits=1):
        coords=MDS(n_components=2,dissimilarity='precomputed',random_state=SEED,n_init=4,
                   normalized_stress='auto',max_iter=500).fit_transform(matrix)
    pd.DataFrame(dict(case_id=ids,mds1=coords[:,0],mds2=coords[:,1])).to_csv(out/'embedding_mds.csv',index=False)
    log('complete',selected_models=selected,sensitivity=sensitivity,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    main()
