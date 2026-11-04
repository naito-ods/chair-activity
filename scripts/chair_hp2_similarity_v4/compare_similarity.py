"""Distinct-case HP2 similarity, conditional neighborhoods and candidate groups."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score,silhouette_score,normalized_mutual_info_score
from sklearn.manifold import MDS
from threadpoolctl import threadpool_limits

SEED=42
CORE=['torque','pedal2']


def log(out,event,**fields):
    row=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),event=event,**fields)
    with (out/'process/similarity_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def quantile(x,w,q):
    good=np.isfinite(x)&np.isfinite(w)&(w>0)
    x=x[good];w=w[good]
    if not len(x):return np.nan
    order=np.argsort(x,kind='stable');x=x[order];w=w[order]
    # Inverse weighted empirical CDF: avoid a boundary moving to the next value
    # solely because e.g. ten weights0.1 sum to0.9999999999999999.
    tolerance=1e-12*max(1.,float(w.sum()))
    return float(x[min(np.searchsorted(np.cumsum(w)+tolerance,q*w.sum()),len(x)-1)])


def case_weights(cases):
    cases=np.asarray(cases,str)
    counts=pd.Series(cases).value_counts()
    return np.array([1/counts[c] for c in cases],float)


def distance(raw,catalog,cases,view):
    """No imputation; equal family weights and case-balanced robust scales."""
    cat=catalog.copy().reset_index(drop=True)
    if view=='combined':
        s,sp=distance(raw,cat[cat.view=='structure'],cases,'structure')
        n,np_=distance(raw,cat[(cat.view=='numeric')&cat.block.isin(CORE)],cases,'numeric')
        return (s+n)/2,sp+np_
    if view=='numeric':cat=cat[(cat.view=='numeric')&cat.block.isin(CORE)].reset_index(drop=True)
    elif view not in ['quality','quality_fraction']:cat=cat[cat.view==view].reset_index(drop=True)
    x=raw[cat.feature].to_numpy(float)
    weights=case_weights(cases)
    scales=[];keep=[];prep=[]
    for j,r in enumerate(cat.itertuples()):
        values=x[:,j].copy()
        if r.transform=='log1p':values=np.log1p(np.maximum(values,0))
        q10,q25,median,q75,q90=[quantile(values,weights,q) for q in [.1,.25,.5,.75,.9]]
        finite=np.isfinite(values)
        if finite.sum()<3:continue
        sd=float(np.sqrt(np.average((values[finite]-np.average(values[finite],weights=weights[finite]))**2,weights=weights[finite])))
        scale=max(q75-q25,.25*(q90-q10),sd if q75-q25<1e-10 else 0)
        if not np.isfinite(scale) or scale<1e-10:continue
        scales.append(np.clip((values-median)/scale,-5,5));keep.append(j)
        prep.append(dict(feature=r.feature,view=r.view,block=r.block,median=median,scale=scale,transform=r.transform))
    if not scales:raise ValueError(f'No varying features: {view}')
    z=np.column_stack(scales);cat=cat.iloc[keep].reset_index(drop=True)
    total=np.zeros((len(raw),len(raw)));families=np.zeros_like(total,dtype=int)
    for _,group in cat.groupby('block',sort=False):
        a=z[:,group.index]
        valid=np.isfinite(a[:,None,:])&np.isfinite(a[None,:,:])
        count=valid.sum(axis=2)
        diff=np.minimum(np.abs(a[:,None,:]-a[None,:,:])/3,1)
        family=np.divide(np.nansum(np.where(valid,diff,np.nan),axis=2),count,
                         out=np.zeros_like(total),where=count>0)
        total+=family;families+=(count>0)
    if np.any(families==0):raise ValueError(f'Undefined pairwise comparison in {view}')
    d=total/families;np.fill_diagonal(d,0)
    assert np.isfinite(d).all() and np.allclose(d,d.T)
    return d,prep


def fit(d,cases,method,k,seed):
    if method=='average':return fcluster(linkage(squareform(d,checks=False),method='average'),k,criterion='maxclust')
    w=case_weights(cases);rng=np.random.default_rng(seed);best=None
    for init in range(8):
        centers=[int(rng.choice(len(d),p=w/w.sum()))]
        while len(centers)<k:
            probabilities=d[:,centers].min(axis=1)*w;probabilities[centers]=0
            remaining=np.setdiff1d(np.arange(len(d)),centers)
            centers.append(int(rng.choice(len(d),p=probabilities/probabilities.sum())) if probabilities.sum()>1e-12 else int(rng.choice(remaining)))
        for _ in range(60):
            labels=d[:,centers].argmin(axis=1);updated=[]
            for cl in range(k):
                ix=np.flatnonzero(labels==cl)
                if len(ix):updated.append(int(ix[np.argmin(d[np.ix_(ix,ix)]@w[ix])]))
                else:
                    remaining=d[:,centers].min(axis=1);remaining[centers+updated]=-1
                    updated.append(int(np.argmax(remaining)))
            if centers==updated:break
            centers=updated
        labels=d[:,centers].argmin(axis=1)
        cost=float(d[:,centers].min(axis=1)@w)
        if best is None or cost<best[0]:best=cost,labels
    return best[1]+1


def nearest(d,cases,i,k=3):
    order=np.argsort(d[i],kind='stable');used={cases[i]};answer=[]
    for j in order:
        if cases[j] in used:continue
        answer.append(int(j));used.add(cases[j])
        if len(answer)>=k:break
    return answer


def partition_repeats(raw,cat,cases,view,method,k,reference,repeats,seed):
    unique=np.array(sorted(set(cases)));rng=np.random.default_rng(seed);rows=[]
    for iteration in range(repeats):
        chosen=rng.choice(unique,max(k+2,int(round(.8*len(unique)))),replace=False)
        ix=np.flatnonzero(np.isin(cases,chosen))
        d,_=distance(raw.iloc[ix],cat,cases[ix],view)
        labels=fit(d,cases[ix],method,k,seed+iteration+1)
        rows.append(dict(iteration=iteration,n_cases=len(chosen),n_units=len(ix),k_actual=len(set(labels)),
            valid_repeat=len(set(labels))==k,ari=float(adjusted_rand_score(reference[ix],labels))))
    return rows


def candidate(view,level,method,k,raw,cat,units,repeats):
    cases=units.case_id.to_numpy(str);d,_=distance(raw,cat,cases,view)
    labels=fit(d,cases,method,k,SEED);sizes=pd.Series(labels).value_counts()
    casecounts=pd.DataFrame(dict(cl=labels,case=cases)).groupby('cl').case.nunique()
    sil=float(silhouette_score(d,labels,metric='precomputed')) if len(sizes)>1 else np.nan
    trial=partition_repeats(raw,cat,cases,view,method,k,labels,repeats,SEED+100)
    values=[r['ari'] for r in trial if r['valid_repeat']]
    median=float(np.median(values)) if values else np.nan
    valid=(len(sizes)==k and sizes.min()>=(8 if level=='block' else 5) and casecounts.min()>=5 and sizes.max()/len(labels)<=.9 and sil>0)
    return dict(model_id=f'{level}__{view}__{method}__k{k}',level=level,view=view,method=method,k=k,
        n_units=len(raw),n_cases=len(set(cases)),silhouette=sil,min_cluster_units=int(sizes.min()),
        min_cluster_cases=int(casecounts.min()),largest_fraction=float(sizes.max()/len(labels)),
        stability_median=median,valid=bool(valid),selection_score=float(sil+.35*median) if values else -1),labels


def neighbor_repeats(raw,cat,units,view,d,repeats=40):
    cases=units.case_id.to_numpy(str);unique=np.array(sorted(set(cases)));rng=np.random.default_rng(SEED+300)
    outcomes={i:[] for i in range(len(raw))};top1={i:[] for i in range(len(raw))}
    for iteration in range(repeats):
        chosen=rng.choice(unique,max(5,int(round(.8*len(unique)))),replace=False)
        ix=np.flatnonzero(np.isin(cases,chosen));subcase=cases[ix]
        nd,_=distance(raw.iloc[ix],cat,subcase,view)
        reference=d[np.ix_(ix,ix)]
        # Both reference and refit see exactly the same remaining candidate cases.
        for j,old in enumerate(ix):
            baseline=nearest(reference,subcase,j);current=nearest(nd,subcase,j)
            b={subcase[q] for q in baseline};a={subcase[q] for q in current}
            outcomes[old].append(len(a&b)/len(b) if b else np.nan)
            top1[old].append(bool(baseline and current and subcase[baseline[0]]==subcase[current[0]]))
    return [dict(unit_id=units.unit_id.iloc[i],case_id=cases[i],view=view,n_present_repeats=len(values),
        conditional_neighbor_retention=float(np.nanmean(values)),top1_retention=float(np.mean(top1[i]))) for i,values in outcomes.items()]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--resume-finalize',action='store_true',help='Finish pair explanations after verified model/neighbor tables already exist')
    args=parser.parse_args();out=args.output.resolve()
    if args.resume_finalize:
        finish_outputs(out)
        return
    if (out/'models.csv').exists():raise FileExistsError('Use a new output directory; previous results are preserved.')
    (out/'distances').mkdir(exist_ok=True)
    units=pd.read_csv(out/'units.csv');features=pd.read_csv(out/'features.csv').set_index('unit_id')
    cat=pd.read_csv(out/'feature_catalog.csv');cf=pd.read_csv(out/'case_features.csv').set_index('case_id')
    cc=pd.read_csv(out/'case_feature_catalog.csv')
    assert len(units)==203 and units.case_id.nunique()==67
    # Case-IQR is retained for description. Primary case distances compare median
    # profiles only, avoiding fabricated heterogeneity for single-block cases.
    cc=cc[(~cc.feature.str.endswith('__iqr'))&(~cc.block.eq('case_structure'))].reset_index(drop=True)
    cu=pd.DataFrame(dict(unit_id=cf.index,case_id=cf.index))
    qualitycat=cat[(cat.view=='quality')&cat.block.str.endswith('_support')].reset_index(drop=True)
    casequalitycat=cc[(cc.view=='quality')&cc.feature.str.contains('n_bins|observed_fraction|eligible_unit_count')].reset_index(drop=True)
    numeric_ok=(features.torque_n_bins>=3)&(features.pedal2_n_bins>=3)
    case_numeric_ok=cf['torque_q50__median'].notna()&cf['pedal2_q50__median'].notna()
    datasets={};plan=dict(seed=SEED,source_units=203,source_cases=67,
        numeric_core=['torque','pedal2'],min_bins_each_signal=3,imputation=False,
        rpm='audit only; tiny high-RPM differences not standardized into main distance',
        scaling='case-balanced median/IQR; q10-q90/SD fallback; log1p for durations; cap normalized difference at3scales',
        family_weights='equal family means within each view; combined=0.5structure+0.5numeric',
        case_distance='median profiles; case IQR retained in exported descriptions, not imputed for one-block cases',
        candidates='kmedoids(case-balanced objective) and average linkage; k2..5; min5distinctcases',
        selection='valid sizes/balance then silhouette+.35*8-repeat median ARI; no forced acceptance',
        final_partition_repeats=60,neighbor_repeats=40,case_subsample_fraction=.8,
        neighbor_validation='reference/refit rankings compared within same remaining candidate cases',
        external_validation=False)
    (out/'analysis_plan.json').write_text(json.dumps(plan,indent=2))
    log(out,'preflight',primary_units=203,primary_cases=67,numeric_units=int(numeric_ok.sum()),numeric_cases=int(units[units.unit_id.isin(features.index[numeric_ok])].case_id.nunique()),case_numeric=int(case_numeric_ok.sum()))
    for level,table,catalog,scope,mask in [('block',features,cat,units,numeric_ok),('case',cf,cc,cu,case_numeric_ok)]:
        for view in ['structure','numeric','combined','context']:
            scope=scope.copy()
            ids=scope.unit_id if view in ['structure','context'] else scope.loc[scope.unit_id.isin(table.index[mask]),'unit_id']
            us=scope.set_index('unit_id').loc[ids].reset_index()
            raw=table.loc[ids].reset_index(drop=True)
            d,prep=distance(raw,catalog,us.case_id.to_numpy(str),view)
            datasets[(level,view)]=(raw,catalog,us,d,prep)
            pd.DataFrame(d,index=ids,columns=ids).to_csv(out/f'distances/{level}_{view}.csv',index_label='unit_id')
    tasks=[]
    for (level,view),(raw,catalog,us,d,prep) in datasets.items():
        if view=='context':continue
        for method in ['kmedoids','average']:
            for k in range(2,6):tasks.append((view,level,method,k,raw,catalog,us))
    result=Parallel(n_jobs=args.workers)(delayed(candidate)(view,level,method,k,raw,catalog,us,8)
        for view,level,method,k,raw,catalog,us in tasks)
    candidates=pd.DataFrame([r[0] for r in result]);candidates.to_csv(out/'model_candidates.csv',index=False)
    lookup={r[0]['model_id']:r[1] for r in result}
    winners=[]
    for key,g in candidates.groupby(['level','view','method']):
        pool=g[g.valid];pool=pool if len(pool) else g
        winners.append(pool.sort_values(['selection_score','k'],ascending=[False,True]).iloc[0].to_dict())
    trials=Parallel(n_jobs=args.workers)(delayed(partition_repeats)(datasets[(r['level'],r['view'])][0],
        datasets[(r['level'],r['view'])][1],datasets[(r['level'],r['view'])][2].case_id.to_numpy(str),
        r['view'],r['method'],int(r['k']),lookup[r['model_id']],60,SEED+1000) for r in winners)
    modelrows=[];alltrials=[];assignments=[];profiles=[];clusters=[]
    for r,trial in zip(winners,trials):
        level,view=r['level'],r['view'];raw,catalog,us,d,prep=datasets[(level,view)]
        labels=lookup[r['model_id']];values=np.array([x['ari'] for x in trial if x['valid_repeat']])
        r.update(stability_median=float(np.median(values)),stability_q25=float(np.quantile(values,.25)),
            stability_q75=float(np.quantile(values,.75)),stability_min=float(values.min()),valid_repeats=len(values),failed_repeats=60-len(values))
        for row in trial:alltrials.append(dict(model_id=r['model_id'],**row))
        qc=qualitycat if level=='block' else casequalitycat
        qd,_=distance(raw,qc,us.case_id.to_numpy(str),'quality')
        qlabels=fit(qd,us.case_id.to_numpy(str),'kmedoids',int(r['k']),SEED)
        r['quality_ari']=float(adjusted_rand_score(labels,qlabels))
        r['status']=('STRUCTURAL_RULE_FAILURE' if not r['valid'] else
                     'EXPLORATORY_UNSTABLE' if r['stability_median']<.75 else 'INTERNALLY_REPEATABLE_CANDIDATE')
        labelmap=dict(zip(us.unit_id,labels));full=units if level=='block' else cu
        for row in full.itertuples():
            assignments.append(dict(level=level,view=view,model_id=r['model_id'],unit_id=row.unit_id,case_id=row.case_id,
                cluster=labelmap.get(row.unit_id,np.nan),status='CLASSIFIED' if row.unit_id in labelmap else 'NO_NUMERIC_SUPPORT'))
        for cl in sorted(set(labels)):
            mask=labels==cl;group=raw.loc[mask]
            cluster_cases=us.loc[mask,'case_id'].nunique()
            clusters.append(dict(model_id=r['model_id'],cluster=int(cl),n_units=int(mask.sum()),n_cases=cluster_cases))
            for feature in catalog[catalog.view.isin(['structure','numeric'])].feature:
                vals=group[feature].dropna()
                profiles.append(dict(model_id=r['model_id'],cluster=int(cl),feature=feature,n_observed=len(vals),
                    n_cases=cluster_cases,raw_median=vals.median(),raw_q25=vals.quantile(.25),raw_q75=vals.quantile(.75)))
        modelrows.append(r)
    models=pd.DataFrame(modelrows);models.to_csv(out/'models.csv',index=False)
    pd.DataFrame(alltrials).to_csv(out/'stability.csv',index=False)
    pd.DataFrame(assignments).to_csv(out/'assignments.csv',index=False)
    pd.DataFrame(clusters).to_csv(out/'cluster_sizes.csv',index=False)
    pd.DataFrame(profiles).to_csv(out/'cluster_profiles.csv',index=False)
    log(out,'partitions_complete',models=len(models))
    neighbor_rows=[];neighbor_stability=[];coords=[];preprocessing={}
    for (level,view),(raw,catalog,us,d,prep) in datasets.items():
        cases=us.case_id.to_numpy(str);preprocessing[f'{level}_{view}']=prep
        for i,row in us.iterrows():
            js=nearest(d,cases,i)
            other=np.flatnonzero(cases!=cases[i]);casebest=pd.DataFrame(dict(case=cases[other],distance=d[i,other])).groupby('case').distance.min()
            ties=int(np.isclose(casebest,casebest.min(),atol=1e-10).sum())
            for rank,j in enumerate(js,1):
                neighbor_rows.append(dict(level=level,view=view,unit_id=row.unit_id,case_id=row.case_id,
                    neighbor_unit_id=us.unit_id.iloc[j],neighbor_case_id=cases[j],rank=rank,distance=float(d[i,j]),n_equal_best_cases=ties))
        nr=neighbor_repeats(raw,catalog,us,view,d)
        neighbor_stability.extend(dict(level=level,**r) for r in nr)
        with threadpool_limits(limits=1):
            xy=MDS(n_components=2,dissimilarity='precomputed',random_state=SEED,n_init=4,max_iter=400,normalized_stress='auto').fit_transform(d)
        coords.extend(dict(level=level,view=view,unit_id=row.unit_id,case_id=row.case_id,x=float(xy[i,0]),y=float(xy[i,1])) for i,row in us.iterrows())
        log(out,'neighbors_complete',level=level,view=view,n_units=len(us))
    neighbors=pd.DataFrame(neighbor_rows)
    neighbors.to_csv(out/'neighbors.csv',index=False)
    pd.DataFrame(neighbor_stability).to_csv(out/'neighbor_stability.csv',index=False)
    pd.DataFrame(coords).to_csv(out/'view_coordinates.csv',index=False)
    (out/'preprocessing.json').write_text(json.dumps(preprocessing,indent=2))
    # Compare metric axes using an identical candidate pool and refitted common scales.
    agreements=[]
    for level in ['block','case']:
        for a,b in [('structure','numeric'),('structure','context'),('numeric','context')]:
            da=datasets[(level,a)];db=datasets[(level,b)]
            common=da[2].set_index('unit_id').index.intersection(db[2].set_index('unit_id').index,sort=False)
            au=da[2].set_index('unit_id').loc[common].reset_index();acase=au.case_id.to_numpy(str)
            ar=da[0].set_axis(da[2].unit_id).loc[common].reset_index(drop=True)
            br=db[0].set_axis(db[2].unit_id).loc[common].reset_index(drop=True)
            ad,_=distance(ar,da[1],acase,a);bd,_=distance(br,db[1],acase,b)
            scores=[]
            for i in range(len(common)):
                aa={acase[j] for j in nearest(ad,acase,i)};bb={acase[j] for j in nearest(bd,acase,i)}
                scores.append(len(aa&bb)/len(aa|bb))
            tr=np.triu_indices(len(common),1)
            agreements.append(dict(level=level,view_a=a,view_b=b,n_common_units=len(common),n_cases=len(set(acase)),
                mean_top3_jaccard=float(np.mean(scores)),distance_spearman=float(spearmanr(ad[tr],bd[tr]).statistic)))
    pd.DataFrame(agreements).to_csv(out/'neighbor_agreement.csv',index=False)
    finish_outputs(out)
    log(out,'complete',models=len(models),neighbor_rows=len(neighbors))


def finish_outputs(out):
    """Recover only descriptive exports; saved partitions/distances are never refit."""
    neighbors=pd.read_csv(out/'neighbors.csv')
    prep=json.loads((out/'preprocessing.json').read_text())
    features=pd.read_csv(out/'features.csv').set_index('unit_id')
    units=pd.read_csv(out/'units.csv').set_index('unit_id')
    assert len(features)==203 and units.case_id.nunique()==67
    differences=[]
    for row in neighbors[(neighbors.level=='block')&(neighbors['rank']==1)].itertuples():
        for field in prep['block_'+row.view]:
            a=float(features.loc[row.unit_id,field['feature']]);b=float(features.loc[row.neighbor_unit_id,field['feature']])
            ta=np.log1p(max(a,0)) if field['transform']=='log1p' and np.isfinite(a) else a
            tb=np.log1p(max(b,0)) if field['transform']=='log1p' and np.isfinite(b) else b
            differences.append(dict(view=row.view,unit_id=row.unit_id,neighbor_unit_id=row.neighbor_unit_id,
                feature=field['feature'],query_raw=a,neighbor_raw=b,
                scaled_difference=min(abs(ta-tb)/field['scale']/3,1) if np.isfinite(ta) and np.isfinite(tb) else np.nan))
    pd.DataFrame(differences).to_csv(out/'neighbor_feature_differences.csv',index=False)
    rng=np.random.default_rng(SEED);examples=[]
    for view in ['structure','numeric','combined']:
        df=pd.read_csv(out/f'distances/block_{view}.csv',index_col=0)
        ids=df.index.to_numpy();d=df.to_numpy();cases=units.loc[ids,'case_id'].to_numpy(str)
        ns=neighbors[(neighbors.level=='block')&(neighbors.view==view)&(neighbors['rank']==1)].sort_values('distance')
        used=set();chosen=[]
        for row in ns.itertuples():
            if row.case_id in used or row.neighbor_case_id in used:continue
            used.update([row.case_id,row.neighbor_case_id]);chosen.append(row)
            if len(chosen)==5:break
        for row in chosen:
            examples.append(dict(view=view,query_unit_id=row.unit_id,neighbor_unit_id=row.neighbor_unit_id,
                query_case_id=row.case_id,neighbor_case_id=row.neighbor_case_id,distance=row.distance,
                kind='nearest',reason='different cases; selected near example, not unbiased performance proof'))
        if chosen:
            row=chosen[0];i=int(np.flatnonzero(ids==row.unit_id)[0])
            eligible=np.flatnonzero((cases!=cases[i])&(ids!=row.neighbor_unit_id))
            q=features.loc[ids,['torque_observed_fraction','pedal2_observed_fraction']].to_numpy(float)
            diff=np.nansum(np.abs(q[eligible]-q[i]),axis=1)
            pool=eligible[np.argsort(diff)[:max(5,len(eligible)//4)]];j=int(rng.choice(pool))
            examples.append(dict(view=view,query_unit_id=ids[i],neighbor_unit_id=ids[j],query_case_id=cases[i],
                neighbor_case_id=cases[j],distance=float(d[i,j]),kind='support_matched_random',
                reason='random distinct-case comparison, matched approximately on observation fractions'))
    # Explicit examples of why structural and numeric similarities are separate.
    nd=pd.read_csv(out/'distances/block_numeric.csv',index_col=0)
    ids=nd.index.to_numpy();n=nd.to_numpy()
    s=pd.read_csv(out/'distances/block_structure.csv',index_col=0).loc[ids,ids].to_numpy()
    cases=units.loc[ids,'case_id'].to_numpy(str)
    for view,near_d,other_d,kind in [('structure',s,n,'structure_close_numeric_far'),
                                    ('numeric',n,s,'numeric_close_structure_far')]:
        candidates=[]
        for i in range(len(ids)):
            other=np.flatnonzero(cases!=cases[i]);j=other[np.argmin(near_d[i,other])]
            percentile=float(np.mean(other_d[i,other]<=other_d[i,j]))
            candidates.append((percentile,i,j))
        seen=set();count=0
        for percentile,i,j in sorted(candidates,reverse=True):
            if percentile<.70 or cases[i] in seen or cases[j] in seen:continue
            seen.update([cases[i],cases[j]]);count+=1
            examples.append(dict(view=view,query_unit_id=ids[i],neighbor_unit_id=ids[j],query_case_id=cases[i],
                neighbor_case_id=cases[j],distance=float(near_d[i,j]),kind=kind,
                reason=f'nearest in named view; other-view distance at percentile {percentile:.3f} within common pool'))
            if count>=2:break
    pd.DataFrame(examples).to_csv(out/'pair_examples.csv',index=False)
    log(out,'descriptive_exports_completed',recovery='float-cast boolean raw context for difference export; saved models untouched',
        neighbor_rows=len(neighbors),pair_examples=len(examples))


if __name__=='__main__':main()
