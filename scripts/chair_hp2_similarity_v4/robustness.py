"""Support, representation, family-weight and same-case granularity audits."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score

from compare_similarity import distance,fit,nearest,case_weights,quantile,log,SEED


def eta(values,labels):
    values=np.asarray(values,float);den=np.sum((values-values.mean())**2)
    return float(sum(sum(labels==cl)*(values[labels==cl].mean()-values.mean())**2 for cl in set(labels))/den) if den else 0.


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output.resolve()
    features=pd.read_csv(out/'features.csv').set_index('unit_id');cat=pd.read_csv(out/'feature_catalog.csv')
    units=pd.read_csv(out/'units.csv').set_index('unit_id')
    cases=pd.read_csv(out/'case_features.csv').set_index('case_id');cc=pd.read_csv(out/'case_feature_catalog.csv')
    models=pd.read_csv(out/'models.csv');assign=pd.read_csv(out/'assignments.csv')
    plan=dict(support_conditions=[[1,0],[5,0],[10,0],[3,.1],[3,.2]],
        structure_conditions=['remove_duration','remove_off_gap','remove_repetition'],
        combined_structure_weights=[.25,.75],
        scope='Sensitivity, not tuning for a higher score. Saved labels stay unchanged.',
        stage_comparison='same cases, identical median-profile feature definitions; primary-containing stages, no new stage-high selection')
    (out/'robustness_plan.json').write_text(json.dumps(plan,indent=2))
    sensitivity=[]
    for minimum,fraction in plan['support_conditions']:
        ok=(features.torque_n_bins>=minimum)&(features.pedal2_n_bins>=minimum)&(
            features.torque_observed_fraction>=fraction)&(features.pedal2_observed_fraction>=fraction)
        ids=features.index[ok];raw=features.loc[ids];casearray=units.loc[ids,'case_id'].to_numpy(str)
        for view in ['numeric','combined']:
            d,_=distance(raw,cat,casearray,view)
            for model in models[(models.level=='block')&(models.view==view)].itertuples():
                labels=fit(d,casearray,model.method,int(model.k),SEED)
                reference=assign[(assign.model_id==model.model_id)&assign.cluster.notna()].set_index('unit_id').cluster
                common=ids.intersection(reference.index)
                labelmap=pd.Series(labels,index=ids)
                sensitivity.append(dict(scenario=f'bins{minimum}_fraction{fraction}',model_id=model.model_id,
                    level='block',view=view,n_units=len(ids),n_cases=len(set(casearray)),n_common_units=len(common),
                    ari=float(adjusted_rand_score(reference.loc[common],labelmap.loc[common])),k_actual=len(set(labels))))
    for level,raw,catalog,scope in [('block',features,cat,units),('case',cases,cc,pd.DataFrame({'case_id':cases.index},index=cases.index))]:
        if level=='case':catalog=catalog[(~catalog.feature.str.endswith('__iqr'))&(~catalog.block.eq('case_structure'))].reset_index(drop=True)
        casearray=scope.case_id.to_numpy(str)
        for scenario,family in [('remove_duration','duration'),('remove_off_gap','off_gap'),('remove_repetition','repetition')]:
            sub=catalog[catalog.block!=family]
            d,_=distance(raw,sub,casearray,'structure')
            for model in models[(models.level==level)&(models.view=='structure')].itertuples():
                labels=fit(d,casearray,model.method,int(model.k),SEED)
                reference=assign[assign.model_id==model.model_id].set_index('unit_id').cluster.loc[raw.index]
                sensitivity.append(dict(scenario=scenario,model_id=model.model_id,level=level,view='structure',
                    n_units=len(raw),n_cases=len(set(casearray)),n_common_units=len(raw),
                    ari=float(adjusted_rand_score(reference,labels)),k_actual=len(set(labels))))
    # Changing combined weights is a diagnosis of dependence, not parameter selection.
    for level in ['block','case']:
        modelrows=models[(models.level==level)&(models.view=='combined')]
        raw,catalog,scope=(features,cat,units) if level=='block' else (cases,cc,pd.DataFrame({'case_id':cases.index},index=cases.index))
        ids=assign[(assign.model_id==modelrows.iloc[0].model_id)&assign.cluster.notna()].unit_id
        raw=raw.loc[ids];scope=scope.loc[ids];casearray=scope.case_id.to_numpy(str)
        if level=='case':catalog=catalog[(~catalog.feature.str.endswith('__iqr'))&(~catalog.block.eq('case_structure'))]
        s,_=distance(raw,catalog,casearray,'structure');n,_=distance(raw,catalog,casearray,'numeric')
        for alpha in [.25,.75]:
            d=alpha*s+(1-alpha)*n
            for model in modelrows.itertuples():
                labels=fit(d,casearray,model.method,int(model.k),SEED)
                reference=assign[(assign.model_id==model.model_id)&assign.cluster.notna()].set_index('unit_id').cluster.loc[ids]
                sensitivity.append(dict(scenario=f'combined_structure_weight{alpha}',model_id=model.model_id,level=level,
                    view='combined',n_units=len(ids),n_cases=len(set(casearray)),n_common_units=len(ids),
                    ari=float(adjusted_rand_score(reference,labels)),k_actual=len(set(labels))))
    pd.DataFrame(sensitivity).to_csv(out/'support_and_weight_sensitivity.csv',index=False)
    # Safe numeric interpretation in every model, including structure-only groups.
    profiles=[];representatives=[]
    for model in models.itertuples():
        group=assign[(assign.model_id==model.model_id)&assign.cluster.notna()]
        raw,catalog=(features,cat) if model.level=='block' else (cases,cc)
        d=pd.read_csv(out/f'distances/{model.level}_{model.view}.csv',index_col=0)
        for cl,part in group.groupby('cluster'):
            ids=part.unit_id.to_numpy();values=raw.loc[ids].copy()
            if model.level=='block':
                for code in ['torque','pedal2','bulk']:
                    fields=catalog[(catalog.view=='numeric')&(catalog.block==code)].feature
                    values.loc[values[code+'_n_bins']<3,fields]=np.nan
            for field in catalog[catalog.view.isin(['structure','numeric'])].feature:
                present=values[field].notna();valid=values.loc[present,field]
                w=case_weights(part.set_index('unit_id').loc[valid.index,'case_id'].to_numpy(str)) if len(valid) else np.array([])
                profiles.append(dict(model_id=model.model_id,cluster=int(cl),feature=field,n_units=len(part),
                    n_observed=len(valid),n_cases=part.case_id.nunique(),
                    n_supported_cases=part.set_index('unit_id').loc[valid.index,'case_id'].nunique(),
                    raw_median=valid.median(),raw_q25=valid.quantile(.25),raw_q75=valid.quantile(.75),
                    case_balanced_median=quantile(valid.to_numpy(float),w,.5) if len(valid) else np.nan))
            matrix=d.loc[ids,ids].to_numpy();weights=case_weights(part.case_id.to_numpy(str));cost=matrix@weights
            med=int(np.argmin(cost));order=np.argsort(matrix[med],kind='stable');chosen=[med];seen={part.iloc[med].case_id}
            for j in order:
                if part.iloc[j].case_id in seen:continue
                seen.add(part.iloc[j].case_id);chosen.append(int(j))
                if len(chosen)==5:break
            for rank,j in enumerate(chosen):
                representatives.append(dict(model_id=model.model_id,cluster=int(cl),rank=rank,unit_id=ids[j],
                    case_id=part.iloc[j].case_id,is_medoid=j==med,distance=float(matrix[med,j])))
    pd.DataFrame(profiles).to_csv(out/'cluster_profiles_supported.csv',index=False)
    pd.DataFrame(representatives).to_csv(out/'model_representatives.csv',index=False)
    # Method agreement is meaningful only within the same unit level and common pool.
    for level in ['block','case']:
        mids=models.loc[models.level==level,'model_id'];matrix=pd.DataFrame(index=mids,columns=mids,dtype=float)
        for a in mids:
            av=assign[(assign.model_id==a)&assign.cluster.notna()].set_index('unit_id').cluster
            for b in mids:
                bv=assign[(assign.model_id==b)&assign.cluster.notna()].set_index('unit_id').cluster
                common=av.index.intersection(bv.index)
                matrix.loc[a,b]=adjusted_rand_score(av.loc[common],bv.loc[common])
        matrix.to_csv(out/f'{level}_model_comparison_ari.csv',index_label='model_id')
    # Granularity: compare typical high-containing stage vs typical high block on same cases.
    sf=pd.read_csv(out/'stage_features.csv').set_index('unit_id');su=pd.read_csv(out/'stage_units.csv').set_index('unit_id')
    sc=pd.read_csv(out/'stage_feature_catalog.csv')
    stagecase=[]
    for case,group in su.groupby('case_id'):
        row=dict(case_id=case)
        for field in sc[sc.view.isin(['structure','numeric'])].itertuples():
            vals=sf.loc[group.index,field.feature]
            if field.view=='numeric':vals=vals[sf.loc[group.index,field.block+'_n_bins']>=3]
            row[field.feature+'__median']=vals.median()
        stagecase.append(row)
    stagecase=pd.DataFrame(stagecase).set_index('case_id');stagecase.to_csv(out/'stage_case_profiles.csv',index_label='case_id')
    comparison=[]
    for view in ['structure','numeric','combined']:
        usecat=cc[cc.feature.isin(stagecase.columns)].copy()
        usecat=usecat[usecat.view.isin(['structure','numeric'])]
        ids=cases.index if view=='structure' else cases.index[(cases.torque_q50__median.notna()&cases.pedal2_q50__median.notna())]
        if view!='structure':ids=ids.intersection(stagecase.index[stagecase.torque_q50__median.notna()&stagecase.pedal2_q50__median.notna()])
        raw=cases.loc[ids];sr=stagecase.loc[ids];casearray=ids.to_numpy(str)
        a,_=distance(raw,usecat,casearray,view);b,_=distance(sr,usecat,casearray,view)
        overlaps=[]
        for i in range(len(ids)):
            aa=set(nearest(a,casearray,i));bb=set(nearest(b,casearray,i))
            overlaps.append(len(aa&bb)/len(aa|bb))
        model=models[(models.level=='case')&(models.view==view)&(models.method=='kmedoids')].iloc[0]
        la=fit(a,casearray,'kmedoids',int(model.k),SEED);lb=fit(b,casearray,'kmedoids',int(model.k),SEED)
        tr=np.triu_indices(len(ids),1)
        comparison.append(dict(view=view,n_common_cases=len(ids),mean_top3_jaccard=float(np.mean(overlaps)),
            distance_spearman=float(spearmanr(a[tr],b[tr]).statistic),partition_ari=float(adjusted_rand_score(la,lb))))
    pd.DataFrame(comparison).to_csv(out/'stage_neighbor_comparison.csv',index=False)
    # Neighbor means with equal query-case weight, and opportunity bias from case size.
    ns=pd.read_csv(out/'neighbor_stability.csv');summaries=[]
    for (level,view),g in ns.groupby(['level','view']):
        caseavg=g.groupby('case_id')[['conditional_neighbor_retention','top1_retention']].mean()
        summaries.append(dict(level=level,view=view,n_units=len(g),n_cases=len(caseavg),
            mean_case_balanced_retention=caseavg.conditional_neighbor_retention.mean(),
            median_case_balanced_retention=caseavg.conditional_neighbor_retention.median(),
            median_case_balanced_top1=caseavg.top1_retention.median()))
    pd.DataFrame(summaries).to_csv(out/'neighbor_stability_summary.csv',index=False)
    neighbors=pd.read_csv(out/'neighbors.csv');opportunity=[]
    nblocks=units.groupby('case_id').size()
    for view,g in neighbors[neighbors.level=='block'].groupby('view'):
        received=g.groupby('neighbor_case_id').size().reindex(nblocks.index,fill_value=0)
        opportunity.append(dict(view=view,spearman_case_blockcount_vs_neighbor_selections=float(spearmanr(nblocks,received).statistic),
            note='More candidate blocks per case give more chances to be nearest; independent case-level analysis is also provided'))
    pd.DataFrame(opportunity).to_csv(out/'neighbor_opportunity_audit.csv',index=False)
    # Dates and original case durations are recorded separately from intended block-duration traits.
    source=out.parent/'chair_hp_sequence_300_v3_20261104'
    manifest=pd.read_csv(source/'case_manifest.csv').set_index('case_id')
    confounds=[]
    for model in models.itertuples():
        g=assign[(assign.model_id==model.model_id)&assign.cluster.notna()]
        dates=manifest.loc[g.case_id,'date'].astype(str).str[:4].to_numpy();dur=manifest.loc[g.case_id,'duration_s'].to_numpy(float)
        labels=g.cluster.to_numpy(int)
        confounds.append(dict(model_id=model.model_id,year_nmi=float(normalized_mutual_info_score(dates,labels)),
            original_case_duration_eta2=eta(dur,labels),source_case_highblock_count_eta2=eta(nblocks.loc[g.case_id],labels),
            single_device=True))
    pd.DataFrame(confounds).to_csv(out/'confound_diagnostics.csv',index=False)
    log(out,'robustness_complete',support_conditions=plan['support_conditions'],stage=comparison)


if __name__=='__main__':main()
