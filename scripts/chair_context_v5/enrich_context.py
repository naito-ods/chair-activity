#!/usr/bin/env python3
"""Descriptive k=3 contextual profiles, held-out signal levels and QC."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

def eta2(values,labels):
    x=np.asarray(values,float);ok=np.isfinite(x);x=x[ok];labels=np.asarray(labels)[ok]
    if not len(x) or np.var(x)==0:return np.nan
    return float(sum(np.sum(labels==l)*(np.mean(x[labels==l])-np.mean(x))**2 for l in set(labels))/np.sum((x-np.mean(x))**2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('Output/chair_context_v5_20261104'));a=ap.parse_args();out=a.output
    if (out/'context_cluster_profiles.csv').exists():raise FileExistsError('Preserve previous output')
    f=pd.read_csv(out/'analysis_case_features.csv');ass=pd.read_csv(out/'context_cluster_assignments.csv');lab=ass[ass.k.eq(3)].set_index('case_id').cluster
    f['context_cluster']=f.case_id.map(lab);assert f.context_cluster.notna().all()
    d=pd.read_csv(out/'context_distances.csv',index_col=0).loc[f.case_id,f.case_id].to_numpy()
    rows=[];reps=[]
    cols=['hp3_observed','hp3_2s','hp1_2s','hp3_before_first_high','hp3_after_last_high','high_hp3_high','macro_mid','macro_mid_before','macro_mid_between','macro_mid_after','macro_mid_scattered','n_mid_dominant_blocks','n_mid_dominant_runs','low_fraction','mid_fraction','high_fraction','first_high_progress','last_high_progress','hp_known_fraction','hp2_observed_coverage','duration_s']
    for cl,g in f.groupby('context_cluster'):
        ix=g.index.to_numpy();med=int(ix[np.argmin(d[np.ix_(ix,ix)].sum(1))]);order=ix[np.argsort(d[med,ix],kind='stable')]
        reps.extend(dict(candidate=f'exploratory_k3_C{int(cl)}',case_id=f.case_id.iloc[j],role='medoid' if j==med else 'member',distance_to_medoid=d[med,j]) for j in order[:6])
        for col in cols:
            x=g[col].astype(float)
            rows.append(dict(cluster=int(cl),n_cases=len(g),feature=col,n_observed=int(x.notna().sum()),median=x.median(),q25=x.quantile(.25),q75=x.quantile(.75),mean=x.mean()))
    pd.DataFrame(rows).to_csv(out/'context_cluster_profiles.csv',index=False)
    prev=pd.read_csv(out/'representatives.csv');pd.concat([prev,pd.DataFrame(reps)],ignore_index=True).to_csv(out/'representatives_with_clusters.csv',index=False)
    # Raw paired values at actual RPM bins. No interpolation or imputation.
    manifest=pd.read_csv(out/'source_manifest.csv').set_index('table');p=pd.read_csv(manifest.loc['hp2_points','path'],low_memory=False);p=p[p.case_id.isin(f.case_id)]
    levels=[]
    for (cid,zone),g in p.groupby(['case_id','zone']):
        for code in ['torque','pedal2','bulk','pedal1']:
            x=g[code].dropna();levels.append(dict(case_id=cid,zone=zone,signal=code,n_bins=len(x),median=float(x.median()) if len(x)>=3 else np.nan,q25=float(x.quantile(.25)) if len(x)>=3 else np.nan,q75=float(x.quantile(.75)) if len(x)>=3 else np.nan))
    level=pd.DataFrame(levels);level['cluster']=level.case_id.map(lab);level.to_csv(out/'raw_numeric_by_case_zone.csv',index=False)
    levsum=level.groupby(['cluster','zone','signal']).agg(n_supported_cases=('median','count'),median_of_case_medians=('median','median'),q25=('median',lambda x:x.quantile(.25)),q75=('median',lambda x:x.quantile(.75))).reset_index();levsum.to_csv(out/'raw_numeric_by_cluster_zone.csv',index=False)
    quality={col:dict(eta2=eta2(f[col],f.context_cluster),cluster_medians=f.groupby('context_cluster')[col].median().to_dict()) for col in ['hp_known_fraction','hp2_observed_coverage','duration_s']}
    qraw=f[['hp_known_fraction','hp2_observed_coverage']].to_numpy();qd=np.mean(np.abs(qraw[:,None,:]-qraw[None,:,:]),axis=2);np.fill_diagonal(qd,0)
    import importlib.util
    spec=importlib.util.spec_from_file_location('context_fit',Path(__file__).resolve().parents[1]/'chair_hp2_similarity_v4/compare_similarity.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    qlabels=mod.fit(qd,f.case_id.to_numpy(),'kmedoids',3,20261104)
    quality['missingness_only_partition_ari']=float(adjusted_rand_score(f.context_cluster,qlabels))
    tri=np.triu_indices(len(f),1);quality['distance_vs_quality_spearman']=float(spearmanr(d[tri],qd[tri]).statistic)
    g=pd.read_csv(out/'case_context_groups.csv');sub=g[g.case_id.isin(f.loc[f.hp_known_fraction.ge(.8),'case_id'])]
    sub.groupby(['group_id','group_label']).size().rename('n_cases').reset_index().to_csv(out/'known80_group_counts.csv',index=False)
    quality['known80_cases']=len(sub);quality['known50_cases']=int(f.hp_known_fraction.ge(.5).sum())
    prior=Path('Output/chair_hp2_similarity_v4_20261104/case_structure_candidates.csv')
    if prior.exists():
        old=pd.read_csv(prior)
        # Compare only if schema explicitly exposes a cluster field.
        clustercol=next((x for x in ['cluster','structure_cluster'] if x in old),None)
        if clustercol:
            joined=f[['case_id','context_cluster']].merge(old[['case_id',clustercol]],on='case_id',validate='one_to_one')
            quality['previous_structure_common_cases']=len(joined);quality['previous_structure_ari']=float(adjusted_rand_score(joined.context_cluster,joined[clustercol]))
    (out/'context_quality_audit.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2)+'\n')
    cross=f[['case_id','context_cluster']].merge(g[['case_id','group_id']],on='case_id');pd.crosstab(cross.context_cluster,cross.group_id).to_csv(out/'context_cluster_rule_cross.csv')
    summary=json.loads((out/'analysis_summary.json').read_text());summary['context_quality_audit']=quality
    summary['report_paragraphs'] += [
        '文脈クラスタの探索ではk=3が比較したk=2〜6の中でsilhouetteと内部ARIの中央値が最大でした（silhouette約0.354、症例抽出ARI中央値約0.817、四分位0.734–0.878）。3群を補助的に表示します。固定した4比較群とは異なる分割であり、独立した処置分類ではありません。',
        'トルク・ペダル2・バルク等の生の値は回転領域別に症例内中央値を計算し、文脈3群間で別途比較します。これらの信号は今回の文脈距離には入れていないため、数値水準が追随するかを見る補助評価です。各症例・領域・信号で3対応観測bin未満は中央値を欠測として残します。',
        f"HP状態既知率80%以上に限定した比較表も保存しました（{len(sub)}症例）。欠測情報のみの3群とのARIは{quality['missingness_only_partition_ari']:.3f}、文脈距離と品質距離のSpearman相関は{quality['distance_vs_quality_spearman']:.3f}です。品質影響が完全に除去された証明にはなりません。"]
    (out/'analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (out/'process/analysis_log.jsonl').open('a') as h:h.write(json.dumps(dict(event='context_profiles_and_numeric_audit_complete',primary_cases=len(f),k=3,raw_numeric_case_zone_signal_rows=len(level)))+'\n')
    print(json.dumps(quality,ensure_ascii=False));print(pd.crosstab(cross.context_cluster,cross.group_id).to_string())
if __name__=='__main__':main()
