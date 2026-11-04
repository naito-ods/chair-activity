#!/usr/bin/env python3
"""Whole-case contextual phenotypes and separate exploratory context clustering.

Inputs are compact v3/v5 exports. Original Stage 2 assignments are never read or
changed. Rule-defined phenotype separation is not evidence of validation.
"""
from pathlib import Path
import argparse, importlib.util, json, datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, silhouette_score

SEED=20261104

def wilson(k,n):
    if not n:return np.nan,np.nan
    z=1.96; p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den; half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half,center+half

def context_distance(frame,families):
    """Fixed semantic ranges, mean absolute differences per family, no filling."""
    matrices=[]
    for _,columns in families.items():
        x=frame[columns].to_numpy(float)
        valid=np.isfinite(x[:,None,:])&np.isfinite(x[None,:,:])
        diff=np.abs(x[:,None,:]-x[None,:,:])
        counts=valid.sum(2)
        matrices.append(np.divide(np.nansum(np.where(valid,diff,np.nan),2),counts,
                                  out=np.full(counts.shape,np.nan),where=counts>0))
    d=np.nanmean(matrices,axis=0)
    if not np.isfinite(d).all():raise ValueError('No comparable context for a pair')
    np.fill_diagonal(d,0)
    return d

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('Output/chair_context_v5_20261104'));ap.add_argument('--repeats',type=int,default=60);a=ap.parse_args();out=a.output
    if (out/'analysis_summary.json').exists():raise FileExistsError('Use a new output directory')
    f=pd.read_csv(out/'case_context_features.csv');scope=pd.read_csv(out/'cohort_scope.csv')
    source=pd.read_csv(out/'source_manifest.csv').set_index('table')
    blocks=pd.read_csv(source.loc['hp2_blocks','path']);runs=pd.read_csv(source.loc['hp_runs','path'])
    main_ids=scope.loc[scope.included_context_high.astype(bool),'case_id']
    f=f.set_index('case_id').loc[main_ids].reset_index();assert len(f)==97 and f.case_id.is_unique
    # Recompute transparent grouping quantities directly from compact sources.
    for hp in [1,3]:
        r=runs[runs.hp.eq(hp)];stats=r.groupby('case_id').active_s.agg(['sum','max','size'])
        for name,col in [('active_s','sum'),('longest_run_s','max'),('n_runs','size')]:
            f[f'hp{hp}_{name}']=f.case_id.map(stats[col]).fillna(0)
    features=[]
    for row in f.itertuples():
        q=blocks[blocks.case_id.eq(row.case_id)];m=q[q.supported.astype(bool)&q.mid_fraction.ge(.6)]
        l=q[q.supported.astype(bool)&q.low_fraction.ge(.6)]
        features.append(dict(case_id=row.case_id,macro_mid_before=int((m.end_s<row.first_high_s).any()),
            macro_mid_after=int((m.start_s>row.last_high_s).any()),
            macro_mid_between=int(((m.start_s>row.first_high_s)&(m.end_s<row.last_high_s)).any()),
            n_macro_low_blocks=len(l),macro_low_before=int((l.end_s<row.first_high_s).any()),
            macro_low_after=int((l.start_s>row.last_high_s).any()),
            macro_low_between=int(((l.start_s>row.first_high_s)&(l.end_s<row.last_high_s)).any())))
    f=f.merge(pd.DataFrame(features),on='case_id',validate='one_to_one')
    f['macro_mid']=f.n_mid_dominant_blocks.ge(1).astype(int)
    f['run_mid']=f.n_mid_dominant_runs.ge(1).astype(int)
    f['hp3_2s']=f.hp3_active_s.ge(2).astype(int);f['hp3_5s']=f.hp3_active_s.ge(5).astype(int)
    f['hp1_2s']=f.hp1_active_s.ge(2).astype(int)
    f['low_known_state']=f.hp_known_fraction.lt(.5)
    f['macro_mid_scattered']=f.n_mid_dominant_blocks.ge(2)&f.macro_mid_between.eq(1)
    f['ramp_only_observed_mid']=f.n_mid_interior_runs.eq(0)&f.any_mid_observed.eq(1)
    # One cross table for comparison; absence is always 'not observed'.
    groups=[]
    for r in f.itertuples():
        hp3=bool(r.hp3_2s); mid=bool(r.macro_mid)
        gid=f'HP3_{int(hp3)}__MID_{int(mid)}'
        label=('HP3 >=2s observed' if hp3 else 'HP3 >=2s not observed')+' / '+('MID-dominant block observed' if mid else 'MID-dominant block not observed')
        groups.append(dict(case_id=r.case_id,group_id=gid,group_label=label,hp_state_support='LOW_KNOWN_LT50%' if r.low_known_state else 'KNOWN_GE50%',old_high_dominant=r.old_high_dominant))
    g=pd.DataFrame(groups);g.to_csv(out/'case_context_groups.csv',index=False)
    gs=g.groupby(['group_id','group_label']).agg(n_cases=('case_id','size'),low_known_cases=('hp_state_support',lambda s:int(s.eq('LOW_KNOWN_LT50%').sum())),old_high_dominant_cases=('old_high_dominant','sum')).reset_index().rename(columns={'group_label':'label'})
    gs.to_csv(out/'context_group_summary.csv',index=False)
    definitions=[
        ('hp3_any','HP3 observed at any duration',f.hp3_observed.eq(1),'At least one observed HP3 positive interval'),
        ('hp3_2s','HP3 >=2s observed',f.hp3_2s.eq(1),'Total HP3 active time >=2s, whole case'),
        ('hp3_before','HP3 before first H',f.hp3_before_first_high.eq(1),'Observed HP3 before first H; unknown-state barrier respected'),
        ('hp3_after','HP3 after last H',f.hp3_after_last_high.eq(1),'Observed HP3 after last H; unknown-state barrier respected'),
        ('hp3_between','H -> HP3 -> H',f.high_hp3_high.eq(1),'HP3 between first/last H, no unknown HP interval across reference span'),
        ('mid_macro','MID-dominant block',f.macro_mid.eq(1),'v3 supported 30s-gap HP2 block with MID positive RPM proxy fraction >=0.6'),
        ('mid_scattered','Repeated MID blocks among H',f.macro_mid_scattered,'At least two MID-dominant blocks, one fully between first and last H'),
        ('mid_before','MID block before H',f.macro_mid_before.eq(1),'Supported MID-dominant block ends before first H'),
        ('mid_after','MID block after H',f.macro_mid_after.eq(1),'Supported MID-dominant block starts after last H'),
        ('mid_between','MID block between H',f.macro_mid_between.eq(1),'Supported MID-dominant block fully between first and last H'),
        ('mid_run','MID-dominant atomic run',f.run_mid.eq(1),'Dominant MID in supported atomic HP2 run; finer-granularity sensitivity'),
        ('mid_ramp_only','MID only at observed run edges',f.ramp_only_observed_mid,'Every MID observation within 3s of its run start or end; ramp candidate, not proof'),
        ('post_hp3_low','HP3 -> LOW-dominant HP2',f.old_post_hp3_low.eq(True),'Existing v3 primary post-HP3 LOW-dominant HP2 block rule'),
        ('high_tail_mid','H -> short MID tail',f.old_high_then_short_mid.eq(True),'Existing v3 corrected MID tail primary rule'),
        ('hp1','HP1 >=2s insertion',f.hp1_2s.eq(1),'Whole-case HP1 total active time >=2s'),
        ('low_high_low','L -> H -> L observed',f.motif_low_high_low.eq(1),'Ordered observed subsequence, unknown HP/RPM gaps >2s reset; not continuous sustained phases'),
    ]
    candidate_rows=[];masks={};members=[]
    rng=np.random.default_rng(SEED)
    for key,label,mask,definition in definitions:
        mask=np.asarray(mask,bool);masks[key]=mask;n=int(mask.sum());lo,hi=wilson(n,len(f))
        boot=np.mean(mask[rng.integers(0,len(f),size=(2000,len(f)))],axis=1)
        candidate_rows.append(dict(candidate=key,label=label,n_cases=n,denominator=len(f),percent=100*n/len(f),ci_low=100*lo,ci_high=100*hi,bootstrap_low=100*np.quantile(boot,.025),bootstrap_high=100*np.quantile(boot,.975),old_high_dominant_n=int((mask&f.old_high_dominant.eq(True)).sum()),low_known_n=int((mask&f.low_known_state).sum()),definition=definition))
        members.extend(dict(candidate=key,case_id=f.case_id.iloc[i]) for i in np.flatnonzero(mask))
    cs=pd.DataFrame(candidate_rows);cs.to_csv(out/'candidate_summary.csv',index=False);pd.DataFrame(members).to_csv(out/'candidate_members.csv',index=False)
    pair=[]
    for x,mx in masks.items():
        for y,my in masks.items():pair.append(dict(candidate=x,other=y,intersection=int((mx&my).sum()),union=int((mx|my).sum()),jaccard=float((mx&my).sum()/(mx|my).sum()) if (mx|my).any() else np.nan))
    pd.DataFrame(pair).to_csv(out/'candidate_overlap.csv',index=False)
    # Fixed semantic distances: categorical flags and bounded raw fractions;
    # equal families prevent many correlated MID flags dominating HP context.
    raw=f.copy()
    raw['mid_block_count_bounded']=np.minimum(raw.n_mid_dominant_blocks,5)/5
    raw['hp3_active_share']=raw.hp3_active_s/raw.duration_s
    families={'hp_presence':['hp3_2s','hp1_2s','hp3_active_share'],
              'hp_order':['hp3_before_first_high','hp3_after_last_high','high_hp3_high'],
              'rpm_composition':['low_fraction','mid_fraction','high_fraction','macro_mid','mid_block_count_bounded'],
              'rpm_placement':['macro_mid_before','macro_mid_between','macro_mid_after','macro_low_before','macro_low_between','macro_low_after'],
              'high_location':['first_high_progress','last_high_progress']}
    d=context_distance(raw,families);ids=f.case_id.to_numpy(str)
    pd.DataFrame(d,index=ids,columns=ids).to_csv(out/'context_distances.csv',index_label='case_id')
    near=[];reps=[]
    for i,cid in enumerate(ids):
        order=[j for j in np.argsort(d[i],kind='stable') if j!=i][:5]
        near.extend(dict(case_id=cid,neighbor_case_id=ids[j],rank=rank,distance=d[i,j]) for rank,j in enumerate(order,1))
    pd.DataFrame(near).to_csv(out/'context_neighbors.csv',index=False)
    scopes={**masks,**{key:g.group_id.eq(key).to_numpy() for key in g.group_id.unique()}}
    for key,mask in scopes.items():
        ix=np.flatnonzero(mask)
        if not len(ix):continue
        med=int(ix[np.argmin(d[np.ix_(ix,ix)].sum(1))]);order=ix[np.argsort(d[med,ix],kind='stable')]
        reps.extend(dict(candidate=key,case_id=ids[j],role='medoid' if j==med else 'member',distance_to_medoid=d[med,j]) for j in order[:6])
    pd.DataFrame(reps).to_csv(out/'representatives.csv',index=False)
    # Reuse independently tested k-medoids implementation; one row per case.
    previous=Path(__file__).resolve().parents[1]/'chair_hp2_similarity_v4/compare_similarity.py'
    spec=importlib.util.spec_from_file_location('similarity_v4',previous);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    models=[];assign=[];trials=[]
    for k in range(2,7):
        labels=module.fit(d,ids,'kmedoids',k,SEED)
        sizes=np.bincount(labels)[1:];aris=[]
        for rep in range(a.repeats):
            ix=np.sort(np.random.default_rng(SEED+1000+rep).choice(len(f),round(.8*len(f)),replace=False))
            rd=context_distance(raw.iloc[ix],families)
            rl=module.fit(rd,ids[ix],'kmedoids',k,SEED+rep+1)
            ari=adjusted_rand_score(labels[ix],rl);aris.append(ari);trials.append(dict(k=k,repeat=rep,n_cases=len(ix),ari=ari))
        models.append(dict(k=k,n_cases=len(f),min_group=int(sizes.min()),max_group=int(sizes.max()),silhouette=silhouette_score(d,labels,metric='precomputed'),ari_median=float(np.median(aris)),ari_q25=float(np.quantile(aris,.25)),ari_q75=float(np.quantile(aris,.75)),rule_cross_ari=adjusted_rand_score(g.group_id,labels),valid_size=bool(sizes.min()>=5)))
        assign.extend(dict(k=k,case_id=c,cluster=int(l)) for c,l in zip(ids,labels))
    models=pd.DataFrame(models);models.to_csv(out/'context_model_comparison.csv',index=False)
    pd.DataFrame(assign).to_csv(out/'context_cluster_assignments.csv',index=False);pd.DataFrame(trials).to_csv(out/'context_case_subsampling.csv',index=False)
    # Main rule-defined grouping sensitivity, not resampling "stability".
    sensitivity=[];base=g.group_id
    for hpthresh in [0,2,5]:
        hp=raw.hp3_active_s.gt(0) if hpthresh==0 else raw.hp3_active_s.ge(hpthresh)
        for level,mid in [('block',raw.macro_mid.eq(1)),('run',raw.run_mid.eq(1))]:
            alt=hp.astype(str)+'_'+mid.astype(str)
            sensitivity.append(dict(hp3_min_s=hpthresh,mid_unit=level,hp3_n=int(hp.sum()),mid_n=int(mid.sum()),group_ari_vs_primary=adjusted_rand_score(base,alt)))
    pd.DataFrame(sensitivity).to_csv(out/'context_definition_sensitivity.csv',index=False)
    hs=[]
    for n in [1,3,5,10]:
        mask=(scope.high_n_bins>=n)&(scope.high_equivalent_s>=2)
        hs.append(dict(min_high_bins=n,min_high_equivalent_s=2,n_cases=int(mask.sum()),primary_retained=int(scope.loc[mask,'case_id'].isin(ids).sum())))
    pd.DataFrame(hs).to_csv(out/'high_support_sensitivity.csv',index=False)
    f.to_csv(out/'analysis_case_features.csv',index=False)
    grouptext='; '.join(f"{r.label}: {r.n_cases}" for r in gs.itertuples())
    summary=dict(version='context-v5',cohort_cases=300,any_high_cases=99,primary_high_cases=97,previous_high_dominant_cases=67,additional_high_cases=int((~f.old_high_dominant.eq(True)).sum()),n_figures_expected_case=97,groups=gs.to_dict('records'),candidates=cs.to_dict('records'),exploratory_models=models.to_dict('records'),context_distance_families=families,report_paragraphs=[
        '前回は高回転区間の反復・連続性とトルク等を主軸にしたため、症例全体の前後文脈による分類を十分に実施していませんでした。今回は文脈を主軸とした別バージョンです。',
        '固定300症例を監査し、HP2高回転36000–44000rpmを3観測bin以上かつ2観測相当秒以上持つ97症例を主対象としました。前回の高回転主体67症例を含みます。高回転1点以上は99症例です。',
        '中回転の点は全97症例で観測され、単なる有無は区分に使えません。主比較は中回転主体の30秒ギャップ区間とし、短い立上り・停止時の通過とは分けて監査しました。',
        'HP3総作動2秒以上 × 中回転主体区間の観測有無の比較表: '+grouptext,
        'これらの比較群は解釈可能なルールで定義した候補です。HP3・中回転・低回転・HP1挿入は重複して現れます。きれいに定義できること自体は自然なクラス分離や処置名の検証にはなりません。',
        'HP状態が不明な区間があるため、HP3等の未観測は未使用を意味しません。低既知率50%未満の症例数と群別内訳を併記しました。数値の観測相当秒は連続測定時間ではありません。',
        '文脈距離はHP種類、HP3順序、回転領域構成、領域の前後配置、高回転の症例内位置を各同じ重みで比較します。DTWは使用していません。異なる症例のメドイドと近傍を表示します。',
        '探索的な文脈クラスタリングもk=2〜6で比較し、症例80%抽出60反復のARIを出力しました。ルール群の閾値感度とクラスタの内部安定性は別指標です。独立症例・処置記録による外部検証は未実施です。'],external_validation=False,seed=SEED,repeats=a.repeats)
    (out/'analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    plan=dict(seed=SEED,source='compact v3 and v5 only',context_distance=families,distance='bounded semantic feature L1, equal-family means, missing pairwise exclusion',primary_rules='HP3 total>=2s x supported MID-dominant30sblock >=1',clustering='k-medoids k2..6, 8 initializations, 60 case80% repeats; no automatic claim of preferred k',prevalence='Wilson95% and 2000 case bootstrap intervals',missing_policy='NaN never filled as numeric zero; absent run counts under observed HP state imply not observed, not not used',previous_assignments_changed=False)
    (out/'analysis_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    with (out/'process/analysis_log.jsonl').open('a') as h:h.write(json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),event='analysis_complete',n_cases=97,candidate_rows=len(cs),models=len(models),case_subsamples=len(trials)))+'\n')
    print(gs.to_string(index=False));print(models.to_string(index=False));print(cs[['candidate','n_cases']].to_string(index=False))

if __name__=='__main__':main()
