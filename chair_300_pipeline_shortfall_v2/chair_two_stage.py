"""Two-stage Hybrid v3.1 activity analysis, using the FIXED ablation sample.

Stage 1: evaluate existing B_state_events clusters + case-group stability.
Stage 2: within selected Stage 1 clusters, optional numeric-only DTW.

No original DATs are modified, no new activity boundaries are inferred.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import warnings
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, silhouette_score


def cli():
    p = argparse.ArgumentParser(description='Fixed activity Stage1 State+Events -> Stage2 Numeric DTW')
    p.add_argument('--ablation-root', type=Path, required=True, help='chair_ablation_100 output')
    p.add_argument('--data-root', type=Path, required=True, help='chair_model_A_results output')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--numeric-codes', default='20742,20743,20108')
    p.add_argument('--target-types', default='HP1,HP2', help='Comma-separated Stage1 dominant types to refine')
    p.add_argument('--min-numeric-coverage', type=float, default=.10)
    p.add_argument('--min-observations', type=int, default=5, help='Actual observed 1-second bins on a core numeric code')
    p.add_argument('--min-group-n', type=int, default=8)
    p.add_argument('--min-group-cases', type=int, default=3)
    p.add_argument('--min-code-fraction', type=float, default=.60, help='Share eligible activity members with this code')
    p.add_argument('--max-bins', type=int, default=64)
    p.add_argument('--band', type=float, default=.25)
    p.add_argument('--max-k', type=int, default=5)
    p.add_argument('--min-subcluster-size', type=int, default=3)
    p.add_argument('--min-subcluster-cases', type=int, default=2)
    p.add_argument('--min-silhouette', type=float, default=.15)
    p.add_argument('--subsample-fraction', type=float, default=.8)
    p.add_argument('--subsample-repeats', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-medoid-plots', type=int, default=8)
    p.add_argument('--max-raw-points', type=int, default=2000)
    a = p.parse_args()
    if not a.ablation_root.is_dir() or not a.data_root.is_dir():
        p.error('ablation-root or data-root does not exist')
    if not (0 < a.band <= 1 and 0 < a.min_code_fraction <= 1 and
            0 <= a.min_numeric_coverage <= 1 and 0 < a.subsample_fraction < 1):
        p.error('Invalid fraction / band argument')
    if min(a.min_observations,a.min_group_n,a.min_group_cases,a.max_bins,
           a.max_k,a.min_subcluster_size,a.min_subcluster_cases) < 1:
        p.error('Integer limits must be positive')
    a.numeric_codes = [s.strip().removeprefix('E') for s in a.numeric_codes.split(',') if s.strip()]
    a.target_types = {s.strip().lower() for s in a.target_types.split(',') if s.strip()}
    if not a.numeric_codes or not a.target_types: p.error('Empty numeric codes or target types')
    a.out.mkdir(parents=True,exist_ok=True)
    return a


def case_path(root, case_id):
    parts = str(case_id).split('__')
    if len(parts) != 3: raise ValueError(f'Unexpected case_id {case_id!r}')
    return root/'cases'/parts[0]/parts[1]/parts[2]


def read_stage1(root):
    assignment = root/'ablation_cluster_assignments.csv'
    fixed = root/'fixed_activity_manifest.csv'
    dist = root/'distances'/'B_state_events.csv'
    if not (assignment.exists() and fixed.exists() and dist.exists()):
        raise FileNotFoundError('Expected ablation_cluster_assignments.csv, fixed_activity_manifest.csv, distances/B_state_events.csv')
    b = pd.read_csv(assignment,dtype={'case_id':str})
    b = b[b.variant=='B_state_events'].copy()
    manifest = pd.read_csv(fixed,dtype={'case_id':str})
    if not len(b): raise ValueError('B_state_events assignments missing')
    keys = ['case_id','activity_id','start_s','end_s']
    if b.duplicated(keys).any() or manifest.duplicated(keys).any():
        raise ValueError('Duplicate activity identities')
    if 'ablation_row' not in b: raise ValueError('B assignments lack ablation_row')
    b = b.sort_values('ablation_row').reset_index(drop=True)
    if not np.array_equal(b.ablation_row.to_numpy(int), np.arange(len(b))):
        raise ValueError('ablation_row indices are not consecutive 0..N-1; refuse ambiguous distance indexing')
    # Verify that the existing rows are the exact same fixed Hybrid v3.1 activities
    check = b.merge(manifest[keys+['ablation_row']], on=keys+['ablation_row'], how='outer', indicator=True)
    if not (check._merge=='both').all() or len(check)!=len(b):
        raise ValueError('Fixed activity manifest disagrees with B assignments')
    D = np.loadtxt(dist,delimiter=',')
    if D.ndim==0: D=np.array([[float(D)]])
    if D.shape!=(len(b),len(b)) or not np.isfinite(D).all():
        raise ValueError('Stage 1 DTW distance shape/values invalid')
    if not np.allclose(D,D.T,atol=1e-6) or not np.allclose(np.diag(D),0,atol=1e-6):
        raise ValueError('Stage 1 distance is not symmetric or diagonal nonzero')
    D = np.maximum((D+D.T)/2,0); np.fill_diagonal(D,0)
    b['cluster'] = b['cluster'].astype(int)
    return b,D


def partition(D,k):
    if len(D)<3 or k<2 or k>=len(D) or np.max(D)<1e-12:
        return None,None
    Z=linkage(squareform(D,checks=False),method='average',optimal_ordering=False)
    labels=fcluster(Z,k,criterion='maxclust')
    return Z,labels if len(set(labels))>=2 else None


def case_stability(D,case_ids,reference,k,a):
    """Case-group subsampling, NOT a fully independent generalization test."""
    uniq=np.array(sorted(set(case_ids)))
    if len(uniq)<5 or k<2:
        return pd.DataFrame(columns=['iteration','n_cases','n_activities','ari_to_reference','k_actual'])
    rng=np.random.default_rng(a.seed)
    rows=[]
    for rep in range(a.subsample_repeats):
        chosen=rng.choice(uniq,size=max(3,int(round(a.subsample_fraction*len(uniq)))),replace=False)
        ix=np.flatnonzero(np.isin(case_ids,chosen))
        if len(ix)<=k+1:continue
        _,lab=partition(D[np.ix_(ix,ix)],k)
        if lab is None:continue
        rows.append({'iteration':rep,'n_cases':len(chosen),'n_activities':len(ix),
                     'ari_to_reference':float(adjusted_rand_score(reference[ix],lab)),
                     'k_actual':len(set(lab))})
    return pd.DataFrame(rows,columns=['iteration','n_cases','n_activities','ari_to_reference','k_actual'])


def read_case_tables(folder):
    names=['raw_1s','normalized_1s','observation_mask_1s']
    tables=[]
    for name in names:
        f=folder/(name+'.csv.gz')
        if not f.exists():raise FileNotFoundError(f)
        d=pd.read_csv(f,compression='gzip',low_memory=False)
        if 'time_s' not in d:raise ValueError(f'Missing time_s in {f}')
        d=d.set_index('time_s')
        tables.append(d)
    return tables


class CaseCache:
    def __init__(self,root,max_cases=1):
        self.root=root;self.max_cases=max_cases;self._cache=OrderedDict()
    def get(self,cid):
        if cid in self._cache:
            self._cache.move_to_end(cid);return self._cache[cid]
        ans=read_case_tables(case_path(self.root,cid))
        self._cache[cid]=ans
        if len(self._cache)>self.max_cases:self._cache.popitem(last=False)
        return ans


def pull_numeric(row,tables,codes,max_bins):
    raw,norm,mask=tables
    t=raw.index.to_numpy(float)
    included=(t>=float(row.start_s))&(t<float(row.end_s))
    n_total=int(included.sum())
    record={'n_time_bins':n_total,'by_code':{}}
    for code in codes:
        col='E'+code
        if col not in raw or col not in norm or col not in mask or n_total==0:
            record['by_code'][code]={'raw':np.array([]),'norm':np.array([]), 't':np.array([]),'coverage':0.,'n':0}
            continue
        x=raw.loc[included,col].to_numpy(float)
        z=norm.loc[included,col].to_numpy(float)
        mm=mask.loc[included,col].to_numpy(float)>0
        good=np.isfinite(x)&np.isfinite(z)&mm
        xx=x[good]; zz=z[good]; tt=t[included][good]-float(row.start_s)
        # samples are kept in order, no value interpolation or invented OFF states
        record['by_code'][code]={'raw':xx,'norm':np.clip(zz,-4,4),'t':tt,
                                 'coverage':float(good.sum()/n_total),'n':int(good.sum())}
    return record


def compress_observed(x,max_bins):
    """Reduce observed-only samples by chronological, contiguous medians; no interpolation."""
    x=np.asarray(x,float)
    if len(x)<=max_bins:return x
    return np.array([np.median(v) for v in np.array_split(x,max_bins)])


def dtw_1d(x,y,band):
    x=np.asarray(x,float);y=np.asarray(y,float)
    n,m=len(x),len(y)
    if not n or not m:return np.nan
    radius=max(2,int(math.ceil(max(n,m)*band)))
    D=np.full((n+1,m+1),np.inf);L=np.zeros((n+1,m+1),dtype=np.int32)
    D[0,0]=0.
    for i in range(1,n+1):
        center=(i-.5)*m/n+.5
        lo=max(1,int(math.floor(center-radius)))
        hi=min(m,int(math.ceil(center+radius)))
        for j in range(lo,hi+1):
            diag,up,left=D[i-1,j-1],D[i-1,j],D[i,j-1]
            if diag<=up and diag<=left:
                prev=diag;l=L[i-1,j-1]
            elif up<=left:
                prev=up;l=L[i-1,j]
            else:
                prev=left;l=L[i,j-1]
            if np.isfinite(prev):
                D[i,j]=prev+min(abs(x[i-1]-y[j-1])/4.,1.)
                L[i,j]=l+1
    return float(D[n,m]/L[n,m]) if L[n,m] else np.nan


def calculate_numeric_matrix(records,codes,max_bins,band):
    n=len(records)
    # robust cohort scale for physical medians and 95th percentiles
    scales={}
    for code in codes:
        features=[]
        for rec in records:
            x=rec['by_code'][code]['raw']
            features.append([np.median(x),np.quantile(x,.95)] if len(x) else [np.nan,np.nan])
        features=np.asarray(features,float)
        dist=[]
        for j in range(2):
            vals=features[:,j];vals=vals[np.isfinite(vals)]
            if len(vals)>1:
                q1,q3=np.quantile(vals,[.25,.75]);q05,q95=np.quantile(vals,[.05,.95])
                dist.append(max(q3-q1,.15*(q95-q05),1e-6))
            else:dist.append(np.nan)
        scales[code]=dist
    D=np.zeros((n,n),float);components=[]
    for i in range(n):
        for j in range(i+1,n):
            dshape=[];damp=[];weights=[];shared=[]
            for code in codes:
                ai=records[i]['by_code'][code];bj=records[j]['by_code'][code]
                if not (ai['n'] and bj['n']):continue
                a=compress_observed(ai['norm'],max_bins)
                b=compress_observed(bj['norm'],max_bins)
                ds=dtw_1d(a,b,band)
                if not np.isfinite(ds):continue
                f1=[np.median(ai['raw']),np.quantile(ai['raw'],.95)]
                f2=[np.median(bj['raw']),np.quantile(bj['raw'],.95)]
                dv=[min(abs(v-w)/sc,4)/4 for v,w,sc in zip(f1,f2,scales[code]) if np.isfinite(sc)]
                da=float(np.median(dv)) if dv else 0.
                # core first channel gets priority; supplementary channels may refine
                weight=1. if code==codes[0] else .35
                dshape.append(ds);damp.append(da);weights.append(weight);shared.append(code)
            if codes[0] not in shared:
                raise ValueError('No mutually observed core code: numeric eligibility gate violated')
            shape=float(np.average(dshape,weights=weights))
            amp=float(np.average(damp,weights=weights))
            # shape and absolute levels both matter; duration is a small, explicit component
            dur=min(abs(math.log(max(records[i]['duration'],1)/max(records[j]['duration'],1)))/4.,1.)
            value=.75*shape+.20*amp+.05*dur
            D[i,j]=D[j,i]=value
            components.append({'i':i,'j':j,'shape_dtw':shape,'raw_level_distance':amp,
                               'duration_distance':dur,'total_distance':value,
                               'shared_codes':','.join(shared),
                               'coverage_difference_core':abs(records[i]['by_code'][codes[0]]['coverage']-
                                                              records[j]['by_code'][codes[0]]['coverage'])})
    return D,pd.DataFrame(components)


def score_partitions(D,df,max_k,a):
    Z=linkage(squareform(D,checks=False),method='average',optimal_ordering=len(D)<=60)
    stats=[];labels_by_k={}
    for k in range(2,min(max_k,len(D)-1)+1):
        lab=fcluster(Z,k,criterion='maxclust')
        groups=[np.flatnonzero(lab==c) for c in np.unique(lab)]
        if len(groups)<2 or len(groups)>=len(D):continue
        sizes=[len(g) for g in groups]
        min_case=min(df.iloc[g].case_id.nunique() for g in groups)
        sil=float(silhouette_score(D,lab,metric='precomputed'))
        valid=(min(sizes)>=a.min_subcluster_size and min_case>=a.min_subcluster_cases
               and max(sizes)/len(df)<=.85 and sil>=a.min_silhouette)
        stats.append({'k_requested':k,'k_actual':len(groups),'silhouette':sil,
                      'min_cluster_size':min(sizes),'min_cluster_cases':min_case,
                      'largest_cluster_fraction':max(sizes)/len(df),'valid':bool(valid)})
        labels_by_k[k]=lab
    t=pd.DataFrame(stats,columns=['k_requested','k_actual','silhouette','min_cluster_size',
                                  'min_cluster_cases','largest_cluster_fraction','valid'])
    okay=t[t.valid] if len(t) else t
    if len(okay):
        winner=okay.sort_values(['silhouette','k_requested'],ascending=[False,True]).iloc[0]
        return Z,t,int(winner.k_requested),labels_by_k[int(winner.k_requested)]
    return Z,t,None,None


def exact_numeric_points(folder,start,end,codes,max_points):
    """Bounded streaming of original event log, for medoid figures only."""
    file=folder/'events_original.csv.gz'
    if not file.exists():return {},False
    want=set(codes);values={c:[] for c in want};strides={c:1 for c in want};seen={c:0 for c in want}
    with gzip.open(file,'rt',encoding='utf-8-sig',errors='replace',newline='') as f:
        reader=csv.DictReader(f)
        for row in reader:
            code=(row.get('event_code') or '').strip()
            if code not in want:continue
            try:
                t=float(row['time_s']);v=float(row['event_value'])
            except (ValueError,TypeError,KeyError):continue
            if not(start<=t<end and np.isfinite(v)):continue
            seen[code]+=1
            if (seen[code]-1)%strides[code]:continue
            values[code].append((t-start,v))
            if len(values[code])>max_points:
                values[code]=values[code][::2];strides[code]*=2
    return values,True


def medoid_plot(folder,row,core,codes,out,max_points):
    raw,_,mask=read_case_tables(folder)
    t=raw.index.to_numpy(float)
    keep=(t>=float(row.start_s))&(t<float(row.end_s))
    time=t[keep]-float(row.start_s)
    exact,has_exact=exact_numeric_points(folder,float(row.start_s),float(row.end_s),codes,max_points)
    fig,axes=plt.subplots(len(codes)+1,1,figsize=(11,2.0*(len(codes)+1)),sharex=True,layout='constrained')
    state_codes=['20039','20011','20101']
    for j,code in enumerate(state_codes):
        col='E'+code
        if col in raw:
            y=raw.loc[keep,col].to_numpy(float)
            axes[0].step(time,y+j*3.,where='post',label=code,linewidth=1.)
    axes[0].set_ylabel('State\n(offset)')
    axes[0].legend(fontsize=8,ncol=3)
    for i,code in enumerate(codes):
        ax=axes[i+1];col='E'+code
        if has_exact and len(exact.get(code,[])):
            z=np.asarray(exact[code],float)
            ax.scatter(z[:,0],z[:,1],s=5,alpha=.7)
            ax.text(.99,.94,f'{len(z)} raw events',transform=ax.transAxes,ha='right',va='top',fontsize=8)
        elif col in raw:
            x=raw.loc[keep,col].to_numpy(float)
            m=mask.loc[keep,col].to_numpy(float)>0
            valid=np.isfinite(x)&m
            ax.scatter(time[valid],x[valid],s=7,alpha=.7)
            ax.text(.99,.94,f'{valid.sum()} observed bins',transform=ax.transAxes,ha='right',va='top',fontsize=8)
        ax.set_ylabel(code+'\n(raw)')
    axes[-1].set_xlabel('Seconds from Activity start')
    axes[0].set_title(f'Stage 2 numeric Medoid | core {core} | {row.case_id} | Activity {row.activity_id}')
    for ax in axes:ax.grid(axis='x',alpha=.2)
    fig.savefig(out,dpi=150);plt.close(fig)


def analyze(a):
    b,D=read_stage1(a.ablation_root)
    n=len(b);n_cases=b.case_id.nunique()
    print(f'Stage 1 fixed B clusters: {n} Activity, {n_cases} cases; no resampling of activities',flush=True)
    b.to_csv(a.out/'stage1_fixed_assignments.csv',index=False)
    np.savetxt(a.out/'stage1_B_distance_copy.csv',D,delimiter=',',fmt='%.8f')
    grp=[]
    for cl,g in b.groupby('cluster',sort=True):
        dominant=str(g.activity_type.fillna('unknown').mode().iloc[0]) if 'activity_type' in g else 'unknown'
        grp.append({'stage1_cluster':int(cl),'n_activities':len(g),'n_cases':g.case_id.nunique(),
                    'dominant_activity_type':dominant,
                    'median_duration_s':float(g.duration_s.median()),
                    'median_numeric_coverage':float(g.numeric_coverage_recomputed.median())
                    if 'numeric_coverage_recomputed' in g else np.nan})
    grp=pd.DataFrame(grp)
    grp.to_csv(a.out/'stage1_cluster_summary.csv',index=False)
    ks=pd.to_numeric(b.get('selected_k',pd.Series([b.cluster.nunique()])),errors='coerce').dropna()
    k=int(ks.iloc[0]) if len(ks) else b.cluster.nunique()
    base_stability=case_stability(D,b.case_id.to_numpy(str),b.cluster.to_numpy(int),k,a)
    base_stability.to_csv(a.out/'stage1_case_subsampling.csv',index=False)
    if len(base_stability):
        st_median=float(base_stability.ari_to_reference.median())
        st_q1=float(base_stability.ari_to_reference.quantile(.25))
        st_q3=float(base_stability.ari_to_reference.quantile(.75))
    else: st_median=st_q1=st_q3=np.nan
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axs[0].bar(grp.stage1_cluster.astype(str),grp.n_activities)
    axs[0].set_xlabel('Stage 1 B cluster');axs[0].set_ylabel('Activity count')
    if len(base_stability):axs[1].hist(base_stability.ari_to_reference,bins=np.linspace(-.2,1,17),edgecolor='white')
    axs[1].set_xlabel('ARI vs fixed Stage 1 B assignments');axs[1].set_ylabel('Case subsamples')
    fig.suptitle('Stage 1: cluster sizes and case-group subsampling stability')
    fig.savefig(a.out/'stage1_stability.png',dpi=160);plt.close(fig)

    groups_to_run=grp[grp.dominant_activity_type.str.lower().isin(a.target_types)]
    if len(groups_to_run)==0:
        warnings.warn('No Stage 1 cluster matches requested target types; see stage1_cluster_summary.csv')
    assignments=[];eligible_rows=[];sub_summaries=[];kdiag=[];pairrows=[];medoidrows=[];sub_stab=[];biasrows=[];cache=CaseCache(a.data_root)
    plots=0
    for rg in groups_to_run.itertuples(index=False):
        cl=int(rg.stage1_cluster)
        members=b[b.cluster==cl].copy().reset_index(drop=True)
        activity_records=[]
        for row in members.itertuples(index=False):
            label=str(row.case_id)
            try:
                obj=pull_numeric(row,cache.get(label),a.numeric_codes,a.max_bins)
            except (OSError,ValueError,KeyError,MemoryError) as e:
                obj={'n_time_bins':0,'by_code':{code:{'n':0,'coverage':0.,'norm':np.array([]),
                        'raw':np.array([]),'t':np.array([])} for code in a.numeric_codes}}
                print(f'WARNING: {label} read failed: {e}',flush=True)
            obj['duration']=float(row.duration_s)
            activity_records.append(obj)
        # Gate on any numeric observation fraction plus sample size
        initially_eligible=[]
        for obj in activity_records:
            coverage=float(np.mean([obj['by_code'][c]['coverage'] for c in a.numeric_codes]))
            best=max(obj['by_code'][c]['n'] for c in a.numeric_codes)
            initially_eligible.append((coverage>=a.min_numeric_coverage and best>=a.min_observations))
        pool=np.flatnonzero(initially_eligible)
        # A frequently observed but CONSTANT numeric channel (e.g., torque=0) is
        # not a useful primary DTW signal. Require actual raw variability.
        counts={code:sum(activity_records[i]['by_code'][code]['n']>=a.min_observations for i in pool)
                for code in a.numeric_codes}
        variable_codes=[]
        for code in a.numeric_codes:
            observed=[activity_records[i]['by_code'][code]['raw'] for i in pool
                      if activity_records[i]['by_code'][code]['n']>=a.min_observations]
            if observed:
                all_values=np.concatenate(observed)
                robust_span=np.quantile(all_values,.95)-np.quantile(all_values,.05)
                if len(all_values)>=3 and np.isfinite(robust_span) and robust_span>1e-8:
                    variable_codes.append(code)
        primary=max(variable_codes,key=lambda c:counts[c]) if variable_codes else a.numeric_codes[0]
        qualified=np.flatnonzero([initially_eligible[i] and
                   activity_records[i]['by_code'][primary]['n']>=a.min_observations
                   for i in range(len(members))])
        usable=[]
        if len(qualified):
            for code in a.numeric_codes:
                frac=sum(activity_records[i]['by_code'][code]['n']>=a.min_observations
                         for i in qualified)/len(qualified)
                if code in variable_codes and (code==primary or frac>=a.min_code_fraction):
                    usable.append(code)
        if primary in usable:usable=[primary]+[x for x in usable if x!=primary]
        for i,row in members.iterrows():
            obj=activity_records[i]
            eligible_rows.append({'case_id':row.case_id,'activity_id':row.activity_id,
                 'stage1_cluster':cl,'activity_type':row.activity_type,
                 'mean_numeric_coverage':np.mean([obj['by_code'][c]['coverage'] for c in a.numeric_codes]),
                 'primary_code':primary,'primary_observation_bins':obj['by_code'][primary]['n'],
                 'stage2_eligible':bool(i in qualified),'per_code_observed_bins':json.dumps({c:obj['by_code'][c]['n'] for c in a.numeric_codes})})
            assignments.append({'case_id':row.case_id,'activity_id':row.activity_id,
                 'stage1_cluster':cl,'stage2_cluster':np.nan,'stage2_eligible':bool(i in qualified),
                 'duration_s':row.duration_s,'start_s':row.start_s,'end_s':row.end_s})
        outcl=a.out/'stage2'/f'B_cluster_{cl:02d}'
        outcl.mkdir(parents=True,exist_ok=True)
        reason=''
        if not variable_codes:reason='no_varying_numeric_signal_in_eligible_activities'
        elif len(qualified)<a.min_group_n:reason=f'eligible_numeric_activities_below_{a.min_group_n}'
        elif members.iloc[qualified].case_id.nunique()<a.min_group_cases:reason='too_few_distinct_cases'
        elif not usable:reason='no_common_numeric_channels'
        if reason:
            sub_summaries.append({'stage1_cluster':cl,'activity_type':rg.dominant_activity_type,
                'n_activity_total':len(members),'n_numeric_eligible':len(qualified),
                'n_eligible_cases':members.iloc[qualified].case_id.nunique(),
                'primary_numeric_code':primary,'used_numeric_codes':','.join(usable),
                'selected_k':np.nan,'silhouette':np.nan,'status':'NOT_CLUSTERED_'+reason,
                'median_case_subsample_ari':np.nan})
            print(f'Stage 2 B{cl}: {reason}, {len(qualified)}/{len(members)} eligible',flush=True)
            continue
        df=members.iloc[qualified].copy().reset_index(drop=True)
        records=[activity_records[i] for i in qualified]
        mat,pairs=calculate_numeric_matrix(records,usable,a.max_bins,a.band)
        np.savetxt(outcl/'numeric_distances.csv',mat,delimiter=',',fmt='%.8f')
        # Diagnostic only: pairwise distances share Activities, so there is no independent-pair p-value.
        if len(pairs)>3 and pairs.coverage_difference_core.std()>1e-10 and pairs.total_distance.std()>1e-10:
            rho=float(spearmanr(pairs.total_distance,pairs.coverage_difference_core).statistic)
        else: rho=np.nan
        biasrows.append({'stage1_cluster':cl,'core_numeric_code':primary,
                         'distance_vs_coverage_difference_spearman':rho,
                         'n_pairs':len(pairs),'note':'descriptive; paired distances are not independent'})
        pairs.insert(0,'stage1_cluster',cl)
        pairrows.append(pairs)
        if np.max(mat)<1e-10:
            reason='identical_numeric_signatures'
            Z=None;diagnostics=pd.DataFrame();chosen=None;lab=None
        else:
            Z,diagnostics,chosen,lab=score_partitions(mat,df,a.max_k,a)
            reason='validated_exploratory' if lab is not None else 'NO_RELIABLE_SUBCLUSTERS'
        if not diagnostics.empty:
            diagnostics.insert(0,'stage1_cluster',cl);kdiag.append(diagnostics)
            diagnostics.to_csv(outcl/'k_diagnostics.csv',index=False)
        if Z is not None:
            fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
            dendrogram(Z,no_labels=len(df)>30,ax=ax)
            ax.set_title(f'Numeric within B{cl} ({rg.dominant_activity_type}) | n={len(df)}')
            ax.set_ylabel('Combined numeric DTW distance')
            fig.savefig(outcl/'numeric_dendrogram.png',dpi=150);plt.close(fig)
        val=np.nan;stable=pd.DataFrame()
        if lab is not None:
            stable=case_stability(mat,df.case_id.to_numpy(str),lab,chosen,a)
            stable.insert(0,'stage1_cluster',cl)
            sub_stab.append(stable)
            val=float(stable.ari_to_reference.median()) if len(stable) else np.nan
            for index,sub in enumerate(lab):
                ix=int(qualified[index]);r=members.iloc[ix]
                for ass in assignments:
                    if ass['stage1_cluster']==cl and ass['case_id']==r.case_id and ass['activity_id']==r.activity_id:
                        ass['stage2_cluster']=int(sub);break
            for subcluster in np.unique(lab):
                ix=np.flatnonzero(lab==subcluster)
                middle=int(ix[np.argmin(mat[np.ix_(ix,ix)].sum(axis=1))])
                r=df.iloc[middle]
                medoidrows.append({'stage1_cluster':cl,'stage2_cluster':int(subcluster),
                    'n_activities':len(ix),'n_cases':df.iloc[ix].case_id.nunique(),
                    'medoid_case_id':r.case_id,'medoid_activity_id':r.activity_id,
                    'used_numeric_codes':','.join(usable)})
                if plots<a.max_medoid_plots:
                    try:
                        medoid_plot(case_path(a.data_root,r.case_id),r,primary,a.numeric_codes,
                                    outcl/f'numeric_subcluster_{subcluster:02d}_medoid_raw.png',a.max_raw_points)
                        plots+=1
                    except (OSError,ValueError,KeyError,MemoryError) as e:
                        warnings.warn(f'Could not draw B{cl} numeric medoid: {e}')
            fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
            counts=pd.Series(lab).value_counts().sort_index()
            ax.bar(counts.index.astype(str),counts.values)
            ax.set_xlabel('Numeric subcluster');ax.set_ylabel('Activity count')
            ax.set_title(f'B{cl} | numeric-only subcluster membership')
            fig.savefig(outcl/'numeric_cluster_sizes.png',dpi=150);plt.close(fig)
            # IMPORTANT: physical raw values are shown on their original scale, not per-activity normalized.
            fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
            sorted_cl=np.unique(lab)
            medians=[[float(np.median(rec['by_code'][primary]['raw']))
                      for rec,l in zip(records,lab) if l==group and len(rec['by_code'][primary]['raw'])]
                     for group in sorted_cl]
            p95s=[[float(np.quantile(rec['by_code'][primary]['raw'],.95))
                      for rec,l in zip(records,lab) if l==group and len(rec['by_code'][primary]['raw'])]
                     for group in sorted_cl]
            coverages=[[rec['by_code'][primary]['coverage'] for rec,l in zip(records,lab) if l==group]
                       for group in sorted_cl]
            axes[0].boxplot(medians,showfliers=True)
            axes[0].set_ylabel(f'Raw median | {primary} (unit unverified)')
            axes[1].boxplot(p95s,showfliers=True)
            axes[1].set_ylabel(f'Raw 95th percentile | {primary}')
            axes[2].boxplot(coverages,showfliers=True)
            axes[2].set_ylabel(f'{primary} observed-bin fraction')
            for ax in axes:
                ax.set_xticks(np.arange(1,len(sorted_cl)+1),[str(c) for c in sorted_cl])
                ax.set_xlabel('Numeric subcluster')
                ax.grid(axis='y',alpha=.2)
            fig.suptitle(f'B{cl} | original numeric level vs observation coverage')
            fig.savefig(outcl/'raw_level_and_coverage_boxplots.png',dpi=160);plt.close(fig)
        sub_summaries.append({'stage1_cluster':cl,'activity_type':rg.dominant_activity_type,
                'n_activity_total':len(members),'n_numeric_eligible':len(qualified),
                'n_eligible_cases':df.case_id.nunique(),'primary_numeric_code':primary,
                'used_numeric_codes':','.join(usable),'selected_k':chosen,
                'silhouette':float(diagnostics.loc[diagnostics.k_requested==chosen,'silhouette'].iloc[0])
                    if chosen is not None else np.nan,
                'status':reason,'median_case_subsample_ari':val})
        print(f'Stage 2 B{cl}: {reason} n={len(df)} selected_k={chosen}',flush=True)
    pd.DataFrame(assignments).to_csv(a.out/'two_stage_assignments.csv',index=False)
    pd.DataFrame(eligible_rows).to_csv(a.out/'numeric_eligibility_details.csv',index=False)
    if sub_summaries:
        summary_plot=pd.DataFrame(sub_summaries).sort_values('stage1_cluster')
        px=np.arange(len(summary_plot))
        fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        axs[0].bar(px,summary_plot.n_activity_total,color='#9da9b4',label='All in Stage1 cluster')
        axs[0].bar(px,summary_plot.n_numeric_eligible,color='#527fb0',label='Numeric eligible')
        axs[0].set_xticks(px,[f'B{v} {t}' for v,t in zip(summary_plot.stage1_cluster,summary_plot.activity_type)])
        axs[0].set_ylabel('Activity count');axs[0].legend(fontsize=8)
        axs[1].bar(px,summary_plot.median_case_subsample_ari.fillna(0),color='#527fb0')
        axs[1].set_ylim(-.1,1)
        axs[1].set_xticks(px,[f'B{v}' for v in summary_plot.stage1_cluster])
        axs[1].set_ylabel('Median case-group subsampling ARI')
        for j,val in enumerate(summary_plot.median_case_subsample_ari):
            if not np.isfinite(val):axs[1].text(j,.03,'N/A',ha='center',va='bottom',fontsize=9)
        for ax in axs:ax.grid(axis='y',alpha=.2)
        fig.suptitle('Stage 2 eligibility and internal stability (N/A = not classified)')
        fig.savefig(a.out/'stage2_eligibility_and_stability.png',dpi=160);plt.close(fig)
    pd.DataFrame(sub_summaries,columns=['stage1_cluster','activity_type','n_activity_total',
        'n_numeric_eligible','n_eligible_cases','primary_numeric_code','used_numeric_codes',
        'selected_k','silhouette','status','median_case_subsample_ari']).to_csv(a.out/'stage2_summary.csv',index=False)
    if kdiag:pd.concat(kdiag,ignore_index=True).to_csv(a.out/'stage2_k_diagnostics.csv',index=False)
    else:pd.DataFrame(columns=['stage1_cluster','k_requested','silhouette','valid']).to_csv(a.out/'stage2_k_diagnostics.csv',index=False)
    if pairrows:pd.concat(pairrows,ignore_index=True).to_csv(a.out/'stage2_pairwise_distances.csv',index=False)
    else:pd.DataFrame(columns=['stage1_cluster','i','j','shape_dtw','raw_level_distance','duration_distance','total_distance']).to_csv(a.out/'stage2_pairwise_distances.csv',index=False)
    pd.DataFrame(biasrows,columns=['stage1_cluster','core_numeric_code',
        'distance_vs_coverage_difference_spearman','n_pairs','note']).to_csv(
        a.out/'stage2_bias_diagnostics.csv',index=False)
    pd.DataFrame(medoidrows,columns=['stage1_cluster','stage2_cluster','n_activities','n_cases','medoid_case_id',
                                    'medoid_activity_id','used_numeric_codes']).to_csv(a.out/'stage2_medoids.csv',index=False)
    if sub_stab:pd.concat(sub_stab,ignore_index=True).to_csv(a.out/'stage2_case_subsampling.csv',index=False)
    else:pd.DataFrame(columns=['stage1_cluster','iteration','n_cases','n_activities','ari_to_reference']).to_csv(a.out/'stage2_case_subsampling.csv',index=False)
    summary={'fixed_activities':n,'fixed_cases':n_cases,'stage1_k':k,
       'stage1_median_subsample_ari':st_median,'stage1_ari_q1':st_q1,'stage1_ari_q3':st_q3,
       'stage2':sub_summaries,
       'limitations':[
           'Stage 1 uses EXISTING B_state_events labels; no activity intervals are changed.',
           'Case subsampling ARI assesses partition consistency within this sample, not generalization to unseen patients.',
           'Numeric Stage 2 is tested only on Activities with sufficient measurements; missingness can cause selection bias.',
           'Observed-only numeric DTW uses measured samples in chronological order; unobserved temporal gaps are not imputed.',
           'Numeric subclusters must satisfy minimal size, distinct-case, balance and silhouette thresholds or remain unclassified.',
           'Input feature codes are a narrow selected set, not every event_code; clinical labels are not inferred.'
       ]}
    (a.out/'analysis_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    print('COMPLETE:',a.out,flush=True)
    return summary


def main():
    analyze(cli())

if __name__=='__main__':main()
