"""Ablation of DTW input groups on FIXED Hybrid v3.1 activity windows.

A: original mixed groups
B: state + event count, no numeric
C: numeric only, coverage eligibility gate (reported explicitly)
D: full groups with one-sided numeric missingness neutralized (sensitivity)

Requires original v3.1.1 modules in the same folder. The program never
resegments data, changes the chosen activities, or edits original files.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import chair_dtw_v3_1 as base

try:
    from numba import njit
    HAS_NUMBA = True
except (ImportError, Exception):
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        return lambda f: f

VARIANTS = ['A_full', 'B_state_events', 'C_numeric_only', 'D_missing_neutral', 'E_mask_only_control']


def arguments():
    p = argparse.ArgumentParser(description='Fixed-activity DTW ablation / missingness audit')
    p.add_argument('--data-root', type=Path, required=True, help='chair_model_A_results with cases/')
    p.add_argument('--baseline-root', type=Path, required=True, help='chair_dtw_v3_1_1_100 with cluster_assignments.csv')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-activities', type=int, default=0, help='0=all original sampled activities')
    p.add_argument('--max-bins', type=int, default=64, help='MUST equal baseline original value for matched comparison')
    p.add_argument('--band', type=float, default=.25)
    p.add_argument('--max-k', type=int, default=8)
    p.add_argument('--numeric-min-coverage', type=float, default=.10,
                   help='Numeric-only variant eligibility mean numeric observation fraction')
    p.add_argument('--numeric-min-bins', type=int, default=3, help='Observed rows required for C')
    p.add_argument('--state-codes', default='20039,20011,20101')
    p.add_argument('--numeric-codes', default='20742,20743,20108')
    p.add_argument('--max-medoid-plots', type=int, default=8, help='max cluster medoid PNGs per variant; 0=no plots')
    a = p.parse_args()
    if a.max_activities < 0 or a.max_bins < 16 or not 0 < a.band <= 1 or a.max_k < 2:
        p.error('Invalid max-activities, max-bins, band, or max-k')
    if not (0 <= a.numeric_min_coverage <= 1) or a.numeric_min_bins < 1:
        p.error('Numeric coverage must be in [0,1]; min-bins >= 1')
    a.state_codes = [c.strip().removeprefix('E') for c in a.state_codes.split(',') if c.strip()]
    a.numeric_codes = [c.strip().removeprefix('E') for c in a.numeric_codes.split(',') if c.strip()]
    if not a.state_codes or not a.numeric_codes:
        p.error('Both state and numeric codes must be specified')
    a.out.mkdir(parents=True, exist_ok=True)
    return a


def restore_fixed_activities(a):
    path = a.baseline_root / 'cluster_assignments.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing v3.1 cluster assignments: {path}')
    df = pd.read_csv(path, dtype={'case_id': str, 'seq': str})
    need = {'case_id','activity_id','start_s','end_s','duration_s'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f'cluster_assignments.csv lacks: {missing}')
    if 'dtw_index' in df:
        df = df.sort_values('dtw_index', kind='stable')
    df = df.reset_index(drop=True)
    df['baseline_row_index'] = np.arange(len(df))
    if a.max_activities:
        df = df.iloc[:a.max_activities].copy().reset_index(drop=True)
    if df.duplicated(subset=['case_id','activity_id','start_s','end_s']).any():
        raise ValueError('Duplicated activity identities in baseline assignments')
    if len(df) < 3:
        raise ValueError('Need >=3 activities')
    return df


def locate_case_dir(data_root, case_id):
    parts = case_id.split('__')
    if len(parts) != 3:
        raise ValueError('Unrecognized case_id: ' + case_id)
    p = data_root / 'cases' / parts[0] / parts[1] / parts[2]
    if not (p / 'raw_1s.csv.gz').exists():
        raise FileNotFoundError(str(p / 'raw_1s.csv.gz'))
    return p


def build_signatures(df, a):
    cache = {}
    features = [None]*len(df)
    failures = []
    ns = len(a.state_codes)
    nn = len(a.numeric_codes)
    for t, (cid, g) in enumerate(df.groupby('case_id', sort=False), 1):
        try:
            folder = locate_case_dir(a.data_root, cid)
            raw, norm, cnt, mask = base.load_files(folder)
            for ix, row in g.iterrows():
                sig = base.segment_to_sequence(
                    row,raw,norm,cnt,mask,a.state_codes,a.numeric_codes,a.max_bins)
                if sig is None:
                    failures.append({'baseline_row_index':int(row.baseline_row_index),'case_id':cid,
                                     'activity_id':row.activity_id,'reason':'no_usable_sequence'})
                else:
                    # For C, distinguish actually observed bins from zero-valued signal.
                    numeric = sig['array'][:,ns:ns+nn]
                    sig['numeric_bins_observed'] = int(np.isfinite(numeric).any(axis=1).sum())
                    sig['numeric_channels_observed'] = int(np.isfinite(numeric).any(axis=0).sum())
                    features[int(ix)] = sig
            del raw, norm, cnt, mask
        except (OSError,ValueError,KeyError,MemoryError) as e:
            for ix,row in g.iterrows():
                failures.append({'baseline_row_index':int(row.baseline_row_index),'case_id':cid,
                                 'activity_id':row.activity_id,'reason':str(e)})
        print(f'Load signatures {t}/{df.case_id.nunique()}: {cid}', flush=True)
    fail_df = pd.DataFrame(failures,columns=['baseline_row_index','case_id','activity_id','reason'])
    fail_df.to_csv(a.out/'read_errors.csv',index=False)
    okay = np.array([i for i,s in enumerate(features) if s is not None],int)
    if len(okay) < 3:raise RuntimeError('Fewer than 3 usable activities; see read_errors.csv')
    if len(okay) != len(df):
        print(f'WARNING: {len(df)-len(okay)} activities unreadable; all variants use the SAME retained rows.', flush=True)
    df = df.iloc[okay].copy().reset_index(drop=True)
    features = [features[i] for i in okay]
    df['ablation_row'] = np.arange(len(df))
    df['numeric_coverage_recomputed'] = [s['coverage'] for s in features]
    df['numeric_bins_observed'] = [s['numeric_bins_observed'] for s in features]
    df['numeric_channels_observed'] = [s['numeric_channels_observed'] for s in features]
    df['event_count_recomputed'] = [s['event_count'] for s in features]
    df.to_csv(a.out/'fixed_activity_manifest.csv',index=False)
    return df,features


@njit(cache=True)
def _dtw_dp(local,band):
    n,m=local.shape
    radius=max(2,int(math.ceil(max(n,m)*band)))
    D=np.full((n+1,m+1),np.inf)
    L=np.zeros((n+1,m+1),np.int64)
    D[0,0]=0.0
    for i in range(1,n+1):
        center=(i-.5)*m/n+.5
        lo=max(1,int(math.floor(center-radius)))
        hi=min(m,int(math.ceil(center+radius)))
        for j in range(lo,hi+1):
            diag=D[i-1,j-1]
            up=D[i-1,j]
            left=D[i,j-1]
            if diag<=up and diag<=left:
                val=diag; pi=i-1;pj=j-1
            elif up<=left:
                val=up;pi=i-1;pj=j
            else:
                val=left;pi=i;pj=j-1
            if math.isinf(val):continue
            D[i,j]=val+local[i-1,j-1]
            L[i,j]=L[pi,pj]+1
    if L[n,m]==0:return 1.0
    return D[n,m]/L[n,m]


def local_cost_numeric_neutral(A,B,ns,nn):
    """D: only jointly measured numeric values inform numeric distance.

    One-sided/mutual missing are NOT filled with zero, and do not count as a match.
    If there are no mutually observed numeric channels at aligned timepoints,
    rely on state/event groups, report absent coverage in pairwise diagnostics.
    """
    x=A[:,:ns];y=B[:,:ns]
    n,m=len(A),len(B)
    total=np.zeros((n,m),dtype=float)
    weights=np.zeros((n,m),dtype=float)
    if ns:
        kk=np.isfinite(x)[:,None,:]&np.isfinite(y)[None,:,:]
        one=np.isfinite(x)[:,None,:]^np.isfinite(y)[None,:,:]
        c=np.where(kk,(x[:,None,:]!=y[None,:,:]).astype(float),np.where(one,.25,0.))
        denom=(kk|one).sum(axis=2)
        yes=denom>0
        s=np.divide(c.sum(axis=2),denom,out=np.zeros((n,m)),where=yes)
        total+=base.GROUP_W['state']*s*yes
        weights+=base.GROUP_W['state']*yes
    xx=A[:,ns:ns+nn];yy=B[:,ns:ns+nn]
    kk=np.isfinite(xx)[:,None,:]&np.isfinite(yy)[None,:,:]
    denom=kk.sum(axis=2)
    if nn:
        delta=np.minimum(np.abs(np.nan_to_num(xx)[:,None,:]-np.nan_to_num(yy)[None,:,:])/4.,1.)
        s=np.divide(np.where(kk,delta,0.).sum(axis=2),denom,
                    out=np.zeros((n,m)),where=denom>0)
        yes=denom>0
        total+=base.GROUP_W['numeric']*s*yes
        weights+=base.GROUP_W['numeric']*yes
    e1=A[:,ns+nn:];e2=B[:,ns+nn:]
    if e1.shape[1]:
        event=np.mean(np.minimum(np.abs(np.nan_to_num(e1)[:,None,:]-np.nan_to_num(e2)[None,:,:]),1.),axis=2)
        total+=base.GROUP_W['events']*event
        weights+=base.GROUP_W['events']
    return np.divide(total,weights,out=np.ones((n,m)),where=weights>0)


def _shape(A,B,ns,nn,band,mode):
    if mode=='A_full':
        local=base.local_cost_matrix(A,B,ns,nn)
    elif mode=='B_state_events':
        ix=list(range(ns))+list(range(ns+nn,2*ns+nn))
        local=base.local_cost_matrix(A[:,ix],B[:,ix],ns,0)
    elif mode=='C_numeric_only':
        local=base.local_cost_matrix(A[:,ns:ns+nn],B[:,ns:ns+nn],0,nn)
    elif mode=='D_missing_neutral':
        local=local_cost_numeric_neutral(A,B,ns,nn)
    elif mode=='E_mask_only_control':
        ma=np.isfinite(A[:,ns:ns+nn]).astype(float)
        mb=np.isfinite(B[:,ns:ns+nn]).astype(float)
        local=np.mean(np.abs(ma[:,None,:]-mb[None,:,:]),axis=2)
    else:raise KeyError(mode)
    return _dtw_dp(local,band)


def build_pairwise(df,features,a):
    ns,nn=len(a.state_codes),len(a.numeric_codes)
    eligible=np.asarray([f['coverage']>=a.numeric_min_coverage
                         and f['numeric_bins_observed']>=a.numeric_min_bins
                         and f['numeric_channels_observed']>=1 for f in features])
    mask=df[['ablation_row','case_id','activity_id','numeric_coverage_recomputed',
             'numeric_bins_observed','numeric_channels_observed']].copy()
    mask['eligible_numeric_only']=eligible
    mask.to_csv(a.out/'numeric_eligibility.csv',index=False)
    n=len(df)
    centers,scale=base.physical_scale(features)
    matrices={name:np.zeros((n,n),float) for name in VARIANTS if name!='C_numeric_only'}
    nums=np.flatnonzero(eligible)
    matrices['C_numeric_only']=np.zeros((len(nums),len(nums)),float)
    num_lookup={int(v):i for i,v in enumerate(nums)}
    pair_rows=[]
    for i in range(n):
        if (i+1)%10==0:print(f'Pairwise {i+1}/{n}, numba={HAS_NUMBA}',flush=True)
        f=features[i]
        for j in range(i+1,n):
            g=features[j]
            duration=min(abs(math.log(max(f['duration'],1)/max(g['duration'],1)))/4.,1.)
            amp=base.amplitude_distance(f,g,centers,scale)
            observed_f=np.isfinite(f['array'][:,ns:ns+nn]).any(axis=0)
            observed_g=np.isfinite(g['array'][:,ns:ns+nn]).any(axis=0)
            numeric_shared=int((observed_f & observed_g).sum())
            # Store pair diagnostics, even when numeric-only is ineligible.
            entry={'i':i,'j':j,'duration_difference_s':abs(f['duration']-g['duration']),
                   'event_count_difference':abs(f['event_count']-g['event_count']),
                   'coverage_difference':abs(f['coverage']-g['coverage']),
                   'numeric_shared_channels':numeric_shared,
                   'numeric_eligible_pair':bool(eligible[i] and eligible[j])}
            for name in ('A_full','B_state_events','D_missing_neutral','E_mask_only_control'):
                shape=_shape(f['array'],g['array'],ns,nn,a.band,name)
                if name=='B_state_events':
                    value=.95*shape+.05*duration
                elif name=='E_mask_only_control':
                    value=shape  # ONLY missingness; deliberately no duration or physical level
                else:
                    value=.75*shape+.20*amp+.05*duration
                matrices[name][i,j]=matrices[name][j,i]=value
                entry[name]=value
            if eligible[i] and eligible[j]:
                shape=_shape(f['array'],g['array'],ns,nn,a.band,'C_numeric_only')
                value=.75*shape+.20*amp+.05*duration
                if numeric_shared==0:
                    # Undefined numeric comparison: avoid a misleading zero/small distance.
                    value=1.0
                x,y=num_lookup[i],num_lookup[j]
                matrices['C_numeric_only'][x,y]=matrices['C_numeric_only'][y,x]=value
                entry['C_numeric_only']=value
            else:
                entry['C_numeric_only']=np.nan
            pair_rows.append(entry)
    pair=pd.DataFrame(pair_rows)
    pair.to_csv(a.out/'pairwise_ablation_distances.csv.gz',index=False,compression='gzip')
    (a.out/'distances').mkdir(exist_ok=True)
    for name,D in matrices.items():
        np.savetxt(a.out/'distances'/f'{name}.csv',D,delimiter=',',fmt='%.8f')
    return matrices,nums,pair


def partition(D,max_k):
    n=len(D)
    if n<3 or np.max(D)<1e-10:
        return {'status':'insufficient_or_identical','labels':None,'k':None,'diagnostics':pd.DataFrame(),'z':None}
    try:
        Z=linkage(squareform(D,checks=False),method='average',optimal_ordering=n<=120)
        choices=[];labels={}
        for k in range(2,min(max_k,n-1)+1):
            l=fcluster(Z,k,criterion='maxclust')
            counts=pd.Series(l).value_counts()
            if len(counts)<2 or len(counts)>=n:continue
            sil=float(silhouette_score(D,l,metric='precomputed'))
            big=float(counts.max()/n);single=int((counts==1).sum())
            balanced=big<=.70 and single/len(counts)<=.20
            choices.append({'k_requested':k,'k_actual':len(counts),'silhouette':sil,
                            'largest_cluster_fraction':big,'singleton_clusters':single,
                            'small_clusters_le2':int((counts<=2).sum()),
                            'passes_balance_check':balanced})
            labels[k]=l
        t=pd.DataFrame(choices)
        if t.empty:return {'status':'no_valid_partition','labels':None,'k':None,'diagnostics':t,'z':Z}
        filtered=t[t.passes_balance_check]
        filtered=filtered if len(filtered) else t
        winner=filtered.sort_values(['silhouette','k_requested'],ascending=[False,True]).iloc[0]
        k=int(winner.k_requested)
        return {'status':'balanced_exploratory' if t.passes_balance_check.any() else 'unbalanced_exploratory',
                'labels':labels[k], 'k':k, 'diagnostics':t,'z':Z,
                'fixed':{kk:labels[kk] for kk in labels}}
    except (ValueError,RuntimeError) as e:
        return {'status':'error_'+str(e),'labels':None,'k':None,'diagnostics':pd.DataFrame(),'z':None}


def rho(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    good=np.isfinite(x)&np.isfinite(y)
    if good.sum()<4 or np.std(x[good])<1e-12 or np.std(y[good])<1e-12:
        return np.nan
    return float(spearmanr(x[good],y[good]).statistic)


def plot_cluster_medoid(out,name,D,labels,subdf,allfeatures,a,original_indices):
    if a.max_medoid_plots<=0:return []
    d=a.out/'medoids'/name
    d.mkdir(parents=True,exist_ok=True)
    selected=[]
    order=sorted(set(labels),key=lambda k:-int((labels==k).sum()))[:a.max_medoid_plots]
    for cl in order:
        group=np.flatnonzero(labels==cl)
        middle=int(group[np.argmin(D[np.ix_(group,group)].sum(axis=1))])
        r=subdf.iloc[middle]
        index=int(original_indices[middle])
        selected.append({'variant':name,'cluster':cl,'n_activities':len(group),
                         'medoid_case_id':r.case_id,'medoid_activity_id':r.activity_id,
                         'start_s':r.start_s,'end_s':r.end_s})
        folder=locate_case_dir(a.data_root,str(r.case_id))
        raw,norm,cnt,mask=base.load_files(folder)
        ix=(raw.index.to_numpy(float)>=float(r.start_s))&(raw.index.to_numpy(float)<float(r.end_s))
        x=raw.index.to_numpy(float)[ix]-float(r.start_s)
        fig,axes=plt.subplots(2+len(a.numeric_codes),1,figsize=(11,2.05*(2+len(a.numeric_codes))),
                              sharex=True,layout='constrained')
        for ci,code in enumerate(a.state_codes):
            col='E'+code
            if col not in raw:continue
            y=raw.loc[ix,col].to_numpy(float)
            axes[0].step(x,y+ci*2.,where='post',label=code,linewidth=1.2)
        axes[0].set_ylabel('State + offset');axes[0].legend(fontsize=8,ncol=3)
        cc=[c for c in a.state_codes if 'E'+c in cnt]
        if cc:
            v=cnt.loc[ix,['E'+c for c in cc]].sum(axis=1).to_numpy(float)
            axes[1].bar(x,v,width=1.0,alpha=.5)
        axes[1].set_ylabel('Events / bin')
        for k,code in enumerate(a.numeric_codes):
            ax=axes[k+2]; col='E'+code
            if col in raw:
                y=raw.loc[ix,col].to_numpy(float)
                m=mask.loc[ix,col].to_numpy(float)>0
                good=np.isfinite(y)&m
                ax.scatter(x[good],y[good],s=7,alpha=.75)
                ax.text(.99,.94,f'{good.sum()} observed bins',transform=ax.transAxes,
                        va='top',ha='right',fontsize=8)
            ax.set_ylabel(code+'\n(raw)')
        axes[0].set_title(f'{name} | cluster {cl} n={len(group)} | {r.case_id} activity={r.activity_id}')
        axes[-1].set_xlabel('Time since activity start (s)')
        for ax in axes: ax.grid(axis='x',alpha=.15)
        fig.savefig(d/f'cluster_{cl:02d}_raw_medoid.png',dpi=135)
        plt.close(fig)
        del raw,norm,cnt,mask
    return selected


def results_and_visualizations(df,features,matrices,nums,pair,a):
    partitions={};all_assign=[];overview=[];krows=[];medoids=[];cluster_details=[]
    for name in VARIANTS:
        ix=nums if name=='C_numeric_only' else np.arange(len(df),dtype=int)
        sub=df.iloc[ix].copy()
        D=matrices[name]
        r=partition(D,a.max_k)
        partitions[name]=(ix,r)
        if not r['diagnostics'].empty:
            t=r['diagnostics'].copy();t.insert(0,'variant',name);krows.append(t)
        assigned=sub.copy()
        assigned['variant']=name
        assigned['cluster']=r['labels'] if r['labels'] is not None else np.nan
        assigned['selected_k']=r['k']
        all_assign.append(assigned)
        if r['labels'] is not None:
            counts=pd.Series(r['labels']).value_counts()
            sil=float(r['diagnostics'].loc[r['diagnostics'].k_requested==r['k'],'silhouette'].iloc[0])
            largest=counts.max()/len(sub)
            singleton=(counts==1).sum()
            for cl in sorted(counts.index):
                members=sub.iloc[np.flatnonzero(r['labels']==cl)]
                present=np.flatnonzero(r['labels']==cl)
                medoid_local=int(present[np.argmin(D[np.ix_(present,present)].sum(axis=1))])
                medoid=sub.iloc[medoid_local]
                cluster_details.append({'variant':name,'cluster':int(cl),
                    'n_activities':len(members),'n_cases':members.case_id.nunique(),
                    'median_duration_s':float(members.duration_s.median()),
                    'median_event_count':float(members.event_count_recomputed.median()),
                    'median_numeric_coverage':float(members.numeric_coverage_recomputed.median()),
                    'zero_coverage_fraction':float((members.numeric_coverage_recomputed==0).mean()),
                    'dominant_activity_type':members.activity_type.mode().iloc[0] if 'activity_type' in members else 'unknown',
                    'medoid_case_id':medoid.case_id,'medoid_activity_id':medoid.activity_id})
            medoids.extend(plot_cluster_medoid(a.out,name,D,r['labels'],sub,features,a,ix))
        else:
            sil=np.nan;largest=np.nan;singleton=np.nan
        overview.append({'variant':name,'n_activities':len(ix),'n_cases':sub.case_id.nunique(),
                         'selected_k':r['k'],'silhouette':sil,'largest_cluster_fraction':largest,
                         'singleton_clusters':singleton,'status':r['status'],
                         'median_numeric_coverage':sub.numeric_coverage_recomputed.median(),
                         'numeric_zero_coverage_fraction':float((sub.numeric_coverage_recomputed==0).mean())})
        if r['z'] is not None:
            fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
            dendrogram(r['z'],no_labels=len(ix)>40,ax=ax,
                       color_threshold=0,above_threshold_color='#53657d')
            ax.set_title(f'{name} | {len(ix)} fixed-activity candidates')
            ax.set_ylabel('DTW-based distance')
            fig.savefig(a.out/f'dendrogram_{name}.png',dpi=135)
            plt.close(fig)

    overview=pd.DataFrame(overview)
    overview.to_csv(a.out/'ablation_summary.csv',index=False)
    pd.DataFrame(cluster_details,columns=['variant','cluster','n_activities','n_cases','median_duration_s',
            'median_event_count','median_numeric_coverage','zero_coverage_fraction',
            'dominant_activity_type','medoid_case_id','medoid_activity_id']).to_csv(
            a.out/'cluster_summary_by_variant.csv',index=False)
    assignment=pd.concat(all_assign,ignore_index=True)
    assignment.to_csv(a.out/'ablation_cluster_assignments.csv',index=False)
    pd.concat(krows,ignore_index=True).to_csv(a.out/'ablation_k_diagnostics.csv',index=False) if krows else None
    pd.DataFrame(medoids,columns=['variant','cluster','n_activities','medoid_case_id','medoid_activity_id','start_s','end_s']).to_csv(a.out/'medoid_index.csv',index=False)

    # Fair comparison: ARI on exact intersection of activity identities; numeric-only is subset.
    for fixed_k in (None,4,8):
        records=[]
        for m1 in VARIANTS:
            ix1,r1=partitions[m1]
            for m2 in VARIANTS:
                ix2,r2=partitions[m2]
                common=np.intersect1d(ix1,ix2)
                l1 = r1['labels'] if fixed_k is None else r1.get('fixed',{}).get(fixed_k)
                l2 = r2['labels'] if fixed_k is None else r2.get('fixed',{}).get(fixed_k)
                value=np.nan
                if l1 is not None and l2 is not None and len(common)>=3:
                    p1=np.searchsorted(ix1,common);p2=np.searchsorted(ix2,common)
                    value=float(adjusted_rand_score(l1[p1],l2[p2]))
                records.append({'variant_1':m1,'variant_2':m2,'intersection_n':len(common),
                                'ari':value,'fixed_k':'optimal' if fixed_k is None else fixed_k})
        out_name='ari_common_best.csv' if fixed_k is None else f'ari_common_k{fixed_k}.csv'
        pd.DataFrame(records).to_csv(a.out/out_name,index=False)

    # Cross-tab: which ORIGINAL A clusters split or merge under each input variant?
    cross_dir=a.out/'cluster_crosstabs'
    cross_dir.mkdir(exist_ok=True)
    base_ix,base_part=partitions['A_full']
    if base_part['labels'] is not None:
        for other in VARIANTS[1:]:
            other_ix,other_part=partitions[other]
            if other_part['labels'] is None:continue
            shared=np.intersect1d(base_ix,other_ix)
            if len(shared)<3:continue
            label_a=base_part['labels'][np.searchsorted(base_ix,shared)]
            label_b=other_part['labels'][np.searchsorted(other_ix,shared)]
            pd.crosstab(pd.Series(label_a,name='A_full cluster'),
                        pd.Series(label_b,name=f'{other} cluster')).to_csv(
                        cross_dir/f'A_full_vs_{other}.csv')

    # Bias audit: pairs are NOT independent, and these coefficients are descriptive.
    bias=[]
    for name in VARIANTS:
        for col in ['duration_difference_s','event_count_difference','coverage_difference']:
            s=pair[name].to_numpy(float)
            bias.append({'variant':name,'factor':col,
                         'spearman_rho':rho(s,pair[col]),
                         'n_pairs':int(np.isfinite(s).sum()),
                         'note':'Descriptive only; pairwise observations not independent'})
    b=pd.DataFrame(bias)
    b.to_csv(a.out/'bias_summary.csv',index=False)

    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    t=pd.concat(krows,ignore_index=True) if krows else pd.DataFrame()
    if len(t):
        for name,g in t.groupby('variant',sort=False):
            ax.plot(g.k_requested,g.silhouette,marker='o',label=f'{name} (n={matrices[name].shape[0]})')
        ax.legend(fontsize=8)
    ax.set_xlabel('Requested cluster count k');ax.set_ylabel('Silhouette (higher is better)')
    ax.set_title('Ablation | silhouette by k (C uses coverage-selected subset)');ax.grid(alpha=.25)
    fig.savefig(a.out/'silhouette_by_k.png',dpi=170);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    means=overview.copy()
    axs[0].bar(means.variant,means.largest_cluster_fraction)
    axs[0].set_ylabel('Largest cluster fraction');axs[0].tick_params(axis='x',rotation=25)
    axs[1].bar(means.variant,means.singleton_clusters)
    axs[1].set_ylabel('Singleton clusters');axs[1].tick_params(axis='x',rotation=25)
    for ax in axs:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Cluster collapse and singleton diagnostic')
    fig.savefig(a.out/'cluster_balance.png',dpi=170);plt.close(fig)

    tab=b.pivot(index='variant',columns='factor',values='spearman_rho').reindex(VARIANTS)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    img=ax.imshow(tab.to_numpy(float),cmap='coolwarm',vmin=-1,vmax=1,aspect='auto')
    ax.set_yticks(np.arange(len(tab)));ax.set_yticklabels(tab.index,fontsize=9)
    ax.set_xticks(np.arange(len(tab.columns)));ax.set_xticklabels(tab.columns,rotation=20,ha='right',fontsize=9)
    for i in range(len(tab)):
        for j in range(len(tab.columns)):
            v=tab.iloc[i,j]
            ax.text(j,i,'N/A' if not np.isfinite(v) else f'{v:+.2f}',ha='center',va='center',fontsize=10)
    fig.colorbar(img,ax=ax,label='Spearman rho (descriptive)')
    ax.set_title('DTW distance vs confounders (activity pairs not independent)')
    fig.savefig(a.out/'distance_bias_heatmap.png',dpi=170);plt.close(fig)

    pivot=pd.read_csv(a.out/'ari_common_best.csv').pivot(index='variant_1',columns='variant_2',values='ari').reindex(index=VARIANTS,columns=VARIANTS)
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    im=ax.imshow(pivot.to_numpy(),cmap='viridis',vmin=0,vmax=1)
    ax.set_xticks(np.arange(len(VARIANTS)));ax.set_xticklabels(VARIANTS,rotation=30,ha='right')
    ax.set_yticks(np.arange(len(VARIANTS)));ax.set_yticklabels(VARIANTS)
    for i in range(len(VARIANTS)):
        for j in range(len(VARIANTS)):
            v=pivot.iloc[i,j]
            ax.text(j,i,'NA' if not np.isfinite(v) else f'{v:.2f}',ha='center',va='center',
                    color='white' if np.isfinite(v) and v<.65 else 'black',fontsize=10)
    fig.colorbar(im,ax=ax,label='ARI on shared activities')
    ax.set_title('Cluster consistency (best k; C evaluated on shared subset)')
    fig.savefig(a.out/'ari_comparison.png',dpi=170);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for idx,name in enumerate(('A_full','D_missing_neutral')):
        g=assignment[assignment.variant==name]
        cohorts=[g.loc[g.cluster==cl,'numeric_coverage_recomputed'].to_numpy(float) for cl in sorted(g.cluster.dropna().unique())]
        axs[idx].boxplot(cohorts,showfliers=False)
        axs[idx].set_xlabel('Cluster ID');axs[idx].set_ylabel('Numeric observation coverage')
        axs[idx].set_title(name);axs[idx].grid(axis='y',alpha=.2)
    fig.suptitle('Are clusters separated by numeric measurement availability?')
    fig.savefig(a.out/'coverage_by_cluster.png',dpi=170);plt.close(fig)

    # Compare A full distances to baseline original matrix when original ordering/parameters match.
    baseline_file=a.baseline_root/'dtw_distance_matrix.csv'
    fidelity={'baseline_distance_file_found':baseline_file.exists(),
              'same_original_activity_count':False,'max_absolute_difference':None,
              'comment':'Expected to match if max-bins, band, input codes and sample are unchanged'}
    if baseline_file.exists():
        try:
            old=np.loadtxt(baseline_file,delimiter=',')
            if old.shape==matrices['A_full'].shape and len(df)==len(pd.read_csv(a.baseline_root/'cluster_assignments.csv')):
                fidelity['same_original_activity_count']=True
                fidelity['max_absolute_difference']=float(np.max(np.abs(old-matrices['A_full'])))
                if fidelity['max_absolute_difference']>1e-5:
                    fidelity['comment']='WARNING: reproduction differs; check --max-bins / --band / codes / missing data'
            else:
                fidelity['comment']='Not directly comparable: activity subset or unreadable rows'
        except (ValueError,OSError) as ex:
            fidelity['comment']='Failed to load original matrix: '+str(ex)
    (a.out/'baseline_fidelity.json').write_text(json.dumps(fidelity,indent=2,ensure_ascii=False),encoding='utf-8')
    report={'n_baseline_selected':len(df),'variants':overview.to_dict('records'),
            'numeric_only_eligibility':{'eligible':int(len(nums)),'ineligible':int(len(df)-len(nums)),
                                        'threshold':a.numeric_min_coverage,'min_bins':a.numeric_min_bins},
            'baseline_fidelity':fidelity,'warnings':[
                'Unsupervised cluster structure does NOT prove clinical operation categories.',
                'Numeric-only C uses a different eligible subset; compare ARI only on shared activities.',
                'E_mask_only_control uses ONLY observation availability (negative control), not clinical signals.',
                'State and event codes are a narrow subset, not all 259 recorded codes.',
                'Distances are diagnostics; correlations across pairs are not independent observations.']}
    (a.out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('COMPLETE. See ablation_summary.csv, baseline_fidelity.json, and plots.',flush=True)


def main():
    a=arguments()
    (a.out/'analysis_config.json').write_text(json.dumps({
        'data_root':str(a.data_root),'baseline_root':str(a.baseline_root),
        'max_activities':a.max_activities,'max_bins':a.max_bins,'band':a.band,'max_k':a.max_k,
        'numeric_min_coverage':a.numeric_min_coverage,'numeric_min_bins':a.numeric_min_bins,
        'state_codes':a.state_codes,'numeric_codes':a.numeric_codes,
        'variants':VARIANTS,'numeric_baseline_group_weights':base.GROUP_W,
        'notes':'Activity intervals come strictly from baseline cluster_assignments.csv. No re-segmentation.'
    },indent=2,ensure_ascii=False),encoding='utf-8')
    df=restore_fixed_activities(a)
    print(f'Fixed v3.1 sample: {len(df)} activities from {df.case_id.nunique()} cases',flush=True)
    df,features=build_signatures(df,a)
    mats,nums,pair=build_pairwise(df,features,a)
    results_and_visualizations(df,features,mats,nums,pair,a)


if __name__=='__main__':main()
