"""Model A v3.1: audit hierarchical Hybrid activities and run small-sample mixed-signal DTW.

Inputs are EXISTING chair_model_A_results and chair_hierarchical_v3 output.
No clinical labels are inferred. No original data are changed.

Usage: python chair_dtw_v3_1.py --data-root ... --hierarchy-root ... --out ...
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import math
import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Bundled unchanged implementation from v3, used to ensure sensitivity uses the SAME boundaries.
import segmentation_core as seg
import chair_hierarchical_v3 as v3
from case_first_sampling import choose_case_first

STATE_CODES = ['20039','20011','20101']
NUMERIC_CODES = ['20742','20743','20108']
EXCLUDE_CODES = {'90012'}
GROUP_W = {'state':.40,'numeric':.35,'events':.25}
TYPE_ORDER = ['HP1','HP2','mixed','syringe_sensor','numeric_activity','other_logged_events','idle','unclassified']


def parse_args():
    p=argparse.ArgumentParser(description='Hybrid v3.1 quality audit + small mixed-DTW (no clinical labels)')
    p.add_argument('--data-root',type=Path,required=True,help='chair_model_A_results')
    p.add_argument('--hierarchy-root',type=Path,required=True,help='chair_hierarchical_v3_100 result directory')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cases',type=int,default=15,help='Number of v3 cases (0=all v3 cases)')
    p.add_argument('--activities',type=int,default=60,help='Max activities for full pairwise DTW')
    p.add_argument('--max-per-case',type=int,default=8)
    p.add_argument('--allow-case-shortfall',action='store_true',
                   help='Allow fewer cases than requested if files/activities are unusable; otherwise fail with audit')
    p.add_argument('--max-bins',type=int,default=96,help='Max length of each activity after masked bin aggregation')
    p.add_argument('--band',type=float,default=.25,help='Sakoe-Chiba radius as a fraction of max activity bins')
    p.add_argument('--snap-tolerance',type=float,default=1.,help='Snap boundary to HP-episode edge if within this value (s)')
    p.add_argument('--sensitivity-tolerance',type=float,default=2.,help='Sensitivity boundary agreement tolerance (s)')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-k',type=int,default=8)
    p.add_argument('--plots-only',action='store_true',help='Render figures from existing saved clustering CSVs; NO DTW recalculation')
    p.add_argument('--plot-mode',choices=['light','medoids','diagnostics','full'],default='light',help='light=low memory overlays (default); medoids=no overlays; diagnostics=no medoids; full=more detail')
    p.add_argument('--refresh-plots',action='store_true',help='Rebuild existing PNG figures; default is to reuse complete PNG files')
    p.add_argument('--no-exact-events',action='store_true',help='Skip raw exact-event CSV: plot from 1-second data only')
    p.add_argument('--exact-max-points',type=int,default=3000,help='Max raw measured points per code in each medoid plot')
    p.add_argument('--num-codes',default=','.join(NUMERIC_CODES),help='Comma-separated numeric code candidates')
    p.add_argument('--state-codes',default=','.join(STATE_CODES),help='Comma-separated state code candidates')
    a=p.parse_args()
    if a.cases<0 or a.activities<2 or a.max_per_case<1 or a.max_bins<16 or not (0<a.band<=1) or a.exact_max_points<10:
        p.error('Invalid cases / activities / max-bins / band')
    a.out.mkdir(parents=True,exist_ok=True)
    a.num_codes=[c.strip().removeprefix('E') for c in a.num_codes.split(',') if c.strip()]
    a.state_codes=[c.strip().removeprefix('E') for c in a.state_codes.split(',') if c.strip()]
    return a


def load_files(case_dir):
    raw=seg.load_table(case_dir,'raw_1s')
    norm=seg.load_table(case_dir,'normalized_1s')
    cnt=seg.load_table(case_dir,'event_counts_1s')
    mask=seg.load_table(case_dir,'observation_mask_1s')
    idx=raw.index
    norm=norm.reindex(index=idx,columns=raw.columns)
    cnt=cnt.reindex(index=idx,columns=raw.columns).fillna(0)
    mask=mask.reindex(index=idx,columns=raw.columns).fillna(0)
    return raw,norm,cnt,mask


class OnDemandCaseData:
    """Keep at most max_loaded cases in RAM; reload older cases from disk."""
    def __init__(self, case_lookup, max_loaded=1):
        self.case_lookup = case_lookup
        self.max_loaded = max(1, int(max_loaded))
        self.cache = OrderedDict()

    def __getitem__(self, case_id):
        if case_id in self.cache:
            self.cache.move_to_end(case_id)
            return self.cache[case_id]
        data = load_files(self.case_lookup[case_id])
        self.cache[case_id] = data
        if len(self.cache) > self.max_loaded:
            self.cache.popitem(last=False)
        return data


def read_exact_activity_events(path, start, end, numeric_codes, max_points=3000):
    """Stream gzip CSV and select only requested raw measurements in an Activity.

    Uses the Python csv reader rather than pandas' full-file C tokenizer.
    Maintains a bounded, deterministic, chronological subsample per code.
    Returns a small DataFrame, not the whole case's event log.
    """
    wanted = set(map(str, numeric_codes))
    buffers = {c: [] for c in wanted}
    seen = {c: 0 for c in wanted}
    strides = {c: 1 for c in wanted}
    with gzip.open(path, 'rt', encoding='utf-8-sig', errors='replace', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'time_s', 'event_code', 'event_value'}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f'Raw-event CSV missing required columns: {path}')
        for row in reader:
            code = (row.get('event_code') or '').strip()
            if code not in wanted:
                continue
            try:
                time_s = float(row['time_s'])
                if not (start <= time_s < end):
                    continue
                value = float(row['event_value'])
                if not (np.isfinite(time_s) and np.isfinite(value)):
                    continue
            except (ValueError, TypeError):
                continue
            seen[code] += 1
            if (seen[code] - 1) % strides[code] != 0:
                continue
            buffers[code].append((time_s, code, value))
            if len(buffers[code]) > max_points:
                buffers[code] = buffers[code][::2]
                strides[code] *= 2
    records = [item for code in numeric_codes for item in buffers.get(code, [])]
    return pd.DataFrame.from_records(records, columns=['time_s','event_code','event_value']), seen


def load_cases(a, intervals):
    choices=sorted(intervals.case_id.unique().tolist())
    sample_file=a.hierarchy_root/'selected_cases.csv'
    if sample_file.exists():
        df=pd.read_csv(sample_file,dtype={'date_folder':str,'chair_folder':str,'seq':str})
        ordering=['__'.join((str(r.date_folder),str(r.chair_folder),f'Chair_log_seq_{r.seq}')) for r in df.itertuples()]
        choices=[x for x in ordering if x in choices]
    if a.cases and len(choices)>a.cases:
        rng=np.random.default_rng(a.seed)
        ix=np.sort(rng.choice(len(choices),size=a.cases,replace=False))
        choices=[choices[i] for i in ix]
    result=[]
    for cid in choices:
        parts=cid.split('__')
        if len(parts)!=3:continue
        d=a.data_root/'cases'/parts[0]/parts[1]/parts[2]
        if all((d/(k+'.csv.gz')).exists() for k in ('raw_1s','normalized_1s','event_counts_1s','observation_mask_1s')):
            result.append((cid,d))
    if not result:raise FileNotFoundError('No matching preprocessed case folders. Check both roots.')
    return result


def clip_refine_boundaries(boundaries,episodes,duration,tolerance,min_activity):
    """Move close interior HP boundaries to nearest episode edge; otherwise drop.

    This correction is explicit and audited; no fabricated HP transitions.
    """
    hp=episodes[episodes.operation_type.isin(('HP1','HP2'))].copy() if len(episodes) else episodes
    proposed=[];audits=[]
    for b in sorted(set(map(float,boundaries))):
        overlaps=hp[(hp.start_s+1e-6<b)&(hp.end_s-1e-6>b)] if len(hp) else hp
        proposal=b;action='unchanged'
        if len(overlaps):
            edges=np.r_[overlaps.start_s.to_numpy(float),overlaps.end_s.to_numpy(float)]
            nearest=float(edges[np.argmin(abs(edges-b))])
            if abs(nearest-b)<=tolerance:
                proposal=nearest;action='snapped_to_hp_episode_edge'
            else:
                action='removed_internal_hp_episode'
        if action!='removed_internal_hp_episode' and min_activity<=proposal<=duration-min_activity:
            proposed.append((proposal,b,action))
        elif action!='removed_internal_hp_episode':
            action='removed_near_case_edge'
        audits.append({'original_s':b,'refined_s':proposal if action not in ('removed_internal_hp_episode','removed_near_case_edge') else np.nan,
                       'action':action})
    # If close boundaries result from a snap, retain first chronologically.
    keep=[]
    for proposal,original,action in sorted(proposed):
        if not keep or proposal-keep[-1][0]>=min_activity:
            keep.append((proposal,original,action))
        else:
            for audit in audits:
                if audit['original_s']==original:
                    audit['action']='removed_too_close_after_refine';audit['refined_s']=np.nan
    return np.array([b[0] for b in keep],dtype=float),audits


def boundary_matching(a,b,tol):
    """One-to-one tolerance match, not just a count comparison."""
    a=np.sort(np.asarray(a,float));b=np.sort(np.asarray(b,float))
    pairs=[]; candidates=[]
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            if abs(x-y)<=tol:candidates.append((abs(x-y),i,j))
    used_a=set();used_b=set()
    for d,i,j in sorted(candidates):
        if i not in used_a and j not in used_b:
            used_a.add(i);used_b.add(j);pairs.append(d)
    p=len(pairs)/len(a) if len(a) else (1. if not len(b) else 0.)
    r=len(pairs)/len(b) if len(b) else (1. if not len(a) else 0.)
    return {'n_match':len(pairs),'match_precision':p,'match_recall':r,
            'boundary_f1':(2*p*r/(p+r)) if p+r>0 else 0.,
            'median_shift_s':float(np.median(pairs)) if pairs else np.nan}


def classify_activity(row,raw,norm,cnt,mask,numeric_codes):
    """Only descriptive code-content categories; unanchored != clinically idle."""
    old=str(row['composition_label'])
    if old!='unanchored':return old,0,0,0
    t=raw.index.to_numpy(float)
    ix=(t>=float(row['start_s']))&(t<float(row['end_s']))
    if not ix.any():return 'unclassified',0,0,0
    arr=cnt.iloc[np.flatnonzero(ix)]
    noncontrol=[c for c in arr.columns if c!='E90012']
    event_count=float(arr[noncontrol].sum().sum())
    distinct_codes=int((arr[noncontrol].sum(axis=0)>0).sum())
    changed=0
    for code in numeric_codes:
        col='E'+code
        if col not in raw:continue
        x=norm.loc[ix,col].to_numpy(float)
        mm=mask.loc[ix,col].to_numpy(float)>0
        x=x[np.isfinite(x)&mm]
        if len(x)>=3 and np.nanpercentile(x,90)-np.nanpercentile(x,10)>.3:
            changed+=1
    if changed>0:return 'numeric_activity',distinct_codes,changed,event_count
    if event_count<2 and distinct_codes<=1:return 'idle',distinct_codes,changed,event_count
    if distinct_codes>=2:return 'other_logged_events',distinct_codes,changed,event_count
    return 'unclassified',distinct_codes,changed,event_count


def segment_to_sequence(row,raw,norm,cnt,mask,state_codes,numeric_codes,max_bins):
    """Adaptive-bin aggregation, not interpolation: missing measurements stay missing."""
    time=raw.index.to_numpy(float)
    ix=np.flatnonzero((time>=row.start_s)&(time<row.end_s))
    if not len(ix):return None
    n=min(len(ix),max_bins)
    groups=np.array_split(ix,n)
    values=np.full((n,len(state_codes)+len(numeric_codes)+len(state_codes)),np.nan,float)
    # state channels: last known reading in the bin, no invented initial zero
    for j,c in enumerate(state_codes):
        col='E'+c
        if col not in raw:continue
        x=raw[col].to_numpy(float)
        m=mask[col].to_numpy(float)>0
        for k,g in enumerate(groups):
            vv=x[g][np.isfinite(x[g]) & m[g]]
            if len(vv):values[k,j]=vv[-1]
    # numeric channels: observed z-score median within each adaptive bin
    ns=len(state_codes)
    for j,c in enumerate(numeric_codes):
        col='E'+c
        if col not in norm:continue
        x=norm[col].to_numpy(float)
        m=mask[col].to_numpy(float)>0
        for k,g in enumerate(groups):
            vv=x[g][np.isfinite(x[g]) & m[g]]
            if len(vv):values[k,ns+j]=float(np.clip(np.median(vv),-4,4))
    # events: log rate, with zeros true zeros (event count always observed)
    for j,c in enumerate(state_codes):
        col='E'+c
        if col not in cnt:continue
        z=cnt[col].to_numpy(float)
        for k,g in enumerate(groups):
            duration_bins=max(float(len(g)),1.)
            values[k,ns+len(numeric_codes)+j]=float(np.clip(np.log1p(z[g].sum()/duration_bins),0,3))
    if np.all(np.isnan(values)):return None
    # raw physical features; keep original raw magnitude for cluster interpretation.
    phys=[];coverage=[]
    for c in numeric_codes:
        col='E'+c
        if col not in raw:
            phys.extend([np.nan,np.nan]);coverage.append(0.)
            continue
        x=raw[col].to_numpy(float)[ix];m=mask[col].to_numpy(float)[ix]>0
        x=x[np.isfinite(x)&m]
        phys.extend([float(np.median(x)) if len(x) else np.nan,
                     float(np.quantile(x,.95)) if len(x) else np.nan])
        coverage.append(len(x)/len(ix))
    # event count is documented for bias audit, not directly used as amplitude feature
    event_n=float(cnt.iloc[ix].drop(columns=['E90012'],errors='ignore').sum().sum())
    return {'array':values,'physical':np.asarray(phys,dtype=float),'duration':float(row.duration_s),
            'coverage':float(np.mean(coverage)) if coverage else 0.,'event_count':event_n,
            'time_s':time[ix], 'index':ix}


def local_cost_matrix(A,B,ns,nn):
    """Group-balanced local distances; missing numeric observations not filled with zero."""
    n,m=A.shape[0],B.shape[0]
    total=np.zeros((n,m),float);weighted=np.zeros((n,m),float)
    # state mismatch: unequal *known* values only; missing status penalized mildly
    if ns:
        x=A[:,:ns];y=B[:,:ns]
        kk=np.isfinite(x)[:,None,:]&np.isfinite(y)[None,:,:]
        either=np.isfinite(x)[:,None,:]^np.isfinite(y)[None,:,:]
        c=np.where(kk,(x[:,None,:]!=y[None,:,:]).astype(float),np.where(either,.25,0.))
        denom=(kk|either).sum(axis=2)
        valid=denom>0
        score=np.divide(c.sum(axis=2),denom,out=np.zeros((n,m)),where=valid)
        total+=GROUP_W['state']*score*valid;weighted+=GROUP_W['state']*valid
    # numeric: compare only simultaneous numeric observations; one-sided missing gets small cost
    if nn:
        x=A[:,ns:ns+nn];y=B[:,ns:ns+nn]
        kk=np.isfinite(x)[:,None,:]&np.isfinite(y)[None,:,:]
        either=np.isfinite(x)[:,None,:]^np.isfinite(y)[None,:,:]
        dist=np.minimum(np.abs(np.nan_to_num(x)[:,None,:]-np.nan_to_num(y)[None,:,:])/4.,1.)
        c=np.where(kk,dist,np.where(either,.25,0.))
        denom=(kk|either).sum(axis=2)
        valid=denom>0
        score=np.divide(c.sum(axis=2),denom,out=np.zeros((n,m)),where=valid)
        total+=GROUP_W['numeric']*score*valid;weighted+=GROUP_W['numeric']*valid
    if ns:
        x=A[:,ns+nn:];y=B[:,ns+nn:]
        score=np.mean(np.minimum(np.abs(np.nan_to_num(x)[:,None,:]-np.nan_to_num(y)[None,:,:]),1.),axis=2)
        total+=GROUP_W['events']*score;weighted+=GROUP_W['events']
    return np.divide(total,weighted,out=np.ones_like(total),where=weighted>0)


def dtw_cost(A,B,ns,nn,band):
    """DTW with length-normalized accumulated cost, modest diagonal corridor.

    A/B use native aggregation (max bins) but may have different lengths.
    """
    costs=local_cost_matrix(A,B,ns,nn)
    n,m=costs.shape
    D=np.full((n+1,m+1),np.inf);L=np.zeros((n+1,m+1),np.int32)
    D[0,0]=0
    radius=max(2,int(np.ceil(max(n,m)*band)))
    for i in range(1,n+1):
        center=(i-.5)*m/n+.5
        lo=max(1,int(np.floor(center-radius)))
        hi=min(m,int(np.ceil(center+radius)))
        for j in range(lo,hi+1):
            # diagonal preference breaks ties and prevents excessive repetitions
            diag=D[i-1,j-1];up=D[i-1,j];left=D[i,j-1]
            if diag<=up and diag<=left:val=diag;pi=i-1;pj=j-1
            elif up<=left:val=up;pi=i-1;pj=j
            else:val=left;pi=i;pj=j-1
            if not math.isfinite(val):continue
            D[i,j]=val+costs[i-1,j-1]
            L[i,j]=L[pi,pj]+1
    return float(D[n,m]/L[n,m]) if L[n,m] else 1.


def physical_scale(features):
    """Robust physical scaling across activities (not per-case min-max)."""
    x=np.array([s['physical'] for s in features],float)
    centers=np.nanmedian(x,axis=0)
    q1=np.nanpercentile(x,25,axis=0)
    q3=np.nanpercentile(x,75,axis=0)
    lo=np.nanpercentile(x,5,axis=0)
    hi=np.nanpercentile(x,95,axis=0)
    scale=np.maximum(q3-q1,(hi-lo)*.15)
    scale[~np.isfinite(scale)|(scale<1e-10)]=np.nan
    return centers,scale


def amplitude_distance(a,b,centers,scale):
    # Absolute observed level differences are kept, normalized by the *cohort* range.
    p=a['physical'];q=b['physical']
    good=np.isfinite(p)&np.isfinite(q)&np.isfinite(scale)
    if not good.any():return 0.  # tracked separately; do not claim amplitudes matched
    dist=np.minimum(np.abs(p[good]-q[good])/scale[good],4.)/4.
    return float(np.median(dist))


def pairwise(features,ns,nn,band):
    n=len(features);D=np.zeros((n,n),float);rows=[]
    centers,scale=physical_scale(features)
    for i in range(n):
        if (i+1)%10==0:print(f'DTW distances: {i+1}/{n} activities',flush=True)
        for j in range(i+1,n):
            a=features[i];b=features[j]
            shape=dtw_cost(a['array'],b['array'],ns,nn,band)
            amp=amplitude_distance(a,b,centers,scale)
            duration=min(abs(np.log(max(a['duration'],1)/max(b['duration'],1)))/4.,1.)
            final=.75*shape+.20*amp+.05*duration
            D[i,j]=D[j,i]=final
            rows.append({'i':i,'j':j,'shape_dtw':shape,'raw_level_distance':amp,
                         'duration_distance':duration,'final_distance':final,
                         'duration_difference_s':abs(a['duration']-b['duration']),
                         'event_count_difference':abs(a['event_count']-b['event_count'])})
    return D,pd.DataFrame(rows),centers,scale


def evaluate_clustering(D,max_k):
    n=len(D)
    if n<3:raise ValueError('At least three activities required')
    condensed=squareform(D,checks=False)
    if np.nanmax(D)<1e-9:
        raise ValueError('All activities have virtually identical distances; clustering not meaningful')
    Z=linkage(condensed,method='average',optimal_ordering=n<=120)
    stats=[];labels_by_k={}
    for k in range(2,min(max_k,n-1)+1):
        lab=fcluster(Z,k,criterion='maxclust')
        sizes=pd.Series(lab).value_counts()
        if len(sizes)<2 or len(sizes)>=n:
            continue
        sil=silhouette_score(D,lab,metric='precomputed')
        singleton=int((sizes==1).sum())
        largest=float(sizes.max()/n)
        viable=(largest<=.70 and singleton/len(sizes)<=.20 and sizes.max()<n-1)
        stats.append({'k_requested':k,'k_actual':len(sizes),'silhouette':float(sil),
                      'largest_cluster_fraction':largest,'singleton_clusters':singleton,
                      'small_clusters_le2':int((sizes<=2).sum()),'passes_balance_check':bool(viable)})
        labels_by_k[k]=lab
    if not stats:
        return Z,np.ones(n,int),pd.DataFrame(),0,'no_valid_partition'
    scores=pd.DataFrame(stats)
    allowed=scores[scores.passes_balance_check]
    candidates=allowed if len(allowed) else scores
    best=candidates.sort_values(['silhouette','k_requested'],ascending=[False,True]).iloc[0]
    choice=int(best.k_requested)
    status='exploratory_balanced' if len(allowed) else 'exploratory_unbalanced_NO_RELIABLE_CLASSES'
    return Z,labels_by_k[choice],scores,choice,status


def render_figures(out,metadata,features,D,Z,labels,numeric_codes,state_codes,cache,case_lookup,
                   no_exact_events=False,exact_max_points=3000):
    n=len(metadata)
    fig,ax=plt.subplots(figsize=(max(11,min(18,n*.13)),7),layout='constrained')
    dendrogram(Z,ax=ax,no_labels=(n>45),leaf_rotation=90,leaf_font_size=7,
               color_threshold=0,above_threshold_color='#53657d')
    ax.set_title(f'Hierarchical clustering | n={n} activities (distance = mixed DTW + raw level)')
    ax.set_ylabel('Distance (not clinical meaning)')
    fig.savefig(out/'cluster_dendrogram.png',dpi=170);plt.close(fig)
    order=leaves_list(Z)
    fig,ax=plt.subplots(figsize=(8,7),layout='constrained')
    im=ax.imshow(D[np.ix_(order,order)],aspect='auto',cmap='viridis',interpolation='nearest')
    fig.colorbar(im,ax=ax,label='Combined distance');ax.set_title('Distance heatmap (dendrogram leaf order)')
    ax.set_xlabel('Activity');ax.set_ylabel('Activity')
    fig.savefig(out/'distance_heatmap.png',dpi=170);plt.close(fig)

    classes=sorted(set(labels))
    for cl in classes:
        members=np.flatnonzero(labels==cl)
        if len(members)>1:
            central=np.argmin(D[np.ix_(members,members)].sum(axis=1))
            medoid_idx=members[central]
        else:medoid_idx=int(members[0])
        r=metadata.iloc[medoid_idx]
        cid=r.case_id
        raw,norm,cnt,mask=cache[cid]
        start=float(r.start_s);end=float(r.end_s)
        # Representative numeric traces: use exact original event values where available.
        exact_file=case_lookup[cid]/'events_original.csv.gz'
        exact=None
        if exact_file.exists() and not no_exact_events:
            try:
                exact, n_seen = read_exact_activity_events(
                    exact_file, start, end, numeric_codes, max_points=exact_max_points
                )
                capped = {code:n for code,n in n_seen.items() if n>exact_max_points}
                if capped:
                    print(f'Raw medoid plot {cid}: bounded sampling for high-density codes {capped}',flush=True)
            except (OSError,ValueError,KeyError,csv.Error,EOFError) as exc:
                print(f'Warning: unable to stream exact events for {cid}: {exc}. '
                      f'Using 1-second raw observations for the medoid plot.',flush=True)
                exact=None
        x=raw.index.to_numpy(float)
        ix=np.where((x>=start)&(x<end))[0]
        fig,axes=plt.subplots(2+len(numeric_codes),1,figsize=(13,max(7,2.25*(2+len(numeric_codes)))),sharex=True,layout='constrained')
        # panel 1 states: separate categories with stepped readings, never assumed zero before first record
        for j,code in enumerate(state_codes):
            col='E'+code
            if col not in raw:continue
            y=raw[col].to_numpy(float)[ix]
            good=np.isfinite(y)
            if good.any():
                axes[0].step(x[ix][good]-start,y[good]+j*3.,where='post',linewidth=1.1,label=code)
        axes[0].set_title(f'Cluster {cl} | n={len(members)} | REAL medoid: {cid} | {start:.1f}–{end:.1f}s')
        axes[0].set_ylabel('States (offset)')
        if axes[0].has_data():axes[0].legend(fontsize=8,loc='upper right',ncol=3)
        for j,code in enumerate(numeric_codes):
            ax=axes[j+1];col='E'+code
            used_exact=False
            if exact is not None:
                observed=exact[(exact.event_code==code)&exact.event_value.notna()]
                if len(observed):
                    ax.scatter(observed.time_s.to_numpy(float)-start,
                               observed.event_value.to_numpy(float),s=6,alpha=.6)
                    used_exact=True
            if not used_exact and col in raw:
                y=raw[col].to_numpy(float)[ix];obs=mask[col].to_numpy(float)[ix]>0
                good=np.isfinite(y)&obs
                if good.any():ax.scatter(x[ix][good]-start,y[good],s=8,alpha=.7)
                else:ax.text(.5,.5,'No observed numeric samples',transform=ax.transAxes,ha='center',va='center')
            elif not used_exact:ax.text(.5,.5,'Code absent',transform=ax.transAxes,ha='center',va='center')
            ax.set_ylabel(f'{code}\nRAW value')
        idx=cnt.index.to_numpy(float);sel=np.where((idx>=start)&(idx<end))[0]
        series=cnt.drop(columns=['E90012'],errors='ignore').sum(axis=1).to_numpy(float)
        axes[-1].step(idx[sel]-start,series[sel],where='post',linewidth=1.)
        axes[-1].set_ylabel('All events\nper bin');axes[-1].set_xlabel('Seconds since Activity start')
        for ax in axes:ax.grid(axis='x',alpha=.2)
        fig.savefig(out/'representatives'/f'cluster_{cl:02d}_raw_medoid.png',dpi=170);plt.close(fig)

        # overlay class members for a few recorded numeric signals: no time interpolation
        fig,axs=plt.subplots(len(numeric_codes),1,figsize=(12,max(3,2.5*len(numeric_codes))),layout='constrained')
        axs=np.atleast_1d(axs)
        # Load each member once, then draw all its numeric codes.
        for member in members[:30]:
            rr=metadata.iloc[member]
            _raw,_,_,_mask=cache[rr.case_id]
            tt=_raw.index.to_numpy(float)
            index=(tt>=rr.start_s)&(tt<rr.end_s)
            if not np.any(index):continue
            for p,code in enumerate(numeric_codes):
                col='E'+code
                if col not in _raw:continue
                yy=_raw[col].to_numpy(float)[index]
                mm=_mask[col].to_numpy(float)[index]>0
                valid=np.isfinite(yy)&mm
                if not np.any(valid):continue
                tnorm=(tt[index][valid]-rr.start_s)/max(rr.duration_s,1e-8)
                axs[p].scatter(tnorm,yy[valid],s=3,alpha=.08,color='#477bb1')
        for p,code in enumerate(numeric_codes):
            ax=axs[p];col='E'+code
            if col in raw:
                y=raw[col].to_numpy(float)[ix]
                m=mask[col].to_numpy(float)[ix]>0
                good=np.isfinite(y)&m
                if good.any():ax.scatter((x[ix][good]-start)/max(end-start,1e-8),y[good],s=13,alpha=.8,color='#cb542d',label='medoid')
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
            ax.set_ylabel(f'{code} raw');ax.grid(alpha=.15)
        axs[-1].set_xlabel('Relative time (only observed points; no interpolation)')
        fig.suptitle(f'Cluster {cl} | observed raw measurements, max 30 members + medoid')
        fig.savefig(out/'representatives'/f'cluster_{cl:02d}_member_overlay.png',dpi=170);plt.close(fig)


def main():
    a=parse_args()
    from memory_safe_plotting import plot_saved_results, finish_saved_diagnostics
    if a.plots_only:
        finish_saved_diagnostics(a.out)
        plot_saved_results(out=a.out,data_root=a.data_root,
                           numeric_codes=a.num_codes,state_codes=a.state_codes,
                           no_exact_events=a.no_exact_events,exact_max_points=a.exact_max_points,
                           mode=a.plot_mode,resume=not a.refresh_plots)
        return
    intervals=pd.read_csv(a.hierarchy_root/'hierarchical_hybrid_activity_intervals.csv',dtype={'seq':str,'case_id':str})
    operations=pd.read_csv(a.hierarchy_root/'level1_operations.csv',dtype={'case_id':str})
    episodes=pd.read_csv(a.hierarchy_root/'level1_episodes.csv',dtype={'case_id':str})
    methods=pd.read_csv(a.hierarchy_root/'activity_intervals_all_methods.csv',dtype={'case_id':str})
    conf_file=a.hierarchy_root/'analysis_config.json'
    conf=json.loads(conf_file.read_text(encoding='utf-8')) if conf_file.exists() else {}
    defaults={'block_gap':12.,'max_syringe_anchor':90.,'idle_gap':15.,'min_activity':15.,'lock_tolerance':1.,'state_min_step':10.,'change_threshold':.45,'smooth_seconds':3.,'max_signals':14}
    defaults.update({k:conf[k] for k in defaults if k in conf})
    params=SimpleNamespace(**defaults)
    gap=float(conf.get('estimated_episode_gap_s',1.5))
    cases=load_cases(a,intervals)
    if a.cases and len(cases)<a.cases and not a.allow_case_shortfall:
        raise RuntimeError(f'Only {len(cases)}/{a.cases} cases have required input files in --hierarchy-root/--data-root. '
                           f'Cannot guarantee case coverage; use --allow-case-shortfall only if intentional.')
    pd.DataFrame([{'case_id':cid,'case_dir':str(d)} for cid,d in cases]).to_csv(a.out/'selected_cases.csv',index=False)
    selected_ids={cid for cid,_ in cases};case_lookup=dict(cases)
    improved=[];audit=[];sensitivity=[];errors=[]
    # Process one full case at a time; do not retain 100 wide dataframes in memory.
    for ni,(cid,d) in enumerate(cases,1):
        try:
            raw,norm,cnt,mask=load_files(d)
            base=intervals[intervals.case_id==cid].sort_values('start_s')
            if base.empty:continue
            duration=float(base.end_s.max())
            ops=operations[operations.case_id==cid].copy()
            eps=episodes[episodes.case_id==cid].copy()
            original=base.start_s.to_numpy(float)[1:]
            refined,audit_lines=clip_refine_boundaries(original,eps,duration,a.snap_tolerance,params.min_activity)
            for q in audit_lines:audit.append({'case_id':cid,**q})
            boundary=np.r_[0.,refined,duration]
            for k,(s,e) in enumerate(zip(boundary[:-1],boundary[1:]),1):
                if e-s<1e-6:continue
                row={'case_id':cid,'seq':cid.rsplit('_',1)[-1], 'activity_id':k,'start_s':s,
                     'end_s':e,'duration_s':e-s,'method':'hierarchical_hybrid_v3_1'}
                within=eps[(eps.end_s>s)&(eps.start_s<e)]
                types=set(within.operation_type.astype(str))
                hp=types.intersection({'HP1','HP2'}); syringe='Syringe_picked' in types
                if len(hp)>1 or (hp and syringe):old='mixed'
                elif hp:old=next(iter(hp))
                elif syringe:old='syringe_sensor'
                else:old='unanchored'
                row['composition_label']=old
                category,n_codes,n_var,event_n=classify_activity(row,raw,norm,cnt,mask,a.num_codes)
                row.update({'activity_type':category,'active_code_count':n_codes,'numeric_changing_code_count':n_var,
                            'total_events':event_n,'contains_hp_episode':bool(hp),
                            'numeric_activity_is_candidate':bool(n_var>0 and old=='unanchored')})
                improved.append(row)

            # Recreate v3.1 boundary timing under pause-gap perturbation, not merely count difference.
            grid=raw.index.to_numpy(float)
            selected=seg.select_signals(raw,cnt,mask,params.max_signals)
            _,change_score=seg.change_edges(raw,cnt,mask,selected,params.state_min_step,
                                            params.change_threshold,float(np.median(np.diff(grid))),params.smooth_seconds)
            change_rows=methods[(methods.case_id==cid)&(methods.method=='change')].sort_values('start_s')
            change_b=change_rows.start_s.to_numpy(float)[1:]
            reference=None
            for factor in (.5,1.,2.):
                alt_eps,_=v3.operation_episodes(ops,gap*factor)
                alt_blocks=v3.coarse_blocks(alt_eps,params.block_gap,params.max_syringe_anchor)
                boundaries,_=v3.chosen_hybrid_times(change_b,change_score,grid,alt_blocks,alt_eps,duration,params)
                boundaries,_=clip_refine_boundaries(boundaries,alt_eps,duration,a.snap_tolerance,params.min_activity)
                if factor==1.:reference=boundaries
                sensitivity.append({'case_id':cid,'gap_factor':factor,'threshold_s':gap*factor,
                                    'n_boundaries':len(boundaries),'boundaries_seconds':json.dumps(boundaries.tolist())})
            assert reference is not None
            for q in sensitivity[-3:]:
                res=boundary_matching(np.asarray(json.loads(q['boundaries_seconds'])),reference,a.sensitivity_tolerance)
                q.update(res)
            print(f'[{ni}/{len(cases)}] {cid}: {len(boundary)-1} activities',flush=True)
        except Exception as ex:
            errors.append({'case_id':cid,'error':str(ex)})
            print('ERROR:',cid,ex,flush=True)
    if not improved:raise RuntimeError('No valid activity windows. See processing_errors.csv')
    table=pd.DataFrame(improved)
    table.to_csv(a.out/'v3_1_all_activity_intervals.csv',index=False)
    pd.DataFrame(audit).to_csv(a.out/'hp_boundary_adjustments.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(a.out/'gap_boundary_sensitivity.csv',index=False)
    pd.DataFrame(errors,columns=['case_id','error']).to_csv(a.out/'processing_errors.csv',index=False)
    case_count=table.case_id.nunique()
    types=table.activity_type.value_counts().rename_axis('activity_type').reset_index(name='n_activities')
    types.to_csv(a.out/'activity_type_inventory.csv',index=False)
    print(f'Available: {case_count} cases, {len(table)} activities. Selecting <= {a.activities} for DTW.',flush=True)

    cache=OnDemandCaseData(case_lookup,max_loaded=1)

    # Case-first sampling: preserve all processable cases BEFORE allocating extra
    # Activity slots; retry a different Activity from the same case when a signal
    # sequence cannot be built. The DTW algorithm itself is unchanged.
    def get_signature(row):
        raw,norm,cnt,mask=cache[str(row.case_id)]
        return segment_to_sequence(row,raw,norm,cnt,mask,a.state_codes,a.num_codes,a.max_bins)

    chosen,dtw_features,type_selection,rejected,coverage = choose_case_first(
        table, target_activities=a.activities, max_per_case=a.max_per_case,
        seed=a.seed, signature_fn=get_signature, type_order=TYPE_ORDER,
        require_all=not a.allow_case_shortfall
    )
    coverage.to_csv(a.out/'case_coverage.csv',index=False)
    type_selection.to_csv(a.out/'selected_type_counts.csv',index=False)
    rejected.to_csv(a.out/'rejected_activity_windows.csv',index=False)
    chosen.to_csv(a.out/'selected_activity_manifest.csv',index=False)

    n_covered=int((coverage.n_selected>0).sum())
    n_requested=len(cases)
    n_selected=len(chosen)
    counts=coverage.n_selected.to_numpy(int)
    coverage_report={
        'cases_requested_cli':int(a.cases),
        'cases_with_inputs':n_requested,
        'cases_with_refined_activity_candidates':int(case_count),
        'cases_represented_in_dtw':n_covered,
        'cases_missing_from_dtw':int(n_requested-n_covered),
        'activities_selected':n_selected,
        'activities_requested_max':int(a.activities),
        'max_per_case':int(a.max_per_case),
        'selection_policy':'one valid Activity per case first, then type-balanced extras',
        'seed':int(a.seed)
    }
    (a.out/'case_coverage_summary.json').write_text(
        json.dumps(coverage_report,indent=2,ensure_ascii=False),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    buckets=pd.Series(counts).value_counts().sort_index()
    ax.bar(buckets.index.astype(str),buckets.to_numpy())
    ax.set_xlabel('Selected activities per case')
    ax.set_ylabel('Number of cases')
    ax.set_title(f'Case-first selection: {n_covered}/{n_requested} cases')
    ax.grid(axis='y',alpha=.2)
    fig.savefig(a.out/'case_coverage.png',dpi=160);plt.close(fig)
    print('Case-first selection:',json.dumps(coverage_report,ensure_ascii=False),flush=True)
    if (n_covered!=n_requested or n_requested!=case_count) and not a.allow_case_shortfall:
        raise RuntimeError(
            f'Coverage shortfall ({n_covered}/{n_requested} cases): see case_coverage.csv and processing_errors.csv. '
            'No clustering was run because --allow-case-shortfall was not specified.')
    if n_selected<3:
        raise ValueError('Fewer than 3 activities have usable signal data')
    labels_table=chosen.reset_index(drop=True).copy()
    labels_table['dtw_index']=np.arange(len(labels_table))
    labels_table['numeric_observation_coverage']=[q['coverage'] for q in dtw_features]
    labels_table['event_count_from_matrix']=[q['event_count'] for q in dtw_features]
    D,pairs,centers,scale=pairwise(dtw_features,len(a.state_codes),len(a.num_codes),a.band)
    Z,labels,k_diag,k,quality=evaluate_clustering(D,a.max_k)
    labels_table['cluster']=labels
    labels_table.to_csv(a.out/'cluster_assignments.csv',index=False)
    np.savetxt(a.out/'dtw_distance_matrix.csv',D,delimiter=',',fmt='%.7f')
    pairs.to_csv(a.out/'pairwise_distance_components.csv',index=False)
    k_diag.to_csv(a.out/'cluster_k_diagnostics.csv',index=False)
    means=[]
    for cl,g in labels_table.groupby('cluster'):
        index=g.dtw_index.to_numpy(int)
        best=index[np.argmin(D[np.ix_(index,index)].sum(axis=1))]
        med=labels_table.iloc[best]
        means.append({'cluster':cl,'n_activities':len(g),'n_cases':g.case_id.nunique(),
                      'median_duration_s':g.duration_s.median(),'median_event_count':g.event_count_from_matrix.median(),
                      'median_numeric_coverage':g.numeric_observation_coverage.median(),
                      'medoid_case_id':med.case_id,'medoid_activity_id':med.activity_id,
                      'medoid_start_s':med.start_s,'medoid_end_s':med.end_s,
                      'dominant_activity_type':g.activity_type.value_counts().index[0]})
    (a.out/'representatives').mkdir(exist_ok=True)
    pd.DataFrame(means).to_csv(a.out/'cluster_summary.csv',index=False)
    # All DTW/cluster outputs have been saved; render in bounded-memory mode.
    # Never reload a whole case just to overlay members.
    del cache
    plot_saved_results(out=a.out,data_root=a.data_root,
                       numeric_codes=a.num_codes,state_codes=a.state_codes,
                       no_exact_events=a.no_exact_events,exact_max_points=a.exact_max_points,
                           mode=a.plot_mode,resume=not a.refresh_plots)

    # Distance bias diagnostics: are durations and event counts again dominating?
    audit_rows=[]
    for col in ('duration_difference_s','event_count_difference'):
        rho,pv=spearmanr(pairs.final_distance,pairs[col])
        audit_rows.append({'factor':col,'spearman_rho':rho,'p_value_naive_pairs_not_independent':pv,
                           'warning':abs(rho)>=.5})
    pd.DataFrame(audit_rows).to_csv(a.out/'distance_bias_diagnostics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,col in zip(axes,('duration_difference_s','event_count_difference')):
        ax.scatter(pairs[col],pairs.final_distance,s=5,alpha=.3)
        ax.set_xlabel(col);ax.set_ylabel('DTW combined distance');ax.grid(alpha=.15)
    fig.suptitle('Distance bias diagnostics (activity pairs not independent)')
    fig.savefig(a.out/'distance_bias_diagnostics.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for cid,grp in table.groupby('activity_type'):
        ax.hist(grp.duration_s,alpha=.5,bins=np.logspace(0,4,35),label=cid)
    ax.set_xscale('log');ax.set_xlabel('Activity duration (seconds)');ax.set_ylabel('Count');ax.legend(fontsize=8)
    fig.savefig(a.out/'activity_types_duration.png',dpi=160);plt.close(fig)
    sens=pd.DataFrame(sensitivity)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for fac,g in sens.groupby('gap_factor'):
        axs[0].boxplot(g.boundary_f1.dropna().to_numpy(),positions=[fac],widths=.25)
        axs[1].boxplot(g.median_shift_s.dropna().to_numpy(),positions=[fac],widths=.25)
    for ax in axs:ax.set_xticks([.5,1.,2.]);ax.set_xlabel('Episode gap multiplier');ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('Boundary F1 vs base (±tolerance)');axs[1].set_ylabel('Median timing difference (s)')
    fig.suptitle('Sensitivity: boundary location, not only activity count')
    fig.savefig(a.out/'gap_boundary_sensitivity.png',dpi=160);plt.close(fig)
    report={'cases_processed':case_count,'cases_represented_in_dtw':n_covered,
            'all_activities':len(table),'dtw_activities':len(labels_table),
            'clusters_k':k,'partition_warning':quality if labels_table.case_id.nunique()>1 else 'single_case_SMOKE_TEST_not_cross_case_validation','distance_weights':{'shape_dtw':.75,'raw_amplitude':.20,'duration':.05},
            'signal_group_weights':GROUP_W,'state_codes':a.state_codes,'numeric_codes':a.num_codes,
            'physical_feature_centers':centers.tolist(),'physical_feature_scales':scale.tolist(),
            'notice':'Exploratory sensor-level classes, not clinical treatment labels. Validation and blinded annotation still required.'}
    (a.out/'analysis_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('physical_feature_centers','physical_feature_scales')},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
