"""Chair-log hierarchical activity segmentation v3.

Level 1: raw timestamp state transitions -> operations -> same-instrument episodes.
Level 2: merge related operation episodes into blocks; select coarse activity
         boundaries from block edges and Change signal, avoiding operation cuts.
Compare the ORIGINAL State and Change methods to the NEW hierarchical Hybrid.
No DTW, no clinical labels or clinical-accuracy claims are produced here.

Read-only against existing chair_model_A_results; outputs separate files.
"""
from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import segmentation_core as seg

# The instrument associations are working hypotheses, not confirmed equipment specifications.
# The syringe code denotes pickup/return candidate, NOT actual air/water delivery.
STATE_SPECS = {
    '20039': {'name':'HP_work', 'values':{1:'HP1',2:'HP2'}, 'off_values':{0}},
    '20011': {'name':'HP_select', 'values':{1:'HP1_selected',2:'HP2_selected'}, 'off_values':{0}},
    '20101': {'name':'Syringe_pickup', 'values':{0:'Syringe_picked'}, 'off_values':{1}},
}
PRIMARY = ('HP1', 'HP2', 'Syringe_picked')
REQ = ['raw_1s', 'event_counts_1s', 'observation_mask_1s']


def parse_args():
    p = argparse.ArgumentParser(description='Hierarchical chair activity model A v3 (NO DTW)')
    p.add_argument('--data-root',type=Path,required=True,help='chair_model_A_results')
    p.add_argument('--activity-root',type=Path,default=None,
                   help='Optional chair_activity_v2_100: reuse selected_cases.csv')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cases',type=int,default=10,help='Max cases to analyze (0=all eligible)')
    p.add_argument('--plot-cases',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--min-duration',type=float,default=60.)
    p.add_argument('--max-duration',type=float,default=7200.)
    p.add_argument('--max-signals',type=int,default=14)
    p.add_argument('--state-min-step',type=float,default=10.)
    p.add_argument('--change-threshold',type=float,default=.45)
    p.add_argument('--smooth-seconds',type=float,default=3.)
    p.add_argument('--episode-gap',type=float,default=0,
                   help='0=estimate from HP operation gaps; else fixed seconds')
    p.add_argument('--episode-min-gap',type=float,default=.5)
    p.add_argument('--episode-max-gap',type=float,default=8.)
    p.add_argument('--episode-default-gap',type=float,default=2.)
    p.add_argument('--block-gap',type=float,default=12.,help='Merge operation episodes into blocks')
    p.add_argument('--idle-gap',type=float,default=15.,help='Give priority to block start/end after idle gap')
    p.add_argument('--min-activity',type=float,default=15.,help='Minimum coarse activity length')
    p.add_argument('--lock-tolerance',type=float,default=1.,help='Tolerance for boundary within operation')
    p.add_argument('--max-syringe-anchor',type=float,default=90.,
                   help='Long syringe pickups retained but not used to merge coarse blocks')
    p.add_argument('--max-plot-events',type=int,default=10000)
    a=p.parse_args()
    if a.cases<0 or a.plot_cases<0 or a.min_activity<=0 or a.block_gap<0:
        p.error('Invalid case/plot/activity options')
    return a


def case_id(path:Path)->str:
    return '__'.join(path.parts[-3:])


def case_table_records(a):
    root=a.data_root
    summary=pd.read_csv(root/'case_summary.csv',dtype={'seq':str,'date_folder':str,'chair_folder':str},low_memory=False)
    summary=summary[summary.status.astype(str).str.startswith('ok_')].copy()
    starts=pd.to_datetime(summary.start,errors='coerce');ends=pd.to_datetime(summary.end,errors='coerce')
    summary['duration_s']=(ends-starts).dt.total_seconds()
    summary=summary[summary.duration_s.between(a.min_duration,a.max_duration)]
    summary['case_dir']=summary.apply(lambda r:root/'cases'/str(r.date_folder)/str(r.chair_folder)/f'Chair_log_seq_{r.seq}',axis=1)
    summary=summary[summary.case_dir.map(lambda d:all((d/(s+'.csv.gz')).exists() for s in REQ))].copy()
    if a.activity_root:
        selected=a.activity_root/'selected_cases.csv'
        if not selected.exists():
            raise FileNotFoundError(f'--activity-root needs selected_cases.csv: {selected}')
        sample=pd.read_csv(selected,dtype={'seq':str,'date_folder':str,'chair_folder':str})
        keys=set(zip(sample.date_folder.astype(str),sample.chair_folder.astype(str),sample.seq.astype(str)))
        summary=summary[summary.apply(lambda r:(str(r.date_folder),str(r.chair_folder),str(r.seq)) in keys,axis=1)]
    if len(summary)==0:raise ValueError('No cases available. Check input root and selected_cases.csv')
    if a.cases and len(summary)>a.cases:
        # stratify by date and duration using stable random seed
        summary=summary.copy()
        summary['bucket']=pd.qcut(summary.duration_s,q=min(4,summary.duration_s.nunique()),duplicates='drop').astype(str)
        rng=np.random.default_rng(a.seed)
        groups=[]
        for _,g in summary.groupby(['bucket','date_folder'],sort=True):
            groups.append(list(rng.permutation(g.index.to_numpy())))
        chosen=[]
        while len(chosen)<a.cases and any(groups):
            for grp in groups:
                if grp and len(chosen)<a.cases:chosen.append(grp.pop())
        summary=summary.loc[chosen].copy()
    else:
        summary=summary.sort_values(['date_folder','chair_folder','seq'])
    return summary.drop(columns=['bucket'],errors='ignore')


def load_original_events(case:Path):
    f=case/'events_original.csv.gz'
    if not f.exists():return None
    d=pd.read_csv(f,compression='gzip',dtype={'event_code':str,'event_value':str},
                  usecols=['event_code','event_value','time_s'],low_memory=False)
    d['time_s']=pd.to_numeric(d.time_s,errors='coerce')
    d['value']=pd.to_numeric(d.event_value,errors='coerce')
    return d.dropna(subset=['time_s']).sort_values('time_s',kind='stable').reset_index(drop=True)


def from_resampled(case:Path,raw:pd.DataFrame,cnt:pd.DataFrame,mask:pd.DataFrame):
    """Fallback: binned times are approximate. DO NOT fake subsecond accuracy."""
    rows=[]
    for code in list(STATE_SPECS)+['20742']:
        col='E'+code
        if col not in raw:continue
        x=raw[col].to_numpy(float)
        if code in STATE_SPECS:
            # state-change from reconstructed sequence, including first known sample
            known=np.isfinite(x)&(mask[col].to_numpy(float)>0)
            ix=np.where(known & (np.r_[True,x[1:]!=x[:-1]]))[0]
        else:
            ix=np.where(np.isfinite(x) & (cnt[col].to_numpy(float)>0))[0]
        for i in ix:rows.append({'event_code':code,'time_s':float(raw.index[i]),'value':float(x[i])})
    return pd.DataFrame(rows,columns=['event_code','time_s','value']).sort_values('time_s',kind='stable')


def state_operations(d:pd.DataFrame, end:float):
    """Recover intervals from observed changes; no fabricated pre-first status.

    HP1/HP2 values are different mutually exclusive active labels; state change
    1->2 closes HP1 and opens HP2 at same timestamp. Duplicate identical readings
    are not operation transitions.
    """
    rows=[]
    for code,spec in STATE_SPECS.items():
        g=d[(d.event_code==code)&d.value.notna()].copy()
        if g.empty:continue
        # last reading at a given timestamp, stable duplicate handling
        g=g.drop_duplicates(subset=['time_s'],keep='last').sort_values('time_s',kind='stable')
        state=None; onset=None; left_censored=False
        last_observed=None; raw_count=0
        for t,v in zip(g.time_s.to_numpy(float),g.value.to_numpy(float)):
            if t < 0 or t > end+1:continue
            vv=int(v) if float(v).is_integer() else None
            new=spec['values'].get(vv,None)
            # Unknown value closes active (does not force 0) and marks observed ambiguity
            if new==state:
                last_observed=vv  # an observed OFF is evidence, not an unknown initial state
                continue
            if state is not None and onset is not None:
                rows.append({'source_code':code,'operation_type':state,'start_s':float(onset),
                             'end_s':float(t),'duration_s':max(0,float(t-onset)),
                             'left_censored':bool(left_censored),'right_censored':False,
                             'unknown_end_value': vv not in spec['off_values'] and new is None,
                             'source':'state_transition'})
            state=new
            onset=float(t) if new is not None else None
            left_censored=(last_observed is None and new is not None)
            last_observed=vv
        if state is not None and onset is not None:
            rows.append({'source_code':code,'operation_type':state,'start_s':float(onset),
                         'end_s':float(end),'duration_s':max(0,end-onset),
                         'left_censored':bool(left_censored),'right_censored':True,
                         'unknown_end_value':False,'source':'state_transition'})
    cols=['source_code','operation_type','start_s','end_s','duration_s','left_censored','right_censored','unknown_end_value','source']
    return pd.DataFrame(rows,columns=cols).sort_values(['start_s','end_s'],kind='stable').reset_index(drop=True)


def numeric_bursts(d:pd.DataFrame,end:float,cutoff=1000.,max_gap=1.):
    """RPM sampling bursts: corroborating evidence only, not verified HP operation."""
    g=d[(d.event_code=='20742')&d.value.notna()].copy()
    if g.empty:return pd.DataFrame(columns=['start_s','end_s','peak_raw','n_samples'])
    g=g.drop_duplicates(subset=['time_s'],keep='last')
    t=g.time_s.to_numpy(float);v=g.value.to_numpy(float)
    is_on=v>=cutoff
    rows=[];start=None;last=None;peak=0.;n=0
    for ti,vi,on in zip(t,v,is_on):
        if start is not None and (not on or (last is not None and ti-last>max_gap)):
            if n>=2:rows.append({'start_s':start,'end_s':last,'peak_raw':peak,'n_samples':n})
            start=None;last=None;peak=0.;n=0
        if on:
            if start is None:start=float(ti)
            last=float(ti);peak=max(peak,float(vi));n+=1
    if start is not None and n>=2:rows.append({'start_s':start,'end_s':last,'peak_raw':peak,'n_samples':n})
    return pd.DataFrame(rows,columns=['start_s','end_s','peak_raw','n_samples'])


def infer_gap(gaps,low,high,default):
    """Two-cluster valley in log gap, with explicit fallback for unimodal/low-N data."""
    x=np.asarray(gaps,dtype=float)
    x=x[np.isfinite(x)&(x>=0.05)&(x<=60)]
    if len(x)<25:return float(default),'fallback_few_gaps',len(x)
    log=np.log(x)
    p25,p75=np.quantile(log,[.25,.75])
    if p75-p25<np.log(2):return float(default),'fallback_unimodal',len(x)
    centers=np.array([p25,p75],dtype=float)
    for _ in range(40):
        nearest=np.argmin(abs(log[:,None]-centers),axis=1)
        if min(np.bincount(nearest,minlength=2))<max(5,.1*len(log)):
            return float(default),'fallback_unbalanced',len(x)
        new=np.array([np.mean(log[nearest==j]) for j in range(2)])
        if np.max(abs(new-centers))<1e-5:break
        centers=new
    centers.sort()
    if centers[1]-centers[0]<np.log(3):return float(default),'fallback_not_bimodal',len(x)
    value=float(np.exp(np.mean(centers)))
    return float(np.clip(value,low,high)),'estimated_log_two_group_midpoint',len(x)


def operation_episodes(operations:pd.DataFrame,gap_s:float):
    """Merge short stops WITHIN same instrument; keep underlying operations intact."""
    result=[]; mapping=[];eid=0
    p=operations[operations.operation_type.isin(PRIMARY)].copy()
    for typ,g in p.groupby('operation_type',sort=True):
        g=g.sort_values(['start_s','end_s'])
        curr=None
        for k,r in g.iterrows():
            if curr is not None and r.start_s-curr['end_s']<=gap_s:
                curr['end_s']=max(curr['end_s'],float(r.end_s))
                curr['n_operations']+=1
                curr['left_censored'] |= bool(r.left_censored)
                curr['right_censored'] |= bool(r.right_censored)
                curr['source_operation_rows'].append(int(k))
                mapping.append((int(k),eid))
            else:
                if curr is not None:result.append(curr)
                eid+=1
                curr={'episode_id':eid,'operation_type':typ,'start_s':float(r.start_s),
                      'end_s':float(r.end_s),'n_operations':1,
                      'left_censored':bool(r.left_censored),'right_censored':bool(r.right_censored),
                      'source_operation_rows':[int(k)]}
                mapping.append((int(k),eid))
        if curr is not None:result.append(curr)
    cols=['episode_id','operation_type','start_s','end_s','duration_s','n_operations',
          'left_censored','right_censored','source_operation_rows']
    episodes=pd.DataFrame(result)
    if episodes.empty:
        episodes=pd.DataFrame(columns=cols)
    else:
        episodes['duration_s']=episodes.end_s-episodes.start_s
        episodes=episodes[cols].sort_values('start_s',kind='stable').reset_index(drop=True)
        episodes['source_operation_rows']=episodes.source_operation_rows.map(lambda a:','.join(map(str,a)))
    return episodes,dict(mapping)


def coarse_blocks(eps:pd.DataFrame,merge_gap:float,max_syringe:float):
    """A block is an ordered set of potentially related instrument episodes.

    Long syringe pickups are excluded from *block merging* but are retained in
    episode files and activity overlap analysis. They cannot bridge 10+ minutes.
    """
    p=eps.copy()
    if p.empty:return pd.DataFrame(columns=['block_id','start_s','end_s','types','n_episodes'])
    p=p[~((p.operation_type=='Syringe_picked')&(p.duration_s>max_syringe))]
    if p.empty:return pd.DataFrame(columns=['block_id','start_s','end_s','types','n_episodes'])
    p=p.sort_values('start_s',kind='stable')
    result=[];cur=None;bid=0
    for r in p.itertuples():
        if cur is not None and float(r.start_s)-cur['end_s']<=merge_gap:
            cur['end_s']=max(cur['end_s'],float(r.end_s));cur['types'].add(r.operation_type)
            cur['n_episodes']+=1
        else:
            if cur is not None:result.append(cur)
            bid+=1;cur={'block_id':bid,'start_s':float(r.start_s),'end_s':float(r.end_s),
                         'types':{r.operation_type},'n_episodes':1}
    if cur is not None:result.append(cur)
    for r in result:r['types']=','.join(sorted(r['types']))
    return pd.DataFrame(result)


def chosen_hybrid_times(change_times:np.ndarray,change_score:np.ndarray,grid:np.ndarray,
                        blocks:pd.DataFrame, episodes:pd.DataFrame, duration:float, a):
    """Coarse segmentation with priorities, not every individual sensor edge.

    Prioritize transitions between blocks, then strong change peaks. Avoid
    cutting through an HP working operation (not the whole episode), so a
    short HP pause can be part of a larger Activity. No fixed 180s split.
    """
    candidates=[]
    for t in change_times:
        k=int(np.argmin(abs(grid-t)))
        strength=float(change_score[k]) if len(change_score) else 0.
        candidates.append((float(t),1.0+min(strength,2.)*.3,'change'))
    if len(blocks):
        b=blocks.sort_values('start_s').reset_index(drop=True)
        for i,r in b.iterrows():
            prev_end=float(b.iloc[i-1].end_s) if i else 0.
            next_start=float(b.iloc[i+1].start_s) if i+1<len(b) else duration
            gap_pre=max(0.,float(r.start_s)-prev_end)
            gap_post=max(0.,next_start-float(r.end_s))
            if gap_pre>=a.idle_gap:
                candidates.append((float(r.start_s),2.7,'block_start'))
            if gap_post>=a.idle_gap:
                candidates.append((float(r.end_s),2.7,'block_end'))
            if i>0 and b.iloc[i-1].types!=r.types and 0<=gap_pre<a.idle_gap:
                candidates.append((float(r.start_s),2.2,'block_type_switch'))

    # Protect genuine HP work pulses; do not mask all long syringe pickups.
    locked=episodes[episodes.operation_type.isin(['HP1','HP2'])] if len(episodes) else episodes
    def cuts_pulse(t):
        return len(locked)>0 and bool(((locked.start_s+a.lock_tolerance<t)&
                                      (locked.end_s-a.lock_tolerance>t)).any())
    candidates=[x for x in candidates if a.min_activity<=x[0]<=duration-a.min_activity and not cuts_pulse(x[0])]
    # Highest-priority candidates win within min-activity separation.
    chosen=[]
    for t,w,kind in sorted(candidates,key=lambda x:(-x[1],x[0])):
        if all(abs(t-v[0])>=a.min_activity for v in chosen):
            chosen.append((t,w,kind))
    chosen.sort(key=lambda x:x[0])
    return np.array([x[0] for x in chosen],float),chosen


def intervals(boundaries,duration,method,case,seq):
    t=np.r_[0,np.sort(np.asarray(boundaries,float)),duration]
    t=t[np.r_[True,np.diff(t)>1e-8]]
    rows=[]
    for i,(x,y) in enumerate(zip(t[:-1],t[1:]),1):
        rows.append({'case_id':case,'seq':seq,'method':method,'activity_id':i,
                     'start_s':round(float(x),3),'end_s':round(float(y),3),
                     'duration_s':round(float(y-x),3)})
    return rows


def split_count(episodes, boundaries):
    b=np.asarray(boundaries,float)
    if len(episodes)==0 or len(b)==0:return 0
    num=0
    for r in episodes.itertuples():
        if r.operation_type in ('HP1','HP2') and r.duration_s>0:
            num+=int(np.sum((b>r.start_s+1e-5)&(b<r.end_s-1e-5)))
    return num


def activity_labels(rows, episodes, cnt):
    """Heuristic composition labels, not trained treatment class labels."""
    for r in rows:
        within=episodes[(episodes.end_s>r['start_s'])&(episodes.start_s<r['end_s'])]
        types=set(within.operation_type)
        hp=types.intersection({'HP1','HP2'})
        syringe='Syringe_picked' in types
        if len(hp)>1 or (hp and syringe):label='mixed'
        elif hp:label=next(iter(hp))
        elif syringe:label='syringe_sensor'
        else:label='unanchored'
        # independent of operation count, only for descriptive interpretation
        time=cnt.index.to_numpy(float)
        p=(time>=r['start_s'])&(time<r['end_s'])
        event_n=float(cnt.iloc[np.where(p)[0]].sum().sum()) if p.any() else 0.
        r['composition_label']=label
        r['overlapping_episodes']=len(within)
        r['event_n']=int(event_n)
        r['event_density']=round(event_n/max(r['duration_s'],1e-10),4)
    return rows


def plot_case(dest, raw,cnt,events,operations,episodes,blocks, boundaries, duration,case_name):
    fig,axs=plt.subplots(6,1,figsize=(16,11),sharex=True,layout='constrained',
                        gridspec_kw={'height_ratios':[2,1,1.5,.6,.6,.6]})
    if events is not None:
        g=events[(events.event_code=='20742')&(events.value.notna())]
        x=g.time_s.to_numpy(float);y=g.value.to_numpy(float)
        if len(x)>12000:
            ix=np.linspace(0,len(x)-1,12000,dtype=int);x=x[ix];y=y[ix]
        axs[0].scatter(x,y,s=.65,color='#dd751c',label='20742 raw observations')
    elif 'E20742' in raw:
        y=raw['E20742'].to_numpy(float);t=raw.index.to_numpy(float)
        good=np.isfinite(y)
        axs[0].scatter(t[good],y[good],s=1,color='#dd751c',label='20742 binned values')
    axs[0].legend(fontsize=8,loc='upper right')
    axs[0].set_ylabel('RPM candidate\nraw value')
    axs[0].set_title(case_name+' | Level 1 operations → Level 2 activities')
    cs=cnt.drop(columns=['E90012'],errors='ignore').sum(axis=1)
    axs[1].plot(cnt.index,cs.values,lw=.6,color='#445873')
    axs[1].set_ylabel('All-code\nevents / bin')
    palette={'HP1':'#2d79b8','HP2':'#df780b','Syringe_picked':'#6d66a6',
             'HP1_selected':'#99cbe0','HP2_selected':'#e7b57a'}
    types=['HP1','HP2','Syringe_picked','HP1_selected','HP2_selected']
    for idx,typ in enumerate(types):
        for r in operations[operations.operation_type==typ].itertuples():
            axs[2].plot([r.start_s,r.end_s],[idx,idx],lw=5,color=palette[typ],solid_capstyle='butt')
    axs[2].set_yticks(range(len(types)));axs[2].set_yticklabels(types,fontsize=8)
    axs[2].set_ylabel('Level 1')
    for r in blocks.itertuples():
        axs[2].axvspan(r.start_s,r.end_s,alpha=.09,color='tab:green')
    for ai,method in enumerate(['state','change','hierarchical_hybrid'],3):
        e=np.r_[0,boundaries[method],duration]
        for j,(l,r) in enumerate(zip(e[:-1],e[1:])):
            axs[ai].axvspan(l,r,alpha=.19,color=('#3a76ba' if j%2==0 else '#d6a456'))
        for x in boundaries[method]:axs[ai].axvline(x,color='#30333a',lw=.55,alpha=.6)
        axs[ai].set_ylabel(f'{method}\n{len(e)-1} parts',fontsize=8)
        axs[ai].set_yticks([])
    for ax in axs:ax.grid(axis='x',alpha=.2)
    axs[-1].set_xlim(0,duration);axs[-1].set_xlabel('Seconds from first 90012 = 1')
    fig.savefig(dest/'hierarchy_overlay.png',dpi=160)
    plt.close(fig)


def global_plots(metrics,episodes,gaps,gap,reason,out):
    if len(metrics):
        fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        methods=['state','change','hierarchical_hybrid']
        for ax,(col,label) in zip(axs.flat,[('n_activities','Activity count / case'),
                                            ('median_duration_s','Median Activity duration (s)'),
                                            ('hp_episode_splits','HP episode cuts / case'),
                                            ('unanchored_fraction','Fraction of unanchored activities')]):
            arr=[metrics[metrics.method==m][col].dropna().to_numpy(float) for m in methods]
            ax.boxplot(arr,showfliers=False)
            ax.set_xticks([1,2,3],['State','Change','Hybrid'])
            ax.set_title(label);ax.grid(axis='y',alpha=.2)
        fig.suptitle('Equal-case distribution (NOT a clinical correctness score)')
        fig.savefig(out/'method_comparison_boxplots.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    x=np.asarray(gaps,float);x=x[np.isfinite(x)&(x>=.05)&(x<=60)]
    if len(x):ax.hist(x,bins=np.logspace(np.log10(.05),np.log10(60),40),alpha=.7)
    ax.axvline(gap,color='tab:red',lw=2,label=f'Episode gap = {gap:.2f} s ({reason})')
    ax.set_xscale('log');ax.set_xlabel('Gap between successive same-HP operations (s)')
    ax.set_ylabel('Frequency');ax.legend(fontsize=9)
    ax.set_title('HP pause distribution / threshold diagnostic')
    fig.savefig(out/'episode_gap_distribution.png',dpi=160);plt.close(fig)
    if len(episodes):
        fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
        for typ,g in episodes.groupby('operation_type'):
            vals=g.duration_s.to_numpy(float)
            ax.hist(vals,bins=np.logspace(-2,4,50),alpha=.5,label=typ)
        ax.set_xscale('log');ax.set_xlabel('Episode duration (s)');ax.set_ylabel('Count')
        ax.legend();ax.set_title('Operation episode durations (raw seconds)')
        fig.savefig(out/'episode_duration_distribution.png',dpi=160);plt.close(fig)


def main():
    a=parse_args();a.out.mkdir(parents=True,exist_ok=True)
    sample=case_table_records(a)
    sample.drop(columns=['case_dir']).to_csv(a.out/'selected_cases.csv',index=False,encoding='utf-8-sig')
    # Pass 1: collect ONLY compact Level-1 operations and gap samples.
    # Do not keep 100 wide 259-column matrices in memory.
    bundles=[];all_gaps=[];errors=[];raw_sources=[]
    for i,r in enumerate(sample.itertuples(),1):
        case=Path(r.case_dir);cid=case_id(case)
        try:
            meta_file=case/'meta.json'
            if meta_file.exists():
                duration=float(json.loads(meta_file.read_text(encoding='utf-8'))['duration_s'])
            else:
                duration=float(r.duration_s)
            d=load_original_events(case)
            source='exact_original_events' if d is not None else 'binned_approximation'
            if d is None:
                raw=seg.load_table(case,'raw_1s');cnt=seg.load_table(case,'event_counts_1s')
                mask=seg.load_table(case,'observation_mask_1s');raw,cnt,mask=seg.align(raw,cnt,mask)
                d=from_resampled(case,raw,cnt,mask)
                del raw,cnt,mask
            ops=state_operations(d,duration)
            hp=ops[ops.operation_type.isin(['HP1','HP2'])]
            for _,g in hp.groupby('operation_type'):
                g=g.sort_values('start_s')
                gaps=np.maximum(0,g.start_s.to_numpy(float)[1:]-g.end_s.to_numpy(float)[:-1])
                all_gaps.extend(gaps.tolist())
            bundles.append({'case':case,'case_id':cid,'seq':str(r.seq),'ops':ops,
                            'duration':duration,'source':source})
            raw_sources.append({'case_id':cid,'input_source':source})
            print(f'Pass 1 [{i}/{len(sample)}] {cid}: {len(ops)} operations / {source}',flush=True)
            del d
        except Exception as ex:
            errors.append({'case_id':cid,'case_dir':str(case),'error':repr(ex)})
            print(f'Pass 1 [{i}/{len(sample)}] ERROR {cid}: {ex}',flush=True)
    gap,reason,n_gaps=infer_gap(all_gaps,a.episode_min_gap,a.episode_max_gap,a.episode_default_gap)
    if a.episode_gap>0:gap=float(a.episode_gap);reason='user_fixed'
    print(f'HP episode merge threshold: {gap:.2f}s ({reason}, n_gaps={n_gaps})',flush=True)

    ops_all=[];eps_all=[];blocks_all=[];rows_all=[];metrics=[];source_rows=[];boundary_rows=[];sensitivity=[]
    for i,rec in enumerate(bundles,1):
        cid=rec['case_id'];seq=rec['seq'];duration=rec['duration'];case=rec['case']
        try:
            raw=seg.load_table(case,'raw_1s');cnt=seg.load_table(case,'event_counts_1s')
            mask=seg.load_table(case,'observation_mask_1s');raw,cnt,mask=seg.align(raw,cnt,mask)
            grid=raw.index.to_numpy(float)
            if len(grid)<5:raise ValueError('Too few time points')
            step=float(np.median(np.diff(grid)))
            selected_signals=seg.select_signals(raw,cnt,mask,a.max_signals)
            state_e,state_score=seg.state_edges(raw,cnt,mask,selected_signals,a.state_min_step,step)
            change_e,change_score=seg.change_edges(raw,cnt,mask,selected_signals,a.state_min_step,
                                                   a.change_threshold,step,a.smooth_seconds)
            bounds={'state':grid[state_e[1:-1]],'change':grid[change_e[1:-1]]}
            bounds={method:ts[(ts>0)&(ts<duration)] for method,ts in bounds.items()}

            ops=rec['ops'].copy();eps,mapping=operation_episodes(ops,gap)
            blocks=coarse_blocks(eps,a.block_gap,a.max_syringe_anchor)
            hyb,selected=chosen_hybrid_times(bounds['change'],change_score,grid,blocks,eps,duration,a)
            bounds['hierarchical_hybrid']=hyb
            # Stress-test the inferred pause threshold; retaining this table is
            # more informative than claiming the single selected value is exact.
            for mult in (0.5, 1.0, 2.0):
                alt_gap=gap*mult
                alt_eps,_=operation_episodes(ops,alt_gap)
                alt_blocks=coarse_blocks(alt_eps,a.block_gap,a.max_syringe_anchor)
                alt_hyb,_=chosen_hybrid_times(bounds['change'],change_score,grid,
                                               alt_blocks,alt_eps,duration,a)
                sensitivity.append({'case_id':cid,'gap_multiplier':mult,'gap_seconds':alt_gap,
                                    'n_episodes':len(alt_eps),'n_blocks':len(alt_blocks),
                                    'n_activities':len(alt_hyb)+1})
            dest=a.out/'cases'/cid;dest.mkdir(parents=True,exist_ok=True)
            ops['operation_id']=np.arange(1,len(ops)+1)
            ops['episode_id']=ops.index.map(mapping)
            ops['case_id']=cid;eps['case_id']=cid;blocks['case_id']=cid
            ops.to_csv(dest/'level1_operations.csv',index=False)
            eps.to_csv(dest/'level1_episodes.csv',index=False)
            blocks.to_csv(dest/'level2_blocks.csv',index=False)
            ops_all.append(ops);eps_all.append(eps);blocks_all.append(blocks)
            for t,w,source in selected:boundary_rows.append({'case_id':cid,'time_s':t,'priority':w,'reason':source})
            for method,b in bounds.items():
                rr=activity_labels(intervals(b,duration,method,cid,seq),eps,cnt)
                rows_all.extend(rr)
                dur=[x['duration_s'] for x in rr]
                metrics.append({'case_id':cid,'seq':seq,'method':method,'duration_s':duration,
                                'n_activities':len(rr),'median_duration_s':float(np.median(dur)),
                                'short_under_5s':int(sum(x<5 for x in dur)),
                                'hp_episode_splits':split_count(eps,b),
                                'unanchored_fraction':float(np.mean([x['composition_label']=='unanchored' for x in rr])),
                                'median_event_density':float(np.median([x['event_density'] for x in rr])),
                                'n_hp_episodes':int(eps.operation_type.isin(['HP1','HP2']).sum()),
                                'n_syringe_episodes':int((eps.operation_type=='Syringe_picked').sum()),
                                'data_precision':rec['source']})
            if i<=a.plot_cases:
                d=load_original_events(case) if rec['source']=='exact_original_events' else None
                plot_case(dest,raw,cnt,d,ops,eps,blocks,bounds,duration,cid)
            print(f'Pass 2 [{i}/{len(bundles)}] {cid}: '
                  +', '.join(f'{m}={len(b)+1}' for m,b in bounds.items()),flush=True)
            del raw,cnt,mask
        except Exception as ex:
            errors.append({'case_id':cid,'case_dir':str(case),'error':repr(ex)})
            print(f'Pass 2 [{i}/{len(bundles)}] ERROR {cid}: {ex}',flush=True)
    def concat(parts,cols):
        return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=cols)
    concat(ops_all,['case_id','operation_type']).to_csv(a.out/'level1_operations.csv',index=False,encoding='utf-8-sig')
    concat(eps_all,['case_id','episode_id','operation_type']).to_csv(a.out/'level1_episodes.csv',index=False,encoding='utf-8-sig')
    concat(blocks_all,['case_id','block_id']).to_csv(a.out/'level2_blocks.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rows_all).to_csv(a.out/'activity_intervals_all_methods.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([r for r in rows_all if r['method']=='hierarchical_hybrid']).to_csv(
        a.out/'hierarchical_hybrid_activity_intervals.csv',index=False,encoding='utf-8-sig')
    md=pd.DataFrame(metrics)
    md.to_csv(a.out/'case_method_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(boundary_rows,columns=['case_id','time_s','priority','reason']).to_csv(
        a.out/'hybrid_boundary_reasons.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(raw_sources).to_csv(a.out/'input_precision_audit.csv',index=False)
    pd.DataFrame(sensitivity,columns=['case_id','gap_multiplier','gap_seconds','n_episodes','n_blocks','n_activities']) \
        .to_csv(a.out/'episode_gap_sensitivity.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(errors,columns=['case_id','case_dir','error']).to_csv(a.out/'processing_errors.csv',index=False)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    cfg.update({'estimated_episode_gap_s':gap,'gap_reason':reason,'n_hp_gaps':n_gaps,
                'n_success':int(md[md.method=='hierarchical_hybrid'].shape[0]),'n_fail':len(errors),
                'evidence_notice':'No human clinical ground truth; composition labels are sensor heuristics. '
                                  'HP and syringe mappings require validation.'})
    (a.out/'analysis_config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    global_plots(md,concat(eps_all,['operation_type','duration_s']),all_gaps,gap,reason,a.out)
    if sensitivity:
        sens=pd.DataFrame(sensitivity)
        fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,(column,label) in zip(axs,[('n_episodes','Operation episode count'),
                                        ('n_activities','Hybrid activity count')]):
            x=[sens[sens.gap_multiplier==mul][column].to_numpy(float) for mul in (.5,1.,2.)]
            ax.boxplot(x,showfliers=False)
            ax.set_xticks([1,2,3],['0.5 ×','1 ×','2 ×'])
            ax.grid(axis='y',alpha=.2);ax.set_title(label)
            ax.set_xlabel('HP pause threshold multiplier')
        fig.suptitle('Gap threshold sensitivity across cases')
        fig.savefig(a.out/'episode_gap_sensitivity.png',dpi=160);plt.close(fig)
    print(f'Complete: {int(md[md.method=="hierarchical_hybrid"].shape[0])}/{len(sample)} cases. Output: {a.out}',flush=True)

if __name__=='__main__':main()
