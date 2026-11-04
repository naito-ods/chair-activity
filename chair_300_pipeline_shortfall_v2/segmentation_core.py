"""Activity segmentation diagnostic v2 (NO clustering / DTW).
Read existing chair_batch output, compare 3 segmentation methods, plot each case.
Python 3.10+. Does not modify original files.
"""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ('gap', 'state', 'change')


def load_table(case_dir, prefix):
    path = case_dir / (prefix + '.csv.gz')
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, compression='gzip', low_memory=False)
    if 'time_s' not in df.columns:
        raise ValueError(f'time_s not found: {path}')
    df = df.set_index('time_s')
    df.index = pd.to_numeric(df.index, errors='coerce')
    df = df.loc[df.index.notna()].sort_index()
    return df.apply(pd.to_numeric, errors='coerce')


def align(raw, cnt, mask):
    cols = sorted(set(raw.columns) | set(cnt.columns) | set(mask.columns))
    times = raw.index
    if len(times) < 5:
        raise ValueError('Too few time bins')
    raw = raw.reindex(index=times, columns=cols)
    cnt = cnt.reindex(index=times, columns=cols).fillna(0)
    mask = mask.reindex(index=times, columns=cols).fillna(0)
    return raw, cnt, mask


def select_signals(raw, cnt, mask, max_signals=14):
    # Use case-level variability and coverage, not only event frequency.
    # Discard 90012 (boundary signal), nearly constant channels, and scant observations.
    choices = []
    for c in raw.columns:
        if c == 'E90012': continue
        y = raw[c].to_numpy(float)
        good = np.isfinite(y) & (mask[c].to_numpy(float) > 0)
        v = y[good]
        if len(v) < 3 or np.unique(v).size < 2: continue
        event_bins = int((cnt[c].to_numpy(float) > 0).sum())
        unique = np.unique(v).size
        binary = np.isin(v, [0,1]).all()
        changes = int(np.count_nonzero(np.diff(v) != 0))
        score = np.log1p(event_bins) * np.log1p(changes) / max(np.sqrt(len(v)), 1)
        if binary: score *= 1.5
        choices.append((score, c, binary, event_bins, changes, unique))
    choices.sort(reverse=True)
    return choices[:max_signals]


def peak_thin(peaks, strength, min_separation_bins):
    order = sorted(set(int(p) for p in peaks if p > 0 and p < len(strength)), key=lambda k: -strength[k])
    keep=[]
    for k in order:
        if all(abs(k-j)>=min_separation_bins for j in keep):keep.append(k)
    return sorted(keep)


def gap_edges(cnt, gap_s, step):
    # Historical baseline: split at sustained periods with zero recorded events.
    activity = (cnt.sum(axis=1).to_numpy(float) > 0)
    n = len(activity); edge=[0]
    pos = np.where(activity)[0]
    if len(pos) > 1:
        for a,b in zip(pos[:-1],pos[1:]):
            if (b-a-1)*step >= gap_s:
                edge.append(int((a+b)//2))
    return sorted(set(edge+[n]))


def state_edges(raw, cnt, mask, signals, min_step_s, step):
    n = len(raw); score=np.zeros(n)
    for _,c,is_binary,_,_,uniq in signals:
        if not is_binary and uniq>8: continue
        y=raw[c].to_numpy(float); observed=mask[c].to_numpy(float)>0
        same_known=observed[1:] & observed[:-1] & np.isfinite(y[1:]) & np.isfinite(y[:-1])
        diff=same_known & (y[1:] != y[:-1])
        score[1:]+=diff.astype(float)
    if score.max()<=0: return [0,n],score
    candidates=np.where(score>0)[0]
    edges=peak_thin(candidates,score,max(1,int(round(min_step_s/step))))
    return [0]+edges+[n],score


def change_edges(raw, cnt, mask, signals, min_step_s, threshold, step, smooth_s):
    # Robust per-channel first differences with explicit missingness;
    # event counts enter as log1p(rate), preventing high-frequency dominance.
    n=len(raw); score=np.zeros(n); channels=0
    half=max(1,int(round(smooth_s/step)))
    for _,c,is_binary,_,_,_ in signals:
        y=raw[c].to_numpy(float); known=np.isfinite(y)&(mask[c].to_numpy(float)>0)
        if known.sum() < 3: continue
        if is_binary: continue  # changes in state handled separately by state method
        med=np.median(y[known]); q1,q3=np.percentile(y[known],[25,75]);scale=max(q3-q1, np.std(y[known])*.15, 1e-8)
        x=np.where(known,(y-med)/scale,0)
        x=np.clip(x,-8,8)
        # Smooth masked values without interpolating across long gaps
        kernel=np.ones(2*half+1)
        numerator=np.convolve(x, kernel,mode='same')
        denom=np.convolve(known.astype(float),kernel,mode='same')
        smooth=np.divide(numerator,denom,out=np.zeros(n),where=denom>0)
        d=np.abs(np.diff(smooth,prepend=smooth[0]))
        valid=(denom>0) & np.r_[False,denom[:-1]>0]
        score += np.where(valid,np.clip(d,0,4),0)
        channels+=1
    # Recorded event intensity changes, standardized per channel
    for _,c,_,_,_,_ in signals:
        x=np.log1p(cnt[c].to_numpy(float))
        if np.std(x)<1e-9: continue
        xx=pd.Series(x).rolling(2*half+1,min_periods=1,center=True).mean().to_numpy()
        score+=np.clip(np.abs(np.diff(xx,prepend=xx[0]))/(np.std(x)+1e-8),0,3)*.35
        channels+=1
    if channels: score/=np.sqrt(channels)
    # local maxima, robust threshold above the typical background
    positive=score[score>0]
    if not len(positive): return [0,n],score
    limit=max(float(threshold),float(np.median(positive)+1.5*np.median(np.abs(positive-np.median(positive)))))
    prev=np.r_[score[0],score[:-1]]; nxt=np.r_[score[1:],score[-1]]
    candidates=np.where((score>=limit)&(score>=prev)&(score>=nxt))[0]
    edges=peak_thin(candidates,score,max(1,int(round(min_step_s/step))))
    return [0]+edges+[n],score


def summarize_edges(edges,time,n,method,seq,case_dir):
    rows=[]
    dt=np.median(np.diff(time))
    for j,(a,b) in enumerate(zip(edges[:-1],edges[1:]),1):
        start=float(time[a]);end=float(time[b]) if b<n else float(time[-1]+dt)
        rows.append({'seq':seq,'case_dir':str(case_dir),'method':method,'activity_id':j,
                     'start_s':round(start,3),'end_s':round(end,3),'duration_s':round(end-start,3),
                     'n_time_bins':int(b-a)})
    return rows


def compare_edges(a,b,time,tolerance=5.):
    aa=np.array([time[k] for k in a[1:-1]])
    bb=np.array([time[k] for k in b[1:-1]])
    if not len(aa) or not len(bb): return np.nan
    return float(np.mean([np.min(np.abs(bb-t)) <= tolerance for t in aa]))


def plot_case(out,raw,cnt,mask,selected,edges,score,case_id):
    t=raw.index.to_numpy(float)
    fig,axs=plt.subplots(5,1,figsize=(17,11),sharex=True,gridspec_kw={'height_ratios':[2.7,1,1,1,1]},layout='constrained')
    numeric=[r for r in selected if not r[2]][:6]
    for _,col,*_ in numeric:
        v=raw[col].to_numpy(float);finite=np.isfinite(v)
        if finite.sum()<3:continue
        q5,q95=np.nanpercentile(v,[5,95]);den=max(q95-q5,1e-9)
        axs[0].plot(t[finite],((v-q5)/den)[finite],lw=.7,alpha=.75,label=col[1:])
    axs[0].set_ylabel('Numeric signals\n5–95% scaled')
    if numeric:axs[0].legend(ncol=6,fontsize=8,loc='upper right')
    axs[0].set_title(f'{case_id} | Activity boundary comparison — diagnostic, not clinical labels')
    for k,method in enumerate(METHODS,1):
        e=edges[method]
        for j,(a,b) in enumerate(zip(e[:-1],e[1:])):
            left=t[a]; right=t[b] if b<len(t) else t[-1]+np.median(np.diff(t))
            axs[k].axvspan(left,right,alpha=.25,color=('tab:blue' if j%2==0 else 'tab:orange'))
        for s in e[1:-1]:axs[k].axvline(t[s],lw=.65,color='black',alpha=.65)
        axs[k].set_ylabel(f'{method}\n{len(e)-1} sections')
        axs[k].set_yticks([])
    axs[4].plot(t,score['change'],lw=1,color='tab:purple',label='change score')
    axs[4].plot(t,score['state'],lw=.8,color='tab:green',alpha=.6,label='state transitions')
    axs[4].legend(fontsize=8,loc='upper right');axs[4].set_ylabel('Boundary scores')
    for ax in axs:ax.grid(axis='x',alpha=.2)
    axs[-1].set_xlabel('Seconds from treatment start (90012)')
    fig.savefig(out/'activity_comparison.png',dpi=160);plt.close(fig)


def sample_cases(summary,root,n,seed,min_s,max_s):
    df=summary[summary.status.astype(str).str.startswith('ok_')].copy()
    df['start_dt']=pd.to_datetime(df['start'],errors='coerce');df['end_dt']=pd.to_datetime(df['end'],errors='coerce')
    df['duration_s']=(df.end_dt-df.start_dt).dt.total_seconds()
    df=df[(df.duration_s>=min_s)&(df.duration_s<=max_s)].copy()
    df['dir']=df.apply(lambda r: root/'cases'/str(r.date_folder)/str(r.chair_folder)/f'Chair_log_seq_{r.seq}',axis=1)
    df=df[df['dir'].map(lambda d:(d/'raw_1s.csv.gz').exists() and (d/'event_counts_1s.csv.gz').exists() and (d/'observation_mask_1s.csv.gz').exists())]
    if df.empty:raise RuntimeError('No eligible case folders found. Check --root and --summary.')
    # Avoid choosing only one date / duration: balanced buckets, random and reproducible.
    df['bucket']=pd.qcut(df.duration_s, q=min(4,df.duration_s.nunique()),duplicates='drop').astype(str)
    df['day']=df.date_folder.astype(str)
    rng=np.random.default_rng(seed)
    groups=[]
    for _,g in df.groupby(['bucket','day'],sort=True):
        indices=rng.permutation(g.index.to_numpy());groups.append(list(indices))
    chosen=[]
    while len(chosen)<n and any(groups):
        for g in groups:
            if g and len(chosen)<n:chosen.append(g.pop())
    return df.loc[chosen].copy()


def main():
    ap=argparse.ArgumentParser(description='Compare three activity-segmentation approaches (NO DTW)')
    ap.add_argument('--root',type=Path,required=True,help='chair_model_A_results folder')
    ap.add_argument('--summary',type=Path,default=None)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cases',type=int,default=10)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--min-duration',type=float,default=60)
    ap.add_argument('--max-duration',type=float,default=7200)
    ap.add_argument('--max-signals',type=int,default=14)
    ap.add_argument('--gap-seconds',type=float,default=12)
    ap.add_argument('--min-boundary-seconds',type=float,default=10)
    ap.add_argument('--change-threshold',type=float,default=.45)
    ap.add_argument('--smooth-seconds',type=float,default=3)
    a=ap.parse_args()
    if a.cases<1 or a.max_signals<1:ap.error('--cases and --max-signals must be positive')
    root=a.root; summary=a.summary or root/'case_summary.csv';out=a.out
    if not summary.exists():ap.error(f'Missing summary: {summary}')
    out.mkdir(parents=True,exist_ok=True)
    selected=sample_cases(pd.read_csv(summary,dtype={'seq':str}),root,a.cases,a.seed,a.min_duration,a.max_duration)
    selected.drop(columns=['dir','bucket','day'],errors='ignore').to_csv(out/'selected_cases.csv',index=False,encoding='utf-8-sig')
    all_rows=[];details=[];boundary_stats=[];error_rows=[]
    for i,(_,record) in enumerate(selected.iterrows(),1):
        case=record['dir'];seq=str(record.seq)
        print(f'[{i}/{len(selected)}] seq {seq}',flush=True)
        try:
            raw=load_table(case,'raw_1s');cnt=load_table(case,'event_counts_1s');mask=load_table(case,'observation_mask_1s')
            raw,cnt,mask=align(raw,cnt,mask)
            t=raw.index.to_numpy(float);step=float(np.median(np.diff(t)))
            if len(t)>20000:raise ValueError('Too many time bins (>20000); select shorter case or increase input resampling interval')
            signals=select_signals(raw,cnt,mask,a.max_signals)
            E1=gap_edges(cnt,a.gap_seconds,step)
            E2,s2=state_edges(raw,cnt,mask,signals,a.min_boundary_seconds,step)
            E3,s3=change_edges(raw,cnt,mask,signals,a.min_boundary_seconds,a.change_threshold,step,a.smooth_seconds)
            edges={'gap':E1,'state':E2,'change':E3};scores={'state':s2,'change':s3}
            dest=out/'cases'/f'{record.date_folder}_{record.chair_folder}_{seq}';dest.mkdir(parents=True,exist_ok=True)
            for method,ed in edges.items():
                rr=summarize_edges(ed,t,len(t),method,seq,case)
                all_rows+=rr
                dur=[z['duration_s'] for z in rr]
                details.append({'seq':seq,'method':method,'n_activities':len(rr),
                                'median_duration_s':float(np.median(dur)),'min_duration_s':float(np.min(dur)),
                                'max_duration_s':float(np.max(dur)), 'duration_s':float(t[-1]+step)})
            boundary_stats.append({'seq':seq,'state_to_change_recall_5s':compare_edges(E2,E3,t),
                                   'change_to_state_recall_5s':compare_edges(E3,E2,t),
                                   'gap_to_change_recall_5s':compare_edges(E1,E3,t)})
            pd.DataFrame([{'code':r[1][1:],'selection_score':r[0], 'binary':r[2],
                           'n_event_bins':r[3], 'n_value_changes':r[4], 'n_unique_values':r[5]}
                          for r in signals]).to_csv(dest/'selected_signal_diagnostics.csv',index=False)
            pd.DataFrame({'time_s':t,'state_change_score':s2,'multivariate_change_score':s3}).to_csv(dest/'boundary_scores.csv',index=False)
            plot_case(dest,raw,cnt,mask,signals,edges,scores,f'{record.date_folder} / {record.chair_folder} / seq {seq}')
        except Exception as exc:
            error_rows.append({'seq':seq,'case_dir':str(case),'error':repr(exc)})
            print(' WARNING:',exc,flush=True)
    pd.DataFrame(all_rows).to_csv(out/'activity_intervals.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(details).to_csv(out/'activity_method_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(boundary_stats).to_csv(out/'boundary_agreement.csv',index=False)
    pd.DataFrame(error_rows,columns=['seq','case_dir','error']).to_csv(out/'processing_errors.csv',index=False)
    (out/'config.json').write_text(json.dumps(vars(a),ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(f'Finished: {len(selected)-len(error_rows)}/{len(selected)} cases. Results: {out}',flush=True)

if __name__=='__main__':main()
