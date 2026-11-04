"""RAM-bounded plotting of existing v3.1 DTW results. No DTW recalculation.

Read only the *selected* event-code columns and activity windows by scanning
compressed tables a row at a time. Never load a whole case into pandas.
"""
from __future__ import annotations
import csv
import gzip
import gc
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform


def _exists_image(path):
    """Resume only if a nontrivial image has already been saved."""
    p=Path(path)
    if not p.is_file() or p.stat().st_size <= 1024:
        return False
    # A previous MemoryError may leave a partially written PNG.
    try:
        with p.open('rb') as fp:
            if fp.read(8) != b'\x89PNG\r\n\x1a\n':
                return False
            fp.seek(-12,2)
            return fp.read(12) == b'\x00\x00\x00\x00IEND\xaeB`\x82'
    except (OSError, ValueError):
        return False


def _save_bounded(fig, path, dpi, problems, kind, cluster=None):
    """Never let image rasterization OOM invalidate finished DTW results."""
    try:
        fig.savefig(path,dpi=dpi)
        return True
    except (MemoryError, OSError, RuntimeError) as ex:
        msg=str(ex).lower()
        if not (isinstance(ex, MemoryError) or any(v in msg for v in ('allocation','out of memory','bad_alloc','not enough memory'))):
            raise
        Path(path).unlink(missing_ok=True)
        problems.append({'cluster':cluster,'case_id':'','kind':kind,'detail':repr(ex)})
        print(f'WARNING: plot was skipped due to memory exhaustion: {path} ({ex})',flush=True)
        return False
    finally:
        plt.close(fig)
        gc.collect()


def _write_warnings(path, problems):
    pd.DataFrame(problems,columns=['cluster','case_id','kind','detail']).to_csv(path,index=False)


def _case_dir(root: Path, case_id: str) -> Path:
    parts = case_id.split('__')
    if len(parts) != 3 or not parts[2].startswith('Chair_log_seq_'):
        raise ValueError(f'Invalid case_id: {case_id!r}')
    return root/'cases'/parts[0]/parts[1]/parts[2]


def _sample_bounded(arrays, max_points):
    """Deterministic chronological decimation for bounded output arrays."""
    if len(arrays[0]) <= max_points:
        return arrays
    ids = np.linspace(0, len(arrays[0])-1, max_points, dtype=int)
    return tuple(a[ids] for a in arrays)


def _stream_window(path: Path, start: float, end: float, codes, *, total_events=False, max_points=2000):
    """Stream a wide gz CSV using csv.reader, selecting only needed fields.

    For event_counts, sum *all* codes except E90012. For raw, select only codes.
    Assumes time_s sorted, as produced by chair_batch. Returned arrays are
    strictly observed values (NaNs stay missing). Memory bounded by max_points.
    """
    times = []
    columns = {c: [] for c in codes}
    count_values = []
    row_number = 0
    stride = 1
    if not path.exists():
        raise FileNotFoundError(path)
    with gzip.open(path, 'rt', encoding='utf-8-sig', errors='replace', newline='') as fp:
        reader = csv.reader(fp)
        header = next(reader)
        idx_time = header.index('time_s')
        selected = {c: header.index('E'+c) for c in codes if 'E'+c in header}
        count_indexes = [k for k, c in enumerate(header) if c!='time_s' and c!='E90012'] if total_events else []
        for fields in reader:
            try:
                t = float(fields[idx_time])
            except (ValueError, IndexError):
                continue
            if t < start:
                continue
            if t >= end:
                break
            row_number += 1
            if (row_number - 1) % stride != 0:
                continue
            times.append(t)
            if total_events:
                count=0.
                for k in count_indexes:
                    try:
                        v=float(fields[k]);count+=v if np.isfinite(v) else 0.
                    except (ValueError, IndexError):
                        continue
                count_values.append(count)
            for c in codes:
                k = selected.get(c)
                if k is not None and k < len(fields):
                    try:
                        columns[c].append(float(fields[k]))
                    except ValueError:
                        columns[c].append(np.nan)
                else:
                    columns[c].append(np.nan)
            # Bound memory even for pathologically long Activities.
            if len(times) > max_points:
                times = times[::2]
                for c in codes: columns[c] = columns[c][::2]
                if total_events: count_values = count_values[::2]
                stride *= 2
    t=np.array(times,dtype=float)
    y={c: np.array(v,dtype=float) for c,v in columns.items()}
    counts=np.array(count_values,dtype=float) if total_events else None
    if len(t)>max_points:
        ix=np.linspace(0,len(t)-1,max_points,dtype=int)
        t=t[ix];y={c:v[ix] for c,v in y.items()}
        if total_events:counts=counts[ix]
    return t,y,counts


def _read_exact(path, start, end, codes, max_points=3000):
    """Raw event values: stream single gzip CSV, no pandas tokenization."""
    buffers={c:[] for c in codes}
    strides={c:1 for c in codes}
    seen={c:0 for c in codes}
    with gzip.open(path,'rt',encoding='utf-8-sig',errors='replace',newline='') as fp:
        reader=csv.DictReader(fp)
        if not {'time_s','event_code','event_value'}.issubset(set(reader.fieldnames or [])):
            raise ValueError(f'Missing event columns: {path}')
        for row in reader:
            c=str(row.get('event_code','')).strip()
            if c not in buffers: continue
            try:
                t=float(row['time_s']);v=float(row['event_value'])
            except (TypeError,ValueError):continue
            if not (start<=t<end and np.isfinite(v)):continue
            seen[c]+=1
            if (seen[c]-1)%strides[c]:continue
            buffers[c].append((t,v))
            if len(buffers[c])>max_points:
                buffers[c]=buffers[c][::2];strides[c]*=2
    return buffers


def plot_saved_results(*, out: Path, data_root: Path, numeric_codes, state_codes,
                       no_exact_events=False, exact_max_points=3000,
                       max_members=8, max_window_points=600,
                       mode='light', resume=True):
    """Generate figures exclusively from saved clustering CSV + narrow CSV streams."""
    out=Path(out);data_root=Path(data_root)
    if mode not in ('light','medoids','diagnostics','full'):
        raise ValueError(f'Invalid plot mode: {mode}')
    if mode == 'full':
        max_members=max(1,min(30,max_members))
        max_window_points=max(600,max_window_points)
        dpi=130
        point_limit=500
    else:
        max_members=max(1,min(8,max_members))
        max_window_points=min(600,max_window_points)
        exact_max_points=min(700,exact_max_points)
        dpi=90
        point_limit=125
    diagnostics=[]
    warn_path=out/'plotting_warnings.csv'
    metadata=pd.read_csv(out/'cluster_assignments.csv',dtype={'case_id':str})
    D=np.loadtxt(out/'dtw_distance_matrix.csv',delimiter=',',ndmin=2)
    if not len(metadata)==len(D) or D.shape!=(len(metadata),len(metadata)):
        raise ValueError('DTW distance matrix and cluster_assignments.csv do not match')
    if 'dtw_index' in metadata.columns:
        metadata=metadata.sort_values('dtw_index').reset_index(drop=True)
    labels=metadata.cluster.to_numpy(int)
    np.fill_diagonal(D,0)
    Z=linkage(squareform(D,checks=False),method='average',optimal_ordering=len(D)<=120)
    representatives=out/'representatives'; representatives.mkdir(exist_ok=True)
    n=len(metadata)
    path=out/'cluster_dendrogram.png'
    if not (resume and _exists_image(path)):
        fig,ax=plt.subplots(figsize=(10,5))
        dendrogram(Z,ax=ax,no_labels=n>45,leaf_rotation=90,leaf_font_size=7,
                   color_threshold=0,above_threshold_color='#53657d')
        ax.set_title(f'Hierarchical clustering | n={n} activities (mixed DTW + raw level)')
        ax.set_ylabel('Distance (not clinical meaning)')
        _save_bounded(fig,path,dpi,diagnostics,'dendrogram_render_memory')
    order=leaves_list(Z)
    path=out/'distance_heatmap.png'
    if not (resume and _exists_image(path)):
        fig,ax=plt.subplots(figsize=(7,5))
        im=ax.imshow(D[np.ix_(order,order)],aspect='auto',cmap='viridis',interpolation='nearest')
        fig.colorbar(im,ax=ax,label='Combined distance')
        ax.set_title('Distance heatmap (dendrogram leaf order)')
        ax.set_xlabel('Activity');ax.set_ylabel('Activity')
        _save_bounded(fig,path,dpi,diagnostics,'distance_heatmap_render_memory')
    if mode == 'diagnostics':
        _write_warnings(warn_path,diagnostics)
        print('Plot mode diagnostics: skipped representatives.',flush=True)
        return diagnostics
    for cl in sorted(np.unique(labels)):
        members=np.flatnonzero(labels==cl)
        medoid_idx=int(members[np.argmin(D[np.ix_(members,members)].sum(axis=1))])
        r=metadata.iloc[medoid_idx]
        cid=str(r.case_id);start=float(r.start_s);end=float(r.end_s)
        case=_case_dir(data_root,cid)
        raw_file=case/'raw_1s.csv.gz'
        print(f'Plot cluster {cl}: {len(members)} members; medoid {cid} [{start:.1f},{end:.1f}]',flush=True)
        medoid_path=representatives/f'cluster_{cl:02d}_raw_medoid.png'
        overlay_path=representatives/f'cluster_{cl:02d}_member_overlay.png'
        need_medoid=not (resume and _exists_image(medoid_path))
        need_overlay=(mode in ('light','full') and not (resume and _exists_image(overlay_path)))
        if not (need_medoid or need_overlay):
            print(f'  Reusing existing plots for cluster {cl}.',flush=True)
            continue
        try:
            t,y,_=_stream_window(raw_file,start,end, list(state_codes)+list(numeric_codes),
                                 max_points=max_window_points)
        except (OSError,ValueError,IndexError,MemoryError,csv.Error) as ex:
            t=np.array([],float);y={c:np.array([],float) for c in list(state_codes)+list(numeric_codes)}
            diagnostics.append({'cluster':int(cl),'case_id':cid,'kind':'medoid_read_error','detail':repr(ex)})
        if need_medoid:
            exact=None
            exact_file=case/'events_original.csv.gz'
            if exact_file.exists() and not no_exact_events:
                try: exact=_read_exact(exact_file,start,end,list(numeric_codes),exact_max_points)
                except (OSError,ValueError,IndexError,MemoryError,csv.Error) as ex:
                    diagnostics.append({'cluster':int(cl),'case_id':cid,'kind':'exact_event_read_error','detail':repr(ex)})
                    print(f'Warning: exact events unavailable: {cid}: {ex}',flush=True)
            fig,axes=plt.subplots(2+len(numeric_codes),1,
                 figsize=(9,max(6,1.5*(2+len(numeric_codes)))),sharex=True)
            axes=np.atleast_1d(axes)
            for j,c in enumerate(state_codes):
                vals=y.get(c,np.array([]));ok=np.isfinite(vals)
                if ok.any(): axes[0].step(t[ok]-start,vals[ok]+3*j,where='post',lw=1.1,label=c)
            axes[0].set_title(f'Cluster {cl} | n={len(members)} | medoid {cid} | {start:.1f}–{end:.1f} s',fontsize=9)
            axes[0].set_ylabel('Raw states (offset)')
            if axes[0].has_data():axes[0].legend(fontsize=7,ncol=3)
            for j,c in enumerate(numeric_codes):
                ax=axes[j+1]; plotted=False
                if exact and exact.get(c):
                    xy=np.asarray(exact[c],float)
                    ax.plot(xy[:,0]-start,xy[:,1],'.',ms=2,alpha=.55,label='Exact original events');plotted=True
                if not plotted:
                    vals=y.get(c,np.array([]));ok=np.isfinite(vals)
                    if ok.any():
                        ax.plot(t[ok]-start, vals[ok],'.',ms=2,alpha=.6,label='Observed raw 1-s bins');plotted=True
                if not plotted:
                    ax.text(.5,.5,'No observed numeric samples',transform=ax.transAxes,ha='center',va='center')
                ax.set_ylabel(f'{c}\nRAW value')
            try:
                te,_,ce=_stream_window(case/'event_counts_1s.csv.gz',start,end,[],total_events=True,
                                      max_points=max_window_points)
                if len(te):axes[-1].step(te-start,ce,where='post',lw=1)
            except (OSError,ValueError,IndexError,MemoryError,csv.Error) as ex:
                diagnostics.append({'cluster':int(cl),'case_id':cid,'kind':'event_count_read_error','detail':repr(ex)})
            axes[-1].set_ylabel('All events\nper bin');axes[-1].set_xlabel('Seconds since Activity start')
            for ax in axes:ax.grid(axis='x',alpha=.2)
            _save_bounded(fig,medoid_path,dpi,diagnostics,'medoid_render_memory',int(cl))
            del exact
        if not need_overlay:
            del t,y
            _write_warnings(warn_path,diagnostics)
            gc.collect()
            continue

        # Compact overlay: a representative subset with a hard per-trace point bound.
        # Axes are intentionally small, no constrained layout, and no huge collections.
        fig,axs=plt.subplots(len(numeric_codes),1,
                             figsize=(9,max(3,1.7*len(numeric_codes))),sharex=True)
        axs=np.atleast_1d(axs)
        selected=members[np.linspace(0,len(members)-1,min(max_members,len(members)),dtype=int)]
        for member in selected:
            rr=metadata.iloc[int(member)]
            try:
                tm,ym,_=_stream_window(_case_dir(data_root,str(rr.case_id))/'raw_1s.csv.gz',
                                       float(rr.start_s),float(rr.end_s),numeric_codes,
                                       max_points=max_window_points)
            except (OSError,ValueError,IndexError,MemoryError,csv.Error) as ex:
                diagnostics.append({'cluster':int(cl),'case_id':str(rr.case_id),'kind':'member_read_error','detail':repr(ex)})
                continue
            for p,c in enumerate(numeric_codes):
                vals=ym[c];ok=np.isfinite(vals)
                if ok.any():
                    xp=(tm[ok]-float(rr.start_s))/max(float(rr.end_s)-float(rr.start_s),1e-8)
                    yp=vals[ok]
                    xp,yp=_sample_bounded((xp,yp),point_limit)
                    axs[p].plot(xp,yp,linestyle='None',marker='.',ms=1.5,alpha=.2,color='#477bb1')
            del tm,ym
        for p,c in enumerate(numeric_codes):
            vals=y.get(c,np.array([]));ok=np.isfinite(vals)
            if ok.any():
                xp=(t[ok]-start)/max(end-start,1e-8)
                yp=vals[ok]
                xp,yp=_sample_bounded((xp,yp),point_limit)
                axs[p].plot(xp,yp,linestyle='None',marker='.',ms=3,alpha=.8,
                            color='#cb542d',label='Medoid (1-s raw)')
            if axs[p].get_legend_handles_labels()[0]: axs[p].legend(fontsize=7)
            axs[p].set_ylabel(f'{c} raw');axs[p].grid(alpha=.15)
        axs[-1].set_xlabel('Relative time (observed 1-s bins; no interpolation)')
        fig.suptitle(f'Cluster {cl} | max {len(selected)} sampled members + medoid',fontsize=10)
        _save_bounded(fig,overlay_path,dpi,diagnostics,'overlay_render_memory',int(cl))
        del t,y
        _write_warnings(warn_path,diagnostics)
        gc.collect()
    _write_warnings(warn_path,diagnostics)
    print(f'Plotting finished. Output: {representatives}',flush=True)
    return diagnostics


def finish_saved_diagnostics(out: Path):
    """Re-create reports located after the old render_figures() call.

    Uses already-saved pairwise/activity CSVs; no DTW or raw-data reload.
    Existing files are preserved where possible.
    """
    from scipy.stats import spearmanr
    import json
    out=Path(out)
    pairpath=out/'pairwise_distance_components.csv'
    if pairpath.exists():
        pairs=pd.read_csv(pairpath)
        audit=[]
        for col in ('duration_difference_s','event_count_difference'):
            if col not in pairs or 'final_distance' not in pairs:continue
            rho,p=spearmanr(pairs.final_distance,pairs[col])
            audit.append({'factor':col,'spearman_rho':rho,
                          'p_value_naive_pairs_not_independent':p,
                          'warning':bool(abs(rho)>=.5)})
        pd.DataFrame(audit).to_csv(out/'distance_bias_diagnostics.csv',index=False)
        if len(pairs):
            fig,axes=plt.subplots(1,2,figsize=(9,3.5))
            for ax,col in zip(axes,('duration_difference_s','event_count_difference')):
                if col in pairs:
                    ax.scatter(pairs[col],pairs.final_distance,s=5,alpha=.3)
                ax.set_xlabel(col);ax.set_ylabel('DTW combined distance');ax.grid(alpha=.15)
            fig.suptitle('Distance bias diagnostics (activity pairs not independent)')
            _save_bounded(fig,out/'distance_bias_diagnostics.png',90,[],'diagnostics_render_memory')
    activitypath=out/'v3_1_all_activity_intervals.csv'
    if activitypath.exists():
        table=pd.read_csv(activitypath)
        if 'activity_type' in table and 'duration_s' in table and len(table):
            fig,ax=plt.subplots(figsize=(7,4))
            for cid,group in table.groupby('activity_type'):
                ax.hist(group.duration_s,alpha=.5,bins=np.logspace(0,4,35),label=cid)
            ax.set_xscale('log');ax.set_xlabel('Activity duration (seconds)')
            ax.set_ylabel('Count');ax.legend(fontsize=8)
            _save_bounded(fig,out/'activity_types_duration.png',90,[],'activity_render_memory')
    sensfile=out/'gap_boundary_sensitivity.csv'
    if sensfile.exists():
        sens=pd.read_csv(sensfile)
        if {'gap_factor','boundary_f1','median_shift_s'}.issubset(sens):
            fig,axs=plt.subplots(1,2,figsize=(9,3.5))
            for fac,g in sens.groupby('gap_factor'):
                for ax,col in zip(axs,('boundary_f1','median_shift_s')):
                    values=g[col].dropna().to_numpy()
                    if len(values):ax.boxplot(values,positions=[float(fac)],widths=.25)
            for ax in axs:ax.set_xticks([.5,1.,2.]);ax.set_xlabel('Episode gap multiplier');ax.grid(axis='y',alpha=.2)
            axs[0].set_ylabel('Boundary F1 vs base (±tolerance)')
            axs[1].set_ylabel('Median timing difference (s)')
            fig.suptitle('Sensitivity: boundary location, not only activity count')
            _save_bounded(fig,out/'gap_boundary_sensitivity.png',90,[],'sensitivity_render_memory')
    # Rebuild the small summary only if original failed before writing it.
    summary_path=out/'analysis_summary.json'
    if not summary_path.exists():
        assn=pd.read_csv(out/'cluster_assignments.csv',dtype={'case_id':str})
        cov_path=out/'case_coverage_summary.json'
        coverage=json.loads(cov_path.read_text(encoding='utf-8')) if cov_path.exists() else {}
        report={'cases_represented_in_dtw':int(assn.case_id.nunique()),
                'dtw_activities':int(len(assn)),
                'clusters_k':int(assn.cluster.nunique()),
                'report_type':'reconstructed from saved results after memory error',
                'cases_with_inputs':coverage.get('cases_with_inputs'),
                'note':'DTW values and clusters were reused, never recomputed. Original unsaved physical scaling parameters are not recoverable from the saved CSVs.'}
        summary_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Recreated post-plot diagnostics from saved CSVs (no DTW computation).',flush=True)
