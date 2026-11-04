#!/usr/bin/env python3
"""Replay/compare fixed Two-stage classes using existing results (no DTW recalculation).

Outputs per Stage-2 subcluster:
- Member overlay at normalized activity progress (%), using observed RAW 1 s values only.
- A representative (medoid) activity on actual time (s) with state and numeric data.
- Class coverage CSV and case-subsample ARI summary.

Only the required columns and time windows are loaded in chunks; missing is NOT zero.
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUM_DEFAULT = '20742,20743,20108'
STATE_DEFAULT = '20039,20011,20101'


def parse_args():
    p = argparse.ArgumentParser(description='Visualize existing Stage-2 subclusters without re-running DTW')
    p.add_argument('--two-stage-root', required=True, type=Path)
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--numeric-codes', default=NUM_DEFAULT)
    p.add_argument('--state-codes', default=STATE_DEFAULT)
    p.add_argument('--max-members', type=int, default=7)
    p.add_argument('--max-points', type=int, default=500)
    p.add_argument('--dpi', type=int, default=110)
    a = p.parse_args()
    a.out = a.out or a.two_stage_root / 'replay_class_examples'
    a.num_codes = [c.strip().removeprefix('E') for c in a.numeric_codes.split(',') if c.strip()]
    a.state_codes = [c.strip().removeprefix('E') for c in a.state_codes.split(',') if c.strip()]
    if a.max_members < 1 or a.max_points < 5 or a.dpi < 30:
        p.error('Invalid max-members, max-points, or dpi')
    if not a.num_codes:
        p.error('At least one numeric code is required')
    return a


def case_folder(root: Path, cid: str):
    parts = str(cid).split('__')
    if len(parts) != 3:
        raise ValueError(f'Unexpected case_id: {cid}')
    return root / 'cases' / parts[0] / parts[1] / parts[2]


def read_window(folder: Path, base: str, start: float, end: float, codes: list[str]):
    f = folder / f'{base}.csv.gz'
    if not f.exists():
        raise FileNotFoundError(f)
    wanted = {'time_s'} | {'E' + c for c in codes}
    pieces = []
    # No full-sized, wide case dataframes are loaded; only requested columns.
    for ch in pd.read_csv(f, compression='gzip',
                          usecols=lambda col: col in wanted,
                          chunksize=3000, low_memory=False):
        if 'time_s' not in ch:
            raise ValueError(f'Missing time_s in {f}')
        tt = pd.to_numeric(ch.time_s, errors='coerce')
        sel = (tt >= start) & (tt < end)
        if sel.any():
            keep = ch.loc[sel].copy()
            for col in keep.columns:
                keep[col] = pd.to_numeric(keep[col], errors='coerce')
            pieces.append(keep)
    if not pieces:
        return pd.DataFrame(columns=['time_s'] + ['E' + c for c in codes])
    return pd.concat(pieces, ignore_index=True)


def activity_signals(data_root: Path, row, num_codes, state_codes):
    folder = case_folder(data_root, str(row.case_id))
    start, end = float(row.start_s), float(row.end_s)
    codes = list(dict.fromkeys(num_codes + state_codes))
    raw = read_window(folder, 'raw_1s', start, end, codes)
    mask = read_window(folder, 'observation_mask_1s', start, end, codes)
    if raw.empty or mask.empty:
        return {c: (np.array([]), np.array([])) for c in codes}
    raw = raw.drop_duplicates('time_s').set_index('time_s')
    mask = mask.drop_duplicates('time_s').set_index('time_s')
    times = raw.index.intersection(mask.index)
    result = {}
    for code in codes:
        col = 'E' + code
        if col not in raw or col not in mask:
            result[code] = (np.array([]), np.array([]))
            continue
        x = raw.loc[times, col].to_numpy(dtype=float)
        observed = mask.loc[times, col].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(observed) & (observed > 0)
        t = times.to_numpy(dtype=float)[valid] - start
        result[code] = (t, x[valid])
    return result


def cap_points(t, x, limit):
    if len(t) <= limit:
        return t, x
    idx = np.unique(np.linspace(0, len(t) - 1, limit, dtype=int))
    return t[idx], x[idx]


def pick_members(group, medoid_row, limit):
    # Prefer an actual stored medoid; then maximize distinct-case representation.
    g = group.copy()
    g = g.sort_values(['case_id', 'activity_id']).reset_index(drop=True)
    selected = []
    if medoid_row is not None:
        match = g[(g.case_id == str(medoid_row.medoid_case_id)) &
                  (g.activity_id.astype(str) == str(medoid_row.medoid_activity_id))]
        if not match.empty:
            selected.append(match.iloc[0])
    if not selected:
        selected.append(g.iloc[0])
    selected_ids = {(str(r.case_id), str(r.activity_id)) for r in selected}
    selected_cases = {str(r.case_id) for r in selected}
    for _, r in g.iterrows():
        key = (str(r.case_id), str(r.activity_id))
        if len(selected) >= limit:
            break
        if str(r.case_id) not in selected_cases and key not in selected_ids:
            selected.append(r)
            selected_cases.add(str(r.case_id))
            selected_ids.add(key)
    for _, r in g.iterrows():
        key = (str(r.case_id), str(r.activity_id))
        if len(selected) >= limit:
            break
        if key not in selected_ids:
            selected.append(r)
            selected_ids.add(key)
    return selected


def display_id(r):
    parts = str(r.case_id).split('__')
    short = parts[-1].replace('Chair_log_seq_', '#') if len(parts) == 3 else str(r.case_id)
    return f'{short} A{r.activity_id}'


def plot_overlay(members, traces, num_codes, out, title, dpi, max_points):
    n = len(num_codes)
    fig, axes = plt.subplots(n, 1, figsize=(11, max(2.6 * n, 3.5)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    colors = plt.get_cmap('tab10')
    for j, code in enumerate(num_codes):
        ax = axes[j]
        for i, (r, rec) in enumerate(zip(members, traces)):
            tt, xx = rec[code]
            if not len(tt):
                continue
            tt, xx = cap_points(tt, xx, max_points)
            duration = max(float(r.end_s) - float(r.start_s), 1e-9)
            ax.scatter(tt * 100.0 / duration, xx, s=(10 if i == 0 else 6),
                       marker='o', alpha=(.8 if i == 0 else .35),
                       color=colors(i % 10), label=display_id(r) if j == 0 else None,
                       rasterized=True)
        ax.set_ylabel(f'{code}\noriginal value')
        ax.grid(alpha=.18)
        if j == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=2, markerscale=1.5)
    axes[-1].set_xlabel('Activity progress (%) — relative time, NOT temporal alignment by DTW')
    axes[-1].set_xlim(-1, 101)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .965])
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_medoid(row, rec, num_codes, state_codes, out, title, dpi, max_points):
    codes = state_codes + num_codes
    fig, axes = plt.subplots(len(codes), 1, figsize=(11, max(2.2 * len(codes), 4)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for j, code in enumerate(codes):
        t, x = rec.get(code, (np.array([]), np.array([])))
        t, x = cap_points(t, x, max_points)
        if j < len(state_codes):
            axes[j].scatter(t, x, s=12, color='tab:blue', alpha=.8)
        else:
            axes[j].scatter(t, x, s=8, color='tab:orange', alpha=.75)
        axes[j].set_ylabel(f'{code}\nraw')
        axes[j].grid(alpha=.18)
        if not len(t):
            axes[j].text(.97, .83, 'No observed points', transform=axes[j].transAxes,
                         ha='right', va='top', fontsize=8)
    axes[-1].set_xlabel('Time from Activity start (s)')
    fig.suptitle(title + ' | 1-second raw observations; missing ≠ OFF', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def main():
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    required = ['stage1_fixed_assignments.csv', 'two_stage_assignments.csv',
                'stage2_medoids.csv', 'stage2_summary.csv']
    missing = [fn for fn in required if not (a.two_stage_root / fn).exists()]
    if missing:
        raise SystemExit('Missing expected Two-stage output files: ' + ', '.join(missing))
    stage1 = pd.read_csv(a.two_stage_root / 'stage1_fixed_assignments.csv')
    labels = pd.read_csv(a.two_stage_root / 'two_stage_assignments.csv', dtype={'case_id': str})
    medoids = pd.read_csv(a.two_stage_root / 'stage2_medoids.csv', dtype={'medoid_case_id': str})
    summaries = pd.read_csv(a.two_stage_root / 'stage2_summary.csv')
    stability_path = a.two_stage_root / 'stage2_case_subsampling.csv'
    stabs = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()
    errors, recurrences, activity_examples = [], [], []
    print(f'Stage1: {len(stage1)} Activities, {stage1.case_id.nunique()} cases')
    if labels.empty or not labels.stage2_cluster.notna().any():
        print('No Stage2 classified activities. Check stage2_summary.csv status. Nothing to replay.')
    for (cl, sub), g in labels.loc[labels.stage2_cluster.notna()].groupby(
            ['stage1_cluster', 'stage2_cluster'], sort=True):
        cl, sub = int(cl), int(sub)
        g = g.sort_values(['case_id', 'activity_id']).reset_index(drop=True)
        m = medoids[(medoids.stage1_cluster == cl) & (medoids.stage2_cluster == sub)]
        medoid = m.iloc[0] if len(m) else None
        chosen = pick_members(g, medoid, a.max_members)
        trace_good = []
        for r in chosen:
            try:
                trace = activity_signals(a.data_root, r, a.num_codes, a.state_codes)
                trace_good.append((r, trace))
                activity_examples.append({'stage1_cluster': cl, 'stage2_cluster': sub,
                                          'case_id': r.case_id, 'activity_id': r.activity_id,
                                          'start_s': r.start_s, 'end_s': r.end_s,
                                          'is_medoid': bool(medoid is not None and
                                              str(r.case_id) == str(medoid.medoid_case_id) and
                                              str(r.activity_id) == str(medoid.medoid_activity_id)),
                                          'n_numeric_observed_points': int(sum(len(trace[c][0]) for c in a.num_codes))})
            except (OSError, MemoryError, ValueError, KeyError) as e:
                errors.append({'stage1_cluster': cl, 'stage2_cluster': sub,
                               'case_id': r.case_id, 'activity_id': r.activity_id, 'error': str(e)})
                warnings.warn(f'Skip activity {r.case_id}/{r.activity_id}: {e}')
        info = summaries[summaries.stage1_cluster == cl]
        status = str(info.iloc[0].status) if len(info) else 'UNKNOWN'
        ari = float(stabs[stabs.stage1_cluster == cl].ari_to_reference.median()) \
            if len(stabs) and 'ari_to_reference' in stabs and len(stabs[stabs.stage1_cluster == cl]) else np.nan
        recurrences.append({'stage1_cluster': cl, 'stage2_cluster': sub,
                            'n_activities': len(g), 'n_distinct_cases': g.case_id.nunique(),
                            'median_case_subsampling_ari_parent_B': ari,
                            'stage2_status_parent_B': status, 'plotted_activities': len(trace_good),
                            'medoid_case_id': medoid.medoid_case_id if medoid is not None else '',
                            'medoid_activity_id': medoid.medoid_activity_id if medoid is not None else ''})
        if not trace_good:
            continue
        tag = f'B{cl:02d}_S{sub:02d}'
        base = f'{tag} | {len(g)} activities / {g.case_id.nunique()} distinct cases | status: {status}'
        try:
            plot_overlay([r for r, _ in trace_good], [s for _, s in trace_good], a.num_codes,
                         a.out / f'{tag}_members_overlay.png', base, a.dpi, a.max_points)
            med = next(((r, tr) for r, tr in trace_good if medoid is not None and
                        str(r.case_id) == str(medoid.medoid_case_id) and
                        str(r.activity_id) == str(medoid.medoid_activity_id)), trace_good[0])
            plot_medoid(med[0], med[1], a.num_codes, a.state_codes,
                        a.out / f'{tag}_medoid_raw_1s.png', f'{tag} representative | {display_id(med[0])}',
                        a.dpi, a.max_points)
            print(f'{tag}: {len(g)} activities from {g.case_id.nunique()} cases; overlay={len(trace_good)}')
        except (MemoryError, OSError, ValueError) as e:
            errors.append({'stage1_cluster': cl, 'stage2_cluster': sub,
                           'case_id': '', 'activity_id': '', 'error': f'plot: {e}'})
            warnings.warn(f'{tag} plotting failed, continuing: {e}')
        finally:
            plt.close('all')
    pd.DataFrame(recurrences, columns=['stage1_cluster', 'stage2_cluster', 'n_activities',
        'n_distinct_cases', 'median_case_subsampling_ari_parent_B', 'stage2_status_parent_B',
        'plotted_activities', 'medoid_case_id', 'medoid_activity_id']).to_csv(
            a.out / 'class_recurrence.csv', index=False)
    pd.DataFrame(activity_examples, columns=['stage1_cluster', 'stage2_cluster', 'case_id',
        'activity_id', 'start_s', 'end_s', 'is_medoid', 'n_numeric_observed_points']).to_csv(
            a.out / 'plotted_activities.csv', index=False)
    pd.DataFrame(errors, columns=['stage1_cluster', 'stage2_cluster', 'case_id',
        'activity_id', 'error']).to_csv(a.out / 'plot_errors.csv', index=False)
    (a.out / 'NOTES.txt').write_text(
        'Plots show raw_1s.csv.gz OBSERVED 1-second values, NOT full original event times.\n'
        'Missing records are excluded, never replaced with zero.\n'
        'Member overlays place observations at percentage of each Activity duration; NO DTW warping or value interpolation is applied.\n'
        'Stage 2 subclusters are displayed unchanged; no clustering was rerun.\n'
        'Distinct cases and within-sample case-subsample ARI are separate measures of recurrence and stability,\n'
        'NOT independent validation on new patients or clinical diagnosis.\n',
        encoding='utf-8')
    print(f'Complete: {a.out}')


if __name__ == '__main__':
    main()
