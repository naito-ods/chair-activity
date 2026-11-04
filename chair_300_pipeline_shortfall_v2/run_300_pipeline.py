"""Run a reproducible 300-case Chair-log DTW -> ablation -> two-stage workflow.

All paths and data are read-only except newly created folders under --output-root.
Uses the bundled, previously validated analytical modules without altering DTW equations.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

INPUTS = ('hierarchical_hybrid_activity_intervals.csv', 'level1_operations.csv',
          'level1_episodes.csv', 'activity_intervals_all_methods.csv')


def arguments():
    p = argparse.ArgumentParser(description='300-case Hybrid v3 -> DTW -> ablation -> two-stage')
    p.add_argument('--output-root', type=Path,
                   default=Path(r'C:\131share\DB\C31\Output'))
    p.add_argument('--data-root', type=Path, default=None)
    p.add_argument('--hierarchy-root', type=Path, default=None)
    p.add_argument('--cases', type=int, default=300)
    p.add_argument('--activities', type=int, default=300,
                   help='>= cases; 300 initially recommended for 16 GB RAM')
    p.add_argument('--max-per-case', type=int, default=2)
    p.add_argument('--max-bins', type=int, default=64,
                   help='Same resolution in DTW baseline and Ablation A')
    p.add_argument('--max-k', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--stage', choices=('all', 'dtw', 'ablation', 'two-stage', 'check'), default='all')
    p.add_argument('--resume', action='store_true',
                   help='Reuse a completed stage if its files pass coverage/identity validation')
    p.add_argument('--allow-case-shortfall', action='store_true',
                   help='Allow fewer than the requested number of cases (e.g., 296/300). All stages record ACTUAL case counts.')
    p.add_argument('--min-case-fraction', type=float, default=0.95,
                   help='Minimum fraction of requested cases retained when --allow-case-shortfall is set (default 0.95)')
    p.add_argument('--dtw-out', type=Path, default=None)
    p.add_argument('--ablation-out', type=Path, default=None)
    p.add_argument('--two-stage-out', type=Path, default=None)
    p.add_argument('--subsample-repeats', type=int, default=50)
    a = p.parse_args()
    a.data_root = a.data_root or a.output_root / 'chair_model_A_results'
    a.hierarchy_root = a.hierarchy_root or a.output_root / 'chair_hierarchical_v3_300'
    a.dtw_out = a.dtw_out or a.output_root / 'chair_dtw_v3_1_300_casefirst'
    a.ablation_out = a.ablation_out or a.output_root / 'chair_ablation_300'
    a.two_stage_out = a.two_stage_out or a.output_root / 'chair_two_stage_300'
    if a.cases < 3 or a.activities < a.cases or a.max_per_case < 1 or a.max_bins < 16:
        p.error('--activities must be >= --cases; other limits invalid')
    if a.activities > a.cases * a.max_per_case:
        p.error('--activities exceeds --cases times --max-per-case')
    if a.subsample_repeats < 1:
        p.error('--subsample-repeats must be positive')
    if not (0 < a.min_case_fraction <= 1):
        p.error('--min-case-fraction must be within (0,1]')
    return a


def counted_ids(path):
    if not path.is_file():
        raise FileNotFoundError(f'Required file absent: {path}')
    ids = set()
    n = 0
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'case_id' not in reader.fieldnames:
            raise ValueError(f'case_id column missing: {path}')
        for r in reader:
            n += 1
            ids.add(str(r['case_id']))
    return len(ids), n


def preflight(a):
    for name in INPUTS:
        p = a.hierarchy_root / name
        if not p.is_file():
            raise FileNotFoundError(f'Hybrid v3 results missing {name}: {p}\n'
                                    'Set --hierarchy-root to your actual 300-case folder.')
    if not (a.data_root / 'cases').is_dir():
        raise FileNotFoundError(f'Preprocessed cases folder absent: {a.data_root / "cases"}')
    n, windows = counted_ids(a.hierarchy_root / INPUTS[0])
    print(f'[CHECK] Hybrid v3: {n} cases, {windows} Activity windows', flush=True)
    if n < a.cases:
        raise ValueError(f'Hybrid results have {n} cases, but --cases {a.cases} specified. '
                         'Do not silently relabel fewer cases as 300.')
    return n


def check_dtw(a):
    root = a.dtw_out
    report_path = root / 'case_coverage_summary.json'
    assign = root / 'cluster_assignments.csv'
    dmat = root / 'dtw_distance_matrix.csv'
    for f in (report_path, assign, dmat):
        if not f.is_file():
            raise FileNotFoundError(f'DTW output missing: {f}')
    with report_path.open(encoding='utf-8') as file:
        report = json.load(file)
    df = pd.read_csv(assign, dtype={'case_id': str})
    n_cases = df.case_id.nunique()
    n_act = len(df)
    report_n = int(report.get('cases_represented_in_dtw', -1))
    if n_cases != report_n:
        raise ValueError(f'DTW case coverage disagrees between CSV={n_cases} and report={report_n}')
    if a.allow_case_shortfall:
        minimum = math.ceil(a.cases * a.min_case_fraction)
        if not (minimum <= n_cases <= a.cases):
            raise ValueError(f'DTW case coverage={n_cases}/{a.cases}, required minimum={minimum}; '
                             'inspect case_coverage.csv')
        print(f'[QC] Shortfall permitted: {n_cases}/{a.cases} actual cases '
              f'(excluded {a.cases-n_cases}; threshold {minimum})', flush=True)
    elif n_cases != a.cases:
        raise ValueError(f'DTW case coverage mismatch: CSV={n_cases}, report={report_n}, '
                         f'requested={a.cases}; inspect case_coverage.csv')
    if n_act != a.activities:
        raise ValueError(f'DTW has {n_act} activities, expected {a.activities}. '
                         'Inspect selected_activity_manifest.csv for shortfall.')
    if df.duplicated(['case_id', 'activity_id', 'start_s', 'end_s']).any():
        raise ValueError('Duplicate DTW activities')
    print(f'[PASS] DTW: {n_cases} cases, {n_act} fixed activities', flush=True)
    return df


def check_ablation(a, baseline):
    root = a.ablation_out
    fixed_path = root / 'fixed_activity_manifest.csv'
    assignment_path = root / 'ablation_cluster_assignments.csv'
    b_dist = root / 'distances' / 'B_state_events.csv'
    fidelity = root / 'baseline_fidelity.json'
    for f in (fixed_path, assignment_path, b_dist, fidelity):
        if not f.exists():
            raise FileNotFoundError(f'Ablation output missing: {f}')
    fixed = pd.read_csv(fixed_path, dtype={'case_id': str})
    keys = ['case_id', 'activity_id', 'start_s', 'end_s']
    a_ids = set(tuple(x) for x in fixed[keys].itertuples(index=False, name=None))
    b_ids = set(tuple(x) for x in baseline[keys].itertuples(index=False, name=None))
    if len(fixed) != len(baseline) or a_ids != b_ids:
        raise ValueError('Ablation lost or changed fixed activities; read_errors.csv may explain.')
    with fidelity.open(encoding='utf-8') as f:
        match = json.load(f)
    err = match.get('max_absolute_difference')
    if err is None or float(err) > 1e-5:
        raise ValueError('Ablation A mismatch with saved DTW baseline: '\
                         f'max_absolute_difference={err}. Check max-bins/band/codes before Stage 2.')
    b = pd.read_csv(assignment_path, dtype={'case_id': str})
    stage1 = b[b.variant == 'B_state_events']
    if len(stage1) != len(baseline) or stage1.case_id.nunique() != baseline.case_id.nunique():
        raise ValueError('Stage1 B does not include all fixed activities/cases.')
    print(f'[PASS] Ablation: fixed {len(fixed)} activities, {baseline.case_id.nunique()} cases; '
          f'A fidelity max diff {float(err):.3g}', flush=True)
    return fixed


def check_two_stage(a, fixed):
    out = a.two_stage_out
    summary = out / 'stage2_summary.csv'
    labels = out / 'two_stage_assignments.csv'
    fixed_out = out / 'stage1_fixed_assignments.csv'
    for f in (summary, labels, fixed_out):
        if not f.is_file():
            raise FileNotFoundError(f'Two-stage output missing: {f}')
    stage1 = pd.read_csv(fixed_out, dtype={'case_id': str})
    if len(stage1) != len(fixed) or stage1.case_id.nunique() != fixed.case_id.nunique():
        raise ValueError('Two-stage Stage1 case/activity set differs from ablation')
    print(f'[PASS] Two-stage: {len(stage1)} fixed activities, {fixed.case_id.nunique()} cases; '
          f'Stage2 summary at {summary}', flush=True)


def execute(cmd, workdir):
    print('\n[RUN] ' + subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(workdir))


def main():
    a = arguments()
    src = Path(__file__).resolve().parent
    for p in (a.dtw_out, a.ablation_out, a.two_stage_out):
        if p.resolve() == a.data_root.resolve() or p.resolve() == a.hierarchy_root.resolve():
            raise ValueError('Refusing to write outputs over preprocessed/hierarchical inputs')
    preflight(a)
    if a.stage == 'check':
        try:
            baseline = check_dtw(a)
            fixed = check_ablation(a, baseline)
            check_two_stage(a, fixed)
        except (FileNotFoundError, ValueError) as e:
            print('[CHECK INCOMPLETE] ' + str(e))
            sys.exit(1)
        return

    stages = ['dtw','ablation','two-stage'] if a.stage == 'all' else [a.stage]
    baseline = None
    fixed = None
    for stage in stages:
        if stage == 'dtw':
            if a.resume:
                try:
                    baseline = check_dtw(a)
                    print('[SKIP] Existing DTW results validated.', flush=True)
                    continue
                except (FileNotFoundError, ValueError) as e:
                    print('[INFO] DTW results not validated; recalculating: ' + str(e), flush=True)
            args = [sys.executable, str(src/'chair_dtw_v3_1.py'),
                    '--data-root', str(a.data_root), '--hierarchy-root', str(a.hierarchy_root),
                    '--out', str(a.dtw_out), '--cases', str(a.cases),
                    '--activities', str(a.activities), '--max-per-case', str(a.max_per_case),
                    '--max-bins', str(a.max_bins), '--max-k', str(a.max_k),
                    '--seed', str(a.seed), '--plot-mode', 'diagnostics', '--no-exact-events']
            if a.allow_case_shortfall:
                args.append('--allow-case-shortfall')
            execute(args,src)
            baseline = check_dtw(a)
        elif stage == 'ablation':
            if baseline is None:
                baseline = check_dtw(a)
            if a.resume:
                try:
                    fixed = check_ablation(a,baseline)
                    print('[SKIP] Existing Ablation results validated.', flush=True)
                    continue
                except (FileNotFoundError, ValueError) as e:
                    print('[INFO] Ablation not validated; recalculating: ' + str(e), flush=True)
            args = [sys.executable, str(src/'chair_ablation.py'),
                    '--data-root', str(a.data_root), '--baseline-root', str(a.dtw_out),
                    '--out', str(a.ablation_out), '--max-bins', str(a.max_bins),
                    '--max-k', str(a.max_k), '--numeric-min-coverage', '0.10',
                    '--max-medoid-plots', '0']
            execute(args,src)
            fixed = check_ablation(a,baseline)
        elif stage == 'two-stage':
            if baseline is None:
                baseline = check_dtw(a)
            if fixed is None:
                fixed = check_ablation(a,baseline)
            if a.resume:
                try:
                    check_two_stage(a,fixed)
                    print('[SKIP] Existing Two-stage results validated.', flush=True)
                    continue
                except (FileNotFoundError, ValueError) as e:
                    print('[INFO] Two-stage not validated; recalculating: ' + str(e), flush=True)
            args = [sys.executable, str(src/'chair_two_stage.py'),
                    '--ablation-root', str(a.ablation_out), '--data-root', str(a.data_root),
                    '--out', str(a.two_stage_out), '--max-bins', str(a.max_bins),
                    '--subsample-repeats', str(a.subsample_repeats), '--max-medoid-plots', '0']
            execute(args,src)
            check_two_stage(a,fixed)
    print('\n[DONE] Requested stages completed and validated.', flush=True)
    print('DTW:      ', a.dtw_out)
    print('Ablation: ', a.ablation_out)
    print('Two-stage:', a.two_stage_out)


if __name__ == '__main__':
    main()
