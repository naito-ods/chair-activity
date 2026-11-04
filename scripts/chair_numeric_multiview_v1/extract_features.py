#!/usr/bin/env python3
"""Versioned case-level, non-DTW features from bounded original event logs.

No source or previous analysis file is changed. Absent target codes remain unknown
in clinical/operation features; only explicit recording-quality indicators use 0.
State exposures describe a documented last-recorded-value model, not proof of
continuous device observation. Numeric levels use observed 1-second-bin medians.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

VERSION = 'numeric_multiview_v1'
STATES = {'20011': [0, 1, 2, 3], '20039': [0, 1, 2, 3],
          '20101': [0, 1], '20774': [0, 1]}
NUMERIC = ['20742', '20108', '20743', '20776', '10014', '10015']
CODES = list(STATES) + NUMERIC
LABELS = {'20011': '選択HP', '20039': '作動中HP', '20742': '回転数',
          '20108': 'バルク開量', '20743': 'トルク', '20101': 'シリンジ',
          '20774': 'スケーラー作動', '20776': 'スケーラー出力',
          '10014': 'ペダル1', '10015': 'ペダル2'}
BLOCKS = {'20011': 'hp_state', '20039': 'hp_state', '20101': 'syringe',
          '20774': 'scaler', '20742': 'hp_numeric', '20108': 'hp_numeric',
          '20743': 'hp_numeric', '20776': 'scaler_numeric',
          '10014': 'pedal', '10015': 'pedal'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def finite_quantile(values, q):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else float('nan')


def state_intervals(g, duration, permitted):
    """Keep explicit unknown intervals; do not assume the pre-first-event state."""
    if g.empty:
        return np.array([]), np.array([]), np.array([])
    g = g.sort_values('time_s', kind='stable').drop_duplicates('time_s', keep='last')
    t = g.time_s.to_numpy(float)
    value = g.numeric_value.to_numpy(float)
    value[~np.isin(value, permitted)] = np.nan
    end = np.r_[t[1:], duration]
    valid = (t < duration) & (end > t)
    return t[valid], end[valid], value[valid]


def interval_summary(t, end, value, lo, hi):
    lengths = np.maximum(0, np.minimum(end, hi) - np.maximum(t, lo))
    known = np.isfinite(value)
    known_s = float(lengths[known].sum())
    active_s = float(lengths[known & (value != 0)].sum())
    return known_s, active_s


def one_case(record):
    source = Path(record['source_path'])
    meta_path = source.parent / 'meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    duration = float(meta['duration_s'])
    if not (duration > 0 and np.isclose(duration, float(record['duration_s']), atol=1e-6)):
        raise ValueError(f'Duration mismatch: {record["case_id"]}')
    if pd.Timestamp(meta['start']) != pd.Timestamp(record['start']) or pd.Timestamp(meta['end']) != pd.Timestamp(record['end']):
        raise ValueError(f'Datetime boundaries mismatch: {record["case_id"]}')
    target = set(CODES) | {'90012'}
    parts = []
    # Only three columns are loaded, and irrelevant event rows are discarded per chunk.
    for ch in pd.read_csv(source, usecols=['time_s', 'event_code', 'event_value'],
                          dtype={'event_code': str}, chunksize=20000, low_memory=False):
        ch.event_code = ch.event_code.str.strip()
        ch = ch[ch.event_code.isin(target)].copy()
        if not ch.empty:
            ch['time_s'] = pd.to_numeric(ch.time_s, errors='coerce')
            ch['numeric_value'] = pd.to_numeric(ch.event_value, errors='coerce')
            ch = ch[np.isfinite(ch.time_s)].copy()
            parts.append(ch[['time_s', 'event_code', 'numeric_value']])
    d = (pd.concat(parts, ignore_index=True) if parts else
         pd.DataFrame(columns=['time_s', 'event_code', 'numeric_value']))
    d = d.sort_values('time_s', kind='stable')
    marker = d[(d.event_code == '90012') & d.numeric_value.isin([0, 1])]
    on = marker[marker.numeric_value == 1]
    if on.empty:
        raise ValueError(f'Missing first ON: {record["case_id"]}')
    first_on = float(on.time_s.iloc[0])
    mv = marker.numeric_value.to_numpy(float)
    is_off_transition = (mv == 0) & (np.r_[np.nan, mv[:-1]] == 1)
    ends = marker[is_off_transition & (marker.time_s > first_on)]
    if ends.empty:
        raise ValueError(f'Missing final OFF: {record["case_id"]}')
    final_off = float(ends.time_s.iloc[-1])
    if not np.isclose(first_on, 0, atol=1e-6) or not np.isclose(final_off, duration, atol=1e-6):
        raise ValueError(f'90012/meta boundary mismatch: {record["case_id"]}, {first_on}, {final_off}, {duration}')
    ncycles = int(((mv == 1) & (np.r_[np.nan, mv[:-1]] != 1)).sum())
    boundary = dict(case_id=record['case_id'], first_on_s=first_on, final_off_s=final_off,
                    duration_s=duration, n_lighting_cycles=ncycles,
                    boundary_policy='first_90012_ON_to_last_90012_OFF',
                    matches_meta=True, matches_selected_manifest=True)
    n_expected_bins = int(math.ceil(duration))
    features = {'case_id': record['case_id']}
    catalog = []
    audits = []
    profiles = []
    state_models = {}
    numeric_events = {}

    def add(code, suffix, value, view, label, block=None):
        name = f'E{code}__{suffix}'
        features[name] = value
        catalog.append(dict(feature=name, block=block or BLOCKS[code], view=view,
                            label=f'{LABELS[code]}: {label}', code=code))

    target_rows = d[(d.event_code != '90012') & (d.time_s >= 0) & (d.time_s < duration)]
    for code in CODES:
        g = target_rows[target_rows.event_code == code].copy()
        valid_numeric = g[np.isfinite(g.numeric_value)].copy()
        vals = valid_numeric.numeric_value.to_numpy(float)
        if len(valid_numeric):
            bins = np.floor(valid_numeric.time_s.to_numpy(float)).astype(int)
            observed = pd.Series(vals, index=bins).groupby(level=0).median()
            bin_values = observed.to_numpy(float)
        else:
            observed = pd.Series(dtype=float)
            bin_values = np.array([])
        n_observed = len(observed)
        audits.append(dict(case_id=record['case_id'], code=code, n_events=len(g),
                           n_numeric=len(vals), n_observed_bins=n_observed,
                           observed_fraction=n_observed / n_expected_bins,
                           first_s=float(g.time_s.min()) if len(g) else np.nan,
                           last_s=float(g.time_s.max()) if len(g) else np.nan,
                           raw_min=float(np.min(vals)) if len(vals) else np.nan,
                           raw_q10=finite_quantile(vals, .1), raw_q50=finite_quantile(vals, .5),
                           raw_q90=finite_quantile(vals, .9),
                           raw_max=float(np.max(vals)) if len(vals) else np.nan,
                           bin_q10=finite_quantile(bin_values, .1),
                           bin_q50=finite_quantile(bin_values, .5),
                           bin_q90=finite_quantile(bin_values, .9)))
        # These zeros explicitly mean absence of recorded events, not absence of device use.
        add(code, 'event_present', int(len(g) > 0), 'quality', '記録イベントあり', 'quality')
        add(code, 'observed_bin_fraction', n_observed / n_expected_bins, 'quality',
            '実測値のある1秒bin割合', 'quality')
        add(code, 'events_per_min', len(g) / (duration / 60), 'volume',
            '記録イベント数/分', 'volume')
        if code in NUMERIC:
            numeric_events[code] = valid_numeric
            for suffix, q in [('q10', .1), ('q50', .5), ('q90', .9)]:
                add(code, suffix, finite_quantile(bin_values, q), 'levels', f'観測1秒中央値の{suffix}')
            add(code, 'iqr', finite_quantile(bin_values, .75) - finite_quantile(bin_values, .25),
                'levels', '観測1秒中央値のIQR')
            for third in range(3):
                lo, hi = duration * third / 3, duration * (third + 1) / 3
                in_third = observed[(observed.index >= lo) & (observed.index < hi)]
                add(code, f'third{third+1}_q50', finite_quantile(in_third, .5), 'coarse',
                    f'進捗第{third+1}三分区間の観測中央値', 'coarse')
            for progress in range(10):
                lo, hi = duration * progress / 10, duration * (progress + 1) / 10
                chunk = observed[(observed.index >= lo) & (observed.index < hi)]
                # Numeric samples stay sparse; no persistence or interpolation is used.
                profiles.append(dict(case_id=record['case_id'], code=code,
                    progress_bin=progress, progress_midpoint=(progress + .5) / 10,
                    raw_median=finite_quantile(chunk, .5), n_observed=len(chunk),
                    known_fraction=np.nan, active_fraction=np.nan))
        else:
            t, end, value = state_intervals(g, duration, STATES[code])
            state_models[code] = (t, end, value)
            known_s, active_s = interval_summary(t, end, value, 0, duration)
            add(code, 'known_fraction', known_s / duration, 'quality',
                '記録値保持モデルで状態既知の時間割合', 'quality')
            for state in STATES[code]:
                amount = np.maximum(0, end - t)[value == state].sum()
                add(code, f'state{state}_fraction', float(amount / known_s) if known_s else np.nan,
                    'operations', f'状態値{state}の時間割合（既知時間分母）')
            add(code, 'active_fraction', active_s / known_s if known_s else np.nan,
                'operations', '非0状態の時間割合（既知時間分母）')
            activation = 0
            changes = 0
            episode_durations = []
            episode_start = None
            # Only observed 0 -> positive transitions start an uncensored episode.
            for i in range(1, len(t)):
                previous, current = value[i-1], value[i]
                if np.isfinite(previous) and np.isfinite(current) and current != previous:
                    changes += 1
                if not np.isfinite(current) or not np.isfinite(previous):
                    episode_start = None
                    continue
                if previous == 0 and current != 0:
                    activation += 1
                    episode_start = t[i]
                if previous != 0 and current == 0:
                    if episode_start is not None:
                        episode_durations.append(t[i] - episode_start)
                    episode_start = None
            add(code, 'activation_per_min', activation / (known_s / 60) if known_s else np.nan,
                'operations', '観測0→非0遷移/既知時間分')
            add(code, 'state_changes_per_min', changes / (known_s / 60) if known_s else np.nan,
                'operations', '観測状態遷移/既知時間分')
            for suffix, q in [('q50', .5), ('q90', .9)]:
                add(code, f'episode_{suffix}_s', finite_quantile(episode_durations, q),
                    'operations', f'両端観測された非0区間長{suffix}（秒）')
            for third in range(3):
                lo, hi = duration * third / 3, duration * (third + 1) / 3
                known, active = interval_summary(t, end, value, lo, hi)
                add(code, f'third{third+1}_active_fraction', active / known if known else np.nan,
                    'coarse', f'進捗第{third+1}三分区間の非0状態割合', 'coarse')
            for progress in range(10):
                lo, hi = duration * progress / 10, duration * (progress + 1) / 10
                known, active = interval_summary(t, end, value, lo, hi)
                chunk = observed[(observed.index >= lo) & (observed.index < hi)]
                profiles.append(dict(case_id=record['case_id'], code=code,
                    progress_bin=progress, progress_midpoint=(progress + .5) / 10,
                    raw_median=finite_quantile(chunk, .5), n_observed=len(chunk),
                    known_fraction=known / (hi - lo),
                    active_fraction=active / known if known else np.nan))
    # Pair numerical samples with the most recent explicitly recorded device state
    # at the *actual event timestamp*, before reducing to one value per second.
    for state_code, states, numeric_codes in [('20039', [1, 2, 3], ['20742', '20743', '20108']),
                                             ('20774', [1], ['20776'])]:
        st, se, sv = state_models[state_code]
        for code in numeric_codes:
            events = numeric_events[code]
            sampled_states = np.full(len(events), np.nan)
            if len(st) and len(events):
                event_times = events.time_s.to_numpy(float)
                ix = np.searchsorted(st, event_times, side='right') - 1
                known = ix >= 0
                sampled_states[known] = sv[ix[known]]
                still_in_interval = np.zeros(len(events), dtype=bool)
                still_in_interval[known] = event_times[known] < se[ix[known]]
                sampled_states[~still_in_interval] = np.nan
            for state in states:
                subset = events[sampled_states == state]
                if len(subset):
                    by_second = pd.Series(subset.numeric_value.to_numpy(float),
                        index=np.floor(subset.time_s.to_numpy(float)).astype(int)).groupby(level=0).median()
                else:
                    by_second = pd.Series(dtype=float)
                for suffix, q in [('q50', .5), ('q90', .9)]:
                    add(code, f'given_E{state_code}_state{state}_{suffix}',
                        finite_quantile(by_second, q), 'conditional',
                        f'{state_code}記録保持状態{state}の時刻に観測された数値の1秒bin{suffix}')
                add(code, f'given_E{state_code}_state{state}_n_bins', len(by_second),
                    'quality', f'{state_code}状態{state}条件付き実測bin数', 'quality')
    select_t, select_end, select_value = state_models['20011']
    run_t, run_end, run_value = state_models['20039']
    overlap_s = 0.0
    equal_s = 0.0
    if len(select_t) and len(run_t):
        breaks = np.unique(np.r_[0, duration, select_t, select_end, run_t, run_end])
        middle = (breaks[:-1] + breaks[1:]) / 2
        widths = np.diff(breaks)
        si = np.searchsorted(select_t, middle, side='right') - 1
        ri = np.searchsorted(run_t, middle, side='right') - 1
        good = (si >= 0) & (ri >= 0)
        gs, gr = np.maximum(si, 0), np.maximum(ri, 0)
        good &= (middle < select_end[gs]) & (middle < run_end[gr])
        good &= np.isfinite(select_value[gs]) & np.isfinite(run_value[gr]) & (run_value[gr] != 0)
        overlap_s = widths[good].sum()
        equal_s = widths[good & (select_value[gs] == run_value[gr])].sum()
    add('20039', 'selected_match_given_running', equal_s / overlap_s if overlap_s else np.nan,
        'operations', '作動非0かつ両状態既知の時間における選択値との一致割合')
    add('20039', 'selected_running_comparable_fraction', overlap_s / duration,
        'quality', '作動非0かつ選択・作動状態比較可能な時間割合', 'quality')
    # Duration is retained for audit and an explicit separate size/volume view.
    features['case_duration_s'] = duration
    catalog.append(dict(feature='case_duration_s', block='volume', view='volume',
                        label='照明最初ONから最後OFFの症例時間（秒）', code='90012'))
    signal_status = 'OK_TARGET_SIGNAL' if len(target_rows) else 'NO_TARGET_SIGNAL'
    manifest = dict(record, signal_status=signal_status, n_target_events=len(target_rows),
                    n_recorded_target_codes=target_rows.event_code.nunique(),
                    source_sha256=sha256(source), meta_sha256=sha256(meta_path))
    return dict(features=features, catalog=catalog, audits=audits, profiles=profiles,
                boundary=boundary, manifest=manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--selection', type=Path, default=None)
    parser.add_argument('--root', type=Path, default=None)
    parser.add_argument('--smoke', action='store_true', help='Validate three representative cases only; do not emit production CSVs')
    args = parser.parse_args()
    root = args.root or Path(__file__).resolve().parents[3]
    # Script lives C31/chair-activity/scripts/chair_numeric_multiview_v1/.
    root = root.resolve()
    if not (root / 'Output').is_dir():
        raise FileNotFoundError(f'C31 root not found: {root}')
    selection = args.selection or root / 'Output/chair_hierarchical_v3_300/selected_cases.csv'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'process').mkdir(exist_ok=True)
    log_path = out / 'process/extraction_log.jsonl'

    def log(event, **details):
        with log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(timestamp_utc=stamp(), event=event, **details), ensure_ascii=False) + '\n')

    started = time.time()
    selected = pd.read_csv(selection, encoding='utf-8-sig')
    assert len(selected) == 300, f'Expected 300 selected cases, got {len(selected)}'
    records = []
    for index, row in enumerate(selected.itertuples()):
        cid = f'{row.date_folder}__{row.chair_folder}__Chair_log_seq_{int(row.seq)}'
        source = root / 'Output/chair_model_A_results/cases' / row.date_folder / row.chair_folder / f'Chair_log_seq_{int(row.seq)}' / 'events_original.csv.gz'
        if not source.is_file() or not (source.parent / 'meta.json').is_file():
            log('preflight_failure', case_id=cid, source_path=str(source), reason='missing_required_input')
            raise FileNotFoundError(source)
        records.append(dict(case_id=cid, date=row.date_folder, chair=row.chair_folder,
                            seq=int(row.seq), duration_s=float(row.duration_s), start=row.start,
                            end=row.end, source_path=str(source), selected_index=index))
    assert len({r['case_id'] for r in records}) == 300
    log('preflight_success', requested_cases=300, input_cases=len(records), workers=args.workers,
        selection_sha256=sha256(selection), smoke=args.smoke,
        boundary='first_90012_ON_to_last_90012_OFF',
        numeric_policy='per_observed_second_median_no_fill',
        state_policy='last_recorded_value_after_first_observation; invalid value creates unknown interval',
        clinical_scope='user_supplied_signal_descriptions; clusters_are_operation_candidates')
    print(f'[PREFLIGHT] 300 requested and available cases; workers={args.workers}', flush=True)
    if args.smoke:
        smoke_records = [records[0], records[len(records)//2], records[-1]]
        results = [one_case(r) for r in smoke_records]
        log('smoke_success', n_cases=len(results), n_features=len(results[0]['features'])-1,
            case_ids=[r['manifest']['case_id'] for r in results])
        print(f'[SMOKE] {len(results)} cases passed; {len(results[0]["features"])-1} features', flush=True)
        return
    outputs = ['features.csv', 'feature_catalog.csv', 'signal_audit.csv', 'case_manifest.csv',
               'boundary_audit.csv', 'extraction_metadata.json', 'coarse_profiles.csv']
    existing = [name for name in outputs if (out/name).exists()]
    if existing:
        raise FileExistsError(f'Refusing to overwrite extraction outputs: {existing}')
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one_case, record): record for record in records}
        for future in as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
                results.append(result)
                log('case_extracted', case_id=record['case_id'], signal_status=result['manifest']['signal_status'],
                    n_target_events=result['manifest']['n_target_events'], n_complete=len(results))
            except Exception as exc:
                failure = dict(case_id=record['case_id'], exception=type(exc).__name__, reason=str(exc))
                failures.append(failure)
                log('case_failure', **failure)
            if (len(results) + len(failures)) % 25 == 0:
                print(f'[EXTRACT] success={len(results)}, failed={len(failures)}, requested=300', flush=True)
    if failures:
        log('extraction_failed', requested=300, actual_success=len(results), failures=failures)
        raise RuntimeError(f'{len(failures)} extraction failures; see {log_path}')
    results.sort(key=lambda r: r['manifest']['selected_index'])
    assert len(results) == 300
    features = pd.DataFrame([r['features'] for r in results])
    catalog = pd.DataFrame(results[0]['catalog'])
    manifest = pd.DataFrame([r['manifest'] for r in results])
    boundary = pd.DataFrame([r['boundary'] for r in results])
    audit = pd.DataFrame([item for r in results for item in r['audits']])
    profiles = pd.DataFrame([item for r in results for item in r['profiles']])
    assert len(features) == features.case_id.nunique() == 300
    assert len(audit) == 3000 and not audit.duplicated(['case_id', 'code']).any()
    assert len(profiles) == 30000 and not profiles.duplicated(['case_id', 'code', 'progress_bin']).any()
    assert set(catalog.feature) == set(features.columns) - {'case_id'}
    assert not catalog.feature.duplicated().any()
    absent_ids = manifest.loc[manifest.signal_status == 'NO_TARGET_SIGNAL', 'case_id'].tolist()
    clinical_features = catalog.loc[catalog.view.isin(['levels', 'operations', 'coarse', 'conditional']), 'feature']
    assert features.set_index('case_id').loc[absent_ids, clinical_features].isna().all().all()
    for name, frame in [('features.csv', features), ('feature_catalog.csv', catalog),
                        ('signal_audit.csv', audit), ('case_manifest.csv', manifest),
                        ('boundary_audit.csv', boundary), ('coarse_profiles.csv', profiles)]:
        frame.to_csv(out/name, index=False)
    metadata = dict(version=VERSION, created_utc=stamp(), requested_cases=300, extracted_cases=300,
        cases_with_target_signal=int((manifest.signal_status != 'NO_TARGET_SIGNAL').sum()),
        no_target_signal_case_ids=absent_ids, exclusions=[], feature_count=len(catalog),
        selection_source=str(selection.resolve()), selection_sha256=sha256(selection),
        script_source=str(Path(__file__).resolve()), script_sha256=sha256(__file__),
        code_meanings=LABELS, meaning_source='user_supplied_in_conversation_2026-11-04',
        boundary_policy='first_90012_ON_to_last_90012_OFF_inclusive_marker; feature_window_[0,duration)',
        numeric_policy='raw values; median within observed integer-second bin; q10/q50/q90/IQR across observed bins; no imputation',
        state_policy='last recorded state held to next same-code record or case end; unknown before first record; invalid states create unknown intervals',
        state_proportion_denominator='known duration only; unknown duration separately recorded',
        state_episode_policy='only 0-to-nonzero activated and explicitly nonzero-to-0 closed episodes; censored runs excluded from episode quantiles',
        quality_zero_meaning='0 event presence/count/observed bins means no matching record, not confirmed OFF/non-use',
        event_count_policy='recording rate only; separate volume view',
        coarse_policy='10 progress bins for visual audit; thirds for optional clustering; observed bins only, no interpolation',
        conditional_policy='numeric events paired by actual timestamp with last recorded same-case state; then per-state per-second medians; no value before first state record',
        feature_views=catalog.groupby('view').size().to_dict(), workers=args.workers,
        chunk_rows=20000, usecols=['time_s','event_code','event_value'], full_wide_tables_loaded=False,
        elapsed_seconds=time.time()-started, python=sys.version,
        numpy_version=np.__version__, pandas_version=pd.__version__)
    (out/'extraction_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    log('extraction_success', requested=300, actual=300, n_no_target_signal=len(absent_ids),
        no_target_signal_case_ids=absent_ids, n_features=len(catalog), elapsed_seconds=time.time()-started)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
