#!/usr/bin/env python3
"""Refine v1 features using user-confirmed scaler-setting and syringe semantics.

20776 is a change-only setting, carried to the next record after initial known
value. 20101=0 means syringe picked up, 1 means stored; pickup is not spraying.
The v1 artifacts and all event inputs stay read-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

VERSION = 'numeric_multiview_recording_semantics_v2'
READ_CODES = {'20774', '20776', '20101', '90012'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def weighted_quantile(value, weight, q):
    value = np.asarray(value, dtype=float)
    weight = np.asarray(weight, dtype=float)
    good = np.isfinite(value) & np.isfinite(weight) & (weight > 0)
    value, weight = value[good], weight[good]
    if not len(value):
        return float('nan')
    order = np.argsort(value, kind='stable')
    value, weight = value[order], weight[order]
    ix = np.searchsorted(np.cumsum(weight), q * weight.sum(), side='left')
    return float(value[min(ix, len(value)-1)])


def state_model(events, code, duration, states=None):
    g = events[events.event_code == code].sort_values('time_s', kind='stable')
    g = g.drop_duplicates('time_s', keep='last')
    if g.empty:
        return np.array([]), np.array([]), np.array([])
    t = g.time_s.to_numpy(float)
    value = g.numeric_value.to_numpy(float)
    if states is not None:
        value[~np.isin(value, states)] = np.nan
    else:
        value[~np.isfinite(value)] = np.nan
    end = np.r_[t[1:], duration]
    keep = (t < duration) & (end > t)
    return t[keep], end[keep], value[keep]


def lookup(model, times):
    t, end, value = model
    answer = np.full(len(times), np.nan)
    if not len(t):
        return answer
    ix = np.searchsorted(t, times, side='right') - 1
    valid = ix >= 0
    valid[valid] &= times[valid] < end[ix[valid]]
    answer[valid] = value[ix[valid]]
    return answer


def refine_case(record):
    source = Path(record['source_path'])
    parts = []
    for chunk in pd.read_csv(source, usecols=['time_s', 'event_code', 'event_value'],
                             dtype={'event_code': str}, chunksize=20000, low_memory=False):
        chunk.event_code = chunk.event_code.str.strip()
        chunk = chunk[chunk.event_code.isin(READ_CODES)].copy()
        if not chunk.empty:
            chunk['time_s'] = pd.to_numeric(chunk.time_s, errors='coerce')
            chunk['numeric_value'] = pd.to_numeric(chunk.event_value, errors='coerce')
            parts.append(chunk[['time_s', 'event_code', 'numeric_value']])
    events = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['time_s', 'event_code', 'numeric_value'])
    events = events[np.isfinite(events.time_s)].sort_values('time_s', kind='stable')
    duration = float(record['duration_s'])
    marker = events[(events.event_code == '90012') & events.numeric_value.isin([0, 1])]
    if marker.empty:
        raise ValueError(f'Missing 90012: {record["case_id"]}')
    first_on = marker[marker.numeric_value == 1].time_s.min()
    mv = marker.numeric_value.to_numpy(float)
    off = marker[(mv == 0) & (np.r_[np.nan, mv[:-1]] == 1) & (marker.time_s > first_on)]
    last_off = off.time_s.max()
    assert np.isclose(first_on, 0) and np.isclose(last_off, duration), record['case_id']
    events = events[(events.time_s >= 0) & (events.time_s < duration)]
    setting = state_model(events, '20776', duration)
    scaler = state_model(events, '20774', duration, [0, 1])
    syringe = state_model(events, '20101', duration, [0, 1])
    breaks = np.unique(np.r_[0, duration, setting[0], setting[1], scaler[0], scaler[1]])
    widths = np.diff(breaks)
    mid = (breaks[:-1] + breaks[1:]) / 2
    state = lookup(scaler, mid)
    values = lookup(setting, mid)
    active = state == 1
    known_active = active & np.isfinite(values)
    active_s = float(widths[active].sum()) if np.isfinite(state).any() else np.nan
    scaler_known_s = float(widths[np.isfinite(state)].sum())
    known_active_s = float(widths[known_active].sum())
    setting_known_s = float(widths[np.isfinite(values)].sum())
    known_active_fraction = known_active_s / active_s if active_s > 0 else np.nan
    x, w = values[known_active], widths[known_active]
    f = {}
    for suffix, q in [('q10', .1), ('q50', .5), ('q90', .9)]:
        f[f'E20776__active_setting_timeweighted_{suffix}'] = weighted_quantile(x, w, q)
    f['E20776__active_setting_timeweighted_iqr'] = weighted_quantile(x, w, .75) - weighted_quantile(x, w, .25)
    f['E20776__active_setting_timeweighted_mean'] = float(np.average(x, weights=w)) if len(x) else np.nan
    f['E20776__setting_known_active_s'] = known_active_s
    f['E20776__scaler_active_s'] = active_s
    f['E20776__setting_known_active_fraction'] = known_active_fraction
    f['E20776__setting_known_case_fraction'] = setting_known_s / duration
    # Correct 20101 meanings. Only explicit 1 -> 0 starts an uncensored pickup.
    t, end, sv = syringe
    known_s = float((end-t)[np.isfinite(sv)].sum())
    pickup_s = float((end-t)[sv == 0].sum())
    pickup_count = 0
    return_count = 0
    completed = []
    pickup_start = None
    left_censored_pickup = bool(len(sv) and sv[0] == 0)
    for i in range(1, len(t)):
        previous, current = sv[i-1], sv[i]
        if not (np.isfinite(previous) and np.isfinite(current)):
            pickup_start = None
            continue
        if previous == 1 and current == 0:
            pickup_count += 1
            pickup_start = t[i]
        elif previous == 0 and current == 1:
            return_count += 1
            if pickup_start is not None:
                completed.append(t[i] - pickup_start)
            pickup_start = None
    f['E20101__pickup_fraction'] = pickup_s / known_s if known_s else np.nan
    f['E20101__pickup_per_min'] = pickup_count / (known_s / 60) if known_s else np.nan
    f['E20101__return_per_min'] = return_count / (known_s / 60) if known_s else np.nan
    for suffix, q in [('q50', .5), ('q90', .9)]:
        f[f'E20101__pickup_complete_episode_{suffix}_s'] = float(np.quantile(completed, q)) if completed else np.nan
    f['E20101__pickup_complete_episode_n'] = len(completed)
    audit = dict(case_id=record['case_id'], duration_s=duration,
        setting_record_count=int((events.event_code == '20776').sum()),
        scaler_record_count=int((events.event_code == '20774').sum()),
        scaler_known_s=scaler_known_s, scaler_active_s=active_s,
        setting_known_s=setting_known_s, setting_known_active_s=known_active_s,
        setting_known_active_fraction=known_active_fraction,
        setting_unknown_active_s=active_s-known_active_s if np.isfinite(active_s) else np.nan,
        active_setting_timeweighted_q50=f['E20776__active_setting_timeweighted_q50'],
        active_setting_timeweighted_mean=f['E20776__active_setting_timeweighted_mean'],
        syringe_known_s=known_s, syringe_pickup_s=pickup_s,
        syringe_pickup_count=pickup_count, syringe_return_count=return_count,
        syringe_complete_pickup_episodes=len(completed),
        syringe_left_censored_pickup=left_censored_pickup,
        syringe_right_censored_pickup=bool(len(sv) and sv[-1] == 0),
        source_sha256=sha256(source))
    profiles = []
    for progress in range(10):
        lo, hi = duration * progress/10, duration * (progress+1)/10
        iw = np.maximum(0, np.minimum(breaks[1:], hi) - np.maximum(breaks[:-1], lo))
        good = known_active & (iw > 0)
        as_ = float(iw[active].sum())
        ks = float(iw[known_active].sum())
        qt = weighted_quantile(values[good], iw[good], .5)
        pickup_overlap = np.maximum(0, np.minimum(end, hi) - np.maximum(t, lo))
        sk = float(pickup_overlap[np.isfinite(sv)].sum())
        sp = float(pickup_overlap[sv == 0].sum())
        profiles.append(dict(case_id=record['case_id'], progress_bin=progress,
            scaler_setting_timeweighted_q50=qt, scaler_active_s=as_,
            scaler_setting_known_active_s=ks,
            scaler_setting_known_active_fraction=ks/as_ if as_ else np.nan,
            syringe_pickup_fraction=sp/sk if sk else np.nan))
    return dict(case_id=record['case_id'], features=f, audit=audit, profiles=profiles)


def make_catalog():
    rows = []
    for suffix in ['q10', 'q50', 'q90', 'iqr', 'mean']:
        rows.append(dict(feature=f'E20776__active_setting_timeweighted_{suffix}',
            block='scaler_numeric', view='levels', code='20776',
            label=f'スケーラー作動中の出力設定・時間加重{suffix}（次の変更まで有効）'))
    for suffix, label in [('setting_known_active_s', '出力設定既知の作動時間（秒）'),
                          ('scaler_active_s', '作動状態1の記録保持時間（秒）'),
                          ('setting_known_active_fraction', '作動時間における設定既知率'),
                          ('setting_known_case_fraction', '症例全時間における設定既知率')]:
        rows.append(dict(feature=f'E20776__{suffix}', block='quality', view='quality',
                         code='20776', label=label))
    for suffix, label in [('pickup_fraction', '取り上げ状態0の時間割合（既知時間分母・噴射時間ではない）'),
                          ('pickup_per_min', '格納1→取り上げ0の遷移/既知時間分'),
                          ('return_per_min', '取り上げ0→格納1の遷移/既知時間分'),
                          ('pickup_complete_episode_q50_s', '両端観測された取り上げ区間長中央値（秒）'),
                          ('pickup_complete_episode_q90_s', '両端観測された取り上げ区間長q90（秒）')]:
        rows.append(dict(feature=f'E20101__{suffix}', block='syringe', view='operations',
                         code='20101', label=f'シリンジ: {label}'))
    rows.append(dict(feature='E20101__pickup_complete_episode_n', block='quality',
                     view='quality', code='20101', label='シリンジ: 両端観測された取り上げ区間数'))
    return pd.DataFrame(rows)


def semantic_catalog(base):
    catalog = base.copy()
    # Historical columns remain accessible but cannot silently enter v2 model views.
    old_scaler = (catalog.code == '20776') & catalog.view.isin(['levels', 'coarse', 'conditional'])
    catalog.loc[old_scaler, 'view'] = 'audit'
    catalog.loc[old_scaler, 'label'] = '旧・設定変更イベント観測値の集約（作動時間分布ではない）: ' + catalog.loc[old_scaler, 'label']
    old_syringe = catalog.feature.isin(['E20101__active_fraction', 'E20101__state0_fraction', 'E20101__state1_fraction',
        'E20101__activation_per_min', 'E20101__episode_q50_s', 'E20101__episode_q90_s'])
    catalog.loc[old_syringe, 'view'] = 'audit'
    catalog.loc[catalog.feature == 'E20101__state0_fraction', 'label'] = 'シリンジ: 取り上げ状態0割合（pickup_fractionの旧別名）'
    catalog.loc[catalog.feature == 'E20101__state1_fraction', 'label'] = 'シリンジ: 格納状態1割合（取り上げ割合の補数）'
    catalog.loc[catalog.feature == 'E20101__active_fraction', 'label'] = '旧非0割合: シリンジ格納状態1の割合（作動・噴射ではない）'
    catalog.loc[catalog.feature == 'E20101__activation_per_min', 'label'] = '旧0→1回数: シリンジ返却・格納回数/分'
    for suffix in ['q50', 'q90']:
        catalog.loc[catalog.feature == f'E20101__episode_{suffix}_s', 'label'] = f'旧非0区間: シリンジ格納区間長{suffix}（秒）'
    for third in [1, 2, 3]:
        catalog.loc[catalog.feature == f'E20101__third{third}_active_fraction', 'label'] = f'シリンジ: 進捗第{third}三分区間の格納状態1割合'
        catalog.loc[catalog.feature == f'E20101__third{third}_active_fraction', 'view'] = 'audit'
    return pd.concat([catalog, make_catalog()], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    base = out.parent
    out.mkdir(parents=True, exist_ok=True)
    (out/'process').mkdir(exist_ok=True)
    process_path = out/'process/extraction_log.jsonl'

    def log(event, **details):
        with process_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(timestamp_utc=now(), event=event, **details), ensure_ascii=False) + '\n')

    required = ['features.csv', 'feature_catalog.csv', 'case_manifest.csv', 'coarse_profiles.csv']
    base_hashes = {name: sha256(base/name) for name in required}
    cases = pd.read_csv(base/'case_manifest.csv')
    assert len(cases) == cases.case_id.nunique() == 300
    records = cases.to_dict('records')
    log('refinement_preflight', version=VERSION, requested_cases=300, input_cases=len(cases),
        workers=args.workers, base_hashes=base_hashes,
        reason='User confirmed change-only 20776 setting persistence and 20101 pickup=0/stored=1',
        old_assumptions='20776 event-bin levels and generic 20101 nonzero activation are retained only as audit')
    print('[PREFLIGHT] 300 cases; new semantics; v1 artifacts read-only.', flush=True)
    if args.smoke:
        results = [refine_case(records[i]) for i in [0, 150, 299]]
        log('refinement_smoke_pass', n_cases=len(results))
        print('[SMOKE] Three cases passed.', flush=True)
        return
    names = ['features.csv','feature_catalog.csv','case_manifest.csv','setting_time_audit.csv',
             'semantic_metadata.json','coarse_profiles.csv']
    if any((out/name).exists() for name in names):
        raise FileExistsError('Refusing to overwrite an existing v2 extraction.')
    start = time.time()
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(refine_case, record): record['case_id'] for record in records}
        for future in as_completed(pending):
            cid = pending[future]
            try:
                results.append(future.result())
                log('case_refined', case_id=cid, n_complete=len(results))
            except Exception as exc:
                failures.append(dict(case_id=cid, exception=type(exc).__name__, reason=str(exc)))
                log('case_failure', **failures[-1])
            if (len(results)+len(failures)) % 50 == 0:
                print(f'[REFINE] success={len(results)} failed={len(failures)} / 300', flush=True)
    if failures:
        log('refinement_failed', failures=failures)
        raise RuntimeError(f'{len(failures)} failures, see process log')
    lookup_results = {r['case_id']:r for r in results}
    assert len(lookup_results) == 300
    ordered = [lookup_results[cid] for cid in cases.case_id]
    new = pd.DataFrame([dict(case_id=r['case_id'], **r['features']) for r in ordered])
    old = pd.read_csv(base/'features.csv')
    features = old.merge(new, on='case_id', validate='one_to_one')
    catalog = semantic_catalog(pd.read_csv(base/'feature_catalog.csv', dtype={'code':str}))
    audit = pd.DataFrame([r['audit'] for r in ordered])
    assert np.array_equal(audit.case_id, cases.case_id)
    assert np.array_equal(audit.source_sha256, cases.source_sha256), 'Source events changed since v1'
    profiles = pd.read_csv(base/'coarse_profiles.csv', dtype={'code':str})
    for r in ordered:
        for entry in r['profiles']:
            select = (profiles.case_id == r['case_id']) & (profiles.progress_bin == entry['progress_bin'])
            scaler_row = select & profiles.code.eq('20776')
            profiles.loc[scaler_row, 'raw_median'] = entry['scaler_setting_timeweighted_q50']
            profiles.loc[scaler_row, 'known_fraction'] = entry['scaler_setting_known_active_fraction']
            profiles.loc[scaler_row, 'setting_known_active_s'] = entry['scaler_setting_known_active_s']
            profiles.loc[scaler_row, 'scaler_active_s'] = entry['scaler_active_s']
            profiles.loc[select & profiles.code.eq('20101'), 'pickup_fraction'] = entry['syringe_pickup_fraction']
    profiles['value_semantics'] = 'v1 observed-bin numeric median or stored-state model'
    profiles.loc[profiles.code.eq('20776'), 'value_semantics'] = 'timeweighted setting median during known scaler-active intervals'
    profiles.loc[profiles.code.eq('20101'), 'value_semantics'] = 'active_fraction=stored(1); pickup_fraction=picked_up(0), not spraying'
    cases['feature_version'] = VERSION
    assert len(features) == features.case_id.nunique() == 300
    assert not catalog.feature.duplicated().any()
    assert set(features.columns)-{'case_id'} == set(catalog.feature)
    untouched = old.columns.drop('case_id')
    assert old[untouched].equals(features[untouched]), 'v1 feature values unexpectedly altered'
    assert np.allclose(features.E20101__pickup_fraction, features.E20101__state0_fraction, equal_nan=True)
    assert np.allclose(features.E20101__return_per_min, features.E20101__activation_per_min, equal_nan=True)
    assert (audit.setting_known_active_s <= audit.scaler_active_s.fillna(0) + 1e-8).all()
    no_target = cases.signal_status.eq('NO_TARGET_SIGNAL')
    semantic_features = catalog.loc[catalog.view.isin(['levels','operations','conditional','coarse']), 'feature']
    assert features.loc[no_target, semantic_features].isna().all().all()
    for name, frame in [('features.csv',features),('feature_catalog.csv',catalog),
                        ('case_manifest.csv',cases),('setting_time_audit.csv',audit),
                        ('coarse_profiles.csv',profiles)]:
        frame.to_csv(out/name, index=False)
    assert all(sha256(base/name) == expected for name,expected in base_hashes.items()), 'v1 source altered'
    metadata = dict(version=VERSION, created_utc=now(), requested_cases=300, actual_cases=300,
        cases_with_target_signal=296, no_target_signal_cases=4,
        no_target_signal_case_ids=cases.loc[no_target,'case_id'].tolist(),
        user_confirmed_definitions={'20776':'Change-only output setting, valid until next record',
                                   '20101':{'0':'syringe picked up','1':'syringe stored',
                                            'caution':'pickup duration is not spray duration'}},
        reason_for_new_version='v1 treated 20776 changes as observed-value levels and 20101 nonzero as generic activation',
        boundary='unchanged: first 90012 ON to last 90012 OFF; features on [0,duration)',
        missing_policy='unknown before first explicit state/setting; nonfinite/invalid state makes unknown; no missing-to-zero imputation',
        setting_summary='exact union of 20776 setting and 20774 state timestamps; timeweighted summaries only in state20774=1 with known setting',
        weighted_quantile_definition='inverse empirical duration-weighted CDF, left threshold',
        syringe_summary='state0 pickup fractions; 1-to-0 pickups and 0-to-1 returns per known minute; only explicitly bounded pickup intervals',
        historical_columns='v1 values retained; obsolete 20776 level/coarse/conditional and 20101 aliases moved to audit view',
        other_numeric='unchanged observed 1-second-bin summaries; persistence not assumed without user confirmation',
        coarse_20776='raw_median now duration-weighted active-setting median; n_observed retains v1 recorded-bin count as audit',
        coarse_20101='active_fraction retains stored state1; new pickup_fraction is state0, not spraying',
        feature_count=len(catalog), feature_views=catalog.groupby('view').size().to_dict(),
        expected_primary_model_views=['levels','operations'],
        complement_warning='20101 state0/state1 fractions are retained only in audit to avoid duplicate/complement weighting against pickup_fraction',
        base_source_hashes=base_hashes, base_sources_unchanged=True,
        event_sources_match_v1_sha256=True,
        script_source=str(Path(__file__).resolve()),script_sha256=sha256(__file__),
        workers=args.workers, chunk_rows=20000, columns_read=['time_s','event_code','event_value'],
        codes_read=sorted(READ_CODES), wide_tables_loaded=False,
        elapsed_seconds=time.time()-start, python=sys.version,
        numpy_version=np.__version__,pandas_version=pd.__version__)
    (out/'semantic_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False),encoding='utf-8')
    log('refinement_complete', actual_cases=300, source_and_v1_hashes_unchanged=True,
        n_features=len(catalog), elapsed_seconds=time.time()-start)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
