"""Independent compact-table audit for the v5 context analysis.

Does not import the extractor; never changes source or previous outputs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score


def review_analysis(output, source):
    f = pd.read_csv(output / 'analysis_case_features.csv').set_index('case_id')
    groups = pd.read_csv(output / 'case_context_groups.csv').set_index('case_id')
    gs = pd.read_csv(output / 'context_group_summary.csv').set_index('group_id')
    expected = 'HP3_' + f.hp3_active_s.ge(2).astype(int).astype(str) + '__MID_' + f.n_mid_dominant_blocks.ge(1).astype(int).astype(str)
    assert expected.equals(groups.group_id.rename(None)), 'rule definitions'
    assert groups.group_id.value_counts().sort_index().equals(gs.n_cases.sort_index().rename('count'))
    members = pd.read_csv(output / 'candidate_members.csv')
    candidates = pd.read_csv(output / 'candidate_summary.csv')
    assert not members.duplicated(['candidate', 'case_id']).any()
    for r in candidates.itertuples():
        ids = members.loc[members.candidate.eq(r.candidate), 'case_id']
        assert len(ids) == r.n_cases and r.denominator == 97
        assert np.isclose(r.percent, 100 * len(ids) / 97)
        assert r.ci_low <= r.percent <= r.ci_high
    d = pd.read_csv(output / 'context_distances.csv', index_col=0).loc[f.index, f.index]
    matrix = d.to_numpy()
    assert np.isfinite(matrix).all() and np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0) and matrix.min() >= 0 and matrix.max() <= 1
    ass = pd.read_csv(output / 'context_cluster_assignments.csv')
    trials = pd.read_csv(output / 'context_case_subsampling.csv')
    models = pd.read_csv(output / 'context_model_comparison.csv')
    assert len(ass) == 485 and not ass.duplicated(['k', 'case_id']).any()
    assert len(trials) == 300 and not trials.duplicated(['k', 'repeat']).any()
    assert trials.ari.between(-1, 1).all() and trials.n_cases.eq(78).all()
    for r in models.itertuples():
        labels = ass[ass.k.eq(r.k)].set_index('case_id').loc[f.index, 'cluster']
        t = trials[trials.k.eq(r.k)]
        assert len(t) == 60 and labels.nunique() == r.k
        assert set(ass.loc[ass.k.eq(r.k), 'case_id']) == set(f.index)
        assert np.isclose(silhouette_score(matrix, labels, metric='precomputed'), r.silhouette)
        assert np.isclose(adjusted_rand_score(groups.group_id, labels), r.rule_cross_ari)
        assert np.isclose(t.ari.median(), r.ari_median)
        assert np.isclose(t.ari.quantile(.25), r.ari_q25)
        assert np.isclose(t.ari.quantile(.75), r.ari_q75)
        assert labels.value_counts().min() == r.min_group
        assert labels.value_counts().max() == r.max_group
    rp = pd.read_csv(output / 'representatives.csv')
    for key, reps in rp.groupby('candidate'):
        ids = groups.index[groups.group_id.eq(key)] if key.startswith('HP3_') else members.loc[members.candidate.eq(key), 'case_id']
        med = reps.loc[reps.role.eq('medoid'), 'case_id']
        assert len(med) == 1 and reps.case_id.is_unique and set(reps.case_id) <= set(ids)
        costs = d.loc[ids, ids].sum(axis=1)
        assert np.isclose(costs.loc[med.iloc[0]], costs.min())
        assert np.allclose(reps.distance_to_medoid, d.loc[med.iloc[0], reps.case_id])
    raw = pd.read_csv(source / 'hp2_points.csv', usecols=['case_id', 'zone', 'torque', 'pedal1', 'pedal2', 'bulk'])
    levels = pd.read_csv(output / 'raw_numeric_by_case_zone.csv')
    raw = raw[raw.case_id.isin(f.index)]
    data = dict(tuple(raw.groupby(['case_id', 'zone'])))
    for r in levels.itertuples():
        x = data[(r.case_id, r.zone)][r.signal].dropna()
        assert len(x) == r.n_bins
        assert np.isclose(x.median(), r.median) if len(x) >= 3 else pd.isna(r.median)
    quality = json.loads((output / 'context_quality_audit.json').read_text())
    assert quality['known80_cases'] == int(f.hp_known_fraction.ge(.8).sum())
    assert quality['known50_cases'] == int(f.hp_known_fraction.ge(.5).sum())
    audit = {'status': 'passed', 'primary_cases': len(f), 'rule_cross_counts': groups.group_id.value_counts().sort_index().to_dict(),
             'candidate_count': len(candidates), 'exploratory_assignment_rows': len(ass),
             'case_subsampling_rows': len(trials), 'recomputed_silhouettes': len(models),
             'raw_numeric_case_zone_signal_rows_checked': len(levels),
             'medoid_scopes_checked': rp.candidate.nunique(),
             'ramp_only_cases_with_macro_mid': int((f.ramp_only_observed_mid & f.macro_mid.eq(1)).sum()),
             'note': 'Trial ARIs and quantiles verified against stored case-subsampling table; individual refits were not independently rerun.'}
    (output / 'process/final_analysis_review.json').write_text(json.dumps(audit, indent=2) + '\n')
    return audit


def reference(source):
    manifest = pd.read_csv(source / 'case_manifest.csv', usecols=['case_id'])
    p = pd.read_csv(source / 'hp2_points.csv', usecols=[
        'case_id', 'run_id', 'time_s', 'rpm', 'weight_s', 'run_start_s', 'run_end_s'])
    runs = pd.read_csv(source / 'hp_runs.csv')
    audit = pd.read_csv(source / 'hp_status_audit.csv', usecols=[
        'case_id', 'duration_s', 'hp_known_s', 'hp1_s', 'hp3_s'])
    blocks = pd.read_csv(source / 'hp2_blocks.csv', usecols=[
        'case_id', 'supported', 'mid_fraction', 'high_dominant'])
    assert len(manifest) == manifest.case_id.nunique() == 300
    assert not p.duplicated(['case_id', 'run_id', 'time_s']).any()
    assert np.isfinite(p.weight_s).all() and (p.weight_s > 0).all()
    high = p[p.rpm.between(36000, 44000)]
    mid = p[(p.rpm > 5000) & (p.rpm < 36000)]
    pos = p[p.rpm > 0]
    out = manifest.merge(audit, on='case_id', validate='one_to_one').set_index('case_id')
    for name, part in [('high', high), ('mid', mid), ('positive', pos)]:
        grouped = part.groupby('case_id').agg(n=('rpm', 'size'), s=('weight_s', 'sum'))
        out[name + '_n'] = grouped.n.reindex(out.index, fill_value=0)
        out[name + '_proxy_s'] = grouped.s.reindex(out.index, fill_value=0)
    out['high_any'] = out.high_n > 0
    out['high_main'] = (out.high_n >= 3) & (out.high_proxy_s >= 2)
    out['mid_supported'] = (out.mid_n >= 3) & (out.mid_proxy_s >= 2)
    out['hp3_seen'] = out.hp3_s > 0
    out['hp1_seen'] = out.hp1_s > 0
    out['hp_known_fraction'] = out.hp_known_s / out.duration_s
    for name, selected in [('old_high_dominant', blocks.high_dominant),
                           ('mid_dominant_block', blocks.supported & (blocks.mid_fraction >= .6))]:
        out[name] = out.index.isin(blocks.loc[selected, 'case_id'])
    interior = mid[(mid.time_s - mid.run_start_s > 3) & (mid.run_end_s - mid.time_s > 3)]
    out['mid_interior_any'] = out.index.isin(interior.case_id)
    q = mid.groupby(['case_id', 'run_id']).agg(n=('rpm', 'size'), s=('weight_s', 'sum'))
    total = pos.groupby(['case_id', 'run_id']).weight_s.sum()
    q['mid_fraction'] = q.s / total
    q = q.reset_index().merge(runs, on=['case_id', 'run_id'], validate='one_to_one')
    q['coverage'] = (q.s / q.mid_fraction) / q.active_s
    run_ok = ((q.n >= 3) & (q.s >= 2) & (q.mid_fraction >= .6)
              & (q.active_s >= 2) & (q.coverage >= .4))
    out['mid_dominant_run'] = out.index.isin(q.loc[run_ok, 'case_id'])
    assert out.high_any.sum() == 99 and out.high_main.sum() == 97
    assert out.old_high_dominant.sum() == 67
    assert out.loc[out.old_high_dominant, 'high_main'].all()
    return out.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('Output/chair_hp_sequence_300_v3_20261104'))
    parser.add_argument('--output', type=Path, default=Path('Output/chair_context_v5_20261104'))
    args = parser.parse_args()
    out = reference(args.source)
    process = args.output / 'process'
    process.mkdir(parents=True, exist_ok=True)
    out.to_csv(process / 'independent_reference.csv', index=False)
    mainrows = out[out.high_main]
    summary = {'requested_cases': len(out), 'high_any': int(out.high_any.sum()),
               'high_main': int(out.high_main.sum()), 'old_high_dominant': int(out.old_high_dominant.sum()),
               'main_counts': {col: int(mainrows[col].sum()) for col in [
                   'hp3_seen', 'hp1_seen', 'mid_supported', 'mid_interior_any',
                   'mid_dominant_run', 'mid_dominant_block']},
               'main_known_hp_fraction_min': float(mainrows.hp_known_fraction.min()),
               'main_known_hp_fraction_median': float(mainrows.hp_known_fraction.median()),
               'source_only_checks': 'passed'}
    extracted = pd.read_csv(args.output / 'case_context_features.csv')
    scope = pd.read_csv(args.output / 'cohort_scope.csv')
    assert len(extracted) == 300 and extracted.case_id.is_unique
    assert len(scope) == 300 and scope.case_id.is_unique
    assert set(extracted.case_id) == set(scope.case_id) == set(out.case_id)
    joined = extracted.set_index('case_id').join(out.set_index('case_id'), rsuffix='_reference')
    comparisons = [('high_supported', 'high_main'), ('any_high_observed', 'high_any'),
                   ('hp3_observed', 'hp3_seen'), ('hp1_observed', 'hp1_seen'),
                   ('old_high_dominant', 'old_high_dominant_reference'),
                   ('mid_supported', 'mid_supported_reference'),
                   ('n_mid_dominant_runs', 'mid_dominant_run'),
                   ('n_mid_dominant_blocks', 'mid_dominant_block'),
                   ('n_mid_interior_runs', 'mid_interior_any')]
    for observed, expected in comparisons:
        assert ((joined[observed] > 0) == (joined[expected] > 0)).all(), observed
    for observed, expected in [('high_n_bins', 'high_n'), ('high_equivalent_s', 'high_proxy_s'),
                               ('mid_n_bins', 'mid_n'), ('mid_equivalent_s', 'mid_proxy_s'),
                               ('hp_known_fraction', 'hp_known_fraction_reference')]:
        valid = joined[observed].notna()
        assert np.allclose(joined.loc[valid, observed], joined.loc[valid, expected]), observed
    assert scope.set_index('case_id').included_context_high.equals(
        out.set_index('case_id').high_main), 'cohort inclusion mismatch'
    source_manifest = pd.read_csv(args.output / 'source_manifest.csv')
    for record in source_manifest.itertuples():
        assert hashlib.sha256(Path(record.path).read_bytes()).hexdigest() == record.sha256
    events = pd.read_csv(args.output / 'context_events.csv', low_memory=False)
    transitions = pd.read_csv(args.output / 'context_transitions.csv')
    assert not events.duplicated(['case_id', 'token_id']).any()
    assert (events.end_s >= events.start_s).all()
    assert (events.active_s >= 0).all()
    assert (events.active_s <= events.end_s - events.start_s + 1e-8).all()
    assert not transitions.loc[transitions.confident_order, ['from_kind', 'to_kind']].isin(['UNKNOWN']).any().any()
    assert transitions.loc[transitions.confident_order, 'unknown_gap_s'].eq(0).all()
    run_events = events[(events.unit_level == 'run') & (events.kind != 'UNKNOWN')]
    source_runs = pd.read_csv(args.source / 'hp_runs.csv')
    assert len(run_events) == len(source_runs)
    numeric = pd.read_csv(args.source / 'hp2_points.csv', usecols=[
        'case_id', 'run_id', 'rpm', 'weight_s', 'second_bin', 'run_start_s', 'run_end_s'])
    numeric['absolute_start'] = np.maximum(numeric.second_bin, numeric.run_start_s)
    numeric['absolute_end'] = np.minimum(numeric.second_bin + 1, numeric.run_end_s)
    numeric['zone'] = np.select([
        numeric.rpm.eq(0), numeric.rpm.gt(0) & numeric.rpm.le(5000),
        numeric.rpm.gt(5000) & numeric.rpm.lt(36000), numeric.rpm.between(36000, 44000),
        numeric.rpm.gt(44000)], ['Z', 'L', 'M', 'H', 'X'], default='UNKNOWN')
    grouped_numeric = dict(tuple(numeric.groupby(['case_id', 'run_id'])))
    covered_points = 0
    for episode in events[(events.unit_level == 'observed_regime') & events.kind.isin(['Z', 'L', 'M', 'H', 'X'])].itertuples():
        samples = grouped_numeric[(episode.case_id, int(float(episode.run_ids)))]
        within = samples[(samples.absolute_start >= episode.start_s - 1e-8)
                         & (samples.absolute_end <= episode.end_s + 1e-8)]
        assert len(within) == int(episode.n_bins), ('episode time/count', episode.token_id)
        assert within.zone.eq(episode.kind).all(), ('episode raw RPM zone', episode.token_id)
        assert np.isclose(within.weight_s.sum(), episode.active_s), ('episode proxy weight', episode.token_id)
        covered_points += len(within)
    assert covered_points == len(numeric), 'Every original RPM bin belongs to exactly one actual-time episode'
    for level, group in events.groupby('unit_level'):
        if level == 'observed_regime':
            continue
        for _, case in group.groupby('case_id'):
            ordered = case.sort_values('start_s')
            assert (ordered.start_s.to_numpy()[1:] >= ordered.end_s.to_numpy()[:-1] - 1e-8).all()
    summary['extracted_feature_comparisons'] = len(comparisons) + 5
    summary['source_hashes_verified'] = len(source_manifest)
    summary['event_rows'] = len(events)
    summary['transition_rows'] = len(transitions)
    summary['raw_points_matched_to_actual_time_episodes'] = covered_points
    summary['v5_output_checks'] = 'passed'
    (process / 'independent_review.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if (args.output / 'context_quality_audit.json').exists():
        print(json.dumps(review_analysis(args.output, args.source), indent=2))


if __name__ == '__main__':
    main()
