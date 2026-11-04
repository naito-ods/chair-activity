"""Leakage, unknown-state, and raw-unit checks for the semantic comparator."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from semantic_baseline import semantic_features


def synthetic_inputs():
    manifest = pd.DataFrame([dict(case_id="synthetic", duration_s=48.)])
    states = [(0., 2., 0.), (2., 8., 3.), (8., 10., 0.),
              (10., 35., 2.), (35., 40., 0.), (40., 43., 3.), (43., 48., 0.)]
    intervals = pd.DataFrame([dict(case_id="synthetic", interval_id=i, start_s=a, end_s=b,
                                   hp=hp, known=True, duration_s=b-a)
                              for i, (a, b, hp) in enumerate(states)])
    runs = intervals[intervals.hp.gt(0)].copy()
    runs["run_id"] = np.arange(1, len(runs)+1)
    runs["active_s"] = runs.duration_s
    hp2run = int(runs.loc[runs.hp.eq(2), "run_id"].iloc[0])
    times = np.arange(10.5, 35., 1.)
    rpm = np.where(times<18., 1000., np.where(times<30., 40000., 20000.))
    points = pd.DataFrame(dict(case_id="synthetic", time_s=times, rpm=rpm,
                               torque=2.+np.arange(len(times))*.01, pedal2=100.,
                               weight_s=1., run_id=hp2run))
    return manifest, intervals, runs, points


def main():
    m, intervals, runs, points = synthetic_inputs()
    original = points.copy(deep=True)
    visible = np.ones((1, 48), bool)
    visible[:, 18:24] = False
    x, names = semantic_features(m, intervals, runs, points, visible)
    hidden = points.time_s.ge(18)&points.time_s.lt(24)
    corrupt = points.copy()
    for col in ["rpm", "torque", "pedal2", "weight_s", "run_id"]:
        corrupt.loc[hidden, col] = 99999999.
    changed, changed_names = semantic_features(m, intervals, runs, corrupt, visible)
    assert names == changed_names
    np.testing.assert_allclose(x, changed, equal_nan=True)
    pd.testing.assert_frame_equal(points, original)
    f = dict(zip(names, x[0]))
    assert f["rpm_q90"] == 40000.  # Raw RPM preserved, not standardized here.
    assert f["quality_rpm_visible_n"] == 19.
    assert f["hp3_before_first_high"] == 1.
    assert f["hp3_after_last_high"] == 1.
    assert f["hp3_case_share"] == 9./48.
    # Hiding every numeric observation must leave the exact HP conditioning
    # unchanged, while unavailable numeric observations remain missing.
    blank, blank_names = semantic_features(m, intervals, runs, points, np.zeros((1, 48), bool))
    b = dict(zip(blank_names, blank[0]))
    assert np.isnan(b["rpm_q50"]) and np.isnan(b["low_fraction"])
    assert b["quality_rpm_visible_n"] == 0.
    assert b["hp3_case_share"] == f["hp3_case_share"]
    # Recorded zero is a distinct numeric observation, not missingness.
    zeros = points.copy()
    zeros["rpm"] = 0.
    zero_x, _ = semantic_features(m, intervals, runs, zeros, np.ones((1,48),bool))
    z = dict(zip(names, zero_x[0]))
    assert z["rpm_q50"] == 0. and z["rpm_zero_fraction"] == 1.
    assert np.isnan(z["low_fraction"])
    # Unknown HP time between HP3 and numeric HIGH blocks a confident ordering.
    unknown = intervals.copy()
    unknown.loc[unknown.start_s.eq(8.), "hp"] = np.nan
    unknown.loc[unknown.start_s.eq(8.), "known"] = False
    u, _ = semantic_features(m, unknown, runs, points, visible)
    uf = dict(zip(names,u[0]))
    assert uf["hp3_before_first_high"] == 0.
    assert uf["quality_hp_known_fraction"] < f["quality_hp_known_fraction"]
    assert not np.isinf(x).any()
    # Real boundary regression: joined RPM at9.9s is visible while torque at
    #10.1s is hidden. Values in one integer-second joined row are not co-timed.
    bm = pd.DataFrame([dict(case_id="boundary", duration_s=20.)])
    bi = pd.DataFrame([dict(case_id="boundary", start_s=0., end_s=20., hp=2., known=True)])
    br = bi.assign(run_id=1, active_s=20.)
    bp = pd.DataFrame([dict(case_id="boundary", run_id=1, time_s=9.9,
                            rpm_time_s=9.9, torque_time_s=10.1, pedal2_time_s=9.8,
                            rpm=40000., torque=123., pedal2=100., weight_s=1.)])
    bx, bn = semantic_features(bm, bi, br, bp, np.array([[True,False]]), bins=2)
    bf = dict(zip(bn,bx[0]))
    assert bf["rpm_q50"]==40000. and bf["pedal2_q50"]==100.
    assert np.isnan(bf["torque_q50"]) and bf["quality_torque_visible_n"]==0.
    bad = bp.copy(); bad["torque"] = 999999999.
    altered, _ = semantic_features(bm, bi, br, bad, np.array([[True,False]]), bins=2)
    np.testing.assert_allclose(bx, altered, equal_nan=True)
    # Reverse boundary: keep visible torque even when joined RPM is hidden.
    reverse = bp.copy()
    reverse["rpm_time_s"] = 10.1; reverse["torque_time_s"] = 9.9
    rx, rn = semantic_features(bm, bi, br, reverse, np.array([[True,False]]), bins=2)
    rf = dict(zip(rn,rx[0]))
    assert np.isnan(rf["rpm_q50"]) and rf["torque_q50"]==123.
    assert np.isnan(rf["high_fraction"])  # Hidden RPM cannot establish a band.
    reverse_bad = reverse.copy(); reverse_bad["rpm"] = 999999999.
    raltered, _ = semantic_features(bm, bi, br, reverse_bad, np.array([[True,False]]), bins=2)
    np.testing.assert_allclose(rx, raltered, equal_nan=True)
    # Partial timing metadata cannot silently fall back to the RPM timestamp.
    try:
        semantic_features(bm, bi, br, bp.drop(columns="torque_time_s"), np.array([[True,False]]), bins=2)
    except ValueError:
        pass
    else:
        raise AssertionError("Partial code-specific timestamp metadata must fail")
    print(json.dumps(dict(tests_passed=10, n_features=len(names),
                          hidden_numeric_content_invariant=True,
                          full_hp_conditioning_retained=True,
                          zero_distinct_from_missing=True,
                          unknown_order_barrier=True,
                          input_frames_unchanged=True,
                          raw_units_preserved=True,
                          per_code_timestamp_boundary_invariant=True,
                          visible_non_rpm_retained_when_rpm_hidden=True,
                          partial_timestamp_metadata_rejected=True)))


if __name__ == "__main__":
    main()
