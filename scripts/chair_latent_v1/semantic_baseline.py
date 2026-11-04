"""Visible-numeric HP/context baseline for annotation-free latent experiments.

All exact HP state intervals, HP runs, and real case durations are allowed
conditioning information. ``visible_bins`` hides NUMERIC observations only.
No precomputed full-data numeric flags/regions/contexts are used. Each numeric
code is masked using its OWN observation timestamp (``rpm_time_s``,
``torque_time_s``, ``pedal2_time_s``), not the RPM representative time of its
joined row. A hidden value cannot affect any numeric statistic, band, block dominance or ordering
feature. Unknown HP intervals remain barriers. Numeric observed-order motifs
also reset at hidden bins or numeric observation gaps longer than two seconds.

This is a leave-parts-out adaptation of the v3/v5 semantic feature families,
not a claim to reproduce their full-data assignments: support and dominance
are recomputed after masking with fixed, previously documented thresholds.
Numeric summaries retain original units; unavailable values stay NaN. A zero
count/flag means not observed in the visible portion, not clinical absence.
Columns prefixed ``quality_`` separately expose support/observation quantities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _weighted_quantile(values, weights, quantile):
    x = np.asarray(values, float)
    w = np.asarray(weights, float)
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not good.any():
        return np.nan
    x, w = x[good], w[good]
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    index = np.searchsorted(np.cumsum(w), quantile * w.sum(), side="left")
    return float(x[min(index, len(x) - 1)])


def _overlap(unknown, start, end):
    if not len(unknown) or end <= start:
        return 0.0
    return float(np.maximum(0, np.minimum(unknown[:, 1], end) -
                            np.maximum(unknown[:, 0], start)).sum())


def _blocks(runs, unknown):
    """Fixed v3 rule: same HP, known OFF gap <=30s; never cross unknown."""
    result, run_map = [], {}
    previous = None
    for r in runs.sort_values("start_s").itertuples():
        gap = r.start_s - previous.end_s if previous is not None else np.nan
        unknown_gap = _overlap(unknown, previous.end_s, r.start_s) if previous is not None else np.nan
        merge = bool(result and int(r.hp) == result[-1]["hp"] and
                     gap <= 30 and unknown_gap <= 1e-8)
        if not merge:
            result.append(dict(hp=int(r.hp), start_s=float(r.start_s), end_s=float(r.end_s),
                               active_s=0.0, n_runs=0, previous_hp=float(previous.hp) if previous is not None else np.nan,
                               unknown_gap_s=unknown_gap))
        block = result[-1]
        run_map[int(r.run_id)] = (len(result)-1, block["active_s"], float(r.start_s))
        block["active_s"] += float(r.end_s-r.start_s)
        block["end_s"] = float(r.end_s)
        block["n_runs"] += 1
        previous = r
    return result, run_map


def _empty(frame, required):
    return pd.DataFrame(columns=required) if frame is None else frame


def _case_features(duration, intervals, runs, points, visible, bins):
    f = {"case_duration_s": float(duration)}
    intervals = intervals.sort_values("start_s")
    runs = runs.sort_values("start_s")
    h = pd.to_numeric(intervals.hp, errors="coerce").to_numpy(float)
    widths = (intervals.end_s-intervals.start_s).to_numpy(float)
    known = np.isfinite(h)
    known_s = float(widths[known].sum())
    unknown = intervals.loc[~known, ["start_s", "end_s"]].to_numpy(float)
    active_s = float(widths[known & (h > 0)].sum())
    f["quality_hp_known_fraction"] = known_s/duration
    f["quality_numeric_visible_fraction"] = float(np.mean(visible))
    for hp in range(4):
        own = float(widths[known & (h == hp)].sum())
        f[f"hp{hp}_known_share"] = own/known_s if known_s > 0 else np.nan
        f[f"hp{hp}_case_share"] = own/duration if known_s > 0 else np.nan
        if hp:
            q = runs[runs.hp.eq(hp)]
            lengths = (q.end_s-q.start_s).to_numpy(float)
            f[f"hp{hp}_observed_2s"] = float(own >= 2) if known_s > 0 else np.nan
            f[f"hp{hp}_active_share"] = own/active_s if active_s > 0 else np.nan
            f[f"hp{hp}_n_runs"] = float(len(q)) if known_s > 0 else np.nan
            for quantile in [.5, .9]:
                f[f"hp{hp}_run_q{int(quantile*100)}_s"] = float(np.quantile(lengths, quantile)) if len(lengths) else np.nan
            f[f"hp{hp}_first_progress"] = float(q.start_s.min()/duration) if len(q) else np.nan
            f[f"hp{hp}_last_progress"] = float(q.end_s.max()/duration) if len(q) else np.nan
    # Known HP state is allowed throughout. Its transitions do not depend on
    # any numeric data, nor on an inferred clinical class.
    transition_counts = {(a, b): 0 for a in [1, 2, 3] for b in [1, 2, 3]}
    rr = list(runs.itertuples())
    for a, b in zip(rr[:-1], rr[1:]):
        if _overlap(unknown, a.end_s, b.start_s) <= 1e-8:
            transition_counts[(int(a.hp), int(b.hp))] += 1
    for (a, b), count in transition_counts.items():
        f[f"hp_transition_{a}_{b}_count"] = float(count) if known_s > 0 else np.nan
    blocks, run_map = _blocks(runs, unknown)
    hp2_s = float(widths[known & (h == 2)].sum())
    p = points.copy()
    timestamp_columns = [name+"_time_s" for name in ["rpm", "torque", "pedal2"]]
    supplied = [name in p for name in timestamp_columns]
    if any(supplied) and not all(supplied):
        raise ValueError("Supply all three per-code timestamps, or none for explicitly co-timed synthetic/legacy input")
    for name in ["rpm", "torque", "pedal2"]:
        if name not in p:
            p[name] = np.nan
        p[name] = pd.to_numeric(p[name], errors="coerce")
        # Numeric values in the same (run, integer-second) joined row can have
        # different representative times and therefore different mask bins.
        # Mask each code independently BEFORE any numeric summaries/regions.
        time_column = name+"_time_s" if all(supplied) else "time_s"
        times = pd.to_numeric(p[time_column], errors="coerce").to_numpy(float)
        valid_time = np.isfinite(times) & (times >= 0) & (times < duration)
        which = np.zeros(len(p), dtype=int)
        which[valid_time] = np.minimum((times[valid_time]/duration*bins).astype(int), bins-1)
        keep = valid_time & visible[which]
        p.loc[~keep, name] = np.nan
    # Dropping fully hidden rows also ensures their metadata cannot influence
    # support/order. Rows with a visible non-RPM code survive for its marginal
    # statistics, but hidden RPM cannot define a band or paired band statistic.
    p = p.loc[p[["rpm", "torque", "pedal2"]].notna().any(axis=1)].copy()
    if all(supplied):
        p["time_s"] = pd.to_numeric(p.rpm_time_s, errors="coerce")
    p = p.sort_values("time_s")
    if "weight_s" not in p:
        p["weight_s"] = 1.0
    p["weight_s"] = pd.to_numeric(p.weight_s, errors="coerce")
    # Only points attached to a known HP2 run can enter numeric HP2 evidence.
    if len(p):
        valid_run = p.run_id.map(lambda x: np.isfinite(x) and int(x) in run_map and blocks[run_map[int(x)][0]]["hp"] == 2)
        p = p.loc[valid_run & p.weight_s.gt(0)].copy()
    p["zone_visible"] = np.select(
        [p.rpm.eq(0), p.rpm.gt(0)&p.rpm.le(5000), p.rpm.gt(5000)&p.rpm.lt(36000),
         p.rpm.ge(36000)&p.rpm.le(44000), p.rpm.gt(44000)],
        ["ZERO", "LOW", "MID", "HIGH", "ABOVE"], default="UNKNOWN")
    positive = p[p.rpm.gt(0)]
    positive_s = float(positive.weight_s.sum())
    rpm = p[p.rpm.notna()]
    rpm_s = float(rpm.weight_s.sum())
    f["quality_rpm_visible_n"] = float(len(rpm))
    f["quality_rpm_visible_coverage"] = rpm_s/hp2_s if hp2_s > 0 else np.nan
    f["rpm_zero_fraction"] = float(p.loc[p.zone_visible.eq("ZERO"), "weight_s"].sum()/rpm_s) if rpm_s > 0 else np.nan
    for signal in ["rpm", "torque", "pedal2"]:
        good = p[signal].notna()
        f[f"quality_{signal}_visible_n"] = float(good.sum())
        f[f"quality_{signal}_visible_s"] = float(p.loc[good, "weight_s"].sum())
        for quantile in [.1, .5, .9]:
            f[f"{signal}_q{int(quantile*100)}"] = _weighted_quantile(p[signal], p.weight_s, quantile)
        f[f"{signal}_iqr"] = _weighted_quantile(p[signal], p.weight_s, .75)-_weighted_quantile(p[signal], p.weight_s, .25)
        for zone in ["LOW", "MID", "HIGH"]:
            q = p[p.zone_visible.eq(zone)]
            f[f"{signal}_{zone.lower()}_q50"] = _weighted_quantile(q[signal], q.weight_s, .5) if q[signal].notna().sum() >= 3 else np.nan
    for zone in ["LOW", "MID", "HIGH", "ABOVE"]:
        q = positive[positive.zone_visible.eq(zone)]
        key = zone.lower()
        f[f"{key}_fraction"] = float(q.weight_s.sum()/positive_s) if positive_s > 0 else np.nan
        f[f"{key}_supported"] = float(len(q) >= 3 and q.weight_s.sum() >= 2) if positive_s > 0 else np.nan
        f[f"{key}_first_progress"] = float(q.time_s.min()/duration) if len(q) else np.nan
        f[f"{key}_last_progress"] = float(q.time_s.max()/duration) if len(q) else np.nan
        f[f"{key}_weighted_progress"] = float(np.average(q.time_s/duration, weights=q.weight_s)) if len(q) else np.nan
        f[f"quality_{key}_visible_n"] = float(len(q))
    high = p[p.zone_visible.eq("HIGH")]
    first_h = float(high.time_s.min()) if len(high) else np.nan
    last_h = float(high.time_s.max()) if len(high) else np.nan
    hp3 = runs[runs.hp.eq(3)]
    before = hp3[hp3.end_s < first_h] if len(high) else hp3.iloc[:0]
    after = hp3[hp3.start_s > last_h] if len(high) else hp3.iloc[:0]
    f["hp3_before_first_high"] = float(any(_overlap(unknown, r.end_s, first_h) <= 1e-8 for r in before.itertuples())) if len(high) else np.nan
    f["hp3_after_last_high"] = float(any(_overlap(unknown, last_h, r.start_s) <= 1e-8 for r in after.itertuples())) if len(high) else np.nan
    between = hp3[(hp3.start_s > first_h)&(hp3.end_s < last_h)] if len(high) else hp3.iloc[:0]
    f["high_hp3_high"] = float(len(between)>0 and _overlap(unknown, first_h, last_h)<=1e-8) if len(high) else np.nan
    # Recompute block support/dominance using ONLY visible numeric evidence.
    block_points = {}
    for r in p.itertuples():
        if np.isfinite(r.run_id) and int(r.run_id) in run_map:
            block_points.setdefault(run_map[int(r.run_id)][0], []).append(r)
    summaries = []
    run_geometry = {int(r.run_id): (float(r.start_s), float(r.end_s))
                    for r in runs.itertuples()}
    for index, b in enumerate(blocks):
        if b["hp"] != 2:
            continue
        q = block_points.get(index, [])
        pos = [r for r in q if np.isfinite(r.rpm) and r.rpm > 0]
        total = sum(r.weight_s for r in pos)
        frac = {z: sum(r.weight_s for r in pos if r.zone_visible == z)/total if total else np.nan for z in ["LOW", "MID", "HIGH"]}
        # v3 support for primary post-HP3 and high-dominance evidence.
        supported = len(pos)>=8 and total/b["active_s"]>=.4 and b["active_s"]>=10
        def active_clock(row):
            _, offset, run_start = run_map[int(row.run_id)]
            return offset+row.time_s-run_start
        early = [r for r in pos if active_clock(r)/b["active_s"]<=.3]
        early_s = sum(r.weight_s for r in early)
        initial_low = sum(r.weight_s for r in early if r.zone_visible=="LOW")/early_s if early_s else np.nan
        observed = sum(r.weight_s for r in q if np.isfinite(r.rpm))
        zero_fraction = sum(r.weight_s for r in q if r.zone_visible=="ZERO")/observed if observed else np.nan
        mid_tail = False
        if pos and pos[-1].zone_visible=="MID" and any(r.zone_visible=="HIGH" for r in pos):
            last_high_row = max((r for r in pos if r.zone_visible=="HIGH"), key=active_clock)
            last_high_clock = active_clock(last_high_row)
            tail = [r for r in pos if active_clock(r)>last_high_clock and r.zone_visible=="MID"]
            if tail:
                def active_edges(row):
                    _, offset, run_start = run_map[int(row.run_id)]
                    _, run_end = run_geometry[int(row.run_id)]
                    second = float(np.floor(row.time_s))
                    return offset+max(second,run_start)-run_start, offset+min(second+1,run_end)-run_start
                tail_start = min(active_edges(r)[0] for r in tail)
                tail_end = max(active_edges(r)[1] for r in tail)
                prefix = [r for r in pos if active_clock(r)<min(active_clock(t) for t in tail)]
                prefix_s = sum(r.weight_s for r in prefix)
                high_s = sum(r.weight_s for r in prefix if r.zone_visible=="HIGH")
                first_bin = max(0,min(bins-1,int(last_high_row.time_s/duration*bins)))
                last_bin = max(0,min(bins-1,int(max(r.time_s for r in tail)/duration*bins)))
                # A masked gap must not manufacture an observed HIGH->MID tail.
                path_visible = bool(visible[first_bin:last_bin+1].all())
                mid_tail = bool(supported and path_visible and b["active_s"]>=30 and
                                frac["HIGH"]>=.6 and high_s>=15 and prefix_s>0 and
                                high_s/prefix_s>=.8 and tail_start/b["active_s"]>=.7 and
                                (tail_end-tail_start)/b["active_s"]<=.25 and len(tail)>=3 and
                                sum(r.weight_s for r in tail)>=2 and
                                b["active_s"]-tail_end<=max(2,.05*b["active_s"]))
        # v5 macroblock flags reused the v3 supported blocks at >=60% dominance.
        summaries.append(dict(**b, supported=supported, fractions=frac,
                              post_hp3=bool(b["previous_hp"]==3 and b["unknown_gap_s"]<=1e-8),
                              initial_low_fraction=initial_low, zero_fraction=zero_fraction,
                              high_then_short_mid=mid_tail))
    supported = [b for b in summaries if b["supported"]]
    supported_any = bool(supported)
    f["quality_n_supported_hp2_blocks"] = float(len(supported))
    f["post_hp3_low_dominant"] = float(any(b["post_hp3"] and b["fractions"]["LOW"]>=.7 and b["initial_low_fraction"]>=.7 for b in supported)) if supported_any else np.nan
    f["high_dominant_observed"] = float(any(b["fractions"]["HIGH"]>=.8 and b["zero_fraction"]<=.1 for b in supported)) if supported_any else np.nan
    f["high_then_short_mid"] = float(any(b["high_then_short_mid"] for b in supported)) if supported_any else np.nan
    for zone in ["LOW", "MID", "HIGH"]:
        dominant = [b for b in supported if b["fractions"][zone]>=.6]
        key = zone.lower()
        f[f"n_{key}_dominant_blocks"] = float(len(dominant)) if supported_any else np.nan
        for placement in ["before", "between", "after"]:
            if placement == "before":
                answer = any(b["end_s"] < first_h for b in dominant)
            elif placement == "after":
                answer = any(b["start_s"] > last_h for b in dominant)
            else:
                answer = any(b["start_s"] > first_h and b["end_s"] < last_h for b in dominant)
            f[f"macro_{key}_{placement}_high"] = float(answer) if len(high) and supported_any else np.nan
    # Pointwise ordered motifs are descriptive only. Hidden numeric bins,
    # unknown HP intervals, and >2s observation gaps break point-to-point paths.
    pairs = [("LOW", "HIGH"), ("HIGH", "MID"), ("HIGH", "LOW"), ("MID", "HIGH")]
    pair_counts = {pair: 0 for pair in pairs}
    previous = None
    for r in p.itertuples():
        if previous is not None:
            a, b = previous.time_s, r.time_s
            first_bin = max(0, min(bins-1, int(a/duration*bins)))
            last_bin = max(0, min(bins-1, int(b/duration*bins)))
            connected = b-a<=2 and visible[first_bin:last_bin+1].all() and _overlap(unknown, a, b)<=1e-8
            pair = (previous.zone_visible, r.zone_visible)
            if connected and pair in pair_counts:
                pair_counts[pair] += 1
        previous = r
    for (a, b), count in pair_counts.items():
        f[f"visible_{a.lower()}_to_{b.lower()}_count"] = float(count) if positive_s>0 else np.nan
    return f


def semantic_features(manifest, intervals, runs, points, visible_bins, bins=48):
    """Return ``(features, names)`` in exact manifest row order.

    ``visible_bins`` is bool ``(len(manifest), bins)`` and affects NUMERIC
    points only, separately by each code's own timestamp. Actual cohort callers
    must supply ``rpm_time_s``, ``torque_time_s``, and ``pedal2_time_s`` from
    original per-code observations, joined on case/run/second-bin. Missing
    timestamp means unavailable evidence for that code. For explicitly co-timed
    synthetic/legacy tables only, omission of ALL three uses ``time_s``.
    Full HP states and durations are shared conditioning data. Inputs are never
    modified. Missing numeric fields retain NaN.
    """
    if manifest.case_id.duplicated().any():
        raise ValueError("Duplicate case_id in manifest")
    visible_bins = np.asarray(visible_bins, bool)
    if visible_bins.shape != (len(manifest), bins):
        raise ValueError("visible_bins shape does not match manifest and bins")
    if not np.all(np.isfinite(manifest.duration_s)) or (manifest.duration_s <= 0).any():
        raise ValueError("Case durations must be finite and positive")
    grouped = [{key: value for key, value in frame.groupby("case_id", sort=False)}
               for frame in [intervals, runs, points]]
    empty_i = intervals.iloc[:0]
    empty_r = runs.iloc[:0]
    empty_p = points.iloc[:0]
    rows = []
    for index, row in enumerate(manifest.itertuples()):
        rows.append(_case_features(float(row.duration_s), grouped[0].get(row.case_id, empty_i),
                                  grouped[1].get(row.case_id, empty_r), grouped[2].get(row.case_id, empty_p),
                                  visible_bins[index], bins))
    result = pd.DataFrame(rows)
    return result.to_numpy(float), result.columns.tolist()
