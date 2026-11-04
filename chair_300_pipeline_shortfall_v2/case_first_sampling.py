"""Case-first activity sampling for chair DTW, without altering segmentation or distance.

One valid activity per input case is reserved BEFORE extra activities are drawn.
If a candidate cannot be represented as a DTW sequence, try the same case's next
candidate.  All selection is deterministic for a given seed and input table.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import json
import numpy as np
import pandas as pd


def choose_case_first(table, target_activities, max_per_case, seed, signature_fn,
                      type_order=(), require_all=True):
    """Return (selected_df, signatures, audit_df, rejected_df, coverage_df).

    signature_fn(row) returns a sequence signature or None. It may raise; errors
    are recorded and the next candidate for that case is tried.
    """
    if table.empty:
        raise ValueError('No activity candidates')
    if 'case_id' not in table or 'activity_id' not in table or 'activity_type' not in table:
        raise ValueError('Missing case_id/activity_id/activity_type in activity table')
    cases = sorted(table.case_id.astype(str).unique())
    if target_activities < len(cases):
        raise ValueError(f'--activities {target_activities} is below {len(cases)} cases. '
                         f'Use --activities >= {len(cases)} to retain one Activity per case.')
    if max_per_case < 1:
        raise ValueError('--max-per-case must be positive')
    rng = np.random.default_rng(seed)
    category_rank = {label: i for i, label in enumerate(type_order)}
    case_arrays = {}
    for cid, group in table.groupby('case_id', sort=True):
        case_arrays[str(cid)] = rng.permutation(group.index.to_numpy()).tolist()

    accepted = []
    features = []
    attempts = []
    selected_set = set()
    per_case = Counter()
    per_type = Counter()
    failures = Counter()
    last_errors = {}
    rejected_set = set()

    def try_one(idx, stage):
        row = table.loc[idx]
        cid = str(row.case_id)
        try:
            sig = signature_fn(row)
            error = '' if sig is not None else 'no_usable_sequence'
        except Exception as exc:
            sig = None
            error = f'{type(exc).__name__}: {str(exc)[:240]}'
        if sig is None:
            failures[cid] += 1
            last_errors[cid] = error
            if idx not in rejected_set:
                attempts.append({'case_id':cid,'activity_id':row.activity_id,
                                 'start_s':row.start_s,'end_s':row.end_s,
                                 'selection_stage':stage,'reason':error})
                rejected_set.add(idx)
            return False
        selected_set.add(idx)
        per_case[cid] += 1
        per_type[str(row.activity_type)] += 1
        item = row.to_dict()
        item['selection_stage'] = stage
        item['selection_rank'] = len(accepted) + 1
        accepted.append(item)
        features.append(sig)
        return True

    # Coverage first: visit each case exactly once and try successive candidates
    # until it contributes one valid activity (or all of its candidates fail).
    for cid in rng.permutation(cases):
        indices = case_arrays[cid]
        # Prefer an underrepresented activity type; randomized tie breaks within
        # categories, and no invented label is used as truth.
        indices = sorted(indices, key=lambda idx: (
            per_type[str(table.loc[idx, 'activity_type'])],
            category_rank.get(str(table.loc[idx, 'activity_type']), 999)
        ))
        for idx in indices:
            if try_one(idx, 'coverage'):
                break

    # Extra budget: round robin over provisional activity types, with per-case
    # caps. No case can receive a second activity until the coverage pass ended.
    remaining = {}
    for activity_type, sub in table.groupby('activity_type', sort=True):
        items = [idx for idx in rng.permutation(sub.index.to_numpy()).tolist()
                 if idx not in selected_set and idx not in rejected_set]
        remaining[str(activity_type)] = items
    keys = sorted(remaining, key=lambda k:(category_rank.get(k, 999), k))
    while len(accepted) < target_activities and any(remaining.values()):
        moved = False
        for kind in keys:
            candidates = remaining[kind]
            while candidates:
                idx = candidates.pop(0)
                cid = str(table.loc[idx, 'case_id'])
                if per_case[cid] >= max_per_case:
                    continue
                if per_case[cid] == 0:
                    # Prevent rescue of failed cases from being counted as a
                    # normal extra. Such cases can be retried here, but failures
                    # were already attempted in coverage pass.
                    stage = 'coverage_retry'
                else:
                    stage = 'extra'
                if try_one(idx, stage):
                    moved = True
                    break
            if len(accepted) >= target_activities:
                break
        if not moved:
            break

    output = pd.DataFrame(accepted).reset_index(drop=True)
    coverage = []
    for cid in cases:
        group = table[table.case_id == cid]
        n = int(per_case[cid])
        coverage.append({'case_id':cid,'n_candidates':len(group),
                         'n_selected':n,'n_signature_failures':failures[cid],
                         'status':'included' if n else 'no_usable_activity',
                         'failure_detail':last_errors.get(cid, '') if n == 0 else '',
                         'selected_activity_ids':json.dumps(output.loc[
                             output.case_id == cid, 'activity_id'].tolist()) if n else '[]'})
    coverage = pd.DataFrame(coverage)
    rejected = pd.DataFrame(attempts, columns=[
        'case_id','activity_id','start_s','end_s','selection_stage','reason'])
    audit = pd.DataFrame([{'activity_type':name,'n_selected':num}
                          for name,num in sorted(per_type.items())],
                         columns=['activity_type','n_selected'])
    missed = coverage[coverage.n_selected == 0]
    if len(missed) and require_all:
        # Caller should SAVE coverage and rejection report before raising.
        pass
    return output, features, audit, rejected, coverage
