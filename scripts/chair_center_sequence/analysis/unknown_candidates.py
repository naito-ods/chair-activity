"""Attach unconfirmed center-model candidates without changing unknown status."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from chair_center_sequence.workflow import save_json, sha


def vote(labels, weights):
    totals = {}
    for label, weight in zip(labels, weights):
        totals[label] = totals.get(label, 0.) + float(weight)
    return sorted(totals, key=lambda label: (-totals[label], label))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--fullfit-dir", default="fullfit")
    parser.add_argument("--oof-dir", default="primary_oof")
    parser.add_argument("--out-prefix", default="")
    args = parser.parse_args()
    root, canonical = Path(args.root), Path(args.canonical)
    output = root / f"{args.out_prefix}unknown_interval_candidates.csv"
    if output.exists():
        raise FileExistsError(output)
    source_hash = sha(canonical)
    a = pd.read_csv(canonical, dtype={"seq": str}, keep_default_na=False)
    original = a.copy(deep=True)
    full = pd.read_csv(root / args.fullfit_dir / "inference.csv.gz", dtype={"case_id": str}, keep_default_na=False)
    oof = pd.read_csv(root / args.oof_dir / "predictions.csv.gz", dtype={"case_id": str}, keep_default_na=False)
    if full.id.tolist() != oof.id.tolist():
        raise ValueError("Fullfit and OOF center IDs disagree")
    rows = []
    for r in a[a.treatment == "unknown"].itertuples():
        ix = np.flatnonzero(full.case_id.eq(r.seq).to_numpy() & full.center_s.ge(r.start_s).to_numpy() & full.center_s.lt(r.end_s).to_numpy())
        candidate = oof_candidate = ""
        top3 = []
        supported = 0.
        if len(ix):
            q = full.iloc[ix]
            weight = np.maximum(0., np.minimum(q.cell_end_s.to_numpy(float), r.end_s) - np.maximum(q.cell_start_s.to_numpy(float), r.start_s))
            supported = float(weight.sum())
            candidate = vote(q.sequence_candidate, weight)[0]
            top3 = vote(q.local_candidate, weight)[:3]
            oof_candidate = vote(oof.sequence_candidate.iloc[ix], weight)[0]
        rows.append({"id": r.id, "seq": r.seq, "start_s": r.start_s, "end_s": r.end_s, "duration_s": r.duration_s,
                     "source_treatment": r.treatment, "source_annotation_status": r.annotation_status, "unknown_reason": r.unknown_reason,
                     "center_sequence_candidate": candidate, "local_top3_by_vote": "|".join(top3),
                     "oof_sequence_candidate": oof_candidate, "candidate_differs_fullfit_vs_oof": candidate != oof_candidate,
                     "centers_inside_interval": len(ix), "grid_supported_s": supported,
                     "grid_supported_fraction": min(1., supported / r.duration_s),
                     "requires_human_review": True, "candidate_human_confirmed": False,
                     "no_center_inside_interval": not len(ix), "temporal_resolution_s": 1.,
                     "interpretation": "fullfit candidate, not clinical truth; vote coverage is not confidence"})
    proposals = pd.DataFrame(rows)
    proposals.to_csv(output, index=False)
    joined = a.merge(proposals[["id", "center_sequence_candidate", "oof_sequence_candidate", "requires_human_review",
                              "candidate_human_confirmed", "no_center_inside_interval"]], on="id", how="left", validate="one_to_one")
    joined.to_csv(root / f"{args.out_prefix}annotations_with_unknown_and_center_candidates.csv", index=False)
    pd.testing.assert_frame_equal(joined[a.columns], original)
    if sha(canonical) != source_hash:
        raise ValueError("Source canonical table changed")
    save_json(root / "process" / f"{args.out_prefix}unknown_candidate_audit.json", {"source_sha256": source_hash, "canonical_rows": len(a),
        "unknown_rows": len(proposals), "rows_with_center_support": int((proposals.centers_inside_interval > 0).sum()),
        "rows_without_center_support": int(proposals.no_center_inside_interval.sum()),
        "source_columns_preserved_exactly": True, "unknown_clinical_accuracy_available": False,
        "all_candidates_require_human_review": True, "fullfit_is_not_heldout_evaluation": True})


if __name__ == "__main__":
    main()
