"""Explicit prepare/evaluate commands; importing this module starts no analysis."""

import argparse
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from . import __version__
from .data import attach_teachers, center_grid, normalize, parent_case_weights
from .features import CHANNELS, CaseLog
from .model import LocalModel
from .sequence import ProcedurePrior


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def versions():
    result = {"method": __version__}
    for name in ["numpy", "pandas", "scikit-learn", "sktime", "numba"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def code_fingerprints():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def prepare(args):
    """Adapt previously aligned caches, verifying source annotations and raw SHA."""
    paths = {name: Path(getattr(args, name)).resolve() for name in
             ["aligned", "manifest", "folds", "events_dir", "annotation_source"]}
    # Hash before the first read, then verify again after extraction. Otherwise
    # an edit after loading metadata could attach a new hash to old contents.
    fingerprints = {str(p): sha(p) for name, p in paths.items() if name != "events_dir"}
    manifest = pd.read_csv(paths["manifest"], dtype={"seq": str}, keep_default_na=False)
    folds = pd.read_csv(paths["folds"], dtype={"seq": str}, keep_default_na=False)
    aligned = pd.read_csv(paths["aligned"], dtype={"seq": str}, keep_default_na=False)
    source = pd.read_csv(paths["annotation_source"], dtype={"seq": str}, keep_default_na=False)
    if not manifest.seq.is_unique or not folds.seq.is_unique or not aligned.id.is_unique or not source.id.is_unique:
        raise ValueError("Duplicate case, fold or source annotation IDs")
    cases = set(manifest.seq)
    if set(folds.seq) != cases or not set(aligned.seq).issubset(cases) or set(aligned.id) != set(source.id):
        raise ValueError("Case/annotation manifests do not agree")
    if folds.fold.nunique() < 2 or folds.fold.isna().any():
        raise ValueError("Need at least two complete case-level folds")
    fold_values = pd.to_numeric(folds.fold, errors="raise").to_numpy(float)
    if not np.isfinite(fold_values).all() or (fold_values < 0).any() or (fold_values % 1 != 0).any():
        raise ValueError("Fold IDs must be finite nonnegative integers")
    folds["fold"] = fold_values.astype(int)
    merged = aligned.set_index("id").loc[source.id]
    if merged.seq.tolist() != source.seq.tolist() or merged.treatment_normalized.map(normalize).tolist() != source.treatment.map(normalize).tolist():
        raise ValueError("Aligned cache disagrees with original source labels")
    from .data import boolean
    if merged.confirmed.map(boolean).tolist() != source.status.eq("確認済み").tolist():
        raise ValueError("Aligned cache disagrees with original confirmation statuses")
    indexed = manifest.set_index("seq")
    origin = pd.to_datetime(source.seq.map(indexed.raw_start), format="mixed")
    for source_name, aligned_name in [("start_datetime", "start_s"), ("end_datetime", "end_s")]:
        relative = (pd.to_datetime(source[source_name], format="mixed") - origin).dt.total_seconds()
        if not np.allclose(relative, merged[aligned_name].to_numpy(float), atol=1e-6, rtol=0):
            raise ValueError("Aligned timestamps disagree with original source/raw origin")
    a = aligned.rename(columns={"seq": "case_id", "id": "annotation_id", "treatment_normalized": "label"})
    a = a[["case_id", "annotation_id", "start_s", "end_s", "label", "confirmed"]]
    a["confirmed"] = a.confirmed.map(boolean)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Preserve earlier outputs; choose a new preparation directory")
    print(json.dumps({"stage": "prepare", "requested_cases": len(cases), "requested_parents": len(a),
                      "training_started": False}), flush=True)
    output.mkdir(parents=True)
    (output / "cases").mkdir()
    all_meta, all_ledger, inventory = [], [], []
    for row in manifest.sort_values("seq").itertuples():
        duration = float(row.raw_duration_s)
        cache = paths["events_dir"] / f"{row.seq}.csv.gz"
        raw = Path(row.source_path)
        raw_hash = sha(raw)
        if raw_hash != row.source_sha256:
            raise ValueError(f"Raw source hash changed: {row.seq}")
        declared = set(source.loc[source.seq == row.seq, "source_sha256"]) - {""}
        if declared and declared != {raw_hash}:
            raise ValueError(f"Source annotation/raw hash mismatch: {row.seq}")
        fingerprints[str(raw)] = raw_hash
        fingerprints[str(cache)] = sha(cache)
        events = pd.read_csv(cache, dtype={"code": str})
        case = CaseLog(events, duration)
        grid = center_grid(row.seq, duration, args.step_s)
        meta, ledger = attach_teachers(grid, a[a.case_id == row.seq], duration)
        summaries = pd.DataFrame([case.summary(t, tuple(args.widths_s), args.mode) for t in grid.center_s])
        tensors = np.stack([case.tensor(t, args.rocket_width_s, args.mode) for t in grid.center_s])
        file = output / "cases" / f"{row.seq}.npz"
        np.savez_compressed(file, summary=summaries.to_numpy(float), tensor=tensors, ids=grid.id.to_numpy(str),
                            summary_names=summaries.columns.to_numpy(str), channels=np.array(CHANNELS))
        meta["fold"] = folds.set_index("seq").loc[row.seq, "fold"]
        all_meta.append(meta)
        all_ledger.append(ledger)
        inventory.append({"case_id": row.seq, "raw_duration_s": duration, "centers": len(meta),
                          "teacher_centers": int(meta.teacher.sum()), "teacher_parents": meta.loc[meta.teacher, "annotation_id"].nunique(),
                          "covered_raw_s": float((meta.cell_end_s - meta.cell_start_s).sum()),
                          "unscanned_tail_s": float(duration - meta.cell_end_s.max())})
        print(json.dumps({"stage": "prepared_case", "case_id": row.seq, "centers": len(meta),
                          "teachers": int(meta.teacher.sum())}), flush=True)
    for path, before in fingerprints.items():
        if sha(path) != before:
            raise ValueError(f"Input changed during preparation: {path}")
    pd.concat(all_meta, ignore_index=True).to_csv(output / "centers.csv.gz", index=False, compression="gzip")
    pd.concat(all_ledger, ignore_index=True).to_csv(output / "annotation_coverage.csv", index=False)
    pd.DataFrame(inventory).to_csv(output / "case_coverage.csv", index=False)
    a.to_csv(output / "annotations.csv", index=False)
    config = {"version": __version__, "step_s": args.step_s, "widths_s": args.widths_s,
              "rocket_width_s": args.rocket_width_s, "mode": args.mode, "requested_cases": len(cases),
              "requested_parents": len(a), "target": "single confirmed interval at center; half-open bounds; no erosion",
              "unknown": "annotation status only, excluded from clinical teachers",
              "code_sha256": code_fingerprints(), "versions": versions()}
    save_json(output / "config.json", config)
    save_json(output / "input_fingerprints.json", fingerprints)
    # Include the configuration and all prepared tables/tensors in the seal.
    seal = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    save_json(output / "prepared_fingerprints.json", seal)
    return output


def evaluate(args):
    root = Path(args.prepared).resolve()
    config = json.loads((root / "config.json").read_text())
    if config["code_sha256"] != code_fingerprints():
        raise ValueError("Code changed since preparation; prepare a separately versioned dataset")
    for relative, before in json.loads((root / "prepared_fingerprints.json").read_text()).items():
        if sha(root / relative) != before:
            raise ValueError(f"Prepared input changed: {relative}")
    meta = pd.read_csv(root / "centers.csv.gz", dtype={"case_id": str}, keep_default_na=False)
    from .data import boolean
    meta["teacher"] = meta.teacher.map(boolean)
    annotations = pd.read_csv(root / "annotations.csv", dtype={"case_id": str}, keep_default_na=False)
    case_coverage = pd.read_csv(root / "case_coverage.csv", dtype={"case_id": str})
    if meta.id.duplicated().any() or meta.case_id.nunique() != config["requested_cases"]:
        raise ValueError("Prepared case/center count mismatch")
    # Preparation concatenates complete cases in this same deterministic order.
    summaries, tensors, ids, summary_names = [], [], [], None
    for case in case_coverage.case_id:
        with np.load(root / "cases" / f"{case}.npz", allow_pickle=False) as p:
            if summary_names is not None and summary_names != p["summary_names"].tolist():
                raise ValueError("Feature schema differs between cases")
            summary_names = p["summary_names"].tolist()
            summaries.append(p["summary"])
            if args.num_kernels:
                tensors.append(p["tensor"])
            ids.extend(p["ids"].tolist())
    if ids != meta.id.tolist():
        raise ValueError("Tensor/metadata order mismatch")
    summary = np.concatenate(summaries)
    tensor = np.concatenate(tensors) if args.num_kernels else None
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Preserve earlier runs; choose a new evaluation directory")
    output.mkdir(parents=True)
    (output / "models").mkdir()
    predictions = meta.copy()
    predictions["local_candidate"] = ""
    predictions["sequence_candidate"] = ""
    predictions["truth_in_train_classes"] = False
    audit = []
    durations = case_coverage.set_index("case_id").raw_duration_s.to_dict()
    print(json.dumps({"stage": "evaluate", "requested_cases": config["requested_cases"],
                      "requested_parents": config["requested_parents"], "known_centers": int(meta.teacher.sum())}), flush=True)
    for fold in sorted(meta.fold.unique()):
        train_cases = sorted(meta.loc[meta.fold != fold, "case_id"].unique())
        validation_cases = sorted(meta.loc[meta.fold == fold, "case_id"].unique())
        tr = np.flatnonzero((meta.fold != fold) & meta.teacher)
        va = np.flatnonzero(meta.fold == fold)
        if set(train_cases) & set(validation_cases) or not len(tr):
            raise ValueError("Case leakage or empty training fold")
        model = LocalModel(args.num_kernels, args.batch_size, args.epochs, args.seed, args.n_jobs)
        model.fit(summary[tr], meta.iloc[tr], tensor[tr] if args.num_kernels else None)
        prior = ProcedurePrior(args.sequence_strength).fit(annotations, model.classes, train_cases, durations)
        p = model.probabilities(summary[va], tensor[va] if args.num_kernels else None)
        predictions.loc[va, "local_candidate"] = np.asarray(model.classes)[p.argmax(axis=1)]
        predictions.loc[va, "truth_in_train_classes"] = meta.target.iloc[va].isin(model.classes).to_numpy()
        for case in validation_cases:
            positions = np.flatnonzero(meta.case_id.iloc[va].to_numpy() == case)
            dest = va[positions]
            predictions.loc[dest, "sequence_candidate"] = prior.decode(p[positions], meta.center_s.iloc[dest])
        bundle = {"local": model, "sequence": prior, "outer_fold": int(fold), "validation_cases": validation_cases,
                  "prepared_config": config, "summary_names": summary_names, "versions": versions()}
        with (output / "models" / f"fold{fold}.pkl").open("wb") as stream:
            pickle.dump(bundle, stream)
        audit.append({"fold": int(fold), "training_cases": train_cases, "validation_cases": validation_cases,
                      "teacher_cases": model.training_cases, "training_centers": len(tr), "query_centers": len(va),
                      "classes": model.classes, "duration_support_cases": prior.duration_support_cases,
                      "mean_duration_s": prior.mean_duration_s.tolist(), "sequence_parent_audit": prior.audit,
                      "query_annotations_used_as_features": False})
        print(json.dumps({"stage": "completed_fold", "fold": int(fold), "train_cases": len(train_cases),
                          "validation_cases": len(validation_cases), "train_centers": len(tr)}), flush=True)
    if predictions.local_candidate.eq("").any() or predictions.sequence_candidate.eq("").any():
        raise ValueError("Missing prediction rows")
    predictions["human_confirmed"] = False
    predictions.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    result, per_label, per_parent = [], [], []
    scored = predictions[predictions.teacher].copy()
    w = parent_case_weights(scored)
    truth = scored.target.to_numpy(str)
    labels = sorted(set(truth))
    for variant in ["local_candidate", "sequence_candidate"]:
        guessed = scored[variant].to_numpy(str)
        result.append({"variant": variant, "case_parent_weighted_accuracy": float(np.average(truth == guessed, weights=w)),
                       "case_parent_weighted_macro_f1": float(f1_score(truth, guessed, labels=labels, average="macro", sample_weight=w, zero_division=0)),
                       "raw_center_accuracy": float(np.mean(truth == guessed)), "scored_cases": int(scored.case_id.nunique()),
                       "scored_parents": int(scored.annotation_id.nunique()), "scored_centers": len(scored),
                       "unseen_truth_centers": int((~scored.truth_in_train_classes).sum())})
        precision, recall, f1, support = precision_recall_fscore_support(truth, guessed, labels=labels, zero_division=0)
        per_label.extend({"variant": variant, "label": label, "precision": float(p), "recall": float(r), "f1": float(f),
                          "center_support": int(n)} for label, p, r, f, n in zip(labels, precision, recall, f1, support))
        for (case, parent), group in scored.groupby(["case_id", "annotation_id"]):
            correct = group[variant].to_numpy() == group.target.to_numpy()
            per_parent.append({"variant": variant, "case_id": case, "annotation_id": parent, "label": group.target.iloc[0],
                               "centers": len(group), "correct_centers": int(correct.sum()), "accuracy": float(correct.mean()),
                               "any_center_detected": bool(correct.any()), "majority_centers_correct": bool(correct.mean() >= .5)})
    pd.DataFrame(per_label).to_csv(output / "per_label_metrics.csv", index=False)
    pd.DataFrame(per_parent).to_csv(output / "parent_detection.csv", index=False)
    save_json(output / "metrics.json", result)
    save_json(output / "fold_audit.json", audit)
    save_json(output / "run_config.json", {"prepared": str(root), "prepared_seal_sha256": sha(root / "prepared_fingerprints.json"),
                                          "parameters": vars(args), "versions": versions(),
                                          "unscored_case_ids": sorted(set(meta.case_id) - set(scored.case_id)),
                                          "scope": "internal grouped evaluation; unknown targets unscored; parent detection is not event-IoU matching",
                                          "confidence": "class-balanced local scores are not calibrated clinical probabilities"})
    return output


def parser():
    p = argparse.ArgumentParser(description="Center-time local logs + procedure sequence; explicit stages only")
    stages = p.add_subparsers(dest="stage", required=True)
    prep = stages.add_parser("prepare", help="Read/verify inputs and extract features; no fitting")
    for name in ["aligned", "manifest", "folds", "events-dir", "annotation-source", "output"]:
        prep.add_argument("--" + name, required=True)
    prep.add_argument("--step-s", type=float, default=1.0)
    prep.add_argument("--widths-s", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    prep.add_argument("--rocket-width-s", type=int, default=30)
    prep.add_argument("--mode", choices=["centered", "past"], default="centered")
    run = stages.add_parser("evaluate", help="Fit/evaluate ONLY when explicitly invoked")
    run.add_argument("--prepared", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--num-kernels", type=int, default=10000, help="0 enables a summary-only ablation")
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--epochs", type=int, default=10)
    run.add_argument("--seed", type=int, default=20261106)
    run.add_argument("--n-jobs", type=int, default=1)
    run.add_argument("--sequence-strength", type=float, default=.2)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    prepare(args) if args.stage == "prepare" else evaluate(args)


if __name__ == "__main__":
    main()
