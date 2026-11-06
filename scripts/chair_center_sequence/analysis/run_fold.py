"""A fixed held-out fold, with class-independent SGD work scheduled on threads."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import chair_center_sequence.model as module
from chair_center_sequence.data import boolean
from chair_center_sequence.sequence import ProcedurePrior
from chair_center_sequence.workflow import code_fingerprints, save_json, sha, versions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--rocket-jobs", type=int, default=2)
    parser.add_argument("--sgd-jobs", type=int, default=3)
    args = parser.parse_args()
    root, output = Path(args.prepared), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((root / "config.json").read_text())
    if config["code_sha256"] != code_fingerprints():
        raise ValueError("Prepared code changed")
    for relative, expected in json.loads((root / "prepared_fingerprints.json").read_text()).items():
        if sha(root / relative) != expected:
            raise ValueError(f"Prepared artifact changed: {relative}")
    meta = pd.read_csv(root / "centers.csv.gz", dtype={"case_id": str}, keep_default_na=False)
    meta["teacher"] = meta.teacher.map(boolean)
    a = pd.read_csv(root / "annotations.csv", dtype={"case_id": str}, keep_default_na=False)
    coverage = pd.read_csv(root / "case_coverage.csv", dtype={"case_id": str})
    summaries, tensors, ids, names = [], [], [], None
    for case in coverage.case_id:
        with np.load(root / "cases" / f"{case}.npz", allow_pickle=False) as saved:
            if names is not None and names != saved["summary_names"].tolist():
                raise ValueError("Feature schema mismatch")
            names = saved["summary_names"].tolist()
            summaries.append(saved["summary"]); tensors.append(saved["tensor"]); ids.extend(saved["ids"].tolist())
    if ids != meta.id.tolist():
        raise ValueError("Center order mismatch")
    summary, tensor = np.concatenate(summaries), np.concatenate(tensors)
    train = np.flatnonzero((meta.fold != args.fold) & meta.teacher)
    query = np.flatnonzero(meta.fold == args.fold)
    training_cases = sorted(meta.loc[meta.fold != args.fold, "case_id"].unique())
    validation_cases = sorted(meta.loc[meta.fold == args.fold, "case_id"].unique())
    if set(training_cases) & set(validation_cases) or not len(query) or not len(train):
        raise ValueError("Invalid fixed fold")
    # Numerical equivalence to the serial estimator must be independently
    # verified before using this scheduling-only override. It changes neither
    # class seed assignment, center order, epochs nor optimizer parameters.
    original = module.SGDClassifier
    def threaded_sgd(**kwargs):
        kwargs["n_jobs"] = args.sgd_jobs
        return original(**kwargs)
    module.SGDClassifier = threaded_sgd
    output.mkdir(parents=True)
    print(json.dumps({"stage": "fixed_fold_start", "fold": args.fold, "train_centers": len(train),
                      "rocket_jobs": args.rocket_jobs, "sgd_jobs": args.sgd_jobs}), flush=True)
    local = module.LocalModel(10000, 256, 10, 20261106, args.rocket_jobs, args.cache_dir)
    local.fit(summary[train], meta.iloc[train], tensor[train])
    prior = ProcedurePrior(.2).fit(a, local.classes, training_cases, coverage.set_index("case_id").raw_duration_s.to_dict())
    p = local.probabilities(summary[query], tensor[query])
    np.savez_compressed(output / "scores.npz", indices=query, ids=meta.id.iloc[query].to_numpy(str),
                        classes=np.array(local.classes), probabilities=p)
    bundle = {"local": local, "sequence": prior, "outer_fold": args.fold, "validation_cases": validation_cases,
              "prepared_config": config, "summary_names": names, "versions": versions(),
              "compute": {"rocket_jobs": args.rocket_jobs, "sgd_jobs": args.sgd_jobs, "numerical_parameters_changed": False}}
    with (output / "model.pkl").open("wb") as stream:
        pickle.dump(bundle, stream)
    save_json(output / "audit.json", {"fold": args.fold, "training_cases": training_cases, "validation_cases": validation_cases,
        "teacher_cases": local.training_cases, "training_centers": len(train), "query_centers": len(query),
        "classes": local.classes, "duration_support_cases": prior.duration_support_cases,
        "mean_duration_s": prior.mean_duration_s.tolist(), "sequence_parent_audit": prior.audit,
        "query_annotations_used_as_features": False, "compute": bundle["compute"]})
    print(json.dumps({"stage": "fixed_fold_complete", "fold": args.fold}), flush=True)


if __name__ == "__main__":
    main()
