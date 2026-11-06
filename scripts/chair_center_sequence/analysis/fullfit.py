"""Refit a prespecified model on all known centers; never score training accuracy."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from chair_center_sequence.data import boolean
from chair_center_sequence.model import LocalModel
from chair_center_sequence.sequence import ProcedurePrior
from chair_center_sequence.workflow import code_fingerprints, save_json, sha, versions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--num-kernels", type=int, default=10000)
    args = parser.parse_args()
    root, output = Path(args.prepared), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((root / "config.json").read_text())
    if config["code_sha256"] != code_fingerprints():
        raise ValueError("Prepared code fingerprint mismatch")
    for relative, expected in json.loads((root / "prepared_fingerprints.json").read_text()).items():
        if sha(root / relative) != expected:
            raise ValueError(f"Prepared artifact changed: {relative}")
    meta = pd.read_csv(root / "centers.csv.gz", dtype={"case_id": str}, keep_default_na=False)
    meta["teacher"] = meta.teacher.map(boolean)
    annotations = pd.read_csv(root / "annotations.csv", dtype={"case_id": str}, keep_default_na=False)
    coverage = pd.read_csv(root / "case_coverage.csv", dtype={"case_id": str})
    summaries, tensors, ids, names = [], [], [], None
    for case in coverage.case_id:
        with np.load(root / "cases" / f"{case}.npz", allow_pickle=False) as saved:
            if names is not None and names != saved["summary_names"].tolist():
                raise ValueError("Summary schema mismatch")
            names = saved["summary_names"].tolist()
            summaries.append(saved["summary"])
            tensors.append(saved["tensor"])
            ids.extend(saved["ids"].tolist())
    if ids != meta.id.tolist():
        raise ValueError("Prepared row order mismatch")
    summary, tensor = np.concatenate(summaries), np.concatenate(tensors)
    train = np.flatnonzero(meta.teacher)
    output.mkdir(parents=True)
    print(json.dumps({"stage": "fullfit_start", "cases": meta.case_id.nunique(), "teachers": len(train),
                      "scored_as_validation": False}), flush=True)
    model = LocalModel(args.num_kernels, 256, 10, 20261106, args.n_jobs, args.cache_dir)
    model.fit(summary[train], meta.iloc[train], tensor[train] if args.num_kernels else None)
    prior = ProcedurePrior(.2).fit(annotations, model.classes, sorted(meta.case_id.unique()),
                                    coverage.set_index("case_id").raw_duration_s.to_dict())
    bundle = {"local": model, "sequence": prior, "prepared_config": config,
              "training_case_ids": sorted(meta.case_id.unique()), "summary_names": names,
              "versions": versions(), "role": "all-case fit; not validation"}
    with (output / "full_model.pkl").open("wb") as stream:
        pickle.dump(bundle, stream)
    p = model.probabilities(summary, tensor if args.num_kernels else None)
    predictions = meta.copy()
    predictions["local_candidate"] = np.array(model.classes)[p.argmax(axis=1)]
    predictions["sequence_candidate"] = ""
    for _, group in meta.groupby("case_id", sort=False):
        ix = group.index.to_numpy()
        predictions.loc[ix, "sequence_candidate"] = prior.decode(p[ix], group.center_s)
    predictions["human_confirmed"] = False
    predictions["inference_role"] = "full-cohort annotation candidate; not held-out evaluation"
    predictions.to_csv(output / "inference.csv.gz", index=False, compression="gzip")
    np.savez_compressed(output / "scores.npz", ids=meta.id.to_numpy(str), classes=np.array(model.classes), probabilities=p)
    save_json(output / "model_card.json", {"training_cases": len(bundle["training_case_ids"]),
        "training_case_ids": bundle["training_case_ids"], "teacher_centers": len(train),
        "teacher_parents": int(meta.loc[train, "annotation_id"].nunique()), "classes": model.classes,
        "annotation_source_preserved": True, "training_accuracy_reported": False,
        "predictions_are_human_confirmed": False, "prepared_seal_sha256": sha(root / "prepared_fingerprints.json"),
        "bundle_sha256": sha(output / "full_model.pkl"), "versions": versions()})
    print(json.dumps({"stage": "fullfit_complete", "model": str(output / "full_model.pkl"),
                      "cases": len(bundle["training_case_ids"]), "classes": len(model.classes)}), flush=True)


if __name__ == "__main__":
    main()
