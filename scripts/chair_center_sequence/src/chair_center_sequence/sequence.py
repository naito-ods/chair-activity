"""Weak first-order procedure prior with class-specific geometric dwell times.

This is a duration-informed HMM, not an explicit-duration HSMM. It never reads
query-case annotations. No transition is forbidden and no minimum dwell time
is imposed. Unknown annotation gaps are censoring, not clinical procedures.
"""

import numpy as np

from .data import UNKNOWN, validate_annotations


class ProcedurePrior:
    def __init__(self, strength=.2, smoothing=.5):
        if not np.isfinite(strength) or strength < 0 or not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("Invalid prior strength or smoothing")
        self.strength = float(strength)
        self.smoothing = float(smoothing)

    def fit(self, annotations, classes, training_cases, durations):
        self.classes = list(classes)
        if not self.classes or len(set(self.classes)) != len(self.classes) or UNKNOWN in self.classes:
            raise ValueError("Classes must be unique known clinical labels")
        self.training_cases = sorted(set(map(str, training_cases)))
        source = annotations.copy()
        source["case_id"] = source.case_id.astype(str)
        selected = source[source.case_id.isin(self.training_cases)]
        self.audit = []
        episodes = []
        counts = np.zeros((len(classes), len(classes)), dtype=float)
        for case in self.training_cases:
            a = validate_annotations(selected[selected.case_id == case], durations[case])
            overlap = np.zeros(len(a), dtype=bool)
            for i, r in enumerate(a.itertuples()):
                overlap[i] = bool(((a.start_s < r.end_s) & (a.end_s > r.start_s) & (a.index != i)).any())
            case_episodes = []
            previous = None
            block = 0
            for i, row in enumerate(a.itertuples()):
                reason = ("overlap" if overlap[i] else "pending_or_unknown" if not row.confirmed or row.label in {"", UNKNOWN}
                          else "class_absent_from_local_training" if row.label not in self.classes else "included")
                self.audit.append({"case_id": case, "annotation_id": row.annotation_id, "reason": reason})
                if reason != "included":
                    previous = None
                    block += 1
                    continue
                contiguous = previous is not None and previous["end_s"] == row.start_s
                if not contiguous:
                    block += 1
                if contiguous and previous["label"] == row.label:
                    # Adjacent equal labels are one visible episode. The source
                    # annotation file and parent-level teacher IDs stay intact.
                    previous["end_s"] = row.end_s
                else:
                    previous = {"case_id": case, "label": row.label, "start_s": row.start_s, "end_s": row.end_s, "block": block}
                    case_episodes.append(previous)
            episodes.extend(case_episodes)
            pairs = []
            for before, after in zip(case_episodes, case_episodes[1:]):
                if before["block"] == after["block"] and before["end_s"] == after["start_s"] and before["label"] != after["label"]:
                    pairs.append((self.classes.index(before["label"]), self.classes.index(after["label"])))
            for i, j in pairs:
                counts[i, j] += 1.0 / len(pairs)
        self.transition_counts = counts
        jump = counts + self.smoothing
        np.fill_diagonal(jump, 0.0)
        totals = jump.sum(axis=1, keepdims=True)
        self.jump = np.divide(jump, totals, out=np.zeros_like(jump), where=totals > 0)
        all_durations = [r["end_s"] - r["start_s"] for r in episodes]
        fallback = float(np.median(all_durations)) if all_durations else 30.0
        self.mean_duration_s = []
        self.duration_support_cases = []
        for label in self.classes:
            per_case = {}
            for r in episodes:
                if r["label"] == label:
                    per_case.setdefault(r["case_id"], []).append(r["end_s"] - r["start_s"])
            means = [np.mean(v) for v in per_case.values()]
            self.mean_duration_s.append(float(np.mean(means)) if means else fallback)
            self.duration_support_cases.append(len(means))
        self.mean_duration_s = np.asarray(self.mean_duration_s)
        return self

    def transition(self, delta_s):
        if not np.isfinite(delta_s) or delta_s <= 0:
            raise ValueError("Consecutive center times must increase")
        if len(self.classes) == 1:
            return np.ones((1, 1))
        # Continuous-time exponential survival. Its discretization has a
        # geometric dwell distribution and permits a one-center procedure.
        exit_prob = -np.expm1(-delta_s / self.mean_duration_s)
        exit_prob = np.clip(exit_prob, 1e-10, 1 - 1e-10)
        matrix = exit_prob[:, None] * self.jump
        np.fill_diagonal(matrix, 1 - exit_prob)
        return matrix

    def decode(self, probabilities, times_s):
        p = np.asarray(probabilities, dtype=float)
        times = np.asarray(times_s, dtype=float)
        if p.shape != (len(times), len(self.classes)) or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("Invalid local probability matrix")
        if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("Decode exactly one chronological case at a time")
        if not len(times):
            return []
        if not np.allclose(p.sum(axis=1), 1):
            raise ValueError("Local rows must sum to one; these are not calibrated confidence scores")
        if self.strength == 0:
            return np.asarray(self.classes)[p.argmax(axis=1)].tolist()
        emission = np.log(np.maximum(p, 1e-15))
        previous = emission[0].copy()  # uniform initial prior
        back = np.zeros(p.shape, dtype=np.int32)
        for t in range(1, len(times)):
            candidates = previous[:, None] + self.strength * np.log(self.transition(times[t] - times[t - 1]))
            back[t] = candidates.argmax(axis=0)
            previous = emission[t] + candidates.max(axis=0)
            previous -= previous.max()
        path = np.zeros(len(times), dtype=np.int32)
        path[-1] = previous.argmax()
        for t in range(len(times) - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return np.asarray(self.classes)[path].tolist()
