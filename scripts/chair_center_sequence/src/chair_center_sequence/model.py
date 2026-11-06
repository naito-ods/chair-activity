"""Batched local classifier with optional ROCKET context and explicit masks."""

import json
import tempfile
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from .data import UNKNOWN, parent_case_weights
from .features import NumericEncoding


class LocalModel:
    def __init__(self, num_kernels=10000, batch_size=256, epochs=10, seed=20261106, n_jobs=1, cache_dir=None, cache_design=True):
        if num_kernels < 0 or batch_size < 1 or epochs < 1:
            raise ValueError("Invalid model configuration")
        self.num_kernels, self.batch_size, self.epochs = int(num_kernels), int(batch_size), int(epochs)
        self.seed, self.n_jobs = int(seed), int(n_jobs)
        self.cache_dir, self.cache_design = cache_dir, bool(cache_design)

    def batches(self, n):
        for start in range(0, n, self.batch_size):
            yield slice(start, min(n, start + self.batch_size))

    def rocket_features(self, tensor):
        z = np.asarray(self.rocket.transform(self.encoding.transform(tensor)), dtype=np.float64)
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite ROCKET features")
        return z

    def design(self, summary, tensor=None):
        summary = np.asarray(summary, dtype=float)
        z = self.summary_scaler.transform(self.imputer.transform(summary))
        z /= np.sqrt(max(1, z.shape[1]))
        if self.num_kernels:
            if tensor is None or len(tensor) != len(summary):
                raise ValueError("ROCKET requires aligned local-context tensors")
            r = self.rocket_scaler.transform(self.rocket_features(tensor))[:, self.keep]
            r /= np.sqrt(max(1, r.shape[1]))
            z = np.concatenate([z, r], axis=1)
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite local design")
        return np.asarray(z, dtype=np.float64)

    def fit(self, summary, meta, tensor=None):
        summary = np.asarray(summary, dtype=float)
        if len(summary) != len(meta) or not len(meta) or not meta.teacher.all():
            raise ValueError("Fit accepts only aligned known TRAIN-case centers")
        if not np.isfinite(summary).any() or np.isinf(summary).any() or meta.target.isin(["", UNKNOWN]).any():
            raise ValueError("Invalid training features or unknown teachers")
        self.classes = sorted(meta.target.astype(str).unique())
        self.training_cases = sorted(meta.case_id.astype(str).unique())
        self.training_parent_ids = sorted(meta.annotation_id.astype(str).unique())
        weights = parent_case_weights(meta)
        self.imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True).fit(summary)
        self.summary_scaler = StandardScaler().fit(self.imputer.transform(summary))
        # Disable amplification below float64 numerical precision; retain the
        # feature and its raw-centered value instead of inventing information.
        self.summary_precision_floor = 1024 * np.finfo(np.float64).eps * np.maximum(1., np.abs(self.summary_scaler.mean_))
        self.summary_unscaled = np.sqrt(self.summary_scaler.var_) <= self.summary_precision_floor
        self.summary_scaler.scale_[self.summary_unscaled] = 1.
        if self.num_kernels:
            from sktime.transformations.rocket import Rocket
            if tensor is None or len(tensor) != len(summary):
                raise ValueError("Training tensors missing or misaligned")
            self.encoding = NumericEncoding().fit(tensor)
            self.rocket = Rocket(num_kernels=self.num_kernels, normalise=False, n_jobs=self.n_jobs, random_state=self.seed)
            # Kernels use the TRAIN tensor shape; no query series is used.
            self.rocket.fit(self.encoding.transform(tensor[:self.batch_size]))
            self.rocket_scaler = StandardScaler(with_mean=False)
            for number, batch in enumerate(self.batches(len(meta))):
                self.rocket_scaler.partial_fit(self.rocket_features(tensor[batch]))
                if number % 40 == 0:
                    print(json.dumps({"stage": "rocket_scaling", "processed": batch.stop, "total": len(meta)}), flush=True)
            std = np.sqrt(self.rocket_scaler.var_)
            threshold = 10 * np.finfo(np.float32).eps * np.maximum(1, np.abs(self.rocket_scaler.mean_))
            self.keep = std > threshold
            self.guard_threshold = threshold
        self.classifier = None
        if len(self.classes) > 1:
            # Explicit class balancing on top of case/parent weights. These
            # scores must not be interpreted as calibrated clinical confidence.
            y = meta.target.to_numpy(str)
            mass = {label: weights[y == label].sum() for label in self.classes}
            balanced = weights * np.array([weights.sum() / (len(self.classes) * mass[v]) for v in y])
            self.classifier = SGDClassifier(loss="log_loss", alpha=.0001, random_state=self.seed, shuffle=False)
            # Cache the exact standardized design once. Recomputing identical
            # convolutions in every SGD epoch is unnecessary. The transient
            # cache is not part of a fitted/pickled model and is deleted on exit.
            with tempfile.TemporaryDirectory(prefix="center_design_", dir=self.cache_dir) as temporary:
                design = None
                if self.cache_design:
                    first = self.design(summary[:1], tensor[:1] if self.num_kernels else None)
                    design = np.memmap(Path(temporary) / "train.float64", mode="w+", dtype=np.float64, shape=(len(meta), first.shape[1]))
                    for number, batch in enumerate(self.batches(len(meta))):
                        design[batch] = self.design(summary[batch], tensor[batch] if self.num_kernels else None)
                        if number % 40 == 0:
                            print(json.dumps({"stage": "design_cache", "processed": batch.stop, "total": len(meta)}), flush=True)
                    design.flush()
                self._train_epochs(summary, tensor, y, balanced, design)
                if design is not None:
                    del design
        return self

    def _train_epochs(self, summary, tensor, y, balanced, design):
        rng = np.random.default_rng(self.seed)
        for epoch in range(self.epochs):
            order = rng.permutation(len(y))
            for batch in self.batches(len(y)):
                ix = order[batch]
                z = np.asarray(design[ix]) if design is not None else self.design(summary[ix], tensor[ix] if self.num_kernels else None)
                self.classifier.partial_fit(z, y[ix], classes=self.classes, sample_weight=balanced[ix])
            print(json.dumps({"stage": "sgd_epoch", "epoch": epoch + 1, "epochs": self.epochs}), flush=True)

    def probabilities(self, summary, tensor=None):
        if tensor is not None and len(tensor) != len(summary):
            raise ValueError("Prediction tensors are misaligned")
        if self.classifier is None:
            return np.ones((len(summary), 1), dtype=float)
        output = []
        for batch in self.batches(len(summary)):
            z = self.design(summary[batch], tensor[batch] if self.num_kernels else None)
            scores = np.asarray(self.classifier.decision_function(z), dtype=np.float64)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite classifier margins")
            # Exactly the normalized one-vs-rest sigmoid used by SGD log-loss,
            # evaluated in log space so all-negative margins cannot underflow
            # into an all-zero row. This does not calibrate the scores.
            log_p = -np.logaddexp(0., -scores)
            log_p -= log_p.max(axis=1, keepdims=True)
            p = np.exp(log_p)
            if not np.isfinite(p).all() or (p.sum(axis=1) <= 0).any():
                raise ValueError("Nonfinite classifier output")
            output.append(p / p.sum(axis=1, keepdims=True))
        return np.concatenate(output) if output else np.empty((0, len(self.classes)))
