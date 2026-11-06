"""Synthetic scientific invariants; no clinical data paths or results."""

import contextlib
import io
import json
import pickle
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

from chair_center_sequence.data import UNKNOWN, attach_teachers, center_grid, parent_case_weights
from chair_center_sequence.features import CaseLog, N_STATE, NUMERIC, NumericEncoding
from chair_center_sequence.model import LocalModel
from chair_center_sequence.sequence import ProcedurePrior
from chair_center_sequence.workflow import main, sha


def annotation(rows, case="A"):
    a = pd.DataFrame(rows, columns=["annotation_id", "start_s", "end_s", "label", "confirmed"])
    a["case_id"] = case
    return a


def events():
    return pd.DataFrame({"time_s": [0., .2, 2., 3., 4., 6., 8.],
                         "code": ["20039", "20742", "20101", "20101", "20101", "20101", "20039"],
                         "value": [0., 0., 0., 0., np.nan, 1., 1.]})


class CenterTests(unittest.TestCase):
    def test_short_interval_is_teacher_even_with_mixed_context(self):
        a = annotation([("long1", 0, 2.2, "other", True), ("short", 2.2, 2.8, "Blowing", True),
                        ("long2", 2.8, 10, "other", True)])
        g, ledger = attach_teachers(center_grid("A", 10), a, 10)
        self.assertEqual(g.loc[g.center_s == 2.5, "target"].item(), "blowing")
        self.assertEqual(ledger.set_index("annotation_id").loc["short", "n_teachers"], 1)
        self.assertLess(g.loc[g.target == "blowing", "boundary_distance_s"].item(), 1)

    def test_boundary_half_open_overlap_pending_and_gap(self):
        a = annotation([("a", 0, 1.5, "one", True), ("b", 1.5, 3, "two", True),
                        ("c", 2.4, 2.8, "two", True), ("p", 3, 4, "three", False)])
        g, _ = attach_teachers(center_grid("A", 5), a, 5)
        self.assertEqual(g.target.tolist(), ["one", "two", UNKNOWN, UNKNOWN, UNKNOWN])
        self.assertEqual(g.unknown_reason.tolist(), ["", "", "overlap", "pending", "unannotated"])

    def test_subgrid_intervals_are_reported_and_finer_grid_recovers(self):
        a = annotation([("tiny", .1, .2, "blowing", True)])
        _, ledger = attach_teachers(center_grid("A", 1), a, 1)
        self.assertEqual(ledger.reason.item(), "no_grid_center")
        fine, _ = attach_teachers(center_grid("A", 1, .05), a, 1)
        self.assertGreater(fine.teacher.sum(), 0)

    def test_grid_does_not_use_annotations_and_keeps_tail(self):
        g = center_grid("A", 2.3)
        self.assertAlmostEqual((g.cell_end_s - g.cell_start_s).sum(), 2.3)
        self.assertAlmostEqual(g.center_s.iloc[-1], 2.15)
        self.assertEqual(len(g), 3)
        with self.assertRaises(ValueError):
            center_grid("A", 1, 0)

    def test_parent_and_case_weights_do_not_count_duplicate_windows_as_cases(self):
        m = pd.DataFrame({"case_id": ["A"] * 5 + ["B"], "annotation_id": ["p"] * 4 + ["q", "r"], "teacher": True})
        w = parent_case_weights(m)
        self.assertAlmostEqual(w[:5].sum(), w[5])
        self.assertAlmostEqual(w[:4].sum(), w[4])

    def test_no_mutation_and_no_boolean_view_side_effects(self):
        a = annotation([("a", 0, 2, "known", True), ("p", 2, 4, "known", False)])
        before = a.copy(deep=True)
        grid = center_grid("A", 4)
        original_grid = grid.copy(deep=True)
        g, _ = attach_teachers(grid, a, 4)
        pd.testing.assert_frame_equal(a, before)
        pd.testing.assert_frame_equal(grid, original_grid)
        self.assertEqual(g.teacher.sum(), 2)


class FeatureTests(unittest.TestCase):
    def test_fully_known_state_never_creates_phantom_unknown_occupancy(self):
        times = np.r_[0., np.linspace(.01, 999.7, 400)]
        e = pd.DataFrame({"time_s": times, "code": "20039", "value": np.arange(len(times)) % 2})
        log = CaseLog(e, 1000.)
        centers = np.arange(10.5, 980, 1.)
        batch = log.summaries(centers)
        for width in [2, 5, 10]:
            self.assertTrue((batch[f"w{width}:20039:state:unknown"] == 0).all())

    def test_repeated_identical_numeric_values_have_exact_zero_std(self):
        e = pd.DataFrame({"time_s": np.linspace(0, 9.9, 100), "code": "20742", "value": .1})
        log = CaseLog(e, 10.)
        centers = np.arange(.5, 10., 1.)
        for source in [pd.DataFrame([log.summary(t) for t in centers]), log.summaries(centers)]:
            for width in [2, 5, 10]:
                self.assertTrue((source[f"w{width}:20742:std"] == 0).all())
                self.assertTrue((source[f"w{width}:20742:mean"] == .1).all())

    def test_exact_fractional_right_boundary_remains_excluded_in_past_mode(self):
        log = CaseLog(pd.DataFrame({"time_s": [.1], "code": ["20742"], "value": [123.]}), 2.)
        scalar = log.summary(.1, widths=(2.,), mode="past")
        batch = log.summaries([.1], widths=(2.,), mode="past")
        self.assertEqual(scalar["w2:20742:observed_count"], 0)
        self.assertEqual(batch["w2:20742:observed_count"].iloc[0], 0)
        scalar_x = log.tensor(.1, mode="past")
        batch_x = log.tensors([.1], mode="past")[0]
        self.assertEqual(scalar_x[N_STATE + len(NUMERIC)].sum(), 0)
        np.testing.assert_array_equal(scalar_x, batch_x)
    def test_vectorized_summary_and_tensor_match_scalar_fractional_logs(self):
        rng = np.random.default_rng(57)
        rows = []
        for code, values in [("20039", [0, 1, 2, 3, np.nan]), ("20101", [0, 1, np.nan]),
                             ("20011", [0, 1, 2, 3]), ("20742", [0, 1234, np.nan]), ("10014", [0, 1])]:
            for time, value in zip(rng.uniform(0, 10.3, 30), rng.choice(values, 30)):
                rows.append((time, code, value))
        log = CaseLog(pd.DataFrame(rows, columns=["time_s", "code", "value"]), 10.3)
        centers = np.r_[np.arange(.5, 10, 1), 10.15, .123, 4.8]
        for mode in ["centered", "past"]:
            scalar = pd.DataFrame([log.summary(t, mode=mode) for t in centers])
            vector = log.summaries(centers, mode=mode)
            self.assertEqual(scalar.columns.tolist(), vector.columns.tolist())
            np.testing.assert_allclose(scalar, vector, rtol=1e-12, atol=1e-12, equal_nan=True)
            scalar_x = np.stack([log.tensor(t, mode=mode) for t in centers])
            vector_x = log.tensors(centers, mode=mode)
            np.testing.assert_allclose(scalar_x, vector_x, rtol=1e-7, atol=1e-7, equal_nan=True)

    def test_initial_unknown_state_persistence_and_invalid_state_reset(self):
        log = CaseLog(events(), 10)
        np.testing.assert_allclose(log.occupancy("20101", 0, 2), [0, 0, 1])
        np.testing.assert_allclose(log.occupancy("20101", 2, 4), [1, 0, 0])
        np.testing.assert_allclose(log.occupancy("20101", 4, 6), [0, 0, 1])
        np.testing.assert_allclose(log.occupancy("20101", 6, 10), [0, 1, 0])

    def test_missing_numeric_is_distinct_from_observed_zero(self):
        log = CaseLog(events(), 10)
        first = log.summary(.5, (1,))
        later = log.summary(2.5, (1,))
        self.assertEqual(first["w1:20742:median"], 0)
        self.assertEqual(first["w1:20742:observed_count"], 1)
        self.assertTrue(np.isnan(later["w1:20742:median"]))
        self.assertEqual(later["w1:20742:observed_count"], 0)

    def test_duplicate_records_are_not_state_changes(self):
        values = CaseLog(events(), 10).summary(3, (2,))
        self.assertEqual(values["w2:20101:records"], 2)
        self.assertEqual(values["w2:20101:changes"], 1)

    def test_edge_windows_use_unknown_padding_and_bound_mask(self):
        x = CaseLog(events(), 10).tensor(.5)
        self.assertTrue(np.isnan(x[N_STATE:N_STATE + len(NUMERIC), 0]).all())
        self.assertEqual(x[-1, 0], 0)
        self.assertEqual(x[4, 0], 1)  # HP state unknown
        self.assertTrue(((x[-1] >= 0) & (x[-1] <= 1)).all())
        np.testing.assert_allclose(x[:5].sum(axis=0), 1)

    def test_future_mutation_leaves_past_features_unchanged(self):
        before = events()
        changed = before.copy()
        changed.loc[changed.time_s > 3, "value"] = 12345
        f = CaseLog(before, 10).summary(3, mode="past")
        g = CaseLog(changed, 10).summary(3, mode="past")
        np.testing.assert_allclose(list(f.values()), list(g.values()), equal_nan=True)

    def test_encoder_preserves_source_nan_and_uses_explicit_sentinel_mask(self):
        x = np.stack([CaseLog(events(), 10).tensor(.5), CaseLog(events(), 10).tensor(5)])
        original = x.copy()
        encoder = NumericEncoding().fit(x[:1])
        encoded = encoder.transform(x)
        np.testing.assert_array_equal(x, original)
        self.assertTrue(np.isfinite(encoded).all())
        missing = ~np.isfinite(x[:, N_STATE, :])
        self.assertTrue((encoded[:, N_STATE, :][missing] == -3).all())
        self.assertTrue((encoded[:, N_STATE + len(NUMERIC), :][missing] == 0).all())


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.a = pd.concat([annotation([("a", 0, 3, "one", True), ("b", 3, 4, "blowing", True),
                                       ("c", 4, 8, "two", True)]),
                            annotation([("z", 0, 8, "two", True)], "HELDOUT")], ignore_index=True)
        self.classes = ["blowing", "one", "two"]
        self.durations = {"A": 8, "HELDOUT": 8}

    def test_heldout_label_mutation_cannot_change_prior_or_predictions(self):
        first = ProcedurePrior().fit(self.a, self.classes, ["A"], self.durations)
        changed = self.a.copy()
        changed.loc[changed.case_id == "HELDOUT", ["label", "start_s", "end_s"]] = ["fabricated", -1000, 99999]
        second = ProcedurePrior().fit(changed, self.classes, ["A"], self.durations)
        np.testing.assert_array_equal(first.transition_counts, second.transition_counts)
        np.testing.assert_array_equal(first.mean_duration_s, second.mean_duration_s)
        p = np.array([[.01, .98, .01], [.98, .01, .01], [.01, .01, .98]])
        self.assertEqual(first.decode(p, [.5, 1.5, 2.5]), second.decode(p, [.5, 1.5, 2.5]))
        self.assertEqual(first.training_cases, ["A"])

    def test_unknown_gap_is_not_a_clinical_transition(self):
        a = annotation([("a", 0, 2, "one", True), ("p", 2, 4, "blowing", False),
                        ("b", 4, 6, "two", True), ("c", 7, 8, "one", True)])
        prior = ProcedurePrior().fit(a, self.classes, ["A"], {"A": 8})
        self.assertEqual(prior.transition_counts.sum(), 0)
        np.testing.assert_allclose(prior.transition(1).sum(axis=1), 1)

    def test_single_center_procedure_is_allowed(self):
        prior = ProcedurePrior().fit(self.a, self.classes, ["A"], self.durations)
        p = np.array([[.00001, .99998, .00001], [.99998, .00001, .00001], [.00001, .00001, .99998]])
        self.assertEqual(prior.decode(p, [.5, 1.5, 2.5]), ["one", "blowing", "two"])
        self.assertTrue((prior.transition(1) > 0).all())

    def test_even_submicrosecond_unknown_gaps_censor_transitions_and_merging(self):
        for kind in ["unannotated", "pending", "explicit_unknown"]:
            rows = [("a", 0, 1, "one", True), ("b", 1.0000005, 2, "two", True)]
            if kind != "unannotated":
                rows.append(("gap", 1, 1.0000005, UNKNOWN if kind == "explicit_unknown" else "one", kind != "pending"))
            prior = ProcedurePrior().fit(annotation(rows), self.classes, ["A"], {"A": 2})
            self.assertEqual(prior.transition_counts.sum(), 0, kind)
        same = annotation([("a", 0, 1, "one", True), ("b", 1.0000005, 2, "one", True)])
        prior = ProcedurePrior().fit(same, ["one"], ["A"], {"A": 2})
        self.assertLess(prior.mean_duration_s[0], 1.1)

    def test_zero_strength_is_exact_local_control_and_bad_order_rejected(self):
        prior = ProcedurePrior(strength=0).fit(self.a, self.classes, ["A"], self.durations)
        p = np.array([[.2, .7, .1], [.6, .1, .3]])
        self.assertEqual(prior.decode(p, [.5, 1.5]), ["one", "blowing"])
        with self.assertRaises(ValueError):
            prior.decode(p, [1.5, .5])


class ModelTests(unittest.TestCase):
    def test_summary_roundoff_variance_is_never_amplified(self):
        g, f, _ = self.fixture()
        f[:, 0] = 1e-17 * np.arange(len(f))
        m = LocalModel(num_kernels=0, epochs=1).fit(f, g)
        self.assertEqual(m.summary_scaler.scale_[0], 1.)
        self.assertTrue(m.summary_unscaled[0])
        query = f[:1].copy(); query[0, 0] = 1.
        self.assertLess(abs(m.design(query)[0, 0]), 2.)

    def test_logspace_ovr_probabilities_remain_finite_for_extreme_margins(self):
        g, f, _ = self.fixture()
        g.loc[g.index[-2:], "target"] = "third"
        m = LocalModel(num_kernels=0, epochs=1).fit(f, g)
        m.classifier.coef_.fill(0)
        m.classifier.intercept_[:] = [-10000., -10001., -10002.]
        p = m.probabilities(f)
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(axis=1), 1)
        expected = np.exp([0., -1., -2.]); expected /= expected.sum()
        np.testing.assert_allclose(p[0], expected)

    def test_cached_design_has_identical_sgd_coefficients_and_predictions(self):
        g, f, x = self.fixture()
        cached = LocalModel(num_kernels=16, epochs=2, batch_size=4, cache_design=True).fit(f, g, x)
        original = LocalModel(num_kernels=16, epochs=2, batch_size=4, cache_design=False).fit(f, g, x)
        np.testing.assert_allclose(cached.classifier.coef_, original.classifier.coef_, rtol=0, atol=1e-13)
        np.testing.assert_allclose(cached.probabilities(f, x), original.probabilities(f, x), rtol=0, atol=1e-13)

    def fixture(self):
        g, _ = attach_teachers(center_grid("A", 10), annotation([("a", 0, 5, "one", True), ("b", 5, 10, "two", True)]), 10)
        log = CaseLog(events(), 10)
        f = pd.DataFrame([log.summary(t) for t in g.center_s]).to_numpy(float)
        x = np.stack([log.tensor(t) for t in g.center_s])
        return g, f, x

    def test_local_serialization_and_predictions_need_no_query_labels(self):
        g, f, _ = self.fixture()
        m = LocalModel(num_kernels=0, epochs=2, batch_size=4).fit(f, g)
        p = m.probabilities(f)
        np.testing.assert_allclose(p.sum(axis=1), 1)
        np.testing.assert_array_equal(p, pickle.loads(pickle.dumps(m)).probabilities(f))
        self.assertEqual(m.training_cases, ["A"])
        self.assertEqual(m.classes, ["one", "two"])

    def test_rocket_branch_and_variance_guard_on_synthetic_only(self):
        g, f, x = self.fixture()
        m = LocalModel(num_kernels=16, epochs=1, batch_size=4).fit(f, g, x)
        p = m.probabilities(f, x)
        self.assertTrue(np.isfinite(p).all())
        self.assertEqual(m.keep.shape, (32,))
        np.testing.assert_allclose(p.sum(axis=1), 1)


class WorkflowTests(unittest.TestCase):
    def test_synthetic_prepare_evaluate_and_preserved_source_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "events").mkdir()
            manifest, aligned, source = [], [], []
            for i in range(4):
                case = str(i)
                raw = root / f"raw{case}.txt"
                raw.write_text("synthetic fixture, not a clinical raw log\n")
                events().to_csv(root / "events" / f"{case}.csv.gz", index=False, compression="gzip")
                manifest.append({"seq": case, "raw_duration_s": 10., "raw_start": "2000-01-01 00:00:00",
                                 "source_path": str(raw), "source_sha256": sha(raw)})
                for j, (start, end, label) in enumerate([(0, 4, "other"), (4, 5, "blowing"), (6, 10, "other")]):
                    aid = f"{case}-{j}"
                    aligned.append({"seq": case, "id": aid, "start_s": start, "end_s": end,
                                    "treatment_normalized": label, "confirmed": True})
                    source.append({"seq": case, "id": aid, "treatment": label, "status": "確認済み",
                                   "start_datetime": str(pd.Timestamp("2000-01-01") + pd.Timedelta(seconds=start)),
                                   "end_datetime": str(pd.Timestamp("2000-01-01") + pd.Timedelta(seconds=end)), "source_sha256": sha(raw)})
            for name, rows in [("manifest", manifest), ("aligned", aligned), ("source", source),
                               ("folds", [{"seq": str(i), "fold": i % 2} for i in range(4)])]:
                pd.DataFrame(rows).to_csv(root / f"{name}.csv", index=False)
            original = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
            prep = ["prepare", "--aligned", str(root / "aligned.csv"), "--manifest", str(root / "manifest.csv"),
                    "--folds", str(root / "folds.csv"), "--events-dir", str(root / "events"), "--annotation-source", str(root / "source.csv"),
                    "--output", str(root / "prepared")]
            run = ["evaluate", "--prepared", str(root / "prepared"), "--output", str(root / "evaluated"), "--num-kernels", "0", "--epochs", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                main(prep)
                main(run)
            q = pd.read_csv(root / "evaluated" / "predictions.csv.gz", keep_default_na=False)
            self.assertEqual(len(q), 40)
            self.assertEqual(q.target.eq("unknown").sum(), 4)
            self.assertTrue(q.local_candidate.ne("").all())
            self.assertFalse(q.human_confirmed.any())
            metrics = json.loads((root / "evaluated" / "metrics.json").read_text())
            self.assertEqual(len(metrics), 2)
            self.assertEqual(metrics[0]["scored_centers"], 36)
            for row in json.loads((root / "evaluated" / "fold_audit.json").read_text()):
                self.assertFalse(set(row["training_cases"]) & set(row["validation_cases"]))
            for path, fingerprint in original.items():
                self.assertEqual(sha(path), fingerprint)
            # A concurrent metadata edit immediately after loading must not
            # attach the edited file's fingerprint to the old in-memory data.
            input_file = root / "manifest.csv"
            input_bytes = input_file.read_bytes()
            actual_read = pd.read_csv
            mutated = False
            def mutate_after_read(path, *args, **kwargs):
                nonlocal mutated
                result = actual_read(path, *args, **kwargs)
                if Path(path) == input_file and not mutated:
                    input_file.write_bytes(input_bytes + b"\n")
                    mutated = True
                return result
            prep_changed = prep[:-1] + [str(root / "prepared_with_changed_input")]
            with mock.patch("chair_center_sequence.workflow.pd.read_csv", side_effect=mutate_after_read), \
                    self.assertRaisesRegex(ValueError, "Input changed during preparation"), \
                    contextlib.redirect_stdout(io.StringIO()):
                main(prep_changed)
            input_file.write_bytes(input_bytes)
            with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                main(run)
            (root / "prepared" / "annotations.csv").write_text("mutated")
            with self.assertRaisesRegex(ValueError, "Prepared input changed"):
                main(run)


if __name__ == "__main__":
    unittest.main()
