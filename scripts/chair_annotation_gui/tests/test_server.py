"""Scientific data integrity, revision conflicts, and export tests with synthetic logs."""
import copy
import csv
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import Store, INSTRUMENTS, digest, new_interval, quality, validate


def fixture(root):
    path = root / "chair_export/2000-01-01/TEST/Chair_log_seq_1.dat"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[content]\noccur_date,occur_time,event_code,event_value\n"
                    "2000/01/01,00:00:00.000,20039,0\n"
                    "2000/01/01,00:00:01.000,20742,0\n"
                    "2000/01/01,00:00:02.000,20742,\n"
                    "bad,time,20742,1200\n"
                    "2000/01/01,00:00:10.000,30117,1\n"
                    "2000/01/01,00:00:21.000,20742,40000\n"
                    "2000/01/01,00:00:30.000,90012,0\n", encoding="utf-8")
    return path


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = fixture(self.root)
        self.original = digest(self.source)
        self.store = Store(self.root, self.root / "Output/test")
        self.cid = next(iter(self.store.cases))
        self.case = self.store.case(self.cid)
        self.doc = self.store.document(self.cid)
        self.doc["annotator"] = "TEST REVIEWER"

    def tearDown(self):
        self.assertEqual(digest(self.source), self.original)
        self.temp.cleanup()

    def complete(self):
        self.doc["intervals"] = [new_interval(0, 30, reviewed=True)]
        return self.doc

    def test_missing_and_zero_distinct_and_bad_rows_audited(self):
        numeric = [e for e in self.case["events"] if e[1] == "20742"]
        self.assertEqual([e[2] for e in numeric], [0, None, 40000])
        self.assertEqual(self.case["invalid_timestamp_rows"], [6])
        self.assertEqual(self.case["raw_rows"], 7)
        self.assertEqual(self.case["event_bins"][10], [1, 0])

    def test_english_vocabulary_exact(self):
        self.assertEqual(len(INSTRUMENTS), 40)
        self.assertIn("articulatin_paper_holder", INSTRUMENTS)
        self.doc["intervals"] = [new_interval(0, 30, instruments=INSTRUMENTS, instrument_status="known")]
        validate(self.doc, self.case)
        self.doc["intervals"][0]["instruments"] = ["日本語器具"]
        with self.assertRaises(ValueError):
            validate(self.doc, self.case)

    def test_gaps_and_nested_overlaps(self):
        self.doc["intervals"] = [new_interval(1, 20), new_interval(5, 10), new_interval(15, 25)]
        q = validate(self.doc, self.case)
        self.assertEqual(len(q["overlaps"]), 2)
        self.assertEqual(q["gaps"], [{"start_s": 0, "end_s": 1}, {"start_s": 25, "end_s": 30}])
        self.assertEqual(q["covered_s"], 24)
        self.assertFalse(q["final_export_allowed"])

    def test_no_procedure_requires_external_evidence(self):
        self.complete()["intervals"][0].update(clinical_status="no_procedure", certainty="confirmed", evidence_sources=["log_inference"])
        self.assertFalse(validate(self.doc, self.case)["final_export_allowed"])
        self.doc["intervals"][0].update(evidence_sources=["video"], evidence_detail="Synthetic video reference")
        self.assertTrue(validate(self.doc, self.case)["final_export_allowed"])

    def test_unknown_can_be_reviewed_without_becoming_negative(self):
        self.assertTrue(validate(self.complete(), self.case)["final_export_allowed"])
        saved = self.store.save(self.doc)["document"]
        export = self.store.export(self.cid, saved["revision"], "final")
        with zipfile.ZipFile(self.store.output / "exports" / export["filename"]) as z:
            row = next(csv.DictReader(io.StringIO(z.read("intervals.csv").decode("utf-8-sig"))))
            self.assertEqual(row["clinical_status"], "unknown")
            self.assertEqual(row["instrument_status"], "unknown")
            self.assertEqual(row["clinical_ground_truth_eligible"], "False")

    def test_revision_conflict_preserves_previous_versions(self):
        initial = copy.deepcopy(self.complete())
        saved = self.store.save(initial)["document"]
        self.assertEqual(saved["revision"], 1)
        with self.assertRaises(RuntimeError):
            self.store.save(initial)
        saved["intervals"][0]["notes"] = "日本語メモ"
        self.assertEqual(self.store.save(saved)["document"]["revision"], 2)
        folder = self.store.output / "cases" / self.store.key(self.cid)
        self.assertEqual(len(list(folder.glob("revision_*.json"))), 2)

    def test_export_round_trip_and_csv_unicode(self):
        row = new_interval(0, 30, clinical_status="treatment", treatment_jp="う蝕除去", instruments=["contra-angle_handpiece", "dental_mirror"], instrument_status="known", instrument_relation="sequential", reviewed=True, notes="=formula\n日本語", evidence_sources=["video"], evidence_detail="example.mp4", certainty="confirmed")
        self.doc["intervals"] = [row]
        saved = self.store.save(self.doc)["document"]
        result = self.store.export(self.cid, 1, "final")
        with zipfile.ZipFile(self.store.output / "exports" / result["filename"]) as z:
            imported = json.loads(z.read("annotations.json"))
            self.assertEqual(imported, saved)
            self.assertTrue(validate(imported, self.case)["final_export_allowed"])
            r = next(csv.DictReader(io.StringIO(z.read("intervals.csv").decode("utf-8-sig"))))
            self.assertEqual(json.loads(r["instruments"]), row["instruments"])
            self.assertEqual(r["treatment_jp"], "う蝕除去")
            self.assertEqual(r["notes"], "'=formula\n日本語")
            self.assertEqual(len(json.loads(z.read("checksums.json"))), 6)

    def test_scope_exclusions_exported(self):
        self.doc["scope"] = {"start_s": 5, "end_s": 20}
        self.doc["intervals"] = [new_interval(5, 20, reviewed=True)]
        self.store.save(self.doc)
        result = self.store.export(self.cid, 1, "final")
        with zipfile.ZipFile(self.store.output / "exports" / result["filename"]) as z:
            m = json.loads(z.read("manifest.json"))
            self.assertEqual(m["exclusions_outside_scope"], [{"start_s": 0, "end_s": 5}, {"start_s": 20, "end_s": 30}])

    def test_malformed_boundaries_ids_hashes_and_confidence(self):
        valid = copy.deepcopy(self.complete())
        for changes in [{"start_s": -1}, {"end_s": 31}, {"end_s": float("nan")}, {"start_s": True}, {"boundary_certainty": "exact"}]:
            doc = copy.deepcopy(valid)
            doc["intervals"][0].update(changes)
            with self.assertRaises(ValueError):
                validate(doc, self.case)
        doc = copy.deepcopy(valid)
        doc["intervals"].append(copy.deepcopy(doc["intervals"][0]))
        with self.assertRaises(ValueError):
            validate(doc, self.case)
        doc = copy.deepcopy(valid)
        doc["source_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            validate(doc, self.case)

    def test_draft_allowed_but_final_rejects_unreviewed(self):
        self.doc["intervals"] = [new_interval(0, 12)]
        self.store.save(self.doc)
        self.store.export(self.cid, 1, "draft")
        with self.assertRaises(ValueError):
            self.store.export(self.cid, 1, "final")

    def test_refuse_existing_analysis_folder(self):
        path = self.root / "Output/unrelated"
        path.mkdir()
        (path / "existing.csv").write_text("do not touch")
        with self.assertRaises(ValueError):
            Store(self.root, path)

    def test_batch_export_is_complete_and_does_not_drop_unreviewed_cases(self):
        self.complete()
        self.doc["intervals"][0]["notes"] = "=formula"
        saved = self.store.save(self.doc)["document"]
        exported = self.store.export_all("final")
        with zipfile.ZipFile(self.store.output / "exports" / exported["filename"]) as z:
            m = json.loads(z.read("collection_manifest.json"))
            self.assertEqual(m["actual_cases"], 1)
            self.assertEqual(m["actual_intervals"], 1)
            r = next(csv.DictReader(io.StringIO(z.read("all_intervals.csv").decode("utf-8-sig"))))
            self.assertEqual(r["notes"], "'=formula")
        saved["intervals"][0]["reviewed"] = False
        self.store.save(saved)
        with self.assertRaisesRegex(ValueError, "未完了症例を除外していません"):
            self.store.export_all("final")
        self.assertEqual(self.store.export_all("draft")["case_count"], 1)


if __name__ == "__main__":
    unittest.main()
