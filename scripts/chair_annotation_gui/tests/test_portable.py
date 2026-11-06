"""Relocation, UTF-8, retained history and launcher-root tests, without real data."""
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import Store, new_interval
from portable import APP_FILES, PREFIX, build_package
from launcher import resolve_root, save_config, read_config
from test_server import fixture


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "元のPC 日本語"
        self.source = fixture(self.root)
        self.original = self.source.read_bytes()
        self.store = Store(self.root, self.root / "Output/annotations")
        self.cid = next(iter(self.store.cases))
        doc = self.store.document(self.cid)
        doc["annotator"] = "確認者"
        doc["intervals"] = [new_interval(0, 30, clinical_status="treatment", treatment_jp="う蝕除去", notes="境界は不明", instruments=["contra-angle_handpiece"], instrument_status="known")]
        doc["media"] = {"filename": "確認動画.mp4", "offset_s": 12.5}
        self.doc = self.store.save(doc)["document"]

    def tearDown(self):
        self.assertEqual(self.source.read_bytes(), self.original)
        self.temp.cleanup()

    def test_application_package_contains_launchers_but_no_patient_data(self):
        output = self.base / "app.zip"
        result = build_package(output)
        self.assertTrue(result["verified"])
        with zipfile.ZipFile(output) as z:
            self.assertTrue(all(PREFIX + f in z.namelist() for f in APP_FILES))
            self.assertFalse(any("/C31/" in n for n in z.namelist()))
            self.assertNotIn(PREFIX + "launcher_config.json", z.namelist())
            windows = z.read(PREFIX + "START_WINDOWS.bat")
            self.assertIn(b"\r\n", windows)
            self.assertNotIn(b"\n", windows.replace(b"\r\n", b""))
            self.assertTrue((z.getinfo(PREFIX + "START_MAC.command").external_attr >> 16) & 0o100)

    def test_transfer_relocates_with_history_and_unicode(self):
        self.doc["intervals"][0]["notes"] += "・修正"
        saved = self.store.save(self.doc)["document"]
        result = self.store.transfer(self.cid, saved["revision"])
        moved = self.base / "移行先 Windows Mac"
        with zipfile.ZipFile(self.store.output / "exports" / result["filename"]) as z:
            z.extractall(moved)
            checksums = json.loads(z.read(PREFIX + "checksums.json"))
            for relative, expected in checksums.items():
                self.assertEqual(hashlib.sha256(z.read(PREFIX + relative)).hexdigest(), expected)
        app = moved / "ChairLogAnnotator"
        root = resolve_root(app_dir=app)
        self.assertEqual(root, app / "C31")
        other = Store(root, root / "Output/chair_continuous_annotations_v1")
        self.assertEqual(other.document(self.cid), saved)
        self.assertEqual(other.case(self.cid)["source_sha256"], self.store.case(self.cid)["source_sha256"])
        folder = other.output / "cases" / other.key(self.cid)
        self.assertEqual(len(list(folder.glob("revision_*.json"))), 2)
        doc = other.document(self.cid)
        doc["intervals"][0]["notes"] += "・別PCで編集"
        self.assertEqual(other.save(doc)["document"]["revision"], 3)
        self.assertEqual(self.store.document(self.cid)["revision"], 2)
        self.assertTrue(other.export(self.cid, 3, "draft")["filename"])

    def test_selected_case_only_and_manifest_lists_video_without_copy(self):
        other_raw = self.source.with_name("Chair_log_seq_2.dat")
        other_raw.write_bytes(self.original)
        refreshed = Store(self.root, self.store.output)
        output = self.base / "selected.zip"
        result = build_package(output, refreshed, [self.cid])
        self.assertEqual(result["case_count"], 1)
        with zipfile.ZipFile(output) as z:
            self.assertFalse(any("Chair_log_seq_2" in n for n in z.namelist()))
            m = json.loads(z.read(PREFIX + "portable_manifest.json"))
            self.assertEqual(m["requested_case_ids"], [self.cid])
            self.assertEqual(m["actual_cases"], 1)
            self.assertFalse(m["videos_included"])
            self.assertEqual(m["video_references"][0]["filename"], "確認動画.mp4")
            self.assertFalse(any(n.endswith(".mp4") for n in z.namelist()))

    def test_utf8_load_does_not_depend_on_windows_locale(self):
        original_read = Path.read_text
        def windows_default(path, encoding=None, **kw):
            return original_read(path, encoding=encoding or "cp932", **kw)
        with mock.patch.object(Path, "read_text", windows_default):
            recreated = Store(self.root, self.store.output)
            self.assertEqual(recreated.document(self.cid)["intervals"][0]["treatment_jp"], "う蝕除去")
            self.assertEqual(recreated.export_all("draft")["case_count"], 1)

    def test_folder_preference_and_bundled_data_priority(self):
        app = self.base / "App"
        app.mkdir()
        save_config(self.root, app)
        self.assertEqual(read_config(app)["root"], str(self.root))
        self.assertEqual(resolve_root(app_dir=app), self.root)
        fixture(app / "C31")
        self.assertEqual(resolve_root(app_dir=app), app / "C31")
        self.assertEqual(resolve_root(self.root / "chair_export", app_dir=app), self.root)

    def test_folder_picker_and_invalid_root(self):
        with mock.patch("launcher.choose_folder", return_value=str(self.root)):
            self.assertEqual(resolve_root(choose=True, app_dir=self.base), self.root)
        with self.assertRaises(ValueError):
            resolve_root(self.base / "missing", app_dir=self.base)

    def test_transfer_rejects_stale_revision_and_existing_package(self):
        with self.assertRaises(RuntimeError):
            self.store.transfer(self.cid, 0)
        output = self.base / "app.zip"
        build_package(output)
        with self.assertRaises(FileExistsError):
            build_package(output)


if __name__ == "__main__":
    unittest.main()
