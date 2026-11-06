"""Build code-only or selected-case portable ZIPs without changing source data."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

APP = Path(__file__).resolve().parent
PREFIX = "ChairLogAnnotator/"
APP_FILES = ["server.py", "launcher.py", "portable.py", "instruments.json", "README.md", "PORTABLE_GUIDE.html",
             "START_WINDOWS.bat", "START_MAC.command", "start.sh", "static/app.js", "static/index.html", "static/style.css"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def build_package(destination, store=None, case_ids=None):
    """Caller holds store.lock during transfer; all saved revisions are retained."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("出力先が既に存在します: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(".pending-" + uuid.uuid4().hex + ".zip")
    ids = list(case_ids or [])
    if store is None and ids:
        raise ValueError("データ転送にはStoreが必要です。")
    if len(ids) != len(set(ids)):
        raise ValueError("転送する症例IDが重複しています。")
    cases, documents, hashes = [], [], {}
    if store is not None:
        from server import validate
        for cid in ids:
            case = store.case(cid)
            doc = store.document(cid)
            validate(doc, case)
            if not doc["revision"]:
                raise ValueError("転送前に症例を保存してください: " + cid)
            documents.append(doc)
            cases.append({"case_id": cid, "seq": case["seq"], "source_sha256": case["source_sha256"],
                          "revision": doc["revision"], "intervals": len(doc["intervals"])})
    with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        def add_bytes(relative, data, executable=False):
            entry = zipfile.ZipInfo(PREFIX + relative)
            entry.create_system = 3
            entry.external_attr = (0o100755 if executable else 0o100644) << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
            hashes[relative] = hashlib.sha256(data).hexdigest()

        def add_file(relative, source, expected=None):
            before = sha(source)
            if expected and before != expected:
                raise ValueError("転送元のSHA256が変化しています: " + str(source))
            archive.write(source, PREFIX + relative)
            if sha(source) != before:
                raise ValueError("転送中にファイルが変化しました: " + str(source))
            hashes[relative] = before

        for name in APP_FILES:
            data = (APP / name).read_bytes()
            if name.endswith(".bat"):
                data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            add_bytes(name, data, name.endswith((".sh", ".command")))
        if store is not None:
            for doc in documents:
                cid = doc["case_id"]
                source = store.cases[cid]["path"]
                add_file("C31/" + source.relative_to(store.root).as_posix(), source, doc["source_sha256"])
                key = store.key(cid)
                folder = store.output / "cases" / key
                for source in sorted(folder.glob("revision_*.json")) + [folder / "latest.json"]:
                    add_file(f"C31/Output/chair_continuous_annotations_v1/cases/{key}/{source.name}", source)
            add_file("C31/Output/chair_continuous_annotations_v1/app_version.json", store.output / "app_version.json")
            # Include only selected cases' legacy references, retaining original provenance.
            legacy_rows = [r for r in store.legacy if any(str(r.get("seq")) == str(d["seq"]) and
                           (not r.get("source_sha256") or r["source_sha256"] == d["source_sha256"]) for d in documents)]
            if legacy_rows:
                data = io.StringIO(newline="")
                writer = csv.DictWriter(data, fieldnames=list(legacy_rows[0]))
                writer.writeheader()
                writer.writerows(legacy_rows)
                add_bytes("C31/ChairLog_annotations1438_1479.csv", data.getvalue().encode("utf-8-sig"))
            audit = store.output / "audit.jsonl"
            if audit.exists():
                selected_lines = []
                with audit.open(encoding="utf-8") as f:
                    for line in f:
                        if line.strip() and json.loads(line).get("case_id") in ids:
                            selected_lines.append(line)
                add_bytes("C31/Output/chair_continuous_annotations_v1/audit.jsonl", "".join(selected_lines).encode("utf-8"))
        manifest = dict(package_version="1.1", created_at=datetime.now(timezone.utc).isoformat(),
                        kind="selected_case_transfer" if store is not None else "application_only",
                        requested_case_ids=ids, requested_cases=len(ids), actual_cases=len(cases), excluded_cases=[],
                        cases=cases, source_annotations_modified=False, videos_included=False,
                        video_references=[dict(case_id=d["case_id"], **d.get("media", {})) for d in documents],
                        original_legacy_file_sha256=store.legacy_hash if store is not None else None,
                        note="Copy video files separately and select them again. Revisions are snapshots; simultaneous edits on different PCs are not merged.")
        add_bytes("portable_manifest.json", encode(manifest))
        add_bytes("START_HERE.txt", ("1. Unzip the whole ChairLogAnnotator folder.\n"
                  "2. Open PORTABLE_GUIDE.html in your browser.\n"
                  "3. Install Python from https://www.python.org/downloads/ if needed.\n"
                  "4. Windows: START_WINDOWS.bat / macOS: START_MAC.command\n"
                  "C31 contains transferred data, if included. Videos must be copied separately.\n").encode("utf-8"))
        archive.writestr(PREFIX + "checksums.json", encode(hashes))
    # Verify the ZIP contents against all source hashes, streaming large members.
    with zipfile.ZipFile(temporary) as archive:
        for relative, expected in hashes.items():
            h = hashlib.sha256()
            with archive.open(PREFIX + relative) as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError("ZIP検証に失敗: " + relative)
    os.replace(temporary, destination)
    return {"filename": destination.name, "case_count": len(cases), "file_count": len(hashes),
            "bytes": destination.stat().st_size, "sha256": sha(destination), "verified": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an application-only Windows/macOS ZIP. No clinical data.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_package(args.output), ensure_ascii=False))
