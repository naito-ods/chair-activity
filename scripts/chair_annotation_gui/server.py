#!/usr/bin/env python3
"""Local, dependency-free annotation editor. Source logs are always read-only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import secrets
import threading
import uuid
import zipfile
from collections import Counter, OrderedDict
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

APP = Path(__file__).resolve().parent
VERSION = "chair-continuous-annotation-v1"
INSTRUMENTS = json.loads((APP / "instruments.json").read_text())
TARGET_CODES = {"20039", "20011", "20101", "20742", "20743", "20108", "20774", "20776", "10014", "10015", "90012"}
ENUMS = {
    "clinical_status": {"treatment", "procedure_unspecified", "no_procedure", "unknown"},
    "instrument_status": {"known", "none", "unknown"},
    "instrument_relation": {"single", "simultaneous", "sequential", "uncertain"},
    "observation_status": {"unreviewed", "events_present", "sparse", "no_target_events", "suspected_recording_gap"},
    "certainty": {"confirmed", "probable", "uncertain"},
    "boundary_certainty": {"exact", "approximate", "unknown"},
}
EVIDENCE = {"video", "direct_observation", "clinical_record", "log_inference", "legacy_annotation", "unknown"}
INDEPENDENT = {"video", "direct_observation", "clinical_record"}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def new_interval(start, end, **fields):
    base = dict(id=str(uuid.uuid4()), start_s=start, end_s=end,
                clinical_status="unknown", treatment_jp="", instruments=[], instrument_status="unknown",
                instrument_relation="uncertain", observation_status="unreviewed", certainty="uncertain",
                evidence_sources=["unknown"], evidence_detail="", boundary_certainty="unknown",
                start_uncertainty_s=None, end_uncertainty_s=None, reviewed=False, notes="")
    base.update(fields)
    return base


def read_log(path):
    rows = []
    invalid = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        header = f.readline()
        if header.strip() != "[content]":
            f.seek(0)
            offset = 2
        else:
            offset = 3
        reader = csv.DictReader(f)
        required = {"occur_date", "occur_time", "event_code", "event_value"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("原ログの列が不足しています: " + str(required))
        for line, r in enumerate(reader, offset):
            try:
                t = datetime.fromisoformat(r["occur_date"].replace("/", "-") + "T" + r["occur_time"])
            except (ValueError, TypeError, AttributeError):
                invalid.append(line)
                continue
            raw = r.get("event_value", "") or ""
            try:
                value = float(raw)
                if not math.isfinite(value):
                    value = None
            except ValueError:
                value = None
            rows.append((t, (r.get("event_code") or "").strip(), value, raw, line))
    if not rows:
        raise ValueError("有効な時刻がありません。ログを空白区間として処理しません。")
    rows.sort(key=lambda r: r[0])
    origin = rows[0][0]
    events = [[round((t - origin).total_seconds(), 6), c, v, raw, line] for t, c, v, raw, line in rows]
    return origin, events[-1][0], events, invalid


def validate(doc, case):
    """Reject malformed documents; retain draft overlaps/gaps as explicit QC issues."""
    if not isinstance(doc, dict) or doc.get("schema_version") != VERSION:
        raise ValueError("対応する注釈JSONではありません。schema_version を確認してください。")
    for k in ("case_id", "source_sha256", "time_origin", "duration_s"):
        if doc.get(k) != case[k]:
            raise ValueError(f"原ログとの不一致: {k}")
    if not isinstance(doc.get("annotator"), str) or not isinstance(doc.get("intervals"), list):
        raise ValueError("annotator / intervals の形式が不正です。")
    if doc.get("study_role") not in {"unassigned", "development", "validation", "held_out"}:
        raise ValueError("症例の用途が不正です。")
    scope = doc.get("scope", {})
    a, b = scope.get("start_s"), scope.get("end_s")
    if not number(a) or not number(b) or not 0 <= a < b <= case["duration_s"]:
        raise ValueError("対象範囲は原ログ内の 開始 < 終了 にしてください。")
    media = doc.get("media", {})
    if not isinstance(media.get("filename", ""), str) or not number(media.get("offset_s", 0)):
        raise ValueError("動画の情報が不正です。")
    seen = set()
    for r in doc["intervals"]:
        rid = r.get("id")
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("区間IDが空欄、または重複しています。")
        seen.add(rid)
        s, e = r.get("start_s"), r.get("end_s")
        if not number(s) or not number(e) or not a <= s < e <= b:
            raise ValueError(f"区間 {rid}: 対象範囲内の 開始 < 終了 にしてください。")
        for k, allowed in ENUMS.items():
            if r.get(k) not in allowed:
                raise ValueError(f"区間 {rid}: {k} が不正です。")
        for k in ("treatment_jp", "evidence_detail", "notes"):
            if not isinstance(r.get(k), str):
                raise ValueError(f"区間 {rid}: {k} は文字列が必要です。")
        if type(r.get("reviewed")) is not bool:
            raise ValueError("確認済みフラグは true/false が必要です。")
        instruments = r.get("instruments")
        if not isinstance(instruments, list) or any(not isinstance(x, str) for x in instruments):
            raise ValueError("器具は英語名の配列で指定してください。")
        if len(instruments) != len(set(instruments)) or not set(instruments) <= set(INSTRUMENTS):
            raise ValueError("器具名が指定された40種類と一致しないか、重複しています。")
        if (r["instrument_status"] == "known") != bool(instruments):
            raise ValueError("器具選択と器具の確認状態が一致しません。")
        if len(instruments) > 1 and r["instrument_relation"] == "single":
            raise ValueError("複数器具の関係を指定してください。")
        if r["clinical_status"] == "treatment" and not r["treatment_jp"].strip():
            raise ValueError("処置名を入力してください。")
        if r["clinical_status"] != "treatment" and r["treatment_jp"].strip():
            raise ValueError("処置名を付ける場合、臨床行為を「処置名あり」にしてください。")
        evidence = r.get("evidence_sources")
        if not isinstance(evidence, list) or not evidence or any(x not in EVIDENCE for x in evidence):
            raise ValueError("確認根拠を選択してください。")
        if len(set(evidence)) != len(evidence) or ("unknown" in evidence and len(evidence) > 1):
            raise ValueError("確認根拠の「不明」は他の根拠と同時に選択できません。")
        for k in ("start_uncertainty_s", "end_uncertainty_s"):
            if r.get(k) is not None and (not number(r[k]) or r[k] < 0):
                raise ValueError("境界の誤差は0以上の秒数、または空欄にしてください。")
        if r["boundary_certainty"] == "exact" and any(r.get(k) != 0 for k in ("start_uncertainty_s", "end_uncertainty_s")):
            raise ValueError("境界が明確な場合、開始・終了の誤差を0にしてください。")
    return quality(doc)


def quality(doc):
    intervals = sorted(doc["intervals"], key=lambda r: (r["start_s"], r["end_s"], r["id"]))
    scope = doc["scope"]
    cursor = scope["start_s"]
    gaps, overlaps, active = [], [], []
    blockers, warnings = [], []
    for r in intervals:
        s, e = r["start_s"], r["end_s"]
        if s > cursor + 1e-6:
            gaps.append({"start_s": cursor, "end_s": s})
        active = [x for x in active if x["end_s"] > s + 1e-6]
        for x in active:
            overlaps.append({"ids": [x["id"], r["id"]], "start_s": s, "end_s": min(e, x["end_s"])})
        active.append(r)
        cursor = max(cursor, e)
        if not r["reviewed"]:
            blockers.append({"id": r["id"], "reason": "未確認の区間"})
        if r["certainty"] == "confirmed" and r["clinical_status"] != "unknown":
            if not INDEPENDENT.intersection(r["evidence_sources"]) or not r["evidence_detail"].strip():
                blockers.append({"id": r["id"], "reason": "確認済みの臨床行為にはログ以外の根拠と説明が必要"})
        if r["clinical_status"] == "no_procedure" and r["certainty"] != "confirmed":
            blockers.append({"id": r["id"], "reason": "処置なしには確認済みの根拠が必要。不明と区別してください"})
        if r["observation_status"] == "unreviewed":
            warnings.append({"id": r["id"], "reason": "ログ観測状況が未確認"})
        if len(r["instruments"]) > 1 and r["instrument_relation"] != "simultaneous":
            warnings.append({"id": r["id"], "reason": "複数器具の同時使用は確認されていません"})
    if cursor < scope["end_s"] - 1e-6:
        gaps.append({"start_s": cursor, "end_s": scope["end_s"]})
    if gaps:
        blockers.append({"reason": f"未記入区間 {len(gaps)} 件。不明区間として明示できます"})
    if overlaps:
        blockers.append({"reason": f"処置区間の重複 {len(overlaps)} 件。主処置の時間区分を調整してください"})
    if not doc["annotator"].strip():
        blockers.append({"reason": "記入者が空欄"})
    total = scope["end_s"] - scope["start_s"]
    uncovered = sum(g["end_s"] - g["start_s"] for g in gaps)
    return dict(n_intervals=len(intervals), gaps=gaps, overlaps=overlaps, blockers=blockers,
                warnings=warnings, covered_s=total - uncovered, scope_s=total,
                coverage_fraction=(total - uncovered) / total,
                unknown_interval_count=sum(r["clinical_status"] == "unknown" for r in intervals),
                final_export_allowed=not blockers)


def csv_data(rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        safe = {}
        for k in fields:
            value = row.get(k, "")
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            safe[k] = value
        writer.writerow(safe)
    return output.getvalue().encode("utf-8-sig")


class Store:
    def __init__(self, root, output):
        self.root = root.resolve()
        self.output = output.resolve()
        if not self.output.is_relative_to(self.root / "Output"):
            raise ValueError("保存先は C31/Output 内の新しい専用フォルダにしてください。")
        marker = self.output / "app_version.json"
        if self.output.exists() and any(self.output.iterdir()) and not marker.exists():
            raise ValueError("既存の別用途フォルダには書き込みません。新しい保存先を指定してください。")
        if marker.exists() and json.loads(marker.read_text()).get("schema_version") != VERSION:
            raise ValueError("保存先のバージョンが一致しません。")
        self.output.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_bytes(json_bytes({"schema_version": VERSION, "created_at": now()}))
        self.cases = {}
        for path in sorted((self.root / "chair_export").glob("*/*/Chair_log_seq_*.dat")):
            match = re.fullmatch(r"Chair_log_seq_(\d+)\.dat", path.name)
            if match and path.resolve().is_relative_to(self.root / "chair_export"):
                cid = f"{path.parent.parent.name}__{path.parent.name}__{path.stem}"
                if cid in self.cases:
                    raise ValueError("症例ID重複: " + cid)
                self.cases[cid] = dict(case_id=cid, seq=int(match[1]), date=path.parent.parent.name,
                                       chair=path.parent.name, path=path)
        if not self.cases:
            raise ValueError("chair_export 以下に原ログがありません。")
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.legacy = []
        self.legacy_hash = None
        legacy = self.root / "ChairLog_annotations1438_1479.csv"
        if legacy.exists():
            self.legacy_hash = digest(legacy)
            with legacy.open(encoding="utf-8-sig", newline="") as f:
                self.legacy = list(csv.DictReader(f))
        print(json.dumps({"requested_cases": "all exported DAT", "actual_cases": len(self.cases),
                          "source_root": str(self.root / "chair_export"), "output": str(self.output),
                          "legacy_intervals": len(self.legacy), "instrument_count": len(INSTRUMENTS)}, ensure_ascii=False), flush=True)

    def key(self, cid):
        if cid not in self.cases:
            raise ValueError("症例がありません。")
        return hashlib.sha256(cid.encode()).hexdigest()[:24]

    def case(self, cid):
        with self.lock:
            self.key(cid)
            path = self.cases[cid]["path"]
            stat = (path.stat().st_mtime_ns, path.stat().st_size)
            if cid in self.cache and self.cache[cid][0] == stat:
                self.cache.move_to_end(cid)
                return self.cache[cid][1]
            origin, duration, events, invalid = read_log(path)
            if duration <= 0:
                raise ValueError("原ログの時間幅が0秒です。連続区間を作成できません。")
            source_hash = digest(path)
            legacy, legacy_errors = [], []
            for row in self.legacy:
                if str(row.get("seq")) != str(self.cases[cid]["seq"]):
                    continue
                try:
                    if row.get("source_sha256") and row["source_sha256"] != source_hash:
                        raise ValueError("原ログSHA256不一致")
                    s = (datetime.fromisoformat(row["start_datetime"]) - origin).total_seconds()
                    e = (datetime.fromisoformat(row["end_datetime"]) - origin).total_seconds()
                    if not 0 <= s < e <= duration:
                        raise ValueError("原ログの時間範囲外")
                    legacy.append(dict(row, start_s=s, end_s=e))
                except (ValueError, KeyError) as exc:
                    legacy_errors.append({"id": row.get("id"), "reason": str(exc)})
            counts = Counter(e[1] for e in events)
            bin_width = max(1, math.ceil(duration / 2400))
            bins = [[0, 0] for _ in range(int(duration // bin_width) + 1)]
            for t, code, *_ in events:
                bins[int(t // bin_width)][0] += 1
                bins[int(t // bin_width)][1] += int(code in TARGET_CODES)
            case = dict(case_id=cid, seq=self.cases[cid]["seq"], source_sha256=source_hash,
                        source_relative_path=str(path.relative_to(self.root)), time_origin=origin.isoformat(),
                        timezone="Asia/Tokyo", duration_s=duration, events=events,
                        raw_rows=len(events) + len(invalid), invalid_timestamp_rows=invalid,
                        codes=dict(counts.most_common()), legacy=legacy, legacy_errors=legacy_errors,
                        legacy_source_sha256=self.legacy_hash, bin_width_s=bin_width, event_bins=bins)
            self.cache[cid] = (stat, case)
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
            return case

    def document(self, cid):
        case = self.case(cid)
        latest = self.output / "cases" / self.key(cid) / "latest.json"
        if latest.exists():
            doc = json.loads(latest.read_text())
            validate(doc, case)
            return doc
        return dict(schema_version=VERSION, case_id=cid, seq=case["seq"], source_sha256=case["source_sha256"],
                    time_origin=case["time_origin"], timezone=case["timezone"], duration_s=case["duration_s"],
                    scope={"start_s": 0, "end_s": case["duration_s"]}, annotator="", study_role="unassigned",
                    revision=0, media={"filename": "", "offset_s": 0}, intervals=[], created_at=now(), updated_at=None)

    def save(self, doc):
        with self.lock:
            if not isinstance(doc, dict):
                raise ValueError("注釈はJSONオブジェクトが必要です。")
            cid = doc.get("case_id")
            case = self.case(cid)
            qc = validate(doc, case)
            current = self.document(cid)
            if type(doc.get("revision")) is not int or doc["revision"] != current["revision"]:
                raise RuntimeError("別のタブで更新されています。ローカルの下書きJSONを保存してから、症例を再読込してください。")
            folder = self.output / "cases" / self.key(cid)
            folder.mkdir(parents=True, exist_ok=True)
            next_revision = current["revision"] + 1
            while (folder / f"revision_{next_revision:06d}.json").exists():
                next_revision += 1
            saved = dict(doc, revision=next_revision, updated_at=now(), created_at=current["created_at"])
            data = json_bytes(saved)
            version = folder / f"revision_{saved['revision']:06d}.json"
            with version.open("xb") as f:
                f.write(data)
            tmp = folder / (".latest-" + uuid.uuid4().hex)
            tmp.write_bytes(data)
            os.replace(tmp, folder / "latest.json")
            with (self.output / "audit.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "save", "at": now(), "case_id": cid, "revision": saved["revision"],
                                    "annotator": saved["annotator"], "intervals": len(saved["intervals"]),
                                    "document_sha256": hashlib.sha256(data).hexdigest()}, ensure_ascii=False) + "\n")
            return {"document": saved, "quality": qc}

    def export(self, cid, revision, mode):
        with self.lock:
            if mode not in {"draft", "final"}:
                raise ValueError("出力モードが不正です。")
            doc = self.document(cid)
            if doc["revision"] != revision or revision < 1:
                raise RuntimeError("保存済みの最新リビジョンから出力してください。")
            case = self.case(cid)
            if digest(self.cases[cid]["path"]) != case["source_sha256"]:
                raise RuntimeError("原ログが変更されています。再読込してください。")
            qc = validate(doc, case)
            if mode == "final" and not qc["final_export_allowed"]:
                raise ValueError("最終出力できません: " + " / ".join(x["reason"] for x in qc["blockers"][:8]))
            rows = []
            origin = datetime.fromisoformat(doc["time_origin"])
            for order, r in enumerate(sorted(doc["intervals"], key=lambda x: (x["start_s"], x["id"])), 1):
                verified = bool(r["reviewed"] and r["certainty"] == "confirmed" and INDEPENDENT.intersection(r["evidence_sources"]) and r["evidence_detail"].strip())
                row = dict(r, case_id=cid, seq=case["seq"], order=order, source_sha256=case["source_sha256"],
                           start_datetime=(origin + timedelta(seconds=r["start_s"])).isoformat(),
                           end_datetime=(origin + timedelta(seconds=r["end_s"])).isoformat(),
                           duration_s=r["end_s"] - r["start_s"], annotator=doc["annotator"], study_role=doc["study_role"],
                           clinical_ground_truth_eligible=verified and r["clinical_status"] != "unknown",
                           instrument_ground_truth_eligible=verified and r["instrument_status"] != "unknown",
                           revision=revision, export_mode=mode)
                rows.append(row)
            fields = ["case_id", "seq", "id", "order", "start_s", "end_s", "start_datetime", "end_datetime", "duration_s",
                      "clinical_status", "treatment_jp", "instruments", "instrument_status", "instrument_relation",
                      "observation_status", "evidence_sources", "evidence_detail", "certainty", "boundary_certainty",
                      "start_uncertainty_s", "end_uncertainty_s", "reviewed", "notes", "annotator", "study_role",
                      "clinical_ground_truth_eligible", "instrument_ground_truth_eligible", "source_sha256", "revision", "export_mode"]
            manifest = {k: v for k, v in case.items() if k not in {"events", "event_bins", "legacy"}}
            manifest.update(exported_at=now(), schema_version=VERSION, revision=revision, export_mode=mode,
                            scope=doc["scope"], exclusions_outside_scope=[{"start_s": 0, "end_s": doc["scope"]["start_s"]},
                            {"start_s": doc["scope"]["end_s"], "end_s": doc["duration_s"]}],
                            interval_count=len(rows), instrument_vocabulary_sha256=digest(APP / "instruments.json"),
                            notes="Human annotations, not model predictions. Unknown is not no_procedure. Instruments are interval-level, not necessarily active throughout.")
            entries = {"annotations.json": json_bytes(doc), "intervals.csv": csv_data(rows, fields),
                       "quality.json": json_bytes(qc), "manifest.json": json_bytes(manifest),
                       "instruments.json": json_bytes(INSTRUMENTS),
                       "README.txt": ("UTF-8。annotations.json が再読込用の完全データです。intervals.csv は表計算・解析用です。\n"
                        "CSVの配列列はJSONです。式として解釈される文字列には先頭に ' を付けています。原文はJSONに保持。\n"
                        "時刻は原ログ先頭からの秒。区間は [start_s, end_s)。日時はAsia/Tokyo。\n"
                        "unknown は処置なし・器具なしを意味しません。未記入は quality.json の gaps に記録します。\n"
                        "final は編集確認済みの出力を示し、全区間の臨床内容が確定していることは意味しません。\n"
                        "複数器具は同時使用とは限りません。instrument_relation を確認してください。\n"
                        "境界誤差のnullは不明です。対象範囲外はmanifest.jsonに記録します。\n"
                        "症例単位で学習・評価を分け、前後関係の評価に正解ラベルを入力しないでください。\n").encode("utf-8")}
            entries["checksums.json"] = json_bytes({k: hashlib.sha256(v).hexdigest() for k, v in entries.items()})
            folder = self.output / "exports"
            folder.mkdir(exist_ok=True)
            name = f"annotations_seq{case['seq']}_r{revision}_{mode}_{uuid.uuid4().hex[:10]}.zip"
            with zipfile.ZipFile(folder / name, "x", zipfile.ZIP_DEFLATED) as z:
                for filename, content in entries.items():
                    z.writestr(filename, content)
            return {"filename": name, "url": "/exports/" + name, "quality": qc}

    def export_all(self, mode):
        """Export every saved case, never silently excluding an incomplete case."""
        with self.lock:
            if mode not in {"draft", "final"}:
                raise ValueError("出力モードが不正です。")
            docs = [json.loads(p.read_text()) for p in sorted((self.output / "cases").glob("*/latest.json"))]
            if not docs:
                raise ValueError("保存済みの症例がありません。")
            failures = []
            for doc in docs:
                case = self.case(doc["case_id"])
                qc = validate(doc, case)
                if mode == "final" and not qc["final_export_allowed"]:
                    failures.append({"case_id": doc["case_id"], "reasons": [b["reason"] for b in qc["blockers"]]})
            if failures:
                raise ValueError("一括最終出力を中止しました。未完了症例を除外していません: " + json.dumps(failures, ensure_ascii=False))
            folder = self.output / "exports"
            folder.mkdir(exist_ok=True)
            name = f"annotations_collection_{mode}_{uuid.uuid4().hex[:10]}.zip"
            temp = folder / (".pending-" + name)
            rows, case_reports, hashes = [], [], {}
            with zipfile.ZipFile(temp, "x", zipfile.ZIP_DEFLATED) as result:
                fields = None
                for doc in docs:
                    exported = self.export(doc["case_id"], doc["revision"], mode)
                    prefix = f"cases/seq{doc['seq']}_{self.key(doc['case_id'])}/"
                    with zipfile.ZipFile(folder / exported["filename"]) as source:
                        for item in source.namelist():
                            data = source.read(item)
                            result.writestr(prefix + item, data)
                            hashes[prefix + item] = hashlib.sha256(data).hexdigest()
                        reader = csv.DictReader(io.StringIO(source.read("intervals.csv").decode("utf-8-sig")))
                        fields = reader.fieldnames
                        rows.extend(reader)
                    case_reports.append({"case_id": doc["case_id"], "seq": doc["seq"], "revision": doc["revision"],
                                         "interval_count": len(doc["intervals"]), "quality": exported["quality"],
                                         "source_sha256": doc["source_sha256"]})
                csv_text = io.StringIO(newline="")
                writer = csv.DictWriter(csv_text, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
                data = csv_text.getvalue().encode("utf-8-sig")
                result.writestr("all_intervals.csv", data)
                hashes["all_intervals.csv"] = hashlib.sha256(data).hexdigest()
                manifest = dict(schema_version=VERSION, exported_at=now(), mode=mode,
                                selection="All cases with a saved annotation document; unsaved cases are outside this export's scope",
                                available_raw_cases=len(self.cases), requested_saved_cases=len(docs), actual_cases=len(docs),
                                actual_intervals=len(rows), excluded_cases=[], cases=case_reports)
                data = json_bytes(manifest)
                result.writestr("collection_manifest.json", data)
                hashes["collection_manifest.json"] = hashlib.sha256(data).hexdigest()
                result.writestr("checksums.json", json_bytes(hashes))
            os.replace(temp, folder / name)
            return {"filename": name, "url": "/exports/" + name, "case_count": len(docs), "interval_count": len(rows)}


def handler(store, token, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, data, code=200, content_type="application/json; charset=utf-8", disposition=None):
            if not isinstance(data, bytes):
                data = json_bytes(data)
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            if disposition:
                self.send_header("Content-Disposition", disposition)
            self.end_headers()
            self.wfile.write(data)

        def check_host(self):
            if self.headers.get("Host") not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
                raise ValueError("localhost からアクセスしてください。")

        def do_GET(self):
            try:
                self.check_host()
                u = urlparse(self.path)
                q = parse_qs(u.query)
                cid = q.get("id", [""])[0]
                if u.path == "/api/config":
                    self.send(dict(token=token, schema_version=VERSION, instruments=INSTRUMENTS, output=str(store.output),
                                   case_count=len(store.cases), target_codes=sorted(TARGET_CODES)))
                elif u.path == "/api/cases":
                    self.send([{k: v for k, v in c.items() if k != "path"} for c in sorted(store.cases.values(), key=lambda x: (x["seq"], x["case_id"]))])
                elif u.path == "/api/case":
                    case = store.case(cid)
                    doc = store.document(cid)
                    self.send(dict(case={k: v for k, v in case.items() if k != "events"}, document=doc, quality=quality(doc)))
                elif u.path == "/api/signal":
                    case = store.case(cid)
                    start = float(q.get("start", ["0"])[0])
                    end = float(q.get("end", [str(case["duration_s"])])[0])
                    if not number(start) or not number(end) or not 0 <= start < end <= case["duration_s"]:
                        raise ValueError("表示範囲が不正です。")
                    codes = q.get("codes", ["20039,20101,20742"])[0].split(",")[:8]
                    signals = {}
                    for code in codes:
                        selected = [e for e in case["events"] if e[1] == code and start <= e[0] <= end]
                        n = len(selected)
                        if n > 4000:
                            selected = [selected[round(i * (n - 1) / 3999)] for i in range(4000)]
                        signals[code] = dict(points=[[e[0], e[2], e[3], e[4]] for e in selected], total=n, displayed=len(selected))
                    self.send(signals)
                elif u.path.startswith("/exports/"):
                    name = u.path.removeprefix("/exports/")
                    if not re.fullmatch(r"annotations_(seq\d+_r\d+|collection)_(draft|final)_[a-f0-9]{10}\.zip", name):
                        raise ValueError("出力ファイル名が不正です。")
                    self.send((store.output / "exports" / name).read_bytes(), content_type="application/zip", disposition=f'attachment; filename="{name}"')
                else:
                    files = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}
                    if u.path not in files:
                        self.send({"error": "Not found"}, 404)
                        return
                    name, kind = files[u.path]
                    self.send((APP / "static" / name).read_bytes(), content_type=kind)
            except (ValueError, KeyError, FileNotFoundError) as exc:
                self.send({"error": str(exc)}, 400)

        def do_POST(self):
            try:
                self.check_host()
                if self.headers.get("X-Annotation-Token") != token:
                    self.send({"error": "ページを再読み込みしてください。"}, 403)
                    return
                if self.headers.get("Origin") not in {None, f"http://127.0.0.1:{port}", f"http://localhost:{port}"}:
                    self.send({"error": "Origin rejected"}, 403)
                    return
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("JSON形式が必要です。")
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 8 * 1024 * 1024:
                    raise ValueError("JSONは8MB以内にしてください。")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("JSONオブジェクトが必要です。")
                if self.path == "/api/save":
                    self.send(store.save(body))
                elif self.path == "/api/validate":
                    self.send(validate(body, store.case(body.get("case_id"))))
                elif self.path == "/api/export":
                    self.send(store.export(body["case_id"], body["revision"], body["mode"]))
                elif self.path == "/api/export_all":
                    self.send(store.export_all(body["mode"]))
                else:
                    self.send({"error": "Not found"}, 404)
            except RuntimeError as exc:
                self.send({"error": str(exc)}, 409)
            except (ValueError, KeyError, TypeError) as exc:
                self.send({"error": str(exc)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=APP.parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "Output" / "chair_continuous_annotations_v1"
    store = Store(root, output)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(store, secrets.token_urlsafe(32), args.port))
    print(f"Annotation GUI: http://127.0.0.1:{args.port}\nSource logs are read-only. Stop with Ctrl+C.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
