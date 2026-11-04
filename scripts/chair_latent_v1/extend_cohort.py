"""Metadata-only cohort planning, then frozen extraction of new v2 cases.

``plan`` reads case-summary metadata and meta.json, and streams file hashes;
it never parses numeric signal observations. ``extract`` MUST see the v2
selection_freeze.json before opening any event CSV. No replacement cases or
partial successful cohort are silently exported. Original data and frozen v1
artifacts are read-only throughout.

Method change from v1: new same-chair cases are selected outside the original
300 by metadata, not the v3 HP2-pattern cohort. Numeric evaluation eligibility
is >=1 supported RPM macro bin AND a known HP2 run. Every requested case stays
in the ledger, including insufficient support and no-target cases.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT/"Output/chair_latent_v2_20261104"
MODEL_ROOT = ROOT/"Output/chair_model_A_results"
ORIGINAL_MANIFEST = ROOT/"Output/chair_hp_sequence_300_v3_20261104/case_manifest.csv"
SUMMARY = MODEL_ROOT/"case_summary.csv"
EXTRACTOR = ROOT/"chair-activity/scripts/chair_hp_sequence_v3/extract_hp_streams.py"
CHAIR = "C512KC1503"
MIN_DURATION = 1549.35
MAX_DURATION = 2597.65
REQUESTED = 200
SEED = 20261105
BINS = 48
CHANNELS = ["OFF_share", "HP1_share", "HP2_share", "HP3_share", "RPM", "torque", "pedal2"]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1048576),b""):
            digest.update(chunk)
    return digest.hexdigest()


def log(out,event,**fields):
    entry=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),event=event,**fields)
    (out/"process").mkdir(parents=True,exist_ok=True)
    with (out/"process/cohort_extension_log.jsonl").open("a") as stream:
        stream.write(json.dumps(entry,ensure_ascii=False)+"\n")
    print(json.dumps(entry,ensure_ascii=False),flush=True)


def _write_json(path,data):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")


def plan(out):
    out.mkdir(parents=True,exist_ok=True)
    (out/"process").mkdir(exist_ok=True)
    if (out/"requested_manifest.csv").exists() or (out/"selection_plan.json").exists():
        raise FileExistsError("Preserve existing cohort plan; use a new output version")
    for path in [SUMMARY,ORIGINAL_MANIFEST,EXTRACTOR]:
        if not path.is_file():
            raise FileNotFoundError(path)
    original=pd.read_csv(ORIGINAL_MANIFEST)
    summary=pd.read_csv(SUMMARY,encoding="utf-8-sig")
    if len(original)!=300 or original.case_id.duplicated().any() or len(summary)!=2914:
        raise ValueError("Original300 / source2914 scope differs from requested plan")
    if not np.isclose(original.duration_s.min(),MIN_DURATION) or not np.isclose(original.duration_s.max(),MAX_DURATION):
        raise ValueError("Original duration range differs; do not silently change selection")
    old_ids=set(original.case_id.astype(str))
    old_hashes=set(original.source_sha256.astype(str))
    summary["duration_s"]=(pd.to_datetime(summary.end,format="mixed")-
                            pd.to_datetime(summary.start,format="mixed")).dt.total_seconds()
    summary["case_id"]=summary.date_folder.astype(str)+"__"+summary.chair_folder.astype(str)+"__Chair_log_seq_"+summary.seq.astype(str)
    if summary.case_id.duplicated().any():
        raise ValueError("Duplicate summary case IDs")
    metadata_candidate=(summary.status.str.startswith("ok",na=False)&summary.chair_folder.eq(CHAIR)&
                        summary.duration_s.between(MIN_DURATION,MAX_DURATION)&~summary.case_id.isin(old_ids))
    candidates=summary.loc[metadata_candidate].copy()
    log(out,"metadata_preflight",source_summary_cases=len(summary),original_cases=len(original),
        metadata_candidates=len(candidates),requested_new_cases=REQUESTED,seed=SEED,
        numeric_signal_values_read=False)
    if len(candidates)!=343:
        raise ValueError(f"Expected343 metadata candidates; actual {len(candidates)}")
    records=[]
    for row in candidates.itertuples():
        folder=MODEL_ROOT/"cases"/str(row.date_folder)/str(row.chair_folder)/("Chair_log_seq_"+str(row.seq))
        source=folder/"events_original.csv.gz";meta_path=folder/"meta.json"
        if not source.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"Planned candidate files missing: {row.case_id}")
        # meta.json contains already-computed case boundaries, not signal values.
        meta=json.loads(meta_path.read_text())
        if not np.isclose(float(meta["duration_s"]),float(row.duration_s)):
            raise ValueError(f"Metadata duration mismatch: {row.case_id}")
        if pd.Timestamp(meta["start"])!=pd.Timestamp(row.start) or pd.Timestamp(meta["end"])!=pd.Timestamp(row.end):
            raise ValueError(f"Metadata time-axis mismatch: {row.case_id}")
        source_hash=sha(source)
        records.append(dict(case_id=str(row.case_id),date=str(row.date_folder),chair=str(row.chair_folder),
            seq=int(row.seq),duration_s=float(row.duration_s),start=str(meta["start"]),end=str(meta["end"]),
            source_path=str(source.resolve()),signal_status="UNAUDITED",source_sha256=source_hash,
            meta_sha256=sha(meta_path),source_summary_status=str(row.status),
            excluded_original_hash=source_hash in old_hashes))
    candidate_manifest=pd.DataFrame(records)
    if candidate_manifest.source_sha256.duplicated().any():
        duplicates=candidate_manifest.loc[candidate_manifest.source_sha256.duplicated(False),["case_id","source_sha256"]]
        duplicates.to_csv(out/"process/duplicate_candidate_hashes.csv",index=False)
        raise ValueError("Duplicate candidate source hashes; stop before random selection")
    eligible=candidate_manifest.loc[~candidate_manifest.excluded_original_hash].copy()
    if len(eligible)<REQUESTED:
        raise ValueError(f"Only {len(eligible)} unique nonoriginal cases for requested{REQUESTED}")
    chosen=np.random.default_rng(SEED).choice(len(eligible),REQUESTED,replace=False)
    requested=eligible.iloc[chosen].copy().reset_index(drop=True)
    requested["selected_index"]=np.arange(len(requested))
    requested=requested[["case_id","date","chair","seq","duration_s","start","end","source_path",
                         "selected_index","signal_status","source_sha256","meta_sha256"]]
    if len(requested)!=REQUESTED or requested.case_id.duplicated().any() or set(requested.case_id)&old_ids:
        raise ValueError("Requested cohort identity/count validation failed")
    if set(requested.source_sha256)&old_hashes:
        raise ValueError("Original source hash overlap")
    candidate_manifest["selected"]=candidate_manifest.case_id.isin(requested.case_id)
    candidate_manifest["selection_reason"]=np.select(
        [candidate_manifest.excluded_original_hash,candidate_manifest.selected],
        ["EXCLUDED_ORIGINAL_SOURCE_HASH","SELECTED_FIXED_RANDOM_SEED_20261105"],
        default="NOT_SELECTED_RANDOM_SAMPLE_NO_REPLACEMENT")
    requested.to_csv(out/"requested_manifest.csv",index=False)
    candidate_manifest.to_csv(out/"candidate_selection_ledger.csv",index=False)
    selection=dict(version="chair-latent-v2-new-cohort-plan",created_utc=datetime.now(timezone.utc).isoformat(),
        source_summary_cases=len(summary),metadata_candidates=len(candidates),
        candidates_after_original_hash_exclusion=len(eligible),requested_cases=REQUESTED,seed=SEED,
        selection="Uniform random200 without replacement from eligible metadata candidates; no numeric values inspected",
        chair=CHAIR,duration_inclusive_s=[MIN_DURATION,MAX_DURATION],original_cases_excluded=300,
        original_case_id_and_hash_overlap=0,numeric_signal_values_read=False,
        metadata_and_byte_hashes_only=True,requested_manifest_sha256=sha(out/"requested_manifest.csv"),
        candidate_ledger_sha256=sha(out/"candidate_selection_ledger.csv"),
        source_summary_sha256=sha(SUMMARY),original_manifest_sha256=sha(ORIGINAL_MANIFEST),
        extractor_path=str(EXTRACTOR),extractor_sha256=sha(EXTRACTOR),
        required_before_numeric_read=str((out/"selection_freeze.json").resolve()),
        numeric_eligibility="NEW v2 support cohort: >=1 supported RPM macro bin and >=1 known HP2 run; not v3 pattern eligibility",
        requested_case_ids=requested.case_id.tolist(),all_metadata_candidate_case_ids=candidate_manifest.case_id.tolist(),
        original_data_readonly=True)
    _write_json(out/"selection_plan.json",selection)
    log(out,"cohort_plan_complete",requested_cases=len(requested),metadata_candidates=len(candidates),
        original_hash_exclusions=int(candidate_manifest.excluded_original_hash.sum()),numeric_signal_values_read=False,
        requested_manifest_sha256=selection["requested_manifest_sha256"])


def _load_extractor():
    spec=importlib.util.spec_from_file_location("chair_hp_v3_frozen_extractor",EXTRACTOR)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def _extract_case(record):
    # Worker independently enforces the freeze's existence before loading data.
    freeze=Path(record.pop("_required_freeze_path"))
    if not freeze.is_file():
        raise RuntimeError("Required v2 selection freeze disappeared")
    return _load_extractor().extract_case(record)


def _macro_export(staging,manifest):
    intervals=pd.read_csv(staging/"hp_intervals.csv")
    runs=pd.read_csv(staging/"hp_runs.csv")
    audit=pd.read_csv(staging/"hp_status_audit.csv").set_index("case_id")
    observation_columns=["case_id","code","time_s","value","hp","run_id","interval_id","second_bin"]
    parts=[]
    for part in pd.read_csv(staging/"observations.csv",usecols=observation_columns,dtype={"code":str},chunksize=50000):
        part=part[part.hp.eq(2)&part.code.isin(["20742","20743","10015"])]
        if len(part):parts.append(part)
    obs=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=observation_columns)
    if len(obs) and obs.duplicated(["case_id","code","interval_id","second_bin"]).any():
        raise ValueError("Duplicate observations in new cohort")
    duration=manifest.set_index("case_id").duration_s
    if len(obs):
        obs["progress_bin"]=np.floor(obs.time_s/obs.case_id.map(duration)*BINS).astype(int).clip(0,BINS-1)
    else:obs["progress_bin"]=pd.Series(dtype=int)
    raw=np.full((len(manifest),BINS,7),np.nan);support=np.zeros_like(raw)
    ig={key:value for key,value in intervals.groupby("case_id",sort=False)}
    og={key:value for key,value in obs.groupby("case_id",sort=False)}
    run_counts=runs[runs.hp.eq(2)].groupby("case_id").size()
    for case_index,row in enumerate(manifest.itertuples()):
        ints=ig[row.case_id];duration_s=float(row.duration_s)
        if not np.isclose((ints.end_s-ints.start_s).sum(),duration_s):
            raise ValueError(f"State interval duration mismatch: {row.case_id}")
        edges=np.linspace(0,duration_s,BINS+1);width=duration_s/BINS
        known=np.zeros(BINS);state=np.zeros((BINS,4))
        for interval in ints.itertuples():
            overlap=np.maximum(0,np.minimum(edges[1:],interval.end_s)-np.maximum(edges[:-1],interval.start_s))
            if np.isfinite(interval.hp):
                known+=overlap;state[:,int(interval.hp)]+=overlap
        good=known>=.5*width
        raw[case_index,good,:4]=state[good]/known[good,None]
        support[case_index,:,:4]=known[:,None]
        points=og.get(row.case_id)
        if points is None:continue
        for (code,bin_index),values in points.groupby(["code","progress_bin"]):
            channel={"20742":4,"20743":5,"10015":6}[code]
            support[case_index,bin_index,channel]=len(values)
            if len(values)>=3:
                raw[case_index,bin_index,channel]=float(values.value.median())
    ledger=manifest[["case_id","date","chair","duration_s","signal_status"]].copy()
    ledger["hp2_run_count"]=ledger.case_id.map(run_counts).fillna(0).astype(int)
    for channel,name in enumerate(CHANNELS):
        ledger[name+"_observed_bins"]=np.isfinite(raw[:,:,channel]).sum(1)
    ledger["eligible"]=(ledger.hp2_run_count.gt(0)&ledger.RPM_observed_bins.gt(0))
    ledger["eligibility_reason"]=np.select(
        [ledger.signal_status.eq("NO_TARGET_SIGNAL"),ledger.hp2_run_count.eq(0),ledger.RPM_observed_bins.eq(0)],
        ["NO_TARGET_SIGNAL","NO_KNOWN_HP2_RUN","NO_SUPPORTED_RPM_MACRO_BIN"],
        default="ELIGIBLE_V2_NEW_NUMERIC_SUPPORT")
    ledger["split"]=np.where(ledger.eligible,"new_case_test","OUTSIDE_V2_NUMERIC_EVALUATION")
    ledger["support_definition"]="v2: >=1 RPM bin with >=3 observations and >=1 known HP2 run"
    # Exact same complementary4 / three-bin hiding construction as v1, with a
    # separately declared seed for new cases. No final-test metric is read here.
    masks=np.ones((len(manifest),4,BINS),bool)
    for case_index in range(len(manifest)):
        order=np.random.default_rng(SEED+100+case_index).permutation(BINS//3)
        for rep in range(4):
            chunks=order[rep*4:(rep+1)*4]
            hidden=np.concatenate([np.arange(3*k,3*k+3) for k in chunks])
            masks[case_index,rep,hidden]=False
        if not ((~masks[case_index]).sum(0)==1).all():
            raise ValueError("Complementary hiding layout failed")
    if ledger.case_id.duplicated().any() or len(ledger)!=REQUESTED:
        raise ValueError("New case ledger identity mismatch")
    np.savez_compressed(staging/"data.npz",raw=raw,support=support,numeric_visible=masks,
                        case_ids=manifest.case_id.to_numpy(str),channels=np.array(CHANNELS))
    ledger.to_csv(staging/"case_ledger.csv",index=False)
    manifest.to_csv(staging/"case_manifest.csv",index=False)
    return dict(actual_cases=len(ledger),eligible_cases=int(ledger.eligible.sum()),
        eligibility_counts=ledger.eligibility_reason.value_counts().to_dict(),
        excluded_case_ids=ledger.loc[~ledger.eligible,"case_id"].tolist(),
        n_hp_intervals=len(intervals),n_hp_runs=len(runs),n_hp2_runs=int(runs.hp.eq(2).sum()),
        exported_numeric_observations=len(obs),raw_shape=list(raw.shape),
        numeric_observed_bins=np.isfinite(raw[:,:,4:]).sum((0,1)).tolist(),
        audit_cases=len(audit),all_requested_cases_retained=True)


def extract(out,workers):
    freeze_path=out/"selection_freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("V2 selection_freeze.json required before extracting ANY numeric observations")
    frozen=json.loads(freeze_path.read_text())
    selection=json.loads((out/"selection_plan.json").read_text())
    if sha(out/"requested_manifest.csv")!=selection["requested_manifest_sha256"]:
        raise ValueError("Requested manifest differs from metadata-only selection plan")
    expected_manifest=frozen.get("fresh_manifest_sha256",frozen.get("requested_manifest_sha256"))
    if expected_manifest is None:
        raise ValueError("Selection freeze must bind fresh_manifest_sha256 or requested_manifest_sha256")
    if expected_manifest!=selection["requested_manifest_sha256"]:
        raise ValueError("Selection freeze requested-manifest hash mismatch")
    if "fresh_requested_cases" in frozen and frozen["fresh_requested_cases"]!=REQUESTED:
        raise ValueError("Freeze requested count differs from200")
    if sha(EXTRACTOR)!=selection["extractor_sha256"] or sha(SUMMARY)!=selection["source_summary_sha256"] or sha(ORIGINAL_MANIFEST)!=selection["original_manifest_sha256"]:
        raise ValueError("Metadata/extractor dependencies changed since plan")
    names=["hp_intervals.csv","hp_runs.csv","observations.csv","transition_audit.csv","hp_status_audit.csv",
           "case_manifest.csv","case_ledger.csv","data.npz","extraction_metadata.json"]
    if any((out/name).exists() for name in names):
        raise FileExistsError("Preserve completed/partial cohort output; incompatible output is never reused")
    staging=out/"process/extraction_staging"
    if staging.exists():
        raise FileExistsError("Previous extraction staging exists; inspect failures instead of silently reusing")
    staging.mkdir(parents=True)
    manifest=pd.read_csv(out/"requested_manifest.csv")
    if len(manifest)!=REQUESTED or manifest.case_id.duplicated().any():
        raise ValueError("Expected200 unique requested cases")
    if "fresh_ids" in frozen and frozen["fresh_ids"]!=manifest.case_id.tolist():
        raise ValueError("Frozen fresh case IDs/order differ from requested manifest")
    old=pd.read_csv(ORIGINAL_MANIFEST)
    if set(manifest.case_id)&set(old.case_id) or set(manifest.source_sha256)&set(old.source_sha256):
        raise ValueError("Original case/source overlap")
    before=[]
    for row in manifest.itertuples():
        source_hash=sha(row.source_path);meta_hash=sha(Path(row.source_path).parent/"meta.json")
        if source_hash!=row.source_sha256 or meta_hash!=row.meta_sha256:
            raise ValueError(f"Raw/meta source changed before extraction: {row.case_id}")
        before.append(dict(case_id=row.case_id,source_sha256=source_hash,meta_sha256=meta_hash))
    pd.DataFrame(before).to_csv(out/"process/new_source_integrity_before.csv",index=False)
    log(out,"extraction_preflight",requested_cases=REQUESTED,freeze_sha256=sha(freeze_path),
        requested_manifest_sha256=sha(out/"requested_manifest.csv"),workers=workers,
        numeric_eligibility="new v2 support, not original v3 pattern eligibility")
    module=_load_extractor();synthetic=module.synthetic_validation()
    columns=[module.INTERVAL_COLS,module.RUN_COLS,module.OBS_COLS,module.TRANSITION_COLS]
    table_names=["hp_intervals.csv","hp_runs.csv","observations.csv","transition_audit.csv"]
    for name,cols in zip(table_names,columns):
        pd.DataFrame(columns=cols).to_csv(staging/name,index=False)
    audits=[];failures=[];case_results=[]
    records=manifest.to_dict("records")
    for record in records:record["_required_freeze_path"]=str(freeze_path.resolve())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(_extract_case,record):record["case_id"] for record in records}
        for count,future in enumerate(as_completed(jobs),1):
            cid=jobs[future]
            try:
                result=future.result()
                for name,cols,values in zip(table_names,columns,result[:4]):
                    if values:pd.DataFrame(values,columns=cols).to_csv(staging/name,index=False,header=False,mode="a")
                audits.append(result[4]);case_results.append(dict(case_id=cid,status="EXTRACTED",reason=""))
            except Exception as exc:
                failures.append(dict(case_id=cid,reason=repr(exc)))
                case_results.append(dict(case_id=cid,status="FAILED",reason=repr(exc)))
            if count%25==0:
                log(out,"extraction_progress",completed=count,requested_cases=REQUESTED,failures=len(failures))
    pd.DataFrame(case_results).to_csv(out/"process/case_extraction_audit.csv",index=False)
    if failures:
        _write_json(out/"process/extraction_failures.json",dict(requested_cases=REQUESTED,successes=len(audits),failures=failures,
            partial_export_reused=False,final_cohort_exported=False))
        raise RuntimeError(f"{len(failures)} requested cases failed; no partial cohort promoted or replacement selected")
    audit=pd.DataFrame(audits).set_index("case_id").loc[manifest.case_id].reset_index()
    if len(audit)!=REQUESTED or audit.case_id.duplicated().any():
        raise ValueError("Extracted audit case mismatch")
    audit["signal_status"]=np.where(audit.n_target_events.eq(0),"NO_TARGET_SIGNAL","OK_TARGET_SIGNAL")
    audit.to_csv(staging/"hp_status_audit.csv",index=False)
    manifest["signal_status"]=audit.signal_status.to_numpy()
    manifest["n_target_events"]=audit.n_target_events.to_numpy()
    manifest["n_recorded_target_codes"]=audit.n_target_codes.to_numpy()
    for name,keys in [("hp_intervals.csv",["case_id","interval_id"]),("hp_runs.csv",["case_id","run_id"]),
                      ("observations.csv",["case_id","code","interval_id","second_bin"])]:
        keys_frame=pd.read_csv(staging/name,usecols=keys)
        if keys_frame.duplicated(keys).any() or not set(keys_frame.case_id)<=set(manifest.case_id):
            raise ValueError(f"Duplicate/incompatible compact keys: {name}")
    details=_macro_export(staging,manifest)
    after=[]
    for row in manifest.itertuples():
        source_hash=sha(row.source_path);meta_hash=sha(Path(row.source_path).parent/"meta.json")
        if source_hash!=row.source_sha256 or meta_hash!=row.meta_sha256:
            raise ValueError(f"Raw/meta source changed after extraction: {row.case_id}")
        after.append(dict(case_id=row.case_id,source_sha256=source_hash,meta_sha256=meta_hash))
    pd.DataFrame(after).to_csv(out/"process/new_source_integrity_after.csv",index=False)
    if before!=after:
        raise ValueError("Source integrity changed during extraction")
    metadata=dict(version="chair-latent-v2-new-cohort-extraction",requested_cases=REQUESTED,**details,
        requested_manifest_sha256=sha(out/"requested_manifest.csv"),selection_freeze_sha256=sha(freeze_path),
        extractor_sha256=sha(EXTRACTOR),exporter_sha256=sha(__file__),seed=SEED,bins=BINS,channels=CHANNELS,
        source_hashes_unchanged=True,original300_case_and_source_overlap=0,
        synthetic_validation=synthetic,scientific_unit="whole original first90012ON-to-finalOFF case",
        numeric_policy="HP2 observed-only per-code median per48 progress bins; >=3 original run/second observations; own actual code timestamps",
        hp_policy="Exact HP interval overlaps; >=50% bin duration known; occupancy normalized by known duration",
        missing="NaN observations, explicit mask/support; no numeric ffill/interpolation; OFF0 differs from unknown",
        eligibility="NEW v2 numeric support: >=1 supported RPM macro bin and >=1 known HP2 run, NOT v3 pattern status",
        hiding="Four complementary layouts; contiguous3binchunks;25% numeric bins hidden each; seed20261105+100+caseindex",
        selection="Metadata-only fixed random200 from343 same-chair duration-matched candidates, no replacement",
        clinical_or_external_patient_validation=False,original_data_readonly=True,
        method_differences_from_v1=["Fresh same-chair cohort outside original300 cases and source hashes",
            "Eligibility uses directly supported RPM macro bins plus HP2 run, not v3 pattern-block eligibility",
            "Separate declared masking seed20261105; original frozen v1 outputs remain unchanged",
            "No hp2_points or clinical/pattern labels generated; raw compact observations remain auditable"])
    _write_json(staging/"extraction_metadata.json",metadata)
    for name in names:
        shutil.move(str(staging/name),str(out/name))
    log(out,"extraction_complete",**details,source_hashes_unchanged=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["plan","extract"])
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if args.workers<1:
        parser.error("workers must be positive")
    if args.stage=="plan":plan(args.output)
    else:extract(args.output,args.workers)


if __name__=="__main__":
    main()
