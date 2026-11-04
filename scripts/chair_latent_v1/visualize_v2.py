"""Read-only visual audit of the frozen v2 fresh-case evaluation.

Only runs after test_summary.json. Model selection is read from the pre-test
freeze. Raw replay cases are selected by a fixed seed without inspecting model
errors. The first nonempty masking layout is chosen per case by target support,
never by gains. No numeric interpolation, ffill, or clinical labels are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualize_latent import (inverse_coordinates,hidden_spans,_sample_rows,
                              _read_case_observations,_save,NUMERIC,ROOT)

SEED=20261105
CANDIDATE="signal_specific_latent_local"
BASELINE="baseline_channelwise"


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):
            digest.update(block)
    return digest.hexdigest()


def metric_figures(out,target,index,freeze,dpi):
    cv=pd.read_csv(out/"crossfit_comparison.csv").set_index("model")
    blends=pd.read_csv(out/"crossfit_blends.csv")
    test=pd.read_csv(out/"test_comparison.csv").set_index("model")
    if cv.index.duplicated().any() or test.index.duplicated().any():
        raise ValueError("Duplicate model comparison rows")
    names=[CANDIDATE,BASELINE,"semantic_kernel","linear4","mask_HP"]
    profile={name:{} for name in names}
    for _,signal,j in NUMERIC:
        channel=signal if signal=="RPM" else signal.lower()
        col=channel+"_error";choice=freeze["candidate_by_channel"][channel]
        baseline=cv.loc[choice["baseline"],col]
        profile[BASELINE][col]=baseline
        if choice["type"]=="baseline":
            profile[CANDIDATE][col]=baseline
        else:
            rows=blends[blends.channel.eq(channel)&np.isclose(blends.mix,float(choice["mix"]))]
            if len(rows)!=1:
                raise ValueError("Frozen channel blend not uniquely present in crossfit table")
            profile[CANDIDATE][col]=float(rows.error.iloc[0])
        for name in ["semantic_kernel","linear4","mask_HP"]:
            profile[name][col]=float(cv.loc[name,col])
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    colors=["#C44E52","#55A868","#4C72B0","#8172B3","#CCB974"]
    for c,(_,signal,j) in enumerate(NUMERIC):
        col=(signal if signal=="RPM" else signal.lower())+"_error"
        for row,title,values in [(0,"Original249 case cross-fit",[profile[n][col] for n in names]),
                                 (1,"Fresh case holdout",test.loc[names,col].tolist())]:
            ax=axes[row,c]
            ax.bar(np.arange(len(names)),values,color=colors)
            ax.set_xticks(np.arange(len(names)),["Candidate","Channelwise\nbaseline","Semantic\ncontext","Linear latent","Mask+HP"],fontsize=8)
            ax.set_title(f"{title}: {signal}")
            ax.set_ylabel("Case-mean standardized signal MSE")
            ax.grid(axis="y",alpha=.2)
    fig.suptitle("V2 channel-specific estimates: red = frozen candidate, green = baseline\nRPM fallback is exactly the baseline; other channels follow pre-test cross-fit choices",fontsize=12)
    fig.tight_layout()
    _save(fig,target/"01_crossfit_and_fresh_per_signal.png",index,"comparison",int(test.n_cases.max()),
          "Per-channel original249 cross-fit table and frozen fresh-case test table; candidate cross-fit values read from preselected channel blends.",dpi=dpi)
    fig,ax=plt.subplots(figsize=(10,5))
    models=test.index.tolist()
    ax.barh(np.arange(len(models)),test.error,
            color=["#C44E52" if n==CANDIDATE else "#55A868" if n==BASELINE else "#4C72B0" for n in models])
    ax.set_yticks(np.arange(len(models)),models,fontsize=9)
    ax.invert_yaxis();ax.grid(axis="x",alpha=.2)
    ax.set_xlabel("Case-mean, channel-balanced standardized numeric MSE")
    ax.set_title("Fresh-case evaluation: all frozen evaluated predictors\nLower is better; plotting does not select or refit a model")
    fig.tight_layout()
    _save(fig,target/"02_fresh_case_model_comparison.png",index,"comparison",int(test.n_cases.max()),
          "All evaluated frozen predictors in stored table order, including exact RPM fallback and explicit mask/HP control.",dpi=dpi)


def coverage(out,target,index,ledger,data,dpi):
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    counts=ledger.eligibility_reason.value_counts(sort=False)
    axes[0].barh(np.arange(len(counts)),counts,color="#4C72B0")
    axes[0].set_yticks(np.arange(len(counts)),counts.index,fontsize=7)
    axes[0].set_xlabel("Requested cases")
    axes[0].set_title("All200 retained; explicit evaluation support")
    counts=ledger.signal_status.value_counts(sort=False)
    axes[1].barh(np.arange(len(counts)),counts,color="#55A868")
    axes[1].set_yticks(np.arange(len(counts)),counts.index,fontsize=8)
    axes[1].set_xlabel("Requested cases")
    axes[1].set_title("Extracted source signal status")
    support=np.isfinite(data["raw"][:,:,4:]).sum(1)
    for c,(_,name,_) in enumerate(NUMERIC):
        axes[2].plot(np.arange(1,len(ledger)+1),np.sort(support[:,c]),label=name)
    axes[2].set_xlabel("All cases, separately sorted per signal")
    axes[2].set_ylabel("Observed48-bin summaries (>=3 observations)")
    axes[2].set_title("Numeric support; unknown is not OFF0")
    axes[2].legend(fontsize=8);axes[2].grid(alpha=.15)
    fig.tight_layout()
    _save(fig,target/"03_all_200_case_support.png",index,"coverage",len(ledger),
          "Every requested case retained, with eligibility reason, source signal status and observed numeric support.",dpi=dpi)


def replay(out,target,index,ledger,data,dpi):
    eligible=np.flatnonzero(ledger.eligible.to_numpy(bool))
    selected=np.random.default_rng(SEED).choice(eligible,min(6,len(eligible)),replace=False)
    if len(selected)!=6:
        raise ValueError("Need six eligible fresh cases for fixed random replay")
    scales=np.load(out/"scaling.npz")
    model_arrays={}
    for name in [CANDIDATE,BASELINE]:
        saved=np.load(out/"test_predictions"/(name+".npz"))
        ids=saved["case_indices"].astype(int)
        if len(np.unique(ids))!=len(ids):
            raise ValueError("Duplicate prediction indices")
        model_arrays[name]=(ids,saved["prediction"].reshape(len(ids),4,48,7))
    case_ids=ledger.case_id.iloc[selected].tolist()
    observations=_read_case_observations(out,case_ids)
    intervals=pd.read_csv(out/"hp_intervals.csv",usecols=["case_id","hp","start_s","end_s"])
    selections=[]
    for rank,i in enumerate(selected,1):
        cid=str(ledger.case_id.iloc[i]);duration=float(ledger.duration_s.iloc[i])
        observed=np.isfinite(data["raw"][i,:,4:])
        masks=data["numeric_visible"][i]
        supported=np.array([((~mask[:,None])&observed).any() for mask in masks])
        nonempty=np.flatnonzero(supported)
        if not len(nonempty):
            raise ValueError(f"Eligible case has no hidden numeric targets: {cid}")
        rep=int(nonempty[0])
        visible=masks[rep];hidden=~visible
        x=(np.arange(48)+.5)*duration/48./60.
        predictions={}
        for name,(indices,array) in model_arrays.items():
            match=np.flatnonzero(indices==i)
            if len(match)!=1:
                raise ValueError(f"Frozen prediction missing/duplicate case: {cid}")
            predictions[name]=inverse_coordinates(array[match[0],rep],scales["center"],scales["scale"])
        hp=intervals[intervals.case_id.eq(cid)]
        points=observations[observations.case_id.eq(cid)]
        fig,axes=plt.subplots(4,1,figsize=(12,8.5),sharex=True,
                              gridspec_kw={"height_ratios":[1.1,2,2,2]})
        for state,color in [(0,"#9E9E9E"),(1,"#4C72B0"),(2,"#DD8452"),(3,"#55A868")]:
            part=hp[hp.hp.eq(state)]
            axes[0].broken_barh([(r.start_s/60.,(r.end_s-r.start_s)/60.) for r in part.itertuples()],(state-.35,.7),facecolors=color)
        part=hp[hp.hp.isna()]
        axes[0].broken_barh([(r.start_s/60.,(r.end_s-r.start_s)/60.) for r in part.itertuples()],(-1.35,.7),facecolors="#D8D8D8")
        axes[0].set_yticks([-1,0,1,2,3],["Unknown","OFF","HP1","HP2","HP3"])
        axes[0].set_ylim(-1.7,3.7);axes[0].set_ylabel("Exact HP state")
        axes[0].set_title(f"Random fresh-case replay {rank}/6 | {cid}\nFirst supported mask layout {rep}; grey = hidden numeric bins; full HP state remains conditioning",fontsize=10)
        target_counts={}
        for c,(code,label,j) in enumerate(NUMERIC):
            ax=axes[c+1];raw=points[points.code.eq(code)].sort_values("time_s")
            shown=_sample_rows(raw,1500,SEED+10*rank+c)
            ax.scatter(shown.time_s/60.,shown.value,s=5,alpha=.45,color="#4C72B0",zorder=2,
                       label=f"Raw HP2 observations ({len(shown)}/{len(raw)} shown)")
            targets=hidden&np.isfinite(data["raw"][i,:,j]);target_counts[label]=int(targets.sum())
            ax.scatter(x[targets],data["raw"][i,targets,j],s=30,facecolors="none",edgecolors="black",zorder=4,
                       label=f"Hidden observed bin median ({targets.sum()} targets)")
            for name,color,marker,offset in [(CANDIDATE,"#C44E52","x",-.07),(BASELINE,"#55A868","+",.07)]:
                value=predictions[name][:,j];good=targets&np.isfinite(value)
                ax.scatter(x[good]+offset,value[good],s=35,marker=marker,color=color,zorder=5,
                           label="Candidate estimate" if name==CANDIDATE else "Baseline estimate")
                bad=int(np.sum(targets&(~np.isfinite(value)|(value<0))))
                if bad:
                    ax.text(.99,.97 if name==CANDIDATE else .85,f"{name}: {bad} nonphysical/nonfinite estimates",
                            transform=ax.transAxes,ha="right",va="top",fontsize=7,color=color)
            ax.set_ylabel(label+" (raw units)");ax.grid(alpha=.15)
            ax.legend(fontsize=6.5,loc="upper left",ncol=2)
        for ax in axes:
            for a,b in hidden_spans(visible,duration):ax.axvspan(a,b,color="#999999",alpha=.12,zorder=0)
            ax.set_xlim(0,duration/60.)
        axes[-1].set_xlabel("Minutes since original case start (no time warping)")
        fig.text(.01,.005,"Crosses are generated bin-median estimates, not observations. Raw code timestamps retained; no numeric interpolation/ffill.",fontsize=8)
        fig.tight_layout(rect=(0,.025,1,1))
        filename=f"replay_{rank:02d}_{cid}.png"
        _save(fig,target/filename,index,"raw_replay",1,
              "Fixed-seed random fresh case; first nonempty layout by observed target support, exact HP and raw own-time numeric scatter.",model=CANDIDATE,dpi=dpi)
        selections.append(dict(selection_rank=rank,case_index=int(i),case_id=cid,seed=SEED,
            mask_replicate=rep,RPM_hidden_targets=target_counts["RPM"],torque_hidden_targets=target_counts["Torque"],
            pedal2_hidden_targets=target_counts["Pedal2"],figure=filename,
            case_selection="Uniform random eligible case, no score/gain filtering",
            layout_selection="First nonempty layout by observed target support, never prediction errors"))
    pd.DataFrame(selections).to_csv(target/"replay_selection.csv",index=False)
    (target/"selection_metadata.json").write_text(json.dumps(dict(seed=SEED,candidate=CANDIDATE,baseline=BASELINE,
        selected_case_ids=case_ids,case_selection="Six uniform random eligible fresh cases without replacement",
        layout_selection="First masking layout with any hidden observed target, independent of errors",
        numeric_policy="Raw HP2 per-code observations, own actual timestamps, no interpolation/ffill",
        estimate_policy="Inverse train scaling then RPM*40000/log1p inverse; no physical clipping; crosses only at hidden supported targets",
        max_raw_points_drawn_per_signal=1500,downsampling="Deterministic for drawing only; analysis and targets unchanged"),indent=2)+"\n")


def integrity(out):
    manifest=pd.read_csv(out/"requested_manifest.csv")
    rows=[]
    for row in manifest.itertuples():
        actual=sha(row.source_path);meta=sha(Path(row.source_path).parent/"meta.json")
        rows.append(dict(case_id=row.case_id,source_matches=actual==row.source_sha256,meta_matches=meta==row.meta_sha256,
                         source_sha256=actual,meta_sha256=meta))
    check=pd.DataFrame(rows)
    if len(check)!=200 or not check.source_matches.all() or not check.meta_matches.all():
        raise ValueError("Original source integrity failed after visualization")
    path=out/"process/source_integrity_after_visualization.csv"
    if path.exists():raise FileExistsError(path)
    check.to_csv(path,index=False)
    return len(check)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"Output/chair_latent_v2_20261104")
    parser.add_argument("--dpi",type=int,default=110)
    args=parser.parse_args();out=args.output
    if not (out/"test_summary.json").is_file():
        raise RuntimeError("Wait for frozen one-shot v2 final evaluation before visualizing")
    if (out/"figure_index.csv").exists():
        raise FileExistsError("Preserve completed v2 visual audit")
    summary=json.loads((out/"test_summary.json").read_text())
    freeze=json.loads((out/"selection_freeze.json").read_text())
    ledger=pd.read_csv(out/"case_ledger.csv");data=np.load(out/"data.npz")
    if len(ledger)!=200 or ledger.case_id.duplicated().any() or not np.array_equal(ledger.case_id.to_numpy(str),data["case_ids"]):
        raise ValueError("Expected unique all200 ledger aligned to arrays")
    if int(ledger.eligible.sum())!=summary["eligible_cases"]:
        raise ValueError("Eligible count differs from frozen test summary")
    target=out/"figures";target.mkdir(exist_ok=True);index=[]
    metric_figures(out,target,index,freeze,args.dpi)
    coverage(out,target,index,ledger,data,args.dpi)
    replay(out,target,index,ledger,data,args.dpi)
    verified=integrity(out)
    pd.DataFrame(index).to_csv(out/"figure_index.csv",index=False)
    print(json.dumps(dict(figures=len(index),raw_replay_cases=6,selection_seed=SEED,
        original_source_cases_verified=verified,source_hashes_unchanged=True,
        figure_index=str(out/"figure_index.csv"))))


if __name__=="__main__":main()
