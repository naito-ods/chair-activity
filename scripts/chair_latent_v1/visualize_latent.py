"""Read-only visual audit of a frozen annotation-free latent experiment.

Run only after ``test_summary.json`` exists. This script selects no models and
does not tune or score them. Six test cases are selected with a fixed random
seed without inspecting gains. Generated estimates and raw observations use
different markers. Numeric observations are never filled or interpolated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT/"Output/chair_latent_v1_20261104"
DEFAULT_SOURCE = ROOT/"Output/chair_hp_sequence_300_v3_20261104"
SEED = 20261104
NUMERIC = [("20742", "RPM", 4), ("20743", "Torque", 5), ("10015", "Pedal2", 6)]


def inverse_coordinates(prediction, center, scale):
    """Invert model coordinates without clipping nonphysical estimates."""
    value = np.asarray(prediction, float).copy()*scale+center
    value[..., 4] *= 40000.
    with np.errstate(over="ignore", invalid="ignore"):
        value[..., 5:7] = np.expm1(value[..., 5:7])
    return value


def hidden_spans(mask, duration):
    hidden = ~np.asarray(mask, bool)
    padded = np.r_[False, hidden, False].astype(int)
    starts = np.flatnonzero(np.diff(padded)==1)
    ends = np.flatnonzero(np.diff(padded)==-1)
    return [(a*duration/len(hidden)/60., b*duration/len(hidden)/60.) for a,b in zip(starts,ends)]


def _sample_rows(frame, maximum, seed):
    if len(frame)<=maximum:
        return frame
    ix = np.sort(np.random.default_rng(seed).choice(len(frame), maximum, replace=False))
    return frame.iloc[ix]


def _read_case_observations(source, case_ids):
    pieces = []
    columns = ["case_id", "code", "time_s", "value", "hp"]
    for frame in pd.read_csv(source/"observations.csv", usecols=columns,
                             dtype={"code": str}, chunksize=50000):
        frame = frame[frame.case_id.isin(case_ids)&frame.hp.eq(2)&frame.code.isin([v[0] for v in NUMERIC])]
        if len(frame):
            pieces.append(frame)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)


def _save(fig, path, index, kind, n_cases, description, model="", dpi=110):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    index.append(dict(figure=path.name, path=str(Path("figures")/path.name), kind=kind,
                      n_cases=n_cases, model=model, description=description))


def comparisons(out, target, summary, freeze, index, dpi):
    development = pd.read_csv(out/"development_comparison.csv")
    test = pd.read_csv(out/"test_comparison.csv")
    if development.model.duplicated().any() or test.model.duplicated().any():
        raise ValueError("Duplicate model rows in comparison tables")
    # Selection within each family is from validation only, never test scores.
    chosen = development.loc[development.groupby("family").validation_error.idxmin()]
    chosen = chosen.sort_values(["validation_error", "model"])
    candidate = summary["candidate"]
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6., len(chosen)*.33)))
    ax = axes[0]
    colors = ["#C44E52" if r.model==candidate else "#4C72B0" for r in chosen.itertuples()]
    ax.barh(np.arange(len(chosen)), chosen.validation_error, color=colors)
    ax.set_yticks(np.arange(len(chosen)), [f"{r.family}: {r.model}" for r in chosen.itertuples()], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Development: best validation model per family")
    ax.set_xlabel("Case-mean standardized numeric MSE (lower is better)")
    ax.grid(axis="x", alpha=.2)
    ax = axes[1]
    colors = ["#C44E52" if name==candidate else "#55A868" if name=="baseline_channelwise" else "#8172B3" for name in test.model]
    ax.barh(np.arange(len(test)), test.test_error, color=colors)
    ax.set_yticks(np.arange(len(test)), test.model, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Final case holdout: all frozen evaluated models")
    ax.set_xlabel("Case-mean standardized numeric MSE (lower is better)")
    ax.grid(axis="x", alpha=.2)
    fig.suptitle("Annotation-free numeric reconstruction comparison\nModel choices frozen before final test; red = candidate, green = channelwise baseline", fontsize=12)
    fig.tight_layout()
    _save(fig, target/"01_model_comparison.png", index, "comparison", int(test.n_cases.max()),
          "Validation family minima selected on validation only; all frozen final-test models displayed without selecting by test scores.", dpi=dpi)
    # Per-channel view prevents a combined score from hiding a signal failure.
    names = list(dict.fromkeys([candidate, "baseline_channelwise", freeze["choices"]["semantic"],
                                freeze["choices"]["linear"], freeze["choices"]["mask_HP"]]))
    chosen_test = test.set_index("model").loc[names]
    fig, axes = plt.subplots(1,3,figsize=(14,4.5))
    for ax, (_, signal, _) in zip(axes, NUMERIC):
        column = signal+"_error" if signal=="RPM" else signal.lower()+"_error"
        ax.bar(np.arange(len(names)), chosen_test[column], color=["#C44E52" if n==candidate else "#55A868" if n=="baseline_channelwise" else "#4C72B0" for n in names])
        ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right", fontsize=7)
        ax.set_title(signal)
        ax.set_ylabel("Case-mean standardized MSE")
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Same hidden observed targets, separated by numeric signal")
    fig.tight_layout()
    _save(fig,target/"02_numeric_channel_comparison.png",index,"comparison",int(test.n_cases.max()),
          "Candidate, channelwise baseline, and frozen semantic/linear/mask-HP controls, separated by signal.",dpi=dpi)


def coverage(ledger, data, target, index, dpi):
    fig, axes = plt.subplots(1,3,figsize=(15,4.5))
    status = ledger.hp2_source_status.value_counts(sort=False)
    axes[0].barh(np.arange(len(status)),status,color="#4C72B0")
    axes[0].set_yticks(np.arange(len(status)),status.index,fontsize=8)
    axes[0].set_xlabel("Cases")
    axes[0].set_title("All 300 cases retained in audit")
    counts=ledger.split.value_counts(sort=False)
    axes[1].barh(np.arange(len(counts)),counts,color="#55A868")
    axes[1].set_yticks(np.arange(len(counts)),counts.index,fontsize=8)
    axes[1].set_xlabel("Cases")
    axes[1].set_title("Frozen case split / explicit non-evaluation")
    observed=np.isfinite(data["raw"][:,:,4:]).sum(axis=1)
    for c,(_,name,_) in enumerate(NUMERIC):
        values=np.sort(observed[:,c])
        axes[2].plot(np.arange(1,len(values)+1),values,label=name)
    axes[2].set_xlabel("All cases, separately sorted by signal support")
    axes[2].set_ylabel("Observed macro bins (>=3 observations)")
    axes[2].set_title("Numeric support; missing stays missing")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    _save(fig,target/"03_all_case_support.png",index,"coverage",len(ledger),
          "Source status, frozen split, and observed numeric bin support for every case. Curves use separately sorted cases per signal.",dpi=dpi)


def replay(out, source, target, summary, data, ledger, index, dpi):
    test_indices = np.flatnonzero(ledger.split.eq("test"))
    if len(test_indices)<6:
        raise ValueError("Need at least six test cases for predetermined random replay")
    # Selection depends only on fixed split, manifest order, and seed.
    selected = np.random.default_rng(SEED).choice(test_indices,6,replace=False)
    candidate = summary["candidate"]
    models = [candidate,"baseline_channelwise"]
    predictions = {}
    for name in models:
        stored=np.load(out/"test_predictions"/(name+".npz"))
        indices=stored["case_indices"].astype(int)
        if len(set(indices.tolist()))!=len(indices):
            raise ValueError("Duplicate prediction case indices")
        a=stored["prediction"].reshape(len(indices),4,data["raw"].shape[1],7)
        predictions[name]=(indices,a)
    scales=np.load(out/"scaling.npz")
    ids=ledger.case_id.iloc[selected].tolist()
    observations=_read_case_observations(source,ids)
    intervals=pd.read_csv(source/"hp_intervals.csv",usecols=["case_id","start_s","end_s","hp"])
    intervals=intervals[intervals.case_id.isin(ids)]
    selected_rows=[]
    for rank,case_index in enumerate(selected,1):
        row=ledger.iloc[case_index];cid=str(row.case_id);duration=float(row.duration_s)
        points=observations[observations.case_id.eq(cid)]
        hp=intervals[intervals.case_id.eq(cid)]
        first_visible=data["numeric_visible"][case_index,0]
        hidden=~first_visible
        bins=len(hidden);x=(np.arange(bins)+.5)*duration/bins/60.
        model_predictions={}
        for name,(indices,values) in predictions.items():
            matches=np.flatnonzero(indices==case_index)
            if len(matches)!=1:
                raise ValueError(f"Prediction case mismatch for {cid}: {name}")
            model_predictions[name]=inverse_coordinates(values[matches[0],0],scales["center"],scales["scale"])
        fig,axes=plt.subplots(4,1,figsize=(12,8.5),sharex=True,gridspec_kw={"height_ratios":[1.1,2,2,2]})
        colors={0:"#9E9E9E",1:"#4C72B0",2:"#DD8452",3:"#55A868"}
        for state in [0,1,2,3]:
            q=hp[hp.hp.eq(state)]
            axes[0].broken_barh([(r.start_s/60.,(r.end_s-r.start_s)/60.) for r in q.itertuples()],
                                (state-.35,.7),facecolors=colors[state])
        q=hp[hp.hp.isna()]
        axes[0].broken_barh([(r.start_s/60.,(r.end_s-r.start_s)/60.) for r in q.itertuples()],
                            (-1.35,.7),facecolors="#D8D8D8")
        axes[0].set_yticks([-1,0,1,2,3],["Unknown","OFF","HP1","HP2","HP3"])
        axes[0].set_ylim(-1.7,3.7)
        axes[0].set_ylabel("Exact HP state")
        axes[0].set_title(f"Random test replay {rank}/6 | {cid}\nReal time; grey bands = hidden numeric bins (first mask layout); HP remains allowed conditioning",fontsize=10)
        for c,(code,label,j) in enumerate(NUMERIC):
            ax=axes[c+1]
            q=points[points.code.eq(code)].sort_values("time_s")
            display=_sample_rows(q,1500,SEED+rank*10+c)
            ax.scatter(display.time_s/60.,display.value,s=5,color="#4C72B0",alpha=.45,label=f"Raw HP2 observations ({len(display)}/{len(q)} shown)",zorder=2)
            observed=np.isfinite(data["raw"][case_index,:,j])
            targets=hidden&observed
            ax.scatter(x[targets],data["raw"][case_index,targets,j],s=30,facecolors="none",edgecolors="black",
                       label="Hidden observed bin median (target)",zorder=4)
            for name,color,marker,offset in [(candidate,"#C44E52","x",-.07),("baseline_channelwise","#55A868","+",.07)]:
                estimate=model_predictions[name][:,j]
                finite=targets&np.isfinite(estimate)
                ax.scatter(x[finite]+offset,estimate[finite],s=35,marker=marker,color=color,
                           label=("Candidate model estimate" if name==candidate else "Baseline model estimate"),zorder=5)
                nonphysical=int(np.sum(targets&np.isfinite(estimate)&(estimate<0)))
                nonfinite=int(np.sum(targets&~np.isfinite(estimate)))
                if nonphysical or nonfinite:
                    ax.text(.99,.97 if name==candidate else .83,f"{name}: {nonphysical} negative / {nonfinite} nonfinite estimates",
                            transform=ax.transAxes,ha="right",va="top",fontsize=7,color=color)
            ax.set_ylabel(label+" (raw units)")
            ax.grid(alpha=.15)
            ax.legend(fontsize=6.5,loc="upper left",ncol=2)
        for ax in axes:
            for start,end in hidden_spans(first_visible,duration):
                ax.axvspan(start,end,color="#999999",alpha=.12,zorder=0)
            ax.set_xlim(0,duration/60.)
        axes[-1].set_xlabel("Minutes since original case start (no time warping)")
        fig.text(.01,.005,"Model crosses are generated bin-median estimates, not observations or physical-state evidence. Raw observations are not connected or filled.",fontsize=8)
        fig.tight_layout(rect=(0,.025,1,1))
        filename=f"replay_{rank:02d}_{cid}.png"
        _save(fig,target/filename,index,"raw_replay",1,
              "Seeded random test case, first masking layout, raw per-code timestamps, exact HP intervals, hidden-bin targets and model estimates.",model=candidate,dpi=dpi)
        selected_rows.append(dict(selection_rank=rank,case_index=int(case_index),case_id=cid,split="test",seed=SEED,
                                  mask_replicate=0,candidate=candidate,baseline="baseline_channelwise",figure=filename,
                                  selection="uniform random without replacement; no error/gain filtering"))
    pd.DataFrame(selected_rows).to_csv(target/"replay_selection.csv",index=False)
    (target/"selection_metadata.json").write_text(json.dumps(dict(seed=SEED,method="Six uniformly random frozen test cases, independent of model errors",
        case_indices=selected.tolist(),case_ids=ids,mask_replicate=0,max_raw_plot_points_per_signal=1500,
        downsampling="deterministic only for drawing; analysis and targets unchanged",inverse="unstandardize; RPM*40000; torque/pedal2 expm1",
        nonphysical_estimates="Displayed unchanged and counted; never clipped into physical range",
        model_estimates="Only hidden bins with observed macro targets are plotted. Generated estimates are distinct from raw scatter.",
        raw_numeric_policy="Actual per-code timestamps, HP2 observations only, no interpolation/ffill",candidate=candidate),indent=2)+"\n")


def latent_coordinates(out,target,index,dpi):
    path=out/"latent_coordinates.csv"
    if not path.exists():
        return
    table=pd.read_csv(path)
    if not all(c in table for c in ["case_id","z1","z2","split"]):
        raise ValueError("latent_coordinates.csv requires case_id,z1,z2,split")
    if table.case_id.duplicated().any():
        raise ValueError("Duplicate latent coordinate case IDs")
    fields=[c for c in ["hp3_before_first_high","high_fraction","mid_fraction","hp_known_fraction"] if c in table]
    fig,axes=plt.subplots(1,1+len(fields),figsize=(5*(1+len(fields)),4.5),squeeze=False)
    ax=axes[0,0]
    for name,g in table.groupby("split",sort=False):
        ax.scatter(g.z1,g.z2,s=18,alpha=.65,label=str(name))
    ax.legend(fontsize=8);ax.set_title("Frozen model coordinates by case split")
    for ax,col in zip(axes[0,1:],fields):
        values=pd.to_numeric(table[col],errors="coerce");good=np.isfinite(values)
        ax.scatter(table.loc[~good,"z1"],table.loc[~good,"z2"],s=18,color="#BBBBBB",label="Unknown")
        sc=ax.scatter(table.loc[good,"z1"],table.loc[good,"z2"],c=values[good],s=18,cmap="viridis",alpha=.7)
        fig.colorbar(sc,ax=ax,shrink=.7);ax.set_title(col.replace("_"," "))
    for ax in axes.ravel():
        ax.set_xlabel("z1");ax.set_ylabel("z2");ax.grid(alpha=.15)
    fig.suptitle("Exploratory continuous coordinates: distances/axes are not treatment classes or real-time trajectories")
    fig.tight_layout()
    _save(fig,target/"04_latent_coordinates.png",index,"latent_coordinates",len(table),
          "Supplied frozen-model coordinates, shown without clustering; overlays are interpretation audits, not learning labels.",dpi=dpi)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--source",type=Path,default=DEFAULT_SOURCE)
    parser.add_argument("--dpi",type=int,default=110)
    args=parser.parse_args();out=args.output
    if not (out/"test_summary.json").exists():
        raise RuntimeError("Final test summary absent; visualization must wait until frozen final evaluation completes")
    if (out/"figures_index.csv").exists():
        raise FileExistsError("Preserve completed figure audit; use a new version for replotting")
    summary=json.loads((out/"test_summary.json").read_text())
    freeze=json.loads((out/"selection_freeze.json").read_text())
    if summary["candidate"]!=freeze["choices"]["candidate"]:
        raise ValueError("Summary candidate differs from pre-test freeze")
    data=np.load(out/"data.npz");ledger=pd.read_csv(out/"case_ledger.csv")
    if ledger.case_id.duplicated().any() or len(ledger)!=300:
        raise ValueError("Expected unique all-300 ledger")
    if not np.array_equal(ledger.case_id.to_numpy(str),data["case_ids"]):
        raise ValueError("Ledger/array case IDs differ")
    target=out/"figures";target.mkdir(exist_ok=True)
    index=[]
    comparisons(out,target,summary,freeze,index,args.dpi)
    coverage(ledger,data,target,index,args.dpi)
    replay(out,args.source,target,summary,data,ledger,index,args.dpi)
    latent_coordinates(out,target,index,args.dpi)
    pd.DataFrame(index).to_csv(out/"figures_index.csv",index=False)
    print(json.dumps(dict(figures=len(index),raw_replay_cases=6,selection_seed=SEED,source_tables_modified=False,
                          figure_index=str(out/"figures_index.csv"))))


if __name__=="__main__":
    main()
