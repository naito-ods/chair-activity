#!/usr/bin/env python3
"""Offline context report from compact v3 streams and audited v5 case features.

Plots preserve observed values. RPM points are run/second medians from the
upstream compact stream; gaps are never interpolated and unknown HP is shown.
No full wide case table is opened. This module owns only figures/report files.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor

_MPL_CACHE = tempfile.TemporaryDirectory(prefix="chair_context_v5_mpl_")
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE.name)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

COLORS = {"H": "#d4503b", "M": "#d69b20", "L": "#2879b4", "Z": "#939393", "X": "#a24eae"}
HP_COLORS = {0: "#edf0f3", 1: "#58a64d", 2: "#4d7bc2", 3: "#984dad"}
FIGURES = []
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def flag(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "1.0", "yes"])


def short(case_id):
    m = re.search(r"seq_(\d+)", str(case_id))
    return "seq " + m.group(1) if m else str(case_id)


def add_figure(fig, root, name, title, section="Overview", caption=""):
    path = root / "figures" / (name + ".png")
    fig.savefig(path, dpi=125, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGURES.append(dict(file=str(path.relative_to(root)), title=title, section=section, caption=caption))


def read_optional(root, filename):
    p = root / filename
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def select_cohort(f):
    for key in ["main_high_cohort", "supported_high", "high_supported", "in_main_cohort", "main_cohort", "eligible_high"]:
        if key in f:
            return f[flag(f[key])].copy()
    ncol = next((c for c in f if c in ["high_n_bins", "h_n_bins", "high_bins", "n_high_bins"]), None)
    scol = next((c for c in f if c in ["high_equivalent_s", "high_proxy_s", "h_proxy_s", "high_weight_s"]), None)
    if ncol and scol:
        return f[(f[ncol] >= 3) & (f[scol] >= 2)].copy()
    raise ValueError("Explicit primary high support flag is required; no implicit cohort inference")


def case_plot(payload):
    case, points, intervals, obs, row, destination = payload
    p = pd.DataFrame(points)
    h = pd.DataFrame(intervals)
    o = pd.DataFrame(obs)
    duration = float(row.get("duration_s", h.end_s.max() if len(h) else p.time_s.max()))
    fig, axes = plt.subplots(5, 1, figsize=(13.5, 8.6), sharex=True,
                             gridspec_kw={"height_ratios": [0.7, 2.1, 1.1, 1.1, 0.85], "hspace": 0.18})
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    for _, q in h.iterrows():
        known = bool(q.known) and pd.notna(q.hp)
        color = HP_COLORS.get(int(q.hp), "#777") if known else "#bcbcbc"
        axes[0].broken_barh([(q.start_s/60, (q.end_s-q.start_s)/60)], (0, 1),
                            facecolors=color, edgecolors="none", hatch=None if known else "///")
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([])
    axes[0].set_ylabel("HP state", rotation=0, labelpad=34, va="center")
    for zone, color in COLORS.items():
        q = p[p.zone == zone]
        axes[1].scatter(q.time_s/60, q.rpm, s=8, c=color, linewidths=0, label=zone)
    axes[1].axhspan(36000, 44000, color=COLORS["H"], alpha=.055)
    axes[1].axhline(5000, color="#888", lw=.6, ls=":")
    axes[1].axhline(36000, color="#888", lw=.6, ls=":")
    axes[1].set_ylim(-1000, 45500)
    axes[1].set_yticks([0, 5000, 20000, 40000], ["0", "5k", "20k", "40k"])
    axes[1].set_ylabel("HP2 RPM")
    for ax, code, label, color in [(axes[2], 20743, "Torque\n(raw)", "#62408f"),
                                    (axes[3], 10015, "Pedal 2\n(raw)", "#237b68")]:
        q = o[(o.code == code) & (o.hp == 2)]
        if len(q):
            ax.scatter(q.time_s/60, q.value, s=7, color=color, linewidths=0)
        else:
            ax.text(.5, .5, "No observations while HP2 active", ha="center", va="center", transform=ax.transAxes, color="#777")
        ax.set_ylabel(label)
    q = p[p.zone == "M"]
    if len(q):
        # Width denotes only the within-run portion of one observed second.
        for _, r in q.iterrows():
            start = max(float(r.run_start_s), np.floor(float(r.time_s)))
            axes[4].broken_barh([(start/60, r.weight_s/60)], (0, 1), facecolors=COLORS["M"], edgecolors="none")
    axes[4].set_ylim(0, 1)
    axes[4].set_yticks([])
    axes[4].set_ylabel("MID bins", rotation=0, labelpad=34)
    axes[4].set_xlabel("Minutes from first lamp ON (actual elapsed time)")
    axes[4].set_xlim(0, duration/60)
    handles = [Patch(facecolor=HP_COLORS[i], label=f"HP{i}" if i else "Known OFF") for i in range(4)]
    handles += [Patch(facecolor="#bcbcbc", hatch="///", label="Unknown HP")]
    handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[z], markersize=5,
                       label=label) for z, label in [("H", "H 36–44k"), ("M", "M >5–<36k"), ("L", "L >0–5k"), ("Z", "RPM=0")]]
    fig.legend(handles=handles, loc="upper center", ncol=9, fontsize=8, bbox_to_anchor=(.53, .965), frameon=False)
    group = row.get("context_family", row.get("context_group", ""))
    fig.suptitle(f"{case}  {group}", fontsize=12, y=1.005)
    fig.text(.01, .005, "Points: observed run/second medians. No interpolation; empty plot regions are not OFF. MID bins do not prove sustained operation.", fontsize=8, color="#555")
    path = Path(destination) / "figures" / "cases" / (case + ".png")
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dict(case_id=case, file=str(path.relative_to(Path(destination))), n_rpm_points=int(p.rpm.notna().sum()),
                n_torque_points=int(((o.code == 20743) & (o.hp == 2)).sum()),
                n_pedal2_points=int(((o.code == 10015) & (o.hp == 2)).sum()))


def binary_columns(f, catalog):
    cols = []
    if len(catalog) and "feature" in catalog:
        for _, r in catalog.iterrows():
            if r.feature in f and str(r.get("kind", "")).lower() in ["binary", "bool", "boolean"]:
                cols.append(r.feature)
    if not cols:
        for col in f:
            vals = set(f[col].dropna().astype(str).str.lower())
            if vals and vals.issubset({"true", "false", "0", "1", "0.0", "1.0"}) and col != "case_id":
                cols.append(col)
    # Cohort/support flags are audited separately, not presented as operation patterns.
    return [c for c in cols if not any(x in c.lower() for x in ["cohort", "eligible", "prior_high", "previous_high", "all_target", "numeric_missing", "old_"])]


def feature_labels(catalog):
    if not len(catalog) or "feature" not in catalog:
        return {}
    return {r.feature: str(r.get("label", r.feature)) for _, r in catalog.iterrows()}


def render_overview(root, f, c, points, hp, catalog):
    labels = feature_labels(catalog)
    binary = binary_columns(c, catalog)
    meaningful = [k for k in binary if 0 < flag(c[k]).sum() < len(c)]
    counts = pd.DataFrame([dict(feature=k, label=labels.get(k, k), n_cases=int(flag(c[k]).sum()),
                                denominator=len(c), percent=100*flag(c[k]).mean()) for k in binary])
    counts.to_csv(root / "visual_prevalence.csv", index=False)
    if len(counts):
        selected = counts.sort_values("n_cases", ascending=False).head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(11.5, max(5, .32*len(selected)+1.6)))
        ax.barh(np.arange(len(selected)), selected.n_cases, color="#346c9e")
        ax.set_yticks(np.arange(len(selected)), selected.feature.str.replace("_", " "))
        for j, n in enumerate(selected.n_cases):
            ax.text(n+.7, j, f"{n}/{len(c)}", va="center", fontsize=9)
        ax.set_xlim(0, len(c)+12)
        ax.set_xlabel("Distinct cases (overlapping observations)")
        ax.set_title("Context observations among supported high-RPM cases")
        ax.spines[["top", "right"]].set_visible(False)
        add_figure(fig, root, "01_context_prevalence", "Context observations: distinct case counts", caption="Multi-label counts overlap. Not-observed flags do not prove non-use. Definitions and support thresholds are available in the feature catalog.")
    if len(meaningful) >= 2:
        # Show at most twenty varied features, retain explicit ordering from catalog.
        selected = meaningful[:20]
        a = np.array([flag(c[k]).to_numpy(dtype=int) for k in selected])
        co = a @ a.T
        fig, ax = plt.subplots(figsize=(11.5, 10))
        im = ax.imshow(co, cmap="Blues", vmin=0, vmax=len(c))
        labs = [k.replace("_", " ") for k in selected]
        ax.set_xticks(np.arange(len(labs)), labs, rotation=70, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(labs)), labs, fontsize=8)
        for i in range(len(labs)):
            for j in range(len(labs)):
                ax.text(j, i, str(co[i,j]), ha="center", va="center", fontsize=6.5, color="white" if co[i,j] > len(c)*.55 else "#222")
        fig.colorbar(im, ax=ax, label="Cases with both observations", shrink=.7)
        ax.set_title("Co-occurrence: HP3, MID and LOW are not exclusive classes")
        add_figure(fig, root, "02_context_overlap", "Overlapping contexts", caption="Each cell counts independent cases having both observed features; diagonal is prevalence.")
        sorting = [k for k in ["context_family", "context_group", "hp3_observed", "mid_substantive", "mid_dominant_run"] if k in c]
        ordered = c.sort_values(sorting + ["case_id"]) if sorting else c.sort_values("case_id")
        a = np.array([flag(ordered[k]).to_numpy(dtype=float) for k in selected]).T
        fig, ax = plt.subplots(figsize=(12, max(10, len(c)*.145)))
        ax.imshow(a, cmap=matplotlib.colors.ListedColormap(["#f0f2f5", "#2e698f"]), aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(selected)), labs, rotation=65, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(c)), [short(x) for x in ordered.case_id], fontsize=6)
        ax.set_title("Case context matrix (observed flags; row = one case)")
        add_figure(fig, root, "03_case_context_matrix", "Context features for every case", caption="Blue = observed according to the feature definition. Grey = flag not met; this is not a claim that an operation was never used.")
    # Entire cases, actual normalized clock; no signal interpolation.
    sortcols = [x for x in ["context_family", "context_group", "hp3_observed"] if x in c]
    ordered = c.sort_values(sortcols+["case_id"])
    fig, axes = plt.subplots(1, 2, figsize=(17, max(11, len(c)*.16)), sharey=True,
                             gridspec_kw={"width_ratios": [1.1, 1]})
    ids = list(ordered.case_id)
    for i, case in enumerate(ids):
        h = hp[hp.case_id == case]
        dur = float(h.end_s.max())
        for state in [0,1,2,3,None]:
            q=h[h.known & (h.hp==state)] if state is not None else h[~h.known | h.hp.isna()]
            if len(q):
                ranges=np.column_stack((100*q.start_s.to_numpy()/dur,100*(q.end_s-q.start_s).to_numpy()/dur))
                axes[0].broken_barh(ranges,(i-.4,.8),facecolors=HP_COLORS.get(state,"#aaa"),edgecolors="none")
        p = points[(points.case_id == case) & points.rpm.notna()]
        for zone in ["L", "M", "H", "Z", "X"]:
            q = p[p.zone == zone]
            axes[1].scatter(100*q.time_s/dur, np.full(len(q), i), s=3, color=COLORS[zone], linewidths=0)
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.set_ylim(len(ids)-.3, -.7)
        ax.set_xlabel("Case elapsed time (%) — no DTW")
        ax.grid(axis="x", alpha=.2)
    axes[0].set_yticks(np.arange(len(ids)), [short(x) for x in ids], fontsize=6)
    axes[0].set_title("Full-case HP state: green HP1 / blue HP2 / purple HP3 / grey unknown")
    axes[1].set_title("Observed HP2 RPM: blue LOW / gold MID / red HIGH / grey zero")
    add_figure(fig, root, "04_all_case_timeline", "Whole-case context atlas", caption="The same case order is shown twice. Time is normalized only for display; gaps stay empty in RPM and unknown HP stays grey.")
    # MID-position distribution: denominator all observed MID proxy support in case.
    rows = []
    for case in ids:
        p = points[points.case_id == case]
        h = hp[hp.case_id == case]
        dur = float(h.end_s.max())
        q = p[p.zone == "M"]
        hist, _ = np.histogram(q.time_s/dur, bins=np.linspace(0,1,11), weights=q.weight_s)
        denom = hist.sum()
        rows.append(hist / denom if denom else np.full(10, np.nan))
    fig, ax = plt.subplots(figsize=(10, max(10, len(c)*.14)))
    im = ax.imshow(np.array(rows), aspect="auto", cmap="YlOrBr", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(ids)), [short(x) for x in ids], fontsize=6)
    ax.set_xticks(range(10), [f"{i*10}–{(i+1)*10}%" for i in range(10)], rotation=45)
    ax.set_xlabel("Case elapsed-time segment")
    ax.set_title("Where MID observations occur within each case")
    fig.colorbar(im, ax=ax, label="Fraction of case's observed MID proxy support", shrink=.55)
    add_figure(fig, root, "05_middle_position", "MID location within cases", caption="Each row sums to one over observed MID proxy support. Empty rows have no MID observations. One-second weighting is not a continuous duration measurement.")
    # Definition ladder prevents treating short ramps as independent MID work.
    ladder = []
    for key, label, mode in [
        ("any_mid_observed", "Any MID observation", "flag"),
        ("mid_supported", "MID support: >=3 bins, >=2 proxy s", "flag"),
        ("n_mid_interior_runs", "MID observed >3 s from both run edges", "count"),
        ("n_mid_dominant_runs", "At least one MID-dominant run", "count"),
        ("n_mid_dominant_blocks", "At least one MID-dominant 30-s macroblock", "count"),
        ("old_high_then_short_mid", "High then short MID (prior strict block rule)", "flag")]:
        if key in c:
            n = int((flag(c[key]) if mode == "flag" else c[key].fillna(0)>0).sum())
            ladder.append((label,n))
    if ladder:
        fig, ax=plt.subplots(figsize=(11,5))
        ax.barh(range(len(ladder)), [x[1] for x in ladder],color=["#e6b54d","#e6b54d","#ca8b20","#985c12","#985c12","#9a659d"][:len(ladder)])
        ax.set_yticks(range(len(ladder)),[x[0] for x in ladder])
        ax.invert_yaxis();ax.set_xlim(0,len(c)+12)
        for i,(_,n) in enumerate(ladder): ax.text(n+.5,i,f"{n}/{len(c)}",va="center")
        ax.set_xlabel("Distinct cases (definitions overlap and are not a nested ladder)")
        ax.set_title("MID: an observation point versus an operation-context candidate")
        add_figure(fig,root,"07_mid_definition_support","MIDの定義を変えると何症例に残るか",caption="The rows are distinct definitions and need not nest. Near-edge exclusion is a timing check, not proof of treatment intent. Exact RPM support/dominance requirements are in feature_catalog.csv and metadata.")
    # Selection effect: previous 67 vs newly included high-observed cases.
    if "old_high_dominant" in c:
        groups=[("Prior high-dominant",c[flag(c.old_high_dominant)]),("Additional high-observed",c[~flag(c.old_high_dominant)])]
        chosen=[k for k in ["hp3_observed","hp1_observed","n_mid_dominant_blocks","n_mid_dominant_runs","old_post_hp3_low","mid_block_after_last_high","block_high_mid_high"] if k in c]
        if chosen:
            fig,ax=plt.subplots(figsize=(11,5.5))
            for j,(label,g) in enumerate(groups):
                vals=[100*((g[k].fillna(0)>0) if k.startswith("n_") else flag(g[k])).mean() for k in chosen]
                ax.barh(np.arange(len(chosen))+(j-.5)*.34,vals,height=.32,label=f"{label} (n={len(g)})",color=["#517ea2","#c09d3c"][j])
            ax.set_yticks(range(len(chosen)),[k.replace("_"," ") for k in chosen]);ax.set_xlim(0,110)
            ax.set_xlabel("Percent of independent cases in each cohort");ax.legend(loc="lower right",fontsize=8)
            ax.set_title("Scope sensitivity: broad high observation versus high-dominant selection")
            add_figure(fig,root,"08_cohort_comparison","67症例に限定した場合との比較",caption="High-dominant status is not used to select the broader primary cohort. Different proportions identify a selection effect, not a treatment difference.")


def render_groups(root, c, p, hp):
    candidates=read_optional(root,"candidate_summary.csv")
    if len(candidates) and "n_cases" in candidates:
        candidates=candidates.sort_values("n_cases",ascending=True)
        fig,ax=plt.subplots(figsize=(12,max(5,.38*len(candidates)+1.8)))
        ax.barh(np.arange(len(candidates)),candidates.percent,color="#386f92")
        if "ci_low" in candidates and "ci_high" in candidates:
            # Summary uses percentages only when CI values are on that scale.
            lo=candidates.ci_low.to_numpy(float); hi=candidates.ci_high.to_numpy(float)
            if np.nanmax(hi)<=1.001: lo,hi=lo*100,hi*100
            v=candidates.percent.to_numpy(float)
            ax.errorbar(v,np.arange(len(candidates)),xerr=np.vstack([np.maximum(0,v-lo),np.maximum(0,hi-v)]),fmt="none",color="#172d42",capsize=2,lw=1)
        ax.set_yticks(np.arange(len(candidates)),candidates.label.astype(str))
        for i,(_,r) in enumerate(candidates.iterrows()):
            endpoint=float(r.get("ci_high",r.percent))
            if endpoint<=1.001: endpoint*=100
            ax.text(min(112,endpoint+1.2),i,f"{int(r.n_cases)}/{int(r.denominator)}",va="center",fontsize=8)
        ax.set_xlim(0,120);ax.set_xlabel("Percent of cases (95% within-cohort prevalence intervals)")
        ax.set_title("Candidate operation contexts — simultaneous, overlapping labels")
        add_figure(fig,root,"00_candidate_prevalence","候補パターンの頻度と不確実性",caption="Counts are distinct cases. Confidence intervals describe prevalence within this selected source population; they do not validate the contextual definition or generalization.")
    assignments = read_optional(root, "context_assignments.csv")
    if not len(assignments):
        assignments = read_optional(root, "case_context_groups.csv")
    representatives = read_optional(root, "representatives_with_clusters.csv")
    if not len(representatives):
        representatives = read_optional(root, "representatives.csv")
    if not len(representatives):
        representatives = read_optional(root, "context_representatives.csv")
    group_col = next((x for x in ["context_family", "context_group", "group_label", "group", "group_id"] if x in assignments), None)
    if group_col and "case_id" in assignments:
        assignments = assignments[assignments.case_id.isin(c.case_id)]
        fig, ax = plt.subplots(figsize=(11, 5))
        counts = assignments.groupby(group_col).case_id.nunique().sort_values()
        ax.barh(range(len(counts)), counts, color="#496b9c")
        ax.set_yticks(range(len(counts)), counts.index.astype(str))
        for i, n in enumerate(counts): ax.text(n+.3, i, str(n), va="center")
        ax.set_xlim(0, max(counts)*1.15)
        ax.set_xlabel("Distinct cases")
        ax.set_title("Interpretable contextual groups")
        add_figure(fig, root, "06_context_groups", "Contextual groups", caption="These are observable operation-context categories. Separation by a rule's own inputs is definitional, not external validation.")
    if len(representatives) and "case_id" in representatives:
        candidate_col = next((x for x in ["candidate", "context_family", "context_group", "group", "group_id"] if x in representatives), None)
        if candidate_col:
            for num, (candidate, rows) in enumerate(representatives.groupby(candidate_col, sort=False)):
                ids = list(dict.fromkeys(rows.case_id[rows.case_id.isin(c.case_id)]))[:5]
                if not ids: continue
                fig, axes = plt.subplots(len(ids), 1, figsize=(12, 2.1*len(ids)), sharex=True, squeeze=False)
                for ax, case in zip(axes[:,0], ids):
                    q = p[(p.case_id == case) & p.rpm.notna()]
                    dur = float(hp.loc[hp.case_id == case, "end_s"].max())
                    for z in COLORS:
                        qq = q[q.zone == z]
                        ax.scatter(100*qq.time_s/dur, qq.rpm, s=7, c=COLORS[z], linewidths=0)
                    h = hp[(hp.case_id == case) & (hp.hp == 3)]
                    for _, r in h.iterrows():
                        ax.axvspan(100*r.start_s/dur, 100*r.end_s/dur, color=HP_COLORS[3], alpha=.18)
                    ax.set_ylim(-1000,45500)
                    ax.set_yticks([0, 20000, 40000], ["0", "20k", "40k"])
                    roles=rows.loc[rows.case_id==case,"role"] if "role" in rows else pd.Series(dtype=str)
                    role=str(roles.iloc[0]) if len(roles) else "member"
                    ax.set_ylabel(short(case)+"\n"+role)
                    ax.grid(axis="x",alpha=.2)
                axes[0,0].set_title(str(candidate)+" — distinct-case representatives (purple bands: HP3)")
                axes[-1,0].set_xlabel("Case elapsed time (%) — no DTW, raw RPM levels")
                add_figure(fig, root, f"group_{num:02d}_members", str(candidate)+": independent cases", "Representatives", "Actual observations from distinct cases. Progress normalization does not warp time. Check individual full-case figures for torque/pedal and observation gaps.")


def render_evaluation(root, c):
    profiles=read_optional(root,"context_cluster_profiles.csv")
    if len(profiles):
        selected=[k for k in ["hp3_2s","hp3_before_first_high","hp3_after_last_high","high_hp3_high","hp1_2s","macro_mid","macro_mid_scattered","macro_mid_before","macro_mid_between","macro_mid_after","first_high_progress","last_high_progress"] if k in set(profiles.feature)]
        groups=sorted(profiles.cluster.unique())
        mat=profiles[profiles.feature.isin(selected)].pivot(index="feature",columns="cluster",values="mean").reindex(selected)
        fig,ax=plt.subplots(figsize=(10,6.5))
        im=ax.imshow(mat,cmap="Blues",vmin=0,vmax=1,aspect="auto")
        ax.set_yticks(range(len(selected)),[k.replace("_"," ") for k in selected])
        counts=profiles.groupby("cluster").n_cases.first()
        ax.set_xticks(range(len(groups)),[f"C{int(k)} (n={counts.loc[k]})" for k in groups])
        for i in range(len(selected)):
            for j in range(len(groups)):
                value=mat.iloc[i,j]
                ax.text(j,i,f"{value:.2f}",ha="center",va="center",color="white" if value>.6 else "#222")
        fig.colorbar(im,ax=ax,label="Fraction of cases / mean case progress",shrink=.7)
        ax.set_title("Exploratory k=3 context profiles")
        add_figure(fig,root,"12_k3_context_profiles","探索的3群で分かれた前後文脈",caption="Binary rows show case proportions; first/last-high rows show mean position within the case. These same features contribute to distance, so this is a descriptive profile rather than independent validation.")
    numeric=read_optional(root,"raw_numeric_by_cluster_zone.csv")
    if len(numeric):
        chosen=["torque","pedal2","bulk"]
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,signal in zip(axes,chosen):
            q=numeric[(numeric.zone=="H") & (numeric.signal==signal)].sort_values("cluster")
            x=np.arange(len(q))
            ax.errorbar(x,q.median_of_case_medians,yerr=np.vstack([q.median_of_case_medians-q.q25,q.q75-q.median_of_case_medians]),fmt="o",capsize=6,color="#577ca1")
            ax.set_xticks(x,[f"C{int(r.cluster)}\nn={int(r.n_supported_cases)}" for _,r in q.iterrows()])
            ax.set_ylabel(signal+" (raw)");ax.set_title("HIGH zone: "+signal)
            ax.spines[["top","right"]].set_visible(False)
        fig.suptitle("Raw numeric levels: median and IQR of supported case medians")
        add_figure(fig,root,"13_k3_raw_numeric","文脈3群と数値水準を別の軸で比較",caption="Numeric values are compared within observed high-RPM bins. A point is the group median of case medians; bars are the case IQR and n is supported cases, not RPM samples. Numeric observations were not used to name clinical treatments.")
    models=read_optional(root,"context_model_comparison.csv")
    resamples=read_optional(root,"context_case_subsampling.csv")
    if len(models):
        fig,axes=plt.subplots(1,2,figsize=(11.5,4))
        axes[0].plot(models.k,models.silhouette,"o-",color="#3c719c")
        axes[0].set(xlabel="Number of clusters k",ylabel="Silhouette",title="Context-distance separation")
        axes[0].set_xticks(models.k)
        if len(resamples):
            ks=list(models.k)
            axes[1].boxplot([resamples.loc[resamples.k==k,"ari"] for k in ks],tick_labels=ks,showfliers=True,
                            flierprops=dict(marker=".",markersize=3))
            axes[1].set(xlabel="Number of clusters k",ylabel="Case-subsampling ARI",title="80% case resampling (60 repeats per k)",ylim=(-.05,1.05))
        for ax in axes: ax.spines[["top","right"]].set_visible(False)
        add_figure(fig,root,"09_exploratory_model_comparison","文脈クラスタの分離度と内部安定性",caption="k is compared on fixed contextual feature definitions. Resampling ARI measures stability inside these cases, not accuracy on independent patients or validity of clinical labels.")
    sensitivity=read_optional(root,"context_definition_sensitivity.csv")
    if len(sensitivity):
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for unit,g in sensitivity.groupby("mid_unit"):
            axes[0].plot(g.hp3_min_s,g.mid_n,"o-",label=f"MID unit: {unit}")
            axes[1].plot(g.hp3_min_s,g.group_ari_vs_primary,"o-",label=f"MID unit: {unit}")
        axes[0].set(xlabel="Minimum total HP3 active seconds",ylabel="Cases meeting MID definition",title="MID granularity changes prevalence")
        axes[1].set(xlabel="Minimum total HP3 active seconds",ylabel="ARI against primary rule categories",title="Definition sensitivity (not clustering stability)",ylim=(0,1.04))
        for ax in axes: ax.legend(fontsize=8);ax.spines[["top","right"]].set_visible(False)
        add_figure(fig,root,"10_definition_sensitivity","文脈定義への依存性",caption="A finer atomic run and a 30-s-gap macroblock answer different questions. Similarity of deterministic categories under a threshold change is distinct from resampling stability.")
    dfile=root/"context_distances.csv"
    if dfile.exists():
        d=pd.read_csv(dfile,index_col=0)
        order=c.sort_values([k for k in ["context_group","group_id","case_id"] if k in c]).case_id.tolist()
        assert set(d.index)==set(order) and set(d.columns)==set(order)
        fig,ax=plt.subplots(figsize=(9,8))
        im=ax.imshow(d.loc[order,order],cmap="viridis",vmin=0,vmax=1,interpolation="nearest")
        ticks=np.arange(0,len(order),5)
        ax.set_xticks(ticks,[short(order[i]) for i in ticks],rotation=90,fontsize=7)
        ax.set_yticks(ticks,[short(order[i]) for i in ticks],fontsize=7)
        fig.colorbar(im,ax=ax,label="Mean bounded difference across five context families",shrink=.75)
        ax.set_title("Whole-case contextual distance (ordered by rule categories)")
        add_figure(fig,root,"11_context_distances","文脈の類似性：全症例間距離",caption="Five feature families receive equal total weight: HP presence, HP order, RPM composition, RPM placement, high-RPM location. Distances retain missing-feature rules documented in analysis_plan.json; no DTW.")


def html_table(df, maxrows=500):
    if not len(df): return ""
    return df.head(maxrows).to_html(index=False, escape=True, float_format=lambda x:f"{x:.4g}", classes="data")


def build_report(root, f, c, case_index, catalog):
    narrative = []
    for filename in ["analysis_summary.json", "context_summary.json", "evaluation_summary.json"]:
        if (root/filename).exists():
            info = json.loads((root/filename).read_text())
            narrative += info.get("report_paragraphs", [])
    parts = ["<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>高回転HP2：症例全体の文脈解析 v5</title>",
             "<style>body{font:16px/1.7 system-ui,sans-serif;color:#223047;max-width:1400px;margin:30px auto;padding:0 22px;background:#fafbfd}h1,h2,h3{line-height:1.35}a{color:#165c9d}section{background:white;border:1px solid #dce3ed;border-radius:10px;padding:20px;margin:20px 0}img{max-width:100%;height:auto}details{margin:14px 0}summary{cursor:pointer;font-weight:600}.data{border-collapse:collapse;font-size:12px;display:block;overflow:auto;max-height:600px}td,th{padding:5px 9px;border-bottom:1px solid #dce3ed;white-space:nowrap}th{position:sticky;top:0;background:#edf3fa}.note{background:#eef3fa;padding:14px;border-left:4px solid #47729c}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}.case{border-bottom:1px solid #dce3ed;padding:8px}input,select,button{font:inherit;padding:7px;margin:5px}figure{margin:25px 0}figcaption{color:#526074;font-size:14px}nav{position:sticky;top:0;background:#fafbfdeF;padding:8px;z-index:2}</style>",
             "<h1>高回転HP2：症例全体の前後文脈解析 v5</h1>",
             "<p>90012の最初のONから最後のOFFまでを対象とし、HP3・中回転・低回転が高回転の前後のどこに現れるかを比較します。DTWは使用していません。既存のStage 2割当は変更していません。</p>",
             f"<p class='note'>固定300症例を監査し、高回転の観測を3 run/秒ビン以上かつ代理観測量2秒以上確認できた<strong>{len(c)}症例</strong>を主解析とします。高回転1点以上は99症例、従来の『高回転主体ブロックあり』は67症例です。対象の拡張を明示して比較します。</p>",
             "<nav><a href='#findings'>結果</a>　<a href='#overview'>全体像</a>　<a href='#representatives'>代表例</a>　<a href='#cases'>症例検索</a>　<a href='#audit'>定義・監査</a></nav>",
             "<section id='findings'><h2>結果と解釈</h2>"]
    for para in narrative: parts.append("<p>"+html.escape(str(para))+"</p>")
    parts += ["<p>HP3の出現と中回転の出現は重複できます。そのため、主分類と併せて重複可能な文脈タグを提示します。中回転1点の出現と、中回転がまとまって観測される操作は区別します。</p>",
              "<p>高回転は36,000〜44,000 rpm（40,000付近の実装上の許容幅±10%）、低回転は0超〜5,000 rpm、中回転は5,000超〜36,000未満です。0 rpm・欠測・HP不明は別扱いです。</p>",
             "<p>まとまった中回転の観測も、意図した処置や作動の連続時間を直接示すものではありません。元の変更記録をrun内1秒ビンに集約した観測であり、数値を前方補完して連続波形を作っていません。中回転主体区間という分類は加減速を完全に除去する方法ではなく、run端からの時間も別に監査しています。</p>"]
    for filename in ["candidate_summary.csv", "context_group_summary.csv", "context_model_comparison.csv", "context_definition_sensitivity.csv", "high_support_sensitivity.csv", "context_cluster_profiles.csv", "raw_numeric_by_cluster_zone.csv"]:
        d=read_optional(root,filename)
        if len(d): parts.append(f"<h3>{html.escape(filename)}</h3>"+html_table(d))
    parts += ["<p>この結果は同一装置・既存症例内の操作パターン候補です。症例単位の内部評価と、新しい患者・装置・臨床ラベルに対する外部検証は別です。ルールで定義した群がその定義の特徴で分かれること自体を妥当性の証拠にはしません。</p></section>"]
    for section, anchor, title in [("Overview", "overview", "症例全体の文脈"), ("Representatives", "representatives", "代表例と異なる症例の比較")]:
        parts.append(f"<section id='{anchor}'><h2>{title}</h2>")
        for item in FIGURES:
            if item["section"] != section: continue
            parts += [f"<figure><h3>{html.escape(item['title'])}</h3><a href='{html.escape(item['file'])}'><img loading='lazy' src='{html.escape(item['file'])}' alt='{html.escape(item['title'])}'></a><figcaption>{html.escape(item['caption'])}</figcaption></figure>"]
        parts.append("</section>")
    indexed = case_index.merge(c, on="case_id", how="left", suffixes=("", "_feature"))
    parts += ["<section id='cases'><h2>全症例の実時間・原数値図</h2><p>症例IDで検索できます。図にはHP状態、HP2回転数、トルク、ペダル2を同じ実時間で表示します。空白はOFFとは限りません。</p><input id='search' placeholder='seq番号・日付・群で検索' aria-label='症例検索'><span id='count'></span>"]
    for _, r in indexed.iterrows():
        labels = [str(r[k]) for k in ["context_family", "context_group", "hp3_position", "mid_pattern"] if k in r and pd.notna(r[k])]
        display = r.case_id + (" | " + " / ".join(labels) if labels else "")
        text = html.escape(display)
        parts.append(f"<details class='case' data-search='{html.escape(display.lower(),quote=True)}'><summary>{text}</summary><p>RPM {int(r.n_rpm_points)}点 / torque {int(r.n_torque_points)}点 / pedal2 {int(r.n_pedal2_points)}点</p><a href='{html.escape(r.file)}'><img loading='lazy' src='{html.escape(r.file)}' alt='{text}'></a></details>")
    parts += ["</section><section id='audit'><h2>定義・監査・再実行</h2><p>順序モチーフは不明HP区間で中断します。中回転主体区間の前・間・後は時系列上の配置であり、間に不明区間がないことを保証しません。HP3未観測は『HP3未使用』と断定しません。中回転の停止前後への偏りと、まとまった中回転操作の候補を別々に確認してください。</p>"]
    parts.append(html_table(catalog))
    parts.append("<h3>解析ファイル</h3><ul>")
    for filename in sorted(root.glob("*.csv")):
        parts.append(f"<li><a href='{filename.name}'>{html.escape(filename.name)}</a></li>")
    for filename in sorted(root.glob("*.json")):
        parts.append(f"<li><a href='{filename.name}'>{html.escape(filename.name)}</a></li>")
    for filename in sorted((root/"process").glob("*")):
        if filename.is_file() and filename.suffix in [".txt", ".json", ".jsonl"]:
            parts.append(f"<li><a href='process/{filename.name}'>process/{html.escape(filename.name)}</a></li>")
    parts += ["</ul></section>",
              "<script>const q=document.getElementById('search');function filter(){let n=0;for(const el of document.querySelectorAll('.case')){const yes=el.dataset.search.includes(q.value.toLowerCase());el.hidden=!yes;if(yes)n++;}document.getElementById('count').textContent=n+'症例';}q.addEventListener('input',filter);filter();</script></html>"]
    (root/"report.html").write_text("\n".join(parts),encoding="utf-8")
    (root/"report.txt").write_text("\n\n".join(["高回転HP2：症例全体の前後文脈解析 v5",f"固定300症例 / 主解析{len(c)}症例 / 高回転1点以上99症例 / 従来高回転主体67症例"]+narrative+["既存Stage 2割当を変更せず、DTWを使用しない新しい文脈解析。HP3と中回転は重複タグを保持。中回転の加減速・停止直前の点と、まとまった操作候補は区別する。数値は観測run内1秒中央値で、欠測の補間・前方補完なし。", "ルールによる分離は定義上の分離であり、内部安定性は外部検証ではない。新規患者・装置・臨床情報による検証が必要。", "図・全症例検索・CSV定義はreport.htmlを参照。"]),encoding="utf-8")


def validate_report(root, case_index):
    from PIL import Image
    page = (root/"report.html").read_text()
    paths = re.findall(r'(?:src|href)=[\'\"]([^\'\"]+)[\'\"]', page)
    broken = [p for p in paths if not p.startswith(("#", "http")) and not (root/p).is_file()]
    images = sorted((root/"figures").rglob("*.png"))
    for p in images:
        with Image.open(p) as im: im.verify()
    assert not broken, broken
    assert case_index.case_id.is_unique
    result = dict(n_case_plots=len(case_index), n_images=len(images), n_links=len(paths), broken_links=broken,
                  source_sha256=_SOURCE_SHA256,
                  all_images_decodable=True, no_wide_case_tables_loaded=True)
    (root/"process"/"visualization_verification.json").write_text(json.dumps(result,indent=2))
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-cases", action="store_true", help="Rebuild summaries/report after existing case figures have been generated")
    args=ap.parse_args()
    root=args.output_root
    (root/"figures"/"cases").mkdir(parents=True,exist_ok=True)
    (root/"process").mkdir(exist_ok=True)
    f=pd.read_csv(root/"case_context_features.csv")
    catalog=read_optional(root,"feature_catalog.csv")
    c=select_cohort(f)
    af=read_optional(root,"analysis_case_features.csv")
    if len(af):
        extra=[x for x in af if x not in c or x=="case_id"]
        c=c.merge(af[extra],on="case_id",how="left",validate="one_to_one")
    groups=read_optional(root,"case_context_groups.csv")
    if len(groups) and "case_id" in groups:
        assert groups.case_id.is_unique
        newcols=[x for x in groups if x not in c or x=="case_id"]
        c=c.merge(groups[newcols],on="case_id",how="left",validate="one_to_one")
        if "group_label" in c: c["context_group"]=c.group_label
    assert len(f)==300 and f.case_id.is_unique
    assert len(c)>0 and c.case_id.is_unique
    p=pd.read_csv(args.source_root/"hp2_points.csv",usecols=["case_id","time_s","run_id","rpm","zone","weight_s","run_start_s","run_end_s"])
    p=p[p.case_id.isin(c.case_id)]
    hp=pd.read_csv(args.source_root/"hp_intervals.csv",usecols=["case_id","start_s","end_s","hp","known"])
    hp=hp[hp.case_id.isin(c.case_id)]
    # Compact long rows, narrowed to two channels before per-case payload creation.
    o=pd.read_csv(args.source_root/"observations.csv",usecols=["case_id","code","time_s","value","hp"])
    o=o[o.case_id.isin(c.case_id) & o.code.isin([20743,10015])]
    case_index_path=root/"case_figure_index.csv"
    if args.skip_cases and case_index_path.exists():
        case_index=pd.read_csv(case_index_path)
        assert set(case_index.case_id)==set(c.case_id), "Existing case plots do not match cohort"
    else:
        payload=[]
        for _,r in c.sort_values("case_id").iterrows():
            case=r.case_id
            payload.append((case,p[p.case_id==case].to_dict("list"),hp[hp.case_id==case].to_dict("list"),
                            o[o.case_id==case].to_dict("list"),r.to_dict(),str(root)))
        with ProcessPoolExecutor(max_workers=max(1,min(args.workers,8))) as executor:
            result=list(executor.map(case_plot,payload))
        case_index=pd.DataFrame(result)
        case_index.to_csv(case_index_path,index=False)
    render_overview(root,f,c,p,hp,catalog)
    render_groups(root,c,p,hp)
    render_evaluation(root,c)
    pd.DataFrame(FIGURES).to_csv(root/"figure_index.csv",index=False)
    build_report(root,f,c,case_index,catalog)
    result=validate_report(root,case_index)
    with (root/"process"/"visualization_log.jsonl").open("a") as log:
        log.write(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),action="report_complete",**result))+"\n")
    print(json.dumps(result))


if __name__=="__main__": main()
