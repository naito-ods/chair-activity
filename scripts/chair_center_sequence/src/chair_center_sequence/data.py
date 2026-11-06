"""Annotation-independent grids and explicitly censored center-time teachers."""

import unicodedata

import numpy as np
import pandas as pd


UNKNOWN = "unknown"


def normalize(value):
    return "" if pd.isna(value) else unicodedata.normalize("NFKC", str(value)).strip().casefold()


def boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).strip().lower() in {"true", "1"}:
        return True
    if str(value).strip().lower() in {"false", "0"}:
        return False
    raise ValueError(f"Invalid confirmed flag: {value!r}")


def validate_annotations(annotations, duration):
    required = {"annotation_id", "start_s", "end_s", "label", "confirmed"}
    if not required.issubset(annotations.columns):
        raise ValueError(f"Missing annotation columns: {sorted(required - set(annotations.columns))}")
    a = annotations.copy()
    if a.annotation_id.isna().any():
        raise ValueError("Annotation IDs must not be missing")
    a["annotation_id"] = a.annotation_id.astype(str)
    if a.annotation_id.eq("").any() or not a.annotation_id.is_unique:
        raise ValueError("Annotation IDs must be nonempty and unique within a case")
    for name in ["start_s", "end_s"]:
        a[name] = pd.to_numeric(a[name], errors="raise")
    bounds = a[["start_s", "end_s"]].to_numpy(float)
    if not np.isfinite(bounds).all() or (a.start_s < 0).any() or (a.end_s > duration).any() or (a.end_s <= a.start_s).any():
        raise ValueError("Annotation bounds must be finite, positive-length, and inside raw bounds")
    a["label"] = a.label.map(normalize)
    a["confirmed"] = a.confirmed.map(boolean)
    return a.sort_values(["start_s", "end_s", "annotation_id"], kind="stable").reset_index(drop=True)


def center_grid(case_id, duration, step_s=1.0):
    """Cell centers depend only on raw bounds; preserve even a short final cell."""
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(step_s) or step_s <= 0:
        raise ValueError("Duration and step_s must be finite and positive")
    starts = np.arange(0, duration, step_s)
    ends = np.minimum(starts + step_s, duration)
    return pd.DataFrame({
        "id": [f"{case_id}:center:{i}" for i in range(len(starts))],
        "case_id": str(case_id), "cell_start_s": starts, "cell_end_s": ends,
        "center_s": (starts + ends) / 2,
    })


def attach_teachers(grid, annotations, duration):
    """A center has a teacher only inside exactly one confirmed known interval.

    Intervals are half-open [start,end). Boundary distance is retained; no fixed
    five-second erosion. Overlapping intervals, including identical labels,
    remain ambiguous. Input windows may span other procedures: the target is
    the center, never a majority label for the entire input window.
    """
    a = validate_annotations(annotations, duration)
    g = grid.copy()
    g["target"] = UNKNOWN
    g["annotation_id"] = ""
    g["teacher"] = False
    g["unknown_reason"] = "unannotated"
    g["boundary_distance_s"] = np.nan
    hits = np.zeros(len(g), dtype=int)
    owner = np.full(len(g), -1, dtype=int)
    centers = g.center_s.to_numpy(float)
    for i, row in enumerate(a.itertuples()):
        inside = (centers >= row.start_s) & (centers < row.end_s)
        hits[inside] += 1
        owner[inside] = i
    g.loc[hits > 1, "unknown_reason"] = "overlap"
    for i, row in enumerate(a.itertuples()):
        chosen = (hits == 1) & (owner == i)
        g.loc[chosen, "annotation_id"] = row.annotation_id
        g.loc[chosen, "boundary_distance_s"] = np.minimum(centers[chosen] - row.start_s, row.end_s - centers[chosen])
        known = row.confirmed and row.label not in {"", UNKNOWN}
        if known:
            g.loc[chosen, ["target", "teacher", "unknown_reason"]] = [row.label, True, ""]
        else:
            g.loc[chosen, "unknown_reason"] = "pending" if not row.confirmed else "explicit_unknown"
    ledger = []
    for row in a.itertuples():
        n = int(((g.annotation_id == row.annotation_id) & g.teacher).sum())
        center_count = int(((centers >= row.start_s) & (centers < row.end_s)).sum())
        reason = ("included" if n else "unconfirmed_or_unknown" if not row.confirmed or row.label in {"", UNKNOWN}
                  else "no_grid_center" if not center_count else "all_centers_ambiguous")
        ledger.append({"case_id": str(g.case_id.iloc[0]), "annotation_id": row.annotation_id,
                       "label": row.label, "duration_s": row.end_s - row.start_s,
                       "n_teachers": n, "n_centers_in_interval": center_count, "reason": reason})
    return g, pd.DataFrame(ledger)


def parent_case_weights(meta):
    """Equal case mass, equal parent mass within a case; no window inflation."""
    if not len(meta) or not meta.teacher.all() or meta.annotation_id.eq("").any():
        raise ValueError("Weights require a nonempty table containing only known teachers")
    q = meta[["case_id", "annotation_id"]].copy()
    counts = q.groupby(["case_id", "annotation_id"]).annotation_id.transform("size")
    parents = q.groupby("case_id").annotation_id.transform("nunique")
    w = 1.0 / (counts.to_numpy(float) * parents.to_numpy(float))
    return w / w.mean()
