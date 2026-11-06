"""Raw-only multi-scale features; labels/boundaries are never accepted here."""

import numpy as np
import pandas as pd


STATE = {"20039": (0, 1, 2, 3), "20011": (0, 1, 2, 3), "20101": (0, 1), "20774": (0, 1)}
NUMERIC = ("20742", "20743", "20108", "10014", "10015")
N_STATE = sum(len(values) + 1 for values in STATE.values())
CHANNELS = ([f"{code}:state:{v}" for code, values in STATE.items() for v in (*values, "unknown")]
            + [f"{c}:observed_median" for c in NUMERIC]
            + [f"{c}:observed_mask" for c in NUMERIC] + ["inside_raw_bounds"])


class CaseLog:
    def __init__(self, events, duration):
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid raw duration")
        if not {"time_s", "code", "value"}.issubset(events.columns):
            raise ValueError("Events require time_s, code, value")
        e = events[["time_s", "code", "value"]].copy()
        e["code"] = e.code.astype(str)
        e["time_s"] = pd.to_numeric(e.time_s, errors="raise")
        e["value"] = pd.to_numeric(e.value, errors="raise")
        if not np.isfinite(e.time_s).all() or ((e.time_s < 0) | (e.time_s > duration)).any() or np.isinf(e.value).any():
            raise ValueError("Invalid event time/value or event outside raw bounds")
        e = e.sort_values("time_s", kind="stable")
        self.duration = float(duration)
        self.events = {}
        self.segments = {}
        for code in (*STATE, *NUMERIC):
            q = e[e.code == code]
            times, values = q.time_s.to_numpy(float), q.value.to_numpy(float)
            self.events[code] = times, values
            if code in STATE:
                q = q.drop_duplicates("time_s", keep="last")
                times, values = q.time_s.to_numpy(float), q.value.to_numpy(float)
                values = np.where(np.isin(values, STATE[code]), values, np.nan)
                # Initial unknown and invalid/unknown states persist until the
                # next SAME-code record. Numeric observations never persist.
                left = np.r_[0.0, times]
                right = np.r_[times, duration]
                value = np.r_[np.nan, values]
                keep = right > left
                self.segments[code] = left[keep], right[keep], value[keep]

    def occupancy(self, code, start, end):
        """Include outside-case input time as unknown, with a separate bound mask."""
        if end <= start:
            raise ValueError("Empty feature window")
        left, right, values = self.segments[code]
        width = np.maximum(0.0, np.minimum(right, end) - np.maximum(left, start))
        known = [width[values == v].sum() / (end - start) for v in STATE[code]]
        unknown = max(0.0, 1.0 - sum(known))
        return np.array(known + [unknown], dtype=float)

    def summary(self, center, widths=(2.0, 5.0, 10.0), mode="centered"):
        if not np.isfinite(center) or not 0 <= center < self.duration:
            raise ValueError("Feature center must be inside raw bounds")
        if mode not in {"centered", "past"}:
            raise ValueError("Feature mode must be centered or past")
        result = {}
        for width in widths:
            if not np.isfinite(width) or width <= 0:
                raise ValueError("Window widths must be finite and positive")
            start, end = (center - width / 2, center + width / 2) if mode == "centered" else (center - width, center)
            prefix = f"w{width:g}:"
            span = max(0.0, min(self.duration, end) - max(0.0, start))
            result[prefix + "available_fraction"] = span / width
            for code, values in STATE.items():
                for value, fraction in zip((*values, "unknown"), self.occupancy(code, start, end)):
                    result[prefix + f"{code}:state:{value}"] = fraction
                times, event_values = self.events[code]
                lo, hi = np.searchsorted(times, [max(0, start), min(end, self.duration)], side="left")
                result[prefix + f"{code}:records"] = float(hi - lo)
                # A state boundary counts a change, not another equal-value
                # heartbeat. Invalid-to-known also counts as a change.
                left, _, segment_values = self.segments[code]
                different = np.r_[False, ~((segment_values[1:] == segment_values[:-1]) |
                                          (np.isnan(segment_values[1:]) & np.isnan(segment_values[:-1])))]
                result[prefix + f"{code}:changes"] = float(((left >= start) & (left < end) & different).sum())
            for code in NUMERIC:
                times, values = self.events[code]
                lo, hi = np.searchsorted(times, [max(0, start), min(end, self.duration)], side="left")
                values = values[lo:hi]
                observed = values[np.isfinite(values)]
                result[prefix + f"{code}:observed_count"] = float(len(observed))
                for name, fn in [("median", np.median), ("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)]:
                    result[prefix + f"{code}:{name}"] = float(fn(observed)) if len(observed) else np.nan
        return result

    def tensor(self, center, width_s=30, mode="centered"):
        if not np.isfinite(center) or not 0 <= center < self.duration:
            raise ValueError("Feature center must be inside raw bounds")
        if not isinstance(width_s, int) or width_s < 11:
            raise ValueError("ROCKET context width must be an integer >=11 seconds")
        if mode not in {"centered", "past"}:
            raise ValueError("Feature mode must be centered or past")
        start = center - width_s / 2 if mode == "centered" else center - width_s
        edges = start + np.arange(width_s + 1)
        x = np.full((len(CHANNELS), width_s), np.nan, dtype=np.float32)
        offset = 0
        for code, values in STATE.items():
            for j in range(width_s):
                x[offset:offset + len(values) + 1, j] = self.occupancy(code, edges[j], edges[j + 1])
            offset += len(values) + 1
        for i, code in enumerate(NUMERIC):
            times, values = self.events[code]
            for j in range(width_s):
                lo, hi = np.searchsorted(times, [max(0, edges[j]), min(self.duration, edges[j + 1])], side="left")
                v = values[lo:hi]
                v = v[np.isfinite(v)]
                x[N_STATE + i, j] = np.median(v) if len(v) else np.nan
            x[N_STATE + len(NUMERIC) + i] = np.isfinite(x[N_STATE + i])
        x[-1] = np.maximum(0, np.minimum(edges[1:], self.duration) - np.maximum(edges[:-1], 0))
        return x


class NumericEncoding:
    """Train-only numeric median/IQR; NaN retained until model input encoding."""
    def fit(self, x):
        self.center, self.scale = [], []
        for j in range(len(NUMERIC)):
            values = x[:, N_STATE + j, :]
            values = values[np.isfinite(values)]
            center = float(np.median(values)) if len(values) else 0.0
            scale = float(np.quantile(values, .75) - np.quantile(values, .25)) if len(values) else 1.0
            if scale <= 1e-12:
                scale = float(np.ptp(values)) if len(values) else 1.0
            self.center.append(center)
            self.scale.append(scale if scale > 1e-12 else 1.0)
        return self

    def transform(self, x):
        result = np.array(x, copy=True, dtype=np.float32)
        for j in range(len(NUMERIC)):
            v = result[:, N_STATE + j, :]
            result[:, N_STATE + j, :] = np.where(np.isfinite(v), (v - self.center[j]) / self.scale[j], -3.0)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite model input after explicit missing-value encoding")
        return result
