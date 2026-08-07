"""B6 gap categorical features (fold-local edges; no TE).

Mining P0/P1 from docs/B6_DATA_MINING.md / artifacts/b6_eda:
- ratio_q5 × region/source
- t3_sfx × code × days_q5
- w_pair × days_q5 / ratio_q5
- days_fixed × cond_q5 / source
- code/version × days_q5
- age_coarse, cond_q5 × source, t_pair × days_q5
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_FIXED_EDGES = np.array([-np.inf, 700, 2500, 5000, 7000, 9000, 10000, np.inf], dtype=float)
DAYS_FIXED_LABELS = (
    "d0_700",
    "d700_2500",
    "d2500_5k",
    "d5k_7k",
    "d7k_9k",
    "d9k_10k",
    "d10k_plus",
)

GAP_CAT_COLS = (
    "gap_days_fixed",
    "gap_t3_sfx",
    "gap_w_pair",
    "gap_age_coarse",
    "gap_ratio5",
    "gap_ratio10",
    "gap_days5",
    "gap_cond5",
    "gap_t3sfx_code_days5",
    "gap_wpair_days5",
    "gap_agec_days5",
    "gap_ratio5_region",
    "gap_ratio5_source",
    "gap_ratio5_code",
    "gap_ratio5_t3sfx",
    "gap_daysfix_source",
    "gap_daysfix_cond5",
    "gap_daysfix_t3sfx",
    "gap_daysfix_code",
    "gap_code_days5",
    "gap_car_days5",
    "gap_version_days5",
    "gap_wpair_ratio5",
    "gap_cond5_source",
    "gap_tpair_days5",
    "gap_rpair_days5",
    "gap_cpair_days5",
    "gap_agec_ratio5",
)


def _quantile_edges(series: pd.Series, bins: int) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return np.array([], dtype=float)
    edges = np.unique(finite.quantile(np.linspace(0, 1, bins + 1)).to_numpy(dtype=float))
    return edges[1:-1] if len(edges) > 1 else np.array([], dtype=float)


def fit_gap_edges(X_tr: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit quantile cutpoints on the training fold only."""
    days = pd.to_numeric(X_tr["days"], errors="coerce")
    cond = pd.to_numeric(X_tr["condition"], errors="coerce")
    ratio = cond / (days.abs() + 1.0)
    return {
        "days5": _quantile_edges(days, 5),
        "cond5": _quantile_edges(cond, 5),
        "ratio5": _quantile_edges(ratio, 5),
        "ratio10": _quantile_edges(ratio, 10),
    }


def _bin_codes(values: pd.Series, edges: np.ndarray, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    codes = np.full(len(numeric), -1, dtype=np.int16)
    valid = np.isfinite(numeric)
    if edges.size:
        codes[valid] = np.searchsorted(edges, numeric[valid], side="right").astype(np.int16)
    else:
        codes[valid] = 0
    return pd.Series(codes, index=values.index).astype(str).radd(prefix + "_")


def _t3_sfx(series: pd.Series) -> pd.Series:
    t3 = series.astype(str)
    parsed = t3.str.extract(r"^(-?\d+(?:\.\d+)?)([A-Za-z])$")
    return parsed[1].fillna("__NONE__").astype(str)


def _days_fixed(days: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(days, errors="coerce").to_numpy(dtype=float)
    idx = np.digitize(np.nan_to_num(numeric, nan=-1.0), DAYS_FIXED_EDGES[1:-1], right=False)
    # digitize with truncated edges → 0..6
    labels = np.array(DAYS_FIXED_LABELS, dtype=object)
    out = labels[np.clip(idx, 0, len(labels) - 1)]
    out = out.astype(object)
    out[~np.isfinite(numeric)] = "__NA__"
    return pd.Series(out, index=days.index, dtype=object)


def add_gap_cats(frame: pd.DataFrame, edges: dict[str, np.ndarray]) -> pd.DataFrame:
    """Append mining gap categorical columns to an already-enriched frame."""
    out = frame.copy()
    days = pd.to_numeric(out["days"], errors="coerce")
    cond = pd.to_numeric(out["condition"], errors="coerce")
    ratio = cond / (days.abs() + 1.0)

    days5 = _bin_codes(days, edges["days5"], "d5")
    cond5 = _bin_codes(cond, edges["cond5"], "c5")
    ratio5 = _bin_codes(ratio, edges["ratio5"], "r5")
    ratio10 = _bin_codes(ratio, edges["ratio10"], "r10")
    days_fixed = _days_fixed(days)

    t3_sfx = _t3_sfx(out["t3"]) if "t3" in out.columns else pd.Series("__NONE__", index=out.index)
    code = out["code"].astype(str) if "code" in out.columns else pd.Series("__NA__", index=out.index)
    region = out["region"].astype(str) if "region" in out.columns else pd.Series("__NA__", index=out.index)
    source = out["source"].astype(str) if "source" in out.columns else pd.Series("__NA__", index=out.index)
    version = out["version"].astype(str) if "version" in out.columns else pd.Series("__NA__", index=out.index)
    # car token from source CAR_x|ENG_y
    car = source.str.extract(r"(CAR_\d+)", expand=False).fillna("__NA__")

    w1 = pd.to_numeric(out.get("w1"), errors="coerce").fillna(-1).astype(int)
    w2 = pd.to_numeric(out.get("w2"), errors="coerce").fillna(-1).astype(int)
    w_pair = w1.astype(str) + "_" + w2.astype(str)

    age = pd.to_numeric(out.get("age_range"), errors="coerce")
    age_coarse = age.clip(upper=8).fillna(-1).astype(int).astype(str)
    age_coarse = age_coarse.where(age.notna(), "__NA__")

    t1 = pd.to_numeric(out.get("t1"), errors="coerce").fillna(0).astype(int).clip(0, 1)
    t2 = pd.to_numeric(out.get("t2"), errors="coerce").fillna(0).astype(int).clip(0, 1)
    t_pair = t1.astype(str) + "_" + t2.astype(str)
    r1 = pd.to_numeric(out.get("r1"), errors="coerce").fillna(0).astype(int).clip(0, 1)
    r2 = pd.to_numeric(out.get("r2"), errors="coerce").fillna(0).astype(int).clip(0, 1)
    r_pair = r1.astype(str) + "_" + r2.astype(str)
    c1 = pd.to_numeric(out.get("c1"), errors="coerce").fillna(0).astype(int).clip(0, 1)
    c2 = pd.to_numeric(out.get("c2"), errors="coerce").fillna(0).astype(int).clip(0, 1)
    c_pair = c1.astype(str) + "_" + c2.astype(str)

    out["gap_days_fixed"] = days_fixed.astype(str)
    out["gap_t3_sfx"] = t3_sfx
    out["gap_w_pair"] = w_pair
    out["gap_age_coarse"] = age_coarse.astype(str)
    out["gap_ratio5"] = ratio5
    out["gap_ratio10"] = ratio10
    out["gap_days5"] = days5
    out["gap_cond5"] = cond5
    out["gap_t3sfx_code_days5"] = (t3_sfx + "|" + code + "|" + days5).astype(str)
    out["gap_wpair_days5"] = (w_pair + "|" + days5).astype(str)
    out["gap_agec_days5"] = (age_coarse.astype(str) + "|" + days5).astype(str)
    out["gap_ratio5_region"] = (ratio5 + "|" + region).astype(str)
    out["gap_ratio5_source"] = (ratio5 + "|" + source).astype(str)
    out["gap_ratio5_code"] = (ratio5 + "|" + code).astype(str)
    out["gap_ratio5_t3sfx"] = (ratio5 + "|" + t3_sfx).astype(str)
    out["gap_daysfix_source"] = (days_fixed.astype(str) + "|" + source).astype(str)
    out["gap_daysfix_cond5"] = (days_fixed.astype(str) + "|" + cond5).astype(str)
    out["gap_daysfix_t3sfx"] = (days_fixed.astype(str) + "|" + t3_sfx).astype(str)
    out["gap_daysfix_code"] = (days_fixed.astype(str) + "|" + code).astype(str)
    out["gap_code_days5"] = (code + "|" + days5).astype(str)
    out["gap_car_days5"] = (car + "|" + days5).astype(str)
    out["gap_version_days5"] = (version + "|" + days5).astype(str)
    out["gap_wpair_ratio5"] = (w_pair + "|" + ratio5).astype(str)
    out["gap_cond5_source"] = (cond5 + "|" + source).astype(str)
    out["gap_tpair_days5"] = (t_pair + "|" + days5).astype(str)
    out["gap_rpair_days5"] = (r_pair + "|" + days5).astype(str)
    out["gap_cpair_days5"] = (c_pair + "|" + days5).astype(str)
    out["gap_agec_ratio5"] = (age_coarse.astype(str) + "|" + ratio5).astype(str)
    return out
