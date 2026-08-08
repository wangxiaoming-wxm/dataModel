"""Residual / anti-monotonic long-exposure features for claim ranking.

Business insight from LL pair audit on B7/closest:
- Correct pairs: claim has higher days, lower condition (model already knows).
- Wrong pairs: claim has *lower* days / *higher* condition — exceptions to the
  exposure monotonicity. Same-region LL pair accuracy ≈0.63.
- Mid condition quintiles inside long have AUC≈0.61–0.63 (worst slice).

Goal: fold-local features that let a hetero model (LGBM/XGB) rank residual risk
after a simple days×condition baseline, without global TE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from insurance_claim.b6pro_long_features import (
    _bin_codes,
    _group_rank,
    _parse_car,
    _quantile_edges,
    _t3_sfx,
)


# Business fixed exposure windows (docs/B6_DATA_MINING.md §1.5)
DAYS_FIXED_EDGES = np.array([700.0, 2500.0, 5000.0, 7000.0, 9000.0, 10000.0])
DAYS_FIXED_LABELS = ("d0_700", "d700_2500", "d2500_5k", "d5k_7k", "d7k_9k", "d9k_10k", "d10k_p")

HEALTHY_TE_KEYS = (
    "te_region_days5",
    "te_ratio_region",
    "te_car_days5",
    "te_wpair_days5",
    "te_t3sfx_code_days5",
    "te_region",
    "te_days_fixed",
)


def _days_fixed(days: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(days, errors="coerce").to_numpy(dtype=float)
    codes = np.searchsorted(DAYS_FIXED_EDGES, numeric, side="right").astype(np.int16)
    codes = np.clip(codes, 0, len(DAYS_FIXED_LABELS) - 1)
    nan = ~np.isfinite(numeric)
    out = pd.Series([DAYS_FIXED_LABELS[i] for i in codes], index=days.index)
    out = out.where(~nan, "__NA__")
    return out.astype(str)


def _oof_te_map(
    keys: pd.Series, y: np.ndarray, prior: float = 10.0
) -> dict[str, float]:
    """Credibility TE from a single training fold (fit only)."""
    global_mean = float(np.mean(y))
    frame = pd.DataFrame({"k": keys.astype(str).to_numpy(), "y": y})
    stats = frame.groupby("k")["y"].agg(["sum", "count"])
    te = (stats["sum"] + prior * global_mean) / (stats["count"] + prior)
    return te.to_dict()


def _apply_te(keys: pd.Series, mapping: dict[str, float], default: float) -> np.ndarray:
    return keys.astype(str).map(mapping).fillna(default).to_numpy(dtype=float)


def fit_resid_edges(X_tr: pd.DataFrame) -> dict[str, np.ndarray]:
    days = pd.to_numeric(X_tr["days"], errors="coerce")
    cond = pd.to_numeric(X_tr["condition"], errors="coerce")
    ratio = cond / (days.abs() + 1.0)
    return {
        "days5": _quantile_edges(days, 5),
        "days10": _quantile_edges(days, 10),
        "cond5": _quantile_edges(cond, 5),
        "ratio5": _quantile_edges(ratio, 5),
    }


def _key_frame(raw: pd.DataFrame, edges: dict[str, np.ndarray]) -> pd.DataFrame:
    days = pd.to_numeric(raw["days"], errors="coerce")
    cond = pd.to_numeric(raw["condition"], errors="coerce")
    ratio = cond / (days.abs() + 1.0)
    region = raw["region"].astype(str) if "region" in raw.columns else pd.Series("__NA__", index=raw.index)
    source = raw["source"].astype(str) if "source" in raw.columns else pd.Series("__NA__", index=raw.index)
    car = _parse_car(source)
    code = raw["code"].astype(str) if "code" in raw.columns else pd.Series("__NA__", index=raw.index)
    t3sfx = _t3_sfx(raw["t3"]) if "t3" in raw.columns else pd.Series("__NONE__", index=raw.index)
    w1 = pd.to_numeric(raw.get("w1"), errors="coerce").fillna(-1).astype(int)
    w2 = pd.to_numeric(raw.get("w2"), errors="coerce").fillna(-1).astype(int)
    w_pair = w1.astype(str) + "_" + w2.astype(str)
    days5 = _bin_codes(days, edges["days5"], "d5")
    ratio5 = _bin_codes(ratio, edges["ratio5"], "r5")
    days_fixed = _days_fixed(days)
    return pd.DataFrame(
        {
            "region": region,
            "car": car,
            "code": code,
            "t3sfx": t3sfx,
            "w_pair": w_pair,
            "days5": days5,
            "ratio5": ratio5,
            "days_fixed": days_fixed,
            "region_days5": (region + "|" + days5).astype(str),
            "ratio_region": (ratio5 + "|" + region).astype(str),
            "car_days5": (car + "|" + days5).astype(str),
            "wpair_days5": (w_pair + "|" + days5).astype(str),
            "t3sfx_code_days5": (t3sfx + "|" + code + "|" + days5).astype(str),
        },
        index=raw.index,
    )


def build_long_resid_matrix(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray | pd.Series,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Numeric+ordinal matrix with fold-local healthy TE and residual numerics."""
    y_tr = np.asarray(y_tr, dtype=float)
    edges = fit_resid_edges(X_tr)
    ktr, kva, kte = _key_frame(X_tr, edges), _key_frame(X_va, edges), _key_frame(X_te, edges)
    prior = 10.0
    gmean = float(np.mean(y_tr))
    te_specs = {
        "te_region_days5": "region_days5",
        "te_ratio_region": "ratio_region",
        "te_car_days5": "car_days5",
        "te_wpair_days5": "wpair_days5",
        "te_t3sfx_code_days5": "t3sfx_code_days5",
        "te_region": "region",
        "te_days_fixed": "days_fixed",
    }
    te_maps = {name: _oof_te_map(ktr[col], y_tr, prior=prior) for name, col in te_specs.items()}

    def numeric_block(raw: pd.DataFrame, keys: pd.DataFrame, fit_ref: pd.DataFrame | None) -> pd.DataFrame:
        days = pd.to_numeric(raw["days"], errors="coerce")
        cond = pd.to_numeric(raw["condition"], errors="coerce")
        ratio = cond / (days.abs() + 1.0)
        liv = pd.to_numeric(raw.get("livability"), errors="coerce")
        age = pd.to_numeric(raw.get("age_range"), errors="coerce")
        cc = pd.to_numeric(raw.get("cc"), errors="coerce")
        V = pd.to_numeric(raw.get("V"), errors="coerce")
        max_g = pd.to_numeric(raw.get("max_g"), errors="coerce")
        x20 = pd.to_numeric(raw.get("x20"), errors="coerce")

        out = pd.DataFrame(index=raw.index)
        out["days"] = days
        out["log_days"] = np.log1p(days.clip(lower=0))
        out["condition"] = cond
        out["ratio"] = ratio.replace([np.inf, -np.inf], np.nan)
        out["livability"] = liv
        out["age_range"] = age
        out["age_coarse"] = age.clip(upper=8)
        out["cc"] = cc
        out["V"] = V
        out["max_g"] = max_g
        out["x20"] = x20
        out["days_x_invcond"] = days / (cond.abs() + 1.0)
        out["long_flag"] = (days >= 3000).astype(float)
        out["ultra_long"] = (days >= 10000).astype(float)
        out["mid_long"] = ((days >= 5000) & (days < 7000)).astype(float)

        # within-group ranks (use raw frame; for va/te ranks vs themselves is ok as relative)
        tmp = pd.DataFrame(
            {"region": keys["region"], "car": keys["car"], "days": days, "condition": cond, "ratio": ratio},
            index=raw.index,
        )
        out["days_pct_region"] = _group_rank(tmp, "region", "days", "days_pct_region")
        out["days_pct_car"] = _group_rank(tmp, "car", "days", "days_pct_car")
        out["cond_pct_region"] = _group_rank(tmp, "region", "condition", "cond_pct_region")
        out["ratio_pct_region"] = _group_rank(tmp, "region", "ratio", "ratio_pct_region")

        # signed deviation from regional median days/cond (anti-monotonic residual anchors)
        if fit_ref is None:
            med_days = tmp.groupby("region")["days"].transform("median")
            med_cond = tmp.groupby("region")["condition"].transform("median")
            # store for apply via map
            numeric_block._med_days = tmp.groupby("region")["days"].median().to_dict()  # type: ignore[attr-defined]
            numeric_block._med_cond = tmp.groupby("region")["condition"].median().to_dict()  # type: ignore[attr-defined]
        else:
            med_days = keys["region"].map(numeric_block._med_days).astype(float)  # type: ignore[attr-defined]
            med_cond = keys["region"].map(numeric_block._med_cond).astype(float)  # type: ignore[attr-defined]
            med_days = med_days.fillna(float(pd.to_numeric(fit_ref["days"], errors="coerce").median()))
            med_cond = med_cond.fillna(float(pd.to_numeric(fit_ref["condition"], errors="coerce").median()))
        out["days_minus_region_med"] = days - med_days
        out["cond_minus_region_med"] = cond - med_cond

        for name, col in te_specs.items():
            out[name] = _apply_te(keys[col], te_maps[name], gmean)

        # keep x0-x18 latent residuals
        for i in range(19):
            c = f"x{i}"
            if c in raw.columns:
                out[c] = pd.to_numeric(raw[c], errors="coerce")

        # ordinal cats as float codes (LGBM can treat as categorical separately)
        for c in ("region", "car", "code", "t3sfx", "w_pair", "days5", "days_fixed", "ratio5"):
            out[f"cat_{c}"] = keys[c].astype("category").cat.codes.astype(float)
        return out

    tr = numeric_block(X_tr, ktr, None)
    va = numeric_block(X_va, kva, X_tr)
    te = numeric_block(X_te, kte, X_tr)

    # days×condition logistic residual score (fold-fit only)
    base_cols = ["log_days", "condition", "ratio"]
    Xb = tr[base_cols].fillna(tr[base_cols].median()).to_numpy()
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(Xb, y_tr)
    for df, name in ((tr, "tr"), (va, "va"), (te, "te")):
        Xm = df[base_cols].fillna(tr[base_cols].median()).to_numpy()
        df["base_logit"] = clf.decision_function(Xm)
        df["base_prob"] = clf.predict_proba(Xm)[:, 1]

    cat_cols = [c for c in tr.columns if c.startswith("cat_")]
    # fillna
    for c in tr.columns:
        med = float(tr[c].median()) if tr[c].notna().any() else 0.0
        tr[c] = tr[c].fillna(med)
        va[c] = va[c].fillna(med)
        te[c] = te[c].fillna(med)
    va = va.reindex(columns=tr.columns)
    te = te.reindex(columns=tr.columns)
    return tr, va, te, cat_cols
