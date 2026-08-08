"""Long-exposure / aging-curve features for auto insurance claim risk.

Business rationale (docs/business_feature_synergy_report.md):
- `days` is the dominant exposure proxy; claim rate rises ~4.9%→14.3% by decile.
- Region×days slopes are heterogeneous (same high-vs-low days OR can flip by region).
- Long exposure (days≳3000) is ~66% of rows / ~79% of claims but B7 max3 AUC≈0.66 there.
- Aging curves: vehicle condition decay vs exposure, car/version risk paths, age clocks.

All quantile edges / within-group ranks are fit on the training fold only (no TE).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from insurance_claim.b6_gap_features import add_gap_cats, fit_gap_edges
from insurance_claim.train_b5_focus import build_b5, enrich
from insurance_claim.train_b6 import build_gap

LONG_EXTRA_CATS = (
    "long_days_fine",
    "long_region_days_fine",
    "long_car_days_fine",
    "long_version_days_fine",
    "long_code_days_fine",
    "long_cond_days_fine",
    "long_region_cond5",
    "long_agec_days_fine",
    "long_wpair_days_fine",
    "long_t3sfx_days_fine",
    "long_flag",
)


def _quantile_edges(series: pd.Series, bins: int) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return np.array([], dtype=float)
    edges = np.unique(finite.quantile(np.linspace(0, 1, bins + 1)).to_numpy(dtype=float))
    return edges[1:-1] if len(edges) > 1 else np.array([], dtype=float)


def _bin_codes(values: pd.Series, edges: np.ndarray, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    codes = np.full(len(numeric), -1, dtype=np.int16)
    valid = np.isfinite(numeric)
    if edges.size:
        codes[valid] = np.searchsorted(edges, numeric[valid], side="right").astype(np.int16)
    else:
        codes[valid] = 0
    return pd.Series(codes, index=values.index).astype(str).radd(prefix + "_")


def _parse_car(source: pd.Series) -> pd.Series:
    s = source.astype(str)
    car = s.str.extract(r"(CAR_\d+)", expand=False)
    return car.fillna("__NA__").astype(str)


def _t3_sfx(series: pd.Series) -> pd.Series:
    t3 = series.astype(str)
    parsed = t3.str.extract(r"^(-?\d+(?:\.\d+)?)([A-Za-z])$")
    return parsed[1].fillna("__NONE__").astype(str)


def _group_rank(frame: pd.DataFrame, group_col: str, value_col: str, out_col: str) -> pd.Series:
    """Percentile rank of value within group (fold-local; uses full frame ranks)."""
    values = pd.to_numeric(frame[value_col], errors="coerce")
    g = frame[group_col].astype(str)
    # rank pct within group; NaN → 0.5
    ranks = values.groupby(g).rank(method="average", pct=True)
    return ranks.fillna(0.5).rename(out_col)


def fit_long_edges(X_tr: pd.DataFrame) -> dict[str, np.ndarray]:
    days = pd.to_numeric(X_tr["days"], errors="coerce")
    cond = pd.to_numeric(X_tr["condition"], errors="coerce")
    return {
        "days_fine": _quantile_edges(days, 10),
        "cond5": _quantile_edges(cond, 5),
    }


def add_long_aging(frame: pd.DataFrame, edges: dict[str, np.ndarray]) -> pd.DataFrame:
    """Append aging-curve categoricals + numeric exposure residuals."""
    out = frame.copy()
    days = pd.to_numeric(out["days"], errors="coerce")
    cond = pd.to_numeric(out["condition"], errors="coerce")
    ratio = cond / (days.abs() + 1.0)

    days_fine = _bin_codes(days, edges["days_fine"], "df")
    cond5 = _bin_codes(cond, edges["cond5"], "c5")
    region = out["region"].astype(str) if "region" in out.columns else pd.Series("__NA__", index=out.index)
    version = out["version"].astype(str) if "version" in out.columns else pd.Series("__NA__", index=out.index)
    code = out["code"].astype(str) if "code" in out.columns else pd.Series("__NA__", index=out.index)
    source = out["source"].astype(str) if "source" in out.columns else pd.Series("__NA__", index=out.index)
    car = _parse_car(source)
    t3_sfx = _t3_sfx(out["t3"]) if "t3" in out.columns else pd.Series("__NONE__", index=out.index)

    w1 = pd.to_numeric(out.get("w1"), errors="coerce").fillna(-1).astype(int)
    w2 = pd.to_numeric(out.get("w2"), errors="coerce").fillna(-1).astype(int)
    w_pair = w1.astype(str) + "_" + w2.astype(str)

    age = pd.to_numeric(out.get("age_range"), errors="coerce")
    age_coarse = age.clip(upper=8).fillna(-1).astype(int).astype(str)
    age_coarse = age_coarse.where(age.notna(), "__NA__")

    long_flag = (days >= 3000).fillna(False).astype(int).astype(str)

    out["long_days_fine"] = days_fine
    out["long_region_days_fine"] = (region + "|" + days_fine).astype(str)
    out["long_car_days_fine"] = (car + "|" + days_fine).astype(str)
    out["long_version_days_fine"] = (version + "|" + days_fine).astype(str)
    out["long_code_days_fine"] = (code + "|" + days_fine).astype(str)
    out["long_cond_days_fine"] = (cond5 + "|" + days_fine).astype(str)
    out["long_region_cond5"] = (region + "|" + cond5).astype(str)
    out["long_agec_days_fine"] = (age_coarse.astype(str) + "|" + days_fine).astype(str)
    out["long_wpair_days_fine"] = (w_pair + "|" + days_fine).astype(str)
    out["long_t3sfx_days_fine"] = (t3_sfx + "|" + days_fine).astype(str)
    out["long_flag"] = long_flag

    # Numeric aging signals (not TE): within-group exposure percentile + cond residual proxy
    tmp = pd.DataFrame(
        {
            "region": region,
            "car": car,
            "version": version,
            "days": days,
            "condition": cond,
            "ratio": ratio,
        },
        index=out.index,
    )
    out["long_days_pct_region"] = _group_rank(tmp, "region", "days", "long_days_pct_region")
    out["long_days_pct_car"] = _group_rank(tmp, "car", "days", "long_days_pct_car")
    out["long_days_pct_version"] = _group_rank(tmp, "version", "days", "long_days_pct_version")
    out["long_cond_pct_region"] = _group_rank(tmp, "region", "condition", "long_cond_pct_region")
    out["long_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)
    out["long_log_days"] = np.log1p(days.clip(lower=0))
    out["long_days_x_invcond"] = days / (cond.abs() + 1.0)
    # exposure beyond regional median (signed)
    med_days = tmp.groupby("region")["days"].transform("median")
    out["long_days_minus_region_med"] = days - med_days
    return out


def build_long_aging(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Gap FE + long-exposure aging cats/numerics (fold-local edges)."""
    tr_g, va_g, te_g, cats_g = build_gap(X_tr, X_va, X_te)
    edges = fit_long_edges(X_tr)

    def aging_block(raw: pd.DataFrame) -> pd.DataFrame:
        enriched = enrich(raw)
        return add_long_aging(enriched, edges)

    atr, ava, ate = aging_block(X_tr), aging_block(X_va), aging_block(X_te)
    extra_cols = [c for c in atr.columns if c.startswith("long_")]

    def merge(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
        block = extra.loc[:, extra_cols].copy()
        out = pd.concat([base.reset_index(drop=True), block.reset_index(drop=True)], axis=1)
        return out.loc[:, ~out.columns.duplicated()]

    tr = merge(tr_g, atr)
    va = merge(va_g, ava).reindex(columns=tr.columns)
    te = merge(te_g, ate).reindex(columns=tr.columns)

    cat_extra = [c for c in LONG_EXTRA_CATS if c in tr.columns]
    cats = list(dict.fromkeys(list(cats_g) + cat_extra))
    tr, va, te = tr.copy(), va.copy(), te.copy()
    for c in cats:
        tr[c] = tr[c].astype(str).fillna("__MISSING__")
        va[c] = va[c].astype(str).fillna("__MISSING__")
        te[c] = te[c].astype(str).fillna("__MISSING__")
    for c in tr.columns:
        if c in cats:
            continue
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
        med = float(tr[c].median()) if tr[c].notna().any() else 0.0
        tr[c] = tr[c].fillna(med)
        va[c] = pd.to_numeric(va[c], errors="coerce").fillna(med)
        te[c] = pd.to_numeric(te[c], errors="coerce").fillna(med)
    return tr, va, te, cats


def build_long_keepx(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """B5/gap path but re-attach x0..x18 (dropped by enrich) + aging cats.

    Hypothesis: latent embeddings help residual ranking inside long exposure
    where tree models over-rely on days main effect.
    """
    tr, va, te, cats = build_long_aging(X_tr, X_va, X_te)
    xcols = [f"x{i}" for i in range(19) if f"x{i}" in X_tr.columns]

    def attach(base: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
        block = raw.loc[:, xcols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        out = pd.concat([base.reset_index(drop=True), block], axis=1)
        return out.loc[:, ~out.columns.duplicated()]

    tr = attach(tr, X_tr)
    va = attach(va, X_va).reindex(columns=tr.columns)
    te = attach(te, X_te).reindex(columns=tr.columns)
    tr, va, te = tr.copy(), va.copy(), te.copy()
    for c in xcols:
        med = float(tr[c].median()) if tr[c].notna().any() else 0.0
        tr[c] = tr[c].fillna(med)
        va[c] = va[c].fillna(med)
        te[c] = te[c].fillna(med)
    return tr, va, te, cats
