"""Aggressive but fold-local plus_ultra features for B6pro (no TE)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from insurance_claim.b6_gap_features import GAP_CAT_COLS, add_gap_cats, fit_gap_edges
from insurance_claim.v10_plus.plus_features import build_plus, parse_frame


def build_plus_ultra(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    tr, va, te, cats = build_plus(X_tr, X_va, X_te)
    edges = fit_gap_edges(X_tr)

    def gap_block(raw: pd.DataFrame) -> pd.DataFrame:
        return add_gap_cats(raw.copy(), edges).loc[:, list(GAP_CAT_COLS)].copy()

    gtr, gva, gte = gap_block(X_tr), gap_block(X_va), gap_block(X_te)

    # Fold-local PCA on x0..x17 (keep a few components)
    xcols = [f"x{i}" for i in range(18) if f"x{i}" in X_tr.columns]
    if xcols:
        Xtr_x = X_tr[xcols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        pca = PCA(n_components=min(6, len(xcols)), random_state=0)
        tr_p = pca.fit_transform(Xtr_x)
        va_p = pca.transform(X_va[xcols].apply(pd.to_numeric, errors="coerce").fillna(0.0))
        te_p = pca.transform(X_te[xcols].apply(pd.to_numeric, errors="coerce").fillna(0.0))
        for i in range(tr_p.shape[1]):
            tr[f"pca_x_{i}"] = tr_p[:, i]
            va[f"pca_x_{i}"] = va_p[:, i]
            te[f"pca_x_{i}"] = te_p[:, i]

    # Extra numeric physics
    for df, raw in ((tr, X_tr), (va, X_va), (te, X_te)):
        days = pd.to_numeric(raw["days"], errors="coerce")
        cond = pd.to_numeric(raw["condition"], errors="coerce")
        df["days_sq"] = (days / 365.25) ** 2
        df["cond_sq"] = cond.fillna(cond.median() if cond.notna().any() else 0) ** 2
        df["days_x_age"] = days * pd.to_numeric(raw.get("age_range"), errors="coerce").fillna(0)
        df["log_cc"] = np.log1p(pd.to_numeric(raw.get("cc"), errors="coerce").clip(lower=0))

    # Extra cats from mining midband ideas
    def extra_cats(raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=raw.index)
        days = pd.to_numeric(raw["days"], errors="coerce")
        # coarser days decile-like via fixed edges
        bins = [-np.inf, 700, 1255, 2500, 4000, 5500, 7000, 8800, 9500, 10200, np.inf]
        labels = [f"dd{i}" for i in range(len(bins) - 1)]
        out["days10"] = pd.cut(days, bins=bins, labels=labels).astype(str)
        out["region_source"] = raw["region"].astype(str) + "|" + raw["source"].astype(str)
        out["code_grades"] = raw["code"].astype(str) + "|" + raw["grades"].astype(str)
        out["liv_age"] = raw["livability"].astype(str) + "|" + raw["age_range"].astype(str)
        out["days10_region"] = out["days10"] + "|" + raw["region"].astype(str)
        out["days10_source"] = out["days10"] + "|" + raw["source"].astype(str)
        return out

    etr, eva, ete = extra_cats(X_tr), extra_cats(X_va), extra_cats(X_te)
    extra_cat_names = list(etr.columns)

    def merge(base, *extras):
        out = base.reset_index(drop=True)
        for e in extras:
            out = pd.concat([out, e.reset_index(drop=True)], axis=1)
        return out.loc[:, ~out.columns.duplicated()]

    tr = merge(tr, gtr, etr)
    va = merge(va, gva, eva).reindex(columns=tr.columns)
    te = merge(te, gte, ete).reindex(columns=tr.columns)
    cats = list(dict.fromkeys(list(cats) + list(GAP_CAT_COLS) + extra_cat_names))
    for c in cats:
        for d in (tr, va, te):
            if c in d.columns:
                d[c] = d[c].astype(str).fillna("__MISSING__")
    return tr, va, te, cats
