"""V10 plus features for B7 (fold-local; no TE in selected path).

Adapted from reference/v10 / V10 package. Fixes t3_num_z to use fold-local
group maps (train-fold med/std) instead of transform+reindex leakage/NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def parse_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month_n"] = pd.to_numeric(
        out["month"].astype(str).str.replace("M", "", regex=False), errors="coerce"
    )
    m = out["t3"].astype(str).str.extract(r"([+-]?\d+(?:\.\d+)?)([A-Za-z]+)?")
    out["t3_num"] = pd.to_numeric(m[0], errors="coerce")
    out["t3_unit"] = m[1].fillna("__none__")
    cm = out["source"].astype(str).str.extract(r"CAR_(\d+)")
    out["car"] = pd.to_numeric(cm[0], errors="coerce")
    out["version_n"] = pd.to_numeric(
        out["version"].astype(str).str.replace("v", "", regex=False), errors="coerce"
    )
    out["grades_ord"] = out["grades"].map({"s": 1, "ss": 2, "sss": 3})
    out["age_range8"] = np.where(out["age_range"] >= 8, 8, out["age_range"]).astype(int)
    return out


def root_features(Xtr, Xva, Xte):
    def base(df, Xtr_ref):
        out = pd.DataFrame(index=df.index)
        out["days"] = df["days"].astype(float)
        out["years"] = out["days"] / 365.25
        out["is_new"] = (out["years"] < 1).astype(int)
        out["days_cap30"] = out["days"].clip(upper=30 * 365.25)
        out["over30"] = (out["days"] > 30 * 365.25).astype(int)
        out["condition_isna"] = df["condition"].isna().astype(int)
        out["condition_winsor"] = df["condition"].clip(
            upper=float(Xtr_ref["condition"].quantile(0.99))
        )
        out["x20_clip"] = df["x20"].clip(
            lower=float(Xtr_ref["x20"].quantile(0.01)),
            upper=float(Xtr_ref["x20"].quantile(0.99)),
        )
        out["x16_clip"] = df["x16"].clip(lower=float(Xtr_ref["x16"].quantile(0.01)))
        out["max_g_over_V"] = df["max_g"] / df["V"].replace(0, np.nan)
        out["V_over_cc"] = df["V"] / df["cc"].replace(0, np.nan)
        out["log_max_g"] = np.log1p(df["max_g"])
        out["w_both"] = ((df["w1"] == 1) & (df["w2"] == 1)).astype(int)
        out["w_neither"] = ((df["w1"] == 0) & (df["w2"] == 0)).astype(int)
        out["t2_and_r2"] = ((df["t2"] == 1) & (df["r2"] == 1)).astype(int)
        out["t1_and_c2"] = ((df["t1"] == 1) & (df["c2"] == 1)).astype(int)
        out["age_range8"] = np.where(df["age_range"] >= 8, 8, df["age_range"]).astype(int)
        for c in [
            "cc",
            "condition",
            "V",
            "x0",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "x9",
            "x10",
            "x11",
            "x12",
            "x13",
            "x14",
            "x15",
            "x16",
            "x17",
            "x18",
            "x20",
            "t1",
            "t2",
            "r1",
            "r2",
            "c1",
            "c2",
            "w1",
            "max_g",
            "age_range",
            "livability",
            "month_n",
            "t3_num",
            "car",
            "version_n",
            "grades_ord",
        ]:
            out[c] = df[c]
        return out

    tr, va, te = base(Xtr, Xtr), base(Xva, Xtr), base(Xte, Xtr)
    edges = np.unique(np.quantile(Xtr["days"].astype(float), np.linspace(0, 1, 6)))[1:-1]
    cond_edges = np.unique(
        np.quantile(Xtr["condition"].dropna().astype(float), np.linspace(0, 1, 5))
    )[1:-1]
    cond_med = float(Xtr["condition"].median())
    for dfx, out in ((Xtr, tr), (Xva, va), (Xte, te)):
        db = pd.Series(
            np.searchsorted(edges, dfx["days"].astype(float), side="right"), index=dfx.index
        ).astype(str)
        cb = pd.Series(
            np.searchsorted(
                cond_edges, dfx["condition"].astype(float).fillna(cond_med), side="right"
            ),
            index=dfx.index,
        ).astype(str)
        out["days5_x_region"] = (db + "|" + dfx["region"].astype(str)).astype(str)
        out["days5_x_condition4"] = (db + "|" + cb).astype(str)
        out["days5_x_source"] = (db + "|" + dfx["source"].astype(str)).astype(str)
        out["region_x_age8"] = (
            dfx["region"].astype(str) + "|" + out["age_range8"].astype(str)
        ).astype(str)
        out["region_x_source"] = (
            dfx["region"].astype(str) + "|" + dfx["source"].astype(str)
        ).astype(str)
        out["region_x_version"] = (
            dfx["region"].astype(str) + "|" + dfx["version"].astype(str)
        ).astype(str)
        out["month"] = dfx["month"].astype(str)
        out["region"] = dfx["region"].astype(str)
        out["source"] = dfx["source"].astype(str)
        out["code"] = dfx["code"].astype(str)
        out["version"] = dfx["version"].astype(str)
        out["t3_unit"] = dfx["t3_unit"].astype(str)
        out["grades"] = dfx["grades"].astype(str)
        out["age_range8"] = out["age_range8"].astype(str)
    cat_names = [
        "month",
        "region",
        "source",
        "code",
        "version",
        "t3_unit",
        "grades",
        "age_range8",
        "days5_x_region",
        "days5_x_condition4",
        "days5_x_source",
        "region_x_age8",
        "region_x_source",
        "region_x_version",
    ]
    return tr, va, te, cat_names


def extra_features(tr, va, te, Xtr, Xva, Xte):
    """Non-TE extras; fold-local maps for t3_num_z."""
    unit_med = Xtr.groupby("t3_unit")["t3_num"].median()
    unit_std = Xtr.groupby("t3_unit")["t3_num"].std().replace(0, np.nan)

    def add(dfx, out):
        out["w_conflict"] = ((dfx["w1"] + dfx["w2"]) != 1).astype(int)
        med = dfx["t3_unit"].map(unit_med)
        std = dfx["t3_unit"].map(unit_std).fillna(unit_std.median()).clip(lower=1e-6)
        out["t3_num_z"] = (dfx["t3_num"] - med) / std
        out["x20_grid"] = np.round(dfx["x20"] / 0.04).astype(int).astype(str)
        out["livability_cat"] = dfx["livability"].astype(str)
        out["region_x_livability"] = (
            dfx["region"].astype(str) + "|" + dfx["livability"].astype(str)
        ).astype(str)
        out["days_x_age8"] = (
            dfx["days"].round(-2).astype(int).astype(str) + "|" + out["age_range8"].astype(str)
        ).astype(str)
        out["log1p_condition"] = np.log1p(dfx["condition"].astype(float).clip(lower=0))
        xs = dfx[[f"x{i}" for i in range(18)]].astype(float)
        out["x_row_mean"] = xs.mean(axis=1)
        out["x_row_std"] = xs.std(axis=1)
        out["x_row_max"] = xs.max(axis=1)
        out["x_row_min"] = xs.min(axis=1)
        out["x_row_abs_mean"] = xs.abs().mean(axis=1)

    add(Xtr, tr)
    add(Xva, va)
    add(Xte, te)
    cats = ["x20_grid", "livability_cat", "region_x_livability", "days_x_age8"]
    return tr, va, te, cats


def prepare(tr, va, te, cat_names):
    tr, va, te = tr.copy(), va.copy(), te.copy()
    for c in cat_names:
        for d in (tr, va, te):
            d[c] = d[c].astype(str).fillna("__MISSING__")
    for c in tr.columns:
        if c in cat_names:
            continue
        for d in (tr, va, te):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        med = float(tr[c].median()) if tr[c].notna().any() else 0.0
        tr[c] = tr[c].fillna(med)
        va[c] = va[c].fillna(med)
        te[c] = te[c].fillna(med)
    return tr, va, te, list(cat_names)


def build_plus(X_tr, X_va, X_te):
    """Return frames + cat column names (for CatBoost cat_features=names)."""
    X_tr = parse_frame(X_tr)
    X_va = parse_frame(X_va)
    X_te = parse_frame(X_te)
    # drop near-id x19 if present (V10 recipe keeps x0-x18, drops x19)
    for d in (X_tr, X_va, X_te):
        if "x19" in d.columns:
            d.drop(columns=["x19"], inplace=True)
        if "id" in d.columns:
            d.drop(columns=["id"], inplace=True)
    tr, va, te, cats = root_features(X_tr, X_va, X_te)
    tr, va, te, extra = extra_features(tr, va, te, X_tr, X_va, X_te)
    cats = cats + extra
    return prepare(tr, va, te, cats)


def build_plus_mine(X_tr, X_va, X_te):
    """Plus + B6 gap cats + FN-oriented crosses (fold-local, no TE)."""
    from insurance_claim.b6_gap_features import GAP_CAT_COLS, add_gap_cats, fit_gap_edges
    from insurance_claim.train_b5_focus import enrich

    tr, va, te, cats = build_plus(X_tr, X_va, X_te)
    edges = fit_gap_edges(X_tr)

    def part(raw):
        g = add_gap_cats(enrich(raw), edges)
        d = parse_frame(raw)
        out = g.loc[:, [c for c in GAP_CAT_COLS if c in g.columns]].copy()
        out["mine_region_month"] = d["region"].astype(str) + "|" + d["month"].astype(str)
        out["mine_source_code"] = d["source"].astype(str) + "|" + d["code"].astype(str)
        out["mine_t3unit_code"] = d["t3_unit"].astype(str) + "|" + d["code"].astype(str)
        car = d["car"].fillna(-1).astype(int).astype(str)
        out["mine_car_code"] = car + "|" + d["code"].astype(str)
        if "x19" in raw.columns:
            out["mine_x19"] = raw["x19"].astype(str)
        return out

    def merge(a, b):
        out = pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1)
        return out.loc[:, ~out.columns.duplicated()]

    gtr, gva, gte = part(X_tr), part(X_va), part(X_te)
    tr = merge(tr, gtr)
    va = merge(va, gva).reindex(columns=tr.columns)
    te = merge(te, gte).reindex(columns=tr.columns)
    extra = [
        c
        for c in tr.columns
        if c.startswith("mine_") or c.startswith("gap_") or c in GAP_CAT_COLS
    ]
    cats = list(dict.fromkeys(list(cats) + extra))
    return prepare(tr, va, te, cats)
