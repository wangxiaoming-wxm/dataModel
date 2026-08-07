#!/usr/bin/env python3
"""Deep EDA on NEW train/test only. All metrics from current CSVs."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUT = Path("/workspace/artifacts/eda_new")
OUT.mkdir(parents=True, exist_ok=True)


def safe_auc(y, s):
    s = np.asarray(s, dtype=float)
    if np.isnan(s).all() or np.nanstd(s) < 1e-12:
        return float("nan")
    mask = np.isfinite(s)
    if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], s[mask]))


def auc_abs(y, s):
    a = safe_auc(y, s)
    if np.isnan(a):
        return float("nan")
    return max(a, 1.0 - a)


def claim_rate_table(series, y, bins=None, top=30):
    s = series.copy()
    if bins is not None:
        s = pd.qcut(pd.to_numeric(s, errors="coerce"), q=bins, duplicates="drop")
    df = pd.DataFrame({"g": s.astype(str), "y": y})
    g = df.groupby("g", dropna=False)["y"].agg(["count", "sum", "mean"]).reset_index()
    g = g.rename(columns={"sum": "claims", "mean": "claim_rate"})
    return g.sort_values("count", ascending=False).head(top)


def psi(expected, actual, bins=10):
    e = pd.to_numeric(expected, errors="coerce").dropna()
    a = pd.to_numeric(actual, errors="coerce").dropna()
    if len(e) < 20 or len(a) < 20:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    e_cnt = np.histogram(e, bins=edges)[0].astype(float)
    a_cnt = np.histogram(a, bins=edges)[0].astype(float)
    e_p = (e_cnt + 1e-6) / (e_cnt.sum() + 1e-6 * len(e_cnt))
    a_p = (a_cnt + 1e-6) / (a_cnt.sum() + 1e-6 * len(a_cnt))
    return float(np.sum((a_p - e_p) * np.log(a_p / e_p)))


def cat_psi(expected, actual, top=50):
    e = expected.astype(str).fillna("__NA__")
    a = actual.astype(str).fillna("__NA__")
    cats = pd.concat([e, a]).value_counts().head(top).index
    e_p = e.value_counts(normalize=True).reindex(cats).fillna(0) + 1e-6
    a_p = a.value_counts(normalize=True).reindex(cats).fillna(0) + 1e-6
    e_p = e_p / e_p.sum()
    a_p = a_p / a_p.sum()
    return float(np.sum((a_p - e_p) * np.log(a_p / e_p)))


def parse_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["month_num"] = df["month"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    out["month_is_m2"] = (df["month"].astype(str) == "M2").astype(int)
    out["month_is_m1"] = (df["month"].astype(str) == "M1").astype(int)
    out["month_rare"] = (~df["month"].astype(str).isin(["M0", "M1", "M2", "M3", "M4"])).astype(int)
    src = df["source"].astype(str)
    out["car_id"] = src.str.extract(r"CAR_(\d+)")[0].astype(float)
    out["eng_id"] = src.str.extract(r"ENG_(\d+)")[0].astype(float)
    out["car_token"] = src.str.extract(r"(CAR_\d+)")[0].fillna("__NA__")
    out["eng_token"] = src.str.extract(r"(ENG_\d+)")[0].fillna("__NA__")
    out["source_has_pipe"] = src.str.contains(r"\|", regex=True).astype(int)
    t3 = df["t3"].astype(str)
    out["t3_num"] = t3.str.extract(r"([-+]?\d+(?:\.\d+)?)")[0].astype(float)
    out["t3_suffix"] = t3.str.extract(r"([A-Za-z]+)$")[0].fillna("__NA__")
    out["t3_prefix2"] = t3.str[:2]
    out["t3_bucket"] = pd.cut(
        out["t3_num"], bins=[0, 4.5, 4.8, 5.0, 5.2, 5.5, 10], include_lowest=True
    ).astype(str)
    ver = df["version"].astype(str)
    out["version_num"] = ver.str.extract(r"(\d+)")[0].astype(float)
    out["version_is_v1"] = (ver == "v1").astype(int)
    out["version_is_early"] = out["version_num"].isin([1, 2, 3, 4]).astype(int)
    out["version_is_late"] = (out["version_num"] >= 10).astype(int)
    out["grades_len"] = df["grades"].astype(str).str.len()
    out["grades_s_count"] = df["grades"].astype(str).str.count("s")
    out["region_len"] = df["region"].astype(str).str.len()
    out["region_is_hexlike"] = df["region"].astype(str).str.fullmatch(r"[0-9a-fA-F]+").astype(int)
    out["w1_eq_w2"] = (df["w1"] == df["w2"]).astype(int)
    out["w1_ne_w2"] = (df["w1"] != df["w2"]).astype(int)
    out["t1_eq_t2"] = (df["t1"] == df["t2"]).astype(int)
    out["r1_eq_r2"] = (df["r1"] == df["r2"]).astype(int)
    out["c1_eq_c2"] = (df["c1"] == df["c2"]).astype(int)
    out["condition_missing"] = df["condition"].isna().astype(int)
    return out


def oof_te_auc(cat, y, n_splits=5, seed=2026, alpha=20.0):
    y = y.astype(int).to_numpy() if hasattr(y, "astype") else np.asarray(y).astype(int)
    cats = pd.Series(cat).astype(str).fillna("__NA__").to_numpy()
    oof = np.zeros(len(y), dtype=float)
    global_mean = float(y.mean())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(np.zeros(len(y)), y):
        stats = pd.DataFrame({"c": cats[tr], "y": y[tr]}).groupby("c")["y"].agg(["sum", "count"])
        te = ((stats["sum"] + alpha * global_mean) / (stats["count"] + alpha)).to_dict()
        oof[va] = np.array([te.get(c, global_mean) for c in cats[va]], dtype=float)
    return (
        float(roc_auc_score(y, oof)),
        float(pd.Series(cats).nunique()),
        float(pd.Series(cats).value_counts().mean()),
    )


def main():
    train = pd.read_csv("/workspace/train.csv")
    test = pd.read_csv("/workspace/test.csv")
    y = train["label"].astype(int)
    report = {
        "meta": {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "claim_rate": float(y.mean()),
            "n_pos": int(y.sum()),
            "n_neg": int((y == 0).sum()),
            "train_missing_condition": int(train["condition"].isna().sum()),
            "test_missing_condition": int(test["condition"].isna().sum()),
            "note": "ALL metrics from NEW train.csv/test.csv only; old OOF discarded",
        }
    }

    uni_rows = []
    feature_cols = [c for c in train.columns if c not in ("id", "label")]
    mi_X, mi_names, discrete = [], [], []
    for c in feature_cols:
        s = train[c]
        if pd.api.types.is_numeric_dtype(s):
            vals = pd.to_numeric(s, errors="coerce")
            vals = vals.fillna(vals.median() if vals.notna().any() else 0)
            auc = safe_auc(y, vals)
            uni_rows.append(
                {
                    "col": c,
                    "type": "numeric",
                    "nunique": int(s.nunique(dropna=False)),
                    "auc": auc,
                    "auc_abs": max(auc, 1 - auc) if np.isfinite(auc) else np.nan,
                    "missing": int(s.isna().sum()),
                    "missing_rate": float(s.isna().mean()),
                }
            )
            mi_X.append(vals.to_numpy())
            mi_names.append(c)
            discrete.append(s.nunique() < 30)
        else:
            rates = y.groupby(s.astype(str)).mean()
            mapped = s.astype(str).map(rates)
            auc = safe_auc(y, mapped)
            oof_a, nun, mean_cnt = oof_te_auc(s, y)
            uni_rows.append(
                {
                    "col": c,
                    "type": "categorical",
                    "nunique": int(s.nunique(dropna=False)),
                    "auc_insample_te": auc,
                    "auc_oof_te": oof_a,
                    "auc_abs": oof_a,
                    "missing": int(s.isna().sum()),
                    "missing_rate": float(s.isna().mean()),
                    "mean_count_per_level": mean_cnt,
                }
            )
            codes, _ = pd.factorize(s.astype(str), sort=True)
            mi_X.append(codes.astype(float))
            mi_names.append(c)
            discrete.append(True)

    mi = mutual_info_classif(
        np.column_stack(mi_X), y, discrete_features=discrete, random_state=2026, n_neighbors=5
    )
    mi_map = {n: float(v) for n, v in zip(mi_names, mi)}
    for row in uni_rows:
        row["mutual_info"] = mi_map.get(row["col"], float("nan"))
    uni_df = pd.DataFrame(uni_rows).sort_values("auc_abs", ascending=False)
    uni_df.to_csv(OUT / "univariate_auc_mi.csv", index=False)
    report["univariate_top20"] = uni_df.head(20).to_dict(orient="records")
    report["univariate_all"] = uni_df.to_dict(orient="records")

    strat = {}
    for c in [
        "month",
        "region",
        "source",
        "t3",
        "version",
        "grades",
        "code",
        "age_range",
        "livability",
        "w1",
        "w2",
        "t1",
        "t2",
        "r1",
        "r2",
        "c1",
        "c2",
    ]:
        strat[c] = claim_rate_table(train[c], y, top=25).to_dict(orient="records")
    for c, bins in [
        ("days", 10),
        ("condition", 10),
        ("cc", 10),
        ("V", 10),
        ("max_g", 10),
    ]:
        strat[f"{c}_q{bins}"] = claim_rate_table(train[c], y, bins=bins, top=25).to_dict(
            orient="records"
        )
    report["claim_rate_strat"] = strat

    drift = {}
    for c in ["days", "condition", "livability", "cc", "V", "max_g", "x18", "x19", "x20"]:
        drift[c] = {
            "psi": psi(train[c], test[c]),
            "train_mean": float(pd.to_numeric(train[c], errors="coerce").mean()),
            "test_mean": float(pd.to_numeric(test[c], errors="coerce").mean()),
            "train_std": float(pd.to_numeric(train[c], errors="coerce").std()),
            "test_std": float(pd.to_numeric(test[c], errors="coerce").std()),
            "train_p50": float(pd.to_numeric(train[c], errors="coerce").median()),
            "test_p50": float(pd.to_numeric(test[c], errors="coerce").median()),
        }
    for c in ["region", "source", "t3", "version", "month", "grades", "code", "age_range"]:
        tr_levels = set(train[c].astype(str))
        te_levels = set(test[c].astype(str))
        drift[c] = {
            "cat_psi": cat_psi(train[c], test[c]),
            "train_nunique": len(tr_levels),
            "test_nunique": len(te_levels),
            "test_only_levels": sorted(te_levels - tr_levels)[:20],
            "train_only_levels": sorted(tr_levels - te_levels)[:20],
            "overlap_rate": float(len(tr_levels & te_levels) / max(1, len(tr_levels | te_levels))),
        }
    report["drift"] = drift

    days_bin = pd.qcut(train["days"], 10, duplicates="drop")
    cond_bin = pd.qcut(train["condition"].fillna(train["condition"].median()), 10, duplicates="drop")
    surface = (
        pd.DataFrame({"days_bin": days_bin.astype(str), "cond_bin": cond_bin.astype(str), "y": y})
        .groupby(["days_bin", "cond_bin"])["y"]
        .agg(["count", "mean"])
        .reset_index()
    )
    surface = surface[surface["count"] >= 30].sort_values("mean", ascending=False)
    report["days_condition_surface_top"] = surface.head(15).to_dict(orient="records")
    report["days_condition_surface_bottom"] = surface.tail(10).to_dict(orient="records")
    report["region_risk"] = (
        pd.DataFrame(
            {
                "region": train["region"].astype(str),
                "y": y,
                "days": train["days"],
                "condition": train["condition"],
            }
        )
        .groupby("region")
        .agg(
            n=("y", "size"),
            claim_rate=("y", "mean"),
            days_mean=("days", "mean"),
            condition_mean=("condition", "mean"),
        )
        .reset_index()
        .sort_values("claim_rate", ascending=False)
        .to_dict(orient="records")
    )

    ptr = parse_features(train)
    parsed_uni = []
    for c in ptr.columns:
        s = ptr[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 15:
            a = safe_auc(y, s.fillna(s.median()))
            parsed_uni.append(
                {
                    "feat": c,
                    "type": "num",
                    "auc": a,
                    "auc_abs": max(a, 1 - a) if np.isfinite(a) else np.nan,
                    "nunique": int(s.nunique()),
                }
            )
        else:
            oof_a, nun, mc = oof_te_auc(s.astype(str), y)
            parsed_uni.append(
                {
                    "feat": c,
                    "type": "cat/flag",
                    "auc_oof_te": oof_a,
                    "auc_abs": oof_a,
                    "nunique": int(pd.Series(s).nunique()),
                    "mean_count": mc,
                }
            )
    parsed_df = pd.DataFrame(parsed_uni).sort_values("auc_abs", ascending=False)
    parsed_df.to_csv(OUT / "parsed_feature_auc.csv", index=False)
    report["parsed_signals"] = parsed_df.to_dict(orient="records")
    for token in ["car_token", "eng_token", "t3_suffix", "version_num", "month_num"]:
        report[f"strat_{token}"] = claim_rate_table(ptr[token], y, top=20).to_dict(orient="records")

    atoms = pd.DataFrame(
        {
            "region": train["region"].astype(str),
            "source": train["source"].astype(str),
            "car": ptr["car_token"].astype(str),
            "eng": ptr["eng_token"].astype(str),
            "t3": train["t3"].astype(str),
            "t3_suf": ptr["t3_suffix"].astype(str),
            "t3_bucket": ptr["t3_bucket"].astype(str),
            "version": train["version"].astype(str),
            "ver_early": ptr["version_is_early"].astype(str),
            "month": train["month"].astype(str),
            "month_num": ptr["month_num"].astype(str),
            "grades": train["grades"].astype(str),
            "code": train["code"].astype(str),
            "age_range": train["age_range"].astype(str),
            "livability": train["livability"].astype(str),
            "days_q10": pd.qcut(train["days"], 10, duplicates="drop").astype(str),
            "cond_q10": pd.qcut(
                train["condition"].fillna(train["condition"].median()), 10, duplicates="drop"
            ).astype(str),
            "days_q5": pd.qcut(train["days"], 5, duplicates="drop").astype(str),
            "cond_q5": pd.qcut(
                train["condition"].fillna(train["condition"].median()), 5, duplicates="drop"
            ).astype(str),
            "w_conflict": ptr["w1_ne_w2"].astype(str),
            "c1": train["c1"].astype(str),
            "c2": train["c2"].astype(str),
        }
    )

    cross2_candidates = [
        ("days_q10", "region"),
        ("days_q10", "source"),
        ("days_q10", "car"),
        ("days_q10", "version"),
        ("days_q10", "t3_suf"),
        ("days_q10", "month"),
        ("days_q10", "livability"),
        ("days_q10", "age_range"),
        ("days_q10", "grades"),
        ("cond_q10", "region"),
        ("cond_q10", "source"),
        ("cond_q10", "car"),
        ("cond_q10", "version"),
        ("cond_q10", "t3_suf"),
        ("cond_q10", "livability"),
        ("region", "source"),
        ("region", "car"),
        ("region", "version"),
        ("region", "month"),
        ("region", "livability"),
        ("region", "t3_suf"),
        ("source", "version"),
        ("source", "month"),
        ("source", "livability"),
        ("car", "version"),
        ("car", "t3_suf"),
        ("car", "month"),
        ("car", "livability"),
        ("version", "month"),
        ("version", "livability"),
        ("version", "t3_suf"),
        ("month", "livability"),
        ("t3_suf", "livability"),
        ("days_q5", "cond_q5"),
        ("days_q10", "cond_q10"),
        ("w_conflict", "region"),
        ("w_conflict", "source"),
        ("grades", "source"),
        ("grades", "region"),
        ("code", "region"),
        ("code", "source"),
        ("age_range", "source"),
        ("age_range", "region"),
        ("age_range", "days_q10"),
    ]
    cross2_rows = []
    for a, b in cross2_candidates:
        key = atoms[a].astype(str) + "|" + atoms[b].astype(str)
        oof_a, nun, mc = oof_te_auc(key, y)
        oof_a1, _, _ = oof_te_auc(atoms[a], y)
        oof_a2, _, _ = oof_te_auc(atoms[b], y)
        cross2_rows.append(
            {
                "cross": f"{a}×{b}",
                "auc_oof_te": oof_a,
                "nunique": nun,
                "mean_count": mc,
                "base_a": oof_a1,
                "base_b": oof_a2,
                "lift_vs_best_base": oof_a - max(oof_a1, oof_a2),
                "sparse_risk": nun > 200 or mc < 25,
            }
        )
    cross2_df = pd.DataFrame(cross2_rows).sort_values("auc_oof_te", ascending=False)
    cross2_df.to_csv(OUT / "cross2_te_upperbound.csv", index=False)
    report["cross2_top"] = cross2_df.head(25).to_dict(orient="records")
    report["cross2_recommended"] = cross2_df[
        (cross2_df["lift_vs_best_base"] > 0.002) & (cross2_df["mean_count"] >= 15)
    ].head(20).to_dict(orient="records")

    cross3_candidates = [
        ("days_q5", "region", "source"),
        ("days_q5", "region", "car"),
        ("days_q5", "region", "version"),
        ("days_q5", "source", "version"),
        ("days_q5", "car", "version"),
        ("days_q5", "region", "month"),
        ("days_q5", "livability", "source"),
        ("days_q5", "livability", "region"),
        ("days_q10", "region", "car"),
        ("cond_q5", "region", "source"),
        ("cond_q5", "region", "version"),
        ("cond_q5", "source", "version"),
        ("days_q5", "cond_q5", "region"),
        ("days_q5", "cond_q5", "source"),
        ("days_q5", "cond_q5", "car"),
        ("days_q5", "cond_q5", "version"),
        ("region", "source", "version"),
        ("region", "car", "version"),
        ("region", "source", "month"),
        ("region", "car", "month"),
        ("source", "version", "month"),
        ("car", "version", "livability"),
        ("region", "version", "livability"),
        ("region", "t3_suf", "version"),
        ("days_q5", "t3_suf", "region"),
        ("days_q5", "grades", "region"),
        ("age_range", "source", "region"),
        ("age_range", "days_q5", "region"),
        ("w_conflict", "days_q5", "region"),
        ("code", "region", "source"),
    ]
    cross3_rows = []
    for a, b, c in cross3_candidates:
        key = atoms[a].astype(str) + "|" + atoms[b].astype(str) + "|" + atoms[c].astype(str)
        oof_a, nun, mc = oof_te_auc(key, y, alpha=30.0)
        cross3_rows.append(
            {
                "cross": f"{a}×{b}×{c}",
                "auc_oof_te": oof_a,
                "nunique": nun,
                "mean_count": mc,
                "sparse_risk": nun > 400 or mc < 12,
            }
        )
    cross3_df = pd.DataFrame(cross3_rows).sort_values("auc_oof_te", ascending=False)
    cross3_df.to_csv(OUT / "cross3_te_upperbound.csv", index=False)
    report["cross3_top"] = cross3_df.head(20).to_dict(orient="records")
    report["cross3_recommended"] = cross3_df[
        (~cross3_df["sparse_risk"]) & (cross3_df["auc_oof_te"] >= 0.60)
    ].to_dict(orient="records")

    xcols = [f"x{i}" for i in range(21)]
    Xx = train[xcols].apply(pd.to_numeric, errors="coerce").fillna(train[xcols].median())
    x_rows = []
    Z = np.column_stack(
        [
            train["days"].to_numpy(),
            train["condition"].fillna(train["condition"].median()).to_numpy(),
            train["livability"].to_numpy(),
            train["age_range"].to_numpy(),
            ptr["car_id"].fillna(-1).to_numpy(),
            ptr["version_num"].fillna(-1).to_numpy(),
            ptr["month_num"].fillna(-1).to_numpy(),
        ]
    )
    Zb = np.column_stack([np.ones(len(Z)), Z])
    for c in xcols:
        a = safe_auc(y, Xx[c])
        beta, _, _, _ = np.linalg.lstsq(Zb, Xx[c].to_numpy(), rcond=None)
        resid = Xx[c].to_numpy() - Zb @ beta
        a_res = safe_auc(y, resid)
        x_rows.append(
            {
                "col": c,
                "nunique": int(train[c].nunique()),
                "auc": a,
                "auc_abs": max(a, 1 - a) if np.isfinite(a) else np.nan,
                "resid_auc": a_res,
                "resid_auc_abs": max(a_res, 1 - a_res) if np.isfinite(a_res) else np.nan,
                "corr_days": float(np.corrcoef(Xx[c], train["days"])[0, 1]),
                "std": float(Xx[c].std()),
                "near_unique": int(train[c].nunique()) > 0.99 * len(train),
            }
        )
    x_df = pd.DataFrame(x_rows).sort_values("auc_abs", ascending=False)
    x_df.to_csv(OUT / "x_features_analysis.csv", index=False)
    report["x_features"] = x_df.to_dict(orient="records")

    Xs = StandardScaler().fit_transform(Xx)
    pca = PCA(n_components=10, random_state=2026)
    pcs = pca.fit_transform(Xs)
    pca_rows = []
    for i in range(pcs.shape[1]):
        a = safe_auc(y, pcs[:, i])
        pca_rows.append(
            {
                "pc": f"PC{i+1}",
                "var_ratio": float(pca.explained_variance_ratio_[i]),
                "auc": a,
                "auc_abs": max(a, 1 - a) if np.isfinite(a) else np.nan,
            }
        )
    skf = StratifiedKFold(5, shuffle=True, random_state=2026)
    pca_oof = {}
    for k in [1, 2, 3, 5, 8, 10]:
        oof = np.zeros(len(y))
        for tr, va in skf.split(pcs, y):
            clf = LogisticRegression(max_iter=500, C=0.5)
            clf.fit(pcs[tr, :k], y.iloc[tr])
            oof[va] = clf.predict_proba(pcs[va, :k])[:, 1]
        pca_oof[f"pc1-{k}"] = float(roc_auc_score(y, oof))
    X_emb = Xx[[f"x{i}" for i in range(18)]]
    row_stats = pd.DataFrame(
        {
            "x_mean": Xx.mean(axis=1),
            "x_std": Xx.std(axis=1),
            "x_min": Xx.min(axis=1),
            "x_max": Xx.max(axis=1),
            "x_absmean": Xx.abs().mean(axis=1),
            "x_l2": np.sqrt((Xx**2).sum(axis=1)),
            "x_range": Xx.max(axis=1) - Xx.min(axis=1),
            "emb_mean": X_emb.mean(axis=1),
            "emb_std": X_emb.std(axis=1),
            "emb_l2": np.sqrt((X_emb**2).sum(axis=1)),
        }
    )
    rs_auc = {
        c: {"auc": safe_auc(y, row_stats[c]), "auc_abs": auc_abs(y, row_stats[c])}
        for c in row_stats.columns
    }
    report["pca"] = {
        "components": pca_rows,
        "cumvar_10": float(pca.explained_variance_ratio_.sum()),
        "oof_logistic": pca_oof,
    }
    report["x_row_stats_auc"] = rs_auc
    corr = Xx.corr().abs()
    high_pairs = []
    for i, a in enumerate(xcols):
        for b in xcols[i + 1 :]:
            v = float(corr.loc[a, b])
            if v >= 0.7:
                high_pairs.append({"a": a, "b": b, "abs_corr": v})
    report["x_high_corr_pairs"] = sorted(high_pairs, key=lambda d: -d["abs_corr"])[:30]

    x_reco = []
    for _, r in x_df.iterrows():
        if r["near_unique"] and r["auc_abs"] < 0.53:
            action, reason = "DROP_or_NO_TE", "near-unique + weak univariate; TE forbidden"
        elif r["auc_abs"] >= 0.55:
            action, reason = "KEEP_raw", "standalone signal"
        elif r["resid_auc_abs"] + 0.005 < r["auc_abs"] and r["auc_abs"] >= 0.52:
            action, reason = "RESIDUALIZE", "signal largely explained by structure"
        elif r["auc_abs"] >= 0.52:
            action, reason = "KEEP_low_priority", "weak but non-null"
        else:
            action, reason = "DROP_candidate", "near-noise"
        x_reco.append(
            {
                "col": r["col"],
                "action": action,
                "reason": reason,
                "auc_abs": r["auc_abs"],
                "resid_auc_abs": r["resid_auc_abs"],
            }
        )
    report["x_recommendations"] = x_reco

    conflict = {
        "w1_eq_w2_rate": float((train["w1"] == train["w2"]).mean()),
        "w1_eq_w2_claim_rate": float(y[train["w1"] == train["w2"]].mean()),
        "w1_ne_w2_claim_rate": float(y[train["w1"] != train["w2"]].mean()),
        "t1_eq_t2_rate": float((train["t1"] == train["t2"]).mean()),
        "r1_eq_r2_rate": float((train["r1"] == train["r2"]).mean()),
        "c1_eq_c2_rate": float((train["c1"] == train["c2"]).mean()),
        "condition_missing_rate_train": float(train["condition"].isna().mean()),
        "condition_missing_claim_rate": float(y[train["condition"].isna()].mean())
        if train["condition"].isna().any()
        else None,
        "condition_present_claim_rate": float(y[train["condition"].notna()].mean()),
    }
    for name, series in [
        ("w1_eq_w2", (train["w1"] == train["w2"]).astype(int)),
        ("condition_missing", train["condition"].isna().astype(int)),
    ]:
        conflict[f"{name}_auc"] = safe_auc(y, series)
    report["conflict_missing"] = conflict

    rare = {}
    for c in ["region", "source", "t3", "version", "month", "grades", "code"]:
        vc = train[c].astype(str).value_counts()
        rare_levels = vc[vc < 50]
        rare_mask = train[c].astype(str).isin(rare_levels.index)
        rare[c] = {
            "n_levels": int(len(vc)),
            "n_rare_lt50": int(len(rare_levels)),
            "rare_row_frac": float(rare_mask.mean()),
            "rare_claim_rate": float(y[rare_mask].mean()) if rare_mask.any() else None,
            "common_claim_rate": float(y[~rare_mask].mean()) if (~rare_mask).any() else None,
            "min_count": int(vc.min()),
            "max_count": int(vc.max()),
        }
    report["rare_categories"] = rare

    d = train["days"].to_numpy()
    cond = train["condition"].fillna(train["condition"].median()).to_numpy()
    physics = {}
    for c in ["days", "condition", "cc", "V", "max_g", "livability", "age_range"]:
        s = pd.to_numeric(train[c], errors="coerce")
        physics[c] = {
            "auc": safe_auc(y, s.fillna(s.median())),
            "auc_log1p": safe_auc(y, np.log1p(s.clip(lower=0).fillna(s.median()))),
            "auc_rank": safe_auc(y, s.rank(method="average")),
        }
    physics["days*condition"] = {"auc": safe_auc(y, d * cond)}
    physics["condition/days"] = {"auc": safe_auc(y, cond / (np.abs(d) + 1))}
    physics["V*days"] = {"auc": safe_auc(y, train["V"] * d)}
    physics["cc/days"] = {"auc": safe_auc(y, train["cc"] / (np.abs(d) + 1))}
    report["numeric_physics"] = physics

    # extra diagnostics used in checklist
    ratio = cond / (np.abs(d) + 1)
    report["extra"] = {
        "cond_over_days_auc": safe_auc(y, ratio),
        "cond_over_days_auc_abs": auc_abs(y, ratio),
        "ratio_q10_oof_te": oof_te_auc(pd.qcut(ratio, 10, duplicates="drop"), y)[0],
        "days_q5_x_ratio_q5_oof_te": oof_te_auc(
            pd.qcut(train["days"], 5, duplicates="drop").astype(str)
            + "|"
            + pd.qcut(ratio, 5, duplicates="drop").astype(str),
            y,
        )[0],
        "livability_oof_te": oof_te_auc(train["livability"], y)[0],
        "livability_numeric_auc": safe_auc(y, train["livability"]),
        "car_eng_bijection": int(
            pd.DataFrame({"car": ptr["car_token"], "eng": ptr["eng_token"]}).drop_duplicates().shape[0]
        ),
        "x19_source_combos": int(train.groupby(["source", "x19"]).ngroups),
    }

    with open(OUT / "eda_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print("META", json.dumps(report["meta"]))
    print("UNI TOP\n", uni_df.head(12).to_string(index=False))
    print("CROSS2 TOP\n", cross2_df.head(10).to_string(index=False))
    print("CROSS3 TOP\n", cross3_df.head(10).to_string(index=False))
    print("EXTRA", json.dumps(report["extra"], indent=2))
    print("DONE", list(OUT.iterdir()))


if __name__ == "__main__":
    main()
