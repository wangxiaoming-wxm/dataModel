#!/usr/bin/env python3
"""B6 feature mining on NEW train.csv/test.csv only.

TE is diagnostic upper-bound only. Final B6 recommendations prefer CatBoost
native string crosses. Numbers must come from current CSVs.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path("/workspace")
OUT = ROOT / "artifacts" / "b6_eda"
OUT.mkdir(parents=True, exist_ok=True)
N_SPLITS = 5
SEED = 2026
B5_OOF_AUC = 0.69817454


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    s = np.asarray(s, dtype=float)
    mask = np.isfinite(s)
    if mask.sum() < 80 or len(np.unique(y[mask])) < 2 or np.nanstd(s[mask]) < 1e-12:
        return float("nan")
    return float(roc_auc_score(y[mask], s[mask]))


def oof_target_encode(keys: pd.Series, y: np.ndarray, n_splits: int = N_SPLITS, seed: int = SEED) -> np.ndarray:
    """Strict fold-local mean TE (global mean prior, no leakage)."""
    keys = keys.astype(str).fillna("__NA__").to_numpy()
    y = np.asarray(y)
    oof = np.full(len(y), np.nan, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = float(y.mean())
    for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
        tr_keys, va_keys = keys[tr_idx], keys[va_idx]
        tr_y = y[tr_idx]
        stats = pd.DataFrame({"k": tr_keys, "y": tr_y}).groupby("k")["y"].agg(["sum", "count"])
        # mild smoothing toward global mean
        means = (stats["sum"] + 10.0 * global_mean) / (stats["count"] + 10.0)
        mapped = pd.Series(va_keys).map(means).fillna(global_mean).to_numpy(dtype=float)
        oof[va_idx] = mapped
    return oof


def leaky_target_encode(keys: pd.Series, y: np.ndarray) -> np.ndarray:
    keys = keys.astype(str).fillna("__NA__")
    y = np.asarray(y)
    global_mean = float(y.mean())
    stats = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    means = (stats["sum"] + 10.0 * global_mean) / (stats["count"] + 10.0)
    return keys.map(means).fillna(global_mean).to_numpy(dtype=float)


def qbin(series: pd.Series, q: int, edges: np.ndarray | None = None) -> tuple[pd.Series, np.ndarray]:
    vals = pd.to_numeric(series, errors="coerce")
    if edges is None:
        qs = np.linspace(0, 1, q + 1)
        edges = np.unique(vals.dropna().quantile(qs).to_numpy(dtype=float))
        if len(edges) < 3:
            labels = pd.Series(["bin_0"] * len(vals), index=vals.index)
            return labels, edges
        cut_edges = edges[1:-1]
    else:
        cut_edges = edges
    codes = np.full(len(vals), -1, dtype=np.int16)
    finite = np.isfinite(vals.to_numpy(dtype=float))
    arr = vals.to_numpy(dtype=float)
    if len(cut_edges):
        codes[finite] = np.searchsorted(cut_edges, arr[finite], side="right").astype(np.int16)
    else:
        codes[finite] = 0
    labels = pd.Series(codes, index=vals.index).astype(str).radd("bin_")
    return labels, np.asarray(cut_edges, dtype=float)


def fixed_day_windows(days: pd.Series) -> pd.Series:
    """Business-ish fixed thresholds (not fold quantile)."""
    d = pd.to_numeric(days, errors="coerce")
    # thresholds chosen near observed decile elbows from prior EDA (~700/2500/5000/7000/9000/10000)
    bins = [-np.inf, 700, 2500, 5000, 7000, 9000, 10000, np.inf]
    labels = ["d0_700", "d700_2500", "d2500_5k", "d5k_7k", "d7k_9k", "d9k_10k", "d10k_plus"]
    return pd.cut(d, bins=bins, labels=labels, right=True).astype(str).fillna("__NA__")


def sparsity_stats(keys: pd.Series) -> dict:
    vc = keys.astype(str).fillna("__NA__").value_counts()
    n = len(keys)
    n_lt20 = int((vc < 20).sum())
    n_lt50 = int((vc < 50).sum())
    row_lt20 = float(vc[vc < 20].sum() / n)
    row_lt50 = float(vc[vc < 50].sum() / n)
    return {
        "nunique": int(vc.shape[0]),
        "mean_count": float(vc.mean()),
        "median_count": float(vc.median()),
        "min_count": int(vc.min()),
        "n_cells_lt20": n_lt20,
        "n_cells_lt50": n_lt50,
        "row_share_lt20": row_lt20,
        "row_share_lt50": row_lt50,
        "sparse_risk": bool(row_lt20 > 0.05 or vc.mean() < 40),
    }


def residualize(y: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Simple residual for ranking: y - p (clipped)."""
    p = np.clip(baseline, 1e-6, 1 - 1e-6)
    return y.astype(float) - p


def parse_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    src = out["source"].astype(str)
    out["car"] = src.str.extract(r"(CAR_\d+)", expand=False).fillna("__NA__")
    out["eng"] = src.str.extract(r"(ENG_\d+)", expand=False).fillna("__NA__")
    t3 = out["t3"].astype(str)
    parsed = t3.str.extract(r"^(-?\d+(?:\.\d+)?)([A-Za-z])$")
    out["t3_num"] = pd.to_numeric(parsed[0], errors="coerce")
    out["t3_sfx"] = parsed[1].fillna("__NA__")
    out["x19_cat"] = out["x19"].astype(str)
    out["x20_cat"] = out["x20"].astype(str)
    out["age_coarse"] = out["age_range"].clip(upper=8).astype(str)
    out["age_raw"] = out["age_range"].astype(str)
    out["w_pair"] = out["w1"].astype(str) + "_" + out["w2"].astype(str)
    out["t_pair"] = out["t1"].astype(str) + "_" + out["t2"].astype(str)
    out["c_pair"] = out["c1"].astype(str) + "_" + out["c2"].astype(str)
    out["r_pair"] = out["r1"].astype(str) + "_" + out["r2"].astype(str)
    days = pd.to_numeric(out["days"], errors="coerce")
    cond = pd.to_numeric(out["condition"], errors="coerce")
    out["cond_over_days"] = cond / (days.abs() + 1.0)
    out["days_x_cond"] = days * cond
    out["log_days"] = np.log1p(days.clip(lower=0))
    out["log_cond"] = np.log1p(cond.clip(lower=0))
    # ratio bins / residualized condition intensity
    out["cond_per_1k_days"] = cond / (days.abs() / 1000.0 + 1e-6)
    out["days_fixed"] = fixed_day_windows(days)
    out["liv_str"] = out["livability"].round(3).astype(str)
    out["version"] = out["version"].astype(str)
    out["code"] = out["code"].astype(str)
    out["grades"] = out["grades"].astype(str)
    out["region"] = out["region"].astype(str)
    out["source"] = out["source"].astype(str)
    out["month"] = out["month"].astype(str)
    return out


def eval_cross(name: str, keys: pd.Series, y: np.ndarray, b5_oof: np.ndarray | None = None) -> dict:
    stats = sparsity_stats(keys)
    oof_te = oof_target_encode(keys, y)
    leaky_te = leaky_target_encode(keys, y)
    auc_oof = safe_auc(y, oof_te)
    auc_leaky = safe_auc(y, leaky_te)
    row = {
        "cross": name,
        "auc_oof_te": auc_oof,
        "auc_leaky_te": auc_leaky,
        "gap_leaky_minus_oof": (auc_leaky - auc_oof) if np.isfinite(auc_leaky) and np.isfinite(auc_oof) else np.nan,
        **stats,
    }
    if b5_oof is not None:
        # correlation of TE score with B5 OOF (heterogeneity proxy)
        a = oof_te
        b = b5_oof
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 100:
            row["corr_with_b5_oof"] = float(np.corrcoef(a[m], b[m])[0, 1])
        else:
            row["corr_with_b5_oof"] = float("nan")
        # residual ranking: does TE separate residual risk?
        resid = residualize(y, b5_oof)
        # AUC of TE vs residual sign is awkward; use corr(|resid|) via spearman-ish:
        # instead: partial signal = AUC of TE on hard subset where B5 is near 0.5
        hard = (b5_oof > 0.08) & (b5_oof < 0.25)
        if hard.sum() > 200 and len(np.unique(y[hard])) > 1:
            row["auc_oof_te_on_b5_midband"] = safe_auc(y[hard], a[hard])
        else:
            row["auc_oof_te_on_b5_midband"] = float("nan")
        # residual AUC: treat residual as continuous target via ranking of |error|
        # Use logistic residual correlation
        row["corr_te_vs_residual"] = float(np.corrcoef(a[m], resid[m])[0, 1]) if m.sum() > 100 else float("nan")
    return row


def numeric_auc_table(df: pd.DataFrame, y: np.ndarray, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        a = safe_auc(y, s)
        a_abs = max(a, 1.0 - a) if np.isfinite(a) else float("nan")
        rows.append(
            {
                "feature": c,
                "auc": a,
                "auc_abs": a_abs,
                "nunique": int(pd.Series(s).nunique(dropna=True)),
                "missing": float(np.isnan(s).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("auc_abs", ascending=False)


def main() -> None:
    tr = pd.read_csv(ROOT / "train.csv")
    te = pd.read_csv(ROOT / "test.csv")
    y = tr["label"].to_numpy()
    b5_pred = np.load(ROOT / "artifacts" / "b5_8seed" / "predictions.npz")
    b5_oof = b5_pred["oof"]
    assert len(b5_oof) == len(tr)

    trp = parse_frame(tr)
    tep = parse_frame(te)

    # fold-local quantile bins fitted on full train for diagnostic (TE itself is OOF)
    days_q5, _ = qbin(trp["days"], 5)
    days_q10, _ = qbin(trp["days"], 10)
    cond_q5, _ = qbin(trp["condition"], 5)
    cond_q10, _ = qbin(trp["condition"], 10)
    ratio_q5, _ = qbin(trp["cond_over_days"], 5)
    ratio_q10, _ = qbin(trp["cond_over_days"], 10)
    ratio_1k_q5, _ = qbin(trp["cond_per_1k_days"], 5)
    t3n_q5, _ = qbin(trp["t3_num"], 5)
    liv_q5, _ = qbin(trp["livability"], 5)

    # B5-style baselines already consumed
    b5_like = {
        "B5:days_q10": days_q10,
        "B5:days_q5×region": days_q5.astype(str) + "|" + trp["region"],
        "B5:days_q5×source": days_q5.astype(str) + "|" + trp["source"],
        "B5:days_q5×x19": days_q5.astype(str) + "|" + trp["x19_cat"],
        "B5:days_q5×x20": days_q5.astype(str) + "|" + trp["x20_cat"],
        "B5:days_q5×age_raw": days_q5.astype(str) + "|" + trp["age_raw"],
        "B5:days_q5×cond_q5": days_q5.astype(str) + "|" + cond_q5.astype(str),
        "B5:region×source×version(dual3-ish)": trp["region"] + "|" + trp["source"] + "|" + trp["version"],
    }

    # Candidate NEW signals for B6 (not fully in B5 recipe)
    candidates = {
        # ratio / condition intensity (B5 has numeric ratio but not cat crosses)
        "ratio_q5": ratio_q5,
        "ratio_q10": ratio_q10,
        "ratio_q5×region": ratio_q5.astype(str) + "|" + trp["region"],
        "ratio_q5×source": ratio_q5.astype(str) + "|" + trp["source"],
        "ratio_q5×t3_sfx": ratio_q5.astype(str) + "|" + trp["t3_sfx"],
        "ratio_q5×code": ratio_q5.astype(str) + "|" + trp["code"],
        "ratio_q5×w_pair": ratio_q5.astype(str) + "|" + trp["w_pair"],
        "ratio_1k_q5×region": ratio_1k_q5.astype(str) + "|" + trp["region"],
        # t3_sfx × code × days
        "t3_sfx": trp["t3_sfx"],
        "t3_sfx×code": trp["t3_sfx"] + "|" + trp["code"],
        "t3_sfx×code×days_q5": trp["t3_sfx"] + "|" + trp["code"] + "|" + days_q5.astype(str),
        "t3_sfx×code×days_fixed": trp["t3_sfx"] + "|" + trp["code"] + "|" + trp["days_fixed"],
        "t3_sfx×days_q5": trp["t3_sfx"] + "|" + days_q5.astype(str),
        "t3_sfx×days_q10": trp["t3_sfx"] + "|" + days_q10.astype(str),
        "t3_sfx×region×days_q5": trp["t3_sfx"] + "|" + trp["region"] + "|" + days_q5.astype(str),
        "t3_num_q5×sfx": t3n_q5.astype(str) + "|" + trp["t3_sfx"],
        "t3_full×days_q5": trp["t3"].astype(str) + "|" + days_q5.astype(str),
        # age_coarse
        "age_coarse": trp["age_coarse"],
        "age_coarse×days_q5": trp["age_coarse"] + "|" + days_q5.astype(str),
        "age_coarse×days_q10": trp["age_coarse"] + "|" + days_q10.astype(str),
        "age_coarse×days_fixed": trp["age_coarse"] + "|" + trp["days_fixed"],
        "age_coarse×region": trp["age_coarse"] + "|" + trp["region"],
        "age_coarse×source": trp["age_coarse"] + "|" + trp["source"],
        "age_coarse×ratio_q5": trp["age_coarse"] + "|" + ratio_q5.astype(str),
        "age_raw×days_q5": trp["age_raw"] + "|" + days_q5.astype(str),
        # w_pair
        "w_pair": trp["w_pair"],
        "w_pair×days_q5": trp["w_pair"] + "|" + days_q5.astype(str),
        "w_pair×days_q10": trp["w_pair"] + "|" + days_q10.astype(str),
        "w_pair×days_fixed": trp["w_pair"] + "|" + trp["days_fixed"],
        "w_pair×region": trp["w_pair"] + "|" + trp["region"],
        "w_pair×source": trp["w_pair"] + "|" + trp["source"],
        "w_pair×ratio_q5": trp["w_pair"] + "|" + ratio_q5.astype(str),
        "w_pair×t3_sfx": trp["w_pair"] + "|" + trp["t3_sfx"],
        # fixed day windows
        "days_fixed": trp["days_fixed"],
        "days_fixed×region": trp["days_fixed"] + "|" + trp["region"],
        "days_fixed×source": trp["days_fixed"] + "|" + trp["source"],
        "days_fixed×cond_q5": trp["days_fixed"] + "|" + cond_q5.astype(str),
        "days_fixed×code": trp["days_fixed"] + "|" + trp["code"],
        "days_fixed×t3_sfx": trp["days_fixed"] + "|" + trp["t3_sfx"],
        "days_fixed×version": trp["days_fixed"] + "|" + trp["version"],
        # other clause pairs
        "t_pair×days_q5": trp["t_pair"] + "|" + days_q5.astype(str),
        "c_pair×days_q5": trp["c_pair"] + "|" + days_q5.astype(str),
        "r_pair×days_q5": trp["r_pair"] + "|" + days_q5.astype(str),
        # code / car / version denser with days (version not in B5 days crosses)
        "code×days_q5": trp["code"] + "|" + days_q5.astype(str),
        "car×days_q5": trp["car"] + "|" + days_q5.astype(str),
        "version×days_q5": trp["version"] + "|" + days_q5.astype(str),
        "version×days_fixed": trp["version"] + "|" + trp["days_fixed"],
        "liv_q5×days_q5": liv_q5.astype(str) + "|" + days_q5.astype(str),
        "cond_q5×region": cond_q5.astype(str) + "|" + trp["region"],
        "cond_q5×source": cond_q5.astype(str) + "|" + trp["source"],
        "cond_q10×source": cond_q10.astype(str) + "|" + trp["source"],
        # known overfit suspects
        "t3_full×code": trp["t3"].astype(str) + "|" + trp["code"],
        "car×version": trp["car"] + "|" + trp["version"],
        "source×version": trp["source"] + "|" + trp["version"],
        "region×days_q5×version": trp["region"] + "|" + days_q5.astype(str) + "|" + trp["version"],
        "car×version×days_q5": trp["car"] + "|" + trp["version"] + "|" + days_q5.astype(str),
        "region×car×version": trp["region"] + "|" + trp["car"] + "|" + trp["version"],
        "age_raw×version": trp["age_raw"] + "|" + trp["version"],
        "month×version": trp["month"] + "|" + trp["version"],
        "t3_full×code×days_q5": trp["t3"].astype(str) + "|" + trp["code"] + "|" + days_q5.astype(str),
        "region×liv_str": trp["region"] + "|" + trp["liv_str"],
    }

    rows = []
    for name, keys in {**b5_like, **candidates}.items():
        row = eval_cross(name, keys, y, b5_oof=b5_oof)
        row["group"] = "b5_like" if name.startswith("B5:") else "candidate"
        rows.append(row)
    cross_df = pd.DataFrame(rows).sort_values("auc_oof_te", ascending=False)
    cross_df.to_csv(OUT / "cross_oof_te_upperbound.csv", index=False)

    # numeric features: residual / ratio variants
    num_cols = [
        "days",
        "condition",
        "cond_over_days",
        "cond_per_1k_days",
        "days_x_cond",
        "log_days",
        "log_cond",
        "t3_num",
        "livability",
        "cc",
        "V",
        "max_g",
        "x19",
        "x20",
        "age_range",
        "w1",
        "w2",
    ]
    num_auc = numeric_auc_table(trp, y, num_cols)
    # residual correlation vs B5 OOF
    resid = residualize(y, b5_oof)
    extra = []
    for c in num_cols:
        s = pd.to_numeric(trp[c], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(s) & np.isfinite(b5_oof)
        corr_b5 = float(np.corrcoef(s[m], b5_oof[m])[0, 1]) if m.sum() > 100 else float("nan")
        corr_r = float(np.corrcoef(s[m], resid[m])[0, 1]) if m.sum() > 100 else float("nan")
        extra.append({"feature": c, "corr_with_b5_oof": corr_b5, "corr_with_residual": corr_r})
    num_auc = num_auc.merge(pd.DataFrame(extra), on="feature")
    num_auc.to_csv(OUT / "numeric_auc_vs_b5.csv", index=False)

    # claim rate tables for key new features
    def rate_table(keys: pd.Series, name: str, min_n: int = 30) -> pd.DataFrame:
        d = pd.DataFrame({"k": keys.astype(str), "y": y})
        g = d.groupby("k")["y"].agg(["count", "sum", "mean"]).reset_index()
        g = g.rename(columns={"sum": "claims", "mean": "claim_rate"})
        g["feature"] = name
        g = g[g["count"] >= min_n].sort_values("claim_rate", ascending=False)
        return g

    rate_parts = [
        rate_table(trp["days_fixed"], "days_fixed", 50),
        rate_table(trp["w_pair"], "w_pair", 20),
        rate_table(trp["age_coarse"], "age_coarse", 20),
        rate_table(trp["t3_sfx"] + "|" + trp["code"], "t3_sfx×code", 20),
        rate_table(trp["t3_sfx"] + "|" + trp["code"] + "|" + days_q5.astype(str), "t3_sfx×code×days_q5", 40),
        rate_table(ratio_q5, "ratio_q5", 50),
        rate_table(trp["days_fixed"] + "|" + trp["region"], "days_fixed×region", 40),
        rate_table(trp["w_pair"] + "|" + days_q5.astype(str), "w_pair×days_q5", 40),
        rate_table(trp["age_coarse"] + "|" + days_q5.astype(str), "age_coarse×days_q5", 40),
    ]
    rates = pd.concat(rate_parts, ignore_index=True)
    rates.to_csv(OUT / "claim_rate_slices.csv", index=False)

    # prioritize healthy high-AUC candidates (not sparse, gap small, not already B5)
    cand = cross_df[cross_df["group"] == "candidate"].copy()
    healthy = cand[
        (cand["sparse_risk"] == False)
        & (cand["mean_count"] >= 50)
        & (cand["row_share_lt20"] <= 0.05)
        & (cand["auc_oof_te"] >= 0.55)
    ].sort_values(["auc_oof_te", "mean_count"], ascending=[False, False])
    healthy.to_csv(OUT / "healthy_high_upperbound.csv", index=False)

    # low corr with B5 among healthy (heterogeneous arms)
    hetero = healthy.copy()
    hetero = hetero.sort_values("corr_with_b5_oof", ascending=True)
    hetero.to_csv(OUT / "hetero_candidates_by_b5_corr.csv", index=False)

    # overfit list: large gap or severe sparsity
    overfit = cand[
        (cand["gap_leaky_minus_oof"] >= 0.04)
        | (cand["row_share_lt20"] >= 0.05)
        | ((cand["mean_count"] < 40) & (cand["nunique"] >= 80))
    ].sort_values("gap_leaky_minus_oof", ascending=False)
    overfit.to_csv(OUT / "overfit_cross_blacklist.csv", index=False)

    # B5 gap analysis: candidates that beat nearest B5 baseline on same axis
    b5_days_region = float(cross_df.loc[cross_df["cross"] == "B5:days_q5×region", "auc_oof_te"].iloc[0])
    b5_days_source = float(cross_df.loc[cross_df["cross"] == "B5:days_q5×source", "auc_oof_te"].iloc[0])
    b5_dc = float(cross_df.loc[cross_df["cross"] == "B5:days_q5×cond_q5", "auc_oof_te"].iloc[0])
    b5_days = float(cross_df.loc[cross_df["cross"] == "B5:days_q10", "auc_oof_te"].iloc[0])

    # train/test coverage for key cats
    coverage_rows = []
    for col in ["t3_sfx", "code", "w_pair", "age_coarse", "days_fixed", "car", "version"]:
        tr_set = set(trp[col].astype(str).unique())
        te_set = set(tep[col].astype(str).unique())
        coverage_rows.append(
            {
                "feature": col,
                "train_nunique": len(tr_set),
                "test_nunique": len(te_set),
                "overlap": len(tr_set & te_set),
                "test_only": len(te_set - tr_set),
                "train_only": len(tr_set - te_set),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT / "train_test_coverage.csv", index=False)

    # summary JSON for report generation
    summary = {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "claim_rate": float(y.mean()),
        "b5_frozen_oof_auc": B5_OOF_AUC,
        "b5_oof_auc_recomputed": float(safe_auc(y, b5_oof)),
        "b5_baselines": {
            "days_q10": b5_days,
            "days_q5×region": b5_days_region,
            "days_q5×source": b5_days_source,
            "days_q5×cond_q5": b5_dc,
        },
        "top_healthy_candidates": healthy.head(20).to_dict(orient="records"),
        "top_hetero_low_b5_corr": hetero.head(15).to_dict(orient="records"),
        "top_overfit": overfit.head(20).to_dict(orient="records"),
        "numeric_top": num_auc.head(12).to_dict(orient="records"),
        "coverage": coverage.to_dict(orient="records"),
        "days_fixed_edges": [-np.inf, 700, 2500, 5000, 7000, 9000, 10000, np.inf],
        "notes": {
            "te_role": "diagnostic upper bound only; prefer CatBoost native crosses",
            "protocol": "5-fold stratified OOF TE with prior strength 10",
        },
    }
    with (OUT / "mining_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("Wrote artifacts to", OUT)
    print("B5 OOF recomputed:", summary["b5_oof_auc_recomputed"])
    print("Top healthy:")
    print(healthy[["cross", "auc_oof_te", "mean_count", "corr_with_b5_oof", "gap_leaky_minus_oof"]].head(15).to_string(index=False))
    print("\nTop overfit:")
    print(overfit[["cross", "auc_oof_te", "gap_leaky_minus_oof", "mean_count", "row_share_lt20"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
