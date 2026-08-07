#!/usr/bin/env python3
"""Business-driven feature synergy audit on the NEW train.csv.

Validates claim-rate spreads, odds ratios, days stratified risk, semantic
crosses, and leaky vs OOF target-encoding AUC gaps. Numbers are computed
from /workspace/train.csv only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

DATA = Path(__file__).resolve().parents[1] / "train.csv"
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "business_feature_synergy"


def odds_ratio(rate: float, base: float) -> float:
    rate = float(np.clip(rate, 1e-9, 1 - 1e-9))
    base = float(np.clip(base, 1e-9, 1 - 1e-9))
    return (rate / (1 - rate)) / (base / (1 - base))


def claim_table(series: pd.Series, y: pd.Series, base: float) -> pd.DataFrame:
    g = (
        pd.DataFrame({"k": series, "y": y})
        .groupby("k", dropna=False)["y"]
        .agg(n="count", claims="sum", rate="mean")
    )
    g["or_vs_base"] = g["rate"].map(lambda r: odds_ratio(r, base))
    return g.sort_values("rate", ascending=False)


def cramer_v(series: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(series, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan")
    chi2, _, _, _ = stats.chi2_contingency(ct)
    n = ct.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def te_auc_pair(key: np.ndarray, y: np.ndarray, seed: int = 2026) -> dict:
    """Compare leaky global TE AUC vs honest 5-fold OOF TE AUC."""
    gmean = pd.Series(y).groupby(key).mean()
    global_te = np.array([gmean[k] for k in key], dtype=float)
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, va in skf.split(np.zeros(len(y)), y):
        m = pd.Series(y[tr]).groupby(key[tr]).mean()
        prior = float(y[tr].mean())
        oof[va] = [m.get(k, prior) for k in key[va]]
    vc = pd.Series(key).value_counts()
    return {
        "cells": int(len(vc)),
        "n_lt_20": int((vc < 20).sum()),
        "row_share_n_lt_20": float(vc[vc < 20].sum() / len(y)),
        "leaky_te_auc": float(roc_auc_score(y, global_te)),
        "oof_te_auc": float(roc_auc_score(y, oof)),
        "auc_gap": float(roc_auc_score(y, global_te) - roc_auc_score(y, oof)),
    }


def main() -> int:
    df = pd.read_csv(DATA)
    y = df["label"].astype(int)
    base = float(y.mean())
    n = len(df)

    df["t3_num"] = df["t3"].astype(str).str.extract(r"([-+]?\d+(?:\.\d+)?)")[0].astype(float)
    df["t3_sfx"] = df["t3"].astype(str).str.extract(r"([A-Za-z]+)$")[0].fillna("NA")
    df["car"] = df["source"].astype(str).str.extract(r"(CAR_\d+)")[0]
    df["eng"] = df["source"].astype(str).str.extract(r"(ENG_\d+)")[0]
    df["days_bin5"] = pd.qcut(df["days"], 5, labels=["D1", "D2", "D3", "D4", "D5"], duplicates="drop").astype(str)
    df["days_bin10"] = pd.qcut(df["days"], 10, duplicates="drop").astype(str)
    df["cond_bin5"] = pd.qcut(
        df["condition"].fillna(df["condition"].median()),
        5,
        labels=["C1", "C2", "C3", "C4", "C5"],
        duplicates="drop",
    ).astype(str)
    df["live_bin"] = pd.qcut(df["livability"], 5, duplicates="drop").astype(str)
    df["w_pair"] = df["w1"].astype(str) + "_" + df["w2"].astype(str)
    df["age_coarse"] = pd.cut(
        df["age_range"], bins=[0, 1, 3, 5, 7, 10], labels=["a1", "a2-3", "a4-5", "a6-7", "a8+"]
    ).astype(str)

    # Univariate highlights
    days_tercile = pd.qcut(df["days"], 3, labels=["low", "mid", "high"])
    days_rates = {
        str(lvl): {
            "n": int((days_tercile == lvl).sum()),
            "rate": float(y[days_tercile == lvl].mean()),
            "or_vs_base": odds_ratio(y[days_tercile == lvl].mean(), base),
        }
        for lvl in ["low", "mid", "high"]
    }

    cond_dec = pd.qcut(df["condition"].fillna(df["condition"].median()), 10, labels=False)
    condition_extremes = {
        "lowest_decile": {
            "rate": float(y[cond_dec == 0].mean()),
            "or_vs_base": odds_ratio(y[cond_dec == 0].mean(), base),
        },
        "highest_decile": {
            "rate": float(y[cond_dec == 9].mean()),
            "or_vs_base": odds_ratio(y[cond_dec == 9].mean(), base),
        },
    }

    region_tbl = claim_table(df["region"], y, base)
    car_tbl = claim_table(df["car"], y, base)
    version_tbl = claim_table(df["version"], y, base)
    age_tbl = claim_table(df["age_range"], y, base)

    # Region explains livability
    oh = OneHotEncoder(sparse_output=False)
    x_reg = oh.fit_transform(df[["region"]])
    live_r2 = float(LinearRegression().fit(x_reg, df["livability"]).score(x_reg, df["livability"]))

    # Days slope heterogeneity by region
    days_slope = []
    for reg, sub in df.groupby("region"):
        if len(sub) < 100:
            continue
        med = sub["days"].median()
        low = float(sub.loc[sub["days"] <= med, "label"].mean())
        high = float(sub.loc[sub["days"] > med, "label"].mean())
        days_slope.append(
            {
                "region": reg,
                "n": int(len(sub)),
                "low_rate": low,
                "high_rate": high,
                "or_high_vs_low": odds_ratio(high, low) if low not in (0, 1) else None,
            }
        )

    # Semantic crosses
    crosses = {
        "region×days_bin5": (df["region"].astype(str) + "|" + df["days_bin5"]).to_numpy(),
        "region×days_bin10": (df["region"].astype(str) + "|" + df["days_bin10"]).to_numpy(),
        "days_bin5×condition_bin5": (df["days_bin5"] + "|" + df["cond_bin5"]).to_numpy(),
        "car×version": (df["car"] + "|" + df["version"]).to_numpy(),
        "source×version": (df["source"].astype(str) + "|" + df["version"]).to_numpy(),
        "t3×code": (df["t3"].astype(str) + "|" + df["code"].astype(str)).to_numpy(),
        "t3_sfx×code": (df["t3_sfx"] + "|" + df["code"].astype(str)).to_numpy(),
        "source×days_bin5": (df["source"].astype(str) + "|" + df["days_bin5"]).to_numpy(),
        "w_pair×days_bin5": (df["w_pair"] + "|" + df["days_bin5"]).to_numpy(),
        "age_coarse×days_bin5": (df["age_coarse"] + "|" + df["days_bin5"]).to_numpy(),
        "region×car": (df["region"].astype(str) + "|" + df["car"]).to_numpy(),
        "days_bin5×version": (df["days_bin5"] + "|" + df["version"]).to_numpy(),
        "t3_sfx×code×days_bin5": (
            df["t3_sfx"] + "|" + df["code"].astype(str) + "|" + df["days_bin5"]
        ).to_numpy(),
        "region×days5×version": (
            df["region"].astype(str) + "|" + df["days_bin5"] + "|" + df["version"]
        ).to_numpy(),
        "car×version×days5": (
            df["car"] + "|" + df["version"] + "|" + df["days_bin5"]
        ).to_numpy(),
    }

    cross_report = {}
    for name, key in crosses.items():
        tbl = claim_table(pd.Series(key), y, base)
        big = tbl[tbl["n"] >= 50]
        cross_report[name] = {
            "cramer_v": cramer_v(pd.Series(key), y),
            "cells": int(len(tbl)),
            "n_lt_20": int((tbl["n"] < 20).sum()),
            "n_lt_50": int((tbl["n"] < 50).sum()),
            "mass_n_ge_50": float(tbl.loc[tbl["n"] >= 50, "n"].sum() / n),
            "rate_span_n_ge_50": float(big["rate"].max() - big["rate"].min()) if len(big) else None,
            "te_auc": te_auc_pair(key, y.to_numpy()),
            "top_n50": big.head(5)[["n", "claims", "rate", "or_vs_base"]].reset_index().to_dict("records"),
            "bot_n50": big.sort_values("rate").head(5)[["n", "claims", "rate", "or_vs_base"]].reset_index().to_dict("records"),
        }

    numeric_corr = (
        df.select_dtypes(include=[np.number])
        .drop(columns=["label"])
        .corrwith(y)
        .abs()
        .sort_values(ascending=False)
        .head(15)
    )

    payload = {
        "data": {"path": str(DATA), "n": n, "claim_rate": base, "positives": int(y.sum())},
        "field_relationships": {
            "corr_cc_V": float(df["cc"].corr(df["V"])),
            "corr_V_max_g": float(df["V"].corr(df["max_g"])),
            "corr_x19_V": float(df["x19"].corr(df["V"])),
            "corr_x20_condition": float(df["x20"].corr(df["condition"])),
            "corr_days_condition": float(df["days"].corr(df["condition"])),
            "livability_R2_on_region": live_r2,
            "code_to_car": {
                str(c): list(pd.crosstab(df["code"], df["car"]).columns[pd.crosstab(df["code"], df["car"]).loc[c] > 0])
                for c in sorted(df["code"].unique())
            },
        },
        "univariate": {
            "days_tercile": days_rates,
            "days_cramer_v_decile": cramer_v(pd.qcut(df["days"], 10, duplicates="drop").astype(str), y),
            "condition_extremes": condition_extremes,
            "condition_cramer_v_decile": cramer_v(
                pd.qcut(df["condition"].fillna(df["condition"].median()), 10, duplicates="drop").astype(str), y
            ),
            "region_extremes_n_ge_30": {
                "high": region_tbl[region_tbl["n"] >= 30].head(5).reset_index().to_dict("records"),
                "low": region_tbl[region_tbl["n"] >= 30].sort_values("rate").head(5).reset_index().to_dict("records"),
            },
            "car": car_tbl.reset_index().to_dict("records"),
            "version_extremes": {
                "high": version_tbl.head(5).reset_index().to_dict("records"),
                "low": version_tbl.sort_values("rate").head(5).reset_index().to_dict("records"),
            },
            "age_range": age_tbl.reset_index().to_dict("records"),
            "w_pair": claim_table(df["w_pair"], y, base).reset_index().to_dict("records"),
            "abs_corr_top15": numeric_corr.round(6).to_dict(),
        },
        "days_slope_by_region": sorted(days_slope, key=lambda r: -(r["or_high_vs_low"] or 0)),
        "crosses": cross_report,
        "recommended_crosses_priority": [
            "region×days_bin5",
            "days_bin5×condition_bin5",
            "source×days_bin5",
            "t3_sfx×code×days_bin5",
            "w_pair×days_bin5",
            "age_coarse×days_bin5",
            "days_bin5×version",
            "region×car",
            "car×version  # CatBoost cat only; avoid crude TE",
            "t3_sfx×code",
            "live_bin×days_bin5  # livability almost region attribute",
            "region×condition_bin5",
        ],
        "overfit_te_crosses": [
            "region×days5×version",
            "car×version×days5",
            "t3×code",
            "source×version / car×version (as TE)",
            "region×days_bin10",
            "age_range×version",
            "month×version",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "audit_numbers.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Compact markdown summary for humans
    lines = [
        "# Business Feature Synergy Audit (numbers)",
        "",
        f"- n={n}, claim_rate={base:.6f}, positives={int(y.sum())}",
        f"- days tercile OR: low={days_rates['low']['or_vs_base']:.3f}, mid={days_rates['mid']['or_vs_base']:.3f}, high={days_rates['high']['or_vs_base']:.3f}",
        f"- condition lowest/highest decile OR: {condition_extremes['lowest_decile']['or_vs_base']:.3f} / {condition_extremes['highest_decile']['or_vs_base']:.3f}",
        f"- livability R² on region: {live_r2:.4f}",
        f"- corr(cc,V)={payload['field_relationships']['corr_cc_V']:.3f}, corr(x19,V)={payload['field_relationships']['corr_x19_V']:.3f}",
        "",
        "## Cross TE leakage gaps (leaky − OOF)",
        "",
    ]
    for name, info in sorted(cross_report.items(), key=lambda kv: -kv[1]["te_auc"]["auc_gap"]):
        te = info["te_auc"]
        lines.append(
            f"- **{name}**: leaky={te['leaky_te_auc']:.4f}, oof={te['oof_te_auc']:.4f}, "
            f"gap={te['auc_gap']:.4f}, cells={te['cells']}, n<20={te['n_lt_20']} "
            f"(row_share={te['row_share_n_lt_20']:.3f}), CramerV={info['cramer_v']:.4f}"
        )
    (OUT / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out_json), "claim_rate": base, "n_crosses": len(cross_report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
