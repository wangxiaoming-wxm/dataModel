"""Mine residuals of B7 max(B6, plus) for new lift toward 0.71."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

OUT = Path("artifacts/b7_eda")
OUT.mkdir(parents=True, exist_ok=True)


def oof_te(y, keys, n_splits=5, seed=2026, alpha=10.0):
    oof = np.zeros(len(y), dtype=float)
    keys = pd.Series(keys).astype(str).fillna("__NA__").to_numpy()
    y = np.asarray(y)
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(keys, y):
        prior = float(y[tr].mean())
        s = pd.Series(y[tr]).groupby(keys[tr]).sum()
        c = pd.Series(y[tr]).groupby(keys[tr]).count()
        te = (s + prior * alpha) / (c + alpha)
        oof[va] = pd.Series(keys[va]).map(te).fillna(prior).to_numpy()
    return float(roc_auc_score(y, oof)), oof


def main():
    train = pd.read_csv("train.csv")
    y = train["label"].astype(int)
    b6 = np.load("artifacts/b6_gapbag_8seed/predictions.npz")
    plus = np.load("reference/v10/oof_plus_h2_10.npz")["oof"]
    eq = 0.5 * (b6["oof_gap"] + b6["oof_gap_bag"])
    fused = np.maximum(eq, plus)
    residual = y.to_numpy() - fused  # positive => under-predicted claims

    # midband where fusion is uncertain
    lo, hi = np.quantile(fused, [0.4, 0.9])
    mid = (fused >= lo) & (fused <= hi)

    days = pd.to_numeric(train["days"], errors="coerce")
    cond = pd.to_numeric(train["condition"], errors="coerce")
    ratio = cond / (days.abs() + 1.0)
    t3 = train["t3"].astype(str)
    t3_sfx = t3.str.extract(r"([A-Za-z]+)$")[0].fillna("__N__")
    w_pair = train["w1"].astype(str) + "_" + train["w2"].astype(str)
    age = pd.to_numeric(train["age_range"], errors="coerce").clip(upper=8).fillna(-1).astype(int).astype(str)
    code = train["code"].astype(str)
    region = train["region"].astype(str)
    source = train["source"].astype(str)
    version = train["version"].astype(str)
    car = source.str.extract(r"(CAR_\d+)")[0].fillna("__NA__")
    # fixed days windows (B6 mining)
    edges = np.array([-np.inf, 700, 2500, 5000, 7000, 9000, 10000, np.inf])
    dfix = pd.cut(days, bins=edges, labels=[f"d{i}" for i in range(7)]).astype(str)
    # qbins on train full for diagnostic only (TE still OOF)
    d5 = pd.qcut(days, 5, duplicates="drop").astype(str)
    r5 = pd.qcut(ratio, 5, duplicates="drop").astype(str)
    c5 = pd.qcut(cond, 5, duplicates="drop").astype(str)

    # who wins: where plus > b6 vs where b6 > plus
    plus_wins = plus > eq
    rows = []
    candidates = {
        "ratio5_region": r5 + "|" + region,
        "ratio5_source": r5 + "|" + source,
        "t3sfx_code_d5": t3_sfx + "|" + code + "|" + d5,
        "w_pair_d5": w_pair + "|" + d5,
        "w_pair_r5": w_pair + "|" + r5,
        "dfix_cond5": dfix + "|" + c5,
        "dfix_source": dfix + "|" + source,
        "car_d5": car + "|" + d5,
        "code_d5": code + "|" + d5,
        "age_r5": age + "|" + r5,
        "version_d5": version + "|" + d5,
        "cond5_source": c5 + "|" + source,
        "t3sfx_dfix_code": t3_sfx + "|" + dfix + "|" + code,
        "region_car_d5": region + "|" + car + "|" + d5,
        "w_pair_t3sfx_d5": w_pair + "|" + t3_sfx + "|" + d5,
        "liv_d5": train["livability"].astype(str) + "|" + d5,
        "grades_d5": train["grades"].astype(str) + "|" + d5,
        "x20bin_region": (pd.qcut(pd.to_numeric(train["x20"], errors="coerce"), 10, duplicates="drop").astype(str) + "|" + region),
        "x_rowmean_d5": (
            pd.qcut(train[[f"x{i}" for i in range(18)]].astype(float).mean(axis=1), 5, duplicates="drop").astype(str)
            + "|"
            + d5
        ),
    }
    for name, key in candidates.items():
        auc, oof = oof_te(y, key)
        corr_f = float(np.corrcoef(oof, fused)[0, 1])
        corr_r = float(np.corrcoef(oof, residual)[0, 1]) if np.std(residual) > 0 else 0.0
        # midband AUC
        if mid.sum() > 100 and y[mid].nunique() > 1:
            auc_mid = float(roc_auc_score(y[mid], oof[mid]))
        else:
            auc_mid = float("nan")
        # sparse
        vc = pd.Series(key).value_counts()
        mean_count = float(vc.mean())
        row_lt20 = float((pd.Series(key).map(vc) < 20).mean())
        rows.append(
            {
                "cross": name,
                "auc_oof_te": auc,
                "auc_midband": auc_mid,
                "corr_fused": corr_f,
                "corr_residual": corr_r,
                "mean_count": mean_count,
                "row_lt20": row_lt20,
                "nunique": int(vc.size),
            }
        )
        print(f"{name}: te={auc:.4f} mid={auc_mid:.4f} corr_f={corr_f:.3f} corr_r={corr_r:.3f} mc={mean_count:.1f}", flush=True)

    df = pd.DataFrame(rows).sort_values(["corr_residual", "auc_oof_te"], ascending=False)
    df.to_csv(OUT / "residual_candidates.csv", index=False)

    # slice claim rates where fusion fails (high residual magnitude among claims)
    train = train.copy()
    train["fused"] = fused
    train["plus_wins"] = plus_wins.astype(int)
    train["resid"] = residual
    fail_pos = (y == 1) & (fused < 0.15)
    fail_neg = (y == 0) & (fused > 0.25)
    summary = {
        "fused_auc": float(roc_auc_score(y, fused)),
        "n_fail_pos_low_score": int(fail_pos.sum()),
        "n_fail_neg_high_score": int(fail_neg.sum()),
        "plus_win_rate": float(plus_wins.mean()),
        "top_residual_corr": df.head(12).to_dict(orient="records"),
        "healthy_high_te": df[(df.auc_oof_te >= 0.55) & (df.row_lt20 <= 0.05) & (df.mean_count >= 50)]
        .head(15)
        .to_dict(orient="records"),
    }
    (OUT / "mining_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps({"fused_auc": summary["fused_auc"], "top": summary["top_residual_corr"][:5]}, indent=2))


if __name__ == "__main__":
    main()
