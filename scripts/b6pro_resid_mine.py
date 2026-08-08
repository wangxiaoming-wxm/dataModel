#!/usr/bin/env python3
"""Mine residual signal vs best max(main, plus) OOF — diagnostic only / candidate crosses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = 0.715


def oof_te(keys: pd.Series, y: np.ndarray, n_splits=5, prior=10.0):
    oof = np.zeros(len(y), dtype=float)
    global_mean = float(y.mean())
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    for tr, va in skf.split(np.zeros(len(y)), y):
        tab = pd.DataFrame({"k": keys.iloc[tr].astype(str), "y": y[tr]})
        g = tab.groupby("k")["y"].agg(["sum", "count"])
        enc = (g["sum"] + prior * global_mean) / (g["count"] + prior)
        oof[va] = keys.iloc[va].astype(str).map(enc).fillna(global_mean).to_numpy()
    return float(roc_auc_score(y, oof)), oof


def main():
    train = pd.read_csv("train.csv")
    y = train["label"].astype(int).to_numpy()
    main = np.load("artifacts/b6pro_main/predictions.npz")
    plus = np.load("reference/v10/oof_plus_h2_10.npz")
    pred = np.maximum(main["oof_main"], plus["oof"])
    base = float(roc_auc_score(y, pred))
    resid = y - pred
    # midband where model is uncertain
    mid = (pred > 0.05) & (pred < 0.35)
    print(f"base_max_auc={base:.6f} mid_frac={mid.mean():.3f}")

    days = pd.to_numeric(train["days"], errors="coerce")
    cond = pd.to_numeric(train["condition"], errors="coerce")
    ratio = cond / (days.abs() + 1)
    # candidate keys
    cands = {}
    cands["region"] = train["region"].astype(str)
    cands["source"] = train["source"].astype(str)
    cands["code"] = train["code"].astype(str)
    cands["version"] = train["version"].astype(str)
    cands["livability"] = train["livability"].astype(str)
    cands["grades"] = train["grades"].astype(str)
    cands["age8"] = pd.Series(
        np.where(train["age_range"] >= 8, 8, train["age_range"]).astype(int).astype(str),
        index=train.index,
    )
    cands["w_pair"] = train["w1"].astype(str) + "_" + train["w2"].astype(str)
    cands["t_pair"] = train["t1"].astype(str) + "_" + train["t2"].astype(str)
    # bins
    for name, s, q in [
        ("days5", days, 5),
        ("cond5", cond, 5),
        ("ratio5", ratio, 5),
    ]:
        edges = np.unique(s[np.isfinite(s)].quantile(np.linspace(0, 1, q + 1)))[1:-1]
        codes = np.searchsorted(edges, s.fillna(s.median()), side="right")
        cands[name] = pd.Series(codes, index=train.index).astype(str)

    # crosses
    keys = list(cands.keys())
    rows = []
    for a in keys:
        auc, _ = oof_te(cands[a], y)
        # residual correlation on midband
        te_auc_mid = None
        if mid.sum() > 100:
            te_auc_mid, _ = oof_te(cands[a][mid].reset_index(drop=True), y[mid])
        rows.append({"feat": a, "oof_te": auc, "mid_te": te_auc_mid})
    for a, b in [
        ("ratio5", "region"),
        ("ratio5", "source"),
        ("days5", "code"),
        ("days5", "w_pair"),
        ("days5", "t_pair"),
        ("cond5", "source"),
        ("age8", "region"),
        ("livability", "region"),
        ("version", "days5"),
        ("grades", "code"),
        ("w_pair", "ratio5"),
        ("t_pair", "ratio5"),
        ("source", "version"),
        ("region", "version"),
        ("region", "livability"),
    ]:
        key = (cands[a] + "|" + cands[b]).astype(str)
        auc, _ = oof_te(key, y)
        mid_te = None
        if mid.sum() > 100:
            mid_te, _ = oof_te(key[mid].reset_index(drop=True), y[mid])
        rows.append({"feat": f"{a}|{b}", "oof_te": auc, "mid_te": mid_te})

    df = pd.DataFrame(rows).sort_values("oof_te", ascending=False)
    out = Path("artifacts/b6pro_resid_mine")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "residual_candidates.csv", index=False)
    summary = {
        "base_max_auc": base,
        "target": TARGET,
        "gap": TARGET - base,
        "top10": df.head(10).to_dict(orient="records"),
        "top_mid": df.sort_values("mid_te", ascending=False).head(10).to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(df.head(15).to_string(index=False))
    print("--- mid ---")
    print(df.sort_values("mid_te", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
