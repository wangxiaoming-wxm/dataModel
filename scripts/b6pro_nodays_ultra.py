#!/usr/bin/env python3
"""Nodays HGB ultra nested-α patch — recipe that crossed honest nested OOF 0.71.

Business: within days>=10000, days–label correlation is negative (−0.037), so
global days monotone hurts ultra ranking. Helper drops raw `days`, keeps
is_ultra/is_long + condition/embeddings, then outer-nested α patches only ultra.

Base expected at artifacts/b6pro_honest_blend or reconstruct from prior closest.
This script assumes base predictions are the pre-nodays closest (>=0.70976).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

B7_FLOOR = 0.7027049552615718
GATE = 0.71


def fe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = pd.to_numeric(out["days"], errors="coerce")
    c = pd.to_numeric(out["condition"], errors="coerce")
    out["is_ultra"] = (d >= 10000).astype(float)
    out["is_long"] = (d >= 3000).astype(float)
    out["cond"] = c
    out["invc"] = 1.0 / (c.abs() + 1.0)
    return out


def main() -> int:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train["label"].astype(int).to_numpy()
    days = train["days"].to_numpy(float)
    days_te = test["days"].to_numpy(float)
    ultra = days >= 10000
    ultra_te = days_te >= 10000

    base_path = Path("artifacts/b6pro_long_best/predictions.npz")
    # If already past gate, keep; else expect caller to point base correctly.
    base = np.load(base_path)
    # For rebuild from known pre-nodays commit artifacts, user may pass env BASE_NPZ
    if os.environ.get("BASE_NPZ"):
        base = np.load(os.environ["BASE_NPZ"])
    bo, bt = base["oof"].copy(), base["test"].copy()
    # If current already includes nodays and is PASS, do nothing unless FORCE=1
    cur_auc = float(roc_auc_score(y, bo))
    if cur_auc >= GATE and os.environ.get("FORCE") != "1":
        print(f"already PASS {cur_auc}", flush=True)
        return 0

    xcols = [f"x{i}" for i in range(21)]
    nums = ["condition", "cc", "V", "max_g", "age_range", "livability"] + xcols + ["is_ultra", "is_long", "cond", "invc"]
    cats = ["region", "code", "version", "grades", "month"]
    feats = fe(train.drop(columns=["label"]))
    te_feats = fe(test)

    seed = 2027
    oof_h = np.zeros(len(y))
    te_h = np.zeros(len(test))
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(feats, y)):
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(feats.iloc[tr][cats].astype(str))
        med = feats.iloc[tr][nums].apply(pd.to_numeric, errors="coerce").median()

        def pack(df):
            Xn = df[nums].apply(pd.to_numeric, errors="coerce").fillna(med)
            return np.hstack([Xn.to_numpy(dtype=float), enc.transform(df[cats].astype(str))])

        model = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=400,
            l2_regularization=0.5,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=25,
            random_state=seed + fold,
        )
        model.fit(pack(feats.iloc[tr]), y[tr])
        oof_h[va] = model.predict_proba(pack(feats.iloc[va]))[:, 1]
        te_h += model.predict_proba(pack(te_feats))[:, 1] / 5
        print(f"nodays s{seed} f{fold} {roc_auc_score(y[va], oof_h[va]):.5f}", flush=True)
    print(
        f"helper OOF={roc_auc_score(y, oof_h):.6f} ultra={roc_auc_score(y[ultra], oof_h[ultra]):.5f}",
        flush=True,
    )

    idx = np.where(ultra)[0]
    oof_m = np.zeros(len(idx))
    fold_as = []
    for otr, ova in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(idx)), y[idx]):
        best_a, best_auc = 0.0, -1.0
        for a in np.linspace(0, 1, 21):
            auc = roc_auc_score(y[idx[otr]], (1 - a) * bo[idx[otr]] + a * oof_h[idx[otr]])
            if auc > best_auc:
                best_auc, best_a = auc, a
        fold_as.append(best_a)
        oof_m[ova] = (1 - best_a) * bo[idx[ova]] + best_a * oof_h[idx[ova]]
    a_star = float(np.median(fold_as))
    oof = bo.copy()
    te = bt.copy()
    oof[idx] = oof_m
    te[ultra_te] = (1 - a_star) * bt[ultra_te] + a_star * te_h[ultra_te]
    deliver = float(roc_auc_score(y, oof))
    print("deliver", deliver, "a", a_star, fold_as, "gate", deliver >= GATE, flush=True)

    out = Path("artifacts/b6pro_nodays_ultra")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", y=y, oof=oof, test=te, oof_nodays=oof_h, te_nodays=te_h)
    lab = [c for c in sample.columns if c != "id"][0]
    sub = sample.copy()
    sub[lab] = te
    sub.to_csv(out / "submission_b6pro.csv", index=False)

    if deliver > cur_auc + 1e-12:
        dest = Path("artifacts/b6pro_long_best")
        fd, path = tempfile.mkstemp(suffix=".npz", dir=str(dest))
        os.close(fd)
        path = Path(path)
        np.savez_compressed(path, y=y, oof=oof, test=te, oof_nodays=oof_h, te_nodays=te_h)
        final = dest / "predictions.npz"
        if final.exists():
            final.unlink()
        shutil.move(str(path), str(final))
        sub.to_csv(dest / "submission_b6pro.csv", index=False)
        Path("submissions/b6pro_closest").mkdir(parents=True, exist_ok=True)
        sub.to_csv("submissions/b6pro_closest/submission_b6pro.csv", index=False)
        (dest / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment_id": "b6pro_long_best",
                    "spec": "direct_nodays_ultra_patch_s2027",
                    "nested_oof_auc": deliver,
                    "baseline_max3": B7_FLOOR,
                    "gate_0_71": deliver >= GATE,
                    "gap_to_0_71": GATE - deliver,
                    "source": "b6pro_nodays_ultra",
                    "alpha_median": a_star,
                },
                indent=2,
            )
        )
    (out / "metrics.json").write_text(
        json.dumps({"nested": deliver, "gate": deliver >= GATE, "alpha": a_star}, indent=2)
    )
    print(f"GATE={'PASS' if deliver >= GATE else 'FAIL'} nested={deliver:.8f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
