#!/usr/bin/env python3
"""FLAML AutoML arm fused with B7 + current region-aging closest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

from insurance_claim.b6pro_fusion import nested_select_rule

B7_FLOOR = 0.7027049552615718
GATE = 0.71


def main() -> int:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = train["label"].astype(int)
    features = train.drop(columns=["label"])
    days = features["days"].to_numpy(dtype=float)
    long = days >= 3000

    b7 = np.load("reference/b7_closest/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    region_arm = np.load("artifacts/b6pro_long_region_aging/predictions.npz")["arm"]

    eng = features.copy()
    eng["log_days"] = np.log1p(eng["days"].clip(lower=0))
    eng["ratio"] = eng["condition"] / (eng["days"].abs() + 1)
    num_cols = [c for c in eng.columns if pd.api.types.is_numeric_dtype(eng[c]) and c != "id"]
    cat_cols = [c for c in eng.columns if c not in num_cols and c != "id"]

    oof = np.zeros(len(y))
    pte = np.zeros(len(test))
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(eng, y)):
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr = eng.iloc[tr].copy()
        Xva = eng.iloc[va].copy()
        Xte_e = test.copy()
        Xte_e["log_days"] = np.log1p(Xte_e["days"].clip(lower=0))
        Xte_e["ratio"] = Xte_e["condition"] / (Xte_e["days"].abs() + 1)
        Xtr[cat_cols] = enc.fit_transform(Xtr[cat_cols].astype(str))
        Xva[cat_cols] = enc.transform(Xva[cat_cols].astype(str))
        Xte_e[cat_cols] = enc.transform(Xte_e[cat_cols].astype(str))
        use = num_cols + cat_cols
        for c in use:
            for df in (Xtr, Xva, Xte_e):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            med = Xtr[c].median()
            Xtr[c] = Xtr[c].fillna(med)
            Xva[c] = Xva[c].fillna(med)
            Xte_e[c] = Xte_e[c].fillna(med)
        automl = AutoML()
        automl.fit(
            Xtr[use],
            y.iloc[tr].to_numpy(),
            task="classification",
            metric="roc_auc",
            time_budget=120,
            estimator_list=["lgbm", "xgboost", "rf", "extra_tree"],
            eval_method="cv",
            n_splits=3,
            seed=42 + fold,
            verbose=0,
        )
        oof[va] = automl.predict_proba(Xva[use])[:, 1]
        pte += automl.predict_proba(Xte_e[use])[:, 1] / 5.0
        print(
            f"fold{fold} best={automl.best_estimator} auc={roc_auc_score(y.iloc[va], oof[va]):.5f}",
            flush=True,
        )

    print(
        "flaml OOF",
        roc_auc_score(y, oof),
        "long",
        roc_auc_score(y.to_numpy()[long], oof[long]),
        "corr",
        np.corrcoef(oof, max3)[0, 1],
        flush=True,
    )
    cands = {
        "b7+flaml": nested_select_rule(y.to_numpy(), [b7["gap"], b7["gap_bag"], b7["plus"], oof]),
        "max3×flaml": nested_select_rule(y.to_numpy(), [max3, oof]),
        "region×flaml": nested_select_rule(y.to_numpy(), [region_arm, oof]),
        "b7+region+flaml": nested_select_rule(
            y.to_numpy(), [b7["gap"], b7["gap_bag"], b7["plus"], region_arm, oof]
        ),
    }
    for k, v in sorted(cands.items(), key=lambda kv: -kv[1]["nested_oof_auc"]):
        print(k, v["nested_oof_auc"], v["selected_rule"], flush=True)
    best_name = max(cands, key=lambda k: cands[k]["nested_oof_auc"])
    best = cands[best_name]
    out = Path("artifacts/b6pro_flaml")
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / "predictions.npz",
        y=y.to_numpy(),
        oof=best["nested_oof"],
        oof_flaml=oof,
        test_flaml=pte,
    )
    metrics = {
        "experiment_id": "b6pro_flaml",
        "best": best_name,
        "nested_oof_auc": best["nested_oof_auc"],
        "flaml_auc": float(roc_auc_score(y, oof)),
        "baseline_max3": B7_FLOOR,
        "gate_0_71": bool(best["nested_oof_auc"] >= GATE),
        "gap_to_0_71": float(GATE - best["nested_oof_auc"]),
        "all": {k: v["nested_oof_auc"] for k, v in cands.items()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    return 0 if metrics["gate_0_71"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
