#!/usr/bin/env python3
"""Anti-correlated residual CatBoost: focus weight where B6 max3 is uncertain/wrong."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.model import build_submission
from insurance_claim.train_b6 import PARAMS_B5, build_gap
from insurance_claim.v10_plus.plus_features import build_plus, parse_frame

TARGET = 0.71


def build_plus_only(X_tr, X_va, X_te):
    Ptr, Pva, Pte = parse_frame(X_tr.copy()), parse_frame(X_va.copy()), parse_frame(X_te.copy())
    for c in ("id", "x19"):
        for d in (Ptr, Pva, Pte):
            if c in d.columns:
                d.drop(columns=[c], inplace=True)
    return build_plus(Ptr, Pva, Pte)


def main() -> int:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train["label"].astype(int).to_numpy()
    features = train.drop(columns=["label"])
    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])

    # uncertainty weight from max3 (fixed stage1 OOF — slight optimism for weights only)
    # Use nested: compute weights from out-of-fold max3 within outer... we use global OOF as approx.
    conf = np.abs(max3 - 0.1)  # distance from base rate-ish; lower = uncertain
    # actually midband uncertainty:
    unc = 1.0 - np.abs(max3 - np.median(max3)) / (np.abs(max3 - np.median(max3)).max() + 1e-6)

    seeds = [2026, 2027, 2028, 2029]
    oofs, tests = [], []
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(features, y)):
            # weights on train: upweight uncertain + errors vs max3
            err = np.abs(y[tr] - max3[tr])
            w = 0.5 + 2.0 * unc[tr] + 3.0 * err
            w = w / w.mean()
            trd, vad, ted, cats = build_plus_only(
                features.iloc[tr].reset_index(drop=True),
                features.iloc[va].reset_index(drop=True),
                test.copy(),
            )
            p = dict(PARAMS_B5)
            p.update(
                iterations=1800,
                learning_rate=0.025,
                depth=7,
                l2_leaf_reg=20,
                random_strength=1.2,
                bagging_temperature=1.5,
                thread_count=4,
                random_seed=seed + fold,
            )
            model = CatBoostClassifier(**p)
            model.fit(
                trd,
                y[tr],
                sample_weight=w,
                eval_set=(vad, y[va]),
                cat_features=cats,
                use_best_model=True,
            )
            oof[va] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5
            print(
                f"residw seed={seed} fold={fold} auc={roc_auc_score(y[va], oof[va]):.5f}",
                flush=True,
            )
        print(f"residw seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
        oofs.append(oof)
        tests.append(pte)

    oof = np.mean(np.vstack(oofs), 0)
    te = np.mean(np.vstack(tests), 0)
    print(
        "solo",
        roc_auc_score(y, oof),
        "corr_max3",
        np.corrcoef(oof, max3)[0, 1],
        "corr_gap",
        np.corrcoef(oof, b7["gap"])[0, 1],
    )
    fused = nested_select_rule(y, [b7["gap"], b7["gap_bag"], b7["plus"], oof])
    print("nested", fused["nested_oof_auc"], fused["selected_rule"])
    print("full", fused["full_data_scores"])
    tp = apply_rule(fused["selected_rule"], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], te])
    out = Path("artifacts/b6pro_residw")
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "predictions.npz", y=y, oof=fused["nested_oof"], test=tp, oof_residw=oof, test_residw=te)
    build_submission(test, sample, tp, out / "submission_b6pro.csv")
    metrics = {
        "experiment_id": "b6pro_residw",
        "oof_auc": float(roc_auc_score(y, oof)),
        "corr_max3": float(np.corrcoef(oof, max3)[0, 1]),
        "nested_oof_auc": fused["nested_oof_auc"],
        "selected_rule": fused["selected_rule"],
        "full_data_scores": fused["full_data_scores"],
        "baseline_max3": float(roc_auc_score(y, max3)),
        "gate_0_71": fused["nested_oof_auc"] >= TARGET,
        "gap_to_0_71": round(TARGET - fused["nested_oof_auc"], 6),
        "public_b7_signal": {"local": 0.702704955, "public": 0.707},
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "stage1_oof_for_weights_only": True,
            "no_oof_weight_search": True,
            "new_data_only": True,
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ["oof_auc", "corr_max3", "nested_oof_auc", "gate_0_71", "gap_to_0_71"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
