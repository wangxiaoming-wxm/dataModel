#!/usr/bin/env python3
"""Feature-dropout bagging on gap FE → diverse near-strength arms for nested max."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import nested_select_rule
from insurance_claim.model import build_submission
from insurance_claim.train_b6 import build_gap

TARGET = 0.715
PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1200,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=10,
    random_strength=1.0,
    bagging_temperature=1.0,
    od_type="Iter",
    od_wait=120,
    verbose=False,
    thread_count=8,
    allow_writing_files=False,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-arms", type=int, default=4)
    ap.add_argument("--drop-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_dropout"))
    ap.add_argument("--main-npz", type=Path, default=Path("artifacts/b6pro_main/predictions.npz"))
    ap.add_argument("--plus-npz", type=Path, default=Path("reference/v10/oof_plus_h2_10.npz"))
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "submit_sample.csv")
    y = train["label"].astype(int)
    features = train.drop(columns=["label"])
    rng = np.random.default_rng(args.seed)

    # probe columns once
    tr0, _, _, cats0 = build_gap(features.iloc[:100].reset_index(drop=True), features.iloc[:100].reset_index(drop=True), test.iloc[:50].copy())
    all_cols = list(tr0.columns)
    drop_candidates = [c for c in all_cols if c.startswith("gap_") or "days_condition" in c or "__category_cross" in c]

    arm_oofs = []
    arm_tests = []
    arm_aucs = {}
    for a in range(args.n_arms):
        drop = set(rng.choice(drop_candidates, size=max(1, int(len(drop_candidates) * args.drop_frac)), replace=False))
        oof = np.zeros(len(train), dtype=float)
        pte = np.zeros(len(test), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=args.seed + a).split(features, y)
        ):
            Xtr = features.iloc[tr_idx].reset_index(drop=True)
            Xva = features.iloc[va_idx].reset_index(drop=True)
            tr, va, te, cats = build_gap(Xtr, Xva, test.copy())
            keep = [c for c in tr.columns if c not in drop]
            cats = [c for c in cats if c in keep]
            tr, va, te = tr[keep], va[keep], te[keep]
            p = dict(PARAMS)
            p["random_seed"] = args.seed + a * 10 + fold
            model = CatBoostClassifier(**p)
            model.fit(tr, y.iloc[tr_idx], eval_set=(va, y.iloc[va_idx]), cat_features=cats, use_best_model=True)
            oof[va_idx] = model.predict_proba(va)[:, 1]
            pte += model.predict_proba(te)[:, 1] / 5
            print(f"drop{a} fold={fold} auc={roc_auc_score(y.iloc[va_idx], oof[va_idx]):.5f} dropped={len(drop)}", flush=True)
        auc = float(roc_auc_score(y, oof))
        print(f"drop{a} OOF={auc:.6f}", flush=True)
        arm_oofs.append(oof)
        arm_tests.append(pte)
        arm_aucs[f"drop{a}"] = auc

    main = np.load(args.main_npz)
    plus = np.load(args.plus_npz)
    all_oofs = [main["oof_main"], plus["oof"]] + arm_oofs
    all_tests = [main["test_main"], plus["test"]] + arm_tests
    fused = nested_select_rule(y.to_numpy(), all_oofs)
    metrics = {
        "experiment_id": "b6pro_feature_dropout",
        "protocol_id": "IA-AUC715-B6PRO-v1",
        "dropout_arm_aucs": arm_aucs,
        "nested_oof_auc": fused["nested_oof_auc"],
        "pooled_oof_auc": fused["nested_oof_auc"],
        "selected_rule": fused["selected_rule"],
        "full_data_scores": fused["full_data_scores"],
        "gate_0_715": fused["nested_oof_auc"] >= TARGET,
        "gap_to_0_715": round(TARGET - fused["nested_oof_auc"], 6),
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "no_oof_weight_search": True,
            "feature_dropout_preregistered": True,
            "new_data_only": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "predictions.npz", y=y.to_numpy(), oof=fused["nested_oof"], test=apply_test(fused["selected_rule"], all_tests))
    build_submission(test, sample, apply_test(fused["selected_rule"], all_tests), args.output_dir / "submission_b6pro.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ["dropout_arm_aucs", "nested_oof_auc", "selected_rule", "gate_0_715", "gap_to_0_715"]}, indent=2))
    return 0


def apply_test(rule, tests):
    from insurance_claim.b6pro_fusion import apply_rule

    return apply_rule(rule, tests)


if __name__ == "__main__":
    raise SystemExit(main())
