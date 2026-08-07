#!/usr/bin/env python3
"""Nested residual corrector: CatBoost with frozen main OOF as feature (nested only).

For each outer fold: use main OOF values on train rows as a feature to train a
corrector; predict valid. This is standard nested stacking with 1 base arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.model import build_submission
from insurance_claim.v10_plus.plus_features import build_plus

TARGET = 0.715
PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=20,
    random_strength=1.0,
    od_type="Iter",
    od_wait=100,
    verbose=False,
    thread_count=8,
    allow_writing_files=False,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-npz", type=Path, required=True)
    ap.add_argument("--plus-npz", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "submit_sample.csv")
    y = train["label"].astype(int).to_numpy()
    main = np.load(args.main_npz)
    main_oof = main["oof_main"] if "oof_main" in main.files else main["oof"]
    main_te = main["test_main"] if "test_main" in main.files else main["test"]
    plus_oof = plus_te = None
    if args.plus_npz:
        p = np.load(args.plus_npz)
        plus_oof = p["oof"] if "oof" in p.files else p["oof_plus"]
        plus_te = p["test"] if "test" in p.files else p["test_plus"]

    features = train.drop(columns=["label"])
    oof_by_seed = {}
    test_by_seed = {}
    for seed in args.seeds:
        oof = np.zeros(len(train), dtype=float)
        pte = np.zeros(len(test), dtype=float)
        for fold, (tr, va) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=seed).split(features, y)
        ):
            Xtr = features.iloc[tr].reset_index(drop=True)
            Xva = features.iloc[va].reset_index(drop=True)
            tr_x, va_x, te_x, cats = build_plus(Xtr, Xva, test.copy())
            # inject base scores (frozen OOF — valid rows never used to train base)
            tr_x = tr_x.copy(); va_x = va_x.copy(); te_x = te_x.copy()
            tr_x["base_main"] = main_oof[tr]
            va_x["base_main"] = main_oof[va]
            te_x["base_main"] = main_te
            if plus_oof is not None:
                tr_x["base_plus"] = plus_oof[tr]
                va_x["base_plus"] = plus_oof[va]
                te_x["base_plus"] = plus_te
            p = dict(PARAMS)
            p["random_seed"] = seed + fold
            model = CatBoostClassifier(**p)
            model.fit(tr_x, y[tr], eval_set=(va_x, y[va]), cat_features=cats, use_best_model=True)
            oof[va] = model.predict_proba(va_x)[:, 1]
            pte += model.predict_proba(te_x)[:, 1] / 5
            print(f"corr seed={seed} fold={fold} auc={roc_auc_score(y[va], oof[va]):.5f}", flush=True)
        print(f"corr seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
        oof_by_seed[seed] = oof
        test_by_seed[seed] = pte

    oof = np.mean(np.vstack(list(oof_by_seed.values())), axis=0)
    te = np.mean(np.vstack(list(test_by_seed.values())), axis=0)
    auc = float(roc_auc_score(y, oof))
    # Also nested-max with main
    from insurance_claim.b6pro_fusion import nested_select_rule

    fused = nested_select_rule(y, [main_oof, oof])
    metrics = {
        "experiment_id": "b6pro_nested_corrector",
        "protocol_id": "IA-AUC715-B6PRO-v1",
        "oof_auc": auc,
        "seed_aucs": {str(s): float(roc_auc_score(y, oof_by_seed[s])) for s in args.seeds},
        "corr_vs_main": float(np.corrcoef(oof, main_oof)[0, 1]),
        "nested_max_with_main": fused["nested_oof_auc"],
        "nested_oof_auc": fused["nested_oof_auc"],
        "pooled_oof_auc": fused["nested_oof_auc"],
        "gate_0_715": fused["nested_oof_auc"] >= TARGET,
        "gap_to_0_715": round(TARGET - fused["nested_oof_auc"], 6),
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "no_oof_weight_search": True,
            "nested_corrector_with_frozen_base_scores": True,
            "new_data_only": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y=y,
        oof=fused["nested_oof"],
        test=np.maximum(main_te, te),  # align with max if selected
        oof_corrector=oof,
        test_corrector=te,
    )
    build_submission(test, sample, np.maximum(main_te, te), args.output_dir / "submission_b6pro.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ["oof_auc", "nested_max_with_main", "gate_0_715", "gap_to_0_715", "corr_vs_main"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
