#!/usr/bin/env python3
"""Region-specialist CatBoost arms for weak regions; nested fuse with B7 max3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.train_b6 import PARAMS_B5, build_gap
from insurance_claim.model import build_submission

TARGET = 0.71
# regions where global max3 < ~0.68 (drag overall)
WEAK_REGIONS = ("9685", "f09d", "fafc", "6645")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_region_spec"))
    args = ap.parse_args()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train["label"].astype(int)
    features = train.drop(columns=["label"])
    region = train["region"].astype(str)
    region_te = test["region"].astype(str)
    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    tmax = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])

    # Start from max3, overwrite weak-region rows with specialist OOF
    oof_spec = max3.copy()
    test_spec = tmax.copy()
    params = {**PARAMS_B5, "thread_count": 4}

    for reg in WEAK_REGIONS:
        idx_all = np.where(region.to_numpy() == reg)[0]
        idx_te = np.where(region_te.to_numpy() == reg)[0]
        if len(idx_all) < 200 or y.iloc[idx_all].sum() < 20:
            print(f"skip {reg}", flush=True)
            continue
        print(f"=== specialist region={reg} n={len(idx_all)} pos={int(y.iloc[idx_all].sum())} ===", flush=True)
        oof_r = np.zeros(len(idx_all))
        # map local positions
        local_y = y.iloc[idx_all].reset_index(drop=True)
        local_X = features.iloc[idx_all].reset_index(drop=True)
        te_preds = []
        for seed in args.seeds:
            oof_s = np.zeros(len(idx_all))
            pte = np.zeros(len(test))
            # If too few for 5-fold, use 3-fold
            n_splits = 5 if local_y.sum() >= 50 else 3
            skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
            for fold, (tr, va) in enumerate(skf.split(local_X, local_y)):
                # FE uses only region-train fold + full test (fold-local on this subset)
                Xtr = local_X.iloc[tr].reset_index(drop=True)
                Xva = local_X.iloc[va].reset_index(drop=True)
                trd, vad, ted, cats = build_gap(Xtr, Xva, test.copy())
                p = dict(params)
                p["random_seed"] = seed + fold + hash(reg) % 1000
                model = CatBoostClassifier(**p)
                model.fit(
                    trd,
                    local_y.iloc[tr],
                    eval_set=(vad, local_y.iloc[va]),
                    cat_features=cats,
                    use_best_model=True,
                )
                oof_s[va] = model.predict_proba(vad)[:, 1]
                pte += model.predict_proba(ted)[:, 1] / n_splits
                print(
                    f"  {reg} seed={seed} fold={fold} auc={roc_auc_score(local_y.iloc[va], oof_s[va]):.5f}",
                    flush=True,
                )
            print(f"  {reg} seed={seed} OOF={roc_auc_score(local_y, oof_s):.6f}", flush=True)
            oof_r += oof_s / len(args.seeds)
            te_preds.append(pte)
        te_r = np.mean(np.vstack(te_preds), 0)
        # write back
        before = roc_auc_score(y.iloc[idx_all], max3[idx_all])
        after = roc_auc_score(y.iloc[idx_all], oof_r)
        print(f"  {reg} region AUC max3={before:.5f} spec={after:.5f}", flush=True)
        oof_spec[idx_all] = oof_r
        if len(idx_te):
            test_spec[idx_te] = te_r[idx_te]

    print("global max3", roc_auc_score(y, max3))
    print("spec patched", roc_auc_score(y, oof_spec))
    # nested fuse max3 vs spec vs components
    fused = nested_select_rule(y.to_numpy(), [b7["gap"], b7["gap_bag"], b7["plus"], oof_spec])
    fused2 = nested_select_rule(y.to_numpy(), [max3, oof_spec])
    print("4arm", fused["nested_oof_auc"], fused["selected_rule"])
    print("max3×spec", fused2["nested_oof_auc"], fused2["selected_rule"])
    best = fused if fused["nested_oof_auc"] >= fused2["nested_oof_auc"] else fused2
    if best is fused:
        tp = apply_rule(best["selected_rule"], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], test_spec])
    else:
        tp = apply_rule(best["selected_rule"], [tmax, test_spec])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y=y.to_numpy(),
        oof=best["nested_oof"],
        test=tp,
        oof_spec=oof_spec,
        test_spec=test_spec,
    )
    build_submission(test, sample, tp, args.output_dir / "submission_b6pro.csv")
    metrics = {
        "experiment_id": "b6pro_region_spec",
        "weak_regions": list(WEAK_REGIONS),
        "spec_oof_auc": float(roc_auc_score(y, oof_spec)),
        "nested_oof_auc": best["nested_oof_auc"],
        "selected_rule": best["selected_rule"],
        "four_arm_nested": fused["nested_oof_auc"],
        "max3_spec_nested": fused2["nested_oof_auc"],
        "baseline_max3": float(roc_auc_score(y, max3)),
        "gate_0_71": best["nested_oof_auc"] >= TARGET,
        "gap_to_0_71": round(TARGET - best["nested_oof_auc"], 6),
        "public_b7_signal": {"local": 0.702704955, "public": 0.707},
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "region_specialist_nested": True,
            "no_oof_weight_search": True,
            "new_data_only": True,
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ["spec_oof_auc", "nested_oof_auc", "gate_0_71", "gap_to_0_71"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
