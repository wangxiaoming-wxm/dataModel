#!/usr/bin/env python3
"""Nested CatBoost stacker: raw fold-local FE + stage1 OOF arms → push past max3."""

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
from insurance_claim.ebm_arm import build_ebm_features
from insurance_claim.model import build_submission

TARGET = 0.71
PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1200,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=12,
    random_strength=0.8,
    od_type="Iter",
    od_wait=80,
    verbose=False,
    thread_count=4,
    allow_writing_files=False,
)


def attach_arm_cols(df: pd.DataFrame, idx, gap, gap_bag, plus) -> pd.DataFrame:
    out = df.copy()
    out["arm_gap"] = gap[idx]
    out["arm_gap_bag"] = gap_bag[idx]
    out["arm_plus"] = plus[idx]
    out["arm_main"] = 0.5 * (gap[idx] + gap_bag[idx])
    out["arm_max3"] = np.maximum.reduce([gap[idx], gap_bag[idx], plus[idx]])
    out["arm_abs_gp"] = np.abs(gap[idx] - plus[idx])
    out["arm_abs_gbp"] = np.abs(gap_bag[idx] - plus[idx])
    out["arm_plus_minus_main"] = plus[idx] - out["arm_main"]
    return out


def prepare_xy(frame: pd.DataFrame):
    cats = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    out = frame.copy()
    for c in out.columns:
        if c in cats:
            out[c] = out[c].astype(str).fillna("__NA__")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            med = float(out[c].median()) if out[c].notna().any() else 0.0
            out[c] = out[c].fillna(med)
    return out, cats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027])
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_stack_raw"))
    args = ap.parse_args()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train["label"].astype(int).to_numpy()
    raw = train.drop(columns=["label"])
    b7 = np.load("reference/b7_closest/predictions.npz")
    gap, gap_bag, plus = b7["gap"], b7["gap_bag"], b7["plus"]
    max3 = np.maximum.reduce([gap, gap_bag, plus])
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")

    # For test arms we have component tests in frozen
    tg, tgb, tp = fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]

    oofs, tests = [], []
    for seed in args.seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(raw, y)):
            # fold-local FE on raw
            Xtr = build_ebm_features(raw.iloc[tr].reset_index(drop=True))
            Xva = build_ebm_features(raw.iloc[va].reset_index(drop=True))
            Xte = build_ebm_features(test.copy())
            Xtr = attach_arm_cols(Xtr, tr, gap, gap_bag, plus)
            Xva = attach_arm_cols(Xva, va, gap, gap_bag, plus)
            # test uses full stage1 test preds (standard stacking)
            Xte = attach_arm_cols(Xte, np.arange(len(test)), tg, tgb, tp)
            Xva = Xva.reindex(columns=Xtr.columns)
            Xte = Xte.reindex(columns=Xtr.columns)
            trd, cats = prepare_xy(Xtr)
            vad, _ = prepare_xy(Xva)
            ted, _ = prepare_xy(Xte)
            vad = vad.reindex(columns=trd.columns)
            ted = ted.reindex(columns=trd.columns)
            p = dict(PARAMS)
            p["random_seed"] = seed + fold
            model = CatBoostClassifier(**p)
            model.fit(
                trd,
                y[tr],
                eval_set=(vad, y[va]),
                cat_features=cats,
                use_best_model=True,
            )
            oof[va] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5
            print(
                f"stack_raw seed={seed} fold={fold} auc={roc_auc_score(y[va], oof[va]):.5f} max3fold={roc_auc_score(y[va], max3[va]):.5f}",
                flush=True,
            )
        print(f"stack_raw seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
        oofs.append(oof)
        tests.append(pte)

    oof = np.mean(np.vstack(oofs), 0)
    te = np.mean(np.vstack(tests), 0)
    print("solo stack", roc_auc_score(y, oof), "max3", roc_auc_score(y, max3))
    print("max(stack,max3)", roc_auc_score(y, np.maximum(oof, max3)))
    fused = nested_select_rule(y, [gap, gap_bag, plus, oof])
    print("4arm nested", fused["nested_oof_auc"], fused["selected_rule"])
    # also nested between stack and max3 only
    fused2 = nested_select_rule(y, [max3, oof])
    print("stack×max3 nested", fused2["nested_oof_auc"], fused2["selected_rule"])

    best = fused if fused["nested_oof_auc"] >= fused2["nested_oof_auc"] else fused2
    if best is fused:
        tp_out = apply_rule(best["selected_rule"], [tg, tgb, tp, te])
    else:
        tp_out = apply_rule(best["selected_rule"], [fr["test"], te] if "test" in fr else [np.maximum.reduce([tg, tgb, tp]), te])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y=y,
        oof=best["nested_oof"],
        test=tp_out,
        oof_stack=oof,
        test_stack=te,
    )
    build_submission(test, sample, tp_out, args.output_dir / "submission_b6pro.csv")
    metrics = {
        "experiment_id": "b6pro_stack_raw",
        "stack_oof_auc": float(roc_auc_score(y, oof)),
        "nested_oof_auc": best["nested_oof_auc"],
        "selected_rule": best["selected_rule"],
        "four_arm_nested": fused["nested_oof_auc"],
        "stack_max3_nested": fused2["nested_oof_auc"],
        "baseline_max3": float(roc_auc_score(y, max3)),
        "gate_0_71": best["nested_oof_auc"] >= TARGET,
        "gap_to_0_71": round(TARGET - best["nested_oof_auc"], 6),
        "public_b7_signal": {"local": 0.702704955, "public": 0.707},
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "stage1_oof_as_features": True,
            "no_oof_weight_search": True,
            "new_data_only": True,
            "note": "stage1 OOF from B7 closest arms; stacker nested by outer SKF",
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ["stack_oof_auc", "nested_oof_auc", "gate_0_71", "gap_to_0_71"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
