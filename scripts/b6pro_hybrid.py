#!/usr/bin/env python3
"""Hybrid arm: B6 gap FE + keep x0-x18 numerics (fold-local; no TE)."""

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
    iterations=1400,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=10,
    random_strength=0.7,
    bagging_temperature=1.0,
    od_type="Iter",
    od_wait=150,
    verbose=False,
    thread_count=8,
    allow_writing_files=False,
)
XCOLS = [f"x{i}" for i in range(19)]  # x0..x18


def build_hybrid(X_tr, X_va, X_te):
    tr, va, te, cats = build_gap(X_tr, X_va, X_te)
    # add x0..x18 if present on raw
    for df, raw in ((tr, X_tr), (va, X_va), (te, X_te)):
        for c in XCOLS:
            if c in raw.columns:
                df[c] = pd.to_numeric(raw[c], errors="coerce")
        # fold-local median fill for x*
        for c in XCOLS:
            if c not in df.columns:
                continue
            med = float(tr[c].median()) if c in tr.columns and tr[c].notna().any() else 0.0
            df[c] = df[c].fillna(med)
    # row stats
    present = [c for c in XCOLS if c in tr.columns]
    if present:
        for df in (tr, va, te):
            xs = df[present]
            df["x_row_mean"] = xs.mean(axis=1)
            df["x_row_std"] = xs.std(axis=1)
            df["x_row_max"] = xs.max(axis=1)
            df["x_row_min"] = xs.min(axis=1)
    return tr, va, te, cats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_hybrid"))
    ap.add_argument("--main-npz", type=Path, default=Path("artifacts/b6pro_main/predictions.npz"))
    ap.add_argument("--plus-npz", type=Path, default=Path("reference/v10/oof_plus_h2_10.npz"))
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "submit_sample.csv")
    y = train["label"].astype(int)
    features = train.drop(columns=["label"])

    oof_by_seed, test_by_seed = {}, {}
    for seed in args.seeds:
        oof = np.zeros(len(train), dtype=float)
        pte = np.zeros(len(test), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=seed).split(features, y)
        ):
            Xtr = features.iloc[tr_idx].reset_index(drop=True)
            Xva = features.iloc[va_idx].reset_index(drop=True)
            tr, va, te, cats = build_hybrid(Xtr, Xva, test.copy())
            p = dict(PARAMS)
            p["random_seed"] = seed + fold
            model = CatBoostClassifier(**p)
            model.fit(tr, y.iloc[tr_idx], eval_set=(va, y.iloc[va_idx]), cat_features=cats, use_best_model=True)
            oof[va_idx] = model.predict_proba(va)[:, 1]
            pte += model.predict_proba(te)[:, 1] / 5
            print(f"hybrid seed={seed} fold={fold} auc={roc_auc_score(y.iloc[va_idx], oof[va_idx]):.5f}", flush=True)
        print(f"hybrid seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
        oof_by_seed[seed] = oof
        test_by_seed[seed] = pte

    oof = np.mean(np.vstack(list(oof_by_seed.values())), axis=0)
    te = np.mean(np.vstack(list(test_by_seed.values())), axis=0)
    main = np.load(args.main_npz)
    plus = np.load(args.plus_npz)
    mo = main["oof_main"]; mt = main["test_main"]
    po = plus["oof"]; pt = plus["test"]
    fused = nested_select_rule(y.to_numpy(), [mo, po, oof])
    metrics = {
        "experiment_id": "b6pro_hybrid_xkeep",
        "protocol_id": "IA-AUC715-B6PRO-v1",
        "oof_auc": float(roc_auc_score(y, oof)),
        "seed_aucs": {str(s): float(roc_auc_score(y, oof_by_seed[s])) for s in args.seeds},
        "corr_vs_main": float(np.corrcoef(oof, mo)[0, 1]),
        "corr_vs_plus": float(np.corrcoef(oof, po)[0, 1]),
        "nested_oof_auc": fused["nested_oof_auc"],
        "pooled_oof_auc": fused["nested_oof_auc"],
        "fusion": {"selected_rule": fused["selected_rule"], "full_data_scores": fused["full_data_scores"]},
        "gate_0_715": fused["nested_oof_auc"] >= TARGET,
        "gap_to_0_715": round(TARGET - fused["nested_oof_auc"], 6),
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "no_oof_weight_search": True,
            "new_data_only": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "predictions.npz", y=y.to_numpy(), oof=fused["nested_oof"], test=np.maximum.reduce([mt, pt, te]), oof_hybrid=oof, test_hybrid=te)
    build_submission(test, sample, np.maximum.reduce([mt, pt, te]), args.output_dir / "submission_b6pro.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ["oof_auc", "nested_oof_auc", "corr_vs_main", "corr_vs_plus", "gate_0_715", "gap_to_0_715"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
