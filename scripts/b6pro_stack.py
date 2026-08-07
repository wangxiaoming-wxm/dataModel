#!/usr/bin/env python3
"""Nested stacking / arm-router for B6pro (honest OOF; no continuous weight search on report OOF).

Uses only frozen arm OOF predictions + optional raw scalars as meta-features.
Outer report score is nested OOF from StratifiedKFold meta-model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

TARGET = 0.715


def load_arm(spec: str):
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    path = Path(parts[0])
    z = np.load(path)
    key = parts[1] if len(parts) > 1 and parts[1] else ("oof" if "oof" in z.files else "oof_main")
    tkey = parts[2] if len(parts) > 2 and parts[2] else ("test" if "test" in z.files else "test_main")
    return name, z[key], z[tkey], z["y"] if "y" in z.files else None


def build_meta(arm_oofs: list[np.ndarray], train: pd.DataFrame | None = None) -> np.ndarray:
    stacked = np.vstack(arm_oofs).T  # n x k
    mean = stacked.mean(axis=1, keepdims=True)
    mx = stacked.max(axis=1, keepdims=True)
    mn = stacked.min(axis=1, keepdims=True)
    std = stacked.std(axis=1, keepdims=True)
    spread = mx - mn
    feats = [stacked, mean, mx, mn, std, spread]
    if len(arm_oofs) >= 2:
        feats.append((arm_oofs[0] - arm_oofs[1]).reshape(-1, 1))
        feats.append(np.abs(arm_oofs[0] - arm_oofs[1]).reshape(-1, 1))
    if train is not None:
        days = pd.to_numeric(train["days"], errors="coerce").fillna(0).to_numpy().reshape(-1, 1)
        cond = pd.to_numeric(train["condition"], errors="coerce").fillna(0).to_numpy().reshape(-1, 1)
        age = pd.to_numeric(train["age_range"], errors="coerce").fillna(0).to_numpy().reshape(-1, 1)
        feats.extend([days / 10000.0, cond, age / 10.0, cond / (np.abs(days) + 1.0)])
    return np.hstack(feats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--model", choices=["logit", "lgb"], default="lgb")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    names, oofs, tests, y = [], [], [], None
    for spec in args.arms:
        n, o, t, yy = load_arm(spec)
        names.append(n)
        oofs.append(o)
        tests.append(t)
        if y is None:
            y = yy
    assert y is not None
    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "submit_sample.csv")

    X = build_meta(oofs, train)
    X_te = build_meta(tests, test)

    oof = np.zeros(len(y), dtype=float)
    pte = np.zeros(len(tests[0]), dtype=float)
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=42)
    fold_aucs = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        if args.model == "logit":
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xva = scaler.transform(X[va])
            Xte = scaler.transform(X_te)
            clf = LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs")
            clf.fit(Xtr, y[tr])
            oof[va] = clf.predict_proba(Xva)[:, 1]
            pte += clf.predict_proba(Xte)[:, 1] / args.folds
        else:
            dtr = lgb.Dataset(X[tr], label=y[tr])
            dva = lgb.Dataset(X[va], label=y[va], reference=dtr)
            params = dict(
                objective="binary",
                metric="auc",
                learning_rate=0.05,
                num_leaves=16,
                min_data_in_leaf=80,
                feature_fraction=0.9,
                bagging_fraction=0.9,
                bagging_freq=1,
                lambda_l2=5.0,
                verbosity=-1,
                seed=42 + fold,
            )
            model = lgb.train(
                params,
                dtr,
                num_boost_round=800,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            oof[va] = model.predict(X[va], num_iteration=model.best_iteration)
            pte += model.predict(X_te, num_iteration=model.best_iteration) / args.folds
        auc = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(auc)
        print(f"stack {args.model} fold={fold} auc={auc:.5f}", flush=True)

    nested = float(roc_auc_score(y, oof))
    metrics = {
        "experiment_id": f"b6pro_stack_{args.model}_{'_'.join(names)}",
        "protocol_id": "IA-AUC715-B6PRO-v1",
        "arm_names": names,
        "arm_aucs": {n: float(roc_auc_score(y, a)) for n, a in zip(names, oofs)},
        "model": args.model,
        "nested_oof_auc": nested,
        "pooled_oof_auc": nested,
        "fold_aucs": fold_aucs,
        "gate_0_715": nested >= TARGET,
        "gap_to_0_715": round(TARGET - nested, 6),
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "no_oof_weight_search": True,
            "nested_stacking": True,
            "meta_features_from_frozen_arm_oof": True,
            "new_data_only": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "predictions.npz", y=y, oof=oof, test=pte)
    from insurance_claim.model import build_submission

    build_submission(test, sample, pte, args.output_dir / "submission_b6pro.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"nested_oof_auc": nested, "gate_0_715": metrics["gate_0_715"], "gap_to_0_715": metrics["gap_to_0_715"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
