#!/usr/bin/env python3
"""Midband residual expert: train CatBoost on rows where main/plus disagree.

Fold-local; no TE. Used as third arm for nested fusion.
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

from insurance_claim.v10_plus.plus_features import build_plus

N_SPLITS = 5
PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=12,
    random_strength=1.0,
    od_type="Iter",
    od_wait=120,
    verbose=False,
    thread_count=8,
    allow_writing_files=False,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-npz", type=Path, required=True)
    ap.add_argument("--plus-npz", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_resid"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    ap.add_argument("--band", type=float, nargs=2, default=[0.05, 0.35])
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    y = train["label"].astype(int).to_numpy()
    main = np.load(args.main_npz)
    plus = np.load(args.plus_npz)
    main_oof = main["oof_main"] if "oof_main" in main.files else main["oof"]
    plus_oof = plus["oof"]
    # disagreement / midband mask for analysis only (fit still uses all rows with sample_weight)
    disagree = np.abs(main_oof - plus_oof)
    mid = (main_oof > args.band[0]) & (main_oof < args.band[1])
    w_base = 1.0 + 3.0 * (disagree / (disagree.max() + 1e-9)) * mid.astype(float)

    features = train.drop(columns=["label"])
    oof_by_seed = {}
    test_by_seed = {}
    for seed in args.seeds:
        oof = np.zeros(len(train), dtype=float)
        pte = np.zeros(len(test), dtype=float)
        for fold, (tr, va) in enumerate(
            StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(features, y)
        ):
            Xtr = features.iloc[tr].reset_index(drop=True)
            Xva = features.iloc[va].reset_index(drop=True)
            ytr = y[tr]
            yva = y[va]
            wtr = w_base[tr]
            tr_x, va_x, te_x, cats = build_plus(Xtr, Xva, test.copy())
            p = dict(PARAMS)
            p["random_seed"] = seed + fold
            model = CatBoostClassifier(**p)
            model.fit(
                tr_x,
                ytr,
                sample_weight=wtr,
                eval_set=(va_x, yva),
                cat_features=cats,
                use_best_model=True,
            )
            oof[va] = model.predict_proba(va_x)[:, 1]
            pte += model.predict_proba(te_x)[:, 1] / N_SPLITS
            print(
                f"resid seed={seed} fold={fold} auc={roc_auc_score(yva, oof[va]):.5f}",
                flush=True,
            )
        print(f"resid seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
        oof_by_seed[seed] = oof
        test_by_seed[seed] = pte

    oof = np.mean(np.vstack(list(oof_by_seed.values())), axis=0)
    te = np.mean(np.vstack(list(test_by_seed.values())), axis=0)
    metrics = {
        "experiment_id": "b6pro_resid_weighted",
        "protocol_id": "IA-AUC715-B6PRO-v1",
        "oof_auc": float(roc_auc_score(y, oof)),
        "seed_aucs": {str(s): float(roc_auc_score(y, oof_by_seed[s])) for s in args.seeds},
        "corr_vs_main": float(np.corrcoef(oof, main_oof)[0, 1]),
        "corr_vs_plus": float(np.corrcoef(oof, plus_oof)[0, 1]),
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "no_oof_weight_search": True,
            "sample_weight_from_oof_disagreement": True,
            "note": "sample weights from frozen main/plus OOF disagreement; not continuous fusion search",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "predictions.npz", y=y, oof=oof, test=te)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"oof_auc": metrics["oof_auc"], **{k: metrics[k] for k in ("corr_vs_main", "corr_vs_plus")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
