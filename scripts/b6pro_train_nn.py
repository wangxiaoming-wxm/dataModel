#!/usr/bin/env python3
"""Quick RealMLP / TabM hetero arm for B6pro (fold-local; no TE)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.ebm_arm import build_ebm_features

TARGET = "label"


def to_xy(df: pd.DataFrame):
    feats = build_ebm_features(df)
    # pytabkit prefers DataFrame with mixed types
    return feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["realmlp", "tabm"], default="realmlp")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_realmlp"))
    ap.add_argument("--epochs", type=int, default=128)
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    y = train[TARGET].astype(int)
    X_all = to_xy(train)
    X_te = to_xy(test)

    oof_by_seed = {}
    test_by_seed = {}
    folds_meta = []
    t0 = time.time()

    for seed in args.seeds:
        oof = np.zeros(len(train), dtype=float)
        pte = np.zeros(len(test), dtype=float)
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X_all, y)):
            Xtr, ytr = X_all.iloc[tr], y.iloc[tr]
            Xva, yva = X_all.iloc[va], y.iloc[va]
            if args.family == "realmlp":
                model = RealMLP_TD_Classifier(
                    n_epochs=args.epochs,
                    device="cpu",
                    n_threads=4,
                    random_state=seed + fold,
                )
            else:
                model = TabM_D_Classifier(
                    n_epochs=args.epochs,
                    device="cpu",
                    n_threads=4,
                    random_state=seed + fold,
                )
            model.fit(Xtr, ytr.to_numpy())
            # predict_proba
            if hasattr(model, "predict_proba"):
                pv = model.predict_proba(Xva)[:, 1]
                pt = model.predict_proba(X_te)[:, 1]
            else:
                pv = np.asarray(model.predict(Xva), dtype=float)
                pt = np.asarray(model.predict(X_te), dtype=float)
                if pv.ndim > 1:
                    pv = pv[:, 1]
                    pt = pt[:, 1]
            oof[va] = pv
            pte += pt / args.folds
            auc = float(roc_auc_score(yva, pv))
            folds_meta.append({"seed": seed, "fold": fold, "valid_auc": auc})
            print(f"{args.family} seed={seed} fold={fold} auc={auc:.5f}", flush=True)
        print(f"{args.family} seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
        oof_by_seed[seed] = oof
        test_by_seed[seed] = pte

    oof = np.mean(np.vstack(list(oof_by_seed.values())), axis=0)
    te = np.mean(np.vstack(list(test_by_seed.values())), axis=0)
    metrics = {
        "experiment_id": f"b6pro_{args.family}",
        "protocol_id": "IA-AUC715-B6PRO-v1",
        "family": args.family,
        "seeds": args.seeds,
        "n_splits": args.folds,
        "oof_auc": float(roc_auc_score(y, oof)),
        "seed_aucs": {str(s): float(roc_auc_score(y, oof_by_seed[s])) for s in args.seeds},
        "folds": folds_meta,
        "elapsed_sec": round(time.time() - t0, 1),
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "no_oof_weight_search": True,
            "new_data_only": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y=y.to_numpy(),
        oof=oof,
        test=te,
        **{f"oof_seed_{s}": oof_by_seed[s] for s in args.seeds},
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"oof_auc": metrics["oof_auc"], "seed_aucs": metrics["seed_aucs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
