#!/usr/bin/env python3
"""TabM / RealMLP heterogeneous arm fused with B7 max3 arms (honest nested)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import nested_select_rule, apply_rule
from insurance_claim.ebm_arm import build_ebm_features
from insurance_claim.model import build_submission
from insurance_claim.realmlp_arm import prepare_fold_frames

TARGET = 0.71


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["tabm", "realmlp"], default="tabm")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_tabm"))
    args = ap.parse_args()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train["label"].astype(int)
    raw = train.drop(columns=["label"])
    raw_te = test.copy()

    oofs, tests = [], []
    for seed in args.seeds:
        oof = np.zeros(len(y), dtype=float)
        pte = np.zeros(len(test), dtype=float)
        for fold, (tr, va) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=seed).split(raw, y)
        ):
            Xtr = build_ebm_features(raw.iloc[tr].reset_index(drop=True))
            Xva = build_ebm_features(raw.iloc[va].reset_index(drop=True))
            Xte = build_ebm_features(raw_te)
            Xtr, Xva, Xte, _ = prepare_fold_frames(Xtr, Xva, Xte)
            tmp = Path(f"/tmp/{args.family}_s{seed}_f{fold}")
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            if args.family == "tabm":
                model = TabM_D_Classifier(
                    device="cpu",
                    random_state=seed + fold,
                    n_cv=1,
                    n_refit=0,
                    n_repeats=1,
                    val_fraction=0.15,
                    n_threads=4,
                    tmp_folder=tmp,
                    verbosity=0,
                    n_epochs=args.epochs,
                    patience=25,
                    batch_size=256,
                    compile_model=False,
                    allow_amp=False,
                    val_metric_name="cross_entropy",
                )
            else:
                model = RealMLP_TD_Classifier(
                    device="cpu",
                    random_state=seed + fold,
                    n_cv=1,
                    n_refit=0,
                    n_repeats=1,
                    val_fraction=0.15,
                    n_threads=4,
                    tmp_folder=tmp,
                    verbosity=0,
                    n_epochs=args.epochs,
                    patience=25,
                    batch_size=256,
                    compile_model=False,
                    allow_amp=False,
                    val_metric_name="cross_entropy",
                )
            model.fit(Xtr, y.iloc[tr].to_numpy())
            oof[va] = model.predict_proba(Xva)[:, 1]
            pte += model.predict_proba(Xte)[:, 1] / 5
            print(
                f"{args.family} seed={seed} fold={fold} auc={roc_auc_score(y.iloc[va], oof[va]):.5f}",
                flush=True,
            )
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"{args.family} seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
        oofs.append(oof)
        tests.append(pte)

    oof = np.mean(np.vstack(oofs), axis=0)
    te = np.mean(np.vstack(tests), axis=0)
    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    arms = [b7["gap"], b7["gap_bag"], b7["plus"], oof]
    fused = nested_select_rule(y.to_numpy(), arms)
    print(
        "solo",
        roc_auc_score(y, oof),
        "corr_main",
        np.corrcoef(oof, 0.5 * (b7["gap"] + b7["gap_bag"]))[0, 1],
        "nested",
        fused["nested_oof_auc"],
        fused["selected_rule"],
    )

    tg, tgb, tp = fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]
    test_pred = apply_rule(fused["selected_rule"], [tg, tgb, tp, te])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y=y.to_numpy(),
        oof=fused["nested_oof"],
        test=test_pred,
        oof_nn=oof,
        test_nn=te,
    )
    build_submission(test, sample, test_pred, args.output_dir / "submission_b6pro.csv")
    metrics = {
        "experiment_id": f"b6pro_{args.family}",
        "family": args.family,
        "oof_auc": float(roc_auc_score(y, oof)),
        "nested_oof_auc": fused["nested_oof_auc"],
        "selected_rule": fused["selected_rule"],
        "full_data_scores": fused["full_data_scores"],
        "baseline_max3": 0.7027049552615718,
        "gate_0_71": fused["nested_oof_auc"] >= TARGET,
        "gap_to_0_71": round(TARGET - fused["nested_oof_auc"], 6),
        "public_b7_signal": {"local": 0.702704955, "public": 0.707},
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "no_oof_weight_search": True,
            "new_data_only": True,
            "reference_plus_bootstrap": True,
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ["oof_auc", "nested_oof_auc", "gate_0_71", "gap_to_0_71"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
