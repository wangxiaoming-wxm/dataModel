#!/usr/bin/env python3
"""TabPFN heterogeneous arm fused with B7 max3 (honest nested)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.ebm_arm import build_ebm_features
from insurance_claim.model import build_submission

TARGET = 0.71


def to_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    out = df.copy()
    cat_idx = []
    cols = list(out.columns)
    for i, c in enumerate(cols):
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype("category").cat.codes.astype(np.int32)
            cat_idx.append(i)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out.to_numpy(dtype=np.float32), cat_idx


def main() -> int:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train["label"].astype(int).to_numpy()
    raw = train.drop(columns=["label"])
    seeds = [2026]  # TabPFN heavy; 1 seed × 5 folds first
    oofs, tests = [], []
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(raw, y)):
            Xtr = build_ebm_features(raw.iloc[tr].reset_index(drop=True))
            Xva = build_ebm_features(raw.iloc[va].reset_index(drop=True))
            Xte = build_ebm_features(test.copy())
            # align columns
            Xva = Xva.reindex(columns=Xtr.columns)
            Xte = Xte.reindex(columns=Xtr.columns)
            Mtr, cats = to_matrix(Xtr)
            Mva, _ = to_matrix(Xva)
            Mte, _ = to_matrix(Xte)
            clf = TabPFNClassifier(
                device="cpu",
                n_estimators=4,
                ignore_pretraining_limits=True,
                random_state=seed + fold,
                n_preprocessing_jobs=2,
                categorical_features_indices=cats,
            )
            # subsample train if too large for memory: keep 4000 stratified
            rng = np.random.RandomState(seed + fold)
            if len(Mtr) > 4000:
                pos = np.where(y[tr] == 1)[0]
                neg = np.where(y[tr] == 0)[0]
                n_pos = min(len(pos), 800)
                n_neg = 4000 - n_pos
                sel = np.concatenate([
                    rng.choice(pos, n_pos, replace=False),
                    rng.choice(neg, n_neg, replace=False),
                ])
                rng.shuffle(sel)
                Mfit, yfit = Mtr[sel], y[tr][sel]
            else:
                Mfit, yfit = Mtr, y[tr]
            print(f"tabpfn fit fold={fold} n={len(Mfit)} d={Mfit.shape[1]}", flush=True)
            clf.fit(Mfit, yfit)
            oof[va] = clf.predict_proba(Mva)[:, 1]
            pte += clf.predict_proba(Mte)[:, 1] / 5
            print(f"tabpfn seed={seed} fold={fold} auc={roc_auc_score(y[va], oof[va]):.5f}", flush=True)
        print(f"tabpfn seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
        oofs.append(oof)
        tests.append(pte)

    oof = np.mean(np.vstack(oofs), 0)
    te = np.mean(np.vstack(tests), 0)
    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    print("solo", roc_auc_score(y, oof), "corr", np.corrcoef(oof, 0.5 * (b7["gap"] + b7["gap_bag"]))[0, 1])
    fused = nested_select_rule(y, [b7["gap"], b7["gap_bag"], b7["plus"], oof])
    print("nested", fused["nested_oof_auc"], fused["selected_rule"], fused["full_data_scores"])
    tp = apply_rule(fused["selected_rule"], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], te])
    out = Path("artifacts/b6pro_tabpfn")
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "predictions.npz", y=y, oof=fused["nested_oof"], test=tp, oof_tabpfn=oof, test_tabpfn=te)
    build_submission(test, sample, tp, out / "submission_b6pro.csv")
    metrics = {
        "experiment_id": "b6pro_tabpfn",
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
            "tabpfn_subsample_train4000": True,
            "no_oof_weight_search": True,
            "new_data_only": True,
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ["oof_auc", "nested_oof_auc", "gate_0_71", "gap_to_0_71"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
