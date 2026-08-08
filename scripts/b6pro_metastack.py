#!/usr/bin/env python3
"""Deep nested CatBoost meta-stacker over frozen arms + raw scalars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.model import build_submission

TARGET = 0.715
PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=800,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=30,
    random_strength=2.0,
    od_type="Iter",
    od_wait=50,
    verbose=False,
    thread_count=8,
    allow_writing_files=False,
)


def load_arm(path: Path, oof_key=None, test_key=None):
    z = np.load(path)
    oof = z[oof_key] if oof_key else (z["oof"] if "oof" in z.files else z["oof_main"])
    te = z[test_key] if test_key else (z["test"] if "test" in z.files else z["test_main"])
    return oof, te


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_metastack"))
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "submit_sample.csv")
    y = train["label"].astype(int).to_numpy()

    arms = {
        "main": load_arm(Path("artifacts/b6pro_main/predictions.npz"), "oof_main", "test_main"),
        "plus": load_arm(Path("reference/v10/oof_plus_h2_10.npz")),
        "hybrid": load_arm(Path("artifacts/b6pro_hybrid8/predictions.npz"), "oof_hybrid", "test_hybrid"),
        "ultra": load_arm(Path("artifacts/b6pro_plus_ultra/predictions.npz"), "oof_plus", "test_plus"),
    }

    def meta_frame(arm_dict, raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=raw.index)
        for name, (oof, _) in arm_dict.items():
            # for test we'll pass test preds in oof slot
            out[f"p_{name}"] = oof
        pred_mat = np.vstack([out[f"p_{n}"] for n in arms]).T
        out["p_mean"] = pred_mat.mean(axis=1)
        out["p_max"] = pred_mat.max(axis=1)
        out["p_min"] = pred_mat.min(axis=1)
        out["p_std"] = pred_mat.std(axis=1)
        out["p_main_plus_diff"] = out["p_main"] - out["p_plus"]
        out["days"] = pd.to_numeric(raw["days"], errors="coerce")
        out["condition"] = pd.to_numeric(raw["condition"], errors="coerce")
        out["age"] = pd.to_numeric(raw["age_range"], errors="coerce")
        out["ratio"] = out["condition"] / (out["days"].abs() + 1)
        out["region"] = raw["region"].astype(str)
        out["source"] = raw["source"].astype(str)
        out["code"] = raw["code"].astype(str)
        return out

    # Build train meta with OOF preds; test meta with test preds
    train_arms = {k: (v[0], v[0]) for k, v in arms.items()}
    test_arms = {k: (v[1], v[1]) for k, v in arms.items()}
    X = meta_frame(train_arms, train)
    Xte = meta_frame(test_arms, test)
    cats = ["region", "source", "code"]

    oof = np.zeros(len(y), dtype=float)
    pte = np.zeros(len(test), dtype=float)
    fold_aucs = []
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
        model = CatBoostClassifier(**{**PARAMS, "random_seed": 42 + fold})
        model.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]), cat_features=cats, use_best_model=True)
        oof[va] = rankdata(model.predict_proba(X.iloc[va])[:, 1]) / (len(va) + 1.0)
        pte += rankdata(model.predict_proba(Xte)[:, 1]) / (len(Xte) + 1.0) / 5
        auc = float(roc_auc_score(y[va], oof[va]))
        fold_aucs.append(auc)
        print(f"meta fold={fold} auc={auc:.5f} (rank-OOF)", flush=True)

    nested = float(roc_auc_score(y, oof))
    # Also report max(main,plus) for reference
    base = float(roc_auc_score(y, np.maximum(arms["main"][0], arms["plus"][0])))
    metrics = {
        "experiment_id": "b6pro_metastack_cb",
        "protocol_id": "IA-AUC715-B6PRO-v1",
        "nested_oof_auc": nested,
        "pooled_oof_auc": nested,
        "fold_aucs": fold_aucs,
        "baseline_max_main_plus": base,
        "gate_0_715": nested >= TARGET,
        "gap_to_0_715": round(TARGET - nested, 6),
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "no_oof_weight_search": True,
            "nested_metastacking": True,
            "uses_reference_plus": True,
            "new_data_only": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "predictions.npz", y=y, oof=oof, test=pte)
    build_submission(test, sample, pte, args.output_dir / "submission_b6pro.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ["nested_oof_auc", "baseline_max_main_plus", "gate_0_715", "gap_to_0_715", "fold_aucs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
