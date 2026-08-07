"""B7 plus arm trainer (V10 H2 recipe; fold-local; no TE)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.model import TARGET, audit_data, build_submission
from insurance_claim.v10_plus.plus_features import build_plus_mine

N_SPLITS_DEFAULT = 10
SEEDS_DEFAULT = (2026, 2027, 2028, 2029)
THREAD_COUNT = 8

PARAMS_H2 = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=2500,
    learning_rate=0.02,
    depth=7,
    l2_leaf_reg=20,
    random_strength=1.0,
    od_type="Iter",
    od_wait=150,
    verbose=False,
    thread_count=THREAD_COUNT,
    allow_writing_files=False,
)


def run_plus(train, test, y, seeds, n_splits, params):
    # V10 drops id/label/x19 before parse
    feats = train.drop(columns=[TARGET])
    oof_by_seed, test_by_seed, folds = {}, {}, []
    for seed in seeds:
        oof = np.zeros(len(train), dtype=float)
        pte = np.zeros(len(test), dtype=float)
        for fold, (a, b) in enumerate(
            StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(feats, y)
        ):
            Xtr = feats.iloc[a].reset_index(drop=True)
            Xva = feats.iloc[b].reset_index(drop=True)
            ytr = y.iloc[a].reset_index(drop=True)
            yva = y.iloc[b].reset_index(drop=True)
            tr, va, te, cats = build_plus_mine(Xtr, Xva, test.copy())
            p = dict(params)
            p["random_seed"] = seed + fold
            m = CatBoostClassifier(**p)
            m.fit(tr, ytr, eval_set=(va, yva), cat_features=cats, use_best_model=True)
            oof[b] = m.predict_proba(va)[:, 1]
            pte += m.predict_proba(te)[:, 1] / n_splits
            auc = float(roc_auc_score(yva, oof[b]))
            folds.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "valid_auc": auc,
                    "best_iter": int(m.get_best_iteration() or -1),
                    "n_features": int(tr.shape[1]),
                }
            )
            print(
                f"plus_mine seed={seed} fold={fold} auc={auc:.5f} best={m.get_best_iteration()} n={tr.shape[1]}",
                flush=True,
            )
        print(f"plus_mine seed={seed} OOF={roc_auc_score(y, oof):.6f}", flush=True)
        oof_by_seed[seed] = oof
        test_by_seed[seed] = pte
    oof = np.mean(np.vstack(list(oof_by_seed.values())), 0)
    te = np.mean(np.vstack(list(test_by_seed.values())), 0)
    return {
        "oof": oof,
        "test": te,
        "oof_by_seed": oof_by_seed,
        "test_by_seed": test_by_seed,
        "oof_auc": float(roc_auc_score(y, oof)),
        "seed_aucs": {str(s): float(roc_auc_score(y, oof_by_seed[s])) for s in seeds},
        "folds": folds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b7_plus_mine"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    ap.add_argument("--folds", type=int, default=N_SPLITS_DEFAULT)
    args = ap.parse_args()
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    audit_data(train, test, sample)
    y = train[TARGET].astype(int)
    t0 = time.time()
    res = run_plus(train, test, y, tuple(args.seeds), args.folds, PARAMS_H2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        oof=res["oof"],
        test=res["test"],
        y=y.to_numpy(),
        **{f"oof_{s}": res["oof_by_seed"][s] for s in args.seeds},
        **{f"test_{s}": res["test_by_seed"][s] for s in args.seeds},
    )
    build_submission(test, sample, res["test"], args.output_dir / "submission_plus.csv")
    metrics = {
        "experiment_id": "b7_plus_mine_h2",
        "oof_auc": res["oof_auc"],
        "seed_aucs": res["seed_aucs"],
        "seeds": list(args.seeds),
        "n_splits": args.folds,
        "params": {k: v for k, v in PARAMS_H2.items() if k != "verbose"},
        "elapsed_sec": round(time.time() - t0, 1),
        "folds": res["folds"],
        "target_encoding": "none",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"oof_auc": res["oof_auc"], "seed_aucs": res["seed_aucs"]}, indent=2))


if __name__ == "__main__":
    main()
