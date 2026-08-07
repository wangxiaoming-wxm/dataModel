"""Quick 1-seed ablation for B6v2 main/hetero + hyperparam variants."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6_builders import build_hetero, build_main
from insurance_claim.model import TARGET
from insurance_claim.train_b5_focus import CAT_PARAMS, N_SPLITS, build_b5
from insurance_claim.train_b6 import build_gap, THREAD_COUNT

SEED = 2026
OUT = Path("artifacts/b6_v2_s1")


def run(name, builder, params, use_best=True):
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = train[TARGET].astype(int)
    feats = train.drop(columns=[TARGET])
    oof = np.zeros(len(train))
    t0 = time.time()
    for fold, (a, b) in enumerate(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(feats, y)):
        Xtr, Xva = feats.iloc[a].reset_index(drop=True), feats.iloc[b].reset_index(drop=True)
        ytr, yva = y.iloc[a].reset_index(drop=True), y.iloc[b].reset_index(drop=True)
        tr, va, te, cats = builder(Xtr, Xva, test.copy())
        p = dict(params)
        p["random_seed"] = SEED + fold
        p["thread_count"] = THREAD_COUNT
        m = CatBoostClassifier(**p)
        if use_best:
            m.fit(tr, ytr, eval_set=(va, yva), cat_features=cats, use_best_model=True, verbose=False)
            best = m.get_best_iteration()
        else:
            m.fit(tr, ytr, cat_features=cats, verbose=False)
            best = p.get("iterations")
        oof[b] = m.predict_proba(va)[:, 1]
        print(f"{name} fold={fold} auc={roc_auc_score(yva, oof[b]):.5f} best={best} n={tr.shape[1]}", flush=True)
    auc = float(roc_auc_score(y, oof))
    print(f"{name} OOF={auc:.6f} sec={time.time()-t0:.1f}", flush=True)
    return auc, oof


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = {**dict(CAT_PARAMS), "thread_count": THREAD_COUNT}
    deep = {**base, "depth": 7, "l2_leaf_reg": 12}
    slow = {**base, "learning_rate": 0.02, "iterations": 2000, "od_wait": 180}
    results = {}
    oofs = {}
    for name, builder, params in [
        ("b5", build_b5, base),
        ("gap", build_gap, base),
        ("main", build_main, base),
        ("hetero", build_hetero, base),
        ("main_deep", build_main, deep),
        ("main_slow", build_main, slow),
        ("hetero_deep", build_hetero, deep),
    ]:
        auc, oof = run(name, builder, params)
        results[name] = auc
        oofs[name] = oof
        np.save(OUT / f"oof_{name}.npy", oof)

    # pre-registered style equal fusions of interest
    for combo in [
        ("b5", "gap"),
        ("b5", "main"),
        ("gap", "main"),
        ("main", "hetero"),
        ("gap", "hetero"),
        ("b5", "gap", "main"),
        ("main", "hetero", "gap"),
    ]:
        arrs = [oofs[c] for c in combo]
        mean = np.mean(np.vstack(arrs), 0)
        rank = np.mean(np.vstack([rankdata(a) for a in arrs]), 0)
        results["mean_" + "+".join(combo)] = float(roc_auc_score(pd.read_csv("train.csv")[TARGET], mean))
        results["rank_" + "+".join(combo)] = float(roc_auc_score(pd.read_csv("train.csv")[TARGET], rank))
        print(f"mean {'+'.join(combo)}={results['mean_'+'+'.join(combo)]:.6f}", flush=True)
        print(f"rank {'+'.join(combo)}={results['rank_'+'+'.join(combo)]:.6f}", flush=True)

    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
