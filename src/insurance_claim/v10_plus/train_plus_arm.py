#!/usr/bin/env python3
"""root_plus 配方全量训练：5 折 × 4 seed，门禁 + 提交。
配方由锁定 holdout 选型（0.67438 > root 0.66502），TE 变体已淘汰。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "shared"))
from features import (  # noqa: E402
    extra_features,
    parse_frame,
    pca_features,
    prepare,
    root_features,
)
from gate_tools import bootstrap_ci_delta, days_stress_auc  # noqa: E402

DATA = Path("/Volumes/pssd/app/ml/正式比赛/data")
OUT = ROOT / "out"
PARAMS = dict(
    loss_function="Logloss", eval_metric="AUC",
    iterations=1500, learning_rate=0.03, depth=6,
    l2_leaf_reg=10, random_strength=0.7,
    od_type="Iter", od_wait=100,
    verbose=False, thread_count=6, allow_writing_files=False,
)
SEEDS = [2026, 2027, 2028, 2029]


def build_plus(Xtr, ytr, Xva, Xte, use_pca=False):
    tr, va, te, cat_names = root_features(Xtr, Xva, Xte)
    tr, va, te, extra_cats = extra_features(tr, va, te, Xtr, Xva, Xte)
    cat_names = cat_names + extra_cats
    if use_pca:
        tr, va, te = pca_features(tr, va, te, Xtr, Xva, Xte)
    return prepare(tr, va, te, cat_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2026,2027,2028,2029")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--config", default="h1")
    ap.add_argument("--tag", default="plus")
    ap.add_argument("--pca", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "submit_sample.csv")
    y = train["label"].astype(int).reset_index(drop=True)
    X = parse_frame(train.drop(columns=["id", "label", "x19"]))
    X_te = parse_frame(test.drop(columns=["id", "x19"]))
    assert list(sample["id"]) == list(test["id"])

    base = np.load(ROOT.parent / "shared" / "baseline_a.npz")
    A_oof, A_auc = base["oof"], float(base["auc"])
    params = dict(PARAMS)
    if args.config == "h2":
        params.update(iterations=2500, learning_rate=0.02, depth=7,
                      l2_leaf_reg=20, random_strength=1.0, od_wait=150)
    elif args.config == "h3":
        params.update(iterations=3000, learning_rate=0.015, depth=8,
                      l2_leaf_reg=30, random_strength=2.0, od_wait=200)
    oof_by_seed, test_by_seed = {}, {}
    t0 = __import__("time").time()
    for seed in seeds:
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=seed)
        oof = np.zeros(len(X), dtype=float)
        pte = np.zeros(len(X_te), dtype=float)
        for fold, (a, b) in enumerate(skf.split(X, y)):
            Xtr, ytr = X.iloc[a], y.iloc[a]
            Xva, yva = X.iloc[b], y.iloc[b]
            tr, va, te, cat_idx = build_plus(Xtr, ytr, Xva, X_te,
                                             use_pca=args.pca)
            p = dict(params)
            p["random_seed"] = seed + fold
            clf = CatBoostClassifier(**p)
            clf.fit(tr, ytr, eval_set=(va, yva), cat_features=cat_idx,
                    use_best_model=True)
            oof[b] = clf.predict_proba(va)[:, 1]
            pte += clf.predict_proba(te)[:, 1] / args.folds
            print(f"[plus] s{seed} f{fold} valid="
                  f"{roc_auc_score(yva, oof[b]):.5f} "
                  f"best={clf.get_best_iteration()}", flush=True)
        oof_by_seed[seed] = oof
        test_by_seed[seed] = pte
        print(f"[plus] s{seed} OOF={roc_auc_score(y, oof):.5f}", flush=True)

    oof_pool = np.mean(np.vstack([oof_by_seed[s] for s in seeds]), axis=0)
    test_pool = np.mean(np.vstack([test_by_seed[s] for s in seeds]), axis=0)
    B_auc = float(roc_auc_score(y, oof_pool))
    corr = float(np.corrcoef(A_oof, oof_pool)[0, 1])
    lo, hi = bootstrap_ci_delta(y, A_oof, oof_pool)
    a_d, b_d = days_stress_auc(y, train["days"], A_oof, oof_pool)
    gain = B_auc - A_auc
    gates = {
        "gain_ge_0.0015": gain >= 0.0015,
        "boot_ci_lower_gt_0": lo > 0,
        "days_stress": b_d >= a_d - 0.01,
    }
    print(f"[{args.tag}] pooled={B_auc:.5f} gain={gain:+.5f} corr={corr:.4f} "
          f"boot=({lo:.5f},{hi:.5f}) days A={a_d:.5f} B={b_d:.5f} "
          f"gates={gates}", flush=True)

    meta = {
        "pooled_oof_auc": B_auc, "gain": gain, "corr_vs_A": corr,
        "boot_ci": [lo, hi], "days_stress": {"A": a_d, "B": b_d},
        "gates": gates,
        "per_seed_oof": {s: float(roc_auc_score(y, oof_by_seed[s]))
                         for s in seeds},
        "elapsed_sec": round(t0 - __import__("time").time(), 1),
    }
    (OUT / f"meta_{args.tag}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    np.savez(OUT / f"oof_{args.tag}.npz", y=y.to_numpy(), oof=oof_pool,
             test=test_pool,
             **{f"oof_seed_{s}": oof_by_seed[s] for s in seeds})
    np.save(OUT / f"test_{args.tag}.npy", test_pool)
    sub = sample.copy()
    sub.iloc[:, 1] = rankdata(test_pool, method="average") / (len(test_pool) + 1)
    sub.to_csv(OUT / f"submission_{args.tag}.csv", index=False)
    print(f"[{args.tag}] saved submission_{args.tag}.csv "
          f"mean={sub.iloc[:,1].mean():.5f} "
          f"nan={sub.iloc[:,1].isna().sum()}", flush=True)


if __name__ == "__main__":
    main()
