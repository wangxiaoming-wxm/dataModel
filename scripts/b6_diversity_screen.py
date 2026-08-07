"""B6 diversity arms: LightGBM on gap cats + numeric; CatBoost gap hyperparam variants.

1-seed screen then optional multi-seed. Protocol: fold-local FE, no TE.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6_gap_features import GAP_CAT_COLS, add_gap_cats, fit_gap_edges
from insurance_claim.model import TARGET
from insurance_claim.train_b5_focus import CAT_PARAMS, N_SPLITS, build_b5, enrich
from insurance_claim.train_b6 import THREAD_COUNT, build_gap

SEED = 2026
OUT = Path("artifacts/b6_div_s1")


def build_lgb_matrix(X_tr, X_va, X_te):
    """Compressed numeric + gap cats (as codes) for LightGBM."""
    edges = fit_gap_edges(X_tr)
    tr_b5, va_b5, te_b5, cats_b5 = build_b5(X_tr, X_va, X_te)

    def gap_frame(raw):
        return add_gap_cats(enrich(raw), edges).loc[:, list(GAP_CAT_COLS)].copy()

    gtr, gva, gte = gap_frame(X_tr), gap_frame(X_va), gap_frame(X_te)

    # Keep a compact numeric set from B5 frame
    num_prefer = [
        c
        for c in tr_b5.columns
        if c not in cats_b5 and pd.api.types.is_numeric_dtype(tr_b5[c])
    ]
    # Cap to avoid huge sparsity: take all numerics (usually ~70)

    def pack(base, gap, cats):
        out = base[num_prefer].copy()
        for c in GAP_CAT_COLS:
            # category codes fold-local from train mapping
            out[c] = gap[c].astype(str)
        return out

    tr = pack(tr_b5, gtr, cats_b5)
    va = pack(va_b5, gva, cats_b5).reindex(columns=tr.columns)
    te = pack(te_b5, gte, cats_b5).reindex(columns=tr.columns)

    cat_cols = list(GAP_CAT_COLS)
    # factorize using train categories
    for c in cat_cols:
        cats = pd.Index(tr[c].astype(str).unique())
        mapping = {k: i for i, k in enumerate(cats)}
        tr[c] = tr[c].astype(str).map(mapping).fillna(-1).astype(int)
        va[c] = va[c].astype(str).map(mapping).fillna(-1).astype(int)
        te[c] = te[c].astype(str).map(mapping).fillna(-1).astype(int)
    for c in num_prefer:
        med = float(pd.to_numeric(tr[c], errors="coerce").median())
        tr[c] = pd.to_numeric(tr[c], errors="coerce").fillna(med)
        va[c] = pd.to_numeric(va[c], errors="coerce").fillna(med)
        te[c] = pd.to_numeric(te[c], errors="coerce").fillna(med)
    return tr, va, te, cat_cols


def run_cb(name, builder, params):
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
        m.fit(tr, ytr, eval_set=(va, yva), cat_features=cats, use_best_model=True, verbose=False)
        oof[b] = m.predict_proba(va)[:, 1]
        print(f"{name} fold={fold} auc={roc_auc_score(yva, oof[b]):.5f} best={m.get_best_iteration()} n={tr.shape[1]}", flush=True)
    auc = float(roc_auc_score(y, oof))
    print(f"{name} OOF={auc:.6f} sec={time.time()-t0:.1f}", flush=True)
    return auc, oof


def run_lgb():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = train[TARGET].astype(int)
    feats = train.drop(columns=[TARGET])
    oof = np.zeros(len(train))
    t0 = time.time()
    for fold, (a, b) in enumerate(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(feats, y)):
        Xtr, Xva = feats.iloc[a].reset_index(drop=True), feats.iloc[b].reset_index(drop=True)
        ytr, yva = y.iloc[a].reset_index(drop=True), y.iloc[b].reset_index(drop=True)
        tr, va, te, cats = build_lgb_matrix(Xtr, Xva, test.copy())
        m = LGBMClassifier(
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            reg_alpha=0.5,
            min_child_samples=40,
            objective="binary",
            n_jobs=THREAD_COUNT,
            random_state=SEED + fold,
            verbose=-1,
        )
        m.fit(
            tr,
            ytr,
            eval_set=[(va, yva)],
            categorical_feature=cats,
            callbacks=[early_stopping(120), log_evaluation(0)],
        )
        oof[b] = m.predict_proba(va)[:, 1]
        print(f"lgb fold={fold} auc={roc_auc_score(yva, oof[b]):.5f} best={m.best_iteration_} n={tr.shape[1]}", flush=True)
    auc = float(roc_auc_score(y, oof))
    print(f"lgb OOF={auc:.6f} sec={time.time()-t0:.1f}", flush=True)
    return auc, oof


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = {**dict(CAT_PARAMS), "thread_count": THREAD_COUNT}
    deep = {**base, "depth": 7, "l2_leaf_reg": 12}
    slow = {**base, "learning_rate": 0.02, "iterations": 2000, "od_wait": 180}
    bag = {**base, "bagging_temperature": 0.5, "random_strength": 1.0}

    results = {}
    oofs = {}
    # reuse known
    results["b5_ref"] = 0.690548
    results["gap_ref"] = 0.691836

    for name, builder, params in [
        ("gap_deep", build_gap, deep),
        ("gap_slow", build_gap, slow),
        ("gap_bag", build_gap, bag),
    ]:
        auc, oof = run_cb(name, builder, params)
        results[name] = auc
        oofs[name] = oof
        np.save(OUT / f"oof_{name}.npy", oof)

    auc, oof = run_lgb()
    results["lgb"] = auc
    oofs["lgb"] = oof
    np.save(OUT / "oof_lgb.npy", oof)

    # load gap ref oof if present from prior screen
    gap_path = Path("artifacts/b6_s1_screen/predictions.npz")
    y = pd.read_csv("train.csv")[TARGET].astype(int)
    if gap_path.exists():
        z = np.load(gap_path)
        oofs["gap"] = z["oof_gap"] if "oof_gap" in z.files else z["oof"]
        # b6_s1 has oof_gap
        if "oof_gap" in z.files:
            oofs["gap"] = z["oof_gap"]
        if "oof_b5" in z.files:
            oofs["b5"] = z["oof_b5"]
        results["gap"] = float(roc_auc_score(y, oofs["gap"]))
        if "b5" in oofs:
            results["b5"] = float(roc_auc_score(y, oofs["b5"]))

    for combo in [
        ("gap", "gap_deep"),
        ("gap", "gap_slow"),
        ("gap", "gap_bag"),
        ("gap", "lgb"),
        ("b5", "gap", "lgb"),
        ("gap", "gap_slow", "lgb"),
    ]:
        if not all(c in oofs for c in combo):
            continue
        arrs = [oofs[c] for c in combo]
        mean = np.mean(np.vstack(arrs), 0)
        rank = np.mean(np.vstack([rankdata(a) for a in arrs]), 0)
        results["mean_" + "+".join(combo)] = float(roc_auc_score(y, mean))
        results["rank_" + "+".join(combo)] = float(roc_auc_score(y, rank))
        print(f"mean {'+'.join(combo)}={results['mean_'+'+'.join(combo)]:.6f}", flush=True)

    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
