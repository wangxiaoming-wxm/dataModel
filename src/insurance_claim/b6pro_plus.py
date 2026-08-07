"""B6pro plus arm: keep x0-x18 root_plus (fold-local; no TE).

Adapted for workspace paths from v10_plus.plus_features.
Optional injection of B6 gap cats for plus_gap variant.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6_gap_features import GAP_CAT_COLS, add_gap_cats, fit_gap_edges
from insurance_claim.b6pro_plus_ultra import build_plus_ultra
from insurance_claim.v10_plus.plus_features import build_plus

N_SPLITS_DEFAULT = 5
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

PARAMS_H2_FIXED = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=600,
    learning_rate=0.02,
    depth=7,
    l2_leaf_reg=20,
    random_strength=1.0,
    verbose=False,
    thread_count=THREAD_COUNT,
    allow_writing_files=False,
)

PARAMS_H3 = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=3000,
    learning_rate=0.015,
    depth=8,
    l2_leaf_reg=30,
    random_strength=2.0,
    od_type="Iter",
    od_wait=200,
    verbose=False,
    thread_count=THREAD_COUNT,
    allow_writing_files=False,
)


def build_plus_gap(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Plus features + fold-local B6 gap categorical crosses."""
    tr, va, te, cats = build_plus(X_tr, X_va, X_te)
    edges = fit_gap_edges(X_tr)

    def gap_block(raw: pd.DataFrame) -> pd.DataFrame:
        # enrich-lite: gap helpers need days/condition/w/t3/code/...
        base = raw.copy()
        with_gap = add_gap_cats(base, edges)
        return with_gap.loc[:, list(GAP_CAT_COLS)].copy()

    gtr, gva, gte = gap_block(X_tr), gap_block(X_va), gap_block(X_te)

    def merge(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
        out = pd.concat([base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        return out.loc[:, ~out.columns.duplicated()]

    tr = merge(tr, gtr)
    va = merge(va, gva).reindex(columns=tr.columns)
    te = merge(te, gte).reindex(columns=tr.columns)
    cats = list(dict.fromkeys(list(cats) + list(GAP_CAT_COLS)))
    for c in cats:
        for d in (tr, va, te):
            if c in d.columns:
                d[c] = d[c].astype(str).fillna("__MISSING__")
    return tr, va, te, cats


def run_plus_arm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: pd.Series,
    seeds: tuple[int, ...],
    *,
    variant: str = "plus",
    params: dict[str, Any] | None = None,
    n_splits: int = N_SPLITS_DEFAULT,
    oof_transform: str = "prob",
    use_best_model: bool = True,
) -> dict[str, Any]:
    """Train plus or plus_gap arm; return pooled OOF/test and per-seed arrays.

    oof_transform:
      - prob: raw predict_proba (may suffer fold-scale mismatch)
      - rank: within-fold rankdata / (n+1) for OOF and test (stable pooling)
    """
    builder = {
        "plus": build_plus,
        "plus_gap": build_plus_gap,
        "plus_ultra": build_plus_ultra,
    }.get(variant)
    if builder is None:
        raise ValueError(f"unknown plus variant: {variant}")
    params_base = dict(params or PARAMS_H2)
    features = train.drop(columns=["label"])
    oof_by_seed: dict[int, np.ndarray] = {}
    test_by_seed: dict[int, np.ndarray] = {}
    fold_rows: list[dict[str, Any]] = []

    for seed in seeds:
        oof = np.zeros(len(train), dtype=float)
        pred_test = np.zeros(len(test), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(
            StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(features, y)
        ):
            X_tr = features.iloc[tr_idx].reset_index(drop=True)
            X_va = features.iloc[va_idx].reset_index(drop=True)
            y_tr = y.iloc[tr_idx].reset_index(drop=True)
            y_va = y.iloc[va_idx].reset_index(drop=True)
            tr, va, te, cats = builder(X_tr, X_va, test.copy())
            p = dict(params_base)
            p["random_seed"] = seed + fold
            model = CatBoostClassifier(**p)
            if use_best_model and "od_type" in p:
                model.fit(tr, y_tr, eval_set=(va, y_va), cat_features=cats, use_best_model=True)
                best = model.get_best_iteration()
            else:
                model.fit(tr, y_tr, cat_features=cats)
                best = p.get("iterations", -1)
            pv = model.predict_proba(va)[:, 1]
            pt = model.predict_proba(te)[:, 1]
            if oof_transform == "rank":
                pv = rankdata(pv) / (len(pv) + 1.0)
                pt = rankdata(pt) / (len(pt) + 1.0)
            oof[va_idx] = pv
            pred_test += pt / n_splits
            auc = float(roc_auc_score(y_va, oof[va_idx]))
            fold_rows.append(
                {
                    "arm": variant,
                    "seed": seed,
                    "fold": fold,
                    "valid_auc": auc,
                    "best_iter": int(best if best is not None else -1),
                    "n_features": int(tr.shape[1]),
                    "n_cats": len(cats),
                    "oof_transform": oof_transform,
                }
            )
            print(
                f"{variant} seed={seed} fold={fold} auc={auc:.5f} best={best} n={tr.shape[1]} xf={oof_transform}",
                flush=True,
            )
        seed_auc = float(roc_auc_score(y, oof))
        print(f"{variant} seed={seed} OOF={seed_auc:.6f}", flush=True)
        oof_by_seed[seed] = oof
        test_by_seed[seed] = pred_test

    oof = np.mean(np.vstack(list(oof_by_seed.values())), axis=0)
    te = np.mean(np.vstack(list(test_by_seed.values())), axis=0)
    return {
        "oof": oof,
        "test": te,
        "oof_by_seed": oof_by_seed,
        "test_by_seed": test_by_seed,
        "oof_auc": float(roc_auc_score(y, oof)),
        "seed_aucs": {str(s): float(roc_auc_score(y, oof_by_seed[s])) for s in seeds},
        "folds": fold_rows,
        "variant": variant,
        "params": {k: v for k, v in params_base.items() if k != "verbose"},
        "n_splits": n_splits,
        "oof_transform": oof_transform,
    }
