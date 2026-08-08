"""B6pro third hetero arm: LightGBM / XGBoost on numeric+OHE lean view (fold-local)."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

N_SPLITS = 5


def _prepare_matrix(X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame):
    drop = [c for c in ("id", "label") if c in X_tr.columns]
    tr = X_tr.drop(columns=drop, errors="ignore").copy()
    va = X_va.drop(columns=[c for c in drop if c in X_va.columns], errors="ignore").copy()
    te = X_te.drop(columns=[c for c in ("id",) if c in X_te.columns], errors="ignore").copy()
    # keep x0-x18 for hetero signal; drop ultra-high-card x19 like plus
    if "x19" in tr.columns:
        tr = tr.drop(columns=["x19"])
        va = va.drop(columns=["x19"], errors="ignore")
        te = te.drop(columns=["x19"], errors="ignore")

    cat_cols = [c for c in tr.columns if tr[c].dtype == object or str(tr[c].dtype) == "string"]
    num_cols = [c for c in tr.columns if c not in cat_cols]

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if cat_cols:
        enc.fit(tr[cat_cols].astype(str))
        tr_c = enc.transform(tr[cat_cols].astype(str))
        va_c = enc.transform(va[cat_cols].astype(str))
        te_c = enc.transform(te[cat_cols].astype(str))
    else:
        tr_c = np.zeros((len(tr), 0))
        va_c = np.zeros((len(va), 0))
        te_c = np.zeros((len(te), 0))

    def num_block(df: pd.DataFrame, ref: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(df), len(num_cols)), dtype=float)
        for i, c in enumerate(num_cols):
            s = pd.to_numeric(df[c], errors="coerce")
            med = float(pd.to_numeric(ref[c], errors="coerce").median())
            out[:, i] = s.fillna(med).to_numpy()
        return out

    tr_n = num_block(tr, tr)
    va_n = num_block(va, tr)
    te_n = num_block(te, tr)
    # simple interactions
    def add_ratio(df: pd.DataFrame) -> np.ndarray:
        days = pd.to_numeric(df.get("days"), errors="coerce").fillna(0).to_numpy()
        cond = pd.to_numeric(df.get("condition"), errors="coerce").fillna(0).to_numpy()
        return np.column_stack([cond / (np.abs(days) + 1.0), np.log1p(np.abs(days))])

    tr_x = np.hstack([tr_n, tr_c, add_ratio(tr)])
    va_x = np.hstack([va_n, va_c, add_ratio(va)])
    te_x = np.hstack([te_n, te_c, add_ratio(te)])
    return tr_x, va_x, te_x


def run_tree_arm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: pd.Series,
    seeds: tuple[int, ...],
    *,
    backend: str = "lgb",
) -> dict[str, Any]:
    features = train.drop(columns=["label"])
    oof_by_seed: dict[int, np.ndarray] = {}
    test_by_seed: dict[int, np.ndarray] = {}
    fold_rows: list[dict[str, Any]] = []

    for seed in seeds:
        oof = np.zeros(len(train), dtype=float)
        pred_test = np.zeros(len(test), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(
            StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(features, y)
        ):
            X_tr = features.iloc[tr_idx].reset_index(drop=True)
            X_va = features.iloc[va_idx].reset_index(drop=True)
            y_tr = y.iloc[tr_idx].to_numpy()
            y_va = y.iloc[va_idx].to_numpy()
            tr_x, va_x, te_x = _prepare_matrix(X_tr, X_va, test.copy())
            if backend == "lgb":
                dtr = lgb.Dataset(tr_x, label=y_tr)
                dva = lgb.Dataset(va_x, label=y_va, reference=dtr)
                params = dict(
                    objective="binary",
                    metric="auc",
                    learning_rate=0.03,
                    num_leaves=48,
                    min_data_in_leaf=40,
                    feature_fraction=0.8,
                    bagging_fraction=0.8,
                    bagging_freq=1,
                    lambda_l2=5.0,
                    verbosity=-1,
                    seed=seed + fold,
                )
                model = lgb.train(
                    params,
                    dtr,
                    num_boost_round=2500,
                    valid_sets=[dva],
                    callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
                )
                oof[va_idx] = model.predict(va_x, num_iteration=model.best_iteration)
                pred_test += model.predict(te_x, num_iteration=model.best_iteration) / N_SPLITS
                best = int(model.best_iteration or -1)
            else:
                model = xgb.XGBClassifier(
                    n_estimators=2500,
                    learning_rate=0.03,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=5.0,
                    objective="binary:logistic",
                    eval_metric="auc",
                    tree_method="hist",
                    early_stopping_rounds=150,
                    random_state=seed + fold,
                    n_jobs=4,
                )
                model.fit(tr_x, y_tr, eval_set=[(va_x, y_va)], verbose=False)
                oof[va_idx] = model.predict_proba(va_x)[:, 1]
                pred_test += model.predict_proba(te_x)[:, 1] / N_SPLITS
                best = int(getattr(model, "best_iteration", -1) or -1)
            auc = float(roc_auc_score(y_va, oof[va_idx]))
            fold_rows.append(
                {"arm": backend, "seed": seed, "fold": fold, "valid_auc": auc, "best_iter": best}
            )
            print(f"{backend} seed={seed} fold={fold} auc={auc:.5f} best={best}", flush=True)
        seed_auc = float(roc_auc_score(y, oof))
        print(f"{backend} seed={seed} OOF={seed_auc:.6f}", flush=True)
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
        "backend": backend,
    }
