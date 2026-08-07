"""CatBoost semantic triple-cross push targeting honest local OOF ~0.69.

Protocol (matches the established CatBoost semantic recipe):
- Features: raw + structured_string + days_condition + dual_category
- dual_category columns include semantic fields; cross_order=3, max_cross_columns=6
- Fold-local block fit only; no target encoding; no OOF weight search
- CatBoost only; equal average across seeds 2026..2029
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.feature_blocks import (
    DaysConditionFeatureBlock,
    DualCategoryFeatureBlock,
    RawFeatureBlock,
    StructuredStringFeatureBlock,
)
from insurance_claim.model import TARGET, audit_data, build_submission

N_SPLITS = 5
SEEDS_4 = (2026, 2027, 2028, 2029)
DUAL_COLS = [
    "region",
    "source",
    "version",
    "age_range",
    "month",
    "livability",
    "condition",
    "t3",
]
CAT_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=900,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=10,
    random_strength=0.7,
    od_type="Iter",
    od_wait=120,
    verbose=False,
    thread_count=-1,
    allow_writing_files=False,
)


def make_blocks():
    return [
        RawFeatureBlock(),
        StructuredStringFeatureBlock(),
        DaysConditionFeatureBlock(),
        DualCategoryFeatureBlock(
            columns=DUAL_COLS,
            max_categories=64,
            cross_order=3,
            max_cross_columns=6,
        ),
    ]


def transform_pair(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts_tr, parts_va, parts_te = [], [], []
    for block in make_blocks():
        parts_tr.append(block.fit_transform(X_tr))
        parts_va.append(block.transform(X_va))
        parts_te.append(block.transform(X_te))
    tr = pd.concat(parts_tr, axis=1)
    va = pd.concat(parts_va, axis=1)
    te = pd.concat(parts_te, axis=1)
    tr = tr.loc[:, ~tr.columns.duplicated()]
    va = va.reindex(columns=tr.columns)
    te = te.reindex(columns=tr.columns)
    return tr, va, te


def prepare_for_cat(
    tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    def is_cat(col: str, series: pd.Series) -> bool:
        if not pd.api.types.is_numeric_dtype(series):
            return True
        name = str(col)
        return (
            name.endswith(
                ("__category", "__category_cross", "__prefix", "__suffix", "__pattern")
            )
            or "__bin_" in name
            or name.endswith(("_bin", "__bin"))
            or "days_condition__bin" in name
        )

    cat_names = [column for column in tr.columns if is_cat(column, tr[column])]
    tr, va, te = tr.copy(), va.copy(), te.copy()
    for column in cat_names:
        tr[column] = tr[column].astype(str).fillna("__MISSING__")
        va[column] = va[column].astype(str).fillna("__MISSING__")
        te[column] = te[column].astype(str).fillna("__MISSING__")
    for column in tr.columns:
        if column in cat_names:
            continue
        tr[column] = pd.to_numeric(tr[column], errors="coerce")
        median = float(tr[column].median()) if tr[column].notna().any() else 0.0
        tr[column] = tr[column].fillna(median)
        va[column] = pd.to_numeric(va[column], errors="coerce").fillna(median)
        te[column] = pd.to_numeric(te[column], errors="coerce").fillna(median)
    return tr, va, te, cat_names


def run_seeds(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seeds: tuple[int, ...],
    y_override: np.ndarray | None = None,
) -> dict[str, Any]:
    y = (
        pd.Series(y_override, name=TARGET).astype(int)
        if y_override is not None
        else train[TARGET].astype(int)
    )
    features = train.drop(columns=[TARGET])
    oof_by_seed: dict[int, np.ndarray] = {}
    test_by_seed: dict[int, np.ndarray] = {}
    fold_rows: list[dict[str, Any]] = []
    started = time.time()

    for seed in seeds:
        folds = list(
            StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(
                features, y
            )
        )
        oof = np.zeros(len(train), dtype=float)
        pred_test = np.zeros(len(test), dtype=float)
        for fold, (train_idx, valid_idx) in enumerate(folds):
            X_tr = features.iloc[train_idx].reset_index(drop=True)
            y_tr = y.iloc[train_idx].reset_index(drop=True)
            X_va = features.iloc[valid_idx].reset_index(drop=True)
            y_va = y.iloc[valid_idx].reset_index(drop=True)
            tr_fe, va_fe, te_fe = transform_pair(X_tr, X_va, test.copy())
            tr_fe, va_fe, te_fe, cat_names = prepare_for_cat(tr_fe, va_fe, te_fe)
            params = dict(CAT_PARAMS)
            params["random_seed"] = seed + fold
            model = CatBoostClassifier(**params)
            model.fit(
                tr_fe,
                y_tr,
                eval_set=(va_fe, y_va),
                cat_features=cat_names,
                use_best_model=True,
                verbose=False,
            )
            valid_pred = model.predict_proba(va_fe)[:, 1]
            test_pred = model.predict_proba(te_fe)[:, 1]
            oof[valid_idx] = valid_pred
            pred_test += test_pred / N_SPLITS
            best_iteration = model.get_best_iteration()
            valid_auc = float(roc_auc_score(y_va, valid_pred))
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "valid_auc": valid_auc,
                    "best_iter": int(best_iteration if best_iteration is not None else -1),
                    "n_features": int(tr_fe.shape[1]),
                    "n_cats": len(cat_names),
                }
            )
            print(
                f"seed={seed} fold={fold} auc={valid_auc:.5f} "
                f"best={best_iteration} n_feat={tr_fe.shape[1]} n_cat={len(cat_names)}",
                flush=True,
            )
        seed_auc = float(roc_auc_score(y, oof))
        oof_by_seed[seed] = oof
        test_by_seed[seed] = pred_test
        print(f"seed={seed} OOF={seed_auc:.5f}", flush=True)

    oof_pool = np.mean(np.vstack([oof_by_seed[seed] for seed in seeds]), axis=0)
    test_pool = np.mean(np.vstack([test_by_seed[seed] for seed in seeds]), axis=0)
    seed_aucs = {
        str(seed): float(roc_auc_score(y, oof_by_seed[seed])) for seed in seeds
    }
    metrics = {
        "recipe": "catboost_semantic_triple_4seed",
        "seeds": list(seeds),
        "pooled_oof_auc": float(roc_auc_score(y, oof_pool)),
        "seed_aucs": seed_aucs,
        "seed_mean": float(np.mean(list(seed_aucs.values()))),
        "seed_std": float(np.std(list(seed_aucs.values()))),
        "pred_mean": float(test_pool.mean()),
        "elapsed_sec": round(time.time() - started, 1),
        "folds": fold_rows,
        "policy": (
            "CatBoost-only semantic triple-cross; fold-local feature blocks; "
            "no TE; equal seed average; no OOF weight search"
        ),
    }
    print(
        f"POOLED OOF={metrics['pooled_oof_auc']:.6f} "
        f"seed_mean={metrics['seed_mean']:.6f}±{metrics['seed_std']:.6f}",
        flush=True,
    )
    return {
        "metrics": metrics,
        "oof": oof_pool,
        "test": test_pool,
        "y": y.to_numpy(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/cat_semantic")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_4))
    parser.add_argument("--shuffled", action="store_true")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "submit_sample.csv")
    audit = audit_data(train, test, sample)

    result = run_seeds(train, test, tuple(args.seeds))
    metrics = result["metrics"]
    metrics["audit"] = {
        "train_rows": audit["train_rows"],
        "test_rows": audit["test_rows"],
        "target_rate": audit["target_rate"],
    }

    if args.shuffled:
        shuffled = train[TARGET].to_numpy().copy()
        np.random.default_rng(2026).shuffle(shuffled)
        shuffled_result = run_seeds(train, test, (args.seeds[0],), y_override=shuffled)
        metrics["shuffled_oof_auc"] = shuffled_result["metrics"]["pooled_oof_auc"]
        metrics["shuffled_pass"] = bool(
            0.47 <= metrics["shuffled_oof_auc"] <= 0.53
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        oof=result["oof"],
        test=result["test"],
        y=result["y"],
    )
    build_submission(
        test, sample, result["test"], args.output_dir / "submission_cat_semantic.csv"
    )
    final_dir = Path("final_candidates_self")
    final_dir.mkdir(parents=True, exist_ok=True)
    build_submission(
        test,
        sample,
        result["test"],
        final_dir / "submission_catboost_069.csv",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (final_dir / "CATBOOST_069.md").write_text(
        "\n".join(
            [
                "# CatBoost 语义三阶冲击 ~0.69",
                "",
                f"- pooled OOF：**{metrics['pooled_oof_auc']:.8f}**",
                f"- seeds：{metrics['seeds']}",
                f"- seed mean±std：{metrics['seed_mean']:.6f}±{metrics['seed_std']:.6f}",
                "- 模型：仅 CatBoost",
                "- 特征：raw + structured_string + days_condition + dual_category(cross_order=3)",
                "- 无 TE、无 OOF 搜权",
                "- 提交：`submission_catboost_069.csv`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "pooled_oof_auc": metrics["pooled_oof_auc"],
                "seed_aucs": metrics["seed_aucs"],
                "shuffled_oof_auc": metrics.get("shuffled_oof_auc"),
                "shuffled_pass": metrics.get("shuffled_pass"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
