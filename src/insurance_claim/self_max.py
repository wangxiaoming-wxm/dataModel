"""Locked self-pipeline maximum honest AUC recipe.

Exploration constraints (hard):
- No target encoding / no label-dependent global statistics
- Frequency encoding fit on the outer training fold only
- Nested early stopping (outer validation labels never touch early stop)
- Fixed equal-rank blend over pre-registered families; no OOF weight search
- Shuffled-label sanity must stay near 0.5

Locked recipe (after clean self-only exploration, ignoring external packages):
- Features: engineer_features_v2 + fold-safe frequency
- Families: CatBoost-A (depth6/lr0.03), CatBoost-B (depth8/lr0.02), XGBoost
- 5 stratified seeds: (2026, 7, 42, 123, 314159)
- Final score: equal-rank average of the three bagged families
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from insurance_claim.model import (
    TARGET,
    _stratified_early_split,
    audit_data,
    build_submission,
    engineer_features,
    rank_normalize,
)

FREQ_COLS = ("region", "t3", "source", "code", "version", "grades", "month")
LOCKED_SEEDS = (2026, 7, 42, 123, 314159)


@dataclass(frozen=True)
class CatVariant:
    name: str
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    iterations: int


LOCKED_CATS = (
    CatVariant("cat_a", depth=6, learning_rate=0.03, l2_leaf_reg=12.0, iterations=1200),
    CatVariant("cat_b", depth=8, learning_rate=0.02, l2_leaf_reg=16.0, iterations=1500),
)


@dataclass(frozen=True)
class SelfMaxConfig:
    folds: int = 5
    seeds: tuple[int, ...] = LOCKED_SEEDS
    early_stopping_rounds: int = 100
    xgb_iterations: int = 1600


def engineer_features_v2(frame: pd.DataFrame) -> pd.DataFrame:
    features = engineer_features(frame)
    days = pd.to_numeric(features.get("days"), errors="coerce")
    if days is not None:
        edges = [-np.inf, 700, 1500, 3000, 5000, 7000, 8600, 10000, np.inf]
        features["days_bin"] = pd.cut(days, bins=edges, labels=False).astype("float")
        features["days_sqrt"] = np.sqrt(days.clip(lower=0))
    condition = pd.to_numeric(features.get("condition"), errors="coerce")
    if condition is not None and days is not None:
        features["days_x_condition"] = days * condition.fillna(condition.median())
    for left, right, name in (
        ("x19", "V", "x19_minus_V"),
        ("cc", "V", "cc_minus_V"),
        ("V", "max_g", "V_minus_max_g"),
        ("x19", "cc", "x19_minus_cc"),
    ):
        if left in features.columns and right in features.columns:
            features[name] = pd.to_numeric(features[left], errors="coerce") - pd.to_numeric(
                features[right], errors="coerce"
            )
    if "age_range" in features.columns and days is not None:
        age = pd.to_numeric(features["age_range"], errors="coerce").replace(0, np.nan)
        features["days_per_age"] = days / age
    return features.replace([np.inf, -np.inf], np.nan)


def add_fold_frequency(
    train_features: pd.DataFrame,
    fit_index: np.ndarray,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_features.copy()
    test_out = test_features.copy()
    fit = train_features.iloc[fit_index]
    for column in FREQ_COLS:
        if column not in train_features.columns:
            continue
        counts = fit[column].astype(str).value_counts(dropna=False)
        mapping = (counts / float(len(fit))).to_dict()
        name = f"{column}_freq"
        train_out[name] = (
            train_features[column].astype(str).map(mapping).fillna(0.0).astype(float)
        )
        test_out[name] = (
            test_features[column].astype(str).map(mapping).fillna(0.0).astype(float)
        )
    return train_out, test_out


def _cat_model(variant: CatVariant, iterations: int, seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=variant.depth,
        learning_rate=variant.learning_rate,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=variant.l2_leaf_reg,
        random_strength=0.8,
        bagging_temperature=0.6,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )


def _xgb_model(
    iterations: int, seed: int, early_stopping_rounds: int | None = None
) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=iterations,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=12.0,
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=early_stopping_rounds,
        random_state=seed,
        n_jobs=-1,
    )


def run_self_max(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: SelfMaxConfig | None = None,
    y_override: np.ndarray | None = None,
) -> dict[str, Any]:
    config = config or SelfMaxConfig()
    y = (
        pd.Series(y_override, name=TARGET).astype(int)
        if y_override is not None
        else train[TARGET].astype(int).reset_index(drop=True)
    )
    train_features = engineer_features_v2(train).reset_index(drop=True)
    test_features = engineer_features_v2(test).reset_index(drop=True)
    family_names = [variant.name for variant in LOCKED_CATS] + ["xgb"]
    bag_oof = {name: np.zeros(len(train)) for name in family_names}
    bag_test = {name: np.zeros(len(test)) for name in family_names}
    fold_rows: list[dict[str, Any]] = []

    for seed in config.seeds:
        splitter = StratifiedKFold(
            n_splits=config.folds, shuffle=True, random_state=seed
        )
        seed_oof = {name: np.zeros(len(train)) for name in family_names}
        seed_test: dict[str, list[np.ndarray]] = {name: [] for name in family_names}
        for fold, (fit_index, valid_index) in enumerate(
            splitter.split(train_features, y)
        ):
            train_fold, test_fold = add_fold_frequency(
                train_features, fit_index, test_features
            )
            cat_cols = [
                column
                for column in train_fold.columns
                if (not pd.api.types.is_numeric_dtype(train_fold[column]))
                or column == "days_bin"
            ]
            num_cols = [column for column in train_fold.columns if column not in cat_cols]
            train_cat = train_fold.copy()
            test_cat = test_fold.copy()
            for column in cat_cols:
                train_cat[column] = train_cat[column].fillna("__NA__").astype(str)
                test_cat[column] = test_cat[column].fillna("__NA__").astype(str)
            median = train_fold[num_cols].astype(float).iloc[fit_index].median()
            train_num = train_fold[num_cols].astype(float).fillna(median)
            test_num = test_fold[num_cols].astype(float).fillna(median)
            early_size = max(2, math.ceil(0.15 * len(fit_index)))
            inner_fit, early_index = _stratified_early_split(
                fit_index, y, early_size, seed + 17 * fold
            )
            fold_metric: dict[str, Any] = {"seed": seed, "fold": fold}
            for variant in LOCKED_CATS:
                fold_seed = seed + 17 * fold
                tuner = _cat_model(variant, variant.iterations, fold_seed)
                tuner.fit(
                    train_cat.iloc[inner_fit],
                    y.iloc[inner_fit],
                    cat_features=cat_cols,
                    eval_set=(train_cat.iloc[early_index], y.iloc[early_index]),
                    early_stopping_rounds=config.early_stopping_rounds,
                    verbose=False,
                )
                best = max(1, tuner.get_best_iteration() + 1)
                model = _cat_model(variant, best, fold_seed)
                model.fit(
                    train_cat.iloc[fit_index],
                    y.iloc[fit_index],
                    cat_features=cat_cols,
                    verbose=False,
                )
                valid_pred = model.predict_proba(train_cat.iloc[valid_index])[:, 1]
                test_pred = model.predict_proba(test_cat)[:, 1]
                seed_oof[variant.name][valid_index] = valid_pred
                seed_test[variant.name].append(test_pred)
                fold_metric[f"{variant.name}_auc"] = float(
                    roc_auc_score(y.iloc[valid_index], valid_pred)
                )
                fold_metric[f"{variant.name}_best"] = best

            fold_seed = seed + 17 * fold
            xgb_tuner = _xgb_model(
                config.xgb_iterations, fold_seed, config.early_stopping_rounds
            )
            xgb_tuner.fit(
                train_num.iloc[inner_fit],
                y.iloc[inner_fit],
                eval_set=[(train_num.iloc[early_index], y.iloc[early_index])],
                verbose=False,
            )
            xgb_best = max(1, xgb_tuner.best_iteration + 1)
            xgb_model = _xgb_model(xgb_best, fold_seed)
            xgb_model.fit(
                train_num.iloc[fit_index], y.iloc[fit_index], verbose=False
            )
            xgb_valid = xgb_model.predict_proba(train_num.iloc[valid_index])[:, 1]
            xgb_test = xgb_model.predict_proba(test_num)[:, 1]
            seed_oof["xgb"][valid_index] = xgb_valid
            seed_test["xgb"].append(xgb_test)
            fold_metric["xgb_auc"] = float(
                roc_auc_score(y.iloc[valid_index], xgb_valid)
            )
            fold_metric["xgb_best"] = xgb_best
            fold_rows.append(fold_metric)

        for name in family_names:
            bag_oof[name] += seed_oof[name] / len(config.seeds)
            bag_test[name] += np.mean(seed_test[name], axis=0) / len(config.seeds)

    oof_rank = np.mean([rank_normalize(bag_oof[name]) for name in family_names], axis=0)
    test_rank = np.mean(
        [rank_normalize(bag_test[name]) for name in family_names], axis=0
    )
    prob_ref = np.mean([bag_test[name] for name in family_names], axis=0)
    order = np.argsort(test_rank, kind="mergesort")
    test_calibrated = np.empty_like(test_rank)
    test_calibrated[order] = np.sort(prob_ref)

    metrics = {
        "recipe": "self_max_locked_v1",
        "config": {
            **asdict(config),
            "seeds": list(config.seeds),
            "cat_variants": [asdict(variant) for variant in LOCKED_CATS],
            "selection_policy": (
                "equal-rank over pre-registered cat_a+cat_b+xgb; "
                "no TE; no OOF weight search"
            ),
        },
        "family_bag_auc": {
            name: float(roc_auc_score(y, bag_oof[name])) for name in family_names
        },
        "final_oof_auc": float(roc_auc_score(y, oof_rank)),
        "folds": fold_rows,
        "baseline_original_blend_auc_mean": 0.64873,
    }
    return {
        "metrics": metrics,
        "oof": oof_rank,
        "test": test_calibrated,
        "y": y.to_numpy(),
        "oof_family": bag_oof,
        "test_family": bag_test,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/self_max"))
    parser.add_argument("--shuffled", action="store_true")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(LOCKED_SEEDS),
    )
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "submit_sample.csv")
    audit = audit_data(train, test, sample)
    config = SelfMaxConfig(seeds=tuple(args.seeds))
    result = run_self_max(train, test, config)
    metrics = result["metrics"]
    metrics["audit"] = audit

    if args.shuffled:
        shuffled = train[TARGET].to_numpy().copy()
        np.random.default_rng(314159).shuffle(shuffled)
        shuffled_result = run_self_max(
            train, test, SelfMaxConfig(seeds=(config.seeds[0],)), y_override=shuffled
        )
        metrics["shuffled_oof_auc"] = shuffled_result["metrics"]["final_oof_auc"]
        metrics["shuffled_pass"] = bool(
            0.47 <= metrics["shuffled_oof_auc"] <= 0.53
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "oof": result["oof"],
        "test": result["test"],
        "y": result["y"],
    }
    for name, values in result["oof_family"].items():
        payload[f"oof_{name}"] = values
        payload[f"test_{name}"] = result["test_family"][name]
    np.savez_compressed(args.output_dir / "predictions.npz", **payload)
    build_submission(
        test, sample, result["test"], args.output_dir / "submission_self_max.csv"
    )
    # Also publish under final_candidates with an explicit self-only name.
    final_dir = Path("final_candidates_self")
    final_dir.mkdir(parents=True, exist_ok=True)
    build_submission(
        test,
        sample,
        result["test"],
        final_dir / "submission_self_max_auc.csv",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (final_dir / "README.md").write_text(
        "\n".join(
            [
                "# 自研流水线最大诚实 AUC（排除外部方案）",
                "",
                f"- 本地 OOF AUC：**{metrics['final_oof_auc']:.8f}**",
                f"- 原始自研基线（Cat+XGB 50/50）：**0.64873**",
                "- 规则：无 TE、无 OOF 搜权、嵌套早停、折内频率编码、固定等权 rank",
                "- 文件：`submission_self_max_auc.csv`",
                "- 复现：`PYTHONPATH=src python3 -m insurance_claim.self_max --shuffled`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "final_oof_auc": metrics["final_oof_auc"],
                "family_bag_auc": metrics["family_bag_auc"],
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
