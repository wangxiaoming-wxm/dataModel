from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

TARGET = "label"
IDENTIFIER = "id"


@dataclass(frozen=True)
class TrainingConfig:
    folds: int = 5
    repeats: int = 2
    seed: int = 2026
    cat_iterations: int = 900
    xgb_iterations: int = 1400
    early_stopping_rounds: int = 120


def audit_data(
    train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame
) -> dict[str, Any]:
    """Validate the competition boundary and return leakage diagnostics."""
    if TARGET not in train or TARGET in test:
        raise ValueError("label must exist only in training data")
    if IDENTIFIER not in train or IDENTIFIER not in test:
        raise ValueError("both datasets must contain id")
    if set(train[TARGET].dropna().unique()) - {0, 1}:
        raise ValueError("label must be binary")
    if train[IDENTIFIER].duplicated().any() or test[IDENTIFIER].duplicated().any():
        raise ValueError("identifiers must be unique")

    overlap = len(set(train[IDENTIFIER]) & set(test[IDENTIFIER]))
    if overlap:
        raise ValueError(f"identifier overlap detected: {overlap}")
    if sample.columns.tolist() != [IDENTIFIER, TARGET]:
        raise ValueError("submission template must contain id,label in that order")
    if sample[IDENTIFIER].tolist() != test[IDENTIFIER].tolist():
        raise ValueError("submission identifiers must match test order")

    train_features = train.drop(columns=[TARGET, IDENTIFIER])
    test_features = test.drop(columns=[IDENTIFIER])
    columns_match = train_features.columns.tolist() == test_features.columns.tolist()
    if not columns_match:
        raise ValueError("training and test feature columns differ")

    shared_rows = len(
        train_features.merge(test_features, how="inner").drop_duplicates()
    )
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(train_features.shape[1]),
        "target_rate": float(train[TARGET].mean()),
        "id_overlap": overlap,
        "duplicate_train_ids": int(train[IDENTIFIER].duplicated().sum()),
        "duplicate_test_ids": int(test[IDENTIFIER].duplicated().sum()),
        "exact_cross_feature_overlap": int(shared_rows),
        "train_test_columns_match": columns_match,
    }


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create target-independent, domain-plausible tabular features."""
    features = frame.drop(columns=[IDENTIFIER, TARGET], errors="ignore").copy()

    if "month" in features:
        features["month_n"] = pd.to_numeric(
            features["month"].astype(str).str.removeprefix("M"), errors="coerce"
        )
    if "t3" in features:
        t3 = features["t3"].astype(str)
        features["t3_value"] = pd.to_numeric(t3.str[:-1], errors="coerce")
        features["t3_kind"] = t3.str[-1:].replace({"nan": "__NA__"})
    if "source" in features:
        source = features["source"].astype(str)
        features["source_car"] = pd.to_numeric(
            source.str.extract(r"CAR_(\d+)", expand=False), errors="coerce"
        )
        features["source_eng"] = pd.to_numeric(
            source.str.extract(r"ENG_(\d+)", expand=False), errors="coerce"
        )
    if "version" in features:
        features["version_n"] = pd.to_numeric(
            features["version"].astype(str).str.removeprefix("v"), errors="coerce"
        )
    if "grades" in features:
        features["grades_n"] = features["grades"].map(
            {"s": 1.0, "ss": 2.0, "sss": 3.0}
        )

    x_columns = [
        column
        for column in features
        if column.startswith("x") and column[1:].isdigit()
    ]
    if x_columns:
        vectors = features[x_columns].apply(pd.to_numeric, errors="coerce")
        features["x_mean"] = vectors.mean(axis=1)
        features["x_std"] = vectors.std(axis=1, ddof=0)
        features["x_min"] = vectors.min(axis=1)
        features["x_max"] = vectors.max(axis=1)
        features["x_l1"] = vectors.abs().sum(axis=1)
        features["x_l2"] = np.sqrt(vectors.pow(2).sum(axis=1))
        features["x_positive_count"] = vectors.gt(0).sum(axis=1)

    for column in ("days", "condition", "cc", "max_g"):
        if column in features:
            values = pd.to_numeric(features[column], errors="coerce")
            features[f"{column}_log1p_abs"] = np.log1p(values.abs())
            if column == "condition":
                features["condition_missing"] = values.isna().astype("int8")

    return features.replace([np.inf, -np.inf], np.nan)


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Map scores to open-interval empirical ranks."""
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average").to_numpy() / (len(series) + 1.0)


def _catboost_frame(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = features.copy()
    categorical = result.select_dtypes(exclude=np.number).columns.tolist()
    result[categorical] = result[categorical].fillna("__NA__").astype(str)
    return result, categorical


def _numeric_frame(features: pd.DataFrame) -> pd.DataFrame:
    return features.select_dtypes(include=np.number).astype(float)


def train_ensemble(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: TrainingConfig = TrainingConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train fixed-complexity repeated-CV models and average test probabilities."""
    y = train[TARGET].astype(int).reset_index(drop=True)
    train_features = engineer_features(train)
    test_features = engineer_features(test)
    cat_train, categorical = _catboost_frame(train_features)
    cat_test, _ = _catboost_frame(test_features)
    xgb_train = _numeric_frame(train_features)
    xgb_test = _numeric_frame(test_features)

    splitter = RepeatedStratifiedKFold(
        n_splits=config.folds,
        n_repeats=config.repeats,
        random_state=config.seed,
    )
    oof_cat = np.zeros((config.repeats, len(train)))
    oof_xgb = np.zeros_like(oof_cat)
    test_cat: list[np.ndarray] = []
    test_xgb: list[np.ndarray] = []
    fold_metrics: list[dict[str, float | int]] = []

    for split_number, (fit_index, valid_index) in enumerate(
        splitter.split(train_features, y)
    ):
        repeat = split_number // config.folds
        fold = split_number % config.folds
        fold_seed = config.seed + split_number

        cat_model = CatBoostClassifier(
            iterations=config.cat_iterations,
            depth=6,
            learning_rate=0.035,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=10,
            random_seed=fold_seed,
            allow_writing_files=False,
            verbose=False,
            thread_count=-1,
        )
        cat_model.fit(
            cat_train.iloc[fit_index],
            y.iloc[fit_index],
            cat_features=categorical,
            eval_set=(cat_train.iloc[valid_index], y.iloc[valid_index]),
            early_stopping_rounds=config.early_stopping_rounds,
            verbose=False,
        )
        cat_valid = cat_model.predict_proba(cat_train.iloc[valid_index])[:, 1]
        oof_cat[repeat, valid_index] = cat_valid
        test_cat.append(cat_model.predict_proba(cat_test)[:, 1])

        xgb_model = XGBClassifier(
            n_estimators=config.xgb_iterations,
            learning_rate=0.025,
            max_depth=3,
            min_child_weight=8,
            subsample=0.82,
            colsample_bytree=0.82,
            reg_alpha=1.0,
            reg_lambda=15.0,
            objective="binary:logistic",
            eval_metric="auc",
            early_stopping_rounds=config.early_stopping_rounds,
            random_state=fold_seed,
            n_jobs=-1,
        )
        xgb_model.fit(
            xgb_train.iloc[fit_index],
            y.iloc[fit_index],
            eval_set=[(xgb_train.iloc[valid_index], y.iloc[valid_index])],
            verbose=False,
        )
        xgb_valid = xgb_model.predict_proba(xgb_train.iloc[valid_index])[:, 1]
        oof_xgb[repeat, valid_index] = xgb_valid
        test_xgb.append(xgb_model.predict_proba(xgb_test)[:, 1])

        fold_metrics.append(
            {
                "repeat": repeat,
                "fold": fold,
                "cat_auc": float(roc_auc_score(y.iloc[valid_index], cat_valid)),
                "xgb_auc": float(roc_auc_score(y.iloc[valid_index], xgb_valid)),
                "cat_best_iteration": int(cat_model.get_best_iteration()),
                "xgb_best_iteration": int(xgb_model.best_iteration),
            }
        )

    repeat_metrics = []
    for repeat in range(config.repeats):
        blend = 0.5 * oof_cat[repeat] + 0.5 * oof_xgb[repeat]
        repeat_metrics.append(
            {
                "repeat": repeat,
                "cat_auc": float(roc_auc_score(y, oof_cat[repeat])),
                "xgb_auc": float(roc_auc_score(y, oof_xgb[repeat])),
                "blend_auc": float(roc_auc_score(y, blend)),
            }
        )

    predictions = 0.5 * np.mean(test_cat, axis=0) + 0.5 * np.mean(
        test_xgb, axis=0
    )
    metrics = {
        "config": asdict(config),
        "selection_policy": "fixed 50/50 blend; no leaderboard feedback",
        "folds": fold_metrics,
        "repeats": repeat_metrics,
        "blend_auc_mean": float(
            np.mean([metric["blend_auc"] for metric in repeat_metrics])
        ),
        "blend_auc_std": float(
            np.std([metric["blend_auc"] for metric in repeat_metrics])
        ),
    }
    return predictions, metrics


def build_submission(
    test: pd.DataFrame,
    sample: pd.DataFrame,
    predictions: np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    """Write predictions while preserving the organizer's exact row order."""
    predictions = np.asarray(predictions, dtype=float)
    if len(predictions) != len(test):
        raise ValueError("prediction count does not match test rows")
    if not np.isfinite(predictions).all() or not ((0 <= predictions) & (predictions <= 1)).all():
        raise ValueError("predictions must be finite and within [0, 1]")
    if sample[IDENTIFIER].tolist() != test[IDENTIFIER].tolist():
        raise ValueError("sample and test identifiers are not aligned")

    submission = sample.copy()
    submission[TARGET] = predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def main() -> None:  # pragma: no cover - exercised by the full training run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "submit_sample.csv")
    audit = audit_data(train, test, sample)
    config = TrainingConfig(folds=args.folds, repeats=args.repeats, seed=args.seed)
    predictions, metrics = train_ensemble(train, test, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_submission(test, sample, predictions, args.output_dir / "submission.csv")
    (args.output_dir / "audit_report.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "cv_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
