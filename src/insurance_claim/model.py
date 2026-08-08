from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
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
    if train.empty or test.empty:
        raise ValueError("training and test data must not be empty")
    if TARGET not in train or TARGET in test:
        raise ValueError("label must exist only in training data")
    if IDENTIFIER not in train or IDENTIFIER not in test:
        raise ValueError("both datasets must contain id")
    if train[TARGET].isna().any() or set(train[TARGET].unique()) != {0, 1}:
        raise ValueError("label must be non-missing and contain both binary classes")
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
    incompatible = [
        column
        for column in train_features
        if pd.api.types.is_numeric_dtype(train_features[column])
        != pd.api.types.is_numeric_dtype(test_features[column])
    ]
    if incompatible:
        raise ValueError(f"training and test feature dtypes differ: {incompatible}")

    train_hashes = pd.util.hash_pandas_object(train_features, index=False).unique()
    test_hashes = pd.util.hash_pandas_object(test_features, index=False).unique()
    shared_rows = len(np.intersect1d(train_hashes, test_hashes, assume_unique=True))
    return {
        "train_rows": len(train),
        "test_rows": len(test),
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
        parsed_t3 = t3.str.extract(r"^(-?\d+(?:\.\d+)?)([A-Za-z])$")
        invalid_t3 = features["t3"].notna() & parsed_t3[0].isna()
        if invalid_t3.any():
            raise ValueError(
                "t3 contains values outside the expected number+letter format"
            )
        features["t3_value"] = pd.to_numeric(parsed_t3[0], errors="coerce")
        features["t3_kind"] = parsed_t3[1].fillna("__NA__")
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
        features["grades_n"] = features["grades"].map({"s": 1.0, "ss": 2.0, "sss": 3.0})

    x_columns = [
        column for column in features if column.startswith("x") and column[1:].isdigit()
    ]
    if x_columns:
        vectors = (
            features[x_columns].apply(pd.to_numeric, errors="coerce").astype(float)
        )
        features["x_mean"] = vectors.mean(axis=1)
        features["x_std"] = vectors.std(axis=1, ddof=0)
        features["x_min"] = vectors.min(axis=1)
        features["x_max"] = vectors.max(axis=1)
        absolute = vectors.abs()
        features["x_l1"] = absolute.sum(axis=1, min_count=1)
        scale = absolute.max(axis=1)
        scaled = vectors.div(scale.replace(0, 1), axis=0)
        features["x_l2"] = scale * np.sqrt(scaled.pow(2).sum(axis=1, min_count=1))
        features["x_positive_count"] = (
            vectors.gt(0).where(vectors.notna()).sum(axis=1, min_count=1)
        )
        features["x_missing_count"] = vectors.isna().sum(axis=1)

    for column in ("days", "condition", "cc", "max_g"):
        if column in features:
            values = pd.to_numeric(features[column], errors="coerce")
            features[f"{column}_log1p_abs"] = np.log1p(values.abs())
            if column == "condition":
                features["condition_missing"] = values.isna().astype("int8")

    return features.replace([np.inf, -np.inf], np.nan)


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Map scores to open-interval empirical ranks."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("rank input must be a non-empty finite one-dimensional array")
    series = pd.Series(array)
    return series.rank(method="average").to_numpy() / (len(series) + 1.0)


def _catboost_frames(
    train_features: pd.DataFrame, test_features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    categorical = train_features.select_dtypes(exclude=np.number).columns.tolist()
    train_result = train_features.copy()
    test_result = test_features.copy()
    train_result[categorical] = train_result[categorical].fillna("__NA__").astype(str)
    test_result[categorical] = test_result[categorical].fillna("__NA__").astype(str)
    return train_result, test_result, categorical


def _validate_config(config: TrainingConfig, y: pd.Series) -> None:
    if config.folds < 2 or config.repeats < 1:
        raise ValueError("folds must be >= 2 and repeats must be >= 1")
    if min(config.cat_iterations, config.xgb_iterations) < 1:
        raise ValueError("model iteration counts must be positive")
    if config.early_stopping_rounds < 1:
        raise ValueError("early stopping rounds must be positive")
    if y.value_counts().min() < 2 * config.folds:
        raise ValueError("each target class must contain at least twice folds samples")


def _cat_model(iterations: int, seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.035,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=10,
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
        learning_rate=0.025,
        max_depth=3,
        min_child_weight=8,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_alpha=1.0,
        reg_lambda=15.0,
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=early_stopping_rounds,
        random_state=seed,
        n_jobs=-1,
    )


def _stratified_early_split(
    fit_index: np.ndarray,
    y: pd.Series,
    desired_early_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split each class explicitly so both partitions retain every class."""
    rng = np.random.default_rng(seed)
    inner_parts: list[np.ndarray] = []
    early_parts: list[np.ndarray] = []
    for target_class in sorted(y.iloc[fit_index].unique()):
        class_index = fit_index[y.iloc[fit_index].to_numpy() == target_class].copy()
        rng.shuffle(class_index)
        proportional = round(desired_early_size * len(class_index) / len(fit_index))
        class_early_size = min(max(1, proportional), len(class_index) - 1)
        early_parts.append(class_index[:class_early_size])
        inner_parts.append(class_index[class_early_size:])
    inner_index = np.concatenate(inner_parts)
    early_index = np.concatenate(early_parts)
    rng.shuffle(inner_index)
    rng.shuffle(early_index)
    return inner_index, early_index


def train_ensemble(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: TrainingConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train fixed-complexity repeated-CV models and average test probabilities."""
    config = config or TrainingConfig()
    if TARGET not in train or train[TARGET].isna().any():
        raise ValueError("training target must be present and non-missing")
    if set(train[TARGET].unique()) != {0, 1}:
        raise ValueError("training target must contain both binary classes")
    y = train[TARGET].astype(int).reset_index(drop=True)
    _validate_config(config, y)
    train_features = engineer_features(train)
    test_features = engineer_features(test)
    if train_features.columns.tolist() != test_features.columns.tolist():
        raise ValueError("engineered training and test columns differ")
    if not len(train_features.columns):
        raise ValueError("at least one engineered feature is required")
    numeric = train_features.select_dtypes(include=np.number).columns.tolist()
    if not numeric:
        raise ValueError("at least one numeric feature is required for XGBoost")
    if numeric != test_features.select_dtypes(include=np.number).columns.tolist():
        raise ValueError("engineered training and test dtypes differ")
    cat_train, cat_test, categorical = _catboost_frames(train_features, test_features)
    xgb_train = train_features[numeric].astype(float)
    xgb_test = test_features[numeric].astype(float)

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
        early_size = max(2, math.ceil(0.15 * len(fit_index)))
        inner_fit, early_index = _stratified_early_split(
            fit_index,
            y,
            desired_early_size=early_size,
            seed=fold_seed,
        )

        cat_tuner = _cat_model(config.cat_iterations, fold_seed)
        cat_tuner.fit(
            cat_train.iloc[inner_fit],
            y.iloc[inner_fit],
            cat_features=categorical,
            eval_set=(cat_train.iloc[early_index], y.iloc[early_index]),
            early_stopping_rounds=config.early_stopping_rounds,
            verbose=False,
        )
        cat_best = max(1, cat_tuner.get_best_iteration() + 1)
        cat_model = _cat_model(cat_best, fold_seed)
        cat_model.fit(
            cat_train.iloc[fit_index],
            y.iloc[fit_index],
            cat_features=categorical,
            verbose=False,
        )
        cat_valid = cat_model.predict_proba(cat_train.iloc[valid_index])[:, 1]
        oof_cat[repeat, valid_index] = cat_valid
        test_cat.append(cat_model.predict_proba(cat_test)[:, 1])

        xgb_tuner = _xgb_model(
            config.xgb_iterations,
            fold_seed,
            config.early_stopping_rounds,
        )
        xgb_tuner.fit(
            xgb_train.iloc[inner_fit],
            y.iloc[inner_fit],
            eval_set=[(xgb_train.iloc[early_index], y.iloc[early_index])],
            verbose=False,
        )
        xgb_best = max(1, xgb_tuner.best_iteration + 1)
        xgb_model = _xgb_model(xgb_best, fold_seed)
        xgb_model.fit(xgb_train.iloc[fit_index], y.iloc[fit_index], verbose=False)
        xgb_valid = xgb_model.predict_proba(xgb_train.iloc[valid_index])[:, 1]
        oof_xgb[repeat, valid_index] = xgb_valid
        test_xgb.append(xgb_model.predict_proba(xgb_test)[:, 1])

        fold_metrics.append(
            {
                "repeat": repeat,
                "fold": fold,
                "cat_auc": float(roc_auc_score(y.iloc[valid_index], cat_valid)),
                "xgb_auc": float(roc_auc_score(y.iloc[valid_index], xgb_valid)),
                "cat_best_iteration": cat_best,
                "xgb_best_iteration": xgb_best,
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

    predictions = 0.5 * np.mean(test_cat, axis=0) + 0.5 * np.mean(test_xgb, axis=0)
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
    in_range = (0 <= predictions) & (predictions <= 1)
    if not np.isfinite(predictions).all() or not in_range.all():
        raise ValueError("predictions must be finite and within [0, 1]")
    if sample[IDENTIFIER].tolist() != test[IDENTIFIER].tolist():
        raise ValueError("sample and test identifiers are not aligned")

    submission = sample.copy()
    submission[TARGET] = predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    audit["file_sha256"] = {
        name: _file_sha256(args.data_dir / name)
        for name in ("train.csv", "test.csv", "submit_sample.csv")
    }
    config = TrainingConfig(folds=args.folds, repeats=args.repeats, seed=args.seed)
    predictions, metrics = train_ensemble(train, test, config)
    metrics["dependencies"] = {
        package: importlib.metadata.version(package)
        for package in ("numpy", "pandas", "scikit-learn", "catboost", "xgboost")
    }

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
