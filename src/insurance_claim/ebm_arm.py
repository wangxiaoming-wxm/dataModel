from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "label"
IDENTIFIER = "id"
FOLD_SEEDS = (42, 7, 123)
SHUFFLE_SEEDS = (0, 1, 2)

EBM_INTERACTIONS = (
    ("days", "condition"),
    ("days_log1p", "condition"),
    ("days", "age_range"),
    ("days", "livability"),
    ("days", "region"),
    ("days", "source"),
    ("condition", "region"),
    ("condition", "source"),
    ("t3_value", "source"),
    ("cc", "V"),
    ("V", "max_g"),
)


@dataclass(frozen=True)
class EBMConfig:
    folds: int = 5
    max_bins: int = 256
    max_interaction_bins: int = 32
    outer_bags: int = 8
    learning_rate: float = 0.03
    max_rounds: int = 4000
    early_stopping_rounds: int = 100
    min_samples_leaf: int = 10
    max_leaves: int = 3
    n_jobs: int = -1


def build_ebm_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic row-local features for a heterogeneous GAM arm."""
    features = frame.drop(columns=[IDENTIFIER, TARGET], errors="ignore").copy()

    month = features["month"].astype("string")
    version = features["version"].astype("string")
    t3 = features["t3"].astype("string")
    source = features["source"].astype("string")

    features["month_number"] = pd.to_numeric(
        month.str.extract(r"^M(\d+)$", expand=False), errors="coerce"
    )
    features["version_number"] = pd.to_numeric(
        version.str.extract(r"^v(\d+)$", expand=False), errors="coerce"
    )
    t3_parts = t3.str.extract(r"^(-?\d+(?:\.\d+)?)([A-Za-z]+)$")
    features["t3_value"] = pd.to_numeric(t3_parts[0], errors="coerce")
    features["t3_suffix"] = t3_parts[1].fillna("__MISSING__")
    source_parts = source.str.extract(r"^CAR_(\d+)\|ENG_(\d+)$")
    features["source_car"] = source_parts[0].fillna("__MISSING__")
    features["source_engine"] = source_parts[1].fillna("__MISSING__")

    days = pd.to_numeric(features["days"], errors="coerce")
    condition = pd.to_numeric(features["condition"], errors="coerce")
    features["days_log1p"] = np.log1p(days.clip(lower=0))
    features["days_sqrt"] = np.sqrt(days.clip(lower=0))
    features["condition_abs"] = condition.abs()
    features["condition_log1p_abs"] = np.log1p(condition.abs())
    features["condition_missing"] = condition.isna().astype("int8")
    features["days_condition_product"] = days * condition

    x_columns = [
        column for column in features if column.startswith("x") and column[1:].isdigit()
    ]
    if x_columns:
        vectors = features[x_columns].apply(pd.to_numeric, errors="coerce")
        features["x_mean"] = vectors.mean(axis=1)
        features["x_std"] = vectors.std(axis=1, ddof=0)
        features["x_min"] = vectors.min(axis=1)
        features["x_max"] = vectors.max(axis=1)
        features["x_l1"] = vectors.abs().sum(axis=1, min_count=1)
        features["x_l2"] = np.sqrt(
            vectors.astype(float).pow(2).sum(axis=1, min_count=1)
        )

    for column in features.select_dtypes(exclude=np.number):
        features[column] = (
            features[column].astype("string").fillna("__MISSING__").astype(str)
        )
    for column in features.select_dtypes(include=np.number):
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features.replace([np.inf, -np.inf], np.nan)


def infer_feature_types(features: pd.DataFrame) -> list[str]:
    return [
        "continuous" if pd.api.types.is_numeric_dtype(features[column]) else "nominal"
        for column in features
    ]


def _cache_signature(
    train_features: pd.DataFrame,
    y: np.ndarray,
    test_features: pd.DataFrame,
    config: EBMConfig,
    prefix: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        pd.util.hash_pandas_object(train_features, index=True).to_numpy().tobytes()
    )
    digest.update(np.asarray(y).tobytes())
    digest.update(
        pd.util.hash_pandas_object(test_features, index=True).to_numpy().tobytes()
    )
    digest.update(
        json.dumps(
            {
                "train_schema": [
                    (column, str(train_features[column].dtype))
                    for column in train_features
                ],
                "test_schema": [
                    (column, str(test_features[column].dtype))
                    for column in test_features
                ],
                "interactions": EBM_INTERACTIONS,
            },
            sort_keys=True,
        ).encode()
    )
    digest.update(Path(__file__).read_bytes())
    digest.update(
        json.dumps(
            {"config": asdict(config), "prefix": prefix},
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()


def _atomic_savez(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _load_matching(path: Path, signature: str):
    saved = np.load(path)
    if "signature" not in saved or str(saved["signature"]) != signature:
        saved.close()
        raise ValueError(f"stale or incompatible cache: {path}")
    return saved


def _model(config: EBMConfig, seed: int, features: pd.DataFrame):
    interactions = [
        pair for pair in EBM_INTERACTIONS if pair[0] in features and pair[1] in features
    ]
    return ExplainableBoostingClassifier(
        feature_names=features.columns.tolist(),
        feature_types=infer_feature_types(features),
        max_bins=config.max_bins,
        max_interaction_bins=config.max_interaction_bins,
        interactions=interactions,
        validation_size=0.15,
        outer_bags=config.outer_bags,
        inner_bags=0,
        learning_rate=config.learning_rate,
        max_rounds=config.max_rounds,
        early_stopping_rounds=config.early_stopping_rounds,
        min_samples_leaf=config.min_samples_leaf,
        max_leaves=config.max_leaves,
        n_jobs=config.n_jobs,
        random_state=seed,
    )


def run_cv_seed(
    train_features: pd.DataFrame,
    y: np.ndarray,
    test_features: pd.DataFrame,
    seed: int,
    output_dir: Path,
    config: EBMConfig,
    prefix: str = "real",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run or resume one complete outer-CV seed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = _cache_signature(train_features, y, test_features, config, prefix)
    final_path = output_dir / f"{prefix}_seed{seed}.npz"
    if final_path.exists():
        saved = _load_matching(final_path, signature)
        metrics = json.loads(str(saved["metrics"]))
        return saved["oof"], saved["test"], metrics

    partial_path = output_dir / f"{prefix}_seed{seed}_partial.npz"
    oof = np.zeros(len(y), dtype=float)
    test_sum = np.zeros(len(test_features), dtype=float)
    completed: list[int] = []
    fold_auc: list[float] = []
    if partial_path.exists():
        saved = _load_matching(partial_path, signature)
        oof = saved["oof"]
        test_sum = saved["test_sum"]
        completed = saved["completed"].astype(int).tolist()
        fold_auc = saved["fold_auc"].astype(float).tolist()

    splitter = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=seed)
    for fold, (fit_index, valid_index) in enumerate(splitter.split(train_features, y)):
        if fold in completed:
            continue
        model = _model(config, seed + fold * 1009, train_features)
        model.fit(train_features.iloc[fit_index], y[fit_index])
        valid_prediction = model.predict_proba(train_features.iloc[valid_index])[:, 1]
        oof[valid_index] = valid_prediction
        test_sum += model.predict_proba(test_features)[:, 1]
        fold_auc.append(float(roc_auc_score(y[valid_index], valid_prediction)))
        completed.append(fold)
        _atomic_savez(
            partial_path,
            oof=oof,
            test_sum=test_sum,
            completed=np.asarray(completed),
            fold_auc=np.asarray(fold_auc),
            signature=signature,
        )
        print(
            f"{prefix} seed={seed} fold={fold} auc={fold_auc[-1]:.6f}",
            flush=True,
        )

    test_prediction = test_sum / config.folds
    metrics = {
        "prefix": prefix,
        "seed": seed,
        "fold_auc": fold_auc,
        "pooled_auc": float(roc_auc_score(y, oof)),
    }
    _atomic_savez(
        final_path,
        oof=oof,
        test=test_prediction,
        metrics=json.dumps(metrics),
        signature=signature,
    )
    partial_path.unlink(missing_ok=True)
    return oof, test_prediction, metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:  # pragma: no cover - exercised by full competition runs
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ebm_arm"))
    parser.add_argument("--mode", choices=("screen", "gate"), default="screen")
    parser.add_argument("--v1-oof", type=Path)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    y = train[TARGET].to_numpy(dtype=int)
    train_features = build_ebm_features(train)
    test_features = build_ebm_features(test)
    if train_features.columns.tolist() != test_features.columns.tolist():
        raise ValueError("engineered train/test columns differ")
    if infer_feature_types(train_features) != infer_feature_types(test_features):
        raise ValueError("engineered train/test dtypes differ")

    config = EBMConfig()
    seeds = FOLD_SEEDS[:1] if args.mode == "screen" else FOLD_SEEDS
    real_results = [
        run_cv_seed(
            train_features,
            y,
            test_features,
            seed,
            args.output_dir,
            config,
        )
        for seed in seeds
    ]
    pooled_oof = np.mean([result[0] for result in real_results], axis=0)
    pooled_test = np.mean([result[1] for result in real_results], axis=0)
    report: dict[str, Any] = {
        "mode": args.mode,
        "config": asdict(config),
        "fold_seeds": list(seeds),
        "seed_auc": [result[2]["pooled_auc"] for result in real_results],
        "pooled_auc": float(roc_auc_score(y, pooled_oof)),
        "data_sha256": {
            name: _sha256(args.data_dir / name) for name in ("train.csv", "test.csv")
        },
        "submission_created": False,
    }

    if args.v1_oof:
        v1_oof = np.load(args.v1_oof)
        report["v1_auc"] = float(roc_auc_score(y, v1_oof))
        report["spearman_vs_v1_oof"] = float(spearmanr(pooled_oof, v1_oof).statistic)

    if args.mode == "gate":
        shuffled_auc = []
        for shuffle_seed in SHUFFLE_SEEDS:
            shuffled_y = np.random.default_rng(shuffle_seed).permutation(y)
            _, _, metrics = run_cv_seed(
                train_features,
                shuffled_y,
                test_features,
                shuffle_seed,
                args.output_dir,
                config,
                prefix="shuffled",
            )
            shuffled_auc.append(metrics["pooled_auc"])
        report["shuffled_seed_auc"] = shuffled_auc
        report["shuffled_mean"] = float(np.mean(shuffled_auc))
        report["passes_shuffled_gate"] = 0.47 < report["shuffled_mean"] < 0.53

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "pooled_predictions.npz",
        oof=pooled_oof,
        test=pooled_test,
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
