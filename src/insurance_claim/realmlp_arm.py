from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pytabkit import RealMLP_TD_Classifier
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.ebm_arm import (
    FOLD_SEEDS,
    SHUFFLE_SEEDS,
    build_ebm_features,
)

TARGET = "label"


@dataclass(frozen=True)
class RealMLPConfig:
    folds: int = 5
    epochs: int = 256
    batch_size: int = 256
    hidden_width: int = 256
    hidden_layers: int = 3
    time_limit_seconds: int = 900
    n_threads: int = 0


def prepare_fold_frames(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Apply fold-local missing-value handling required by RealMLP."""
    train_out = train.copy()
    valid_out = valid.copy()
    test_out = test.copy()
    categorical = train_out.select_dtypes(exclude=np.number).columns.tolist()
    numeric = [column for column in train_out if column not in categorical]

    for column in categorical:
        for frame in (train_out, valid_out, test_out):
            frame[column] = (
                frame[column].astype("string").fillna("__MISSING__").astype(str)
            )
    for column in numeric:
        median = pd.to_numeric(train_out[column], errors="coerce").median()
        fill_value = float(median) if pd.notna(median) else 0.0
        for frame in (train_out, valid_out, test_out):
            frame[column] = (
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(fill_value)
                .astype(float)
            )
    return train_out, valid_out, test_out, categorical


def _model(config: RealMLPConfig, seed: int, tmp_folder: Path):
    threads = config.n_threads or None
    return RealMLP_TD_Classifier(
        device="cpu",
        random_state=seed,
        n_cv=1,
        n_refit=0,
        n_repeats=1,
        val_fraction=0.15,
        n_threads=threads,
        tmp_folder=tmp_folder,
        verbosity=1,
        val_metric_name="cross_entropy",
        n_epochs=config.epochs,
        batch_size=config.batch_size,
        hidden_width=config.hidden_width,
        n_hidden_layers=config.hidden_layers,
        use_ls=False,
    )


def run_cv_seed(
    train_features: pd.DataFrame,
    y: np.ndarray,
    test_features: pd.DataFrame,
    seed: int,
    output_dir: Path,
    config: RealMLPConfig,
    prefix: str = "real",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run one outer-CV seed with fold-level durable checkpoints."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"{prefix}_seed{seed}.npz"
    if final_path.exists():
        saved = np.load(final_path)
        return saved["oof"], saved["test"], json.loads(str(saved["metrics"]))

    partial_path = output_dir / f"{prefix}_seed{seed}_partial.npz"
    oof = np.zeros(len(y), dtype=float)
    test_sum = np.zeros(len(test_features), dtype=float)
    completed: list[int] = []
    fold_auc: list[float] = []
    if partial_path.exists():
        saved = np.load(partial_path)
        oof = saved["oof"]
        test_sum = saved["test_sum"]
        completed = saved["completed"].astype(int).tolist()
        fold_auc = saved["fold_auc"].astype(float).tolist()

    splitter = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=seed)
    for fold, (fit_index, valid_index) in enumerate(splitter.split(train_features, y)):
        if fold in completed:
            continue
        fit, valid, test, categorical = prepare_fold_frames(
            train_features.iloc[fit_index],
            train_features.iloc[valid_index],
            test_features,
        )
        model = _model(
            config,
            seed + fold * 1009,
            output_dir / "tmp" / f"{prefix}_{seed}_{fold}",
        )
        model.fit(
            fit,
            y[fit_index],
            cat_col_names=categorical,
            time_to_fit_in_seconds=config.time_limit_seconds,
        )
        valid_prediction = model.predict_proba(valid)[:, 1]
        oof[valid_index] = valid_prediction
        test_sum += model.predict_proba(test)[:, 1]
        fold_auc.append(float(roc_auc_score(y[valid_index], valid_prediction)))
        completed.append(fold)
        np.savez_compressed(
            partial_path,
            oof=oof,
            test_sum=test_sum,
            completed=np.asarray(completed),
            fold_auc=np.asarray(fold_auc),
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
    np.savez_compressed(
        final_path,
        oof=oof,
        test=test_prediction,
        metrics=json.dumps(metrics),
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
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/realmlp_arm")
    )
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

    config = RealMLPConfig()
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
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ("pytabkit", "torch")
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
        report["passes_shuffled_gate"] = report["shuffled_mean"] < 0.53

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
