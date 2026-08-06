#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

BOOTSTRAP_SEED = 314159
BOOTSTRAP_SAMPLES = 3000


def rank(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / (len(values) + 1.0)


def calibrate_order(
    ranking_score: np.ndarray, reference_probability: np.ndarray
) -> np.ndarray:
    calibrated = np.empty_like(ranking_score, dtype=float)
    calibrated[np.argsort(ranking_score, kind="mergesort")] = np.sort(
        reference_probability
    )
    return calibrated


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_bootstrap(
    y: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, object]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = []
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, len(y), size=len(y))
        if np.unique(y[indices]).size != 2:
            continue
        differences.append(
            roc_auc_score(y[indices], candidate[indices])
            - roc_auc_score(y[indices], reference[indices])
        )
    return {
        "seed": BOOTSTRAP_SEED,
        "samples": len(differences),
        "ci95": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
    }


def write_submission(
    sample: pd.DataFrame,
    test_ids: np.ndarray,
    prediction: np.ndarray,
    path: Path,
) -> dict[str, object]:
    if not np.array_equal(sample["id"].astype(str).to_numpy(), test_ids):
        raise ValueError("sample and package test IDs differ")
    if (
        not np.isfinite(prediction).all()
        or not ((prediction >= 0) & (prediction <= 1)).all()
    ):
        raise ValueError("predictions must be finite within [0, 1]")
    output = sample.copy()
    output["label"] = prediction
    output.to_csv(path, index=False)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(output),
        "min": float(prediction.min()),
        "max": float(prediction.max()),
        "mean": float(prediction.mean()),
        "unique_predictions": int(pd.Series(prediction).nunique()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("final_candidates"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    sample = pd.read_csv(args.data_dir / "submit_sample.csv")
    y = train["label"].to_numpy(dtype=int)
    package_y = np.load(args.package_dir / "y.npy")
    test_ids = np.load(args.package_dir / "test_id.npy", allow_pickle=True).astype(str)
    if not np.array_equal(y, package_y):
        raise ValueError("package labels do not match official training labels")

    v1_oof = np.load(args.package_dir / "oof_v1_3rd.npy")
    v1_test = np.load(args.package_dir / "test_v1_3rd.npy")
    v7_oof = np.load(args.package_dir / "oof_v7_lgbm.npy")
    v7_test = np.load(args.package_dir / "test_v7_lgbm.npy")
    upside_oof = (rank(v1_oof) + rank(v7_oof)) / 2
    upside_test_rank = (rank(v1_test) + rank(v7_test)) / 2
    upside_test = calibrate_order(upside_test_rank, v1_test)

    safe_path = args.output_dir / "submission_1_safe_v1_anchor.csv"
    upside_path = args.output_dir / "submission_2_upside_v1_v7_equal_rank.csv"
    files = {
        "safe_v1": write_submission(sample, test_ids, v1_test, safe_path),
        "upside_v1_v7": write_submission(sample, test_ids, upside_test, upside_path),
    }

    report = {
        "recommendation_order": [
            "safe_v1",
            "upside_v1_v7_conditional",
        ],
        "known_public_anchor": {
            "candidate": "safe_v1",
            "local_oof": float(roc_auc_score(y, v1_oof)),
            "known_public": 0.70236,
            "warning": "prediction-identical to the known third-place file",
        },
        "upside_v1_v7": {
            "method": "fixed equal-rank average; no weight search",
            "local_oof": float(roc_auc_score(y, upside_oof)),
            "v7_local_oof": float(roc_auc_score(y, v7_oof)),
            "delta_vs_v1": float(
                roc_auc_score(y, upside_oof) - roc_auc_score(y, v1_oof)
            ),
            "delta_bootstrap": paired_bootstrap(y, upside_oof, v1_oof),
            "spearman_oof_vs_v1": float(spearmanr(upside_oof, v1_oof).statistic),
            "spearman_test_vs_v1": float(spearmanr(upside_test, v1_test).statistic),
            "public_estimate": "approximately 0.70-0.71; no public anchor",
            "warning": (
                "V7 package lacks full training source/per-seed/shuffled "
                "artifacts; conditional second shot only"
            ),
        },
        "rejected": {
            "minimax_0_55v5_0_45v3": (
                "OOF-searched weights violate R07 and algebraically increase "
                "the V5 component that already dragged V3 public performance"
            ),
            "grok": "honest OOF 0.66165, materially below V1",
        },
        "files": files,
    }
    report_path = args.output_dir / "FINAL_CANDIDATES.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
