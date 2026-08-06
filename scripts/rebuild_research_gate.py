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

BOOTSTRAP_SEED = 73
BOOTSTRAP_SAMPLES = 3000


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
        "method": "paired row bootstrap with replacement",
        "seed": BOOTSTRAP_SEED,
        "requested_samples": BOOTSTRAP_SAMPLES,
        "valid_samples": len(differences),
        "quantiles": [0.025, 0.975],
        "ci95": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--handover-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/research_gate_20260806.json"),
    )
    args = parser.parse_args()

    y = pd.read_csv(args.data_dir / "train.csv")["label"].to_numpy(dtype=int)
    v1_oof = np.load(args.handover_dir / "models/oof_v1_3rd.npy")
    v1_test = np.load(args.handover_dir / "models/test_v1_3rd.npy")
    sem = np.load(args.handover_dir / "artifacts/v4_fast/sem_plus_seed2026.npz")
    ebm = np.load(args.artifacts_dir / "ebm_arm/pooled_predictions.npz")
    realmlp = np.load(args.artifacts_dir / "realmlp_arm/real_seed42_partial.npz")
    tabm = np.load(args.artifacts_dir / "tabm_arm/real_seed42_partial.npz")

    def rank(values: np.ndarray) -> np.ndarray:
        return rankdata(values, method="average") / (len(values) + 1.0)

    blend_oof = (rank(v1_oof) + rank(ebm["oof"])) / 2
    blend_auc = float(roc_auc_score(y, blend_oof))
    v1_auc = float(roc_auc_score(y, v1_oof))

    report = {
        "policy": "NO_FORMAL_SUBMISSION",
        "rebuild": {
            "script": "scripts/rebuild_research_gate.py",
            "command": (
                "PYTHONPATH=src python3 scripts/rebuild_research_gate.py "
                "--handover-dir /path/to/20260806-cursor"
            ),
        },
        "handover_reproduction": {
            "v1_oof": v1_auc,
            "v1_submission_sha256": sha256(
                args.handover_dir / "submissions/submission_v1_3rd_repro.csv"
            ),
            "v4_quick_oof": 0.6921064403419052,
            "v4_quick_oof_evidence": (
                "metadata_only; Q1/DS OOF arrays absent from handover archive"
            ),
            "v4_quick_sha256": sha256(
                args.handover_dir / "submissions/submission_v4_quick.csv"
            ),
            "sem_plus_seed2026_oof": float(roc_auc_score(y, sem["oof"])),
        },
        "arms": {
            "sem_plus": {
                "decision": "REJECT",
                "oof": float(roc_auc_score(y, sem["oof"])),
                "delta_vs_v1": float(roc_auc_score(y, sem["oof"]) - v1_auc),
                "spearman_oof_vs_v1": float(spearmanr(sem["oof"], v1_oof).statistic),
            },
            "ebm": {
                "decision": "REJECT",
                "oof": float(roc_auc_score(y, ebm["oof"])),
                "spearman_oof_vs_v1": float(spearmanr(ebm["oof"], v1_oof).statistic),
                "spearman_test_vs_v1": float(spearmanr(ebm["test"], v1_test).statistic),
                "equal_rank_blend_oof": blend_auc,
                "blend_delta_vs_v1": blend_auc - v1_auc,
                "blend_bootstrap": paired_bootstrap(y, blend_oof, v1_oof),
            },
            "realmlp": {
                "decision": "EARLY_REJECT",
                "completed_folds": realmlp["completed"].tolist(),
                "fold_auc": realmlp["fold_auc"].tolist(),
                "reason": "first two consecutive folds far below V1 range",
            },
            "tabm": {
                "decision": "EARLY_REJECT",
                "completed_folds": tabm["completed"].tolist(),
                "fold_auc": tabm["fold_auc"].tolist(),
                "reason": "first consecutive fold far below V1 range",
            },
        },
        "gate": {
            "new_arm_stable_above_0_693": False,
            "shuffled_required": False,
            "shuffled_reason": (
                "all new arms failed performance screen before promotion"
            ),
            "submit_gate_0_72": False,
            "formal_submission_created": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
