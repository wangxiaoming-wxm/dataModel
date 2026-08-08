#!/usr/bin/env python3
"""Fuse multiple OOF npz arms with nested discrete rules (B6pro)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from insurance_claim.b6pro_fusion import RULE_NAMES_EXT, apply_rule_test, nested_select_rule
from insurance_claim.model import build_submission

TARGET = 0.715


def load_arm(path: Path, oof_key: str | None, test_key: str | None):
    z = np.load(path)
    oof = z[oof_key] if oof_key else (z["oof"] if "oof" in z.files else z["oof_main"])
    if test_key:
        te = z[test_key]
    elif "test" in z.files:
        te = z["test"]
    elif "test_main" in z.files:
        te = z["test_main"]
    else:
        te = None
    y = z["y"] if "y" in z.files else None
    return oof, te, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="name=path[:oof_key[:test_key]]")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    ap.add_argument("--extended-rules", action="store_true", default=True)
    args = ap.parse_args()

    names, oofs, tests = [], [], []
    y = None
    for spec in args.arms:
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        path = Path(parts[0])
        oof_key = parts[1] if len(parts) > 1 and parts[1] else None
        test_key = parts[2] if len(parts) > 2 and parts[2] else None
        oof, te, y_arm = load_arm(path, oof_key, test_key)
        if y is None and y_arm is not None:
            y = y_arm
        names.append(name)
        oofs.append(oof)
        tests.append(te)

    assert y is not None
    assert all(t is not None for t in tests)

    t0 = time.time()
    nested = nested_select_rule(y, oofs, rules=RULE_NAMES_EXT if len(oofs) >= 2 else None)
    rule = nested["selected_rule"]
    test_pred = apply_rule_test(rule, tests)
    corrs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            corrs[f"{names[i]}__{names[j]}"] = float(np.corrcoef(oofs[i], oofs[j])[0, 1])

    metrics = {
        "experiment_id": f"b6pro_multifuse_{'_'.join(names)}",
        "protocol_id": "IA-AUC715-B6PRO-v1",
        "arm_names": names,
        "arm_aucs": {n: float(roc_auc_score(y, a)) for n, a in zip(names, oofs)},
        "corr": corrs,
        "fusion": {
            "selected_rule": nested["selected_rule"],
            "fold_rules": nested["fold_rules"],
            "consistent_all_folds": nested["consistent_all_folds"],
            "full_data_scores": nested["full_data_scores"],
            "rules_used": nested.get("rules_used"),
            "rule_selection": "nested_5fold",
        },
        "nested_oof_auc": nested["nested_oof_auc"],
        "pooled_oof_auc": nested["nested_oof_auc"],
        "full_selected_auc": nested["full_selected_auc"],
        "gate_0_715": bool(nested["nested_oof_auc"] >= TARGET),
        "gap_to_0_715": round(TARGET - float(nested["nested_oof_auc"]), 6),
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "no_oof_weight_search": True,
            "fusion_rules_preregistered": True,
            "rule_selection_nested": True,
            "new_data_only": True,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y=y,
        oof=nested["nested_oof"],
        test=test_pred,
        oof_full_selected=nested["full_selected_oof"],
        **{f"oof_{n}": a for n, a in zip(names, oofs)},
        **{f"test_{n}": t for n, t in zip(names, tests)},
    )
    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "submit_sample.csv")
    build_submission(test, sample, test_pred, args.output_dir / "submission_b6pro.csv")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "nested_oof_auc": metrics["nested_oof_auc"],
        "selected_rule": metrics["fusion"]["selected_rule"],
        "arm_aucs": metrics["arm_aucs"],
        "gate_0_715": metrics["gate_0_715"],
        "gap_to_0_715": metrics["gap_to_0_715"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
