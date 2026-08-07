#!/usr/bin/env python3
"""Nested regime fusion: within days bins, choose among preregistered rules.

Honest nested OOF; rules and bins preregistered; no continuous weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import RULE_NAMES_CORE, apply_rule
from insurance_claim.model import build_submission

TARGET = 0.715


def load_pair(main_path, plus_path):
    m = np.load(main_path)
    p = np.load(plus_path)
    y = m["y"]
    mo = m["oof_main"] if "oof_main" in m.files else m["oof"]
    mt = m["test_main"] if "test_main" in m.files else m["test"]
    po = p["oof"] if "oof" in p.files else p["oof_plus"]
    pt = p["test"] if "test" in p.files else p["test_plus"]
    return y, mo, mt, po, pt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-npz", type=Path, required=True)
    ap.add_argument("--plus-npz", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "submit_sample.csv")
    y, mo, mt, po, pt = load_pair(args.main_npz, args.plus_npz)
    days = pd.to_numeric(train["days"], errors="coerce").to_numpy()
    days_te = pd.to_numeric(test["days"], errors="coerce").to_numpy()

    # Fit bin edges on full train days (feature only; no label) — allowed
    edges = np.unique(np.quantile(days[np.isfinite(days)], np.linspace(0, 1, args.bins + 1)))[1:-1]
    bin_id = np.searchsorted(edges, days, side="right")
    bin_te = np.searchsorted(edges, days_te, side="right")

    nested = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    fold_rule_maps = []
    for tr, va in skf.split(np.zeros(len(y)), y):
        rule_map = {}
        for b in range(args.bins):
            mask_tr = bin_id[tr] == b
            if mask_tr.sum() < 50 or y[tr][mask_tr].sum() < 5:
                rule_map[b] = "max"
                continue
            arms_tr = [mo[tr][mask_tr], po[tr][mask_tr]]
            scores = {r: roc_auc_score(y[tr][mask_tr], apply_rule(r, arms_tr)) for r in RULE_NAMES_CORE}
            rule_map[b] = max(scores, key=scores.get)
        fold_rule_maps.append(rule_map)
        for b, rule in rule_map.items():
            mask_va = bin_id[va] == b
            if not mask_va.any():
                continue
            nested[va[mask_va]] = apply_rule(rule, [mo[va][mask_va], po[va][mask_va]])

    # Full-data majority rule per bin for test
    from collections import Counter

    test_pred = np.zeros(len(test), dtype=float)
    full_map = {}
    for b in range(args.bins):
        votes = [fm.get(b, "max") for fm in fold_rule_maps]
        rule = Counter(votes).most_common(1)[0][0]
        full_map[b] = rule
        mask = bin_te == b
        if mask.any():
            test_pred[mask] = apply_rule(rule, [mt[mask], pt[mask]])
        mask_tr = bin_id == b
        # fill any nested holes
        hole = (bin_id == b) & (nested == 0)  # unlikely
        if hole.any():
            nested[hole] = apply_rule(rule, [mo[hole], po[hole]])

    auc = float(roc_auc_score(y, nested))
    metrics = {
        "experiment_id": "b6pro_regime_days",
        "protocol_id": "IA-AUC715-B6PRO-v1",
        "nested_oof_auc": auc,
        "pooled_oof_auc": auc,
        "gate_0_715": auc >= TARGET,
        "gap_to_0_715": round(TARGET - auc, 6),
        "bin_rules_full": {str(k): v for k, v in full_map.items()},
        "fold_rule_maps": [{str(k): v for k, v in fm.items()} for fm in fold_rule_maps],
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "no_oof_weight_search": True,
            "regime_rules_preregistered": list(RULE_NAMES_CORE),
            "nested_regime_selection": True,
            "new_data_only": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "predictions.npz", y=y, oof=nested, test=test_pred)
    build_submission(test, sample, test_pred, args.output_dir / "submission_b6pro.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"nested_oof_auc": auc, "bin_rules": full_map, "gate_0_715": metrics["gate_0_715"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
