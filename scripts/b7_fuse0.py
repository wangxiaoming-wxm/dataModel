"""B7 baseline fuse: B6 gap/gap_bag × V10 plus with nested discrete rules.

Uses frozen OOFs for immediate honest nested score; does not mutate B6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from insurance_claim.b7_fusion import (
    FUSION_RULES,
    all_pair_fusions,
    fuse_pair,
    fuse_three_max,
    fuse_three_mean,
    nested_select_pair,
)
from insurance_claim.model import TARGET, build_submission


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b7_fuse0"))
    ap.add_argument(
        "--main-npz",
        type=Path,
        default=Path("artifacts/b6_gapbag_8seed/predictions.npz"),
        help="B6 gap+gap_bag predictions (provides oof_gap, oof_gap_bag, test_*)",
    )
    ap.add_argument(
        "--plus-npz",
        type=Path,
        default=Path("reference/v10/oof_plus_h2_10.npz"),
    )
    ap.add_argument(
        "--plus-test",
        type=Path,
        default=Path("reference/v10/test_plus_h2_10.npy"),
    )
    ap.add_argument("--main-arm", choices=["gap_bag", "gap", "equal_b6"], default="gap_bag")
    args = ap.parse_args()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train[TARGET].astype(int).to_numpy()

    main_z = np.load(args.main_npz)
    plus_z = np.load(args.plus_npz)
    oof_plus = plus_z["oof"]
    te_plus = np.load(args.plus_test)

    if args.main_arm == "gap_bag":
        oof_a, te_a, a_name = main_z["oof_gap_bag"], main_z["test_gap_bag"], "gap_bag"
    elif args.main_arm == "gap":
        oof_a, te_a, a_name = main_z["oof_gap"], main_z["test_gap"], "gap"
    else:
        oof_a = 0.5 * (main_z["oof_gap"] + main_z["oof_gap_bag"])
        te_a = 0.5 * (main_z["test_gap"] + main_z["test_gap_bag"])
        a_name = "equal_b6"

    # Align lengths
    assert len(oof_a) == len(y) == len(oof_plus)

    nested = nested_select_pair(oof_a, oof_plus, y)
    rule = nested["selected_rule"]
    full = all_pair_fusions(oof_a, oof_plus)
    full_aucs = {k: float(roc_auc_score(y, v)) for k, v in full.items()}

    # Three-way with gap + gap_bag + plus (mean / max only; disclosure)
    oof_gap, oof_bag = main_z["oof_gap"], main_z["oof_gap_bag"]
    three = {
        "mean3": float(roc_auc_score(y, fuse_three_mean([oof_gap, oof_bag, oof_plus]))),
        "max3": float(roc_auc_score(y, fuse_three_max([oof_gap, oof_bag, oof_plus]))),
        "max(equal_b6,plus)": float(
            roc_auc_score(y, np.maximum(0.5 * (oof_gap + oof_bag), oof_plus))
        ),
    }

    # Primary delivery: nested-selected rule on (main_arm, plus)
    oof_final = fuse_pair(oof_a, oof_plus, rule)
    # For rank_mean on test, use ranks then min-max to (0,1)
    if rule == "rank_mean":
        from scipy.stats import rankdata

        te_final = 0.5 * (rankdata(te_a) + rankdata(te_plus))
        te_final = (te_final - te_final.min()) / (te_final.max() - te_final.min() + 1e-12)
    else:
        te_final = fuse_pair(te_a, te_plus, rule)

    # Prefer full-data application of the nested-selected rule (V10 style)
    oof_submit = full[rule]
    te_submit = te_final
    pooled = float(roc_auc_score(y, oof_submit))

    metrics = {
        "experiment_id": "b7_fuse0_nested",
        "main_arm": a_name,
        "plus_source": str(args.plus_npz),
        "nested": {
            "selected_rule": rule,
            "nested_oof_auc": nested["nested_oof_auc"],
            "votes": nested["nested_rule_votes"],
            "consistent_all_folds": nested["consistent_all_folds"],
            "full_data_scores": full_aucs,
        },
        "three_arm_disclosure": three,
        "arm_aucs": {
            a_name: float(roc_auc_score(y, oof_a)),
            "plus": float(roc_auc_score(y, oof_plus)),
            "gap": float(roc_auc_score(y, oof_gap)),
            "gap_bag": float(roc_auc_score(y, oof_bag)),
        },
        "corr_main_plus": float(np.corrcoef(oof_a, oof_plus)[0, 1]),
        "pooled_oof_auc": pooled,
        "nested_oof_auc": nested["nested_oof_auc"],
        "gate_0_71": bool(nested["nested_oof_auc"] >= 0.71),
        "gap_to_0_71": round(0.71 - nested["nested_oof_auc"], 6),
        "baseline_b6": 0.6989746962571622,
        "baseline_v10_nested": 0.7013149650619108,
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "no_oof_weight_search": True,
            "fusion_rules_preregistered": list(FUSION_RULES),
            "rule_selection": "nested_5fold",
            "b6_freeze_untampered": True,
            "new_data_only": True,
        },
        "data_sha256": {
            "train": _sha(Path("train.csv")),
            "test": _sha(Path("test.csv")),
        },
    }

    # shuffled sanity on fused scores (permute plus)
    rng = np.random.default_rng(2026)
    plus_shuf = oof_plus.copy()
    rng.shuffle(plus_shuf)
    sh_max = float(roc_auc_score(y, np.maximum(oof_a, plus_shuf)))
    metrics["shuffled_plus_max_auc"] = sh_max
    metrics["shuffled_plus_max_pass"] = bool(sh_max < 0.66)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "predictions.npz",
        oof=oof_submit,
        test=te_submit,
        y=y,
        nested_oof=nested["nested_oof"],
        oof_main=oof_a,
        oof_plus=oof_plus,
        test_main=te_a,
        test_plus=te_plus,
    )
    build_submission(test, sample, te_submit, out / "submission_b7.csv")
    Path("submissions").mkdir(exist_ok=True)
    build_submission(test, sample, te_submit, Path("submissions") / "submission_b7_fuse0.csv")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: metrics[k] for k in (
        "main_arm", "nested", "three_arm_disclosure", "arm_aucs", "corr_main_plus",
        "pooled_oof_auc", "nested_oof_auc", "gate_0_71", "gap_to_0_71",
        "shuffled_plus_max_auc", "shuffled_plus_max_pass",
    )}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
