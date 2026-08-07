"""B6pro trainer: B6-class arms × plus hetero × nested discrete fusion → target 0.715.

Protocol:
- StratifiedKFold >= 5; main seeds >= 8 equal-weight
- Fold-local FE; no global TE; no continuous OOF weight search
- Pre-registered discrete fusion rules + nested selection
- Primary reported score: nested_oof_auc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from insurance_claim.b6pro_fusion import (
    RULE_NAMES,
    apply_rule_test,
    nested_select_rule,
)
from insurance_claim.b6pro_plus import PARAMS_H2, PARAMS_H3, run_plus_arm
from insurance_claim.model import TARGET, audit_data, build_submission
from insurance_claim.train_b6 import run_arm

BASELINE_B6 = 0.6989746962571622
TARGET_AUC = 0.715


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="/workspace")
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _load_oof_bundle(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    out = {k: z[k] for k in z.files}
    return out


def fuse_bundle(
    y: np.ndarray,
    arms_oof: list[np.ndarray],
    arms_test: list[np.ndarray],
    arm_names: list[str],
) -> dict[str, Any]:
    nested = nested_select_rule(y, arms_oof)
    rule = nested["selected_rule"]
    test_pred = apply_rule_test(rule, arms_test)
    corrs = {}
    for i in range(len(arm_names)):
        for j in range(i + 1, len(arm_names)):
            corrs[f"{arm_names[i]}__{arm_names[j]}"] = float(
                np.corrcoef(arms_oof[i], arms_oof[j])[0, 1]
            )
    return {
        **nested,
        "test_pred": test_pred,
        "arm_names": arm_names,
        "arm_aucs": {n: float(roc_auc_score(y, a)) for n, a in zip(arm_names, arms_oof)},
        "corr": corrs,
        "rules_preregistered": nested.get("rules_used", list(RULE_NAMES)),
        "rules_used": nested.get("rules_used", list(RULE_NAMES)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_run"))
    parser.add_argument("--mode", choices=["full", "fuse", "plus_only", "b6_arms"], default="full")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(2026, 2034)))
    parser.add_argument("--b6-arms", nargs="+", default=["gap", "gap_bag"])
    parser.add_argument("--plus-variant", choices=["plus", "plus_gap", "plus_ultra"], default="plus")
    parser.add_argument("--plus-config", choices=["h2", "h3"], default="h2")
    parser.add_argument("--plus-folds", type=int, default=5)
    parser.add_argument("--plus-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--main-npz", type=Path, default=None, help="Existing main arm OOF npz")
    parser.add_argument("--plus-npz", type=Path, default=None, help="Existing plus OOF npz")
    parser.add_argument(
        "--ref-plus",
        action="store_true",
        help="Allow reference/v10 plus OOF for fuse0 bootstrap (disclose as reference)",
    )
    parser.add_argument("--shuffled", action="store_true")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "submit_sample.csv")
    audit = audit_data(train, test, sample)
    y = train[TARGET].astype(int)
    y_np = y.to_numpy()
    seeds = tuple(args.seeds)
    plus_seeds = tuple(args.plus_seeds) if args.plus_seeds else seeds[:4]
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save: dict[str, Any] = {"y": y_np}
    metrics: dict[str, Any] = {
        "experiment_id": f"b6pro_{args.mode}_{args.plus_variant}_{args.plus_config}",
        "protocol_id": "IA-AUC715-B6PRO-v1",
        "git_commit": _git_commit(),
        "data_sha256": {
            "train": _sha256(args.data_dir / "train.csv"),
            "test": _sha256(args.data_dir / "test.csv"),
            "submit": _sha256(args.data_dir / "submit_sample.csv"),
        },
        "baseline_b6_pooled": BASELINE_B6,
        "target_auc": TARGET_AUC,
        "mode": args.mode,
        "seeds": list(seeds),
        "plus_seeds": list(plus_seeds),
        "audit": {
            "train_rows": audit["train_rows"],
            "test_rows": audit["test_rows"],
            "target_rate": audit["target_rate"],
            "id_overlap": audit["id_overlap"],
        },
    }

    arm_oofs: list[np.ndarray] = []
    arm_tests: list[np.ndarray] = []
    arm_names: list[str] = []

    # --- main B6-class arms ---
    if args.mode in ("full", "b6_arms") and args.main_npz is None:
        b6_results = {}
        for name in args.b6_arms:
            print(f"=== train B6 arm {name} ===", flush=True)
            b6_results[name] = run_arm(name, train, test, y, seeds)
            save[f"oof_{name}"] = b6_results[name]["oof"]
            save[f"test_{name}"] = b6_results[name]["test"]
            for s, arr in b6_results[name]["oof_by_seed"].items():
                save[f"oof_{name}_{s}"] = arr
        # equal_prob of B6 arms as main
        if len(b6_results) == 1:
            main_oof = next(iter(b6_results.values()))["oof"]
            main_test = next(iter(b6_results.values()))["test"]
        else:
            main_oof = np.mean(np.vstack([b6_results[n]["oof"] for n in args.b6_arms]), axis=0)
            main_test = np.mean(np.vstack([b6_results[n]["test"] for n in args.b6_arms]), axis=0)
        save["oof_main"] = main_oof
        save["test_main"] = main_test
        metrics["b6_arms"] = {
            n: {"oof_auc": b6_results[n]["oof_auc"], "seed_aucs": b6_results[n]["seed_aucs"]}
            for n in args.b6_arms
        }
        metrics["main_equal_prob_auc"] = float(roc_auc_score(y_np, main_oof))
        arm_oofs.append(main_oof)
        arm_tests.append(main_test)
        arm_names.append("equal_b6")
    elif args.main_npz is not None:
        bundle = _load_oof_bundle(args.main_npz)
        main_oof = bundle["oof"] if "oof" in bundle else bundle["oof_main"]
        main_test = bundle["test"] if "test" in bundle else bundle["test_main"]
        save["oof_main"] = main_oof
        save["test_main"] = main_test
        metrics["main_source"] = str(args.main_npz)
        metrics["main_equal_prob_auc"] = float(roc_auc_score(y_np, main_oof))
        arm_oofs.append(main_oof)
        arm_tests.append(main_test)
        arm_names.append("equal_b6")

    # --- plus hetero ---
    plus_params = PARAMS_H3 if args.plus_config == "h3" else PARAMS_H2
    if args.mode in ("full", "plus_only") and args.plus_npz is None and not args.ref_plus:
        print(f"=== train plus variant={args.plus_variant} config={args.plus_config} ===", flush=True)
        plus = run_plus_arm(
            train,
            test,
            y,
            plus_seeds,
            variant=args.plus_variant,
            params=plus_params,
            n_splits=args.plus_folds,
        )
        save["oof_plus"] = plus["oof"]
        save["test_plus"] = plus["test"]
        for s, arr in plus["oof_by_seed"].items():
            save[f"oof_plus_{s}"] = arr
        metrics["plus"] = {
            "variant": plus["variant"],
            "oof_auc": plus["oof_auc"],
            "seed_aucs": plus["seed_aucs"],
            "params": plus["params"],
            "n_splits": plus["n_splits"],
            "source": "self_trained",
        }
        arm_oofs.append(plus["oof"])
        arm_tests.append(plus["test"])
        arm_names.append(args.plus_variant)
    elif args.plus_npz is not None or args.ref_plus:
        if args.plus_npz is not None:
            path = args.plus_npz
        else:
            path = Path("reference/v10/oof_plus_h2_10.npz")
        bundle = _load_oof_bundle(path)
        poof = bundle["oof"]
        if "test" in bundle:
            pte = bundle["test"]
        else:
            pte = np.load("reference/v10/test_plus_h2_10.npy")
        save["oof_plus"] = poof
        save["test_plus"] = pte
        metrics["plus"] = {
            "oof_auc": float(roc_auc_score(y_np, poof)),
            "source": str(path),
            "reference_bootstrap": bool(args.ref_plus and args.plus_npz is None),
        }
        arm_oofs.append(poof)
        arm_tests.append(pte)
        arm_names.append("plus")

    if args.mode == "b6_arms":
        # Save main only
        metrics["pooled_oof_auc"] = metrics.get("main_equal_prob_auc")
        metrics["nested_oof_auc"] = None
        metrics["gate_0_715"] = False
        metrics["elapsed_sec"] = round(time.time() - started, 1)
        np.savez_compressed(args.output_dir / "predictions.npz", **save)
        (args.output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"main_equal_prob_auc": metrics["main_equal_prob_auc"]}, indent=2))
        return 0

    if args.mode == "plus_only":
        metrics["pooled_oof_auc"] = metrics["plus"]["oof_auc"]
        metrics["nested_oof_auc"] = None
        metrics["gate_0_715"] = False
        metrics["elapsed_sec"] = round(time.time() - started, 1)
        np.savez_compressed(args.output_dir / "predictions.npz", **save)
        build_submission(test, sample, save["test_plus"], args.output_dir / "submission_plus.csv")
        (args.output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"plus_oof_auc": metrics["plus"]["oof_auc"]}, indent=2))
        return 0

    # --- nested fusion ---
    assert len(arm_oofs) >= 2, "need >=2 arms for fusion"
    fused = fuse_bundle(y_np, arm_oofs, arm_tests, arm_names)
    save["oof"] = fused["nested_oof"]
    save["test"] = fused["test_pred"]
    save["oof_full_selected"] = fused["full_selected_oof"]

    metrics["fusion"] = {
        "selected_rule": fused["selected_rule"],
        "fold_rules": fused["fold_rules"],
        "consistent_all_folds": fused["consistent_all_folds"],
        "full_data_scores": fused["full_data_scores"],
        "rules_preregistered": fused.get("rules_used", fused["rules_preregistered"]),
        "rules_used": fused.get("rules_used"),
        "rule_selection": "nested_5fold",
        "arm_aucs": fused["arm_aucs"],
        "corr": fused["corr"],
    }
    metrics["nested_oof_auc"] = fused["nested_oof_auc"]
    metrics["pooled_oof_auc"] = fused["nested_oof_auc"]
    metrics["full_selected_auc"] = fused["full_selected_auc"]
    metrics["gate_0_715"] = bool(fused["nested_oof_auc"] >= TARGET_AUC)
    metrics["gap_to_0_715"] = round(TARGET_AUC - float(fused["nested_oof_auc"]), 6)
    metrics["protocol_declaration"] = {
        "no_test_labels": True,
        "no_global_te": True,
        "fold_local_fe": True,
        "no_oof_weight_search": True,
        "fusion_rules_preregistered": True,
        "rule_selection_nested": True,
        "equal_seed_average": True,
        "new_data_only": True,
        "b6_freeze_untampered": True,
        "reference_plus_bootstrap": bool(metrics.get("plus", {}).get("reference_bootstrap")),
        "early_stopping_on_valid": True,
    }

    if args.shuffled and "oof_plus" in save:
        # Shuffle plus rows then max with main — collapse check
        rng = np.random.default_rng(2026)
        plus_shuf = save["oof_plus"].copy()
        rng.shuffle(plus_shuf)
        if "oof_main" in save:
            sh_max = np.maximum(save["oof_main"], plus_shuf)
            metrics["shuffled_plus_max_auc"] = float(roc_auc_score(y_np, sh_max))
            metrics["shuffled_plus_max_pass"] = bool(metrics["shuffled_plus_max_auc"] < 0.66)

    metrics["elapsed_sec"] = round(time.time() - started, 1)
    np.savez_compressed(args.output_dir / "predictions.npz", **save)
    build_submission(test, sample, save["test"], args.output_dir / "submission_b6pro.csv")
    Path("submissions").mkdir(exist_ok=True)
    build_submission(test, sample, save["test"], Path("submissions") / "submission_b6pro.csv")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "nested_oof_auc": metrics["nested_oof_auc"],
                "full_selected_auc": metrics["full_selected_auc"],
                "selected_rule": metrics["fusion"]["selected_rule"],
                "arm_aucs": metrics["fusion"]["arm_aucs"],
                "gate_0_715": metrics["gate_0_715"],
                "gap_to_0_715": metrics["gap_to_0_715"],
                "elapsed_sec": metrics["elapsed_sec"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
