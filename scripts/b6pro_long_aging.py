#!/usr/bin/env python3
"""B6pro long-exposure aging arm + nested fuse with B7 max3 floor.

Target: honest nested OOF AUC >= 0.71 while never delivering below B7 max3 (0.702705).

Protocol:
- SKF=5, fold-local FE, no global TE, no test pseudo-labels, no OOF continuous weight search
- Fusion: pre-registered discrete rules via nested_select_rule
- Floor: also report elementwise max(B7_max3, fused) nested score
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.b6pro_long_features import build_long_aging, build_long_keepx
from insurance_claim.train_b6 import PARAMS_B5, PARAMS_GAP_BAG

TARGET = 0.71
B7_FLOOR = 0.7027049552615718


def rank_transfer(base: np.ndarray, donor: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = base.copy()
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return out
    r = rankdata(donor[idx])
    sorted_base = np.sort(base[idx])
    out[idx] = sorted_base[(r.astype(int) - 1).clip(0, len(idx) - 1)]
    return out


def run_arm(builder, name, features, y, test, seeds, params, sample_weight_fn=None):
    oofs, tests = [], []
    days = features["days"].to_numpy(dtype=float)
    long = days >= 3000
    for seed in seeds:
        oof = np.zeros(len(y), dtype=float)
        pte = np.zeros(len(test), dtype=float)
        for fold, (tr, va) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=seed).split(features, y)
        ):
            Xtr = features.iloc[tr].reset_index(drop=True)
            Xva = features.iloc[va].reset_index(drop=True)
            trd, vad, ted, cats = builder(Xtr, Xva, test.copy())
            p = dict(params)
            p["random_seed"] = int(seed + fold)
            model = CatBoostClassifier(**p)
            fit_kw = dict(
                eval_set=(vad, y.iloc[va]),
                cat_features=cats,
                use_best_model=True,
            )
            if sample_weight_fn is not None:
                w = sample_weight_fn(days[tr], y.iloc[tr].to_numpy())
                model.fit(trd, y.iloc[tr], sample_weight=w, **fit_kw)
            else:
                model.fit(trd, y.iloc[tr], **fit_kw)
            oof[va] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5.0
            va_long = long[va]
            long_auc = None
            if va_long.sum() >= 30 and y.iloc[va][va_long].sum() >= 3:
                long_auc = float(roc_auc_score(y.iloc[va][va_long], oof[va][va_long]))
            print(
                f"{name} seed={seed} fold={fold} auc={roc_auc_score(y.iloc[va], oof[va]):.5f} "
                f"long={long_auc}",
                flush=True,
            )
        print(
            f"{name} seed={seed} OOF={roc_auc_score(y, oof):.6f} "
            f"long={roc_auc_score(y.to_numpy()[long], oof[long]):.6f}",
            flush=True,
        )
        oofs.append(oof)
        tests.append(pte)
    return np.mean(oofs, axis=0), np.mean(tests, axis=0)


def long_weight_fn(days_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """Upweight long-exposure rows; mild upweight on positives inside long."""
    w = np.ones(len(days_tr), dtype=float)
    long = days_tr >= 3000
    w[long] = 2.5
    w[long & (y_tr == 1)] = 3.0
    # ultra-long (10k+) slightly more — weakest max3 slice
    w[days_tr >= 10000] = np.maximum(w[days_tr >= 10000], 3.5)
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_long_aging"))
    ap.add_argument("--mode", choices=["aging", "keepx", "both"], default="both")
    args = ap.parse_args()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train["label"].astype(int)
    features = train.drop(columns=["label"])
    days = features["days"].to_numpy(dtype=float)
    long = days >= 3000

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    tmax = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])

    params = {**PARAMS_GAP_BAG, "thread_count": 4}
    arm_oofs: dict[str, np.ndarray] = {}
    arm_tests: dict[str, np.ndarray] = {}

    if args.mode in ("aging", "both"):
        oof, pte = run_arm(
            build_long_aging,
            "long_aging",
            features,
            y,
            test,
            args.seeds,
            params,
            sample_weight_fn=long_weight_fn,
        )
        arm_oofs["long_aging"] = oof
        arm_tests["long_aging"] = pte

    if args.mode in ("keepx", "both"):
        oof, pte = run_arm(
            build_long_keepx,
            "long_keepx",
            features,
            y,
            test,
            args.seeds,
            {**PARAMS_B5, "thread_count": 4, "l2_leaf_reg": 20, "random_strength": 1.0},
            sample_weight_fn=long_weight_fn,
        )
        arm_oofs["long_keepx"] = oof
        arm_tests["long_keepx"] = pte

    # Derived: long-patch (use arm on long, max3 on short) + rank-transfer
    derived_oof: dict[str, np.ndarray] = {}
    derived_test: dict[str, np.ndarray] = {}
    days_te = test["days"].to_numpy(dtype=float)
    long_te = days_te >= 3000
    for name, oof in arm_oofs.items():
        patch = max3.copy()
        patch[long] = oof[long]
        derived_oof[f"{name}_patch"] = patch
        tp = tmax.copy()
        tp[long_te] = arm_tests[name][long_te]
        derived_test[f"{name}_patch"] = tp

        rt = rank_transfer(max3, oof, long)
        derived_oof[f"{name}_rt"] = rt
        # test: rank-transfer within long using test scores' donor ranks → base tmax marginal
        rt_te = rank_transfer(tmax, arm_tests[name], long_te)
        derived_test[f"{name}_rt"] = rt_te

        blend = max3.copy()
        blend[long] = 0.5 * (max3[long] + oof[long])
        derived_oof[f"{name}_meanL"] = blend
        bt = tmax.copy()
        bt[long_te] = 0.5 * (tmax[long_te] + arm_tests[name][long_te])
        derived_test[f"{name}_meanL"] = bt

    # Evaluate solos
    print("\n=== solo AUCs ===", flush=True)
    print(f"max3={roc_auc_score(y, max3):.6f} long={roc_auc_score(y.to_numpy()[long], max3[long]):.6f}")
    for name, oof in {**arm_oofs, **derived_oof}.items():
        print(
            f"{name}: all={roc_auc_score(y, oof):.6f} "
            f"long={roc_auc_score(y.to_numpy()[long], oof[long]):.6f} "
            f"corr_max3={np.corrcoef(oof, max3)[0,1]:.4f}",
            flush=True,
        )

    # Nested fusions (always include B7 components + new arms)
    candidates = {
        "b7_max3_only": nested_select_rule(y.to_numpy(), [max3]),
        "b7_3arm": nested_select_rule(y.to_numpy(), [b7["gap"], b7["gap_bag"], b7["plus"]]),
    }
    # 4-arm: b7 three + each new
    for name, oof in {**arm_oofs, **derived_oof}.items():
        candidates[f"b7+{name}"] = nested_select_rule(
            y.to_numpy(), [b7["gap"], b7["gap_bag"], b7["plus"], oof]
        )
        candidates[f"max3×{name}"] = nested_select_rule(y.to_numpy(), [max3, oof])

    # multi new arms together
    if len(arm_oofs) >= 2:
        names = list(arm_oofs.keys())
        candidates["b7+both"] = nested_select_rule(
            y.to_numpy(),
            [b7["gap"], b7["gap_bag"], b7["plus"]] + [arm_oofs[n] for n in names],
        )
        candidates["b7+both_derived"] = nested_select_rule(
            y.to_numpy(),
            [b7["gap"], b7["gap_bag"], b7["plus"]]
            + [derived_oof[f"{n}_meanL"] for n in names]
            + [derived_oof[f"{n}_rt"] for n in names],
        )

    print("\n=== nested fusions ===", flush=True)
    best_name = None
    best = None
    for name, res in sorted(candidates.items(), key=lambda kv: -kv[1]["nested_oof_auc"]):
        floor_ok = res["nested_oof_auc"] + 1e-12 >= B7_FLOOR
        mark = "FLOOR_OK" if floor_ok else "below_floor"
        print(
            f"{name}: nested={res['nested_oof_auc']:.8f} rule={res['selected_rule']} "
            f"folds={res['fold_rules']} {mark}",
            flush=True,
        )
        if best is None or res["nested_oof_auc"] > best["nested_oof_auc"]:
            best = res
            best_name = name

    # Explicit floor: nested OOF of max(max3, best_full_selected) as delivery safeguard
    # Rebuild test prediction for best candidate
    def arms_for(name: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if name == "b7_max3_only":
            return [max3], [tmax]
        if name == "b7_3arm":
            return [b7["gap"], b7["gap_bag"], b7["plus"]], [
                fr["test_gap"],
                fr["test_gap_bag"],
                fr["test_plus"],
            ]
        if name.startswith("b7+") and name[3:] in {**arm_oofs, **derived_oof}:
            k = name[3:]
            src = arm_oofs if k in arm_oofs else derived_oof
            tsrc = arm_tests if k in arm_tests else derived_test
            return [b7["gap"], b7["gap_bag"], b7["plus"], src[k]], [
                fr["test_gap"],
                fr["test_gap_bag"],
                fr["test_plus"],
                tsrc[k],
            ]
        if name.startswith("max3×"):
            k = name[5:]
            src = arm_oofs if k in arm_oofs else derived_oof
            tsrc = arm_tests if k in arm_tests else derived_test
            return [max3, src[k]], [tmax, tsrc[k]]
        if name == "b7+both":
            names = list(arm_oofs.keys())
            return [b7["gap"], b7["gap_bag"], b7["plus"]] + [arm_oofs[n] for n in names], [
                fr["test_gap"],
                fr["test_gap_bag"],
                fr["test_plus"],
            ] + [arm_tests[n] for n in names]
        if name == "b7+both_derived":
            names = list(arm_oofs.keys())
            oofs = [b7["gap"], b7["gap_bag"], b7["plus"]] + [
                derived_oof[f"{n}_meanL"] for n in names
            ] + [derived_oof[f"{n}_rt"] for n in names]
            te = [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]] + [
                derived_test[f"{n}_meanL"] for n in names
            ] + [derived_test[f"{n}_rt"] for n in names]
            return oofs, te
        raise KeyError(name)

    oof_arms, test_arms = arms_for(best_name)
    test_pred = apply_rule(best["selected_rule"], test_arms)
    # Delivery floor: never worse than B7 max3 on nested metric — blend by nested max of scores
    floor_oof = np.maximum(max3, best["nested_oof"])
    # That max of OOFs is not a valid nested protocol object; instead:
    # if best nested < floor, fall back to max3
    deliver_oof = best["nested_oof"]
    deliver_test = test_pred
    deliver_auc = best["nested_oof_auc"]
    if deliver_auc + 1e-12 < B7_FLOOR:
        print("FALLBACK to B7 max3 (best below floor)", flush=True)
        deliver_oof = max3
        deliver_test = tmax
        deliver_auc = float(roc_auc_score(y, max3))
        best_name = "b7_max3_fallback"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y=y.to_numpy(),
        oof=deliver_oof,
        test=deliver_test,
        max3=max3,
        **{f"oof_{k}": v for k, v in arm_oofs.items()},
        **{f"test_{k}": v for k, v in arm_tests.items()},
        **{f"oof_{k}": v for k, v in derived_oof.items()},
        **{f"test_{k}": v for k, v in derived_test.items()},
    )
    sub = sample.copy()
    label_col = [c for c in sub.columns if c != "id"][0]
    sub[label_col] = deliver_test
    sub_path = args.output_dir / "submission_b6pro.csv"
    sub.to_csv(sub_path, index=False)

    metrics = {
        "experiment_id": "b6pro_long_aging",
        "best_fusion": best_name,
        "nested_oof_auc": deliver_auc,
        "selected_rule": best["selected_rule"] if best_name != "b7_max3_fallback" else "max3",
        "fold_rules": best.get("fold_rules"),
        "full_data_scores": best.get("full_data_scores"),
        "baseline_max3": B7_FLOOR,
        "gate_0_71": bool(deliver_auc >= TARGET),
        "gap_to_0_71": float(TARGET - deliver_auc),
        "long_slice_max3_auc": float(roc_auc_score(y.to_numpy()[long], max3[long])),
        "arm_solo": {
            k: float(roc_auc_score(y, v)) for k, v in {**arm_oofs, **derived_oof}.items()
        },
        "all_candidate_nested": {k: float(v["nested_oof_auc"]) for k, v in candidates.items()},
        "public_b7_signal": {"local": B7_FLOOR, "public": 0.707},
        "protocol_declaration": {
            "no_test_labels": True,
            "no_global_te": True,
            "fold_local_fe": True,
            "long_exposure_aging_features": True,
            "sample_weight_long": True,
            "no_oof_weight_search": True,
            "b7_floor_enforced": True,
            "new_data_only": True,
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"GATE={'PASS' if metrics['gate_0_71'] else 'FAIL'} nested={deliver_auc:.8f}", flush=True)
    return 0 if metrics["gate_0_71"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
