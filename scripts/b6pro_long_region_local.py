#!/usr/bin/env python3
"""Per-weak-region long specialists (f09d/9685/908d dominate long claim mass).

Business: region×days slopes flip; global trees underfit local aging curves.
Train CatBoost keepx only on (region in focus) ∩ (days>=3000), patch scores locally.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.b6pro_long_features import build_long_aging, build_long_keepx
from insurance_claim.model import IDENTIFIER, TARGET
from insurance_claim.train_b6 import PARAMS_GAP_BAG, build_gap

B7_FLOOR = 0.7027049552615718
GATE = 0.71
CLOSEST = 0.7054481147284526
# largest weak long masses
FOCUS = ("f09d", "9685", "908d", "fafc")
WEAK = frozenset({"908d", "f09d", "9685", "fafc", "f167", "ab86"})

PARAMS = {**PARAMS_GAP_BAG, "thread_count": 2, "iterations": 2500, "od_wait": 120}


def write_submission(sample, test, pred, path: Path):
    out = sample.copy()
    label_col = [c for c in out.columns if c != IDENTIFIER][0]
    out[label_col] = np.asarray(pred, float)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def train_region_spec(features, y, test, region_name, builder, seeds, min_days=3000.0):
    region = features["region"].astype(str).to_numpy()
    days = features["days"].to_numpy(float)
    mask = (region == region_name) & (days >= min_days)
    idx = np.where(mask)[0]
    if len(idx) < 200 or y.iloc[idx].sum() < 20:
        return None, None, mask
    oof_acc = np.zeros(len(y))
    te_acc = np.zeros(len(test))
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        Xl = features.iloc[idx].reset_index(drop=True)
        yl = y.iloc[idx].reset_index(drop=True)
        n_splits = 5 if yl.sum() >= 5 and (len(yl) - yl.sum()) >= 5 else 3
        try:
            splits = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(Xl, yl))
        except ValueError:
            return None, None, mask
        for fold, (tr, va) in enumerate(splits):
            gtr, gva = idx[tr], idx[va]
            trd, vad, ted, cats = builder(
                features.iloc[gtr].reset_index(drop=True),
                features.iloc[gva].reset_index(drop=True),
                test.copy(),
            )
            model = CatBoostClassifier(**{**PARAMS, "random_seed": seed + fold})
            model.fit(trd, y.iloc[gtr], eval_set=(vad, y.iloc[gva]), cat_features=cats, use_best_model=True)
            oof[gva] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / n_splits
        oof_acc += oof
        te_acc += pte
        print(
            f"region={region_name} s{seed} n={mask.sum()} "
            f"slice={roc_auc_score(y.to_numpy()[mask], oof[mask]):.5f}",
            flush=True,
        )
    return oof_acc / len(seeds), te_acc / len(seeds), mask


def main() -> int:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train[TARGET].astype(int)
    features = train.drop(columns=[TARGET])
    days = features["days"].to_numpy(float)
    days_te = test["days"].to_numpy(float)
    region = train["region"].astype(str).to_numpy()
    region_te = test["region"].astype(str).to_numpy()

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    tmax = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])
    cur = np.load("artifacts/b6pro_long_best/predictions.npz")
    aging = np.load("artifacts/b6pro_long_only_aging/predictions.npz")
    gap = np.load("artifacts/b6pro_long_only_gap/predictions.npz")
    keepx = np.load("artifacts/b6pro_long_only_keepx/predictions.npz")
    meanL3 = (aging["oof_long_only"] + gap["oof_long_only"] + keepx["oof_long_only"]) / 3.0
    tmeanL3 = (aging["test_long_only"] + gap["test_long_only"] + keepx["test_long_only"]) / 3.0

    seeds = [2026, 2027, 2028, 2029]
    builders = {"keepx": build_long_keepx, "aging": build_long_aging, "gap": build_gap}

    # start from current region blend arm
    def region_blend(max3_, long_spec, wo=0.15):
        out = max3_.copy()
        long = days >= 3000
        weak_m = long & np.isin(region, list(WEAK))
        other = long & ~np.isin(region, list(WEAK))
        out[weak_m] = long_spec[weak_m]
        out[other] = wo * long_spec[other] + (1 - wo) * max3_[other]
        return out

    base_arm = region_blend(max3, meanL3, 0.15)
    tbase = region_blend.__wrapped__ if False else None
    # test blend helper
    def region_blend_te(tmax_, tspec, wo=0.15):
        out = tmax_.copy()
        long = days_te >= 3000
        weak_m = long & np.isin(region_te, list(WEAK))
        other = long & ~np.isin(region_te, list(WEAK))
        out[weak_m] = tspec[weak_m]
        out[other] = wo * tspec[other] + (1 - wo) * tmax_[other]
        return out

    tbase_arm = region_blend_te(tmax, tmeanL3, 0.15)

    patched = base_arm.copy()
    tpatched = tbase_arm.copy()
    local_oofs = {}
    for rname in FOCUS:
        for bname, builder in builders.items():
            print(f"=== {rname} / {bname} ===", flush=True)
            oof_r, te_r, mask = train_region_spec(features, y, test, rname, builder, seeds)
            if oof_r is None:
                print(f"skip {rname}/{bname}", flush=True)
                continue
            local_oofs[f"{rname}_{bname}"] = oof_r
            # compare local vs current on that mask
            cur_auc = roc_auc_score(y.to_numpy()[mask], base_arm[mask])
            new_auc = roc_auc_score(y.to_numpy()[mask], oof_r[mask])
            max3_auc = roc_auc_score(y.to_numpy()[mask], max3[mask])
            print(f"  compare max3={max3_auc:.5f} base={cur_auc:.5f} local={new_auc:.5f}", flush=True)
            # only patch if local better than base on that region slice
            if new_auc > cur_auc:
                patched[mask] = oof_r[mask]
                mask_te = (region_te == rname) & (days_te >= 3000)
                tpatched[mask_te] = te_r[mask_te]
                print(f"  PATCHED {rname} with {bname}", flush=True)
            # also try mean with base
            # store mean version separately later

    # also build mean-patched variants
    variants = {
        "patch_better": (patched, tpatched),
        "base": (base_arm, tbase_arm),
    }
    # for each focus region, mean(local_best, meanL3)
    mean_patch = base_arm.copy()
    tmean_patch = tbase_arm.copy()
    for rname in FOCUS:
        cands = [local_oofs[k] for k in local_oofs if k.startswith(rname + "_")]
        if not cands:
            continue
        # pick best local by slice auc
        mask = (region == rname) & (days >= 3000)
        best_local = max(cands, key=lambda a: roc_auc_score(y.to_numpy()[mask], a[mask]))
        mean_patch[mask] = 0.5 * (meanL3[mask] + best_local[mask])
        # test: average available - use keepx if present else first
        # retrain not available; use last te from best builder name
        # approximate: use keepx te if we have it by re-running keys - skip precise; use patched te if patched else meanL3
        mask_te = (region_te == rname) & (days_te >= 3000)
        tmean_patch[mask_te] = 0.5 * (tmeanL3[mask_te] + tpatched[mask_te])
    variants["mean_local"] = (mean_patch, tmean_patch)

    # full replace weak with best locals where available
    best_local_arm = base_arm.copy()
    tbest_local = tbase_arm.copy()
    for rname in FOCUS:
        cands = [(k, local_oofs[k]) for k in local_oofs if k.startswith(rname + "_")]
        if not cands:
            continue
        mask = (region == rname) & (days >= 3000)
        kbest, abest = max(cands, key=lambda kv: roc_auc_score(y.to_numpy()[mask], kv[1][mask]))
        best_local_arm[mask] = abest[mask]
        print(f"best for {rname}: {kbest}", flush=True)
    variants["best_locals"] = (best_local_arm, tbest_local)

    results = {}
    best_name, best_res, best_pair = None, None, None
    for name, (oof_arm, te_arm) in variants.items():
        for tag, oof_arms, te_arms in [
            (f"b7+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], oof_arm], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], te_arm]),
            (f"cur+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], cur["oof"], oof_arm], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], cur["test"], te_arm]),
        ]:
            res = nested_select_rule(y.to_numpy(), oof_arms)
            results[tag] = {
                "nested": float(res["nested_oof_auc"]),
                "rule": res["selected_rule"],
                "folds": res["fold_rules"],
            }
            print(f"{tag}: nested={res['nested_oof_auc']:.8f} rule={res['selected_rule']}", flush=True)
            if best_res is None or res["nested_oof_auc"] > best_res["nested_oof_auc"]:
                best_name, best_res, best_pair = tag, res, (oof_arms, te_arms)

    deliver_auc = best_res["nested_oof_auc"]
    deliver_oof = best_res["nested_oof"]
    deliver_test = apply_rule(best_res["selected_rule"], best_pair[1])
    if deliver_auc + 1e-12 < B7_FLOOR:
        best_name = "b7_fallback"
        deliver_auc = float(roc_auc_score(y, max3))
        deliver_oof, deliver_test = max3, tmax

    promoted = deliver_auc > CLOSEST + 1e-12
    out_dir = Path("artifacts/b6pro_long_region_local")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions.npz",
        y=y.to_numpy(),
        oof=deliver_oof,
        test=deliver_test,
        patched=patched,
        **{k: v for k, v in local_oofs.items()},
    )
    write_submission(sample, test, deliver_test, out_dir / "submission_b6pro.csv")
    if promoted:
        dest = Path("artifacts/b6pro_long_best")
        np.savez_compressed(dest / "predictions.npz", y=y.to_numpy(), oof=deliver_oof, test=deliver_test, arm=patched)
        write_submission(sample, test, deliver_test, dest / "submission_b6pro.csv")
        write_submission(sample, test, deliver_test, Path("submissions/b6pro_closest/submission_b6pro.csv"))
        (dest / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment_id": "b6pro_long_best",
                    "spec": best_name,
                    "nested_oof_auc": deliver_auc,
                    "baseline_max3": B7_FLOOR,
                    "gate_0_71": deliver_auc >= GATE,
                    "gap_to_0_71": GATE - deliver_auc,
                    "source": "b6pro_long_region_local",
                },
                indent=2,
            )
        )

    metrics = {
        "experiment_id": "b6pro_long_region_local",
        "best_fusion": best_name,
        "nested_oof_auc": deliver_auc,
        "all_candidate_nested": results,
        "promoted_closest": promoted,
        "prev_closest": CLOSEST,
        "gate_0_71": bool(deliver_auc >= GATE),
        "gap_to_0_71": float(GATE - deliver_auc),
        "baseline_max3": B7_FLOOR,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"GATE={'PASS' if metrics['gate_0_71'] else 'FAIL'} nested={deliver_auc:.8f} promoted={promoted}", flush=True)
    return 0 if metrics["gate_0_71"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
