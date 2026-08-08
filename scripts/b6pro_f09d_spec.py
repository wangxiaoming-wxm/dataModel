#!/usr/bin/env python3
"""f09d (and other mega-weak) long-region specialists — highest-leverage lift path.

Sensitivity: raising f09d-long AUC 0.605→0.65 alone yields overall ≈0.711.
f09d long n≈2689 (~14% claim) — enough mass for a dedicated CatBoost.
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
CLOSEST = 0.7054764270400189
# leverage-ranked weak regions
FOCUS = ("f09d", "9685", "908d", "fafc", "f167", "ab86")
WEAK = frozenset(FOCUS)

PARAMS = {
    **PARAMS_GAP_BAG,
    "depth": 8,
    "l2_leaf_reg": 10,
    "learning_rate": 0.025,
    "iterations": 4000,
    "od_wait": 200,
    "random_strength": 1.5,
    "bagging_temperature": 0.8,
    "thread_count": 4,
}


def write_submission(sample, test, pred, path: Path):
    out = sample.copy()
    label_col = [c for c in out.columns if c != IDENTIFIER][0]
    out[label_col] = np.asarray(pred, float)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def train_focus(features, y, test, regions, builder, seeds, min_days=3000.0, expand_pool=None):
    """Train on (region in pool) ∩ long; evaluate/patch on focus regions."""
    region = features["region"].astype(str).to_numpy()
    days = features["days"].to_numpy(float)
    pool = tuple(expand_pool) if expand_pool else tuple(regions)
    mask_pool = np.isin(region, list(pool)) & (days >= min_days)
    idx = np.where(mask_pool)[0]
    if len(idx) < 300 or y.iloc[idx].sum() < 30:
        return None, None, mask_pool

    oof_acc = np.zeros(len(y))
    te_acc = np.zeros(len(test))
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        Xl = features.iloc[idx].reset_index(drop=True)
        yl = y.iloc[idx].reset_index(drop=True)
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xl, yl)):
            gtr, gva = idx[tr], idx[va]
            trd, vad, ted, cats = builder(
                features.iloc[gtr].reset_index(drop=True),
                features.iloc[gva].reset_index(drop=True),
                test.copy(),
            )
            model = CatBoostClassifier(**{**PARAMS, "random_seed": int(seed + fold)})
            model.fit(trd, y.iloc[gtr], eval_set=(vad, y.iloc[gva]), cat_features=cats, use_best_model=True)
            oof[gva] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5.0
        print(
            f"pool={pool} seed={seed} pool_auc={roc_auc_score(y.to_numpy()[mask_pool], oof[mask_pool]):.5f}",
            flush=True,
        )
        for r in regions:
            m = (region == r) & (days >= min_days)
            if m.sum() and y.to_numpy()[m].sum() and y.to_numpy()[m].sum() < m.sum():
                print(f"  {r}: {roc_auc_score(y.to_numpy()[m], oof[m]):.5f}", flush=True)
        oof_acc += oof
        te_acc += pte
    return oof_acc / len(seeds), te_acc / len(seeds), mask_pool


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
    long = days >= 3000
    long_te = days_te >= 3000

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    tmax = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])
    cur = np.load("artifacts/b6pro_long_best/predictions.npz")
    base = cur["oof"].copy()
    tbase = cur["test"].copy()
    aging = np.load("artifacts/b6pro_long_only_aging/predictions.npz")
    gap = np.load("artifacts/b6pro_long_only_gap/predictions.npz")
    keepx = np.load("artifacts/b6pro_long_only_keepx/predictions.npz")
    meanL3 = (aging["oof_long_only"] + gap["oof_long_only"] + keepx["oof_long_only"]) / 3.0
    tmeanL3 = (aging["test_long_only"] + gap["test_long_only"] + keepx["test_long_only"]) / 3.0

    seeds = [2026, 2027, 2028, 2029, 2030, 2031]
    builders = {"keepx": build_long_keepx, "aging": build_long_aging, "gap": build_gap}

    # Experiments: single-region and pooled-weak specialists
    runs = []
    # 1) f09d alone with each builder
    for bname, builder in builders.items():
        runs.append((f"f09d_{bname}", ("f09d",), builder, None))
    # 2) f09d+9685 pool (two largest weak)
    runs.append(("f09d9685_keepx", ("f09d", "9685"), build_long_keepx, ("f09d", "9685")))
    # 3) all weak pool
    runs.append(("allweak_keepx", FOCUS, build_long_keepx, FOCUS))
    # 4) f09d trained with transfer from all weak pool but only patch f09d
    runs.append(("f09d_xfer_keepx", ("f09d",), build_long_keepx, FOCUS))

    local = {}
    for name, focus, builder, pool in runs:
        print(f"\n=== RUN {name} ===", flush=True)
        oof_r, te_r, _ = train_focus(features, y, test, focus, builder, seeds, expand_pool=pool)
        if oof_r is None:
            print("skip", name, flush=True)
            continue
        local[name] = (oof_r, te_r, focus)
        for r in focus:
            m = (region == r) & long
            print(
                f"SUMMARY {name} {r}: local={roc_auc_score(y.to_numpy()[m], oof_r[m]):.5f} "
                f"base={roc_auc_score(y.to_numpy()[m], base[m]):.5f} "
                f"max3={roc_auc_score(y.to_numpy()[m], max3[m]):.5f}",
                flush=True,
            )

    # Build patched arms: for each region, pick best local if better than base
    def patch_best(mode="replace"):
        arm = base.copy()
        tarm = tbase.copy()
        for r in FOCUS:
            m = (region == r) & long
            m_te = (region_te == r) & long_te
            cands = []
            for name, (oof_r, te_r, focus) in local.items():
                if r not in focus:
                    continue
                cands.append((roc_auc_score(y.to_numpy()[m], oof_r[m]), name, oof_r, te_r))
            if not cands:
                continue
            cands.sort(reverse=True)
            best_auc, best_name, oof_r, te_r = cands[0]
            base_auc = roc_auc_score(y.to_numpy()[m], base[m])
            print(f"pick {r}: {best_name} {best_auc:.5f} vs base {base_auc:.5f}", flush=True)
            if best_auc > base_auc:
                if mode == "replace":
                    arm[m] = oof_r[m]
                    tarm[m_te] = te_r[m_te]
                elif mode == "mean":
                    arm[m] = 0.5 * (base[m] + oof_r[m])
                    tarm[m_te] = 0.5 * (tbase[m_te] + te_r[m_te])
                elif mode == "meanL":
                    arm[m] = 0.5 * (meanL3[m] + oof_r[m])
                    tarm[m_te] = 0.5 * (tmeanL3[m_te] + te_r[m_te])
                elif mode == "w07":
                    arm[m] = 0.7 * oof_r[m] + 0.3 * base[m]
                    tarm[m_te] = 0.7 * te_r[m_te] + 0.3 * tbase[m_te]
        return arm, tarm

    variants = {}
    for mode in ("replace", "mean", "meanL", "w07"):
        variants[mode] = patch_best(mode)

    # also: only patch f09d with its best
    for mode in ("replace", "mean", "w07"):
        arm = base.copy()
        tarm = tbase.copy()
        m = (region == "f09d") & long
        m_te = (region_te == "f09d") & long_te
        cands = [
            (roc_auc_score(y.to_numpy()[m], oof_r[m]), oof_r, te_r)
            for name, (oof_r, te_r, focus) in local.items()
            if "f09d" in focus
        ]
        if cands:
            cands.sort(reverse=True)
            _, oof_r, te_r = cands[0]
            if mode == "replace":
                arm[m] = oof_r[m]
                tarm[m_te] = te_r[m_te]
            elif mode == "mean":
                arm[m] = 0.5 * (base[m] + oof_r[m])
                tarm[m_te] = 0.5 * (tbase[m_te] + te_r[m_te])
            else:
                arm[m] = 0.7 * oof_r[m] + 0.3 * base[m]
                tarm[m_te] = 0.7 * te_r[m_te] + 0.3 * tbase[m_te]
            variants[f"f09d_only_{mode}"] = (arm, tarm)

    results = {}
    best_name, best_res, best_pair = None, None, None
    for name, (oof_arm, te_arm) in variants.items():
        direct = float(roc_auc_score(y, oof_arm))
        long_a = float(roc_auc_score(y.to_numpy()[long], oof_arm[long]))
        f09 = float(roc_auc_score(y.to_numpy()[(region == "f09d") & long], oof_arm[(region == "f09d") & long]))
        for tag, oof_arms, te_arms in [
            (f"direct_{name}", [oof_arm], [te_arm]),
            (f"b7+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], oof_arm], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], te_arm]),
            (f"max3×{name}", [max3, oof_arm], [tmax, te_arm]),
        ]:
            if len(oof_arms) == 1:
                res = {"nested_oof_auc": direct, "nested_oof": oof_arm, "selected_rule": "mean", "fold_rules": ["mean"]}
            else:
                res = nested_select_rule(y.to_numpy(), oof_arms)
            results[tag] = float(res["nested_oof_auc"])
            if best_res is None or res["nested_oof_auc"] > best_res["nested_oof_auc"]:
                best_name, best_res, best_pair = tag, res, (oof_arms, te_arms)
        print(f"{name}: direct={direct:.8f} long={long_a:.5f} f09d={f09:.5f}", flush=True)

    deliver_auc = best_res["nested_oof_auc"]
    deliver_oof = best_res["nested_oof"]
    deliver_test = apply_rule(best_res["selected_rule"], best_pair[1]) if len(best_pair[1]) > 1 else best_pair[1][0]
    if deliver_auc + 1e-12 < B7_FLOOR:
        best_name = "b7_fallback"
        deliver_auc = float(roc_auc_score(y, max3))
        deliver_oof, deliver_test = max3, tmax

    promoted = deliver_auc > CLOSEST + 1e-12
    out_dir = Path("artifacts/b6pro_f09d_spec")
    out_dir.mkdir(parents=True, exist_ok=True)
    save = {"y": y.to_numpy(), "oof": deliver_oof, "test": deliver_test}
    for name, (oof_r, te_r, focus) in local.items():
        save[f"oof_{name}"] = oof_r
        save[f"test_{name}"] = te_r
    np.savez_compressed(out_dir / "predictions.npz", **save)
    write_submission(sample, test, deliver_test, out_dir / "submission_b6pro.csv")
    if promoted:
        dest = Path("artifacts/b6pro_long_best")
        np.savez_compressed(dest / "predictions.npz", y=y.to_numpy(), oof=deliver_oof, test=deliver_test, arm=deliver_oof)
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
                    "source": "b6pro_f09d_spec",
                },
                indent=2,
            )
        )

    metrics = {
        "experiment_id": "b6pro_f09d_spec",
        "best_fusion": best_name,
        "nested_oof_auc": deliver_auc,
        "all_candidate_nested": results,
        "promoted_closest": promoted,
        "prev_closest": CLOSEST,
        "gate_0_71": bool(deliver_auc >= GATE),
        "gap_to_0_71": float(GATE - deliver_auc),
        "baseline_max3": B7_FLOOR,
        "f09d_base_auc": float(roc_auc_score(y.to_numpy()[(region == "f09d") & long], base[(region == "f09d") & long])),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "all_candidate_nested"}, indent=2), flush=True)
    print("TOP", sorted(results.items(), key=lambda kv: -kv[1])[:12], flush=True)
    print(f"GATE={'PASS' if metrics['gate_0_71'] else 'FAIL'} nested={deliver_auc:.8f} promoted={promoted}", flush=True)
    return 0 if metrics["gate_0_71"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
