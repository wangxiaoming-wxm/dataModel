#!/usr/bin/env python3
"""Long-exposure PairLogit / Lossguide specialists + region blend with B7 floor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.b6pro_long_features import build_long_aging
from insurance_claim.train_b6 import PARAMS_GAP_BAG, build_gap

B7_FLOOR = 0.7027049552615718
GATE = 0.71
WEAK = frozenset({"908d", "f09d", "9685", "fafc", "f167", "ab86"})


def blend_region(base, arm, mask, region, ww=1.0, wo=0.15):
    out = base.copy()
    weak = mask & np.isin(region, list(WEAK))
    other = mask & ~np.isin(region, list(WEAK))
    out[weak] = ww * arm[weak] + (1 - ww) * base[weak]
    out[other] = wo * arm[other] + (1 - wo) * base[other]
    return out


def run_lossguide(features, y, test, days, seeds):
    mask = days >= 3000
    idx = np.where(mask)[0]
    params = {
        **PARAMS_GAP_BAG,
        "grow_policy": "Lossguide",
        "max_leaves": 64,
        "depth": None,
        "iterations": 2000,
        "learning_rate": 0.02,
        "l2_leaf_reg": 12,
        "thread_count": 4,
    }
    # CatBoost may not like depth=None with some versions - use depth=6 + lossguide
    params = {
        **PARAMS_GAP_BAG,
        "grow_policy": "Lossguide",
        "max_leaves": 48,
        "depth": 8,
        "iterations": 1800,
        "learning_rate": 0.025,
        "l2_leaf_reg": 8,
        "thread_count": 4,
    }
    oof_acc = np.zeros(len(y))
    te_acc = np.zeros(len(test))
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        Xl = features.iloc[idx]
        yl = y.iloc[idx]
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xl, yl)):
            gtr, gva = idx[tr], idx[va]
            trd, vad, ted, cats = build_long_aging(
                features.iloc[gtr].reset_index(drop=True),
                features.iloc[gva].reset_index(drop=True),
                test.copy(),
            )
            model = CatBoostClassifier(**{**params, "random_seed": int(seed + fold)})
            model.fit(
                trd,
                y.iloc[gtr],
                eval_set=(vad, y.iloc[gva]),
                cat_features=cats,
                use_best_model=True,
            )
            oof[gva] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5.0
            print(
                f"lossguide seed={seed} fold={fold} auc={roc_auc_score(y.iloc[gva], oof[gva]):.5f}",
                flush=True,
            )
        print(
            f"lossguide seed={seed} slice={roc_auc_score(y.to_numpy()[mask], oof[mask]):.6f}",
            flush=True,
        )
        oof_acc += oof
        te_acc += pte
    return oof_acc / len(seeds), te_acc / len(seeds)


def run_pairlogit(features, y, test, days, seeds):
    """CatBoost PairLogit with group_id=region on long rows."""
    mask = days >= 3000
    idx = np.where(mask)[0]
    oof_acc = np.zeros(len(y))
    te_acc = np.zeros(len(test))
    region_all = features["region"].astype(str).to_numpy()
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        Xl = features.iloc[idx].reset_index(drop=True)
        yl = y.iloc[idx].reset_index(drop=True)
        # map local->global
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xl, yl)):
            gtr, gva = idx[tr], idx[va]
            trd, vad, ted, cats = build_gap(
                features.iloc[gtr].reset_index(drop=True),
                features.iloc[gva].reset_index(drop=True),
                test.copy(),
            )
            # PairLogit requires queryIds grouped (contiguous). Sort by region.
            group_tr = region_all[gtr]
            group_va = region_all[gva]
            order_tr = np.argsort(group_tr, kind="mergesort")
            order_va = np.argsort(group_va, kind="mergesort")
            inv_va = np.empty_like(order_va)
            inv_va[order_va] = np.arange(len(order_va))
            trd_s = trd.iloc[order_tr].reset_index(drop=True)
            vad_s = vad.iloc[order_va].reset_index(drop=True)
            ytr_s = y.iloc[gtr].to_numpy()[order_tr]
            yva_s = y.iloc[gva].to_numpy()[order_va]
            gtr_s = group_tr[order_tr]
            gva_s = group_va[order_va]
            # test: sort by region then unsort predictions
            group_te = test["region"].astype(str).to_numpy()
            order_te = np.argsort(group_te, kind="mergesort")
            inv_te = np.empty_like(order_te)
            inv_te[order_te] = np.arange(len(order_te))
            ted_s = ted.iloc[order_te].reset_index(drop=True)
            gte_s = group_te[order_te]
            train_pool = Pool(trd_s, ytr_s, cat_features=cats, group_id=gtr_s)
            va_pool = Pool(vad_s, yva_s, cat_features=cats, group_id=gva_s)
            te_pool = Pool(ted_s, cat_features=cats, group_id=gte_s)
            model = CatBoostRanker(
                loss_function="PairLogit",
                eval_metric="AUC",
                iterations=1200,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=10,
                random_seed=int(seed + fold),
                od_type="Iter",
                od_wait=100,
                verbose=False,
                thread_count=4,
                allow_writing_files=False,
            )
            try:
                model.fit(train_pool, eval_set=va_pool, use_best_model=True)
                s_va = model.predict(va_pool)[inv_va]
                s_te = model.predict(te_pool)[inv_te]
                s_va = 1 / (1 + np.exp(-(s_va - np.mean(s_va)) / (np.std(s_va) + 1e-6)))
                s_te = 1 / (1 + np.exp(-(s_te - np.mean(s_te)) / (np.std(s_te) + 1e-6)))
                oof[gva] = s_va
                pte += s_te / 5.0
                print(
                    f"pairlogit seed={seed} fold={fold} auc={roc_auc_score(y.iloc[gva], oof[gva]):.5f}",
                    flush=True,
                )
            except Exception as e:
                print(f"pairlogit failed seed={seed} fold={fold}: {e}", flush=True)
                clf = CatBoostClassifier(
                    **{**PARAMS_GAP_BAG, "random_seed": seed + fold, "thread_count": 4}
                )
                clf.fit(
                    trd,
                    y.iloc[gtr],
                    eval_set=(vad, y.iloc[gva]),
                    cat_features=cats,
                    use_best_model=True,
                )
                oof[gva] = clf.predict_proba(vad)[:, 1]
                pte += clf.predict_proba(ted)[:, 1] / 5.0
        print(
            f"pairlogit seed={seed} slice={roc_auc_score(y.to_numpy()[mask], oof[mask]):.6f}",
            flush=True,
        )
        oof_acc += oof
        te_acc += pte
    return oof_acc / len(seeds), te_acc / len(seeds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lossguide", "pairlogit", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027])
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_long_rank"))
    args = ap.parse_args()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train["label"].astype(int)
    features = train.drop(columns=["label"])
    days = features["days"].to_numpy(dtype=float)
    days_te = test["days"].to_numpy(dtype=float)
    region = features["region"].astype(str).to_numpy()
    region_te = test["region"].astype(str).to_numpy()
    long = days >= 3000
    long_te = days_te >= 3000

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    tmax = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])
    prev = np.load("artifacts/b6pro_long_best/predictions.npz")["arm"]

    arms = {}
    tests = {}
    if args.mode in ("lossguide", "both"):
        oof, pte = run_lossguide(features, y, test, days, args.seeds)
        arms["lossguide"] = oof
        tests["lossguide"] = pte
    if args.mode in ("pairlogit", "both"):
        oof, pte = run_pairlogit(features, y, test, days, args.seeds)
        arms["pairlogit"] = oof
        tests["pairlogit"] = pte

    # also mean with existing aging/gap/keepx
    lo_a = np.load("artifacts/b6pro_long_only_aging/predictions.npz")["oof_long_only"]
    lo_g = np.load("artifacts/b6pro_long_only_gap/predictions.npz")["oof_long_only"]
    lo_k = np.load("artifacts/b6pro_long_only_keepx/predictions.npz")["oof_long_only"]
    te_a = np.load("artifacts/b6pro_long_only_aging/predictions.npz")["test_long_only"]
    te_g = np.load("artifacts/b6pro_long_only_gap/predictions.npz")["test_long_only"]
    te_k = np.load("artifacts/b6pro_long_only_keepx/predictions.npz")["test_long_only"]

    best = None
    best_name = None
    best_te = None
    results = {}
    for name, oof in arms.items():
        print(name, "slice", roc_auc_score(y.to_numpy()[long], oof[long]), flush=True)
        for wo in (0.15, 0.2, 0.3):
            rb = blend_region(max3, oof, long, region, 1.0, wo)
            res = nested_select_rule(y.to_numpy(), [b7["gap"], b7["gap_bag"], b7["plus"], rb])
            key = f"{name}_wo{wo}"
            results[key] = res["nested_oof_auc"]
            print(key, res["nested_oof_auc"], flush=True)
            if best is None or res["nested_oof_auc"] > best["nested_oof_auc"]:
                best, best_name = res, key
                best_te = blend_region(tmax, tests[name], long_te, region_te, 1.0, wo)
        # ensemble with prev specialists
        ens = np.mean([oof, lo_a, lo_g, lo_k], axis=0)
        tens = np.mean([tests[name], te_a, te_g, te_k], axis=0)
        rb = blend_region(max3, ens, long, region, 1.0, 0.15)
        res = nested_select_rule(y.to_numpy(), [b7["gap"], b7["gap_bag"], b7["plus"], rb])
        key = f"ens4_{name}"
        results[key] = res["nested_oof_auc"]
        print(key, res["nested_oof_auc"], flush=True)
        if res["nested_oof_auc"] > best["nested_oof_auc"]:
            best, best_name = res, key
            best_te = blend_region(tmax, tens, long_te, region_te, 1.0, 0.15)
        # fuse with previous best arm
        res2 = nested_select_rule(y.to_numpy(), [b7["gap"], b7["gap_bag"], b7["plus"], prev, rb])
        key2 = f"prev+{key}"
        results[key2] = res2["nested_oof_auc"]
        print(key2, res2["nested_oof_auc"], flush=True)

    deliver_auc = best["nested_oof_auc"]
    deliver_oof = best["nested_oof"]
    deliver_test = apply_rule(best["selected_rule"], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], best_te])
    if deliver_auc + 1e-12 < B7_FLOOR:
        deliver_auc, deliver_oof, deliver_test = B7_FLOOR, max3, tmax
        best_name = "b7_fallback"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y=y.to_numpy(),
        oof=deliver_oof,
        test=deliver_test,
        **{f"oof_{k}": v for k, v in arms.items()},
        **{f"test_{k}": v for k, v in tests.items()},
    )
    sub = sample.copy()
    sub[sub.columns[1]] = deliver_test
    sub.to_csv(args.output_dir / "submission_b6pro.csv", index=False)
    metrics = {
        "experiment_id": "b6pro_long_rank",
        "best": best_name,
        "nested_oof_auc": float(deliver_auc),
        "baseline_max3": B7_FLOOR,
        "prev_closest": 0.7054481147284526,
        "gate_0_71": bool(deliver_auc >= GATE),
        "gap_to_0_71": float(GATE - deliver_auc),
        "all": results,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    return 0 if metrics["gate_0_71"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
