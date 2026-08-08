#!/usr/bin/env python3
"""Retrain plus_gap (2-4 seeds) and fuse with B7 + current long closest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import nested_select_rule
from insurance_claim.b6pro_plus import PARAMS_H2, build_plus_gap

WEAK = frozenset({"908d", "f09d", "9685", "fafc", "f167", "ab86"})


def main() -> int:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = train["label"].astype(int)
    features = train.drop(columns=["label"])
    days = features["days"].to_numpy(float)
    long = days >= 3000
    region = train["region"].astype(str).to_numpy()

    b7 = np.load("reference/b7_closest/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    cur = np.load("artifacts/b6pro_long_best/predictions.npz")["arm"]
    ref_plus = b7["plus"]

    seeds = [2026, 2027, 2028, 2029]
    oofs, tests = [], []
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        for fold, (tr, va) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=seed).split(features, y)
        ):
            trd, vad, ted, cats = build_plus_gap(
                features.iloc[tr].reset_index(drop=True),
                features.iloc[va].reset_index(drop=True),
                test.copy(),
            )
            model = CatBoostClassifier(**{**PARAMS_H2, "random_seed": seed + fold, "thread_count": 4})
            model.fit(
                trd,
                y.iloc[tr],
                eval_set=(vad, y.iloc[va]),
                cat_features=cats,
                use_best_model=True,
            )
            oof[va] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5.0
            print(
                f"plus_gap s{seed} f{fold} {roc_auc_score(y.iloc[va], oof[va]):.5f}",
                flush=True,
            )
        print(
            f"plus_gap s{seed} OOF={roc_auc_score(y, oof):.6f} "
            f"long={roc_auc_score(y.to_numpy()[long], oof[long]):.6f}",
            flush=True,
        )
        oofs.append(oof)
        tests.append(pte)
    oof_pg = np.mean(oofs, 0)
    te_pg = np.mean(tests, 0)
    print("ref plus", roc_auc_score(y, ref_plus), "new plus_gap", roc_auc_score(y, oof_pg), flush=True)

    results = {}
    for name, arms in [
        ("b7+pg", [b7["gap"], b7["gap_bag"], b7["plus"], oof_pg]),
        ("b7+cur+pg", [b7["gap"], b7["gap_bag"], b7["plus"], cur, oof_pg]),
        ("max3×pg", [max3, oof_pg]),
        ("replace_plus", [b7["gap"], b7["gap_bag"], oof_pg]),
        ("max_ref_new_plus", [b7["gap"], b7["gap_bag"], np.maximum(ref_plus, oof_pg)]),
        ("b7+cur+maxplus", [b7["gap"], b7["gap_bag"], np.maximum(ref_plus, oof_pg), cur]),
    ]:
        res = nested_select_rule(y.to_numpy(), arms)
        results[name] = res["nested_oof_auc"]
        print(name, res["nested_oof_auc"], res["selected_rule"], flush=True)

    def blend(base, arm, wo=0.15):
        out = base.copy()
        mask = long
        weak = mask & np.isin(region, list(WEAK))
        other = mask & ~np.isin(region, list(WEAK))
        out[weak] = arm[weak]
        out[other] = wo * arm[other] + (1 - wo) * base[other]
        return out

    rb = blend(max3, oof_pg, 0.15)
    results["pg_region"] = nested_select_rule(
        y.to_numpy(), [b7["gap"], b7["gap_bag"], b7["plus"], rb]
    )["nested_oof_auc"]
    print("pg region", results["pg_region"], flush=True)

    best = max(results.values())
    out = Path("artifacts/b6pro_plus_gap2")
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "predictions.npz", y=y.to_numpy(), oof=oof_pg, test=te_pg)
    metrics = {
        "plus_gap_auc": float(roc_auc_score(y, oof_pg)),
        "long": float(roc_auc_score(y.to_numpy()[long], oof_pg[long])),
        "best_nested": float(best),
        "all": results,
        "prev_closest": 0.7054481147284526,
        "gate_0_71": bool(best >= 0.71),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    return 0 if metrics["gate_0_71"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
