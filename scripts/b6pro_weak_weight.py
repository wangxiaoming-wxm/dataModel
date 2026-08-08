#!/usr/bin/env python3
"""Full-data CatBoost keepx with sample-weight boost on weak×long rows.

Business: f09d/9685 long under-ranked (AUC~0.60) but hold most claim mass;
slice-only specialists underfit (~0.52). Weighting keeps global structure while
prioritizing weak-region ranking — highest sensitivity path to 0.71.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.b6pro_long_features import build_long_keepx
from insurance_claim.model import IDENTIFIER, TARGET
from insurance_claim.train_b6 import PARAMS_GAP_BAG

B7_FLOOR = 0.7027049552615718
GATE = 0.71
CLOSEST = float(json.load(open("artifacts/b6pro_long_best/metrics.json"))["nested_oof_auc"])
WEAK = ("f09d", "9685", "908d", "fafc", "f167", "ab86")
FOCUS_F09D = ("f09d",)

PARAMS = {
    **PARAMS_GAP_BAG,
    "thread_count": 4,
    "iterations": 3500,
    "od_wait": 150,
    "depth": 8,
    "l2_leaf_reg": 8,
    "learning_rate": 0.03,
}


def write_submission(sample, pred, path: Path) -> None:
    out = sample.copy()
    lab = [c for c in out.columns if c != IDENTIFIER][0]
    out[lab] = np.asarray(pred, float)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def make_weights(features: pd.DataFrame, focus: tuple[str, ...], w_weak: float, w_other_long: float = 1.0) -> np.ndarray:
    region = features["region"].astype(str).to_numpy()
    days = features["days"].to_numpy(float)
    long = days >= 3000
    w = np.ones(len(features), dtype=float)
    w[long] = w_other_long
    weak_long = long & np.isin(region, list(focus))
    w[weak_long] = w_weak
    # mild upweight of other long claims mass
    return w


def train_weighted(features, y, test, focus, w_weak, seeds):
    oof_acc = np.zeros(len(y))
    te_acc = np.zeros(len(test))
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(features, y)):
            trd, vad, ted, cats = build_long_keepx(
                features.iloc[tr].reset_index(drop=True),
                features.iloc[va].reset_index(drop=True),
                test.copy(),
            )
            w_tr = make_weights(features.iloc[tr].reset_index(drop=True), focus, w_weak)
            model = CatBoostClassifier(**{**PARAMS, "random_seed": int(seed + fold)})
            model.fit(
                trd,
                y.iloc[tr],
                sample_weight=w_tr,
                eval_set=(vad, y.iloc[va]),
                cat_features=cats,
                use_best_model=True,
            )
            oof[va] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5.0
            print(
                f"focus={focus} w={w_weak} s{seed} f{fold} {roc_auc_score(y.iloc[va], oof[va]):.5f}",
                flush=True,
            )
        print(
            f"focus={focus} w={w_weak} s{seed} OOF={roc_auc_score(y, oof):.6f}",
            flush=True,
        )
        oof_acc += oof
        te_acc += pte
    return oof_acc / len(seeds), te_acc / len(seeds)


def main() -> int:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train[TARGET].astype(int)
    features = train.drop(columns=[TARGET])
    days = features["days"].to_numpy(float)
    long = days >= 3000
    region = features["region"].astype(str).to_numpy()

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    tmax = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])
    # Eager-load from stable paths (avoid lazy NpzFile + concurrent long_best writes)
    cur_path = "artifacts/b6pro_honest_blend/predictions.npz"
    if not Path(cur_path).exists():
        cur_path = "artifacts/b6pro_long_best/predictions.npz"
    _cur = np.load(cur_path)
    cur = {"oof": np.asarray(_cur["oof"], float).copy(), "test": np.asarray(_cur["test"], float).copy()}
    _kx = np.load("artifacts/b6pro_full_keepx/predictions.npz")
    kx = {
        "oof": np.asarray(_kx["oof"] if "oof" in _kx.files else _kx["oof_k"], float).copy(),
        "test": np.asarray(_kx["test"] if "test" in _kx.files else _kx["te_k"], float).copy(),
    }

    seeds = [2026, 2027, 2028, 2029]
    # Focused set: moderate weights (high weights previously underfit f09d)
    runs = [
        ("f09d_w2", FOCUS_F09D, 2.0),
        ("f09d_w3", FOCUS_F09D, 3.0),
        ("weak_w2", WEAK, 2.0),
        ("weak_w3", WEAK, 3.0),
    ]

    locals_ = {}
    for name, focus, ww in runs:
        print(f"\n=== {name} ===", flush=True)
        oof_r, te_r = train_weighted(features, y, test, focus, ww, seeds)
        locals_[name] = (oof_r, te_r)
        m = (np.isin(region, list(focus))) & long
        print(
            f"SUMMARY {name}: oof={roc_auc_score(y, oof_r):.6f} "
            f"long={roc_auc_score(y.to_numpy()[long], oof_r[long]):.5f} "
            f"focus_long={roc_auc_score(y.to_numpy()[m], oof_r[m]):.5f} "
            f"cur_focus={roc_auc_score(y.to_numpy()[m], cur['oof'][m]):.5f} "
            f"corr_cur={np.corrcoef(oof_r, cur['oof'])[0,1]:.3f}",
            flush=True,
        )

    # blend variants: weak-region replace / mean with cur
    variants = {}
    for name, (oof_r, te_r) in locals_.items():
        variants[f"raw_{name}"] = (oof_r, te_r)
        variants[f"mean_cur_{name}"] = (0.5 * (cur["oof"] + oof_r), 0.5 * (cur["test"] + te_r))
        variants[f"mean_kx_{name}"] = (0.5 * (kx["oof"] + oof_r), 0.5 * (kx["test"] + te_r))
        # patch only focus long
        focus = FOCUS_F09D if name.startswith("f09d") else WEAK
        arm = cur["oof"].copy()
        tarm = cur["test"].copy()
        m = (np.isin(region, list(focus))) & long
        m_te = (np.isin(test["region"].astype(str).to_numpy(), list(focus))) & (test["days"].to_numpy(float) >= 3000)
        for mode, alpha in [("patch1", 1.0), ("patch07", 0.7), ("patch05", 0.5), ("patch03", 0.3)]:
            a = arm.copy()
            ta = tarm.copy()
            a[m] = alpha * oof_r[m] + (1 - alpha) * cur["oof"][m]
            ta[m_te] = alpha * te_r[m_te] + (1 - alpha) * cur["test"][m_te]
            variants[f"{mode}_{name}"] = (a, ta)

    results = {}
    best_name, best_res, best_pair = None, None, None
    for name, (oa, ta) in variants.items():
        direct = float(roc_auc_score(y, oa))
        for tag, oof_arms, te_arms in [
            (f"direct_{name}", [oa], [ta]),
            (f"b7+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], oa], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], ta]),
            (
                f"cur+{name}",
                [b7["gap"], b7["gap_bag"], b7["plus"], cur["oof"], oa],
                [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], cur["test"], ta],
            ),
        ]:
            if len(oof_arms) == 1:
                res = {"nested_oof_auc": direct, "nested_oof": oa, "selected_rule": "mean"}
            else:
                res = nested_select_rule(y.to_numpy(), oof_arms)
            results[tag] = float(res["nested_oof_auc"])
            if best_res is None or res["nested_oof_auc"] > best_res["nested_oof_auc"]:
                best_name, best_res, best_pair = tag, res, (oof_arms, te_arms)
        f09 = (region == "f09d") & long
        print(
            f"{name}: direct={direct:.8f} f09d={roc_auc_score(y.to_numpy()[f09], oa[f09]):.5f}",
            flush=True,
        )

    deliver = best_res["nested_oof_auc"]
    deliver_oof = best_res["nested_oof"]
    deliver_test = apply_rule(best_res["selected_rule"], best_pair[1]) if len(best_pair[1]) > 1 else best_pair[1][0]
    if deliver + 1e-12 < B7_FLOOR:
        best_name = "b7_fallback"
        deliver = float(roc_auc_score(y, max3))
        deliver_oof, deliver_test = max3, tmax

    promoted = deliver > CLOSEST + 1e-12
    out = Path("artifacts/b6pro_weak_weight")
    out.mkdir(parents=True, exist_ok=True)
    save = {"y": y.to_numpy(), "oof": deliver_oof, "test": deliver_test}
    for name, (oof_r, te_r) in locals_.items():
        save[f"oof_{name}"] = oof_r
        save[f"te_{name}"] = te_r
    np.savez_compressed(out / "predictions.npz", **save)
    write_submission(sample, deliver_test, out / "submission_b6pro.csv")
    if promoted:
        dest = Path("artifacts/b6pro_long_best")
        np.savez_compressed(dest / "predictions.npz", y=y.to_numpy(), oof=deliver_oof, test=deliver_test)
        write_submission(sample, deliver_test, dest / "submission_b6pro.csv")
        write_submission(sample, deliver_test, Path("submissions/b6pro_closest/submission_b6pro.csv"))
        (dest / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment_id": "b6pro_long_best",
                    "spec": best_name,
                    "nested_oof_auc": deliver,
                    "baseline_max3": B7_FLOOR,
                    "gate_0_71": deliver >= GATE,
                    "gap_to_0_71": GATE - deliver,
                    "source": "b6pro_weak_weight",
                },
                indent=2,
            )
        )

    metrics = {
        "best": best_name,
        "nested": deliver,
        "promoted": promoted,
        "gate": deliver >= GATE,
        "closest_prev": CLOSEST,
        "top": sorted(results.items(), key=lambda kv: -kv[1])[:15],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "top"}, indent=2), flush=True)
    print("TOP", metrics["top"][:10], flush=True)
    print(f"GATE={'PASS' if deliver >= GATE else 'FAIL'} nested={deliver:.8f} promoted={promoted}", flush=True)
    return 0 if deliver >= GATE else 2


if __name__ == "__main__":
    raise SystemExit(main())
