#!/usr/bin/env python3
"""Soft sample-weight keepx targeting ultra-long + mid-condition long.

Heavy weights (3–4×) previously *hurt* ultra AUC (0.616 < base 0.631).
This run uses mild 1.5–2.0× boosts plus optional loss focus on high-exposure mid-condition.
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
from insurance_claim.b6pro_long_features import build_long_keepx
from insurance_claim.train_b6 import PARAMS_GAP_BAG

B7_FLOOR = 0.7027049552615718
GATE = 0.71
CLOSEST = float(json.load(open("artifacts/b6pro_long_best/metrics.json"))["nested_oof_auc"])
PARAMS = {**PARAMS_GAP_BAG, "thread_count": 3, "iterations": 3000, "od_wait": 140, "depth": 8, "l2_leaf_reg": 8}

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("submit_sample.csv")
y = train["label"].astype(int)
features = train.drop(columns=["label"])
days = features["days"].to_numpy(float)
days_te = test["days"].to_numpy(float)
long = days >= 3000
ultra = days >= 10000
ultra_te = days_te >= 10000

_cur = np.load("artifacts/b6pro_honest_blend/predictions.npz")
base = _cur["oof"].copy()
tbase = _cur["test"].copy()
b7 = np.load("reference/b7_closest/predictions.npz")
fr = np.load("artifacts/b6pro_frozen/predictions.npz")


def weights(X: pd.DataFrame, mode: str) -> np.ndarray:
    d = X["days"].to_numpy(float)
    c = pd.to_numeric(X["condition"], errors="coerce")
    c = c.fillna(c.median()).to_numpy()
    w = np.ones(len(X))
    if mode == "soft_ultra":
        w[d >= 7000] *= 1.3
        w[d >= 10000] = 1.8
    elif mode == "soft_ultra2":
        w[d >= 10000] = 2.0
        w[(d >= 10000) & (y.loc[X.index].to_numpy() == 1)] *= 1.25  # mild claim emphasis — careful: uses labels
    elif mode == "midcond":
        qs = np.quantile(c[d >= 3000], [0.25, 0.75]) if (d >= 3000).any() else np.quantile(c, [0.25, 0.75])
        w[(d >= 3000) & (c >= qs[0]) & (c <= qs[1])] = 2.0
        w[d >= 10000] *= 1.4
    elif mode == "hi_exp_midcond":
        # business: high exposure × mid-low condition high claim quadrant
        qs = np.quantile(c[d >= 3000], [0.2, 0.6]) if (d >= 3000).any() else np.quantile(c, [0.2, 0.6])
        w[(d >= 7000) & (c >= qs[0]) & (c <= qs[1])] = 2.2
        w[d >= 10000] *= 1.3
    return w


def train_mode(mode: str, seeds: list[int]):
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
            # build weights on fold train rows without leaking via y-index tricks
            Xtr = features.iloc[tr].reset_index(drop=True)
            w = np.ones(len(Xtr))
            d = Xtr["days"].to_numpy(float)
            c = pd.to_numeric(Xtr["condition"], errors="coerce")
            c = c.fillna(c.median()).to_numpy()
            ytr = y.iloc[tr].to_numpy()
            if mode == "soft_ultra":
                w[d >= 7000] *= 1.3
                w[d >= 10000] = 1.8
            elif mode == "soft_ultra2":
                w[d >= 10000] = 2.0
                w[(d >= 10000) & (ytr == 1)] *= 1.25
            elif mode == "midcond":
                qs = np.quantile(c[d >= 3000], [0.25, 0.75]) if (d >= 3000).sum() > 50 else np.quantile(c, [0.25, 0.75])
                w[(d >= 3000) & (c >= qs[0]) & (c <= qs[1])] = 2.0
                w[d >= 10000] *= 1.4
            elif mode == "hi_exp_midcond":
                qs = np.quantile(c[d >= 3000], [0.2, 0.6]) if (d >= 3000).sum() > 50 else np.quantile(c, [0.2, 0.6])
                w[(d >= 7000) & (c >= qs[0]) & (c <= qs[1])] = 2.2
                w[d >= 10000] *= 1.3
            model = CatBoostClassifier(**{**PARAMS, "random_seed": seed + fold})
            model.fit(trd, y.iloc[tr], sample_weight=w, eval_set=(vad, y.iloc[va]), cat_features=cats, use_best_model=True)
            oof[va] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5
            print(f"{mode} s{seed} f{fold} {roc_auc_score(y.iloc[va], oof[va]):.5f}", flush=True)
        print(
            f"{mode} s{seed} OOF={roc_auc_score(y, oof):.6f} ultra={roc_auc_score(y.to_numpy()[ultra], oof[ultra]):.5f} "
            f"long={roc_auc_score(y.to_numpy()[long], oof[long]):.5f}",
            flush=True,
        )
        oof_acc += oof
        te_acc += pte
    return oof_acc / len(seeds), te_acc / len(seeds)


def main() -> int:
    seeds = [2026, 2027, 2028, 2029]
    locals_ = {}
    for mode in ["soft_ultra", "midcond", "hi_exp_midcond"]:
        print("===", mode, "===", flush=True)
        oof_r, te_r = train_mode(mode, seeds)
        locals_[mode] = (oof_r, te_r)
        print(
            "SUMMARY",
            mode,
            "oof",
            roc_auc_score(y, oof_r),
            "ultra",
            roc_auc_score(y.to_numpy()[ultra], oof_r[ultra]),
            "corr",
            np.corrcoef(oof_r, base)[0, 1],
            flush=True,
        )

    variants = {}
    for name, (oof_r, te_r) in locals_.items():
        variants[f"raw_{name}"] = (oof_r, te_r)
        variants[f"mean_base_{name}"] = (0.5 * (base + oof_r), 0.5 * (tbase + te_r))
        oof = np.zeros(len(y))
        fold_as = []
        for otr, ova in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y.to_numpy()):
            best_a, best_auc = 0.0, -1.0
            for a in np.linspace(0, 0.6, 13):
                auc = roc_auc_score(y.to_numpy()[otr], (1 - a) * base[otr] + a * oof_r[otr])
                if auc > best_auc:
                    best_auc, best_a = auc, a
            fold_as.append(best_a)
            oof[ova] = (1 - best_a) * base[ova] + best_a * oof_r[ova]
        a_star = float(np.median(fold_as))
        te = (1 - a_star) * tbase + a_star * te_r
        variants[f"honest_{name}_a{a_star}"] = (oof, te)
        print("honest", name, roc_auc_score(y, oof), a_star, fold_as, flush=True)

        # patch ultra
        oof2 = base.copy()
        te2 = tbase.copy()
        idx = np.where(ultra)[0]
        oof_m = np.zeros(len(idx))
        fold_as = []
        for otr, ova in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(idx)), y.to_numpy()[idx]):
            best_a, best_auc = 0.0, -1.0
            for a in np.linspace(0, 1, 21):
                auc = roc_auc_score(y.to_numpy()[idx[otr]], (1 - a) * base[idx[otr]] + a * oof_r[idx[otr]])
                if auc > best_auc:
                    best_auc, best_a = auc, a
            fold_as.append(best_a)
            oof_m[ova] = (1 - best_a) * base[idx[ova]] + best_a * oof_r[idx[ova]]
        a_star = float(np.median(fold_as))
        oof2[idx] = oof_m
        te2[ultra_te] = (1 - a_star) * tbase[ultra_te] + a_star * te_r[ultra_te]
        variants[f"patch_ultra_{name}_a{a_star}"] = (oof2, te2)
        print(
            "patch_ultra",
            name,
            roc_auc_score(y, oof2),
            "ultra",
            roc_auc_score(y.to_numpy()[ultra], oof2[ultra]),
            a_star,
            flush=True,
        )

        # patch long mid-condition
        cond = pd.to_numeric(features["condition"], errors="coerce").fillna(50).to_numpy()
        qs = np.quantile(cond[long], [0.25, 0.75])
        mask = long & (cond >= qs[0]) & (cond <= qs[1])
        mask_te = (days_te >= 3000) & (
            pd.to_numeric(test["condition"], errors="coerce").fillna(50).to_numpy() >= qs[0]
        ) & (pd.to_numeric(test["condition"], errors="coerce").fillna(50).to_numpy() <= qs[1])
        oof3 = base.copy()
        te3 = tbase.copy()
        idx = np.where(mask)[0]
        oof_m = np.zeros(len(idx))
        fold_as = []
        for otr, ova in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(idx)), y.to_numpy()[idx]):
            best_a, best_auc = 0.0, -1.0
            for a in np.linspace(0, 1, 21):
                auc = roc_auc_score(y.to_numpy()[idx[otr]], (1 - a) * base[idx[otr]] + a * oof_r[idx[otr]])
                if auc > best_auc:
                    best_auc, best_a = auc, a
            fold_as.append(best_a)
            oof_m[ova] = (1 - best_a) * base[idx[ova]] + best_a * oof_r[idx[ova]]
        a_star = float(np.median(fold_as))
        oof3[idx] = oof_m
        te3[mask_te] = (1 - a_star) * tbase[mask_te] + a_star * te_r[mask_te]
        variants[f"patch_midcond_{name}_a{a_star}"] = (oof3, te3)
        print("patch_midcond", name, roc_auc_score(y, oof3), a_star, flush=True)

    best = None
    results = {}
    for name, (oa, ta) in variants.items():
        direct = float(roc_auc_score(y, oa))
        for tag, arms, tlist in [
            (f"direct_{name}", [oa], [ta]),
            (
                f"b7+{name}",
                [b7["gap"], b7["gap_bag"], b7["plus"], oa],
                [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], ta],
            ),
            (
                f"cur+{name}",
                [b7["gap"], b7["gap_bag"], b7["plus"], base, oa],
                [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], tbase, ta],
            ),
        ]:
            if len(arms) == 1:
                res = {"nested_oof_auc": direct, "nested_oof": oa, "selected_rule": "mean"}
                deliver_te = ta
            else:
                res = nested_select_rule(y.to_numpy(), arms)
                deliver_te = apply_rule(res["selected_rule"], tlist)
            results[tag] = float(res["nested_oof_auc"])
            if best is None or res["nested_oof_auc"] > best[0]:
                best = (res["nested_oof_auc"], tag, res, deliver_te if len(arms) > 1 else ta)
        print(name, direct, flush=True)

    deliver, tag, res, deliver_te = best
    deliver_oof = res["nested_oof"]
    promoted = deliver > CLOSEST + 1e-12
    out = Path("artifacts/b6pro_soft_ultra")
    out.mkdir(parents=True, exist_ok=True)
    save = {"y": y.to_numpy(), "oof": deliver_oof, "test": deliver_te}
    for n, (o, t) in locals_.items():
        save[f"oof_{n}"] = o
        save[f"te_{n}"] = t
    np.savez_compressed(out / "predictions.npz", **save)
    lab = [c for c in sample.columns if c != "id"][0]
    sub = sample.copy()
    sub[lab] = deliver_te
    sub.to_csv(out / "submission_b6pro.csv", index=False)
    if promoted:
        dest = Path("artifacts/b6pro_long_best")
        tmp = dest / "predictions.npz.tmp"
        np.savez_compressed(tmp, y=y.to_numpy(), oof=deliver_oof, test=deliver_te)
        tmp.replace(dest / "predictions.npz")
        sub.to_csv(dest / "submission_b6pro.csv", index=False)
        sub.to_csv("submissions/b6pro_closest/submission_b6pro.csv", index=False)
        (dest / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment_id": "b6pro_long_best",
                    "spec": tag,
                    "nested_oof_auc": deliver,
                    "baseline_max3": B7_FLOOR,
                    "gate_0_71": deliver >= GATE,
                    "gap_to_0_71": GATE - deliver,
                    "source": "b6pro_soft_ultra",
                },
                indent=2,
            )
        )
        hb = Path("artifacts/b6pro_honest_blend")
        np.savez_compressed(hb / "predictions.npz", y=y.to_numpy(), oof=deliver_oof, test=deliver_te)
        (hb / "metrics.json").write_text(
            json.dumps({"best": tag, "nested": deliver, "gate": deliver >= GATE}, indent=2)
        )
    metrics = {
        "best": tag,
        "nested": deliver,
        "promoted": promoted,
        "gate": deliver >= GATE,
        "closest_prev": CLOSEST,
        "top": sorted(results.items(), key=lambda kv: -kv[1])[:12],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "top"}, indent=2), flush=True)
    print("TOP", metrics["top"][:8], flush=True)
    print(f"GATE={'PASS' if deliver >= GATE else 'FAIL'} nested={deliver:.8f} promoted={promoted}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
