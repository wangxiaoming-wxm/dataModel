#!/usr/bin/env python3
"""Heterogeneous HGB with regime-slope features; nested ultra/midcond patch onto closest.

Probe (2-seed): patch_ultra nested ≈0.70929 (ultra 0.635) vs closest 0.70890.
Low corr (~0.60) to CatBoost closest is the diversity source.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule

B7_FLOOR = 0.7027049552615718
GATE = 0.71
CLOSEST = float(json.load(open("artifacts/b6pro_long_best/metrics.json"))["nested_oof_auc"])

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("submit_sample.csv")
y = train["label"].astype(int).to_numpy()
days = train["days"].to_numpy(float)
days_te = test["days"].to_numpy(float)
long = days >= 3000
ultra = days >= 10000
ultra_te = days_te >= 10000

_cur = np.load("artifacts/b6pro_honest_blend/predictions.npz")
base = _cur["oof"].copy()
tbase = _cur["test"].copy()
b7 = np.load("reference/b7_closest/predictions.npz")
fr = np.load("artifacts/b6pro_frozen/predictions.npz")

NUM = [
    "days",
    "condition",
    "cc",
    "V",
    "max_g",
    "age_range",
    "livability",
    "x18",
    "x19",
    "x20",
    "cap10",
    "ex10",
    "cap7",
    "ex7",
    "c_s",
    "c_m",
    "c_l",
    "c_u",
    "band",
]
CAT = ["region", "code", "version", "grades", "month"]


def fe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = pd.to_numeric(out["days"], errors="coerce")
    c = pd.to_numeric(out["condition"], errors="coerce")
    out["cap10"] = d.clip(upper=10000)
    out["ex10"] = (d - 10000).clip(lower=0)
    out["cap7"] = d.clip(upper=7000)
    out["ex7"] = (d - 7000).clip(lower=0)
    out["c_s"] = c * (d < 3000).astype(float)
    out["c_m"] = c * ((d >= 3000) & (d < 7000)).astype(float)
    out["c_l"] = c * ((d >= 7000) & (d < 10000)).astype(float)
    out["c_u"] = c * (d >= 10000).astype(float)
    out["band"] = pd.cut(d, [-0.1, 3000, 7000, 10000, 1e9], labels=[0, 1, 2, 3]).astype(float)
    return out


def train_hgb(seeds: list[int]):
    feats = fe(train.drop(columns=["label"]))
    te_feats = fe(test)
    oof_acc = np.zeros(len(y))
    te_acc = np.zeros(len(test))
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(feats, y)):
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            enc.fit(feats.iloc[tr][CAT].astype(str))
            med = feats.iloc[tr][NUM].apply(pd.to_numeric, errors="coerce").median()

            def pack(df):
                Xn = df[NUM].apply(pd.to_numeric, errors="coerce").fillna(med)
                Xc = enc.transform(df[CAT].astype(str))
                return np.hstack([Xn.to_numpy(dtype=float), Xc])

            model = HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.05,
                max_iter=400,
                l2_regularization=0.5,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=25,
                random_state=seed + fold,
            )
            model.fit(pack(feats.iloc[tr]), y[tr])
            oof[va] = model.predict_proba(pack(feats.iloc[va]))[:, 1]
            pte += model.predict_proba(pack(te_feats))[:, 1] / 5
            print(f"hgb_reg s{seed} f{fold} {roc_auc_score(y[va], oof[va]):.5f}", flush=True)
        print(
            f"hgb_reg s{seed} OOF={roc_auc_score(y, oof):.6f} ultra={roc_auc_score(y[ultra], oof[ultra]):.5f} "
            f"long={roc_auc_score(y[long], oof[long]):.5f} corr={np.corrcoef(oof, base)[0, 1]:.4f}",
            flush=True,
        )
        oof_acc += oof
        te_acc += pte
    return oof_acc / len(seeds), te_acc / len(seeds)


def nested_patch(base_oof, help_oof, base_te, help_te, mask, mask_te):
    oof = base_oof.copy()
    te = base_te.copy()
    idx = np.where(mask)[0]
    oof_m = np.zeros(len(idx))
    fold_as = []
    for otr, ova in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(idx)), y[idx]):
        best_a, best_auc = 0.0, -1.0
        for a in np.linspace(0, 1, 21):
            auc = roc_auc_score(y[idx[otr]], (1 - a) * base_oof[idx[otr]] + a * help_oof[idx[otr]])
            if auc > best_auc:
                best_auc, best_a = auc, a
        fold_as.append(best_a)
        oof_m[ova] = (1 - best_a) * base_oof[idx[ova]] + best_a * help_oof[idx[ova]]
    a_star = float(np.median(fold_as))
    oof[idx] = oof_m
    te[mask_te] = (1 - a_star) * base_te[mask_te] + a_star * help_te[mask_te]
    return oof, te, a_star, fold_as


def main() -> int:
    seeds = [2026, 2027, 2028, 2029]
    print("=== hgb regime ===", flush=True)
    oof_r, te_r = train_hgb(seeds)
    print(
        "SUMMARY",
        roc_auc_score(y, oof_r),
        "ultra",
        roc_auc_score(y[ultra], oof_r[ultra]),
        "corr",
        np.corrcoef(oof_r, base)[0, 1],
        flush=True,
    )

    cond = pd.to_numeric(train["condition"], errors="coerce").fillna(1).to_numpy()
    qs = np.quantile(cond[long], [0.25, 0.75])
    cte = pd.to_numeric(test["condition"], errors="coerce").fillna(1).to_numpy()
    mid = long & (cond >= qs[0]) & (cond <= qs[1])
    mid_te = (days_te >= 3000) & (cte >= qs[0]) & (cte <= qs[1])

    variants = {"raw": (oof_r, te_r)}
    # honest global
    oof = np.zeros(len(y))
    fold_as = []
    for otr, ova in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
        best_a, best_auc = 0.0, -1.0
        for a in np.linspace(0, 0.5, 11):
            auc = roc_auc_score(y[otr], (1 - a) * base[otr] + a * oof_r[otr])
            if auc > best_auc:
                best_auc, best_a = auc, a
        fold_as.append(best_a)
        oof[ova] = (1 - best_a) * base[ova] + best_a * oof_r[ova]
    a_star = float(np.median(fold_as))
    te = (1 - a_star) * tbase + a_star * te_r
    variants[f"honest_a{a_star}"] = (oof, te)
    print("honest", roc_auc_score(y, oof), a_star, fold_as, flush=True)

    for tag, mask, mask_te in [
        ("ultra", ultra, ultra_te),
        ("long", long, days_te >= 3000),
        ("midcond", mid, mid_te),
        ("ultra_or_mid", ultra | mid, ultra_te | mid_te),
    ]:
        oof2, te2, a_star, fas = nested_patch(base, oof_r, tbase, te_r, mask, mask_te)
        variants[f"patch_{tag}_a{a_star}"] = (oof2, te2)
        print(
            f"patch_{tag}",
            roc_auc_score(y, oof2),
            "slice",
            roc_auc_score(y[mask], oof2[mask]),
            a_star,
            fas,
            flush=True,
        )

    # sequential: patch ultra then midcond on result (nested each stage on outer)
    oof_u, te_u, a_u, _ = nested_patch(base, oof_r, tbase, te_r, ultra, ultra_te)
    oof_um, te_um, a_m, _ = nested_patch(oof_u, oof_r, te_u, te_r, mid & ~ultra, mid_te & ~ultra_te)
    variants[f"seq_ultra_mid_au{a_u}_am{a_m}"] = (oof_um, te_um)
    print("seq_ultra_mid", roc_auc_score(y, oof_um), a_u, a_m, flush=True)

    best = None
    results = {}
    for name, (oa, ta) in variants.items():
        direct = float(roc_auc_score(y, oa))
        for tag, arms, tlist in [
            (f"direct_{name}", [oa], [ta]),
            (f"b7+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], oa], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], ta]),
            (f"cur+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], base, oa], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], tbase, ta]),
        ]:
            if len(arms) == 1:
                res = {"nested_oof_auc": direct, "nested_oof": oa, "selected_rule": "mean"}
                dte = ta
            else:
                res = nested_select_rule(y, arms)
                dte = apply_rule(res["selected_rule"], tlist)
            results[tag] = float(res["nested_oof_auc"])
            if best is None or res["nested_oof_auc"] > best[0]:
                best = (res["nested_oof_auc"], tag, res["nested_oof"], dte)
        print(name, direct, flush=True)

    deliver, tag, deliver_oof, deliver_te = best
    promoted = deliver > CLOSEST + 1e-12
    out = Path("artifacts/b6pro_hgb_regime")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "predictions.npz",
        y=y,
        oof=deliver_oof,
        test=deliver_te,
        oof_hgb=oof_r,
        te_hgb=te_r,
    )
    lab = [c for c in sample.columns if c != "id"][0]
    sub = sample.copy()
    sub[lab] = deliver_te
    sub.to_csv(out / "submission_b6pro.csv", index=False)
    if promoted:
        dest = Path("artifacts/b6pro_long_best")
        tmp = dest / "predictions.npz.tmp"
        np.savez_compressed(tmp, y=y, oof=deliver_oof, test=deliver_te)
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
                    "source": "b6pro_hgb_regime",
                    "note": "honest nested patch; regime-slope HGB helper",
                },
                indent=2,
            )
        )
        hb = Path("artifacts/b6pro_honest_blend")
        np.savez_compressed(hb / "predictions.npz", y=y, oof=deliver_oof, test=deliver_te)
        (hb / "metrics.json").write_text(
            json.dumps({"best": tag, "nested": deliver, "gate": deliver >= GATE, "source": "hgb_regime"}, indent=2)
        )
    metrics = {
        "best": tag,
        "nested": deliver,
        "promoted": promoted,
        "gate": deliver >= GATE,
        "closest_prev": CLOSEST,
        "solo": float(roc_auc_score(y, oof_r)),
        "solo_ultra": float(roc_auc_score(y[ultra], oof_r[ultra])),
        "top": sorted(results.items(), key=lambda kv: -kv[1])[:12],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "top"}, indent=2), flush=True)
    print("TOP", metrics["top"][:8], flush=True)
    print(f"GATE={'PASS' if deliver >= GATE else 'FAIL'} nested={deliver:.8f} promoted={promoted}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
