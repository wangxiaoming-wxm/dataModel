#!/usr/bin/env python3
"""Regime-slope keepx: fix days/condition slope flips across exposure regimes.

Data evidence on closest OOF:
- short: low condition → high claim (classic)
- mid(3–7k): condition slope FLIPS (higher cond → slightly higher claim)
- ultra(≥10k): days-label corr turns NEGATIVE (−0.037); mid-cond tertile AUC≈0.555

Add fold-local regime features so CatBoost can learn different slopes instead of
one global days monotone that hurts ultra ranking.
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
PARAMS = {**PARAMS_GAP_BAG, "thread_count": 4, "iterations": 3200, "od_wait": 150, "depth": 8, "l2_leaf_reg": 8}

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


def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = pd.to_numeric(out["days"], errors="coerce")
    c = pd.to_numeric(out["condition"], errors="coerce")
    out["reg_days_cap7k"] = d.clip(upper=7000)
    out["reg_days_cap10k"] = d.clip(upper=10000)
    out["reg_days_excess7k"] = (d - 7000).clip(lower=0)
    out["reg_days_excess10k"] = (d - 10000).clip(lower=0)
    out["reg_ultra"] = (d >= 10000).astype(int).astype(str)
    out["reg_band"] = pd.cut(d, bins=[-0.1, 3000, 7000, 10000, 1e9], labels=["s", "m", "l", "u"]).astype(str)
    out["reg_cond_x_short"] = c * (d < 3000).astype(float)
    out["reg_cond_x_mid"] = c * ((d >= 3000) & (d < 7000)).astype(float)
    out["reg_cond_x_long7"] = c * ((d >= 7000) & (d < 10000)).astype(float)
    out["reg_cond_x_ultra"] = c * (d >= 10000).astype(float)
    out["reg_invcond_x_ultra"] = (1.0 / (c.abs() + 1.0)) * (d >= 10000).astype(float)
    out["reg_band_cond"] = out["reg_band"] + "|" + (c.fillna(-1).round(1).astype(str))
    # vehicle value interactions in ultra (hard_pos had lower V/x19)
    if "V" in out.columns:
        out["reg_V_x_ultra"] = pd.to_numeric(out["V"], errors="coerce") * (d >= 10000).astype(float)
    if "x19" in out.columns:
        out["reg_x19_x_ultra"] = pd.to_numeric(out["x19"], errors="coerce") * (d >= 10000).astype(float)
    if "x20" in out.columns:
        out["reg_x20_x_ultra"] = pd.to_numeric(out["x20"], errors="coerce") * (d >= 10000).astype(float)
    return out


def build_regime(X_tr, X_va, X_te):
    tr0, va0, te0, cats = build_long_keepx(X_tr, X_va, X_te)
    atr, ava, ate = add_regime(X_tr), add_regime(X_va), add_regime(X_te)
    extra = [c for c in atr.columns if c.startswith("reg_")]

    def merge(base, extra_df):
        block = extra_df.loc[:, extra].reset_index(drop=True)
        out = pd.concat([base.reset_index(drop=True), block], axis=1)
        return out.loc[:, ~out.columns.duplicated()]

    tr, va, te = merge(tr0, atr), merge(va0, ava), merge(te0, ate)
    va, te = va.reindex(columns=tr.columns), te.reindex(columns=tr.columns)
    cat_extra = ["reg_ultra", "reg_band", "reg_band_cond"]
    cats = list(dict.fromkeys(list(cats) + [c for c in cat_extra if c in tr.columns]))
    tr, va, te = tr.copy(), va.copy(), te.copy()
    for c in cats:
        tr[c] = tr[c].astype(str).fillna("__MISSING__")
        va[c] = va[c].astype(str).fillna("__MISSING__")
        te[c] = te[c].astype(str).fillna("__MISSING__")
    for c in tr.columns:
        if c in cats:
            continue
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
        med = float(tr[c].median()) if tr[c].notna().any() else 0.0
        tr[c] = tr[c].fillna(med)
        va[c] = pd.to_numeric(va[c], errors="coerce").fillna(med)
        te[c] = pd.to_numeric(te[c], errors="coerce").fillna(med)
    return tr, va, te, cats


def train(seeds):
    oof_acc = np.zeros(len(y))
    te_acc = np.zeros(len(test))
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(features, y)):
            trd, vad, ted, cats = build_regime(
                features.iloc[tr].reset_index(drop=True),
                features.iloc[va].reset_index(drop=True),
                test.copy(),
            )
            model = CatBoostClassifier(**{**PARAMS, "random_seed": seed + fold})
            model.fit(trd, y.iloc[tr], eval_set=(vad, y.iloc[va]), cat_features=cats, use_best_model=True)
            oof[va] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5
            print(f"regime s{seed} f{fold} {roc_auc_score(y.iloc[va], oof[va]):.5f}", flush=True)
        print(
            f"regime s{seed} OOF={roc_auc_score(y, oof):.6f} ultra={roc_auc_score(y.to_numpy()[ultra], oof[ultra]):.5f} "
            f"long={roc_auc_score(y.to_numpy()[long], oof[long]):.5f}",
            flush=True,
        )
        oof_acc += oof
        te_acc += pte
    return oof_acc / len(seeds), te_acc / len(seeds)


def main() -> int:
    # start soft ultra in parallel only if weak not holding CPU — use 4 seeds
    seeds = [2026, 2027, 2028, 2029]
    print("=== regime slope keepx ===", flush=True)
    oof_r, te_r = train(seeds)
    print(
        "SUMMARY",
        roc_auc_score(y, oof_r),
        "ultra",
        roc_auc_score(y.to_numpy()[ultra], oof_r[ultra]),
        "corr",
        np.corrcoef(oof_r, base)[0, 1],
        flush=True,
    )

    variants = {"raw": (oof_r, te_r), "mean": (0.5 * (base + oof_r), 0.5 * (tbase + te_r))}
    # honest α
    oof = np.zeros(len(y))
    fold_as = []
    for otr, ova in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y.to_numpy()):
        best_a, best_auc = 0.0, -1.0
        for a in np.linspace(0, 0.7, 15):
            auc = roc_auc_score(y.to_numpy()[otr], (1 - a) * base[otr] + a * oof_r[otr])
            if auc > best_auc:
                best_auc, best_a = auc, a
        fold_as.append(best_a)
        oof[ova] = (1 - best_a) * base[ova] + best_a * oof_r[ova]
    a_star = float(np.median(fold_as))
    te = (1 - a_star) * tbase + a_star * te_r
    variants[f"honest_a{a_star}"] = (oof, te)
    print("honest", roc_auc_score(y, oof), a_star, fold_as, flush=True)

    # patch ultra / long / midcond
    cond = pd.to_numeric(features["condition"], errors="coerce").fillna(1).to_numpy()
    qs = np.quantile(cond[long], [0.25, 0.75])
    for tag, mask, mask_te in [
        ("ultra", ultra, ultra_te),
        ("long", long, days_te >= 3000),
        (
            "midcond",
            long & (cond >= qs[0]) & (cond <= qs[1]),
            (days_te >= 3000)
            & (pd.to_numeric(test["condition"], errors="coerce").fillna(1).to_numpy() >= qs[0])
            & (pd.to_numeric(test["condition"], errors="coerce").fillna(1).to_numpy() <= qs[1]),
        ),
    ]:
        oof2 = base.copy()
        te2 = tbase.copy()
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
        oof2[idx] = oof_m
        te2[mask_te] = (1 - a_star) * tbase[mask_te] + a_star * te_r[mask_te]
        variants[f"patch_{tag}_a{a_star}"] = (oof2, te2)
        print(f"patch_{tag}", roc_auc_score(y, oof2), "slice", roc_auc_score(y.to_numpy()[mask], oof2[mask]), a_star, flush=True)

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
                res = nested_select_rule(y.to_numpy(), arms)
                dte = apply_rule(res["selected_rule"], tlist)
            results[tag] = float(res["nested_oof_auc"])
            if best is None or res["nested_oof_auc"] > best[0]:
                best = (res["nested_oof_auc"], tag, res["nested_oof"], dte)
        print(name, direct, flush=True)

    deliver, tag, deliver_oof, deliver_te = best
    promoted = deliver > CLOSEST + 1e-12
    out = Path("artifacts/b6pro_regime")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", y=y.to_numpy(), oof=deliver_oof, test=deliver_te, oof_regime=oof_r, te_regime=te_r)
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
                    "source": "b6pro_regime",
                },
                indent=2,
            )
        )
        hb = Path("artifacts/b6pro_honest_blend")
        np.savez_compressed(hb / "predictions.npz", y=y.to_numpy(), oof=deliver_oof, test=deliver_te)
        (hb / "metrics.json").write_text(json.dumps({"best": tag, "nested": deliver, "gate": deliver >= GATE}, indent=2))
    metrics = {
        "best": tag,
        "nested": deliver,
        "promoted": promoted,
        "gate": deliver >= GATE,
        "closest_prev": CLOSEST,
        "solo": float(roc_auc_score(y, oof_r)),
        "solo_ultra": float(roc_auc_score(y.to_numpy()[ultra], oof_r[ultra])),
        "top": sorted(results.items(), key=lambda kv: -kv[1])[:12],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "top"}, indent=2), flush=True)
    print("TOP", metrics["top"][:8], flush=True)
    print(f"GATE={'PASS' if deliver >= GATE else 'FAIL'} nested={deliver:.8f} promoted={promoted}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
