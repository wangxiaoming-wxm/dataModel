#!/usr/bin/env python3
"""HistGradientBoosting residual corrector on ultra / mid-condition long.

Heterogeneous arm (not CatBoost) to lift high-leverage slices where closest is weak:
ultra≈0.631, long condQ2≈0.620. Fold-local features only; outer nested α / patch.
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

NUM = ["days", "condition", "cc", "V", "max_g", "age_range", "livability", "x18", "x19", "x20"]
CAT = ["region", "code", "version", "grades", "month", "t3", "source"]


def build_matrix(tr_df, va_df, te_df):
    """Fold-local ordinal cats + numeric exposure×condition interactions."""
    def enrich(df):
        out = df.copy()
        d = pd.to_numeric(out["days"], errors="coerce")
        c = pd.to_numeric(out["condition"], errors="coerce")
        out["_log_days"] = np.log1p(d.clip(lower=0))
        out["_cond"] = c
        out["_days_x_invcond"] = d / (c.abs() + 1.0)
        out["_ratio"] = c / (d.abs() + 1.0)
        out["_ultra"] = (d >= 10000).astype(float)
        out["_long"] = (d >= 3000).astype(float)
        # car from source
        car = out["source"].astype(str).str.extract(r"(CAR_\d+)", expand=False).fillna("__NA__")
        out["_car"] = car
        return out

    tr, va, te = enrich(tr_df), enrich(va_df), enrich(te_df)
    num_cols = NUM + ["_log_days", "_cond", "_days_x_invcond", "_ratio", "_ultra", "_long"]
    cat_cols = CAT + ["_car"]

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(tr[cat_cols].astype(str))

    def pack(df):
        Xn = df[num_cols].apply(pd.to_numeric, errors="coerce")
        for c in num_cols:
            med = float(tr[c].pipe(pd.to_numeric, errors="coerce").median()) if c in tr else 0.0
            # use train med from outer scope via tr
            pass
        meds = {c: float(pd.to_numeric(tr[c], errors="coerce").median()) for c in num_cols}
        Xn = Xn.fillna(meds)
        Xc = enc.transform(df[cat_cols].astype(str))
        return np.hstack([Xn.to_numpy(dtype=float), Xc])

    return pack(tr), pack(va), pack(te)


def train_hgb(seeds, sample_mask_fn=None):
    oof_acc = np.zeros(len(y))
    te_acc = np.zeros(len(test))
    feats = train.drop(columns=["label"])
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(feats, y)):
            Xtr, Xva, Xte = build_matrix(feats.iloc[tr], feats.iloc[va], test)
            sw = None
            if sample_mask_fn is not None:
                w = np.ones(len(tr))
                w *= sample_mask_fn(feats.iloc[tr].reset_index(drop=True))
                sw = w
            model = HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.05,
                max_iter=400,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=seed + fold,
            )
            model.fit(Xtr, y[tr], sample_weight=sw)
            oof[va] = model.predict_proba(Xva)[:, 1]
            pte += model.predict_proba(Xte)[:, 1] / 5
            print(f"hgb s{seed} f{fold} {roc_auc_score(y[va], oof[va]):.5f}", flush=True)
        print(
            f"hgb s{seed} OOF={roc_auc_score(y, oof):.6f} ultra={roc_auc_score(y[ultra], oof[ultra]):.5f} "
            f"long={roc_auc_score(y[long], oof[long]):.5f}",
            flush=True,
        )
        oof_acc += oof
        te_acc += pte
    return oof_acc / len(seeds), te_acc / len(seeds)


def midcond_w(X):
    d = X["days"].to_numpy(float)
    c = pd.to_numeric(X["condition"], errors="coerce").fillna(1.0).to_numpy()
    w = np.ones(len(X))
    qs = np.quantile(c[d >= 3000], [0.25, 0.75]) if (d >= 3000).sum() > 30 else np.quantile(c, [0.25, 0.75])
    w[(d >= 3000) & (c >= qs[0]) & (c <= qs[1])] = 2.0
    w[d >= 10000] *= 1.5
    return w


def main() -> int:
    seeds = [2026, 2027, 2028]
    print("=== plain hgb ===", flush=True)
    oof_p, te_p = train_hgb(seeds, None)
    print("=== midcond-w hgb ===", flush=True)
    oof_m, te_m = train_hgb(seeds, midcond_w)
    locals_ = {"plain": (oof_p, te_p), "midw": (oof_m, te_m)}

    variants = {}
    for name, (oof_r, te_r) in locals_.items():
        print(
            "SUMMARY",
            name,
            roc_auc_score(y, oof_r),
            "ultra",
            roc_auc_score(y[ultra], oof_r[ultra]),
            "corr",
            np.corrcoef(oof_r, base)[0, 1],
            flush=True,
        )
        variants[f"raw_{name}"] = (oof_r, te_r)
        # honest global α
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
        variants[f"honest_{name}_a{a_star}"] = (oof, te)
        print("honest", name, roc_auc_score(y, oof), a_star, fold_as, flush=True)

        for tag_mask, mask, mask_te in [
            ("ultra", ultra, ultra_te),
            (
                "midcond",
                long
                & (
                    pd.to_numeric(train["condition"], errors="coerce").fillna(1).to_numpy()
                    >= np.quantile(pd.to_numeric(train["condition"], errors="coerce").fillna(1).to_numpy()[long], 0.25)
                )
                & (
                    pd.to_numeric(train["condition"], errors="coerce").fillna(1).to_numpy()
                    <= np.quantile(pd.to_numeric(train["condition"], errors="coerce").fillna(1).to_numpy()[long], 0.75)
                ),
                (days_te >= 3000),
            ),
        ]:
            if tag_mask == "midcond":
                cond = pd.to_numeric(train["condition"], errors="coerce").fillna(1).to_numpy()
                qs = np.quantile(cond[long], [0.25, 0.75])
                mask = long & (cond >= qs[0]) & (cond <= qs[1])
                cte = pd.to_numeric(test["condition"], errors="coerce").fillna(1).to_numpy()
                mask_te = (days_te >= 3000) & (cte >= qs[0]) & (cte <= qs[1])
            oof2 = base.copy()
            te2 = tbase.copy()
            idx = np.where(mask)[0]
            if len(idx) < 100:
                continue
            oof_m = np.zeros(len(idx))
            fold_as = []
            for otr, ova in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(idx)), y[idx]):
                best_a, best_auc = 0.0, -1.0
                for a in np.linspace(0, 1, 21):
                    auc = roc_auc_score(y[idx[otr]], (1 - a) * base[idx[otr]] + a * oof_r[idx[otr]])
                    if auc > best_auc:
                        best_auc, best_a = auc, a
                fold_as.append(best_a)
                oof_m[ova] = (1 - best_a) * base[idx[ova]] + best_a * oof_r[idx[ova]]
            a_star = float(np.median(fold_as))
            oof2[idx] = oof_m
            te2[mask_te] = (1 - a_star) * tbase[mask_te] + a_star * te_r[mask_te]
            variants[f"patch_{tag_mask}_{name}_a{a_star}"] = (oof2, te2)
            print(f"patch_{tag_mask}", name, roc_auc_score(y, oof2), a_star, flush=True)

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
    out = Path("artifacts/b6pro_hgb_mid")
    out.mkdir(parents=True, exist_ok=True)
    save = {"y": y, "oof": deliver_oof, "test": deliver_te}
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
                    "source": "b6pro_hgb_mid",
                },
                indent=2,
            )
        )
        hb = Path("artifacts/b6pro_honest_blend")
        np.savez_compressed(hb / "predictions.npz", y=y, oof=deliver_oof, test=deliver_te)
        (hb / "metrics.json").write_text(json.dumps({"best": tag, "nested": deliver, "gate": deliver >= GATE}, indent=2))
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
