#!/usr/bin/env python3
"""Contextual nested stacker: arm logits + exposure/condition/region context.

Hypothesis: closest fails on ultra & mid-condition long because a static blend
cannot re-weight arms by business context. Learn fold-honest LR/HGB on
[logit(arms), days, condition, flags, region ordinal] with nested C / early-stop.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

B7_FLOOR = 0.7027049552615718
GATE = 0.71
CLOSEST = float(json.load(open("artifacts/b6pro_long_best/metrics.json"))["nested_oof_auc"])

ARM_PATHS = {
    "cur": "artifacts/b6pro_honest_blend/predictions.npz",
    "pick": "artifacts/b6pro_region_pick/predictions.npz",
    "b2": "artifacts/b6pro_region_blend2/predictions.npz",
    "b3": "artifacts/b6pro_region_blend3/predictions.npz",
    "nest": "artifacts/b6pro_nest_div/predictions.npz",
    "ebm": "artifacts/b6pro_ebm/predictions.npz",
    "flaml": "artifacts/b6pro_flaml/predictions.npz",
    "keepx": "artifacts/b6pro_full_keepx/predictions.npz",
}


def load_arm(path: str):
    m = np.load(path)
    oof = m["oof"].copy()
    te = m["test"].copy() if "test" in m.files else m["test_flaml"].copy()
    return oof, te


def main() -> int:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train["label"].astype(int).to_numpy()
    days = train["days"].to_numpy(float)
    days_te = test["days"].to_numpy(float)
    cond = pd.to_numeric(train["condition"], errors="coerce")
    cond = cond.fillna(cond.median()).to_numpy()
    cond_te = pd.to_numeric(test["condition"], errors="coerce").fillna(float(np.median(cond))).to_numpy()
    region = train["region"].astype(str).to_numpy()
    region_te = test["region"].astype(str).to_numpy()
    long = days >= 3000
    ultra = days >= 10000

    oofs, tes = {}, {}
    for k, p in ARM_PATHS.items():
        oofs[k], tes[k] = load_arm(p)

    arm_names = list(oofs)
    # context
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(region.reshape(-1, 1))

    def make_X(oof_dict, d, c, r, q_lo: float, q_hi: float):
        parts = [logit(np.clip(oof_dict[n], 1e-6, 1 - 1e-6)) for n in arm_names]
        rcode = enc.transform(r.reshape(-1, 1)).ravel()
        ctx = np.column_stack(
            [
                np.log1p(np.clip(d, 0, None)),
                c,
                d / (np.abs(c) + 1.0),
                c / (np.abs(d) + 1.0),
                (d >= 3000).astype(float),
                (d >= 7000).astype(float),
                (d >= 10000).astype(float),
                rcode,
            ]
        )
        ultra_f = (d >= 10000).astype(float)
        mid = ((d >= 3000) & (c >= q_lo) & (c <= q_hi)).astype(float)
        inter = []
        for p in parts[:4]:
            inter.append(p * ultra_f)
            inter.append(p * mid)
        return np.column_stack(parts + [ctx] + inter)

    results = {}
    best = None

    # --- nested LR with nested C; fold-local mid-condition quantiles ---
    Cs = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    oof_lr = np.zeros(len(y))
    fold_C = []
    te_folds = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
        long_tr = days[tr] >= 3000
        qs = np.quantile(cond[tr][long_tr], [0.25, 0.75]) if long_tr.sum() > 50 else np.quantile(cond[tr], [0.25, 0.75])
        Xtr = make_X({k: v[tr] for k, v in oofs.items()}, days[tr], cond[tr], region[tr], qs[0], qs[1])
        Xva = make_X({k: v[va] for k, v in oofs.items()}, days[va], cond[va], region[va], qs[0], qs[1])
        Xte = make_X(tes, days_te, cond_te, region_te, qs[0], qs[1])
        bestC, best_auc = Cs[0], -1.0
        for C in Cs:
            oo = np.zeros(len(tr))
            for tr2, va2 in StratifiedKFold(5, shuffle=True, random_state=1).split(Xtr, y[tr]):
                lr = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
                lr.fit(Xtr[tr2], y[tr][tr2])
                oo[va2] = lr.predict_proba(Xtr[va2])[:, 1]
            auc = roc_auc_score(y[tr], oo)
            if auc > best_auc:
                best_auc, bestC = auc, C
        fold_C.append(bestC)
        lr = LogisticRegression(C=bestC, max_iter=3000, solver="lbfgs")
        lr.fit(Xtr, y[tr])
        oof_lr[va] = lr.predict_proba(Xva)[:, 1]
        te_folds.append(lr.predict_proba(Xte)[:, 1])
    te_lr = np.mean(te_folds, axis=0)
    auc_lr = float(roc_auc_score(y, oof_lr))
    print("nested_lr", auc_lr, fold_C, "ultra", roc_auc_score(y[ultra], oof_lr[ultra]), "long", roc_auc_score(y[long], oof_lr[long]), flush=True)
    results["nested_lr"] = auc_lr
    if best is None or auc_lr > best[0]:
        best = (auc_lr, "nested_lr", oof_lr, te_lr)

    # --- nested HGB meta ---
    oof_h = np.zeros(len(y))
    te_folds = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
        long_tr = days[tr] >= 3000
        qs = np.quantile(cond[tr][long_tr], [0.25, 0.75]) if long_tr.sum() > 50 else np.quantile(cond[tr], [0.25, 0.75])
        Xtr = make_X({k: v[tr] for k, v in oofs.items()}, days[tr], cond[tr], region[tr], qs[0], qs[1])
        Xva = make_X({k: v[va] for k, v in oofs.items()}, days[va], cond[va], region[va], qs[0], qs[1])
        Xte = make_X(tes, days_te, cond_te, region_te, qs[0], qs[1])
        model = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=0,
        )
        model.fit(Xtr, y[tr])
        oof_h[va] = model.predict_proba(Xva)[:, 1]
        te_folds.append(model.predict_proba(Xte)[:, 1])
    te_h = np.mean(te_folds, axis=0)
    auc_h = float(roc_auc_score(y, oof_h))
    print("nested_hgb", auc_h, "ultra", roc_auc_score(y[ultra], oof_h[ultra]), "long", roc_auc_score(y[long], oof_h[long]), flush=True)
    results["nested_hgb"] = auc_h
    if auc_h > best[0]:
        best = (auc_h, "nested_hgb", oof_h, te_h)

    # honest α blend with cur
    cur = oofs["cur"]
    tcur = tes["cur"]
    for name, oof_m, te_m in [("lr", oof_lr, te_lr), ("hgb", oof_h, te_h)]:
        oof = np.zeros(len(y))
        fold_as = []
        for otr, ova in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
            best_a, best_auc = 0.0, -1.0
            for a in np.linspace(0, 1, 21):
                auc = roc_auc_score(y[otr], (1 - a) * cur[otr] + a * oof_m[otr])
                if auc > best_auc:
                    best_auc, best_a = auc, a
            fold_as.append(best_a)
            oof[ova] = (1 - best_a) * cur[ova] + best_a * oof_m[ova]
        a_star = float(np.median(fold_as))
        te = (1 - a_star) * tcur + a_star * te_m
        auc = float(roc_auc_score(y, oof))
        print(f"honest_cur+{name}", auc, a_star, fold_as, flush=True)
        results[f"honest_cur+{name}"] = auc
        if auc > best[0]:
            best = (auc, f"honest_cur+{name}", oof, te)

    # patch ultra only with meta
    for name, oof_m, te_m in [("lr", oof_lr, te_lr), ("hgb", oof_h, te_h)]:
        oof = cur.copy()
        te = tcur.copy()
        idx = np.where(ultra)[0]
        oof_s = np.zeros(len(idx))
        fold_as = []
        for otr, ova in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(idx)), y[idx]):
            best_a, best_auc = 0.0, -1.0
            for a in np.linspace(0, 1, 21):
                auc = roc_auc_score(y[idx[otr]], (1 - a) * cur[idx[otr]] + a * oof_m[idx[otr]])
                if auc > best_auc:
                    best_auc, best_a = auc, a
            fold_as.append(best_a)
            oof_s[ova] = (1 - best_a) * cur[idx[ova]] + best_a * oof_m[idx[ova]]
        a_star = float(np.median(fold_as))
        oof[idx] = oof_s
        te[days_te >= 10000] = (1 - a_star) * tcur[days_te >= 10000] + a_star * te_m[days_te >= 10000]
        auc = float(roc_auc_score(y, oof))
        print(f"patch_ultra_{name}", auc, "ultra", roc_auc_score(y[ultra], oof[ultra]), a_star, flush=True)
        results[f"patch_ultra_{name}"] = auc
        if auc > best[0]:
            best = (auc, f"patch_ultra_{name}", oof, te)

    deliver, tag, deliver_oof, deliver_te = best
    # also try nested_select with B7
    from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    for name, oof_m, te_m in [(tag, deliver_oof, deliver_te), ("lr", oof_lr, te_lr), ("hgb", oof_h, te_h)]:
        for tname, arms, tlist in [
            (f"b7+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], oof_m], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], te_m]),
            (f"b7+cur+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], cur, oof_m], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], tcur, te_m]),
        ]:
            res = nested_select_rule(y, arms)
            results[tname] = float(res["nested_oof_auc"])
            if res["nested_oof_auc"] > deliver:
                deliver = float(res["nested_oof_auc"])
                tag = tname
                deliver_oof = res["nested_oof"]
                deliver_te = apply_rule(res["selected_rule"], tlist)

    promoted = deliver > CLOSEST + 1e-12
    out = Path("artifacts/b6pro_ctx_stack")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", y=y, oof=deliver_oof, test=deliver_te, oof_lr=oof_lr, oof_hgb=oof_h)
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
                    "source": "b6pro_ctx_stack",
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
        "top": sorted(results.items(), key=lambda kv: -kv[1])[:15],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "top"}, indent=2), flush=True)
    print("TOP", metrics["top"][:10], flush=True)
    print(f"GATE={'PASS' if deliver >= GATE else 'FAIL'} nested={deliver:.8f} promoted={promoted}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
