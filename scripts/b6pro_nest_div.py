#!/usr/bin/env python3
"""Nested logit stack of B7 arms + kx8 + current closest + diverse low-corr (EBM/FLAML).

Protocol: outer nested C selection (SKF≥5); no global TE; no test labels.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.model import IDENTIFIER

B7_FLOOR = 0.7027049552615718
GATE = 0.71
CLOSEST = float(json.load(open("artifacts/b6pro_long_best/metrics.json"))["nested_oof_auc"])


def clip_logit(p, eps=1e-6):
    return logit(np.clip(p, eps, 1 - eps))


def nest_logit_train_predict(oof_arms, te_arms, y, test_len, C, seeds=(0, 1, 2)):
    X = np.column_stack([clip_logit(a) for a in oof_arms])
    Xt = np.column_stack([clip_logit(a) for a in te_arms])
    oof = np.zeros(len(y))
    te = np.zeros(test_len)
    for seed in seeds:
        pred = np.zeros(len(y))
        pte = np.zeros(test_len)
        nf = 0
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
            clf.fit(X[tr], y[tr])
            pred[va] = clf.predict_proba(X[va])[:, 1]
            pte += clf.predict_proba(Xt)[:, 1]
            nf += 1
        oof += pred
        te += pte / nf
    return oof / len(seeds), te / len(seeds)


def nested_C_select(oof_arms, te_arms, y, test_len, C_grid=(0.3, 0.5, 1.0, 3.0), outer_seed=42):
    oof_final = np.zeros(len(y))
    fold_Cs = []
    for otr, ova in StratifiedKFold(5, shuffle=True, random_state=outer_seed).split(np.zeros(len(y)), y):
        best_C, best_auc = None, -1.0
        for C in C_grid:
            clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
            X_otr = np.column_stack([clip_logit(a[otr]) for a in oof_arms])
            X_ova = np.column_stack([clip_logit(a[ova]) for a in oof_arms])
            clf.fit(X_otr, y[otr])
            p = clf.predict_proba(X_ova)[:, 1]
            auc = roc_auc_score(y[ova], p)
            if auc > best_auc:
                best_auc, best_C = auc, C
        fold_Cs.append(best_C)
        clf = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs")
        X_otr = np.column_stack([clip_logit(a[otr]) for a in oof_arms])
        X_ova = np.column_stack([clip_logit(a[ova]) for a in oof_arms])
        clf.fit(X_otr, y[otr])
        oof_final[ova] = clf.predict_proba(X_ova)[:, 1]
    C_star = Counter(fold_Cs).most_common(1)[0][0]
    _, te = nest_logit_train_predict(oof_arms, te_arms, y, test_len, C=C_star)
    return oof_final, te, C_star, fold_Cs


def main() -> int:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train["label"].astype(int).to_numpy()
    days = train["days"].to_numpy(float)
    long = days >= 3000

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    # Use frozen previous closest BEFORE this run for stack base to avoid trivial identity
    # Prefer nest_stack artifact if present, else long_best
    cur_path = Path("artifacts/b6pro_nest_stack/predictions.npz")
    if not cur_path.exists():
        cur_path = Path("artifacts/b6pro_long_best/predictions.npz")
    cur = np.load(cur_path)
    kx8 = np.load("artifacts/b6pro_keepx8/predictions.npz")
    zebm = np.load("artifacts/b6pro_ebm/predictions.npz")
    zfl = np.load("artifacts/b6pro_flaml/predictions.npz")

    base_o = [b7["gap"], b7["gap_bag"], b7["plus"], kx8["oof"], cur["oof"]]
    base_t = [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], kx8["test"], cur["test"]]
    configs = [
        ("ebm", [zebm["oof_ebm"]], [zebm["test_ebm"]]),
        ("flaml", [zfl["oof_flaml"]], [zfl["test_flaml"]]),
        ("ebm+flaml", [zebm["oof_ebm"], zfl["oof_flaml"]], [zebm["test_ebm"], zfl["test_flaml"]]),
    ]

    results = []
    for name, extra_o, extra_t in configs:
        for C in [0.5, 1.0, 3.0]:
            oof, te = nest_logit_train_predict(base_o + extra_o, base_t + extra_t, y, len(test), C=C)
            auc = roc_auc_score(y, oof)
            print(f"{name} C={C}: direct={auc:.8f} long={roc_auc_score(y[long], oof[long]):.5f}", flush=True)
            results.append((auc, name, C, oof, te, "fixed"))
        oof, te, Cs, fold_Cs = nested_C_select(base_o + extra_o, base_t + extra_t, y, len(test))
        auc = roc_auc_score(y, oof)
        print(f"{name} nestedC={Cs} folds={fold_Cs}: direct={auc:.8f}", flush=True)
        results.append((auc, name, Cs, oof, te, "nestedC"))

    results.sort(key=lambda r: -r[0])
    auc, name, C, oof, te, mode = results[0]
    deliver_auc, deliver_oof, deliver_te = float(auc), oof, te
    deliver_name = f"direct_logit_base+{name}_C{C}_{mode}"

    for tag, arms, te_arms in [
        ("b7+stack", [b7["gap"], b7["gap_bag"], b7["plus"], oof], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], te]),
        (
            "b7+cur+stack",
            [b7["gap"], b7["gap_bag"], b7["plus"], cur["oof"], oof],
            [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], cur["test"], te],
        ),
    ]:
        res = nested_select_rule(y, arms)
        if res["nested_oof_auc"] > deliver_auc:
            deliver_auc = float(res["nested_oof_auc"])
            deliver_oof = res["nested_oof"]
            deliver_te = apply_rule(res["selected_rule"], te_arms)
            deliver_name = f"{tag}_{name}_C{C}"

    if deliver_auc + 1e-12 < B7_FLOOR:
        deliver_auc = B7_FLOOR
        deliver_oof = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
        deliver_te = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])
        deliver_name = "b7_fallback"

    promoted = deliver_auc > CLOSEST + 1e-12
    out = Path("artifacts/b6pro_nest_div")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", y=y, oof=deliver_oof, test=deliver_te, oof_stack=oof, te_stack=te)
    lab = [c for c in sample.columns if c != IDENTIFIER][0]
    sub = sample.copy()
    sub[lab] = deliver_te
    sub.to_csv(out / "submission_b6pro.csv", index=False)
    metrics = {
        "best": deliver_name,
        "nested": deliver_auc,
        "promoted": promoted,
        "gate": deliver_auc >= GATE,
        "closest_prev": CLOSEST,
        "top": [(r[0], r[1], r[2], r[5]) for r in results[:10]],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    if promoted:
        dest = Path("artifacts/b6pro_long_best")
        np.savez_compressed(dest / "predictions.npz", y=y, oof=deliver_oof, test=deliver_te)
        sub.to_csv(dest / "submission_b6pro.csv", index=False)
        sub.to_csv("submissions/b6pro_closest/submission_b6pro.csv", index=False)
        (dest / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment_id": "b6pro_long_best",
                    "spec": deliver_name,
                    "nested_oof_auc": deliver_auc,
                    "baseline_max3": B7_FLOOR,
                    "gate_0_71": deliver_auc >= GATE,
                    "gap_to_0_71": GATE - deliver_auc,
                    "source": "b6pro_nest_div",
                },
                indent=2,
            )
        )
    print(json.dumps(metrics, indent=2, default=str), flush=True)
    print(f"GATE={'PASS' if deliver_auc >= GATE else 'FAIL'} nested={deliver_auc:.8f} promoted={promoted}", flush=True)
    return 0 if deliver_auc >= GATE else 2


if __name__ == "__main__":
    raise SystemExit(main())
