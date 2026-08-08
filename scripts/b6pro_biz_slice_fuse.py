#!/usr/bin/env python3
"""Business-slice nested pick/blend over existing diverse arms.

Targets high-leverage failure modes from data mining:
- ultra days>=10k (AUC≈0.631)
- long mid-condition quartile (condQ2≈0.620)
- weak regions f09d/908d/9685 on long

Protocol: outer SKF nested arm/α selection per slice; no global TE; no OOF weight search on report score.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule

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
    "nest_stack": "artifacts/b6pro_nest_stack/predictions.npz",
    "f09d": "artifacts/b6pro_f09d_blend/predictions.npz",
    "iso": "artifacts/b6pro_iso_resid/predictions.npz",
}


def load_arm(path: str) -> tuple[np.ndarray, np.ndarray]:
    m = np.load(path)
    oof = m["oof"].copy()
    if "test" in m.files:
        te = m["test"].copy()
    elif "test_flaml" in m.files:
        te = m["test_flaml"].copy()
    elif "te_cls" in m.files:
        te = m["te_cls"].copy()
    else:
        raise KeyError(path)
    return oof, te


def make_slices(days, cond, region, qs):
    s = np.full(len(days), "short", dtype=object)
    s[days >= 3000] = "long"
    s[(days >= 3000) & (cond >= qs[0]) & (cond < qs[1])] = "long_c1"
    s[(days >= 3000) & (cond >= qs[1]) & (cond < qs[2])] = "long_c2"
    s[(days >= 3000) & (cond < qs[0])] = "long_c0"
    s[(days >= 3000) & (cond >= qs[2])] = "long_c3"
    s[days >= 10000] = "ultra"
    for w in ("f09d", "908d", "9685"):
        s[(region == w) & (days >= 3000) & (days < 10000)] = f"weak_{w}"
        s[(region == w) & (days >= 10000)] = f"ultra_{w}"
    return s


def nested_slice_pick(y, sid, sid_te, oofs, tes, improve: float, seed: int = 0):
    names = list(oofs)
    oof = np.zeros(len(y))
    te = np.zeros(len(sid_te))
    fold_maps = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        mapping = {}
        for s in sorted(set(sid)):
            msk = np.zeros(len(y), dtype=bool)
            msk[tr] = True
            msk &= sid == s
            if msk.sum() < 60 or y[msk].sum() < 4 or (y[msk] == 0).sum() < 4:
                mapping[s] = "cur"
                continue
            best_n, best = "cur", roc_auc_score(y[msk], oofs["cur"][msk])
            for n in names:
                if n == "cur":
                    continue
                auc = roc_auc_score(y[msk], oofs[n][msk])
                if auc > best + improve:
                    best, best_n = auc, n
            mapping[s] = best_n
        fold_maps.append(mapping)
        for i in va:
            oof[i] = oofs[mapping[sid[i]]][i]
        for i, s in enumerate(sid_te):
            te[i] += tes[mapping.get(s, "cur")][i]
    te /= len(fold_maps)
    return oof, te, fold_maps


def nested_slice_blend(y, sid, sid_te, oofs, tes, alphas, seed: int = 0):
    helpers = [n for n in oofs if n != "cur"]
    oof = np.zeros(len(y))
    te = np.zeros(len(sid_te))
    fold_maps = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        mapping = {}
        for s in sorted(set(sid)):
            msk = np.zeros(len(y), dtype=bool)
            msk[tr] = True
            msk &= sid == s
            if msk.sum() < 60 or y[msk].sum() < 4:
                mapping[s] = ("cur", 0.0)
                continue
            best = ("cur", 0.0)
            best_auc = roc_auc_score(y[msk], oofs["cur"][msk])
            for h in helpers:
                for a in alphas:
                    sc = (1 - a) * oofs["cur"][msk] + a * oofs[h][msk]
                    auc = roc_auc_score(y[msk], sc)
                    if auc > best_auc:
                        best_auc = auc
                        best = (h, float(a))
            mapping[s] = best
        fold_maps.append(mapping)
        for i in va:
            h, a = mapping[sid[i]]
            oof[i] = (1 - a) * oofs["cur"][i] + a * oofs[h][i]
        for i, s in enumerate(sid_te):
            h, a = mapping.get(s, ("cur", 0.0))
            te[i] += (1 - a) * tes["cur"][i] + a * tes[h][i]
    te /= len(fold_maps)
    return oof, te, fold_maps


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
    qs = np.quantile(cond[long], [0.25, 0.5, 0.75])

    oofs, tes = {}, {}
    for k, p in ARM_PATHS.items():
        oofs[k], tes[k] = load_arm(p)
        print(k, float(roc_auc_score(y, oofs[k])), flush=True)

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")

    sid = make_slices(days, cond, region, qs)
    sid_te = make_slices(days_te, cond_te, region_te, qs)
    print("slices", {k: int((sid == k).sum()) for k in sorted(set(sid))}, flush=True)

    variants = {}
    for improve in (0.0, 0.001, 0.002):
        oof, te, maps = nested_slice_pick(y, sid, sid_te, oofs, tes, improve)
        variants[f"pick_i{improve}"] = (oof, te, maps)
        print(
            f"pick_i{improve}",
            float(roc_auc_score(y, oof)),
            "long",
            float(roc_auc_score(y[long], oof[long])),
            "ultra",
            float(roc_auc_score(y[ultra], oof[ultra])),
            flush=True,
        )
        for s in sorted(set(sid)):
            print(" ", s, Counter(m[s] for m in maps).most_common(2), flush=True)

    alphas = np.linspace(0, 0.8, 9)
    oof, te, maps = nested_slice_blend(y, sid, sid_te, oofs, tes, alphas)
    variants["blend"] = (oof, te, maps)
    print(
        "blend",
        float(roc_auc_score(y, oof)),
        "long",
        float(roc_auc_score(y[long], oof[long])),
        "ultra",
        float(roc_auc_score(y[ultra], oof[ultra])),
        flush=True,
    )

    # coarse slices
    sid_c = np.full(len(days), "short", dtype=object)
    sid_c[long] = "long"
    sid_c[ultra] = "ultra"
    sid_c[(days >= 3000) & (cond >= qs[1]) & (cond < qs[2])] = "long_c2"
    sid_c[(region == "f09d") & long] = "f09d_long"
    sid_c[(region == "908d") & long] = "r908d_long"
    sid_te_c = np.full(len(days_te), "short", dtype=object)
    sid_te_c[days_te >= 3000] = "long"
    sid_te_c[days_te >= 10000] = "ultra"
    sid_te_c[(days_te >= 3000) & (cond_te >= qs[1]) & (cond_te < qs[2])] = "long_c2"
    sid_te_c[(region_te == "f09d") & (days_te >= 3000)] = "f09d_long"
    sid_te_c[(region_te == "908d") & (days_te >= 3000)] = "r908d_long"

    oof, te, maps = nested_slice_pick(y, sid_c, sid_te_c, oofs, tes, 0.001)
    variants["coarse_pick"] = (oof, te, maps)
    print("coarse_pick", float(roc_auc_score(y, oof)), flush=True)
    oof, te, maps = nested_slice_blend(y, sid_c, sid_te_c, oofs, tes, alphas)
    variants["coarse_blend"] = (oof, te, maps)
    print("coarse_blend", float(roc_auc_score(y, oof)), flush=True)

    # outer nest α between cur and best biz variant
    best = None
    results = {}
    for name, (oa, ta, _) in variants.items():
        direct = float(roc_auc_score(y, oa))
        # honest α with cur
        oof = np.zeros(len(y))
        fold_as = []
        for otr, ova in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
            best_a, best_auc = 0.0, -1.0
            for a in np.linspace(0, 1.0, 21):
                auc = roc_auc_score(y[otr], (1 - a) * oofs["cur"][otr] + a * oa[otr])
                if auc > best_auc:
                    best_auc, best_a = auc, a
            fold_as.append(best_a)
            oof[ova] = (1 - best_a) * oofs["cur"][ova] + best_a * oa[ova]
        a_star = float(np.median(fold_as))
        te = (1 - a_star) * tes["cur"] + a_star * ta
        honest = float(roc_auc_score(y, oof))
        print(f"honest_cur+{name}", honest, a_star, fold_as, flush=True)
        for tag, arms, tlist in [
            (f"direct_{name}", [oa], [ta]),
            (f"honest_{name}", [oof], [te]),
            (
                f"b7+{name}",
                [b7["gap"], b7["gap_bag"], b7["plus"], oa],
                [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], ta],
            ),
            (
                f"b7+honest_{name}",
                [b7["gap"], b7["gap_bag"], b7["plus"], oof],
                [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], te],
            ),
        ]:
            if len(arms) == 1:
                res = {"nested_oof_auc": float(roc_auc_score(y, arms[0])), "nested_oof": arms[0], "selected_rule": "mean"}
                deliver_te = tlist[0]
            else:
                res = nested_select_rule(y, arms)
                deliver_te = apply_rule(res["selected_rule"], tlist)
            results[tag] = float(res["nested_oof_auc"])
            if best is None or res["nested_oof_auc"] > best[0]:
                best = (res["nested_oof_auc"], tag, res["nested_oof"], deliver_te)

    deliver, tag, deliver_oof, deliver_te = best
    promoted = deliver > CLOSEST + 1e-12
    out = Path("artifacts/b6pro_biz_slice")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", y=y, oof=deliver_oof, test=deliver_te)
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
                    "source": "b6pro_biz_slice",
                },
                indent=2,
            )
        )
        # keep honest_blend in sync only if clearly better
        hb = Path("artifacts/b6pro_honest_blend")
        np.savez_compressed(hb / "predictions.npz", y=y, oof=deliver_oof, test=deliver_te)
        (hb / "metrics.json").write_text(
            json.dumps({"best": tag, "nested": deliver, "gate": deliver >= GATE}, indent=2)
        )
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
