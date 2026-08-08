#!/usr/bin/env python3
"""Anti-days / days-dropout arm: force residual signal beyond exposure monotonicity.

LL-pair audit: wrong pairs are anti-monotonic in days/condition. Models that
cannot rely on raw days must use region×car×latent structure — diverse vs B7.
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
from insurance_claim.b6pro_long_features import build_long_keepx, build_long_aging
from insurance_claim.model import IDENTIFIER, TARGET
from insurance_claim.train_b6 import PARAMS_GAP_BAG, build_gap

B7_FLOOR = 0.7027049552615718
GATE = 0.71
CLOSEST = 0.7054481147284526
WEAK = frozenset({"908d", "f09d", "9685", "fafc", "f167", "ab86"})

PARAMS = {**PARAMS_GAP_BAG, "thread_count": 4, "iterations": 3000, "od_wait": 150}


def drop_days_cols(tr, va, te, cats):
    """Remove raw days and obvious monotonic transforms; keep bin cats & residuals."""
    drop_exact = {
        "days",
        "log_days",
        "long_log_days",
        "days_x_cond",
        "days_x_invcond",
        "long_days_x_invcond",
        "long_days_minus_region_med",
        "cond_over_days",
        "days_times_cond",
    }
    # keep long_days_fine / region_days bins — they are discrete risk curves
    cols = [c for c in tr.columns if c not in drop_exact and not c.startswith("te_")]
    # also drop numeric columns that are pure days aliases if present
    def filt(df):
        out = df.loc[:, [c for c in cols if c in df.columns]].copy()
        return out

    tr2, va2, te2 = filt(tr), filt(va), filt(te)
    cats2 = [c for c in cats if c in tr2.columns]
    va2 = va2.reindex(columns=tr2.columns)
    te2 = te2.reindex(columns=tr2.columns)
    return tr2, va2, te2, cats2


def noise_days(tr, va, te, cats, rng):
    """Keep days but permute within fold train; va/te get train-marginal noise."""
    tr2, va2, te2 = tr.copy(), va.copy(), te.copy()
    if "days" in tr2.columns:
        vals = tr2["days"].to_numpy().copy()
        rng.shuffle(vals)
        tr2["days"] = vals
        # assign va/te random draws from train days marginal
        va2["days"] = rng.choice(tr["days"].to_numpy(), size=len(va2))
        te2["days"] = rng.choice(tr["days"].to_numpy(), size=len(te2))
    return tr2, va2, te2, cats


def write_submission(sample, test, pred, path: Path):
    out = sample.copy()
    label_col = [c for c in out.columns if c != IDENTIFIER][0]
    out[label_col] = np.asarray(pred, float)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def region_blend(max3, long_spec, region, days, wo):
    out = max3.copy()
    long = days >= 3000
    weak_m = long & np.isin(region, list(WEAK))
    other = long & ~np.isin(region, list(WEAK))
    out[weak_m] = long_spec[weak_m]
    out[other] = wo * long_spec[other] + (1 - wo) * max3[other]
    return out


def run_arm(features, y, test, builder, mode: str, seeds, long_only=False):
    days = features["days"].to_numpy(float)
    if long_only:
        mask = days >= 3000
        idx = np.where(mask)[0]
    else:
        idx = np.arange(len(y))
        mask = np.ones(len(y), bool)

    oof_acc = np.zeros(len(y))
    te_acc = np.zeros(len(test))
    for seed in seeds:
        oof = np.zeros(len(y))
        pte = np.zeros(len(test))
        Xl = features.iloc[idx].reset_index(drop=True)
        yl = y.iloc[idx].reset_index(drop=True)
        rng = np.random.default_rng(seed)
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xl, yl)):
            gtr, gva = idx[tr], idx[va]
            trd, vad, ted, cats = builder(
                features.iloc[gtr].reset_index(drop=True),
                features.iloc[gva].reset_index(drop=True),
                test.copy(),
            )
            if mode == "drop":
                trd, vad, ted, cats = drop_days_cols(trd, vad, ted, cats)
            elif mode == "noise":
                trd, vad, ted, cats = noise_days(trd, vad, ted, cats, rng)
            model = CatBoostClassifier(**{**PARAMS, "random_seed": seed + fold})
            model.fit(trd, y.iloc[gtr], eval_set=(vad, y.iloc[gva]), cat_features=cats, use_best_model=True)
            oof[gva] = model.predict_proba(vad)[:, 1]
            pte += model.predict_proba(ted)[:, 1] / 5.0
            print(
                f"{mode}{'L' if long_only else 'F'} s{seed} f{fold} "
                f"auc={roc_auc_score(y.iloc[gva], oof[gva]):.5f}",
                flush=True,
            )
        print(
            f"{mode}{'L' if long_only else 'F'} s{seed} OOF={roc_auc_score(y, oof):.6f} "
            f"slice={roc_auc_score(y.to_numpy()[mask], oof[mask]):.6f}",
            flush=True,
        )
        oof_acc += oof
        te_acc += pte
    return oof_acc / len(seeds), te_acc / len(seeds)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    ap.add_argument("--builder", choices=["gap", "aging", "keepx"], default="keepx")
    ap.add_argument("--mode", choices=["drop", "noise"], default="drop")
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/b6pro_nodays"))
    args = ap.parse_args()

    builders = {"gap": build_gap, "aging": build_long_aging, "keepx": build_long_keepx}
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("submit_sample.csv")
    y = train[TARGET].astype(int)
    features = train.drop(columns=[TARGET])
    days = features["days"].to_numpy(float)
    days_te = test["days"].to_numpy(float)
    region = train["region"].astype(str).to_numpy()
    region_te = test["region"].astype(str).to_numpy()
    long = days >= 3000

    b7 = np.load("reference/b7_closest/predictions.npz")
    fr = np.load("artifacts/b6pro_frozen/predictions.npz")
    max3 = np.maximum.reduce([b7["gap"], b7["gap_bag"], b7["plus"]])
    tmax = np.maximum.reduce([fr["test_gap"], fr["test_gap_bag"], fr["test_plus"]])
    cur = np.load("artifacts/b6pro_long_best/predictions.npz")
    aging = np.load("artifacts/b6pro_long_only_aging/predictions.npz")
    gap = np.load("artifacts/b6pro_long_only_gap/predictions.npz")
    keepx = np.load("artifacts/b6pro_long_only_keepx/predictions.npz")
    meanL3 = (aging["oof_long_only"] + gap["oof_long_only"] + keepx["oof_long_only"]) / 3.0
    tmeanL3 = (aging["test_long_only"] + gap["test_long_only"] + keepx["test_long_only"]) / 3.0

    oof_nd, te_nd = run_arm(
        features, y, test, builders[args.builder], args.mode, args.seeds, long_only=args.long_only
    )
    print(
        "nodays",
        roc_auc_score(y, oof_nd),
        "long",
        roc_auc_score(y.to_numpy()[long], oof_nd[long]),
        "corr(max3)",
        float(np.corrcoef(oof_nd, max3)[0, 1]),
        flush=True,
    )

    specs = {
        "raw": (oof_nd, te_nd),
        "max_m3": (np.maximum(max3, oof_nd), np.maximum(tmax, te_nd)),
        "mean_m3": (0.5 * (max3 + oof_nd), 0.5 * (tmax + te_nd)),
    }
    # long patch / region blends combining with meanL3
    for wo in (0.0, 0.15, 0.3):
        mix = 0.5 * (meanL3 + oof_nd)
        tmix = 0.5 * (tmeanL3 + te_nd)
        specs[f"rb_mix_wo{wo}"] = (
            region_blend(max3, mix, region, days, wo),
            region_blend(tmax, tmix, region_te, days_te, wo),
        )
        specs[f"rb_nd_wo{wo}"] = (
            region_blend(max3, oof_nd, region, days, wo),
            region_blend(tmax, te_nd, region_te, days_te, wo),
        )
    # long-only replace with mean(nd, meanL3)
    patch = max3.copy()
    patch[long] = 0.5 * (max3[long] + oof_nd[long])
    tpatch = tmax.copy()
    tpatch[days_te >= 3000] = 0.5 * (tmax[days_te >= 3000] + te_nd[days_te >= 3000])
    specs["meanL_nd"] = (patch, tpatch)

    results = {}
    best_name, best_res, best_pair = None, None, None
    for name, (oof_arm, te_arm) in specs.items():
        for tag, oof_arms, te_arms in [
            (f"b7+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], oof_arm], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], te_arm]),
            (f"cur+{name}", [b7["gap"], b7["gap_bag"], b7["plus"], cur["oof"], oof_arm], [fr["test_gap"], fr["test_gap_bag"], fr["test_plus"], cur["test"], te_arm]),
            (f"max3×{name}", [max3, oof_arm], [tmax, te_arm]),
        ]:
            res = nested_select_rule(y.to_numpy(), oof_arms)
            results[tag] = float(res["nested_oof_auc"])
            print(f"{tag}: nested={res['nested_oof_auc']:.8f}", flush=True)
            if best_res is None or res["nested_oof_auc"] > best_res["nested_oof_auc"]:
                best_name, best_res, best_pair = tag, res, (oof_arms, te_arms)

    deliver_auc = best_res["nested_oof_auc"]
    deliver_oof = best_res["nested_oof"]
    deliver_test = apply_rule(best_res["selected_rule"], best_pair[1])
    if deliver_auc + 1e-12 < B7_FLOOR:
        best_name = "b7_fallback"
        deliver_auc = float(roc_auc_score(y, max3))
        deliver_oof, deliver_test = max3, tmax

    promoted = deliver_auc > CLOSEST + 1e-12
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y=y.to_numpy(),
        oof=deliver_oof,
        test=deliver_test,
        oof_nd=oof_nd,
        test_nd=te_nd,
    )
    write_submission(sample, test, deliver_test, args.output_dir / "submission_b6pro.csv")
    if promoted:
        dest = Path("artifacts/b6pro_long_best")
        np.savez_compressed(dest / "predictions.npz", y=y.to_numpy(), oof=deliver_oof, test=deliver_test, arm=oof_nd)
        write_submission(sample, test, deliver_test, dest / "submission_b6pro.csv")
        write_submission(sample, test, deliver_test, Path("submissions/b6pro_closest/submission_b6pro.csv"))
        (dest / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment_id": "b6pro_long_best",
                    "spec": best_name,
                    "nested_oof_auc": deliver_auc,
                    "baseline_max3": B7_FLOOR,
                    "gate_0_71": deliver_auc >= GATE,
                    "gap_to_0_71": GATE - deliver_auc,
                    "source": "b6pro_nodays",
                },
                indent=2,
            )
        )

    metrics = {
        "experiment_id": "b6pro_nodays",
        "builder": args.builder,
        "mode": args.mode,
        "long_only": args.long_only,
        "best_fusion": best_name,
        "nested_oof_auc": deliver_auc,
        "nd_oof_auc": float(roc_auc_score(y, oof_nd)),
        "nd_long_auc": float(roc_auc_score(y.to_numpy()[long], oof_nd[long])),
        "nd_corr_max3": float(np.corrcoef(oof_nd, max3)[0, 1]),
        "all_candidate_nested": results,
        "promoted_closest": promoted,
        "prev_closest": CLOSEST,
        "gate_0_71": bool(deliver_auc >= GATE),
        "gap_to_0_71": float(GATE - deliver_auc),
        "baseline_max3": B7_FLOOR,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "all_candidate_nested"}, indent=2), flush=True)
    print("TOP", sorted(results.items(), key=lambda kv: -kv[1])[:8], flush=True)
    print(f"GATE={'PASS' if metrics['gate_0_71'] else 'FAIL'} nested={deliver_auc:.8f} promoted={promoted}", flush=True)
    return 0 if metrics["gate_0_71"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
