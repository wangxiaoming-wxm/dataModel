#!/usr/bin/env python3
"""Within-region days-isotonic residual ranker (anti-monotonic correction).

Business: claim rate rises with days overall, but wrong pairs are anti-monotonic
(low days / high condition yet claim). Fit fold-local isotonic P(y|days) globally,
then train a residual model on (y - p_iso) / ranking within region×days_bin to
capture local deviations — especially in f09d/9685 long.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

from insurance_claim.b6pro_fusion import apply_rule, nested_select_rule
from insurance_claim.b6pro_long_features import build_long_keepx
from insurance_claim.model import IDENTIFIER, TARGET
from insurance_claim.train_b6 import PARAMS_GAP_BAG

B7_FLOOR = 0.7027049552615718
GATE = 0.71
CLOSEST = float(json.load(open("artifacts/b6pro_long_best/metrics.json"))["nested_oof_auc"])
WEAK = frozenset({"f09d", "9685", "908d", "fafc", "f167", "ab86"})

PARAMS_CLS = {**PARAMS_GAP_BAG, "thread_count": 4, "iterations": 3000, "od_wait": 140}
PARAMS_REG = dict(
    loss_function="RMSE",
    iterations=2500,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=8,
    od_type="Iter",
    od_wait=120,
    random_strength=1.0,
    thread_count=4,
    verbose=False,
    allow_writing_files=False,
)


def write_submission(sample, pred, path: Path) -> None:
    out = sample.copy()
    lab = [c for c in out.columns if c != IDENTIFIER][0]
    out[lab] = np.asarray(pred, float)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def fit_iso_days(days_tr, y_tr, days_va, days_te):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    iso.fit(days_tr, y_tr)
    return iso.predict(days_va), iso.predict(days_te), iso.predict(days_tr)


def numeric_matrix(X_tr, X_va, X_te):
    """Ordinal-encode cats + keep numerics + engineered anti-mono signals."""
    frames = [X_tr.copy(), X_va.copy(), X_te.copy()]
    for fr in frames:
        fr["log_days"] = np.log1p(pd.to_numeric(fr["days"], errors="coerce").clip(lower=0))
        cond = pd.to_numeric(fr["condition"], errors="coerce")
        days = pd.to_numeric(fr["days"], errors="coerce")
        fr["cond"] = cond
        fr["ratio"] = cond / (days.abs() + 1.0)
        fr["days_x_invcond"] = days / (cond.abs() + 1.0)
        # parse
        if "source" in fr.columns:
            fr["car"] = fr["source"].astype(str).str.extract(r"(CAR_\d+)", expand=False).fillna("__NA__")
        if "t3" in fr.columns:
            fr["t3_sfx"] = fr["t3"].astype(str).str.extract(r"([A-Za-z])$", expand=False).fillna("__NONE__")
    num_cols = [
        c
        for c in frames[0].columns
        if pd.api.types.is_numeric_dtype(frames[0][c]) and c not in (IDENTIFIER,)
    ]
    cat_cols = [c for c in frames[0].columns if c not in num_cols and c != IDENTIFIER]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr_cat = enc.fit_transform(frames[0][cat_cols].astype(str))
    va_cat = enc.transform(frames[1][cat_cols].astype(str))
    te_cat = enc.transform(frames[2][cat_cols].astype(str))
    def pack(fr, cat):
        num = fr[num_cols].apply(pd.to_numeric, errors="coerce")
        med = num.median()
        num = num.fillna(med)
        return np.hstack([num.to_numpy(dtype=float), cat])
    return pack(frames[0], tr_cat), pack(frames[1], va_cat), pack(frames[2], te_cat)


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
    cur = np.load("artifacts/b6pro_long_best/predictions.npz")

    seeds = [2026, 2027, 2028, 2029]
    oof_cls = np.zeros(len(y))
    te_cls = np.zeros(len(test))
    oof_res = np.zeros(len(y))
    te_res = np.zeros(len(test))
    oof_blend = np.zeros(len(y))
    te_blend = np.zeros(len(test))

    for seed in seeds:
        oof_c = np.zeros(len(y))
        oof_r = np.zeros(len(y))
        pte_c = np.zeros(len(test))
        pte_r = np.zeros(len(test))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(features, y)):
            Xtr = features.iloc[tr].reset_index(drop=True)
            Xva = features.iloc[va].reset_index(drop=True)
            ytr = y.iloc[tr].to_numpy()
            # iso baseline
            pva_iso, pte_iso, ptr_iso = fit_iso_days(
                Xtr["days"].to_numpy(float), ytr, Xva["days"].to_numpy(float), test["days"].to_numpy(float)
            )
            # classification keepx with residual features
            trd, vad, ted, cats = build_long_keepx(Xtr, Xva, test.copy())
            for df, piso in ((trd, ptr_iso), (vad, pva_iso), (ted, pte_iso)):
                df["iso_days"] = piso
                df["resid_target_proxy"] = 0.0  # placeholder numeric
            # weight anti-mono: within long, upweight if condition high but days low relative to region
            w = np.ones(len(Xtr))
            dtr = Xtr["days"].to_numpy(float)
            ctr = pd.to_numeric(Xtr["condition"], errors="coerce").fillna(50).to_numpy()
            rtr = Xtr["region"].astype(str).to_numpy()
            long_tr = dtr >= 3000
            # high condition = better car? business says LOW condition = worse. So anti-mono claim: low days + low condition
            # upweight weak regions
            w[np.isin(rtr, list(WEAK)) & long_tr] *= 4.0
            # upweight low-days-within-long (3000-5000) where anti-mono denser
            w[(dtr >= 3000) & (dtr < 5000)] *= 1.5

            model = CatBoostClassifier(**{**PARAMS_CLS, "random_seed": seed + fold})
            model.fit(trd, y.iloc[tr], sample_weight=w, eval_set=(vad, y.iloc[va]), cat_features=cats, use_best_model=True)
            oof_c[va] = model.predict_proba(vad)[:, 1]
            pte_c += model.predict_proba(ted)[:, 1] / 5.0

            # residual regressor on y - iso
            Xtrn, Xvan, Xten = numeric_matrix(Xtr, Xva, test.copy())
            # append iso
            Xtrn = np.column_stack([Xtrn, ptr_iso])
            Xvan = np.column_stack([Xvan, pva_iso])
            Xten = np.column_stack([Xten, pte_iso])
            resid_tr = ytr.astype(float) - ptr_iso
            reg = CatBoostRegressor(**{**PARAMS_REG, "random_seed": seed + fold})
            reg.fit(Xtrn, resid_tr, eval_set=(Xvan, y.iloc[va].to_numpy(float) - pva_iso), use_best_model=True)
            rva = reg.predict(Xvan)
            rte = reg.predict(Xten)
            # convert residual to probability-like: iso + resid, clip
            oof_r[va] = np.clip(pva_iso + rva, 1e-4, 1 - 1e-4)
            pte_r += np.clip(pte_iso + rte, 1e-4, 1 - 1e-4) / 5.0
            print(
                f"s{seed} f{fold} cls={roc_auc_score(y.iloc[va], oof_c[va]):.5f} "
                f"res={roc_auc_score(y.iloc[va], oof_r[va]):.5f}",
                flush=True,
            )
        print(
            f"s{seed} OOF cls={roc_auc_score(y, oof_c):.6f} res={roc_auc_score(y, oof_r):.6f} "
            f"f09d_cls={roc_auc_score(y.to_numpy()[(region=='f09d')&long], oof_c[(region=='f09d')&long]):.5f}",
            flush=True,
        )
        oof_cls += oof_c
        te_cls += pte_c
        oof_res += oof_r
        te_res += pte_r
        oof_blend += 0.5 * (oof_c + oof_r)
        te_blend += 0.5 * (pte_c + pte_r)

    n = len(seeds)
    oof_cls /= n
    te_cls /= n
    oof_res /= n
    te_res /= n
    oof_blend /= n
    te_blend /= n

    variants = {
        "cls": (oof_cls, te_cls),
        "res": (oof_res, te_res),
        "blend": (oof_blend, te_blend),
        "mean_cur_cls": (0.5 * (cur["oof"] + oof_cls), 0.5 * (cur["test"] + te_cls)),
        "mean_cur_res": (0.5 * (cur["oof"] + oof_res), 0.5 * (cur["test"] + te_res)),
        "mean_cur_blend": (0.5 * (cur["oof"] + oof_blend), 0.5 * (cur["test"] + te_blend)),
    }
    # patch weak long with cls
    for alpha, tag in [(1.0, "patch1"), (0.5, "patch05"), (0.3, "patch03")]:
        arm = cur["oof"].copy()
        tarm = cur["test"].copy()
        m = np.isin(region, list(WEAK)) & long
        m_te = np.isin(test["region"].astype(str), list(WEAK)) & (test["days"].to_numpy(float) >= 3000)
        arm[m] = alpha * oof_cls[m] + (1 - alpha) * cur["oof"][m]
        tarm[m_te] = alpha * te_cls[m_te] + (1 - alpha) * cur["test"][m_te]
        variants[f"{tag}_weak_cls"] = (arm, tarm)
        arm2 = cur["oof"].copy()
        tarm2 = cur["test"].copy()
        m2 = (region == "f09d") & long
        m2_te = (test["region"].astype(str) == "f09d") & (test["days"].to_numpy(float) >= 3000)
        arm2[m2] = alpha * oof_cls[m2] + (1 - alpha) * cur["oof"][m2]
        tarm2[m2_te] = alpha * te_cls[m2_te] + (1 - alpha) * cur["test"][m2_te]
        variants[f"{tag}_f09d_cls"] = (arm2, tarm2)

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
        print(f"{name}: direct={direct:.8f} f09d={roc_auc_score(y.to_numpy()[f09], oa[f09]):.5f}", flush=True)

    deliver = best_res["nested_oof_auc"]
    deliver_oof = best_res["nested_oof"]
    deliver_test = apply_rule(best_res["selected_rule"], best_pair[1]) if len(best_pair[1]) > 1 else best_pair[1][0]
    if deliver + 1e-12 < B7_FLOOR:
        best_name = "b7_fallback"
        deliver = float(roc_auc_score(y, max3))
        deliver_oof, deliver_test = max3, tmax

    promoted = deliver > CLOSEST + 1e-12
    out = Path("artifacts/b6pro_iso_resid")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "predictions.npz",
        y=y.to_numpy(),
        oof=deliver_oof,
        test=deliver_test,
        oof_cls=oof_cls,
        te_cls=te_cls,
        oof_res=oof_res,
        te_res=te_res,
    )
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
                    "source": "b6pro_iso_resid",
                },
                indent=2,
            )
        )

    metrics = {
        "best": best_name,
        "nested": deliver,
        "cls": float(roc_auc_score(y, oof_cls)),
        "res": float(roc_auc_score(y, oof_res)),
        "promoted": promoted,
        "gate": deliver >= GATE,
        "closest_prev": CLOSEST,
        "top": sorted(results.items(), key=lambda kv: -kv[1])[:12],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "top"}, indent=2), flush=True)
    print("TOP", metrics["top"], flush=True)
    print(f"GATE={'PASS' if deliver >= GATE else 'FAIL'} nested={deliver:.8f} promoted={promoted}", flush=True)
    return 0 if deliver >= GATE else 2


if __name__ == "__main__":
    raise SystemExit(main())
