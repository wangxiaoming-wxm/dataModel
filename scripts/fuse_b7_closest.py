"""Build B7 closest submission: max(gap, gap_bag, plus).

Default sources are the frozen artifacts committed in this repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
CLAIMED_CLOSEST = 0.7027049552615718


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--b6",
        type=Path,
        default=ROOT / "artifacts/b6_frozen/predictions.npz",
        help="npz with oof_gap/oof_gap_bag/test_gap/test_gap_bag/y",
    )
    ap.add_argument(
        "--plus-oof",
        type=Path,
        default=ROOT / "reference/v10/oof_plus_h2_10.npz",
    )
    ap.add_argument(
        "--plus-test",
        type=Path,
        default=ROOT / "reference/v10/test_plus_h2_10.npy",
    )
    ap.add_argument(
        "--sample",
        type=Path,
        default=ROOT / "data/submit_sample.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "submissions/submission_b7_closest_honest.csv",
    )
    args = ap.parse_args()

    b6 = np.load(args.b6)
    plus_npz = np.load(args.plus_oof)
    plus_oof = plus_npz["oof"]
    plus_test = np.load(args.plus_test)
    y = b6["y"]
    gap, gap_bag = b6["oof_gap"], b6["oof_gap_bag"]
    oof = np.maximum(np.maximum(gap, gap_bag), plus_oof)
    test = np.maximum(np.maximum(b6["test_gap"], b6["test_gap_bag"]), plus_test)
    auc = float(roc_auc_score(y, oof))
    print(f"closest_max3_oof_auc={auc:.12f} claimed={CLAIMED_CLOSEST:.12f} abs_err={abs(auc-CLAIMED_CLOSEST):.2e}")

    sample = pd.read_csv(args.sample)
    # sample may use label column for probabilities
    out = sample.copy()
    if "label" not in out.columns:
        raise SystemExit(f"submit sample missing label column: {list(out.columns)}")
    if len(out) != len(test):
        raise SystemExit(f"length mismatch sample={len(out)} test_pred={len(test)}")
    out["label"] = test
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
