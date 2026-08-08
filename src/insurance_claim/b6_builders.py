"""B6v2 builders: integrated main + orthogonal hetero (fold-local, no TE)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from insurance_claim.b6_gap_features import GAP_CAT_COLS, add_gap_cats, fit_gap_edges
from insurance_claim.feature_blocks import (
    DaysConditionFeatureBlock,
    DualCategoryFeatureBlock,
    RawFeatureBlock,
    StructuredStringFeatureBlock,
)
from insurance_claim.train_b5_focus import enrich, prepare_for_cat

DUAL_MAIN = [
    "region",
    "source",
    "x19_cat",
    "x20_cat",
    "age_coarse",
    "code",
    "w_pair",
    "version",
]

DUAL_HETERO = [
    "w_pair",
    "code",
    "t3_sfx",
    "age_coarse",
    "region",
    "source",
    "version",
    "grades",
]


def _attach_biz_tokens(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t3 = out["t3"].astype(str)
    parsed = t3.str.extract(r"^(-?\d+(?:\.\d+)?)([A-Za-z])$")
    out["t3_sfx"] = parsed[1].fillna("__NONE__").astype(str)
    w1 = pd.to_numeric(out.get("w1"), errors="coerce").fillna(-1).astype(int)
    w2 = pd.to_numeric(out.get("w2"), errors="coerce").fillna(-1).astype(int)
    out["w_pair"] = w1.astype(str) + "_" + w2.astype(str)
    age = pd.to_numeric(out.get("age_range"), errors="coerce")
    age_c = age.clip(upper=8).fillna(-1).astype(int).astype(str)
    out["age_coarse"] = age_c.where(age.notna(), "__NA__")
    out["code"] = out["code"].astype(str) if "code" in out.columns else "__NA__"
    out["grades"] = out["grades"].astype(str) if "grades" in out.columns else "__NA__"
    return out


def _merge_gap(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    te: pd.DataFrame,
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    edges = fit_gap_edges(X_tr)

    def gap_only(raw: pd.DataFrame) -> pd.DataFrame:
        return add_gap_cats(enrich(raw), edges).loc[:, list(GAP_CAT_COLS)].copy()

    gtr, gva, gte = gap_only(X_tr), gap_only(X_va), gap_only(X_te)

    def merge(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
        out = pd.concat([base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        return out.loc[:, ~out.columns.duplicated()]

    tr = merge(tr, gtr)
    va = merge(va, gva).reindex(columns=tr.columns)
    te = merge(te, gte).reindex(columns=tr.columns)
    return prepare_for_cat(tr, va, te)


def build_main(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """B5-like FE with dual/days centered on mining tokens + gap cats."""
    raw_tr, raw_va, raw_te = X_tr, X_va, X_te
    X_tr = _attach_biz_tokens(enrich(X_tr))
    X_va = _attach_biz_tokens(enrich(X_va))
    X_te = _attach_biz_tokens(enrich(X_te))
    parts_tr, parts_va, parts_te = [], [], []
    for block in [
        RawFeatureBlock(drop_near_id_latent=False),
        StructuredStringFeatureBlock(columns=["source", "t3", "region", "code"]),
        DaysConditionFeatureBlock(
            quantile_bins=(5, 10, 20),
            categorical_cross_columns=("region", "source", "x19_cat", "x20_cat", "age_coarse", "code"),
            categorical_cross_bins=(10,),
        ),
        DualCategoryFeatureBlock(
            columns=DUAL_MAIN, max_categories=128, cross_order=3, max_cross_columns=6
        ),
    ]:
        parts_tr.append(block.fit_transform(X_tr))
        parts_va.append(block.transform(X_va))
        parts_te.append(block.transform(X_te))
    tr = pd.concat(parts_tr, axis=1).loc[:, lambda d: ~d.columns.duplicated()]
    va = pd.concat(parts_va, axis=1).loc[:, lambda d: ~d.columns.duplicated()].reindex(columns=tr.columns)
    te = pd.concat(parts_te, axis=1).loc[:, lambda d: ~d.columns.duplicated()].reindex(columns=tr.columns)
    return _merge_gap(tr, va, te, raw_tr, raw_va, raw_te)


def build_hetero(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Orthogonal view: no x19/x20 dual; w_pair/code/t3_sfx first + gap cats."""
    raw_tr, raw_va, raw_te = X_tr, X_va, X_te
    X_tr = _attach_biz_tokens(enrich(X_tr))
    X_va = _attach_biz_tokens(enrich(X_va))
    X_te = _attach_biz_tokens(enrich(X_te))
    parts_tr, parts_va, parts_te = [], [], []
    for block in [
        RawFeatureBlock(drop_near_id_latent=True),
        StructuredStringFeatureBlock(columns=["source", "t3", "version", "code", "grades"]),
        DaysConditionFeatureBlock(
            quantile_bins=(5, 10, 20),
            categorical_cross_columns=("region", "source", "code", "w_pair", "t3_sfx"),
            categorical_cross_bins=(5, 10),
            include_single_axis_crosses=True,
        ),
        DualCategoryFeatureBlock(
            columns=DUAL_HETERO, max_categories=64, cross_order=2, max_cross_columns=6
        ),
    ]:
        parts_tr.append(block.fit_transform(X_tr))
        parts_va.append(block.transform(X_va))
        parts_te.append(block.transform(X_te))
    tr = pd.concat(parts_tr, axis=1).loc[:, lambda d: ~d.columns.duplicated()]
    va = pd.concat(parts_va, axis=1).loc[:, lambda d: ~d.columns.duplicated()].reindex(columns=tr.columns)
    te = pd.concat(parts_te, axis=1).loc[:, lambda d: ~d.columns.duplicated()].reindex(columns=tr.columns)
    return _merge_gap(tr, va, te, raw_tr, raw_va, raw_te)
