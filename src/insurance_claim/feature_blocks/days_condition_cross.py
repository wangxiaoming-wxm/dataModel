"""Days×condition risk-surface as categorical crosses (no claim-rate TE).

Fold-local quantile edges only; emit CatBoost-native string categories.
No high-cardinality pair-TE.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .base import FeatureBlock

_SOURCE_RE = re.compile(r"^([A-Za-z]+)_?(\d+)\|([A-Za-z]+)_?(\d+)$")


class DaysConditionCrossFeatureBlock(FeatureBlock):
    name = "days_condition_cross"
    version = "1.1"

    def __init__(
        self,
        days_bins: int = 5,
        condition_bins: int = 5,
        days_bins_hi: int = 10,
        with_region: bool = True,
        with_source_car: bool = True,
        with_version: bool = True,
        with_t3_sfx: bool = True,
        with_code: bool = True,
    ) -> None:
        super().__init__()
        self.days_bins = int(days_bins)
        self.condition_bins = int(condition_bins)
        self.days_bins_hi = int(days_bins_hi)
        self.with_region = bool(with_region)
        self.with_source_car = bool(with_source_car)
        self.with_version = bool(with_version)
        self.with_t3_sfx = bool(with_t3_sfx)
        self.with_code = bool(with_code)
        self.days_edges_: np.ndarray = np.array([-np.inf, np.inf])
        self.days_edges_hi_: np.ndarray = np.array([-np.inf, np.inf])
        self.cond_edges_: np.ndarray = np.array([-np.inf, np.inf])

    def fit(self, X_train: pd.DataFrame, y_train=None) -> "DaysConditionCrossFeatureBlock":
        self._validate_frame(X_train)
        source = self._without_targets(X_train)
        self.input_columns_ = [
            c
            for c in ("days", "condition", "region", "source", "version", "t3", "code", "t3_sfx", "car_token")
            if c in source
        ]
        days = pd.to_numeric(source.get("days", pd.Series(dtype=float)), errors="coerce")
        cond = pd.to_numeric(source.get("condition", pd.Series(dtype=float)), errors="coerce")
        self.days_edges_ = self._edges(days, self.days_bins)
        self.days_edges_hi_ = self._edges(days, self.days_bins_hi)
        self.cond_edges_ = self._edges(cond, self.condition_bins)
        self._finalize(self._transform_source(source), fit_stage=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._ensure_fitted()
        self._validate_frame(X)
        return self._finalize(self._transform_source(self._without_targets(X)), fit_stage=False)

    def manifest(self) -> dict[str, object]:
        payload = super().manifest()
        payload.update(
            {
                "days_bins": self.days_bins,
                "condition_bins": self.condition_bins,
                "days_bins_hi": self.days_bins_hi,
                "with_region": self.with_region,
                "with_source_car": self.with_source_car,
                "with_version": self.with_version,
                "with_t3_sfx": self.with_t3_sfx,
                "with_code": self.with_code,
                "days_edges": self.days_edges_.tolist(),
                "days_edges_hi": self.days_edges_hi_.tolist(),
                "cond_edges": self.cond_edges_.tolist(),
            }
        )
        return payload

    @staticmethod
    def _edges(values: pd.Series, bins: int) -> np.ndarray:
        finite = values[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))]
        if finite.empty:
            return np.asarray([-np.inf, np.inf])
        edges = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return np.asarray([-np.inf, np.inf])
        edges = edges.astype(float)
        edges[0], edges[-1] = -np.inf, np.inf
        return edges

    @staticmethod
    def _bin(values: pd.Series, edges: np.ndarray, prefix: str) -> pd.Series:
        codes = pd.cut(values, edges, labels=False, include_lowest=True).fillna(-1).astype(int)
        return codes.astype(str).radd(f"{prefix}_")

    @staticmethod
    def _parse_car(source_raw: pd.Series) -> pd.Series:
        def _one(value: object) -> str:
            text = (
                "__MISSING__"
                if value is None or (isinstance(value, float) and np.isnan(value))
                else str(value)
            )
            match = _SOURCE_RE.match(text.strip())
            if not match:
                return "__NA__"
            return f"{match.group(1)}_{match.group(2)}"

        return source_raw.map(_one).astype(str)

    @staticmethod
    def _t3_sfx(values: pd.Series) -> pd.Series:
        def _one(value: object) -> str:
            text = str(value)
            match = re.match(r"^[-+]?\d+(?:\.\d+)?([A-Za-z]*)$", text)
            if not match:
                return "__NONE__"
            return match.group(1) or "__NONE__"

        return values.astype(str).map(_one)

    def _transform_source(self, source: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=source.index)
        days = pd.to_numeric(
            source.get("days", pd.Series(index=source.index, dtype=float)), errors="coerce"
        )
        cond = pd.to_numeric(
            source.get("condition", pd.Series(index=source.index, dtype=float)),
            errors="coerce",
        )
        db5 = self._bin(days, self.days_edges_, "d5")
        db10 = self._bin(days, self.days_edges_hi_, "d10")
        cb5 = self._bin(cond, self.cond_edges_, "c5")

        out["days_q5"] = db5.astype(str)
        out["days_q10"] = db10.astype(str)
        out["cond_q5"] = cb5.astype(str)
        out["days_q5__X__cond_q5__category_cross"] = (db5 + "|" + cb5).astype(str)

        # Fixed exposure windows (business-ish breakpoints around density modes).
        out["days_win"] = (
            pd.cut(
                days,
                bins=[-np.inf, 400, 700, 860, 2000, 7600, 9100, np.inf],
                labels=["w0", "w1", "w2", "w3", "w4", "w5", "w6"],
            )
            .astype(str)
            .fillna("__NA__")
        )
        out["days_win__X__cond_q5__category_cross"] = (
            out["days_win"] + "|" + cb5
        ).astype(str)

        region = None
        if self.with_region and "region" in source:
            region = source["region"].astype(str).fillna("__NA__")
            out["days_q5__X__region__category_cross"] = (db5 + "|" + region).astype(str)
            out["days_win__X__region__category_cross"] = (
                out["days_win"] + "|" + region
            ).astype(str)
            out["days_q10__X__region__category_cross"] = (db10 + "|" + region).astype(str)

        car = None
        if "car_token" in source:
            car = source["car_token"].astype(str).fillna("__NA__")
        elif self.with_source_car and "source" in source:
            car = self._parse_car(source["source"])
        if car is not None:
            out["src_car"] = car
            out["days_q5__X__src_car__category_cross"] = (db5 + "|" + car).astype(str)
            out["days_win__X__src_car__category_cross"] = (
                out["days_win"] + "|" + car
            ).astype(str)

        version = None
        if self.with_version and "version" in source:
            version = source["version"].astype(str).fillna("__NA__")
            out["days_q5__X__version__category_cross"] = (db5 + "|" + version).astype(str)
            out["days_win__X__version__category_cross"] = (
                out["days_win"] + "|" + version
            ).astype(str)

        t3_sfx = None
        if self.with_t3_sfx:
            if "t3_sfx" in source:
                t3_sfx = source["t3_sfx"].astype(str).fillna("__NONE__")
            elif "t3" in source:
                t3_sfx = self._t3_sfx(source["t3"])
            if t3_sfx is not None:
                out["days_q5__X__t3_sfx__category_cross"] = (db5 + "|" + t3_sfx).astype(str)

        if self.with_code and "code" in source:
            code = source["code"].astype(str).fillna("__NA__")
            out["days_q5__X__code__category_cross"] = (db5 + "|" + code).astype(str)

        if region is not None and version is not None:
            out["days_q10__X__region__X__version__category_cross"] = (
                db10 + "|" + region + "|" + version
            ).astype(str)
        if region is not None and car is not None:
            out["days_q10__X__region__X__car__category_cross"] = (
                db10 + "|" + region + "|" + car
            ).astype(str)
        if region is not None:
            out["days_q10__X__region__X__cond_q5__category_cross"] = (
                db10 + "|" + region + "|" + cb5
            ).astype(str)
        if car is not None and version is not None:
            out["days_q5__X__car__X__version__category_cross"] = (
                db5 + "|" + car + "|" + version
            ).astype(str)
        if car is not None and t3_sfx is not None:
            out["days_q5__X__car__X__t3_sfx__category_cross"] = (
                db5 + "|" + car + "|" + t3_sfx
            ).astype(str)
        return out


__all__ = ["DaysConditionCrossFeatureBlock"]
