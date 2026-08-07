"""Domain-aware parsing for insurance claim composite fields.

Extracts vehicle / policy / geography / time semantics from structured
strings such as ``t3="5.2P"``, ``source="CAR_0|ENG_709"``, ``version="v9"``.
All transforms are label-free and therefore fold-safe.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .base import FeatureBlock


_T3_RE = re.compile(r"^([-+]?\d+(?:\.\d+)?)([A-Za-z]*)$")
_SOURCE_RE = re.compile(r"^([A-Za-z]+)_?(\d+)\|([A-Za-z]+)_?(\d+)$")
_VERSION_RE = re.compile(r"^v?(\d+)$", re.IGNORECASE)
_MONTH_RE = re.compile(r"^M?(\d+)$", re.IGNORECASE)
_GRADES_ORD = {"s": 1, "ss": 2, "sss": 3}


class DomainParseFeatureBlock(FeatureBlock):
    """Parse composite columns into numeric magnitudes and semantic tokens."""

    name = "domain_parse"
    version = "1.0"

    def __init__(
        self,
        t3_column: str = "t3",
        source_column: str = "source",
        version_column: str = "version",
        grades_column: str = "grades",
        month_column: str = "month",
        t3_quantile_bins: int = 10,
    ) -> None:
        super().__init__()
        self.t3_column = t3_column
        self.source_column = source_column
        self.version_column = version_column
        self.grades_column = grades_column
        self.month_column = month_column
        self.t3_quantile_bins = int(t3_quantile_bins)
        if self.t3_quantile_bins < 2:
            raise ValueError("t3_quantile_bins must be >= 2")
        self.t3_edges_: np.ndarray = np.array([], dtype=float)
        self.t3_median_: float = 0.0

    def fit(self, X_train: pd.DataFrame, y_train=None) -> "DomainParseFeatureBlock":
        self._validate_frame(X_train)
        source = self._without_targets(X_train)
        self.input_columns_ = [
            column
            for column in (
                self.t3_column,
                self.source_column,
                self.version_column,
                self.grades_column,
                self.month_column,
                "w1",
                "w2",
                "condition",
                "code",
                "region",
            )
            if column in source
        ]
        t3_num = self._parse_t3_num(
            source.get(self.t3_column, pd.Series(index=source.index, dtype="object"))
        )
        finite = t3_num[np.isfinite(t3_num.to_numpy(dtype=float))]
        self.t3_median_ = float(finite.median()) if not finite.empty else 0.0
        self.t3_edges_ = self._quantile_edges(finite, self.t3_quantile_bins)
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
                "t3_column": self.t3_column,
                "source_column": self.source_column,
                "version_column": self.version_column,
                "grades_column": self.grades_column,
                "month_column": self.month_column,
                "t3_quantile_bins": self.t3_quantile_bins,
                "t3_median": self.t3_median_,
                "t3_edges": self.t3_edges_.tolist(),
            }
        )
        return payload

    def _transform_source(self, source: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=source.index)

        t3_raw = (
            self._as_string(source[self.t3_column])
            if self.t3_column in source
            else pd.Series("__MISSING__", index=source.index)
        )
        t3_num = self._parse_t3_num(t3_raw)
        t3_sfx = t3_raw.map(self._t3_suffix)
        t3_filled = t3_num.fillna(self.t3_median_)
        output["t3_num"] = t3_filled.astype(float)
        output["t3_sfx"] = t3_sfx.astype(str)
        output["t3_bin"] = self._apply_bins(t3_num, self.t3_edges_)
        output["t3_key"] = output["t3_bin"].astype(str) + "|" + output["t3_sfx"].astype(str)

        source_raw = (
            self._as_string(source[self.source_column])
            if self.source_column in source
            else pd.Series("__MISSING__", index=source.index)
        )
        parsed_source = source_raw.map(self._parse_source)
        output["car_prefix"] = parsed_source.map(lambda row: row[0]).astype(str)
        output["car_id"] = parsed_source.map(lambda row: row[1]).astype(str)
        output["eng_prefix"] = parsed_source.map(lambda row: row[2]).astype(str)
        output["eng_id"] = parsed_source.map(lambda row: row[3]).astype(str)
        output["car_n"] = pd.to_numeric(output["car_id"], errors="coerce").fillna(-1).astype(float)
        output["eng_n"] = pd.to_numeric(output["eng_id"], errors="coerce").fillna(-1).astype(float)
        output["car_token"] = (
            output["car_prefix"].astype(str) + "_" + output["car_id"].astype(str)
        )

        version_raw = (
            self._as_string(source[self.version_column])
            if self.version_column in source
            else pd.Series("__MISSING__", index=source.index)
        )
        ver_n = version_raw.map(self._parse_version_num)
        output["ver_n"] = ver_n.astype(float)
        output["ver_era"] = ver_n.map(self._version_era).astype(str)

        grades_raw = (
            self._as_string(source[self.grades_column])
            if self.grades_column in source
            else pd.Series("__MISSING__", index=source.index)
        )
        output["grades_ord"] = grades_raw.str.lower().map(_GRADES_ORD).fillna(0).astype(float)
        output["grades_token"] = grades_raw.str.lower().astype(str)

        month_raw = (
            self._as_string(source[self.month_column])
            if self.month_column in source
            else pd.Series("__MISSING__", index=source.index)
        )
        output["month_n"] = month_raw.map(self._parse_month_num).astype(float)

        if "condition" in source:
            output["condition_missing"] = (
                pd.to_numeric(source["condition"], errors="coerce").isna().astype("int8")
            )
        else:
            output["condition_missing"] = np.int8(0)

        if "w1" in source and "w2" in source:
            w1 = pd.to_numeric(source["w1"], errors="coerce")
            w2 = pd.to_numeric(source["w2"], errors="coerce")
            output["w_conflict"] = (w1 == w2).fillna(False).astype("int8")
            output["w_sum"] = (w1.fillna(0) + w2.fillna(0)).astype(float)
            output["w_xor"] = (w1.fillna(-1) != w2.fillna(-1)).astype("int8")
        else:
            output["w_conflict"] = np.int8(0)
            output["w_sum"] = 0.0
            output["w_xor"] = np.int8(0)

        code = (
            self._as_string(source["code"])
            if "code" in source
            else pd.Series("__MISSING__", index=source.index)
        )
        region = (
            self._as_string(source["region"])
            if "region" in source
            else pd.Series("__MISSING__", index=source.index)
        )
        output["car_code_key"] = output["car_id"].astype(str) + "|" + code
        output["t3sfx_code_key"] = output["t3_sfx"].astype(str) + "|" + code
        output["ver_era_region_key"] = output["ver_era"].astype(str) + "|" + region
        output["car_ver_key"] = output["car_token"].astype(str) + "|" + version_raw.astype(str)
        output["code_grades_key"] = code.astype(str) + "|" + output["grades_token"].astype(str)
        output["t3sfx_car_key"] = output["t3_sfx"].astype(str) + "|" + output["car_token"].astype(str)
        return output

    @staticmethod
    def _parse_t3_num(values: pd.Series) -> pd.Series:
        extracted = values.astype(str).str.extract(_T3_RE, expand=True)[0]
        return pd.to_numeric(extracted, errors="coerce")

    @staticmethod
    def _t3_suffix(value: str) -> str:
        match = _T3_RE.match(str(value))
        if not match:
            return "__NONE__"
        suffix = match.group(2)
        return suffix if suffix else "__NONE__"

    @staticmethod
    def _parse_source(value: str) -> tuple[str, str, str, str]:
        match = _SOURCE_RE.match(str(value))
        if not match:
            return ("__NONE__", "__MISSING__", "__NONE__", "__MISSING__")
        return match.group(1), match.group(2), match.group(3), match.group(4)

    @staticmethod
    def _parse_version_num(value: str) -> float:
        match = _VERSION_RE.match(str(value).strip())
        return float(match.group(1)) if match else np.nan

    @staticmethod
    def _version_era(value: float) -> str:
        if not np.isfinite(value):
            return "__MISSING__"
        if value <= 4:
            return "early"
        if value <= 10:
            return "mid"
        return "late"

    @staticmethod
    def _parse_month_num(value: str) -> float:
        match = _MONTH_RE.match(str(value).strip())
        return float(match.group(1)) if match else np.nan

    @staticmethod
    def _quantile_edges(values: pd.Series, bins: int) -> np.ndarray:
        if values.empty:
            return np.array([], dtype=float)
        edges = np.unique(
            values.quantile(np.linspace(0.0, 1.0, bins + 1)).to_numpy(dtype=float)
        )
        return edges[1:-1] if len(edges) > 1 else np.array([], dtype=float)

    @staticmethod
    def _apply_bins(values: pd.Series, edges: np.ndarray) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        output = np.full(len(numeric), -1, dtype=np.int16)
        valid = np.isfinite(numeric)
        if edges.size:
            output[valid] = np.searchsorted(edges, numeric[valid], side="right").astype(
                np.int16
            )
        else:
            output[valid] = 0
        return pd.Series(output, index=values.index, dtype="int16").astype(str).radd("bin_")


DomainParseBlock = DomainParseFeatureBlock

__all__ = ["DomainParseFeatureBlock", "DomainParseBlock"]
