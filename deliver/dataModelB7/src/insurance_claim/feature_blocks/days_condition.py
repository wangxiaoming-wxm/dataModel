"""Fold-local multi-scale features for the strongest numeric interaction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .base import FeatureBlock


class DaysConditionFeatureBlock(FeatureBlock):
    """Create leakage-safe ``days``/``condition`` transforms and interactions."""

    name = "days_condition"
    version = "1.0"

    def __init__(
        self,
        days_column: str = "days",
        condition_column: str = "condition",
        quantile_bins: int | Iterable[int] = (5, 10, 20),
        categorical_cross_columns: Iterable[str] | None = None,
        categorical_cross_bins: Iterable[int] | None = None,
        include_single_axis_crosses: bool = True,
    ) -> None:
        super().__init__()
        self.days_column = days_column
        self.condition_column = condition_column
        if isinstance(quantile_bins, (int, np.integer)):
            quantile_bins = (int(quantile_bins),)
        self.quantile_bins = tuple(sorted({int(value) for value in quantile_bins if int(value) >= 2}))
        if not self.quantile_bins:
            raise ValueError("quantile_bins must contain a positive bin count >= 2")
        self.categorical_cross_columns = tuple(categorical_cross_columns or ())
        requested_cross_bins = tuple(int(value) for value in (categorical_cross_bins or self.quantile_bins))
        self.categorical_cross_bins = tuple(sorted(set(requested_cross_bins))) if self.categorical_cross_columns else ()
        unknown_bins = set(self.categorical_cross_bins).difference(self.quantile_bins)
        if unknown_bins:
            raise ValueError(f"categorical_cross_bins must be present in quantile_bins: {sorted(unknown_bins)}")
        self.include_single_axis_crosses = bool(include_single_axis_crosses)
        self.medians_: dict[str, float] = {}
        self.edges_: dict[str, dict[int, np.ndarray]] = {"days": {}, "condition": {}}

    def fit(self, X_train: pd.DataFrame, y_train=None) -> "DaysConditionFeatureBlock":
        self._validate_frame(X_train)
        source = self._without_targets(X_train)
        requested_columns = (self.days_column, self.condition_column, *self.categorical_cross_columns)
        self.input_columns_ = [column for column in requested_columns if column in source]
        for logical, column in (("days", self.days_column), ("condition", self.condition_column)):
            values = pd.to_numeric(source[column], errors="coerce") if column in source else pd.Series(dtype=float)
            finite = values[np.isfinite(values)]
            self.medians_[logical] = float(finite.median()) if not finite.empty else 0.0
            for bins in self.quantile_bins:
                self.edges_[logical][bins] = self._quantile_edges(finite, bins)
        self._finalize(self._transform_source(source), fit_stage=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._ensure_fitted()
        self._validate_frame(X)
        return self._finalize(self._transform_source(self._without_targets(X)), fit_stage=False)

    def manifest(self) -> dict[str, object]:
        payload = super().manifest()
        payload.update({
            "days_column": self.days_column,
            "condition_column": self.condition_column,
            "quantile_bins": list(self.quantile_bins),
            "categorical_cross_columns": list(self.categorical_cross_columns),
            "categorical_cross_bins": list(self.categorical_cross_bins),
            "include_single_axis_crosses": self.include_single_axis_crosses,
            "medians": dict(self.medians_),
            "edges": {
                name: {str(bins): edges.tolist() for bins, edges in mapping.items()}
                for name, mapping in self.edges_.items()
            },
        })
        return payload

    def _transform_source(self, source: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=source.index)
        days = self._numeric(source, self.days_column)
        condition = self._numeric(source, self.condition_column)
        days_filled = days.fillna(self.medians_.get("days", 0.0))
        condition_filled = condition.fillna(self.medians_.get("condition", 0.0))
        output["days__filled"] = days_filled.astype(float)
        output["condition__filled"] = condition_filled.astype(float)
        output["days__log1p"] = np.log1p(days_filled.clip(lower=0)).astype(float)
        output["condition__log1p"] = np.log1p(condition_filled.clip(lower=0)).astype(float)
        output["days__missing"] = days.isna().astype("int8")
        output["condition__missing"] = condition.isna().astype("int8")
        output["days_condition__product"] = (days_filled * condition_filled).astype(float)
        output["days_condition__ratio"] = (condition_filled / (days_filled.abs() + 1.0)).astype(float)
        output["days_condition__missing"] = (days.isna() | condition.isna()).astype("int8")

        first_bins = self.quantile_bins[0]
        for logical, values in (("days", days), ("condition", condition)):
            for bins in self.quantile_bins:
                output[f"{logical}__bin_{bins}"] = self._apply_bins(values, self.edges_[logical][bins])
            # Compatibility aliases match the existing FeatureBuilder names.
            output[f"{logical}_bin"] = output[f"{logical}__bin_{first_bins}"]
        for bins in self.quantile_bins:
            output[f"days_condition__bin_{bins}"] = (
                output[f"days__bin_{bins}"].astype(str) + "__" + output[f"condition__bin_{bins}"].astype(str)
            )
        for column in self.categorical_cross_columns:
            category = self._as_string(source[column]) if column in source else pd.Series("__MISSING__", index=source.index)
            for bins in self.categorical_cross_bins:
                if self.include_single_axis_crosses:
                    output[f"days__bin_{bins}__X__{column}"] = output[f"days__bin_{bins}"].astype(str) + "|" + category
                    output[f"condition__bin_{bins}__X__{column}"] = output[f"condition__bin_{bins}"].astype(str) + "|" + category
                output[f"days_condition__bin_{bins}__X__{column}"] = output[f"days_condition__bin_{bins}"].astype(str) + "|" + category
        output["days_condition_bin"] = output[f"days_condition__bin_{first_bins}"]
        return output

    @staticmethod
    def _quantile_edges(values: pd.Series, bins: int) -> np.ndarray:
        if values.empty:
            return np.array([], dtype=float)
        edges = np.unique(values.quantile(np.linspace(0.0, 1.0, bins + 1)).to_numpy(dtype=float))
        return edges[1:-1] if len(edges) > 1 else np.array([], dtype=float)

    @staticmethod
    def _apply_bins(values: pd.Series, edges: np.ndarray) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        output = np.full(len(numeric), -1, dtype=np.int16)
        valid = np.isfinite(numeric)
        if edges.size:
            output[valid] = np.searchsorted(edges, numeric[valid], side="right").astype(np.int16)
        else:
            output[valid] = 0
        return pd.Series(output, index=values.index, dtype="int16").astype(str).radd("bin_")

    @staticmethod
    def _numeric(source: pd.DataFrame, column: str) -> pd.Series:
        if column not in source:
            return pd.Series(np.nan, index=source.index, dtype=float)
        return pd.to_numeric(source[column], errors="coerce")


DaysConditionBlock = DaysConditionFeatureBlock

__all__ = ["DaysConditionFeatureBlock", "DaysConditionBlock"]
