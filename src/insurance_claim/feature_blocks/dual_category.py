"""Dual lexical/numeric representations for categorical columns."""

from __future__ import annotations

from itertools import combinations
from collections.abc import Iterable

import pandas as pd

from .base import FeatureBlock


class DualCategoryFeatureBlock(FeatureBlock):
    """Expose both a safe string category and a fold-local integer code.

    The code vocabulary is fitted on the training fold.  Unknown validation or
    test categories always map to ``-1`` and never alter the fitted mapping.
    """

    name = "dual_category"
    version = "1.0"

    def __init__(self, columns: Iterable[str] | None = None, max_categories: int = 64, cross_order: int = 3, max_cross_columns: int = 5) -> None:
        super().__init__()
        self.columns = tuple(columns) if columns is not None else None
        self.max_categories = int(max_categories)
        if self.max_categories < 1:
            raise ValueError("max_categories must be positive")
        if cross_order not in {1, 2, 3, 4} or max_cross_columns < 1:
            raise ValueError("cross_order must be 1..4 and max_cross_columns positive")
        self.cross_order = int(cross_order)
        self.max_cross_columns = int(max_cross_columns)
        self.columns_: list[str] = []
        self.cross_columns_: list[str] = []
        self.vocabularies_: dict[str, dict[str, int]] = {}
        self.frequencies_: dict[str, dict[str, float]] = {}

    def fit(self, X_train: pd.DataFrame, y_train=None) -> "DualCategoryFeatureBlock":
        self._validate_frame(X_train)
        source = self._without_targets(X_train)
        if self.columns is None:
            selected = []
            for column in source.columns:
                if pd.api.types.is_object_dtype(source[column]) or pd.api.types.is_string_dtype(source[column]) or isinstance(source[column].dtype, pd.CategoricalDtype):
                    selected.append(column)
                elif source[column].nunique(dropna=True) <= self.max_categories:
                    selected.append(column)
        else:
            selected = list(self.columns)
        self.columns_ = [column for column in selected if column in source]
        self.cross_columns_ = self.columns_[: self.max_cross_columns]
        self.input_columns_ = list(self.columns_)
        for column in self.columns_:
            values = self._as_string(source[column])
            self.vocabularies_[column] = {value: index for index, value in enumerate(values.drop_duplicates())}
            self.frequencies_[column] = {str(key): float(value) for key, value in values.value_counts(normalize=True).items()}
        self._finalize(self._transform_source(source), fit_stage=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._ensure_fitted()
        self._validate_frame(X)
        return self._finalize(self._transform_source(self._without_targets(X)), fit_stage=False)

    def manifest(self) -> dict[str, object]:
        payload = super().manifest()
        payload.update({"columns": list(self.columns_), "max_categories": self.max_categories, "cross_order": self.cross_order, "cross_columns": self.cross_columns_, "vocabulary_sizes": {key: len(value) for key, value in self.vocabularies_.items()}})
        return payload

    def _transform_source(self, source: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=source.index)
        for column in self.columns_:
            values = self._as_string(source[column]) if column in source else pd.Series("__MISSING__", index=source.index)
            output[f"{column}__category"] = values
            output[f"{column}__category_code"] = values.map(self.vocabularies_.get(column, {})).fillna(-1).astype("int32")
            output[f"{column}__frequency"] = values.map(self.frequencies_.get(column, {})).fillna(0.0).astype(float)
        string_values = {
            column: self._as_string(source[column]) if column in source else pd.Series("__MISSING__", index=source.index)
            for column in self.cross_columns_
        }
        for order in range(2, min(self.cross_order, len(self.cross_columns_)) + 1):
            for columns in combinations(self.cross_columns_, order):
                name = "__X__".join(map(str, columns)) + "__category_cross"
                combined = string_values[columns[0]]
                for column in columns[1:]:
                    combined = combined + "|" + string_values[column]
                output[name] = combined.astype(str)
        return output


DualCategoryBlock = DualCategoryFeatureBlock

__all__ = ["DualCategoryFeatureBlock", "DualCategoryBlock"]
