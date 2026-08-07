"""Contracts shared by leakage-aware feature blocks.

Feature blocks deliberately use a small sklearn-like API.  A block owns only
the statistics needed for its own output and freezes its output schema at
``fit`` time.  This makes it safe to fit one instance per cross-validation
fold and prevents validation/test rows from changing the feature columns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


TARGET_COLUMNS = frozenset({"label", "target", "y"})


class FeatureBlock(ABC):
    """Base interface for a single composable feature block.

    Subclasses implement ``fit`` and ``transform`` and should call
    :meth:`_validate_frame` before reading input columns.  ``feature_names_``
    is intentionally populated during fit, so a transform cannot silently
    introduce columns that the model did not see in its training fold.
    """

    name = "feature_block"
    version = "1.0"
    uses_target = False
    uses_test_features = False
    requires_fit = True

    def __init__(self) -> None:
        self._fitted = False
        self.feature_names_: list[str] = []
        self.input_columns_: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: Any = None) -> "FeatureBlock":
        """Learn fold-local state from training rows and return ``self``."""

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform rows using state learned by :meth:`fit`."""

    def fit_transform(self, X_train: pd.DataFrame, y_train: Any = None) -> pd.DataFrame:
        """Fit on a training fold and transform that fold."""

        return self.fit(X_train, y_train=y_train).transform(X_train)

    def fit_with_reference(
        self,
        X_train: pd.DataFrame,
        X_reference: pd.DataFrame,
        y_train: Any = None,
    ) -> "FeatureBlock":
        """Fit a block that explicitly consumes an unlabeled reference set."""

        del X_train, X_reference, y_train
        raise RuntimeError(f"{self.__class__.__name__} does not support reference-data fitting")

    def get_feature_names(self) -> list[str]:
        """Return a copy of the frozen output schema."""

        self._ensure_fitted()
        return list(self.feature_names_)

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serialisable audit description of this block."""

        return {
            "name": self.name,
            "version": self.version,
            "fitted": self.is_fitted,
            "uses_target": bool(self.uses_target),
            "uses_test_features": bool(self.uses_test_features),
            "requires_fit": bool(self.requires_fit),
            "input_columns": list(self.input_columns_),
            "feature_names": list(self.feature_names_),
        }

    def _ensure_fitted(self) -> None:
        if self.requires_fit and not self._fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before transform or inspection")

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"X must be a pandas.DataFrame, got {type(frame)!r}")
        if frame.columns.duplicated().any():
            raise ValueError("X contains duplicate column names")

    @staticmethod
    def _without_targets(frame: pd.DataFrame) -> pd.DataFrame:
        # ``id`` is a linkage field rather than a model feature.  Keeping this
        # rule in the common helper prevents target-encoding/dual-category
        # auto-selection from accidentally learning an identifier vocabulary.
        drop = [column for column in frame.columns if str(column).lower() in TARGET_COLUMNS or str(column).lower() == "id"]
        return frame.drop(columns=drop).copy()

    @staticmethod
    def _as_string(values: pd.Series) -> pd.Series:
        return values.astype("string").fillna("__MISSING__").astype(str)

    def _finalize(self, output: pd.DataFrame, *, fit_stage: bool) -> pd.DataFrame:
        """Freeze or apply the output schema and normalise category strings."""

        if fit_stage:
            self.feature_names_ = list(output.columns)
            self._fitted = True
            return output.reindex(columns=self.feature_names_)

        self._ensure_fitted()
        result = output.copy()
        for column in self.feature_names_:
            if column not in result:
                result[column] = "__MISSING__" if column.endswith(("__category", "__prefix", "__suffix", "__pattern", "__bin")) else 0.0
        return result.reindex(columns=self.feature_names_)


__all__ = ["FeatureBlock", "TARGET_COLUMNS"]
