"""Features extracted from composite string columns."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .base import FeatureBlock


_NUMBER_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)")
_TOKEN_RE = re.compile(r"[^A-Za-z0-9]+")


class StructuredStringFeatureBlock(FeatureBlock):
    """Extract stable lexical and numeric descriptors from string columns."""

    name = "structured_string"
    version = "1.0"

    def __init__(self, columns: Iterable[str] | None = None) -> None:
        super().__init__()
        self.columns = tuple(columns) if columns is not None else None
        self.columns_: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train=None) -> "StructuredStringFeatureBlock":
        self._validate_frame(X_train)
        source = self._without_targets(X_train)
        requested = list(self.columns) if self.columns is not None else [
            column for column in source.columns
            if pd.api.types.is_object_dtype(source[column])
            or pd.api.types.is_string_dtype(source[column])
            or isinstance(source[column].dtype, pd.CategoricalDtype)
        ]
        self.columns_ = [column for column in requested if column in source.columns]
        self.input_columns_ = list(self.columns_)
        self._finalize(self._transform_source(source), fit_stage=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._ensure_fitted()
        self._validate_frame(X)
        return self._finalize(self._transform_source(self._without_targets(X)), fit_stage=False)

    def manifest(self) -> dict[str, object]:
        payload = super().manifest()
        payload.update({"columns": list(self.columns_)})
        return payload

    def _transform_source(self, source: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=source.index)
        for column in self.columns_:
            values = self._as_string(source[column]) if column in source else pd.Series("__MISSING__", index=source.index)
            missing = values.eq("__MISSING__")
            # Prefix/suffix split on common composite delimiters.  Keeping the
            # original lexical values out of this block avoids a hidden raw
            # feature and makes composition explicit.
            pieces = values.str.split(_TOKEN_RE, n=1, expand=True)
            prefix = pieces[0].replace("", "__MISSING__").fillna("__MISSING__")
            # ``Series.str.rsplit`` treats a compiled regex as a literal
            # pattern on some pandas versions, so extract the final token
            # explicitly for cross-version behaviour.
            suffix = values.str.extract(r"(?:^|[-_|:/\s])([^\-_|:/\s]+)$", expand=False).replace("", "__MISSING__").fillna("__MISSING__")
            number = pd.to_numeric(values.str.extract(_NUMBER_RE, expand=False), errors="coerce")
            output[f"{column}__prefix"] = prefix.astype(str)
            output[f"{column}__suffix"] = suffix.astype(str)
            output[f"{column}__number"] = number.astype(float)
            output[f"{column}__length"] = values.str.len().astype(float)
            output[f"{column}__digit_count"] = values.str.count(r"\d").astype(float)
            output[f"{column}__alpha_count"] = values.str.count(r"[A-Za-z]").astype(float)
            output[f"{column}__special_count"] = values.str.count(r"[^A-Za-z0-9]").astype(float)
            output[f"{column}__pattern"] = values.map(self._pattern)
            output[f"{column}__missing"] = missing.astype("int8")
        return output

    @staticmethod
    def _pattern(value: str) -> str:
        if value == "__MISSING__":
            return "MISSING"
        chars = []
        for char in value:
            chars.append("A" if char.isalpha() else "9" if char.isdigit() else "_")
        return "".join(chars) or "EMPTY"


StructuredStringBlock = StructuredStringFeatureBlock

__all__ = ["StructuredStringFeatureBlock", "StructuredStringBlock"]
