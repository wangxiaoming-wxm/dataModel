"""Raw feature block with stable columns and safe categorical missing values."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .base import FeatureBlock


class RawFeatureBlock(FeatureBlock):
    """Keep source columns while dropping linkage/target fields.

    Raw categorical values are represented as strings with an explicit missing
    token.  No category vocabulary is learned, so unseen validation categories
    are passed through safely and can be encoded by a model adapter later.
    """

    name = "raw"
    version = "1.0"

    def __init__(
        self,
        drop_columns: Iterable[str] | None = None,
        drop_near_id_latent: bool = False,
        near_id_latent_prefix: str = "x",
        keep_latent: Iterable[str] | None = ("x19", "x20"),
    ) -> None:
        super().__init__()
        self.drop_columns = tuple(drop_columns or ("id", "label", "target", "y"))
        self.drop_near_id_latent = bool(drop_near_id_latent)
        self.near_id_latent_prefix = near_id_latent_prefix
        self.keep_latent = tuple(keep_latent or ())
        self.categorical_columns_: list[str] = []

    def _drop_latent(self, source: pd.DataFrame) -> pd.DataFrame:
        if not self.drop_near_id_latent:
            return source
        drop = []
        for column in source.columns:
            name = str(column)
            if name in self.keep_latent:
                continue
            if name.startswith(self.near_id_latent_prefix) and name[1:].isdigit():
                # Drop near-row-unique anonymized latents x0..x18 by default.
                idx = int(name[1:])
                if idx <= 18:
                    drop.append(column)
        return source.drop(columns=drop, errors="ignore")

    def fit(self, X_train: pd.DataFrame, y_train=None) -> "RawFeatureBlock":
        self._validate_frame(X_train)
        source = X_train.drop(columns=list(self.drop_columns), errors="ignore").copy()
        source = self._drop_latent(source)
        self.input_columns_ = list(source.columns)
        transformed = self._transform_source(source)
        self.categorical_columns_ = [
            column for column in transformed.columns if not pd.api.types.is_numeric_dtype(transformed[column])
        ]
        self._finalize(transformed, fit_stage=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._ensure_fitted()
        self._validate_frame(X)
        source = X.drop(columns=list(self.drop_columns), errors="ignore").copy()
        source = self._drop_latent(source)
        result = self._transform_source(source)
        for column in self.feature_names_:
            if column not in result:
                result[column] = "__MISSING__" if column in self.categorical_columns_ else np.nan
        result = result.reindex(columns=self.feature_names_)
        for column in self.categorical_columns_:
            result[column] = self._as_string(result[column])
        return result

    def manifest(self) -> dict[str, object]:
        payload = super().manifest()
        payload.update({"drop_columns": list(self.drop_columns), "categorical_columns": list(self.categorical_columns_)})
        return payload

    def _transform_source(self, source: pd.DataFrame) -> pd.DataFrame:
        output = source.copy()
        for column in output.columns:
            if pd.api.types.is_object_dtype(output[column]) or pd.api.types.is_string_dtype(output[column]) or isinstance(output[column].dtype, pd.CategoricalDtype):
                output[column] = self._as_string(output[column])
        return output


RawBlock = RawFeatureBlock

__all__ = ["RawFeatureBlock", "RawBlock"]
