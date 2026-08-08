"""Vehicle physics ratios and exposure residuals (label-free, fold-local)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureBlock


class NumericPhysicsFeatureBlock(FeatureBlock):
    """cc / V / max_g ratios plus OLS residuals of days/condition."""

    name = "numeric_physics"
    version = "1.0"

    def __init__(
        self,
        keep_latent: tuple[str, ...] = ("x19", "x20"),
        residual_latent: tuple[str, ...] = ("x0", "x1", "x5", "x14"),
        drop_raw_latent: bool = True,
    ) -> None:
        super().__init__()
        self.keep_latent = tuple(keep_latent)
        self.residual_latent = tuple(residual_latent)
        self.drop_raw_latent = bool(drop_raw_latent)
        self.medians_: dict[str, float] = {}
        self.days_coef_: np.ndarray = np.zeros(3)
        self.days_intercept_: float = 0.0
        self.cond_coef_: np.ndarray = np.zeros(3)
        self.cond_intercept_: float = 0.0
        self.latent_coef_: dict[str, np.ndarray] = {}
        self.latent_intercept_: dict[str, float] = {}
        self.x20_cond_edges_: np.ndarray = np.array([], dtype=float)
        self.x20_cond_means_: dict[int, float] = {}
        self.x20_global_mean_: float = 0.0

    def fit(self, X_train: pd.DataFrame, y_train=None) -> "NumericPhysicsFeatureBlock":
        self._validate_frame(X_train)
        source = self._without_targets(X_train)
        cols = [
            "days",
            "condition",
            "cc",
            "V",
            "max_g",
            *self.keep_latent,
            *self.residual_latent,
        ]
        self.input_columns_ = [c for c in cols if c in source]
        for column in ("days", "condition", "cc", "V", "max_g", "x19", "x20"):
            if column in source:
                values = pd.to_numeric(source[column], errors="coerce")
                finite = values[np.isfinite(values)]
                self.medians_[column] = float(finite.median()) if not finite.empty else 0.0
            else:
                self.medians_[column] = 0.0

        design = self._design(source)
        days = pd.to_numeric(source.get("days", pd.Series(dtype=float)), errors="coerce").fillna(
            self.medians_["days"]
        )
        cond = pd.to_numeric(
            source.get("condition", pd.Series(dtype=float)), errors="coerce"
        ).fillna(self.medians_["condition"])
        self.days_coef_, self.days_intercept_ = self._ols(design, days)
        self.cond_coef_, self.cond_intercept_ = self._ols(design, cond)

        exposure = np.column_stack(
            [
                days.to_numpy(dtype=float),
                cond.to_numpy(dtype=float),
                np.ones(len(source)),
            ]
        )
        for column in self.residual_latent:
            if column not in source:
                continue
            target = pd.to_numeric(source[column], errors="coerce")
            med = float(target[np.isfinite(target)].median()) if target.notna().any() else 0.0
            self.medians_[column] = med
            filled = target.fillna(med)
            coef, intercept = self._ols(exposure[:, :2], filled)
            self.latent_coef_[column] = coef
            self.latent_intercept_[column] = intercept

        if "x20" in source:
            x20 = pd.to_numeric(source["x20"], errors="coerce")
            self.x20_global_mean_ = float(x20[np.isfinite(x20)].mean()) if x20.notna().any() else 0.0
            self.x20_cond_edges_ = self._quantile_edges(cond, 5)
            bins = self._apply_bin_codes(cond, self.x20_cond_edges_)
            tmp = pd.DataFrame({"bin": bins, "x20": x20})
            means = tmp.groupby("bin")["x20"].mean()
            self.x20_cond_means_ = {int(k): float(v) for k, v in means.items() if np.isfinite(v)}

        self._finalize(self._transform_source(source), fit_stage=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._ensure_fitted()
        self._validate_frame(X)
        return self._finalize(self._transform_source(self._without_targets(X)), fit_stage=False)

    def _design(self, source: pd.DataFrame) -> np.ndarray:
        cc = pd.to_numeric(source.get("cc", pd.Series(index=source.index)), errors="coerce").fillna(
            self.medians_.get("cc", 0.0)
        )
        v = pd.to_numeric(source.get("V", pd.Series(index=source.index)), errors="coerce").fillna(
            self.medians_.get("V", 0.0)
        )
        mg = pd.to_numeric(
            source.get("max_g", pd.Series(index=source.index)), errors="coerce"
        ).fillna(self.medians_.get("max_g", 0.0))
        return np.column_stack([cc.to_numpy(dtype=float), v.to_numpy(dtype=float), mg.to_numpy(dtype=float)])

    @staticmethod
    def _ols(X: np.ndarray, y: pd.Series) -> tuple[np.ndarray, float]:
        yv = y.to_numpy(dtype=float)
        mask = np.isfinite(yv) & np.all(np.isfinite(X), axis=1)
        if mask.sum() < X.shape[1] + 2:
            return np.zeros(X.shape[1]), float(np.nanmean(yv)) if np.isfinite(yv).any() else 0.0
        Xd = np.column_stack([X[mask], np.ones(mask.sum())])
        try:
            coef, _, _, _ = np.linalg.lstsq(Xd, yv[mask], rcond=None)
        except np.linalg.LinAlgError:
            return np.zeros(X.shape[1]), float(np.nanmean(yv[mask]))
        return coef[:-1], float(coef[-1])

    @staticmethod
    def _quantile_edges(values: pd.Series, bins: int) -> np.ndarray:
        finite = values[np.isfinite(values)]
        if finite.empty:
            return np.array([], dtype=float)
        edges = np.unique(finite.quantile(np.linspace(0, 1, bins + 1)).to_numpy(dtype=float))
        return edges[1:-1] if len(edges) > 1 else np.array([], dtype=float)

    @staticmethod
    def _apply_bin_codes(values: pd.Series, edges: np.ndarray) -> np.ndarray:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        output = np.full(len(numeric), -1, dtype=np.int16)
        valid = np.isfinite(numeric)
        if edges.size:
            output[valid] = np.searchsorted(edges, numeric[valid], side="right").astype(np.int16)
        else:
            output[valid] = 0
        return output

    def _transform_source(self, source: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=source.index)
        cc = pd.to_numeric(source.get("cc", pd.Series(index=source.index)), errors="coerce").fillna(
            self.medians_.get("cc", 0.0)
        )
        v = pd.to_numeric(source.get("V", pd.Series(index=source.index)), errors="coerce").fillna(
            self.medians_.get("V", 0.0)
        )
        mg = pd.to_numeric(
            source.get("max_g", pd.Series(index=source.index)), errors="coerce"
        ).fillna(self.medians_.get("max_g", 0.0))
        days = pd.to_numeric(source.get("days", pd.Series(index=source.index)), errors="coerce").fillna(
            self.medians_.get("days", 0.0)
        )
        cond = pd.to_numeric(
            source.get("condition", pd.Series(index=source.index)), errors="coerce"
        ).fillna(self.medians_.get("condition", 0.0))

        out["cc_log1p"] = np.log1p(cc.clip(lower=0)).astype(float)
        out["max_g_log1p"] = np.log1p(mg.clip(lower=0)).astype(float)
        out["cc_per_V"] = (cc / (v.abs() + 1e-6)).astype(float)
        out["V_per_max_g"] = (v / (mg.abs() + 1.0)).astype(float)
        out["cc_per_max_g"] = (cc / (mg.abs() + 1.0)).astype(float)
        try:
            out["V_bin"] = (
                pd.qcut(v.rank(method="first"), q=10, labels=False)
                .astype(str)
                .radd("vbin_")
            )
        except ValueError:
            out["V_bin"] = "vbin_0"

        design = np.column_stack([cc.to_numpy(dtype=float), v.to_numpy(dtype=float), mg.to_numpy(dtype=float)])
        out["days_profile_resid"] = (
            days.to_numpy(dtype=float) - design @ self.days_coef_ - self.days_intercept_
        ).astype(float)
        out["cond_profile_resid"] = (
            cond.to_numpy(dtype=float) - design @ self.cond_coef_ - self.cond_intercept_
        ).astype(float)

        for column in self.keep_latent:
            if column in source:
                values = pd.to_numeric(source[column], errors="coerce").fillna(
                    self.medians_.get(column, 0.0)
                )
                out[f"{column}__kept"] = values.astype(float)

        if "x20" in source:
            x20 = pd.to_numeric(source["x20"], errors="coerce").fillna(self.medians_.get("x20", 0.0))
            bins = self._apply_bin_codes(cond, self.x20_cond_edges_)
            expected = np.array(
                [self.x20_cond_means_.get(int(b), self.x20_global_mean_) for b in bins],
                dtype=float,
            )
            out["x20_resid_cond"] = (x20.to_numpy(dtype=float) - expected).astype(float)

        exposure = np.column_stack(
            [days.to_numpy(dtype=float), cond.to_numpy(dtype=float)]
        )
        for column in self.residual_latent:
            if column not in source or column not in self.latent_coef_:
                continue
            values = pd.to_numeric(source[column], errors="coerce").fillna(
                self.medians_.get(column, 0.0)
            )
            pred = exposure @ self.latent_coef_[column] + self.latent_intercept_[column]
            out[f"{column}__resid_dc"] = (values.to_numpy(dtype=float) - pred).astype(float)

        # Compress near-ID latent noise into row stats (no raw x0-x18).
        latent_cols = [f"x{i}" for i in range(0, 19) if f"x{i}" in source]
        if latent_cols:
            mat = source[latent_cols].apply(pd.to_numeric, errors="coerce")
            out["latent_row_mean"] = mat.mean(axis=1).astype(float)
            out["latent_row_std"] = mat.std(axis=1).fillna(0.0).astype(float)
            out["latent_row_min"] = mat.min(axis=1).astype(float)
            out["latent_row_max"] = mat.max(axis=1).astype(float)
            out["latent_row_l2"] = np.sqrt((mat.fillna(0.0) ** 2).sum(axis=1)).astype(float)
        return out


__all__ = ["NumericPhysicsFeatureBlock"]
