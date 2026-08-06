import numpy as np
import pandas as pd
import pytest

from insurance_claim.realmlp_arm import (
    RealMLPConfig,
    prepare_fold_frames,
    run_cv_seed,
)


def test_prepare_fold_frames_uses_training_medians_only() -> None:
    train = pd.DataFrame(
        {
            "numeric": [1.0, np.nan, 3.0],
            "all_missing": [np.nan, np.nan, np.nan],
            "category": ["a", None, "b"],
        }
    )
    valid = pd.DataFrame(
        {
            "numeric": [1000.0, np.nan],
            "all_missing": [5.0, np.nan],
            "category": ["c", None],
        }
    )
    test = valid.copy()

    train_out, valid_out, test_out, categories = prepare_fold_frames(train, valid, test)

    assert train_out["numeric"].tolist() == [1.0, 2.0, 3.0]
    assert valid_out["numeric"].tolist() == [1000.0, 2.0]
    assert test_out["numeric"].tolist() == [1000.0, 2.0]
    assert train_out["all_missing"].tolist() == [0.0, 0.0, 0.0]
    assert categories == ["category"]
    assert train_out["category"].tolist() == ["a", "__MISSING__", "b"]


def test_realmlp_cv_seed_writes_and_reuses_checkpoint(monkeypatch, tmp_path) -> None:
    class FakeModel:
        def fit(
            self,
            features,
            target,
            cat_col_names,
            time_to_fit_in_seconds,
        ):
            return self

        def predict_proba(self, features):
            score = 1 / (1 + np.exp(-features["signal"].to_numpy(dtype=float)))
            return np.column_stack([1 - score, score])

    monkeypatch.setattr(
        "insurance_claim.realmlp_arm._model",
        lambda config, seed, tmp_folder: FakeModel(),
    )
    train_features = pd.DataFrame(
        {"signal": [-2.0, -1.0, 1.0, 2.0] * 4, "category": ["a", "b"] * 8}
    )
    y = np.array([0, 0, 1, 1] * 4)
    test_features = train_features.iloc[:4].copy()
    config = RealMLPConfig(folds=2, epochs=1)

    first = run_cv_seed(train_features, y, test_features, 42, tmp_path, config)
    second = run_cv_seed(train_features, y, test_features, 42, tmp_path, config)

    assert first[2]["pooled_auc"] == pytest.approx(1.0)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
