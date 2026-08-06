import numpy as np
import pandas as pd
import pytest

from insurance_claim.ebm_arm import (
    EBM_INTERACTIONS,
    EBMConfig,
    build_ebm_features,
    infer_feature_types,
    run_cv_seed,
)


def test_build_ebm_features_is_row_local_and_excludes_id_label() -> None:
    frame = pd.DataFrame(
        {
            "id": ["a", "b"],
            "label": [0, 1],
            "month": ["M1", "M12"],
            "source": ["CAR_2|ENG_262", "CAR_10|ENG_651"],
            "version": ["v3", "v18"],
            "t3": ["4.79E", "5.20P"],
            "region": ["6645", "f09d"],
            "code": ["A", "B"],
            "grades": ["ss", "sss"],
            "days": [100.0, 1000.0],
            "condition": [0.1, np.nan],
            "age_range": [1, 2],
            "livability": [0.2, 0.4],
            "cc": [1000.0, 2000.0],
            "V": [8.0, 10.0],
            "max_g": [100.0, 200.0],
            "x0": [0.1, -0.1],
            "x1": [0.2, -0.2],
        }
    )

    features = build_ebm_features(frame)

    assert "id" not in features and "label" not in features
    assert features["month_number"].tolist() == [1.0, 12.0]
    assert features["source_car"].tolist() == ["2", "10"]
    assert features["source_engine"].tolist() == ["262", "651"]
    assert features["t3_value"].tolist() == [4.79, 5.2]
    assert features["condition_missing"].tolist() == [0, 1]
    assert features["x_l2"].tolist() == pytest.approx([np.sqrt(0.05), np.sqrt(0.05)])


def test_feature_types_and_interactions_match_engineered_schema() -> None:
    frame = pd.DataFrame(
        {
            "id": ["a", "b"],
            "month": ["M1", "M2"],
            "source": ["CAR_1|ENG_1", "CAR_2|ENG_2"],
            "version": ["v1", "v2"],
            "t3": ["4.1E", "4.2P"],
            "region": ["r1", "r2"],
            "code": ["A", "B"],
            "grades": ["s", "ss"],
            "days": [1.0, 2.0],
            "condition": [0.1, 0.2],
            "age_range": [1, 2],
            "livability": [0.2, 0.3],
            "cc": [1.0, 2.0],
            "V": [1.0, 2.0],
            "max_g": [1.0, 2.0],
            "x0": [0.0, 1.0],
        }
    )
    features = build_ebm_features(frame)

    feature_types = infer_feature_types(features)

    assert len(feature_types) == features.shape[1]
    assert feature_types[features.columns.get_loc("source")] == "nominal"
    assert feature_types[features.columns.get_loc("days")] == "continuous"
    assert all(
        left in features and right in features for left, right in EBM_INTERACTIONS
    )


def test_cv_seed_writes_and_reuses_checkpoint(monkeypatch, tmp_path) -> None:
    class FakeModel:
        def fit(self, features, target):
            return self

        def predict_proba(self, features):
            score = 1 / (1 + np.exp(-features["signal"].to_numpy(dtype=float)))
            return np.column_stack([1 - score, score])

    monkeypatch.setattr(
        "insurance_claim.ebm_arm._model",
        lambda config, seed, features: FakeModel(),
    )
    train_features = pd.DataFrame(
        {"signal": [-2.0, -1.0, 1.0, 2.0] * 4, "category": ["a", "b"] * 8}
    )
    y = np.array([0, 0, 1, 1] * 4)
    test_features = train_features.iloc[:4].copy()
    config = EBMConfig(folds=2)

    first = run_cv_seed(train_features, y, test_features, 42, tmp_path, config)
    second = run_cv_seed(train_features, y, test_features, 42, tmp_path, config)

    assert first[2]["pooled_auc"] == pytest.approx(1.0)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert (tmp_path / "real_seed42.npz").exists()
    assert not (tmp_path / "real_seed42_partial.npz").exists()

    with pytest.raises(ValueError, match="incompatible cache"):
        run_cv_seed(
            train_features,
            y,
            test_features,
            42,
            tmp_path,
            EBMConfig(folds=2, max_bins=64),
        )
