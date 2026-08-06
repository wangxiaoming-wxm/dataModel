from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from insurance_claim.model import (
    TrainingConfig,
    _stratified_early_split,
    audit_data,
    build_submission,
    engineer_features,
    rank_normalize,
    train_ensemble,
)


def sample_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "id": ["a1", "b2", "c3", "d4"],
            "month": ["M1", "M2", "M1", "M3"],
            "t3": ["4.5E", "5.2P", "4.8E", "5.0P"],
            "source": [
                "CAR_1|ENG_100",
                "CAR_2|ENG_200",
                "CAR_1|ENG_100",
                "CAR_3|ENG_300",
            ],
            "grades": ["s", "ss", "sss", "ss"],
            "version": ["v1", "v2", "v3", "v4"],
            "days": [10.0, 20.0, 30.0, 40.0],
            "condition": [0.1, np.nan, 0.3, 0.4],
            "x0": [0.1, -0.2, 0.3, -0.4],
            "x1": [0.2, 0.1, -0.1, -0.2],
            "label": [0, 1, 0, 1],
        }
    )
    test = train.drop(columns="label").copy()
    test["id"] = ["e5", "f6", "g7", "h8"]
    test["days"] += 0.5
    sample = pd.DataFrame({"id": test["id"], "label": 0})
    return train, test, sample


def test_audit_accepts_clean_competition_data() -> None:
    train, test, sample = sample_frames()

    report = audit_data(train, test, sample)

    assert report["target_rate"] == pytest.approx(0.5)
    assert report["id_overlap"] == 0
    assert report["exact_cross_feature_overlap"] == 0
    assert report["train_test_columns_match"] is True


def test_audit_rejects_identifier_overlap() -> None:
    train, test, sample = sample_frames()
    test.loc[0, "id"] = train.loc[0, "id"]
    sample.loc[0, "id"] = test.loc[0, "id"]

    with pytest.raises(ValueError, match="identifier overlap"):
        audit_data(train, test, sample)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda tr, te, sub: tr.drop(columns="label"), "label must exist"),
        (
            lambda tr, te, sub: tr.assign(label=[0, 2, 0, 1]),
            "label must be non-missing",
        ),
        (
            lambda tr, te, sub: tr.assign(id=["a1", "a1", "c3", "d4"]),
            "identifiers must be unique",
        ),
    ],
)
def test_audit_rejects_invalid_training_data(mutate, message: str) -> None:
    train, test, sample = sample_frames()
    train = mutate(train, test, sample)

    with pytest.raises(ValueError, match=message):
        audit_data(train, test, sample)


def test_audit_rejects_misaligned_submission() -> None:
    train, test, sample = sample_frames()
    sample = sample.iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError, match="submission identifiers"):
        audit_data(train, test, sample)


def test_audit_rejects_feature_dtype_mismatch() -> None:
    train, test, sample = sample_frames()
    test["days"] = test["days"].astype(str)

    with pytest.raises(ValueError, match="feature dtypes differ"):
        audit_data(train, test, sample)


def test_engineering_removes_identifiers_and_adds_semantic_features() -> None:
    train, _, _ = sample_frames()

    features = engineer_features(train.drop(columns="label"))

    assert "id" not in features
    assert {"month_n", "t3_value", "t3_kind", "source_car", "source_eng"} <= set(
        features
    )
    assert {"x_mean", "x_std", "x_l1", "x_l2"} <= set(features)
    assert np.isfinite(features.select_dtypes(include=np.number).dropna()).all().all()


def test_engineering_handles_minimum_integer_without_overflow() -> None:
    frame = pd.DataFrame({"id": ["a"], "x0": [np.iinfo(np.int64).min], "x1": [0]})

    features = engineer_features(frame)

    assert features.loc[0, "x_l1"] > 0
    assert features.loc[0, "x_l2"] > 0


def test_rank_normalize_is_bounded_and_order_preserving() -> None:
    ranked = rank_normalize(np.array([0.8, 0.1, 0.4]))

    assert ranked.min() > 0
    assert ranked.max() < 1
    assert list(np.argsort(ranked)) == [1, 2, 0]


def test_rank_normalize_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="non-empty finite"):
        rank_normalize(np.array([0.2, np.nan]))


def test_submission_preserves_sample_order_and_probabilities(tmp_path: Path) -> None:
    _, test, sample = sample_frames()
    predictions = np.array([0.2, 0.8, 0.4, 0.6])

    output = build_submission(test, sample, predictions, tmp_path / "submission.csv")

    assert output["id"].tolist() == sample["id"].tolist()
    assert output["label"].tolist() == pytest.approx(predictions.tolist())
    assert pd.read_csv(tmp_path / "submission.csv").equals(output)


def test_submission_rejects_invalid_probabilities(tmp_path: Path) -> None:
    _, test, sample = sample_frames()

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        build_submission(
            test,
            sample,
            np.array([0.2, 1.1, 0.4, 0.6]),
            tmp_path / "submission.csv",
        )


def test_train_ensemble_returns_repeated_cv_metrics() -> None:
    train, test, _ = sample_frames()
    train = pd.concat([train] * 15, ignore_index=True)
    train["id"] = [f"train-{index}" for index in range(len(train))]
    test = pd.concat([test] * 3, ignore_index=True)
    test["id"] = [f"test-{index}" for index in range(len(test))]
    config = TrainingConfig(
        folds=2,
        repeats=1,
        seed=7,
        cat_iterations=5,
        xgb_iterations=5,
        early_stopping_rounds=2,
    )

    predictions, metrics = train_ensemble(train, test, config)

    assert predictions.shape == (len(test),)
    assert ((0 <= predictions) & (predictions <= 1)).all()
    assert len(metrics["folds"]) == 2
    assert len(metrics["repeats"]) == 1
    assert metrics["selection_policy"].startswith("fixed 50/50")


def test_train_ensemble_rejects_invalid_cv_configuration() -> None:
    train, test, _ = sample_frames()

    with pytest.raises(ValueError, match="folds must be"):
        train_ensemble(train, test, TrainingConfig(folds=1))


def test_train_ensemble_rejects_missing_or_single_class_target() -> None:
    train, test, _ = sample_frames()
    train["label"] = 0

    with pytest.raises(ValueError, match="both binary classes"):
        train_ensemble(train, test, TrainingConfig(folds=2))


def test_train_ensemble_supports_minimum_valid_stratified_data() -> None:
    train, test, _ = sample_frames()
    train = pd.concat([train, train], ignore_index=True)
    train["id"] = [f"minimum-{index}" for index in range(len(train))]
    config = TrainingConfig(
        folds=2,
        repeats=1,
        cat_iterations=1,
        xgb_iterations=1,
        early_stopping_rounds=1,
    )

    predictions, _ = train_ensemble(train, test, config)

    assert len(predictions) == len(test)


def test_inner_early_split_retains_rare_class_in_both_partitions() -> None:
    y = pd.Series([0] * 996 + [1] * 4)
    fit_index = np.arange(len(y))

    inner, early = _stratified_early_split(fit_index, y, 150, seed=7)

    assert set(y.iloc[inner]) == {0, 1}
    assert set(y.iloc[early]) == {0, 1}
    assert not set(inner) & set(early)
    assert set(inner) | set(early) == set(fit_index)
