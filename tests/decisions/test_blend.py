import pandas as pd
import pytest

from baram.decisions.blend import apply_blend, fit_two_model_blend
from baram.exceptions import ContractError

CAPACITIES = {1: 21600.0, 2: 21600.0, 3: 21000.0}


def _prediction_frames() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    labels: list[dict[str, object]] = []
    first: list[dict[str, object]] = []
    second: list[dict[str, object]] = []
    for group, capacity in CAPACITIES.items():
        for index in range(3):
            key = {
                "forecast_id": f"{group}-{index}",
                "forecast_kst_dtm": pd.Timestamp("2023-01-01") + pd.Timedelta(index + 1, unit="h"),
                "group_id": group,
            }
            actual = 0.5 * capacity
            labels.append({**key, "actual_kwh": actual})
            first.append({**key, "prediction_kwh": actual})
            second.append({**key, "prediction_kwh": actual + 0.1 * capacity})
    return {"a": pd.DataFrame(first), "b": pd.DataFrame(second)}, pd.DataFrame(labels)


def test_blend_weights_are_convex_and_choose_perfect_parent() -> None:
    """Catches nonconvex weights or a tie-break away from the simpler best parent."""
    predictions, labels = _prediction_frames()
    policy = fit_two_model_blend(
        predictions,
        labels,
        CAPACITIES,
        "1" * 64,
        "2" * 64,
        weight_step=0.05,
    )
    for weights in policy.weights_by_group.values():
        assert weights["a"] == 1.0
        assert weights["b"] == 0.0
        assert sum(weights.values()) == pytest.approx(1.0)
    blended = apply_blend(policy, predictions)
    assert blended["prediction_kwh"].tolist() == labels["actual_kwh"].tolist()


def test_blend_rejects_key_mismatch() -> None:
    """Catches blending predictions for different validation rows."""
    predictions, labels = _prediction_frames()
    predictions["b"] = predictions["b"].iloc[:-1]
    with pytest.raises(ContractError, match="keys"):
        fit_two_model_blend(predictions, labels, CAPACITIES, "1" * 64, "2" * 64)


def test_frozen_blend_applies_to_future_matching_parent_models() -> None:
    """Catches OOF evidence hashes being mistaken for future-row prediction hashes."""
    predictions, labels = _prediction_frames()
    policy = fit_two_model_blend(
        predictions,
        labels,
        CAPACITIES,
        "1" * 64,
        "2" * 64,
    )
    future = {
        model_id: frame.assign(
            forecast_id="future-" + frame["forecast_id"],
            forecast_kst_dtm=frame["forecast_kst_dtm"] + pd.Timedelta(365, unit="D"),
        )
        for model_id, frame in predictions.items()
    }
    result = apply_blend(policy, future)
    assert len(result) == len(labels)
    assert result["forecast_id"].str.startswith("future-").all()
