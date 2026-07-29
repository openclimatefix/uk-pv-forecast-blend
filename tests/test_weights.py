"""
Tests:
1. test_get_horizon_maes           - Horizon MAE function runs and covers all model names
2. test_get_blend_weights          - Blend weights sum to ≤1 and include expected models
3. test_get_blend_weights_cloud    - Blend weights handle excluded/intraday model combinations (parametrized)
"""

import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import time_machine
from forecast_blend.weights import (
    ALL_MODEL_NAMES,
    infer_backup_model_name,
    get_horizon_maes,
    get_blend_weights,
)


def test_get_horizon_maes():
    # Just check the function can run
    df = get_horizon_maes()
    
    # Check all the expected models are present
    assert set(ALL_MODEL_NAMES) - set(df.columns) == set()


def test_infer_backup_model_name_prefers_longest_horizon_then_lowest_mae():
    df = pd.DataFrame(
        {
            "short": [1.0, 2.0, None],
            "long_higher_mae": [3.0, 3.0, 3.0],
            "long_lower_mae": [2.0, 2.0, 2.0],
        },
        index=pd.to_timedelta(["30min", "60min", "90min"]),
    )

    assert infer_backup_model_name(df) == "long_lower_mae"


def test_infer_backup_model_name_returns_first_when_no_valid_mae():
    df = pd.DataFrame(
        {
            "a": [None, None],
            "b": [None, None],
        },
        index=pd.to_timedelta(["30min", "1h"]),
    )

    assert infer_backup_model_name(df) == "a"


def _make_metadata_df(model_names, forecast_creation_time):
    """Build a minimal metadata DataFrame like the one returned by _fetch_latest_forecast_metadata_from_dp."""
    return pd.DataFrame([
        {
            "created_utc": forecast_creation_time,
            "forecast_creation_time": forecast_creation_time,
            "location_id": f"uuid-{name}",
            "name": name,
        }
        for name in model_names
    ])


@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
async def test_get_blend_weights():
    t0 = pd.Timestamp("2023-01-01 00:00", tz="UTC")
    # pvnet_ecmwf and pvnet_day_ahead are current (no delay)
    forecast_time = pd.Timestamp("2023-01-01 00:00", tz="UTC").to_pydatetime()
    metadata_df = _make_metadata_df(["pvnet_ecmwf", "pvnet_day_ahead"], forecast_time)

    mock_client = MagicMock()
    with patch(
        "forecast_blend.weights.get_latest_forecast_metadata",
        new=AsyncMock(return_value=metadata_df),
    ):
        weights_df = await get_blend_weights(client=mock_client, t0=t0)

    assert set(weights_df.columns).issubset(set(ALL_MODEL_NAMES))
    assert (weights_df.sum(axis=1).round(6) <= 1.0001).all()
    assert len(weights_df) > 0


# Test with and without excluding the pvnet_cloud model
test_settings = [(None, "pvnet_cloud"),  (["pvnet_cloud"], "pvnet_v2")]
@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("exclude_models, intraday_model", test_settings)
@pytest.mark.skip(reason="Cloudcasting MAE weights are not available.")
async def test_get_blend_weights_cloud(exclude_models, intraday_model):
    t0 = pd.Timestamp("2023-01-01 00:00", tz="UTC")

    forecast_time = pd.Timestamp("2023-01-01 00:00", tz="UTC").to_pydatetime()

    available = ["pvnet_day_ahead", "pvnet_v2", "pvnet_ecmwf", "pvnet_cloud"]
    if exclude_models:
        available = [m for m in available if m not in exclude_models]
    metadata_df = _make_metadata_df(available, forecast_time)

    mock_client = MagicMock()
    with patch(
        "forecast_blend.weights.get_latest_forecast_metadata",
        new=AsyncMock(return_value=metadata_df),
    ):
        weights_df = await get_blend_weights(client=mock_client, t0=t0, exclude_models=exclude_models)

    assert intraday_model in weights_df.columns or "pvnet_day_ahead" in weights_df.columns
    assert len(weights_df) > 0
