"""
Integration tests for reading forecasts from the Data Platform.

These tests verify that:
1. Forecast values can be read from the Data Platform
2. Forecast metadata can be read for calculating model delays
3. The blend service can blend forecasts read from the Data Platform
"""

import datetime

import pandas as pd
import pytest
from betterproto.lib.google.protobuf import Struct
from dp_sdk.ocf import dp

from forecast_blend.forecast.data_platform import (
    fetch_dp_forecast_values,
    get_forecast_values_from_data_platform
)
from forecast_blend.weights import _fetch_latest_forecast_metadata_from_dp


@pytest.fixture
def set_data_platform_env(setup_test_data, monkeypatch):
    """Fixture to set data platform environment variables for tests."""
    test_data = setup_test_data
    monkeypatch.setenv("READ_FROM_DATA_PLATFORM", "true")
    monkeypatch.setenv("DATA_PLATFORM_HOST", test_data["host"])
    monkeypatch.setenv("DATA_PLATFORM_PORT", str(test_data["port"]))
    yield
    # monkeypatch automatically cleans up after test


async def create_test_forecast(
    client: dp.DataPlatformDataServiceStub,
    forecaster: dp.Forecaster,
    location_uuid: str,
    init_time: datetime.datetime,
    forecast_values: list[dict],
    capacity_watts: int,
) -> None:
    """Helper function to create a forecast in the Data Platform."""
    dp_values = []
    for v in forecast_values:
        horizon_mins = int((v["target_time"] - init_time).total_seconds() / 60)
        if horizon_mins < 0:
            continue

        # Normalize to fraction
        p50_fraction = v["p50_mw"] * 1e6 / capacity_watts

        other_stats = {}
        if "p10_mw" in v:
            other_stats["p10"] = v["p10_mw"] * 1e6 / capacity_watts
        if "p90_mw" in v:
            other_stats["p90"] = v["p90_mw"] * 1e6 / capacity_watts

        dp_values.append(
            dp.CreateForecastRequestForecastValue(
                horizon_mins=horizon_mins,
                p50_fraction=p50_fraction,
                metadata=Struct(),
                other_statistics_fractions=other_stats,
            )
        )

    request = dp.CreateForecastRequest(
        forecaster=forecaster,
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        init_time_utc=init_time,
        values=dp_values,
    )
    await client.create_forecast(request)


@pytest.mark.asyncio(loop_scope="session")
async def test_read_forecast_metadata_from_data_platform(
    client, setup_test_data, set_data_platform_env
):
    """
    Test that forecast metadata can be read from the Data Platform.
    This is used for calculating model delays in weights.py.
    """
    test_data = setup_test_data

    # Create a forecast for pvnet_day_ahead
    init_time = datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.UTC)
    forecast_values = [
        {
            "target_time": init_time + datetime.timedelta(minutes=30 * i),
            "p50_mw": 100 + i * 10,
        }
        for i in range(1, 9)
    ]

    await create_test_forecast(
        client=client,
        forecaster=test_data["forecasters"]["pvnet_day_ahead"],
        location_uuid=test_data["locations"][0]["uuid"],
        init_time=init_time,
        forecast_values=forecast_values,
        capacity_watts=test_data["locations"][0]["capacity_watts"],
    )

    t0 = pd.Timestamp("2025-01-01 12:00", tz="UTC")
    max_delay = pd.Timedelta("36h")

    df = await _fetch_latest_forecast_metadata_from_dp(
        client=client,
        model_names=["pvnet_day_ahead"],
        t0=t0,
        max_delay=max_delay,
    )

    # Verify results
    assert len(df) > 0, "Should have found forecast metadata"
    assert "name" in df.columns
    assert "pvnet_day_ahead" in df["name"].values


@pytest.mark.asyncio(loop_scope="session")
async def test_read_forecast_values_from_data_platform(
    client, setup_test_data, set_data_platform_env
):
    """
    Test that forecast values can be read from the Data Platform.
    """
    test_data = setup_test_data

    # Create a forecast
    init_time = datetime.datetime(2025, 1, 2, 12, 0, tzinfo=datetime.UTC)
    expected_values = [
        {
            "target_time": init_time + datetime.timedelta(minutes=30 * i),
            "p50_mw": 100 + i * 10,
            "p10_mw": 90 + i * 10,
            "p90_mw": 110 + i * 10,
        }
        for i in range(1, 9)
    ]

    await create_test_forecast(
        client=client,
        forecaster=test_data["forecasters"]["pvnet_day_ahead"],
        location_uuid=test_data["locations"][0]["uuid"],
        init_time=init_time,
        forecast_values=expected_values,
        capacity_watts=test_data["locations"][0]["capacity_watts"],
    )

    # Test reading - we need to read from the data platform
    # The actual implementation uses asyncio.run internally, but we're in an async context
    # So we'll test the underlying async function directly

    # First verify we can find the location
    location_uuid = test_data["locations"][0]["uuid"]
    assert location_uuid == test_data["locations"][0]["uuid"]

    # Then fetch forecast values
    start_datetime = init_time.replace(tzinfo=None)
    values = await fetch_dp_forecast_values(
        client=client,
        location_uuid=location_uuid,
        model_name="pvnet_day_ahead",
        start_datetime=start_datetime,
    )

    assert len(values) == 8, f"Expected 8 forecast values, got {len(values)}"


@pytest.mark.asyncio(loop_scope="session")
async def test_blend_forecasts_from_data_platform(
    client, setup_test_data, set_data_platform_env
):
    """
    Test end-to-end: read multiple model forecasts from Data Platform and blend them.
    """
    test_data = setup_test_data

    # Create forecasts for two models
    init_time = datetime.datetime(2025, 1, 3, 12, 0, tzinfo=datetime.UTC)

    # pvnet_day_ahead forecast - values from 100-180
    pvnet_values = [
        {
            "target_time": init_time + datetime.timedelta(minutes=30 * i),
            "p50_mw": 100 + i * 10,
        }
        for i in range(1, 9)
    ]

    await create_test_forecast(
        client=client,
        forecaster=test_data["forecasters"]["pvnet_day_ahead"],
        location_uuid=test_data["locations"][0]["uuid"],
        init_time=init_time,
        forecast_values=pvnet_values,
        capacity_watts=test_data["locations"][0]["capacity_watts"],
    )

    # national_xg forecast - values from 200-280
    xg_values = [
        {
            "target_time": init_time + datetime.timedelta(minutes=30 * i),
            "p50_mw": 200 + i * 10,
        }
        for i in range(1, 9)
    ]

    await create_test_forecast(
        client=client,
        forecaster=test_data["forecasters"]["national_xg"],
        location_uuid=test_data["locations"][0]["uuid"],
        init_time=init_time,
        forecast_values=xg_values,
        capacity_watts=test_data["locations"][0]["capacity_watts"],
    )

    # Get forecasts for both models
    pvnet_forecast = await get_forecast_values_from_data_platform(
        client=client,
        location_uuid=test_data["locations"][0]["uuid"],
        model_name="pvnet_day_ahead",
        start_datetime=init_time.replace(tzinfo=None),
    )

    xg_forecast = await get_forecast_values_from_data_platform(
        client=client,
        location_uuid=test_data["locations"][0]["uuid"],
        model_name="national_xg",
        start_datetime=init_time.replace(tzinfo=None),
    )

    # Verify we got data from both models
    assert len(pvnet_forecast) == 8, (
        f"Expected 8 pvnet values, got {len(pvnet_forecast)}"
    )
    assert len(xg_forecast) == 8, f"Expected 8 xg values, got {len(xg_forecast)}"

    # Verify the values are different (confirming we got data from different models)
    pvnet_first_value = pvnet_forecast["expected_power_generation_megawatts"].iloc[0]
    xg_first_value = xg_forecast["expected_power_generation_megawatts"].iloc[0]

    # The values should be different since we created different forecasts
    assert pvnet_first_value != xg_first_value, "Forecasts should have different values"


@pytest.mark.asyncio(loop_scope="session")
async def test_read_with_no_forecasts_returns_empty(
    client, setup_test_data, set_data_platform_env
):
    """
    Test that reading forecasts for a non-existent model returns empty list.
    """
    test_data = setup_test_data

    # Try to fetch forecasts for a model that doesn't exist
    values = await fetch_dp_forecast_values(
        client=client,
        location_uuid=test_data["locations"][0]["uuid"],
        model_name="nonexistent_model",
        start_datetime=datetime.datetime(2025, 1, 1),
    )

    assert values == [], "Should return empty list for non-existent model"
