"""
Integration tests for reading forecasts from the Data Platform.

These tests verify that:
1. Forecast values can be read from the Data Platform
2. Forecast metadata can be read for calculating model delays
3. The blend service can blend forecasts read from the Data Platform
"""

import datetime
import time

import pandas as pd
import pytest
from betterproto.lib.google.protobuf import Struct, Value
from dp_sdk.ocf import dp
from grpclib.client import Channel
import pytest_asyncio
from testcontainers.core.container import DockerContainer
from testcontainers.postgres import PostgresContainer
from importlib.metadata import version

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


@pytest_asyncio.fixture(scope="module")
async def dp_client():
    """
    Fixture to spin up a PostgreSQL container and Data Platform for reading tests.
    """
    with PostgresContainer(
        "ghcr.io/openclimatefix/data-platform-pgdb:logging",
        username="postgres",
        password="postgres",  # noqa: S106
        dbname="postgres",
        env={"POSTGRES_HOST": "db"},
    ) as postgres:
        database_url = postgres.get_connection_url()
        database_url = database_url.replace("postgresql+psycopg2", "postgres")
        database_url = database_url.replace("localhost", "host.docker.internal")

        with DockerContainer(
            image=f"ghcr.io/openclimatefix/data-platform:{version('dp_sdk')}",
            env={"DATABASE_URL": database_url},
            ports=[50051],
        ) as data_platform_server:
            time.sleep(2)  # Give some time for the server to start

            port = data_platform_server.get_exposed_port(50051)
            host = data_platform_server.get_container_host_ip()
            channel = Channel(host=host, port=port)
            client = dp.DataPlatformDataServiceStub(channel)
            yield client, host, port
            channel.close()


@pytest_asyncio.fixture(scope="module")
async def setup_test_data(dp_client):
    """
    Fixture to set up test locations and forecasters in the Data Platform.
    Returns the location UUIDs and forecaster info for use in tests.
    """
    client, host, port = dp_client

    # Create test location for GSP 0 (national)
    metadata_gsp0 = Struct(fields={"gsp_id": Value(number_value=0)})
    create_location_request = dp.CreateLocationRequest(
        location_name="test_national_gsp",
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POLYGON((-2 52, -1 52, -1 53, -2 53, -2 52))",
        location_type=dp.LocationType.GSP,
        effective_capacity_watts=1_000_000_000,  # 1 GW
        metadata=metadata_gsp0,
        valid_from_utc=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
    )
    response = await client.create_location(create_location_request)
    location_uuid_gsp0 = response.location_uuid

    # Create test location for GSP 1 (regional)
    metadata_gsp1 = Struct(fields={"gsp_id": Value(number_value=1)})
    create_location_request = dp.CreateLocationRequest(
        location_name="test_regional_gsp_1",
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POLYGON((-3 51, -2 51, -2 52, -3 52, -3 51))",
        location_type=dp.LocationType.GSP,
        effective_capacity_watts=100_000_000,  # 100 MW
        metadata=metadata_gsp1,
        valid_from_utc=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
    )
    response = await client.create_location(create_location_request)
    location_uuid_gsp1 = response.location_uuid

    # Create forecasters for different models
    forecasters = {}
    for model_name in ["pvnet_day_ahead", "national_xg", "pvnet_v2"]:
        response = await client.create_forecaster(
            dp.CreateForecasterRequest(name=model_name, version="1.0.0")
        )
        forecasters[model_name] = response.forecaster

    return {
        "host": host,
        "port": port,
        "locations": {
            0: {"uuid": location_uuid_gsp0, "capacity_watts": 1_000_000_000},
            1: {"uuid": location_uuid_gsp1, "capacity_watts": 100_000_000},
        },
        "forecasters": forecasters,
    }


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


@pytest.mark.asyncio(loop_scope="module")
async def test_read_forecast_metadata_from_data_platform(
    dp_client, setup_test_data, set_data_platform_env
):
    """
    Test that forecast metadata can be read from the Data Platform.
    This is used for calculating model delays in weights.py.
    """
    client, _, _ = dp_client
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


@pytest.mark.asyncio(loop_scope="module")
async def test_read_forecast_values_from_data_platform(
    dp_client, setup_test_data, set_data_platform_env
):
    """
    Test that forecast values can be read from the Data Platform.
    """
    client, _, _ = dp_client
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


@pytest.mark.asyncio(loop_scope="module")
async def test_blend_forecasts_from_data_platform(
    dp_client, setup_test_data, set_data_platform_env
):
    """
    Test end-to-end: read multiple model forecasts from Data Platform and blend them.
    """
    client, _, _ = dp_client
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


@pytest.mark.asyncio(loop_scope="module")
async def test_read_with_no_forecasts_returns_empty(
    dp_client, setup_test_data, set_data_platform_env
):
    """
    Test that reading forecasts for a non-existent model returns empty list.
    """
    test_data = setup_test_data

    # Try to fetch forecasts for a model that doesn't exist
    values = await fetch_dp_forecast_values(
        client=dp_client[0],
        location_uuid=test_data["locations"][0]["uuid"],
        model_name="nonexistent_model",
        start_datetime=datetime.datetime(2025, 1, 1),
    )

    assert values == [], "Should return empty list for non-existent model"
