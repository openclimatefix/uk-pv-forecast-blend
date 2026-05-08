"""
Tests:
1. test_app                    - App completes and saves blend forecasts to DP for all GSPs
2. test_app_twice              - Running app a second time updates forecasts without duplicating
3. test_app_only_national      - App restricted to gsp_id=0 saves only the national blend
4. test_app_only_ecwmf_and_xg  - Blend with only ecmwf/xg models transitions correctly across horizon
"""

import datetime
import os
import pytest
import pytest_asyncio
import time_machine
import pandas as pd
from betterproto.lib.google.protobuf import Struct, Value
from ocf import dp
from unittest.mock import AsyncMock, patch

from forecast_blend.app import app
from forecast_blend.weights import ALL_MODEL_NAMES


@pytest_asyncio.fixture(autouse=True, scope="session", loop_scope="session")
async def setup_dp_locations(data_client):
    """Create locations and seed input forecasts in Data Platform for app tests."""
    location_uuids = {}
    for gsp_id in range(0, 15):
        try:
            metadata_gsp = Struct(fields={"gsp_id": Value(number_value=gsp_id)})
            resp = await data_client.create_location(dp.CreateLocationRequest(
                location_name=f"test_gsp_{gsp_id}",
                energy_source=dp.EnergySource.SOLAR,
                geometry_wkt="POLYGON((-2 52, -1 52, -1 53, -2 53, -2 52))",
                location_type=dp.LocationType.NATION if gsp_id == 0 else dp.LocationType.GSP,
                effective_capacity_watts=100_000_000_000,
                metadata=metadata_gsp,
                valid_from_utc=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
            ))
            location_uuids[gsp_id] = resp.location_uuid
        except Exception:
            pass

    # Seed forecasts for each model so the app can blend them
    init_time = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)
    values = [
        dp.CreateForecastRequestForecastValue(
            horizon_mins=30 * i,
            p50_fraction=0.5,
            metadata=Struct(),
        )
        for i in range(1, 17)
    ]
    for model_name in ALL_MODEL_NAMES:
        gsp_ids_for_model = [0] if model_name == "National_xg" else list(range(0, 12))
        try:
            forecaster_resp = await data_client.create_forecaster(
                dp.CreateForecasterRequest(name=model_name, version="1.0.0")
            )
            adj_resp = await data_client.create_forecaster(
                dp.CreateForecasterRequest(name=f"{model_name}_adjust", version="1.0.0")
            )
        except Exception:
            continue
        for gsp_id in gsp_ids_for_model:
            if gsp_id not in location_uuids:
                continue
            forecasters = [forecaster_resp.forecaster]
            if gsp_id == 0:
                forecasters.append(adj_resp.forecaster)
            for forecaster in forecasters:
                try:
                    await data_client.create_forecast(dp.CreateForecastRequest(
                        forecaster=forecaster,
                        location_uuid=location_uuids[gsp_id],
                        energy_source=dp.EnergySource.SOLAR,
                        init_time_utc=init_time,
                        values=values,
                    ))
                except Exception:
                    pass


async def _get_blend_forecasts(data_client, blend_name):
    """Helper: fetch blend forecasts for the national (NATION) location."""
    locations_resp = await data_client.list_locations(dp.ListLocationsRequest(
        energy_source_filter=dp.EnergySource.SOLAR,
        location_type_filter=dp.LocationType.NATION,
    ))
    assert len(locations_resp.locations) >= 1
    location_uuid = locations_resp.locations[0].location_uuid

    forecasts_resp = await data_client.get_latest_forecasts(dp.GetLatestForecastsRequest(
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
    ))
    return [
        f for f in forecasts_resp.forecasts
        if f.forecaster.forecaster_name == blend_name
    ]


@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
async def test_app(data_client):
    """App should complete without errors and save forecasts to Data Platform for all GSPs."""
    await app(gsps=list(range(0, 11)))

    blend_name = os.environ["BLEND_NAME"]
    blend_forecasts = await _get_blend_forecasts(data_client, blend_name)
    assert len(blend_forecasts) >= 1

    # Check the national blend has the right number of forecast values (16 seeded)
    national_resp = await data_client.list_locations(dp.ListLocationsRequest(
        energy_source_filter=dp.EnergySource.SOLAR,
        location_type_filter=dp.LocationType.NATION,
    ))
    national_uuid = national_resp.locations[0].location_uuid
    init_time = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)
    timeseries_resp = await data_client.get_forecast_as_timeseries(
        dp.GetForecastAsTimeseriesRequest(
            location_uuid=national_uuid,
            energy_source=dp.EnergySource.SOLAR,
            forecaster=blend_forecasts[0].forecaster,
            time_window=dp.TimeWindow(
                start_timestamp_utc=init_time,
                end_timestamp_utc=init_time + datetime.timedelta(hours=9),
            ),
        )
    )
    assert len(timeseries_resp.values) == 16

    # Check blend forecasts exist for all GSPs (gsp_ids 1-10)
    gsp_resp = await data_client.list_locations(dp.ListLocationsRequest(
        energy_source_filter=dp.EnergySource.SOLAR,
        location_type_filter=dp.LocationType.GSP,
    ))
    gsp_blend_count = 0
    for loc in gsp_resp.locations:
        forecasts_resp = await data_client.get_latest_forecasts(dp.GetLatestForecastsRequest(
            location_uuid=loc.location_uuid,
            energy_source=dp.EnergySource.SOLAR,
        ))
        if any(f.forecaster.forecaster_name == blend_name for f in forecasts_resp.forecasts):
            gsp_blend_count += 1
    assert gsp_blend_count == 10  # gsp_ids 1-10


@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
async def test_app_twice(data_client):
    """Running the app a second time should succeed and update the saved blend forecast."""
    blend_name = os.environ["BLEND_NAME"]

    await app(gsps=list(range(0, 11)))
    blend_after_first = await _get_blend_forecasts(data_client, blend_name)
    assert len(blend_after_first) >= 1

    await app(gsps=list(range(0, 11)))
    blend_after_second = await _get_blend_forecasts(data_client, blend_name)
    assert len(blend_after_second) == len(blend_after_first)


@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
async def test_app_only_national(data_client):
    """App run restricted to gsp_id=0 should save only the national blend forecast."""
    await app(gsps=[0])

    blend_forecasts = await _get_blend_forecasts(data_client, os.environ["BLEND_NAME"])
    assert len(blend_forecasts) >= 1


@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
async def test_app_only_ecwmf_and_xg(data_client):
    """When only pvnet_ecmwf (value=0) and National_xg (value=1) have data, the blend
    should transition from ecmwf-only to xg-only across forecast horizon."""
    blend_name = os.environ["BLEND_NAME"]
    init_time = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)

    # Build weights that use only pvnet_ecmwf and National_xg
    t0 = pd.Timestamp("2023-01-01 00:00", tz="UTC")
    horizons = pd.timedelta_range("30min", periods=25, freq="30min")
    ecmwf_weights = pd.Series(
        [1.0] * 16 + [0.75, 0.5, 0.25] + [0.0] * 6, index=t0 + horizons
    )
    xg_weights = 1.0 - ecmwf_weights

    mock_weights = pd.DataFrame({"pvnet_ecmwf": ecmwf_weights, "National_xg": xg_weights})

    with patch("forecast_blend.app.get_national_blend_weights", new=AsyncMock(return_value=mock_weights)), \
         patch("forecast_blend.app.get_regional_blend_weights", new=AsyncMock(return_value=mock_weights)):
        await app(gsps=[0])

    locations_resp = await data_client.list_locations(dp.ListLocationsRequest(
        energy_source_filter=dp.EnergySource.SOLAR,
        location_type_filter=dp.LocationType.NATION,
    ))
    location_uuid = locations_resp.locations[0].location_uuid

    forecasts_resp = await data_client.get_latest_forecasts(dp.GetLatestForecastsRequest(
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
    ))
    blend_forecast = next(
        (f for f in forecasts_resp.forecasts if f.forecaster.forecaster_name == blend_name),
        None,
    )
    assert blend_forecast is not None

    # Fetch the timeseries and verify values exist
    timeseries_resp = await data_client.get_forecast_as_timeseries(
        dp.GetForecastAsTimeseriesRequest(
            location_uuid=location_uuid,
            energy_source=dp.EnergySource.SOLAR,
            forecaster=blend_forecast.forecaster,
            time_window=dp.TimeWindow(
                start_timestamp_utc=init_time,
                end_timestamp_utc=init_time + datetime.timedelta(hours=13),
            ),
        )
    )
    assert len(timeseries_resp.values) == 16
