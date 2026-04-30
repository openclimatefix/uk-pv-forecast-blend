import asyncio
import datetime
import os
import pytest
import pytest_asyncio
import time_machine
import pandas as pd
from betterproto.lib.google.protobuf import Struct, Value
from grpclib.client import Channel
from ocf import dp
from unittest.mock import AsyncMock, patch

from forecast_blend.app import app
from forecast_blend.weights import ALL_MODEL_NAMES


@pytest_asyncio.fixture(autouse=True, scope="module")
async def setup_dp_locations(dp_client):
    """Create locations and seed input forecasts in Data Platform for app tests."""
    host, port = dp_client
    channel = Channel(host=host, port=port)
    client = dp.DataPlatformDataServiceStub(channel)

    location_uuids = {}
    for gsp_id in range(0, 15):
        try:
            metadata_gsp = Struct(fields={"gsp_id": Value(number_value=gsp_id)})
            resp = await client.create_location(dp.CreateLocationRequest(
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
    for model_name in ALL_MODEL_NAMES:
        gsp_ids_for_model = [0] if model_name == "National_xg" else list(range(0, 12))
        for gsp_id in gsp_ids_for_model:
            if gsp_id not in location_uuids:
                continue
            try:
                forecaster_resp = await client.create_forecaster(
                    dp.CreateForecasterRequest(name=model_name, version="1.0.0")
                )
                values = [
                    dp.CreateForecastRequestForecastValue(
                        horizon_mins=30 * i,
                        p50_fraction=0.5,
                        metadata=Struct(),
                    )
                    for i in range(1, 17)
                ]
                await client.create_forecast(dp.CreateForecastRequest(
                    forecaster=forecaster_resp.forecaster,
                    location_uuid=location_uuids[gsp_id],
                    energy_source=dp.EnergySource.SOLAR,
                    init_time_utc=init_time,
                    values=values,
                ))
                # Also create adjuster forecasts for national (gsp_id=0)
                if gsp_id == 0:
                    adj_resp = await client.create_forecaster(
                        dp.CreateForecasterRequest(name=f"{model_name}_adjust", version="1.0.0")
                    )
                    await client.create_forecast(dp.CreateForecastRequest(
                        forecaster=adj_resp.forecaster,
                        location_uuid=location_uuids[gsp_id],
                        energy_source=dp.EnergySource.SOLAR,
                        init_time_utc=init_time,
                        values=values,
                    ))
            except Exception:
                pass

    channel.close()


async def _get_blend_forecasts(dp_client, blend_name):
    """Helper: fetch blend forecasts for the national (NATION) location."""
    host, port = dp_client
    channel = Channel(host=host, port=port)
    client = dp.DataPlatformDataServiceStub(channel)

    locations_resp = await client.list_locations(dp.ListLocationsRequest(
        energy_source_filter=dp.EnergySource.SOLAR,
        location_type_filter=dp.LocationType.NATION,
    ))
    assert len(locations_resp.locations) >= 1
    location_uuid = locations_resp.locations[0].location_uuid

    forecasts_resp = await client.get_latest_forecasts(dp.GetLatestForecastsRequest(
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
    ))
    channel.close()
    return [
        f for f in forecasts_resp.forecasts
        if f.forecaster.forecaster_name == blend_name
    ]


@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
async def test_app(dp_client):
    """App should complete without errors and save forecasts to Data Platform."""
    asyncio.run(app(gsps=list(range(0, 11))))

    blend_forecasts = await _get_blend_forecasts(dp_client, os.environ["BLEND_NAME"])
    assert len(blend_forecasts) >= 1


@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
async def test_app_twice(dp_client):
    """Running the app a second time should succeed and update the saved blend forecast."""
    blend_name = os.environ["BLEND_NAME"]

    asyncio.run(app(gsps=list(range(0, 11))))
    blend_after_first = await _get_blend_forecasts(dp_client, blend_name)
    assert len(blend_after_first) >= 1

    asyncio.run(app(gsps=list(range(0, 11))))
    blend_after_second = await _get_blend_forecasts(dp_client, blend_name)
    assert len(blend_after_second) >= 1


@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
async def test_app_only_national(dp_client):
    """App run restricted to gsp_id=0 should save only the national blend forecast."""
    asyncio.run(app(gsps=[0]))

    blend_forecasts = await _get_blend_forecasts(dp_client, os.environ["BLEND_NAME"])
    assert len(blend_forecasts) >= 1


@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
async def test_app_only_ecwmf_and_xg(dp_client):
    """When only pvnet_ecmwf (value=0) and National_xg (value=1) have data, the blend
    should transition from ecmwf-only to xg-only across forecast horizon."""
    host, port = dp_client
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
        asyncio.run(app(gsps=[0]))

    # Read back from DP
    channel = Channel(host=host, port=port)
    client = dp.DataPlatformDataServiceStub(channel)

    locations_resp = await client.list_locations(dp.ListLocationsRequest(
        energy_source_filter=dp.EnergySource.SOLAR,
        location_type_filter=dp.LocationType.NATION,
    ))
    location_uuid = locations_resp.locations[0].location_uuid

    forecasts_resp = await client.get_latest_forecasts(dp.GetLatestForecastsRequest(
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
    ))
    blend_forecast = next(
        (f for f in forecasts_resp.forecasts if f.forecaster.forecaster_name == blend_name),
        None,
    )
    assert blend_forecast is not None

    # Stream values and check the blend transitions from ecmwf (0 MW) toward xg (1 MW)
    stream = client.stream_forecast_data(dp.StreamForecastDataRequest(
        energy_source=dp.EnergySource.SOLAR,
        location_uuid=location_uuid,
        forecasters=blend_forecast.forecaster,
        time_window=dp.TimeWindow(
            start_timestamp_utc=init_time,
            end_timestamp_utc=init_time + datetime.timedelta(hours=13),
        ),
    ))
    values = [v async for v in stream]
    channel.close()

    assert len(values) > 0
    # Early steps should be near 0 (ecmwf-dominated), late steps near xg_fraction
    early_value = values[0].p50_fraction
    late_value = values[-1].p50_fraction
    assert late_value > early_value
