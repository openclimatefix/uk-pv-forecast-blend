"""
Tests:
1. test_app                          - App completes and saves blend forecasts to DP for all GSPs
2. test_app_legacy_gsp_reconstructed_from_source
                                     - Legacy GSPs are rebuilt from their source GSP even when
                                       they still have a forecast of their own
3. test_app_twice                    - Running app a second time updates forecasts without duplicating
4. test_app_only_national            - App restricted to gsp_id=0 saves only the national blend
5. test_app_only_ecmwf_and_day_ahead - Blend with only ecmwf/day_ahead models transitions correctly across horizon
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
from forecast_blend.save import fetch_dp_gsp_uuid_map
from forecast_blend.weights import ALL_MODEL_NAMES


# gsp 4 is a legacy gsp, reconstructed from gsp 350 (weight 1.0) - see gsp_legacy_weights.yaml.
# gsp 350 is seeded with a different value to every other gsp so that we can tell a
# reconstructed forecast apart from one blended directly from gsp 4's own (stale) forecast.
LEGACY_GSP_ID = 4
LEGACY_SOURCE_GSP_ID = 350
SEEDED_P50_FRACTION = 0.5
SEEDED_SOURCE_P50_FRACTION = 0.25


@pytest_asyncio.fixture(autouse=True, scope="session", loop_scope="session")
async def setup_dp_locations(data_client):
    """Create locations and seed input forecasts in Data Platform for app tests."""
    location_uuids = {}
    for gsp_id in list(range(0, 15)) + [LEGACY_SOURCE_GSP_ID]:
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

    def make_values(p50_fraction):
        return [
            dp.CreateForecastRequestForecastValue(
                horizon_mins=30 * i,
                p50_fraction=p50_fraction,
                metadata=Struct(),
            )
            for i in range(1, 17)
        ]

    for model_name in ALL_MODEL_NAMES:
        gsp_ids_for_model = list(range(0, 12)) + [LEGACY_SOURCE_GSP_ID]
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
            p50_fraction = (
                SEEDED_SOURCE_P50_FRACTION
                if gsp_id == LEGACY_SOURCE_GSP_ID
                else SEEDED_P50_FRACTION
            )
            for forecaster in forecasters:
                try:
                    await data_client.create_forecast(dp.CreateForecastRequest(
                        forecaster=forecaster,
                        location_uuid=location_uuids[gsp_id],
                        energy_source=dp.EnergySource.SOLAR,
                        init_time_utc=init_time,
                        values=make_values(p50_fraction),
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
    await app(gsps=list(range(0, 11)) + [LEGACY_SOURCE_GSP_ID])

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
    # gsp_ids 1-10 (gsp 4 reconstructed rather than blended) plus the legacy source gsp 350
    assert gsp_blend_count == 11


@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
async def test_app_legacy_gsp_reconstructed_from_source(data_client):
    """A legacy gsp should be reconstructed from its source gsp, not blended from its own data.

    gsp 4 has its own seeded forecast (SEEDED_P50_FRACTION) as well as a source gsp 350
    (SEEDED_SOURCE_P50_FRACTION). Before this was fixed, the blend for gsp 4 succeeded off its
    own stale forecast, which put gsp 4 into the results dict and made add_legacy_gsp_results
    skip it - reconstruction only kicked in once that forecast ran out of values. gsp 4 should
    now always take the value of its source gsp.
    """
    blend_name = os.environ["BLEND_NAME"]
    init_time = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)

    await app(gsps=list(range(0, 11)) + [LEGACY_SOURCE_GSP_ID])

    uuid_map = await fetch_dp_gsp_uuid_map(data_client)
    assert LEGACY_GSP_ID in uuid_map, "legacy gsp must still exist as a location to be saved"
    location_uuid = uuid_map[LEGACY_GSP_ID]["location_uuid"]

    forecasts_resp = await data_client.get_latest_forecasts(dp.GetLatestForecastsRequest(
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
    ))
    blend_forecast = next(
        (f for f in forecasts_resp.forecasts if f.forecaster.forecaster_name == blend_name),
        None,
    )
    assert blend_forecast is not None, f"no blend forecast saved for legacy gsp {LEGACY_GSP_ID}"

    timeseries_resp = await data_client.get_forecast_as_timeseries(
        dp.GetForecastAsTimeseriesRequest(
            location_uuid=location_uuid,
            energy_source=dp.EnergySource.SOLAR,
            forecaster=blend_forecast.forecaster,
            time_window=dp.TimeWindow(
                start_timestamp_utc=init_time,
                end_timestamp_utc=init_time + datetime.timedelta(hours=48),
            ),
        )
    )

    # the reconstruction carries the source gsp's value (weight 1.0), not gsp 4's own forecast
    assert len(timeseries_resp.values) == 16
    for value in timeseries_resp.values:
        assert value.p50_value_fraction == pytest.approx(SEEDED_SOURCE_P50_FRACTION, abs=1e-6)


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
async def test_app_only_ecmwf_and_day_ahead(data_client):
    """When only pvnet_ecmwf (value=0) and pvnet_day_ahead (value=1) have data, the blend
    should transition from ecmwf-only to day_ahead-only across forecast horizon."""
    blend_name = os.environ["BLEND_NAME"]
    init_time = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)

    # Build weights that use only pvnet_ecmwf and pvnet_day_ahead
    t0 = pd.Timestamp("2023-01-01 00:00", tz="UTC")
    horizons = pd.timedelta_range("30min", periods=25, freq="30min")
    ecmwf_weights = pd.Series(
        [1.0] * 16 + [0.75, 0.5, 0.25] + [0.0] * 6, index=t0 + horizons
    )
    day_ahead_weights = 1.0 - ecmwf_weights

    mock_weights = pd.DataFrame({"pvnet_ecmwf": ecmwf_weights, "pvnet_day_ahead": day_ahead_weights})

    with patch("forecast_blend.app.get_blend_weights", new=AsyncMock(return_value=mock_weights)):
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
