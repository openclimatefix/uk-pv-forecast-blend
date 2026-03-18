import datetime

import pandas as pd
import pytest
from betterproto.lib.google.protobuf import Struct, Value
from dp_sdk.ocf import dp

from forecast_blend.save import save_forecast_to_data_platform


@pytest.mark.asyncio(loop_scope="session")
async def test_save_to_generation_to_data_platform(client):
    """
    Test saving data to the Data Platform.
    This test uses the `data_platform` fixture to ensure that the Data Platform service
    is running and can accept data.
    """
    # setup: add location - gsp 0
    metadata = Struct(fields={"gsp_id": Value(number_value=0)})
    create_location_request = dp.CreateLocationRequest(
        location_name="gsp0",
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POINT(0 0)",
        location_type=dp.LocationType.GSP,
        effective_capacity_watts=1_000_000,
        metadata=metadata,
        valid_from_utc=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
    )
    create_location_response = await client.create_location(create_location_request)
    location_uuid = create_location_response.location_uuid

    # setup: make fake data
    fake_data = pd.DataFrame(
        {
            "p10_mw": [0.3] * 24,
            "p50_mw": [0.5] * 24,
            "p90_mw": [0.7] * 24,
            "adjust_mw": [0.1] * 24,
            "target_datetime_utc": pd.Timestamp("2025-01-01")
            + pd.timedelta_range(
                start=0,
                periods=24,
                freq="30min",
            ),
        },
    )

    # Test the functyion
    _ = await save_forecast_to_data_platform(
        forecast_values_by_gsp_id={0: fake_data},
        locations_uuid_and_capacity_by_gsp_id={0: {'location_uuid': location_uuid, 'effective_capacity_watts': 1_000_000}},
        client=client,
        model_tag="test_model",
        init_time_utc=datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC),
        metadata=Struct(fields={"test_key": Value(string_value="test_value")})
    )

    # check: read from the data platform to check it was saved
    list_forecasters_response = await client.list_forecasters(dp.ListForecastersRequest())
    forecaster_names = {f.forecaster_name for f in list_forecasters_response.forecasters}
    assert "test_model" in forecaster_names
    assert "test_model_adjust" in forecaster_names

    # check: There is a forecast object
    get_latest_forecasts_request = dp.GetLatestForecastsRequest(
        energy_source=dp.EnergySource.SOLAR,
        pivot_timestamp_utc=datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC),
        location_uuid=location_uuid,
    )
    get_latest_forecasts_response = await client.get_latest_forecasts(
        get_latest_forecasts_request,
    )
    assert len(get_latest_forecasts_response.forecasts) == 2
    forecast = get_latest_forecasts_response.forecasts[0]
    assert forecast.forecaster.forecaster_name == "test_model"
    assert forecast.metadata.fields["test_key"].string_value == "test_value"
    forecast_adjust = get_latest_forecasts_response.forecasts[1]
    assert forecast_adjust.forecaster.forecaster_name == "test_model_adjust"

    # check: the number of forecast values
    stream_forecast_data_request = dp.StreamForecastDataRequest(
        energy_source=dp.EnergySource.SOLAR,
        location_uuid=location_uuid,
        forecasters=forecast.forecaster,
        time_window=dp.TimeWindow(
            start_timestamp_utc=datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC),
            end_timestamp_utc=datetime.datetime(2025, 1, 2, tzinfo=datetime.UTC),
        ),
    )
    stream_forecast_data_response = client.stream_forecast_data(
        stream_forecast_data_request,
    )
    count = 0
    async for d in stream_forecast_data_response:
        assert d.p50_fraction == 0.5
        count += 1
    assert count == 24
