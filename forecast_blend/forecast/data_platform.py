"""data platform interactions for forecasts."""
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
from loguru import logger
from dp_sdk.ocf import dp

def read_from_data_platform() -> bool:
    """Check if we should read from Data Platform instead of database."""
    result = os.getenv("READ_FROM_DATA_PLATFORM", "false").lower() == "true"
    return result


def get_data_platform_connection() -> tuple[str, int]:
    """Get the Data Platform host and port from environment variables."""
    host = os.getenv("DATA_PLATFORM_HOST", "localhost")
    port = int(os.getenv("DATA_PLATFORM_PORT", "50051"))
    return host, port

async def fetch_dp_forecast_values(
    client: dp.DataPlatformDataServiceStub,
    location_uuid: str,
    model_name: str,
    start_datetime: datetime | None,
):
    """Fetch forecast values from Data Platform for a specific location and model."""
    logger.debug(
        f"Fetching forecasts from Data Platform "
        f"for location {location_uuid} and model {model_name}"
    )
    # Get the latest forecasts for this location
    response = await client.get_latest_forecasts(
        dp.GetLatestForecastsRequest(
            location_uuid=location_uuid,
            energy_source=dp.EnergySource.SOLAR,
        )
    )

    logger.debug(f"Received {len(response.forecasts)} forecasts from Data Platform")

    # Filter by model name (forecaster tag)
    matching_forecasts = [
        f for f in response.forecasts if f.forecaster.forecaster_name == model_name
    ]

    if not matching_forecasts:
        logger.warning(
            f"No forecasts found for model '{model_name}' at location {location_uuid}"
        )
        return []

    # Take the most recent forecast
    forecast = matching_forecasts[0]

    timeseries_response = await client.get_forecast_as_timeseries(
        dp.GetForecastAsTimeseriesRequest(
            location_uuid=location_uuid,
            energy_source=dp.EnergySource.SOLAR,
            forecaster=forecast.forecaster,
            time_window=dp.TimeWindow(
                start_timestamp_utc=forecast.initialization_timestamp_utc,
                end_timestamp_utc=forecast.initialization_timestamp_utc
                + timedelta(hours=48),
            ),
        )
    )

    forecast_values = timeseries_response.values
    if start_datetime:
        # Ensure start_datetime is timezone-aware for comparison
        if start_datetime.tzinfo is None:
            start_datetime = start_datetime.replace(tzinfo=timezone.utc)
        filtered_values = []
        for v in forecast_values:
            target_time = v.target_timestamp_utc
            if target_time.tzinfo is None:
                target_time = target_time.replace(tzinfo=timezone.utc)
            if target_time >= start_datetime:
                filtered_values.append(v)
        forecast_values = filtered_values

    return forecast_values


async def get_forecast_values_from_data_platform(
    client,
    location_uuid: str,
    model_name: str,
    start_datetime: datetime | None,
) -> pd.DataFrame:
    """Get forecast values from Data Platform.

    Args:
        client: Data Platform client
        location_uuid: Location UUID
        model_name: Model/forecaster name
        start_datetime: Optional start datetime filter

    Returns:
        DataFrame with columns: target_time, expected_power_generation_megawatts,
        adjust_mw, created_utc, properties, model_name
    """
    logger.debug(
        f"Getting forecast values from Data Platform for location {location_uuid}, "
        f"model {model_name}, start_datetime {start_datetime}"
    )

    dp_values = await fetch_dp_forecast_values(
        client=client,
        location_uuid=location_uuid,
        model_name=model_name,
        start_datetime=start_datetime,
    )

    if not dp_values:
        logger.debug(f"No forecast values from Data Platform for model {model_name}")
        return pd.DataFrame(
            columns=[
                "target_time",
                "expected_power_generation_megawatts",
                "adjust_mw",
                "created_utc",
                "properties",
                "model_name",
            ]
        )

    # Convert Data Platform forecast values directly to DataFrame rows
    rows = []
    now = datetime.now(timezone.utc)
    for dp_value in dp_values:
        properties = {}
        if hasattr(dp_value, "plevel_10") and dp_value.plevel_10:
            properties["10"] = dp_value.plevel_10
        if hasattr(dp_value, "plevel_90") and dp_value.plevel_90:
            properties["90"] = dp_value.plevel_90

        rows.append(
            {
                "target_time": dp_value.target_timestamp_utc,
                "expected_power_generation_megawatts": dp_value.p50_value_fraction
                * dp_value.effective_capacity_watts
                / 1_000_000,
                # TODO: load adjuster values from the Data Platform
                # using the model "{model_name}_adjust". This should only apply for
                # national forecasts (gsp_id == 0).
                # Currently adjust_mw is set to 0 because DP does not return it directly.
                "adjust_mw": 0, 
                "created_utc": now,
                "properties": properties,
                "model_name": model_name,
            }
        )

    df = pd.DataFrame(rows)
    logger.debug(f"Converted {len(df)} forecast values from Data Platform")
    return df
