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
    gsp_id: int,
) -> pd.DataFrame:
    """Get forecast values from Data Platform.

    Args:
        client: Data Platform client
        location_uuid: Location UUID
        model_name: Model/forecaster name
        start_datetime: Optional start datetime filter
        gsp_id: GSP ID (used to determine if adjuster values should be loaded)

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

    # For national forecasts (gsp_id=0), fetch adjuster values to calculate adjust_mw
    adjuster_values_by_time = {}
    if gsp_id == 0:
        adjuster_model_name = f"{model_name}_adjust"
        adjuster_values = await fetch_dp_forecast_values(
            client=client,
            location_uuid=location_uuid,
            model_name=adjuster_model_name,
            start_datetime=start_datetime,
        )
        for v in adjuster_values:
            target_time = v.target_timestamp_utc.replace(microsecond=0)

            capacity_mw = v.effective_capacity_watts / 1_000_000
            adjuster_values_by_time[target_time] = (
                v.p50_value_fraction * capacity_mw
            )

    # Convert Data Platform forecast values directly to DataFrame rows
    rows = []
    now = datetime.now(timezone.utc)
    for dp_value in dp_values:
        properties = {}
        # Probabilistic values are stored in other_statistics_fractions dict
        # They need to be converted from fractions back to MW
        other_stats = getattr(dp_value, "other_statistics_fractions", {}) or {}
        capacity_mw = dp_value.effective_capacity_watts / 1_000_000
        target_time = dp_value.target_timestamp_utc.replace(microsecond=0)
        if "p10" in other_stats:
            properties["10"] = other_stats["p10"] * capacity_mw
        if "p90" in other_stats:
            properties["90"] = other_stats["p90"] * capacity_mw

        main_p50_mw = dp_value.p50_value_fraction * capacity_mw

        # Calculate adjust_mw for national forecasts (gsp_id=0)
        # adjust_mw = main_forecast - adjuster_forecast
        adjust_mw = 0.0
        if gsp_id == 0:
            adjuster_value = adjuster_values_by_time.get(target_time)

            if adjuster_value is not None:
                adjust_mw = main_p50_mw - adjuster_value
            else:
                logger.warning(
                    f"No adjuster value found for target_time {target_time} "
                    f"using model {adjuster_model_name}, defaulting adjust_mw to 0.0"
                )
        rows.append(
            {
                "target_time": dp_value.target_timestamp_utc,
                "expected_power_generation_megawatts": main_p50_mw,
                "adjust_mw": adjust_mw,
                "created_utc": now,
                "properties": properties,
                "model_name": model_name,
            }
        )

    df = pd.DataFrame(rows)
    logger.debug(f"Converted {len(df)} forecast values from Data Platform")
    return df
