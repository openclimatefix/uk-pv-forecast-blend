""" Blends forecasts together

1. get forecast values for each model
2. blends them together

"""

import json
from datetime import datetime, timedelta
from loguru import logger

import pandas as pd
from nowcasting_datamodel.read.read import get_forecast_values_latest
from nowcasting_datamodel.models import ForecastValue
from sqlalchemy.orm.session import Session

from forecast_blend.utils import (
    blend_forecasts_together,
    convert_list_forecast_values_to_df,
)

import os

from grpclib.client import Channel
from dp_sdk.ocf import dp


def read_from_data_platform() -> bool:
    return os.getenv("READ_FROM_DATA_PLATFORM", "false").lower() == "true"

async def _fetch_dp_forecast_values(
    location_uuid: str,
    model_name: str,
    start_datetime: datetime | None,
):
    """Fetch forecast values from Data Platform for a specific location and model."""
    host = os.getenv("DATA_PLATFORM_HOST", "localhost")
    port = int(os.getenv("DATA_PLATFORM_PORT", "50051"))

    logger.debug(
        f"Fetching forecasts from Data Platform at {host}:{port} "
        f"for location {location_uuid} and model {model_name}"
    )

    async with Channel(host=host, port=port) as channel:
        client = dp.DataPlatformDataServiceStub(channel)
        
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
            f for f in response.forecasts 
            if f.forecaster.forecaster_name == model_name
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
                    end_timestamp_utc=forecast.initialization_timestamp_utc + timedelta(hours=48),
                )
            )
        )

        forecast_values = timeseries_response.values
        if start_datetime:
            forecast_values = [
                v for v in forecast_values
                if v.target_timestamp_utc.replace(tzinfo=None) >= start_datetime
            ]
        
        return forecast_values


async def _fetch_dp_gsp_location_uuid(gsp_id: int):
    """Fetch the location UUID for a given GSP ID from Data Platform."""
    host = os.getenv("DATA_PLATFORM_HOST", "localhost")
    port = int(os.getenv("DATA_PLATFORM_PORT", "50051"))

    async with Channel(host=host, port=port) as channel:
        client = dp.DataPlatformDataServiceStub(channel)
        
        # List all locations and find the one matching the GSP ID
        response = await client.list_locations(
            dp.ListLocationsRequest(
                energy_source_filter=dp.EnergySource.SOLAR,
                location_type_filter=dp.LocationType.GSP,
            )
        )
        
        for location in response.locations:
            # Check if metadata contains gsp_id
            metadata = location.metadata

            if metadata and hasattr(metadata, "fields"):
                if "gsp_id" in metadata.fields:
                    gsp_field = metadata.fields["gsp_id"]
                    location_gsp_id = int(gsp_field.number_value)

                    if location_gsp_id == gsp_id:
                        return location.location_uuid

        
        raise ValueError(f"No location found for GSP ID {gsp_id} in Data Platform")


def _get_forecast_values_from_db(
    session: Session,
    gsp_id: int,
    model_name: str,
    start_datetime: datetime | None,
):
    """Get forecast values from the database (nowcasting_datamodel)."""
    return get_forecast_values_latest(
        session=session,
        gsp_id=gsp_id,
        start_datetime=start_datetime,
        model_name=model_name,
    )


async def _get_forecast_values_from_data_platform(
    gsp_id: int,
    model_name: str,
    start_datetime: datetime | None,
):
    """Get forecast values from Data Platform."""
    logger.debug(
        f"Getting forecast values from Data Platform for GSP {gsp_id}, "
        f"model {model_name}, start_datetime {start_datetime}"
    )
    
    # Get location UUID for this GSP
    location_uuid = await _fetch_dp_gsp_location_uuid(gsp_id)

    dp_values = await _fetch_dp_forecast_values(
        location_uuid=location_uuid,
        model_name=model_name,
        start_datetime=start_datetime,
)


    # Convert Data Platform forecast values to ForecastValue objects
    forecast_values = []
    for dp_value in dp_values:
        # Convert from Data Platform format to nowcasting_datamodel format
        # Note: adjust_mw might not be available in DP, defaulting to 0
        properties = {}
        if hasattr(dp_value, 'plevel_10') and dp_value.plevel_10:
            properties['10'] = dp_value.plevel_10
        if hasattr(dp_value, 'plevel_90') and dp_value.plevel_90:
            properties['90'] = dp_value.plevel_90
            
        forecast_values.append(
            ForecastValue(
                target_time=dp_value.target_timestamp_utc.replace(tzinfo=None),
                expected_power_generation_megawatts = dp_value.p50_value_fraction * dp_value.effective_capacity_watts/ 1_000_000,
                adjust_mw=0,  # adjust_mw is not provided by DP, setting to 0
                properties=properties,
            )
        )

    logger.debug(f"Converted {len(forecast_values)} forecast values from Data Platform")
    return forecast_values


async def get_blend_forecast_values_latest(
    session: Session,
    gsp_id: int,
    weights_df: pd.DataFrame,
    start_datetime: datetime | None = None,
) -> pd.DataFrame:
    """
    Get forecast values

    :param session: database session
    :param gsp_id: gsp id, to filter query on
    :param weights_df: dataframe of weights to use for blending,
        see structure in weights.py - get_national_blend_weights
    :param start_datetime: optional to filterer target_time by start_datetime
        If None is given then all are returned.

    return: DataFrame of blended forecast values, with the following columns
            - target_datetime_utc
            - p50_mw
            - adjust_mw
            - p10_mw (if available)
            - p90_mw (if available)
    """

    logger.info(
        f"Getting blend forecast for gsp_id {gsp_id} and start_datetime {start_datetime}"
    )

    model_names = weights_df.columns

    # get forecast for the different models
    forecast_values_all_model = []
    for model_name in model_names:
        if read_from_data_platform():
            forecast_values_one_model = await _get_forecast_values_from_data_platform(
                gsp_id=gsp_id,
                start_datetime=start_datetime,
                model_name=model_name,
            )
        else:
            forecast_values_one_model = _get_forecast_values_from_db(
                session=session,
                gsp_id=gsp_id,
                start_datetime=start_datetime,
                model_name=model_name,
            )


        if len(forecast_values_one_model) == 0:
            logger.debug(
                f"No forecast values for {model_name} for gsp_id {gsp_id} "
                f"and start_datetime {start_datetime}"
            )
        else:
            logger.debug(
                f"Found {len(forecast_values_one_model)} values for {model_name} "
                f"for gsp_id {gsp_id} and start_datetime {start_datetime}. "
                f"First value is {forecast_values_one_model[0].target_time}"
            )
            forecast_values_all_model.append([model_name, forecast_values_one_model])

    # make into dataframe
    forecast_values_all_model = convert_list_forecast_values_to_df(forecast_values_all_model)

    # blend together
    forecast_values_blended = blend_forecasts_together(forecast_values_all_model, weights_df)

    # add properties
    if gsp_id == 0:
        # currently only blend for national
        forecast_values_blended = add_p_levels_to_forecast_values(
            blended_df=forecast_values_blended,
            all_model_df=forecast_values_all_model,
            weights_df=weights_df,
        )

    # rename to dataframe columns
    forecast_values_blended = forecast_values_blended.rename(
        columns={
            "target_time": "target_datetime_utc",
            "expected_power_generation_megawatts": "p50_mw",
        }
    )

    return forecast_values_blended


def add_p_levels_to_forecast_values(
    blended_df: pd.DataFrame,
    all_model_df: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add properties to blended forecast values.

    We normalize all properties by the "expected_power_generation_megawatts" value,
    and renormalize by the blended "expected_power_generation_megawatts" value.
    This makes sure that plevels 10 and 90 surround the blended value.

    :param blended_df: DataFrame of blended forecast values
    :param all_model_df: DataFrame of all forecast values for all models
    :param weights_df: DataFrame of weights for each model
    :return: DataFrame of blended forecast values with added properties
    """

    logger.debug("Adding properties to blended forecast values")
    all_model_df.reset_index(inplace=True, drop=True)

    # get properties out of json if they exist
    if "properties" in all_model_df.columns:
        properties_only_df = pd.json_normalize(all_model_df["properties"])
        properties_only_df.rename(
            columns={"10": "plevel_10", "90": "plevel_90"}, inplace=True
        )
    else:
        # If properties are missing, create empty columns
        properties_only_df = pd.DataFrame(columns=["plevel_10", "plevel_90"])

    properties_only_df = pd.concat(
        [all_model_df[["target_time", "model_name", "adjust_mw"]], properties_only_df],
        axis=1,
    )
    properties_only_df.reset_index(inplace=True, drop=True)

    logger.debug(f"properties_only_df {properties_only_df}")

    assert (
        "target_time" in properties_only_df.columns
    ), f"target_time must be in properties_only_df {properties_only_df.columns}"

    # blend together the p values if they exist
    blended_on_p_values = None
    for p_level in ["plevel_10", "plevel_90"]:
        if p_level in properties_only_df.columns:
            blended_on_p_value = blend_forecasts_together(
                properties_only_df, weights_df, column_name_to_blend=p_level
            )
            blended_on_p_value = blended_on_p_value[["target_time", p_level]]

            # join all plevels back together in `blended_on_p_values`
            if blended_on_p_values is None:
                blended_on_p_values = blended_on_p_value
            else:
                blended_on_p_values = blended_on_p_values.merge(
                    blended_on_p_value, on="target_time", how="outer"
                )

    if blended_on_p_values is not None:
        # add properties to blended forecast values if they exist
        blended_df = blended_df.merge(blended_on_p_values, on=["target_time"], how="left")

        # format plevels back to dict
        blended_df.rename(columns={"plevel_10": "10", "plevel_90": "90"}, inplace=True)
        blended_df["properties"] = blended_df[["10", "90"]].apply(
            lambda x: json.loads(x.to_json()) if pd.notnull(x).all() else {}, axis=1
        )

        # rename
        blended_df.rename(columns={"10": "p10_mw", "90": "p90_mw"}, inplace=True)

    else:
        # If blended_on_p_values is None, assign an empty dictionary to properties
        blended_df["properties"] = [{}] * len(blended_df)

    

    return blended_df

