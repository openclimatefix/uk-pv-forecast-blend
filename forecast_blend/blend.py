""" Blends forecasts together

1. get forecast values for each model
2. blends them together

"""

import asyncio
import json
from datetime import datetime
from loguru import logger

import pandas as pd
from sqlalchemy.orm.session import Session

from forecast_blend.save import fetch_dp_gsp_uuid_map
from forecast_blend.utils import (
    blend_forecasts_together,
    convert_list_forecast_values_to_df,
    get_data_platform_connection,
    read_from_data_platform,
)

from forecast_blend.forecast.data_platform import get_forecast_values_from_data_platform
from forecast_blend.forecast.database import get_forecast_values_from_db

from grpclib.client import Channel
from dp_sdk.ocf import dp



async def get_blend_forecast_values_latest(
    session: Session,
    gsp_id: int,
    weights_df: pd.DataFrame,
    start_datetime: datetime | None = None,
    gsp_uuid_map: dict | None = None,
    dp_client: dp.DataPlatformDataServiceStub | None = None,
) -> pd.DataFrame:
    """
    Get forecast values

    :param session: database session
    :param gsp_id: gsp id, to filter query on
    :param weights_df: dataframe of weights to use for blending,
        see structure in weights.py - get_national_blend_weights
    :param start_datetime: optional to filterer target_time by start_datetime
        If None is given then all are returned.
    :param gsp_uuid_map: optional pre-fetched GSP UUID map from Data Platform
    :param dp_client: optional pre-created Data Platform client

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
    forecast_values_all_model_dfs = []
    if read_from_data_platform():

        # Use passed-in client and map, or create new ones if not provided
        client = dp_client
        local_channel = None
        if client is None:
            host, port = get_data_platform_connection()
            local_channel = Channel(host=host, port=port)
            client = dp.DataPlatformDataServiceStub(local_channel)

        try:
            # Use passed-in map or fetch it
            uuid_map = gsp_uuid_map
            if uuid_map is None:
                uuid_map = await fetch_dp_gsp_uuid_map(client)

            if gsp_id not in uuid_map:
                raise ValueError(f"GSP {gsp_id} not found in Data Platform")

            location_uuid = uuid_map[gsp_id]["location_uuid"]

            tasks = [
                get_forecast_values_from_data_platform(
                    client=client,
                    location_uuid=location_uuid,
                    model_name=model_name,
                    start_datetime=start_datetime,
                )
                for model_name in model_names
            ]

            results = await asyncio.gather(*tasks)

            # Results are DataFrames, collect non-empty ones
            for df in results:
                if len(df) > 0:
                    forecast_values_all_model_dfs.append(df)
                else:
                    model_name = df["model_name"].iloc[0] if len(df) > 0 else "unknown"
                    logger.debug(
                        f"No forecast values for {model_name} "
                        f"for gsp_id {gsp_id}"
                    )
        finally:
            # Close the channel only if we created it locally
            if local_channel is not None:
                local_channel.close()

        # Concatenate all DataFrames
        if forecast_values_all_model_dfs:
            forecast_values_all_model = pd.concat(forecast_values_all_model_dfs, axis=0)
            forecast_values_all_model.reset_index(inplace=True, drop=True)
            forecast_values_all_model.sort_values(by=["target_time"], inplace=True)
        else:
            forecast_values_all_model = pd.DataFrame()

    else:
        forecast_values_all_model_list = []
        for model_name in model_names:
            forecast_values_one_model = get_forecast_values_from_db(
                session=session,
                gsp_id=gsp_id,
                start_datetime=start_datetime,
                model_name=model_name,
            )

            if len(forecast_values_one_model) > 0:
                forecast_values_all_model_list.append(
                    [model_name, forecast_values_one_model]
                )
            else:
                logger.debug(
                    f"No forecast values for {model_name} "
                    f"for gsp_id {gsp_id}"
                )

        # make into dataframe
        forecast_values_all_model = convert_list_forecast_values_to_df(forecast_values_all_model_list)

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

