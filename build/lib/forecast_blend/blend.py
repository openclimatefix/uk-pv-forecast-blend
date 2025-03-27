""" Blends forecasts together

1. creates weights for blending
2. get forecast values for each model
3. blends them together

"""

import json
from datetime import datetime
from typing import List, Optional

import pandas as pd
import structlog
from nowcasting_datamodel.models.forecast import (
    ForecastValue,
    ForecastValueSevenDaysSQL,
)
from nowcasting_datamodel.read.read import (
    get_forecast_values,
    get_forecast_values_latest,
)
from sqlalchemy.orm.session import Session

from utils import (
    blend_forecasts_together,
    check_forecast_created_utc,
    convert_df_to_list_forecast_values,
    convert_list_forecast_values_to_df,
)
from weights import make_weights_df

logger = structlog.stdlib.get_logger()


def get_blend_forecast_values_latest(
    session: Session,
    gsp_id: int,
    start_datetime: Optional[datetime] = None,
    model_names: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    forecast_horizon_minutes: Optional[int] = None,
) -> List[ForecastValue]:
    """
    Get forecast values

    :param session: database session
    :param gsp_id: gsp id, to filter query on
    :param start_datetime: optional to filterer target_time by start_datetime
        If None is given then all are returned.
    :param model_names: list of model names to use for blending
    :param weights: list of weights to use for blending, see structure in make_weights_df
    :param forecast_horizon_minutes: The forecast horizon to blend together

    return: List of forecasts values blended from different models
    """

    logger.info(
        f"Getting blend forecast for gsp_id {gsp_id} and start_datetime {start_datetime}"
    )

    if model_names is None:
        model_names = ["cnn", "National_xg"]
    if len(model_names) > 1:
        weights_df = make_weights_df(
            model_names,
            weights,
            start_datetime,
            forecast_horizon_minutes=forecast_horizon_minutes,
        )
    else:
        weights_df = None

    # get forecast for the different models
    forecast_values_all_model = []
    for model_name in model_names:
        if forecast_horizon_minutes is None:
            forecast_values_one_model = get_forecast_values_latest(
                session=session,
                gsp_id=gsp_id,
                start_datetime=start_datetime,
                model_name=model_name,
            )
        else:
            forecast_values_one_model = get_forecast_values(
                session=session,
                gsp_ids=[gsp_id],
                start_datetime=start_datetime,
                only_return_latest=True,
                forecast_horizon_minutes=forecast_horizon_minutes,
                model=ForecastValueSevenDaysSQL,
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

    # check the created_utc is valid for each forecast
    forecast_values_all_model_valid = check_forecast_created_utc(
        forecast_values_all_model
    )

    # This is a hard code fix, and soon we plan to improve blending, by using dynamic weights from the
    # MAE results from the ML experiments #39
    # if pvnet_v2 is not in model names, then change the weights for pvnet_v2 to pvnet_ecmwf
    if "pvnet_v2" not in [model_name for model_name, _ in forecast_values_all_model_valid]:
        logger.warning(f"Changing pvnet_v2 to pvnet_ecmwf in weights_df, as pvnet v2 model has not be loaded, "
                       f"or is out of date. This is a hard code fix, and soon we plan to improve blending. ")
        if (weights_df is not None) and ("pvnet_v2" in weights_df.columns):
            weights_df.rename(columns={"pvnet_v2": "pvnet_ecmwf"}, inplace=True)

    # make into dataframe
    forecast_values_all_model = convert_list_forecast_values_to_df(
        forecast_values_all_model_valid
    )

    # blend together
    forecast_values_blended = blend_forecasts_together(
        forecast_values_all_model, weights_df
    )

    # add properties
    if gsp_id == 0:
        # currently only blend for national
        forecast_values_blended = add_p_levels_to_forecast_values(
            blended_df=forecast_values_blended,
            all_model_df=forecast_values_all_model,
            weights_df=weights_df,
        )

    # convert back to list of forecast values
    forecast_values = convert_df_to_list_forecast_values(forecast_values_blended)

    return forecast_values


def add_p_levels_to_forecast_values(
    blended_df: pd.DataFrame,
    all_model_df: pd.DataFrame,
    weights_df: pd.DataFrame,
):
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

    logger.debug(f"Adding properties to blended forecast values")
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

    else:
        # If blended_on_p_values is None, assign an empty dictionary to properties
        blended_df["properties"] = [{}] * len(blended_df)

    return blended_df

