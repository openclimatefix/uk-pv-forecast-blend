"""Functions to make weights for blending"""

import asyncio
import numpy as np
import os
import pandas as pd
from collections.abc import Collection
from typing import Awaitable, Callable
from datetime import timezone
from loguru import logger

from ocf import dp


DAY_AHEAD_MODEL_NAMES = frozenset({"pvnet_ukv_only", "pvnet_day_ahead", "pvnet_ecmwf"})
INTRADAY_MODEL_NAMES = frozenset({"pvnet_v2", "pvnet_cloud"})
ALL_MODEL_NAMES = DAY_AHEAD_MODEL_NAMES | INTRADAY_MODEL_NAMES

BLEND_KERNEL = [0.75, 0.5, 0.25]
MIN_FORECAST_HORIZON = pd.Timedelta("30min")


def get_horizon_maes() -> pd.DataFrame:
    """Loads Mean Absolute Error (MAE) curves for different forecast horizons from a CSV file

    Returns:
        A pandas DataFrame with pd.Timedelta horizons as the index and MAEs per model
        in the columns.
    """
    filepath = os.path.dirname(os.path.realpath(__file__)) + '/data/model_horizon_mae_scores.csv'
    df_maes = pd.read_csv(filepath, index_col=0).reset_index(names="horizon")
    df_maes["horizon"] = pd.to_timedelta(df_maes["horizon"]).to_numpy()
    df_maes = df_maes.set_index("horizon")

    # ----------------------------------------------------------------------------------------------
    # NOTE: This is a hack which sets the pvnet_cloud model MAE to 10% higher than the pvnet_v2
    # model for the first (30 minutes) horizon. This is a temporary fix for some production 
    # weirdness
    # TODO: Remove when no longer required
    h_limit = pd.Timedelta("30min")
    df_maes.loc[h_limit, "pvnet_cloud"] = df_maes.loc[h_limit, "pvnet_v2"] * 1.1
    # ----------------------------------------------------------------------------------------------

    return df_maes


async def _list_national_locations(client: dp.DataPlatformDataServiceStub) -> list:
    """List solar NATION locations from the Data Platform."""
    response = await client.list_locations(
        dp.ListLocationsRequest(
            energy_source_filter=dp.EnergySource.SOLAR,
            location_type_filter=dp.LocationType.NATION,
        )
    )
    return response.locations


async def _list_gsp_locations(client: dp.DataPlatformDataServiceStub) -> list:
    """List solar GSP locations from the Data Platform."""
    response = await client.list_locations(
        dp.ListLocationsRequest(
            energy_source_filter=dp.EnergySource.SOLAR,
            location_type_filter=dp.LocationType.GSP,
        )
    )
    return response.locations


async def _fetch_latest_forecast_metadata_from_dp(
    client: dp.DataPlatformDataServiceStub,
    model_names: set[str],
    t0: pd.Timestamp,
    max_delay: pd.Timedelta,
    list_locations_fn: Callable[[dp.DataPlatformDataServiceStub], Awaitable[list]],
) -> pd.DataFrame:

    """Gets the most recent forecast metadata from Data Platform.

    Args:
        model_names: Model names (forecaster tags) to filter to
        t0: The blend forecast init-time
        max_delay: Max delay with respect to t0 to consider using forecast in the blend
        list_locations_fn: Async function returning the locations to fetch forecasts for
            (e.g. `_list_national_locations` or `_list_gsp_locations`)

    Returns:
        A pandas DataFrame with columns 'created_utc', 'forecast_creation_time',
        'location_id', and 'name' (model name) representing the latest forecasts
    """

    t0_datetime = t0.to_pydatetime().replace(tzinfo=timezone.utc)
    earliest_creation_time = t0_datetime - max_delay

    locations = await list_locations_fn(client)

    tasks = [
        asyncio.create_task(
            client.get_latest_forecasts(
                dp.GetLatestForecastsRequest(
                    location_uuid=location.location_uuid,
                    energy_source=dp.EnergySource.SOLAR,
                )
            )
        )
        for location in locations
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    rows = []
    for response in responses:
        try:
            for forecast in response.forecasts:
                forecaster_name = forecast.forecaster.forecaster_name

                if forecaster_name not in model_names:
                    continue

                init_time = forecast.initialization_timestamp_utc

                if init_time and init_time >= earliest_creation_time:
                    rows.append({
                        "created_utc": init_time,
                        "forecast_creation_time": init_time,
                        "location_id": forecast.location_uuid,
                        "name": forecaster_name,
                    })

        except Exception as e:
            logger.warning(f"Skipping failed forecast response: {e}")
            continue

    if not rows:
        logger.warning("No forecast metadata found from Data Platform")
        return pd.DataFrame(columns=['created_utc', 'forecast_creation_time', 'location_id', 'name'])

    df = pd.DataFrame(rows)
    df["forecast_creation_time"] = pd.to_datetime(df["forecast_creation_time"], utc=True)
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    return df


async def get_latest_forecast_metadata(
    client: dp.DataPlatformDataServiceStub,
    model_names: set[str],
    t0: pd.Timestamp,
    max_delay: pd.Timedelta,
    location_type: str,
) -> pd.DataFrame:
    """Get latest forecast metadata from Data Platform.

    Args:
        client: Data Platform stub to use for requests
        model_names: Model names to filter to
        t0: The blend forecast init-time
        max_delay: Max delay with respect to t0 to consider using forecast in the blend
        location_type: "national" or "regional" — selects which locations to list

    Returns:
        A pandas DataFrame with forecast metadata
    """
    if location_type == "national":
        list_locations_fn = _list_national_locations
    elif location_type == "regional":
        list_locations_fn = _list_gsp_locations
    else:
        raise ValueError(
            f"Unknown location_type: {location_type!r}. Expected 'national' or 'regional'."
        )

    logger.info("Reading forecast metadata from Data Platform")
    return await _fetch_latest_forecast_metadata_from_dp(
        client=client,
        model_names=model_names,
        t0=t0,
        max_delay=max_delay,
        list_locations_fn=list_locations_fn,
    )


def get_model_delays(df_forecast_ids: pd.DataFrame, t0: pd.Timestamp) -> dict[str, pd.Timedelta]:
    """Construct a dictionary of how delayed each model is
    
    Args:
        df_forecast_ids: DataFrame of the most recent forecast IDs for each model and location. 
            See `get_latest_forecast_metadata()`.
        t0: The blend forecast init time
    
    """
    
    # Filter to the most recent forecast ID per model
    df_forecast_ids = df_forecast_ids.loc[
        df_forecast_ids.groupby("name")["forecast_creation_time"].idxmax()
    ]
    
    # Approximate the forecast initialisation time
    df_forecast_ids["initialisation_datetime_utc"] = (
        df_forecast_ids["forecast_creation_time"].dt.floor("30min")
    )
    
    df_forecast_ids["delay"] = t0 - df_forecast_ids["initialisation_datetime_utc"]
    
    return dict(df_forecast_ids.set_index("name").delay)


def delay_horizon_maes(
    df_mae: pd.DataFrame, 
    delays_dict: dict[str, pd.Timedelta]
) -> pd.DataFrame:
    """Construct the expected horizon-MAE for each model run with given delays
    
    Args:
        df_mae: The horizon-MAE results for each model without runtime delays
        delays_dict: The current delay of each model

    Returns:
        A DataFrame of horizon-MAE values for each model run with the given delays
    """
    
    mae_models = set(df_mae.columns)
    delay_models = set(delays_dict.keys())
    
    if len(mae_models - delay_models)>0:
        skipped_models = mae_models - delay_models
        logger.warning(
            f"The following models have saved MAES but were not available {skipped_models}. "
            "These models will be skipped in consideration for the blend."
        
        )
    if len(delay_models - mae_models)>0:
        skipped_models = delay_models - mae_models
        logger.warning(
            f"The following models are available but do not have saved MAEs {skipped_models}. "
            "These models will be skipped in consideration for the blend."
        )
    
    delayed_model_horizon_mae_dfs = []

    for model_name in mae_models.intersection(delay_models):

        df = pd.DataFrame(
            df_mae[model_name].values,
            index=df_mae.index - delays_dict[model_name],
            columns=[model_name],
        )

        delayed_model_horizon_mae_dfs.append(df)

    df_delayed_maes = pd.concat(delayed_model_horizon_mae_dfs, axis=1).sort_index()
    
    return df_delayed_maes.loc[MIN_FORECAST_HORIZON:].dropna(axis=0, how="all")


def make_avg_mae_func(n_hours: int) -> Callable[[pd.Series], float]:
    """Make function to calculate average MAE over the first N hours of forecast horizon
    
    Args:
        hours: The number of hours to average over

    Returns:
        A function which calculates the average MAE over the first N hours of forecast horizon
    """

    def calculate_nhr_avg_mae(horizon_mae: pd.Series) -> float:
        """Calculate the average MAE over the first N hours of forecast horizon
        
        Args:
            horizon_mae: The expected horizon-MAE values
        """
        return horizon_mae.loc[MIN_FORECAST_HORIZON:f"{n_hours}h"].iloc[:-1].mean()
    
    return calculate_nhr_avg_mae


def make_blend_weights_array(size: int, blend_start_index: int, kernel: list[float]) -> np.ndarray:
    """Construct a numpy array of blend weights.
    
    Args:
        size: The length of the array
        blend_start_index: The index where the blend starts
        kernel: The values used during the blend steps
        
    Examples:
        >>> make_blend_weights_array(8, 1, [.75, 0.5, 0.25])
        array([1.  , 0.75, 0.5 , 0.25, 0.  , 0.  , 0.  , 0.  ])
        
        >>> make_blend_weights_array(6, 1, [])
        array([1., 0., 0., 0., 0., 0.])
    """

    k_size = len(kernel)

    if blend_start_index <= -k_size:
        raise ValueError(
            "The blend_start_index chosen sets the entire blend kernel to be before the first "
            "forecast step. Please choose a larger blend_start_index."
        )
    if blend_start_index >= size:
        raise ValueError(
            "The blend_start_index chosen sets the entire blend kernel to be after the last "
            "forecast step. Please choose a smaller blend_start_index."
        )

    start_index = max(blend_start_index, 0)
    end_index = min(blend_start_index + k_size, size)

    weights = np.zeros(size)
    weights[:start_index] = 1
    kernel_slice = slice(start_index - blend_start_index, end_index - blend_start_index)
    weights[start_index:end_index] = kernel[kernel_slice]
    return weights


def index_of_last_non_nan_value(x: np.ndarray) -> int:
    """Find the index position of the last non-NaN value in the array

    Args:
        x: The input array
    """
    if np.isnan(x).any():
        return int(np.isnan(x).argmax())-1
    else:
        return len(x) - 1


def infer_backup_model_name(df_mae: pd.DataFrame) -> str:
    """Infer the backup model name from the DataFrame of horizon-MAE values
    
    The backup model is the one which has the longest forecast horizon in the input DataFrame. Ties 
    are broken by lowest average MAE.

    Args:
        df_mae: DataFrame of horizon-MAE values for all models under consideration. These should 
            have been modified for the delay of the latest model run.
    
    Returns:
        The name of the backup model
    """

    # Store the model names, max horizon, and average MAEs
    candidates: list[tuple[str, pd.Timedelta, float]] = []

    for model_name in df_mae.columns:
        model_mae = df_mae[model_name].dropna()
        if model_mae.empty:
            continue

        candidates.append((model_name, model_mae.index.max(), model_mae.mean()))

    if not candidates:
        logger.warning("Cannot infer backup model from empty MAE data. Returning first.")
        return df_mae.columns[0]

    # Choose the model with the longest forecast horizon, and if tied, the lowest average MAE
    backup_model_name, _, _ = max(candidates, key=lambda item: (item[1], -item[2]))
    return backup_model_name


def calculate_best_pairwise_blend(
    df_mae: pd.DataFrame, 
    backup_model_name: str, 
    kernel: list[float], 
    score_func: Callable[[pd.Series], float],
) -> pd.DataFrame:
    """Calculate the blend across models which minimise the score function.
    
    This function chooses up to one model which will be blended into the backup model
    
    Args:
        df_mae: DataFrame of horizon-MAE values for all models under consideration. These should 
            have been modified for the delay of the latest model run.
        backup_model_name: The default model which all other models will be blended into
        kernel: The blend kernel
        score_func: The function of MAE which will be minimised
    """
    
    assert backup_model_name in df_mae.columns
    
    kernel = np.array(kernel)
    assert (kernel>0).all()
    assert (kernel<1).all()
    assert (np.diff(kernel)<=0).all()
    
    # Not all models predict out to the max horizon in the input DataFrame
    # We fill these model MAE horizons with a high value so they will be penalised
    # and therefore not selected
    fill_val = np.nanmax(df_mae.values)*10
    df_mae_filled = df_mae.fillna(fill_val)
    
    # Initialise best blend results
    # These are set to use only the backup model
    best_score = score_func(df_mae_filled[backup_model_name])
    best_model = backup_model_name
    best_weights = None
    
    
    backup_last_non_nan_idx = index_of_last_non_nan_value(df_mae[backup_model_name])
    
    for model in [c for c in df_mae.columns if c!=backup_model_name]:
        
        last_non_nan_idx = index_of_last_non_nan_value(df_mae[model])
        
        if last_non_nan_idx>=backup_last_non_nan_idx:
            
            intraday_weights = np.ones(len(df_mae))
            
            mae_series = (
                intraday_weights * df_mae_filled[model]
                + (1-intraday_weights) * df_mae_filled[backup_model_name]
            )
            score = score_func(mae_series)
            
            if score < best_score:
                best_score = score
                best_weights = intraday_weights
                best_model = model
                
        else:
            max_blend_start_pos = last_non_nan_idx - len(kernel) + 1
            min_blend_start_pos = -len(kernel) + 1
            for position in range(min_blend_start_pos, max_blend_start_pos + 1):

                # We don't allow the blend to start before the first forecast step
                if df_mae.index[position]<MIN_FORECAST_HORIZON:
                    continue

                candidate_weights = make_blend_weights_array(
                    size=len(df_mae), 
                    blend_start_index=position, 
                    kernel=kernel
                )

                # These are the expected horizon-MAEs for the blend position
                candidate_mae_series = (
                    candidate_weights * df_mae_filled[model]
                    + (1-candidate_weights) * df_mae_filled[backup_model_name]
                )
                score = score_func(candidate_mae_series)

                if score < best_score:
                    best_score = score
                    best_weights = candidate_weights
                    best_model = model

    if best_model==backup_model_name:
        backup_model_weights = np.ones(len(df_mae))
        backup_model_weights[df_mae[backup_model_name].isna()] = np.nan
        blend_results = {backup_model_name:backup_model_weights}

    else:
        backup_model_weights = 1 - best_weights
        backup_model_weights[df_mae[backup_model_name].isna()] = np.nan
        best_weights[df_mae[best_model].isna()] = np.nan
        blend_results = {best_model:best_weights, backup_model_name:backup_model_weights}
    
    return pd.DataFrame(blend_results, index=df_mae.index)


def add_best_candidate_to_blend(
    df_mae: pd.DataFrame,
    df_blend_weights: pd.DataFrame,
    candidate_names: set[str],
    kernel: list[float],
    score_func: Callable[[pd.Series], float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend up to one candidate model into the previous stage's blend

    Args:
        df_mae: DataFrame of horizon-MAE values for all models under consideration. These should 
            have been modified for the delay of the latest model run. Must also contain a
            "blend" column holding the horizon-MAE of the previous stage's blend.
        df_blend_weights: The blend weights from the previous stage, with one column per model
            already in the blend. These will be scaled by how much the previous stage's blend is
            used in this stage.
        candidate_names: Models which may be blended into the previous stage's blend. Up to one
            of these will be chosen. None of these should already be in df_blend_weights.
        kernel: The blend kernel
        score_func: The function of MAE which will be minimised

    Returns:
        A tuple of:
        - df_mae with the "blend" column updated to the horizon-MAE of the new blend
        - The blend weights for the new blend, with one column per model used
    """
    if missing := {"blend", *candidate_names} - set(df_mae.columns):
        raise ValueError(
            f"df_mae is missing required columns: {sorted(missing)}. It must contain "
            "a 'blend' column (the previous stage's blend MAE) and a column for each candidate "
            f"model. Available columns: {list(df_mae.columns)}"
        )

    if overlap := candidate_names & set(df_blend_weights.columns):
        raise ValueError(
            f"Candidates {sorted(overlap)} are already in the blend from a previous stage"
        )

    # Calculate the optimal blend weights for the candidates against the current blend
    df_candidate_blend_weights = calculate_best_pairwise_blend(
        df_mae[[*sorted(candidate_names), "blend"]],
        backup_model_name="blend",
        kernel=kernel,
        score_func=score_func,
    )

    # Calculate the weighted horizon-MAE for the new blend
    mask = df_mae[df_candidate_blend_weights.columns].notnull().any(axis=1)
    blend_mae = (
        (df_candidate_blend_weights.fillna(0) * df_mae[df_candidate_blend_weights.columns])
        .sum(skipna=True, axis=1)
        .where(mask)
    )
    df_new_delayed_mae = df_mae.assign(blend=blend_mae)

    # The previous stage blend weights must be multiplied by how much we are using that blend
    df_scaled_blend_weights = df_blend_weights.mul(df_candidate_blend_weights["blend"], axis=0)

    df_new_blend_weights = pd.concat(
        [df_scaled_blend_weights, df_candidate_blend_weights.drop(columns="blend")],
        axis=1
    )

    # Remove models whose weight has been reduced to zero at every horizon by this stage
    df_new_blend_weights = df_new_blend_weights.loc[:, df_new_blend_weights.sum(axis=0) > 0]

    return df_new_delayed_mae, df_new_blend_weights


def calculate_hierarchical_blend(
    df_mae: pd.DataFrame,
    model_delays_dict: dict[str, pd.Timedelta],
) -> pd.DataFrame:
    """Calculate blend weights from the horizon-MAE curves and current model delays

    We create a hierarchical blend using up to 4 models. A backup model is chosen as the model
    with the longest available forecast horizon. Then up to one of the other day-ahead models is
    blended into it. Then, optionally, pvnet_cloud is blended in. Finally, up to one of the
    remaining intraday models is blended in. At each stage a model is only blended in if it
    improves the expected MAE over that stage's scoring window.

    Args:
        df_mae: Horizon-MAE values for each candidate model (columns), without delays
        model_delays_dict: Current delay of each model. Must contain a delay for every model
            in df_mae

    Returns:
        A DataFrame of blend weights:
        - Index: Forecast horizon (pd.Timedelta)
        - Columns: The models used in the blend. Models which are unavailable or not chosen do not 
          appear
        - Values: Weights between 0 and 1. At each horizon the weights of the available
          models sum to 1. Weights are NaN where a model has no forecast for that horizon
    """

    if df_mae.empty:
        raise ValueError("df_mae is empty")

    if missing_delays := set(df_mae.columns) - set(model_delays_dict):
        raise ValueError(
            f"Models {sorted(missing_delays)} are in df_mae but have no delay in "
            "model_delays_dict. Remove them from df_mae if they are unavailable."
        )

    # Construct the expected horizon-MAE values for each model run with the current delays
    df_delayed_mae = delay_horizon_maes(df_mae, model_delays_dict)

    # Drop candidate models which have no future predictions and therefore have all NaN MAE curves
    df_delayed_mae = df_delayed_mae.dropna(axis=1, how="all")

    if df_delayed_mae.empty:
        raise ValueError("No models have forecasts available for the future")

    all_candidate_models = set(df_delayed_mae.columns)

    # Use the model with the longest forecast horizon as the backup model
    backup_model_name = infer_backup_model_name(df_delayed_mae)

    # Initialize the blend weights to be 100% backup model
    df_blend_weights = pd.DataFrame(
        {backup_model_name: np.ones(len(df_delayed_mae))}, 
        index=df_delayed_mae.index
    )

    # Initialize the blend MAE to be the backup model's MAE
    df_delayed_mae["blend"] = df_delayed_mae[backup_model_name]

    # Blend stage 1: Blend in the available day-ahead models (excluding the backup model)
    other_day_ahead_models = (
        all_candidate_models & DAY_AHEAD_MODEL_NAMES - set(df_blend_weights.columns)
    )
    if other_day_ahead_models:
        df_delayed_mae, df_blend_weights = add_best_candidate_to_blend(
            df_mae=df_delayed_mae,
            df_blend_weights=df_blend_weights,
            candidate_names=other_day_ahead_models, 
            kernel=BLEND_KERNEL, 
            score_func=make_avg_mae_func(36),
        )

    # Blend stage 2: Blend the pvnet_cloud model into the previous stage's blend unless its not
    # included or it was chosen as the backup model
    if "pvnet_cloud" in all_candidate_models and "pvnet_cloud" not in df_blend_weights.columns:
        df_delayed_mae, df_blend_weights = add_best_candidate_to_blend(
            df_mae=df_delayed_mae,
            df_blend_weights=df_blend_weights,
            candidate_names={"pvnet_cloud"},
            kernel=BLEND_KERNEL,
            score_func=make_avg_mae_func(8),
        )

    # Blend stage 3: Blend the other available intraday models into the previous stage's blend
    other_intraday_models = (
        all_candidate_models & INTRADAY_MODEL_NAMES - set(df_blend_weights.columns) - {"pvnet_cloud"}
    )

    if other_intraday_models:
        df_delayed_mae, df_blend_weights = add_best_candidate_to_blend(
            df_mae=df_delayed_mae,
            df_blend_weights=df_blend_weights,
            candidate_names=other_intraday_models,
            kernel=BLEND_KERNEL,
            score_func=make_avg_mae_func(8),
        )

    return df_blend_weights


async def get_blend_weights(
    client: dp.DataPlatformDataServiceStub,
    t0: pd.Timestamp, 
    exclude_models: Collection[str] | None = None,
) -> pd.DataFrame:
    """Get the blend weights for the forecast initialised at t0

    Loads the horizon-MAE curves, finds the current delay of each model from the Data
    Platform, and calculates the blend weights. See `calculate_hierarchical_blend` for
    how the blend is constructed.

    Args:
        client: Data Platform client used to find the latest forecast for each model
        t0: The forecast initialisation time
        exclude_models: These models will not be included in the blend

    Returns:
        A DataFrame of blend weights as returned by `calculate_hierarchical_blend`, but
        indexed by target time (t0 + horizon) rather than forecast horizon
    """

    all_candidate_models = set(ALL_MODEL_NAMES) - set(exclude_models or ())

    df_mae = get_horizon_maes()
    
    if len(all_candidate_models - set(df_mae.columns))!=0:
        skipped_models = all_candidate_models - set(df_mae.columns)
        raise ValueError(
            f"The following models are missing from the input DataFrame: {skipped_models}. "
            "Please ensure that the input DataFrame contains MAE-horizon values for all models "
            "under consideration."
        )
    
    # The maximum forecast horizon across all models
    max_horizon = df_mae.index.max()
    
    # Find how delayed the most recent forecast of each model is
    df_latest_forecast_ids = await get_latest_forecast_metadata(
        client=client,
        model_names=all_candidate_models, 
        t0=t0, 
        max_delay=max_horizon,
        location_type="national",
    )

    # Candidate models which have not run recently will not appear in the delays dict, and so will
    # be left out of the blend
    model_delays_dict = get_model_delays(df_latest_forecast_ids, t0)

    available_models = set(model_delays_dict)

    if not available_models:
        raise ValueError(
            f"None of the candidate models {sorted(all_candidate_models)} have run recently, "
            "so no blend can be made"
        )
    if not_run_recently := all_candidate_models - available_models:
        logger.warning(
            f"Models {sorted(not_run_recently)} have not run recently and will not be blended"
        )

    logger.info(f"Model delays: {model_delays_dict}")

    df_blend_weights = calculate_hierarchical_blend(
        df_mae=df_mae[sorted(available_models)],
        model_delays_dict=model_delays_dict,
    )

    # Convert forecast horizons to target times
    df_blend_weights.index = df_blend_weights.index + t0

    return df_blend_weights


def backfill_weights(df: pd.DataFrame, start_datetime: pd.Timestamp) -> pd.DataFrame:
    """Backfill weights for blending models
    
    Args:
        df: The DataFrame of weights
        start_datetime: The start time to backfill from
    
    Returns:
        A DataFrame with backfilled weights
    """
    
    fill_times = pd.date_range(start_datetime, df.index.min() - pd.Timedelta("30min"), freq="30min")

    df_filled = df.reindex(fill_times, method="bfill")
    
    return pd.concat([df_filled, df])
