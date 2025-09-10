"""Functions to make weights for blending"""

import numpy as np
import os
import pandas as pd
from typing import Callable
from datetime import timezone
from loguru import logger

from sqlalchemy.orm import Session
from nowcasting_datamodel.models import ForecastSQL, MLModelSQL



DAY_AHEAD_MODEL_NAMES = ["pvnet_day_ahead", "National_xg"]
INTRADAY_MODEL_NAMES = ["pvnet_v2", "pvnet_ecmwf", "pvnet_cloud"]
ALL_MODEL_NAMES = DAY_AHEAD_MODEL_NAMES + INTRADAY_MODEL_NAMES

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
    return df_maes.set_index("horizon")


def get_latest_forecast_metadata(
    session: Session, 
    model_names: list[str], 
    t0: pd.Timestamp, 
    max_delay: pd.Timedelta,
) -> pd.DataFrame:
    """Gets the most recent forecast IDs and creation times for specified models for each location.

    Args:
        session: Database session
        model_names: List of model names to filter to
        t0: The blend forecast init-time
        max_delay: Max delay with respect to t0 to consider using forecast in the blend

    Returns:
        A pandas DataFrame with columns 'created_utc', 'id', 'forecast_creation_time',
        'location_id', and 'name' (model name) representing the latest forecasts for each
        (location, model) combination
    """

    t0_datetime = t0.to_pydatetime().replace(tzinfo=timezone.utc)
    earliest_creation_time = t0_datetime - max_delay

    # Define the columns we want to pull
    forecast_columns = [
        ForecastSQL.created_utc, 
        ForecastSQL.id, 
        ForecastSQL.forecast_creation_time, 
        ForecastSQL.location_id,
        MLModelSQL.name,
    ]
    
    # Build query that joins the ForecastSQL and MLModelSQL tables, applies filters, 
    # and takes the most recent row for each (model, location) pair
    query = (
        session.query(*forecast_columns)
        .distinct(ForecastSQL.location_id, MLModelSQL.name)
        .join(MLModelSQL, ForecastSQL.model_id == MLModelSQL.id)
        .filter(ForecastSQL.created_utc >= earliest_creation_time)
        .filter(MLModelSQL.name.in_(model_names))
        .order_by(ForecastSQL.location_id, MLModelSQL.name, ForecastSQL.forecast_creation_time.desc())
    )

    return pd.read_sql(query.statement, query.session.bind)


def _get_most_recent_row(df: pd.DataFrame) -> pd.Series:
    return df.sort_values("forecast_creation_time", ascending=False).iloc[0]


def get_model_delays(df_forecast_ids: pd.DataFrame, t0: pd.Timestamp) -> dict[str, pd.Timedelta]:
    """Construct a dictionary of how delayed each model is
    
    Args:
        df_forecast_ids: DataFrame of the most recent forecast IDs for each model and location. 
            See `get_latest_forecast_metadata()`.
        t0: The blend forecast init time
    
    """
    
    # Filter to the most recent forecast ID per model
    df_forecast_ids = (
        df_forecast_ids.groupby("name")[df_forecast_ids.columns]
        .apply(_get_most_recent_row)
    )
    
    # TODO: Use the saved forecast initialisation time when available
    # https://github.com/openclimatefix/nowcasting_datamodel/issues/315
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
    
    Arga:
        df_mae: The horizon-MAE results for each model without runtime delays
        delays_dict: The current delay of each model

    Returns:
        A DataFrame of horizon-MAE values for each model run with the given delays
    """
    
    mae_models = set(df_mae.columns)
    delay_models = set(delays_dict.keys())
    
    if len(mae_models - delay_models)>0:
        skipped_models = mae_models - delay_models
        logger.warn(
            f"The following models have saved MAES but were not available {skipped_models}. "
            "These models will be skipped in consideration for the blend."
        
        )
    if len(delay_models - mae_models)>0:
        skipped_models = delay_models - mae_models
        logger.warn(
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
    
    weights = np.zeros(size)
    weights[:blend_start_index] = 1
    weights[blend_start_index:blend_start_index+len(kernel)] = kernel
    return weights


def index_of_last_non_nan_value(x: np.ndarray) -> int:
    """Find the index position of the last non-NaN value in the arreay

    Args:
        x: The input array
    """
    if np.isnan(x).any():
        return int(np.isnan(x).argmax())-1
    else:
        return len(x) - 1


def calculate_optimal_blend_weights(
    df_mae: pd.DataFrame, 
    backup_model_name: str, 
    kernel: list[float], 
    score_func: Callable[[pd.Series], float],
) -> pd.DataFrame:
    """Calculate the blend across models which minimise the score function.
    
    This function chooses up to one model which will be blended into the backup model
    
    Args:
        df_mae: DataFrame of horizon-MAE values for all models under consideration. These should have beeen
            modified for the delay of the latest model run.
        backup_model_name: The default model which all other models will be blended into
        kernel: The blend kernel
        score_func: The function of MAE which will be minimised
    """
    
    assert backup_model_name in df_mae.columns
    
    kernel = np.array(kernel)
    assert (kernel>0).all()
    assert (kernel<1).all()
    assert (np.diff(kernel)<=0).all()
    
    # Not all models predict out to the max horizon in the input DataFrme
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
            for position in range(max_blend_start_pos + 1):

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


def get_national_blend_weights(
    session: Session, 
    t0: pd.Timestamp, 
    exclude_models: list[str] | None = None,
) -> pd.DataFrame:
    """Determines optimal time-varying weights for blending multiple forecast models.

    This function calculates weights for combining various day-ahead and intraday models based
    on their historical performance (MAE) and current operational delay. 
    
    We create a hierarchical blend where up to one day-ahead model is chosen and blended into 
    National_xg. We also choose up to one intraday model to blend into the day-ahead blend.
    
    Args:
        session: The database session
        t0: The forecast initialisation time
        exclude_models: These models will not be excluded from the blend

    Returns:
        A pandas DataFrame containing the optimal blend weights:
        - Index: Forecast horizon relative to `t0` (pd.Timedelta).
        - Columns: Names of the models included in the final blend (subset of
          `DAY_AHEAD_MODEL_NAMES` and `INTRADAY_MODEL_NAMES`).
        - Values: Float weights (between 0.0 and 1.0) assigned to each model
          at each forecast horizon. At any given horizon, weights for available
          models should sum approximately to 1.0. Weights will be NaN where a
          model was inherently unavailable (e.g., NaN in its original MAE curve
          at that horizon). Models excluded due to excessive delay or missing
          MAE data will not appear as columns or will have zero/NaN weights.
    """
    
    df_mae = get_horizon_maes()

    if exclude_models is None:
        all_model_names = [*ALL_MODEL_NAMES]
    else:
        all_model_names = [m for m in ALL_MODEL_NAMES if m not in exclude_models]
        df_mae = df_mae.drop(columns=exclude_models)
    
    # We need to have MAE-horizon values for all potential models
    assert len(set(all_model_names) - set(df_mae.columns))==0
    
    # The maximum forecast horizon of any of the models
    max_horizon = df_mae.index.max()
    
    # Find how delayed the most recent forecast of each model is
    df_latest_forecast_ids = get_latest_forecast_metadata(
        session=session, 
        model_names=all_model_names, 
        t0=t0, 
        max_delay=max_horizon,
    )
    model_delays_dict = get_model_delays(df_latest_forecast_ids, t0)
    
    # If the model has not run recently it will not appear in the recent forecasts in the database. 
    # Therefore it will not appear in model_delays_dict. Add it with a delay of the maximum forecast 
    # horizon so it will be present, but disregarded in further steps
    model_delays_dict = {m: model_delays_dict.get(m, max_horizon) for m in all_model_names}

    # Construct the expected horizon-MAE values for each model run with the current delays
    df_delayed_mae = delay_horizon_maes(df_mae, model_delays_dict)
    
    # Calculate the optimal blend weights between day-ahead models
    df_da_model_weights = calculate_optimal_blend_weights(
        df_delayed_mae[all_model_names], 
        backup_model_name="National_xg", 
        kernel=BLEND_KERNEL, 
        score_func=make_avg_mae_func(36),
    )
    
    # Calculate the expected horizon-MAE for the day-ahead blend computed above
    mask = ~df_delayed_mae[all_model_names].isnull().all(axis=1)
    df_delayed_mae["da_blend"] = (
        df_da_model_weights.fillna(0)*df_delayed_mae[DAY_AHEAD_MODEL_NAMES]
    ).sum(skipna=True, axis=1).where(mask)
    
    # Calculate the optimal blend weights for blending the intraday models into the day-ahead blend
    df_intraday_model_weights = calculate_optimal_blend_weights(
        df_delayed_mae[all_model_names+["da_blend"]], 
        backup_model_name="da_blend",
        kernel=BLEND_KERNEL,
        score_func=make_avg_mae_func(8),
    )
    
    # The day-ahead blend weights must be multiplied by how much we are using the day-ahead blend
    # at each step
    for col in df_da_model_weights.columns:
        df_da_model_weights[col] = df_da_model_weights[col]*df_intraday_model_weights["da_blend"]
    
    df_intraday_model_weights = df_intraday_model_weights.drop(columns="da_blend")

    df_all_weights = pd.concat([df_da_model_weights, df_intraday_model_weights], axis=1)
    
    # Filter out any models which are not used in this blend
    df_all_weights = df_all_weights.loc[:, df_all_weights.sum(axis=0) > 0]

    # Shift the index to be relative to the forecast initialisation time
    df_all_weights.index = df_all_weights.index + t0

    return df_all_weights


def get_regional_blend_weights(
    session: Session, 
    t0: pd.Timestamp, 
    exclude_models: list[str] = None,
) -> pd.DataFrame:
    """Determines optimal time-varying weights for blending multiple forecast models.

    This function calculates weights for combining various day-ahead and intraday models based
    on their historical performance (MAE) and current operational delay. 
    
    We create a blend where up to one intraday model is blended into pvnet_day_ahead.
    
    Args:
        session: The database session
        t0: The forecast initialisation time
        exclude_models: These models will not be excluded from the blend

    Returns:
        A pandas DataFrame containing the optimal blend weights:
        - Index: Forecast horizon relative to `t0` (pd.Timedelta).
        - Columns: Names of the models included in the final blend (subset of
          `DAY_AHEAD_MODEL_NAMES` and `INTRADAY_MODEL_NAMES`).
        - Values: Float weights (between 0.0 and 1.0) assigned to each model
          at each forecast horizon. At any given horizon, weights for available
          models should sum approximately to 1.0. Weights will be NaN where a
          model was inherently unavailable (e.g., NaN in its original MAE curve
          at that horizon). Models excluded due to excessive delay or missing
          MAE data will not appear as columns or will have zero/NaN weights.
    """

    # National_xg is not a regional model
    if exclude_models is None:
        exclude_models = ["National_xg"]
    else:
        exclude_models = exclude_models + ["National_xg"]

    df_mae = get_horizon_maes().drop(columns=exclude_models)

    # We need to have MAE-horizon values for all potential models
    all_regional_models = [m for m in ALL_MODEL_NAMES if m not in exclude_models]
    assert len(set(all_regional_models) - set(df_mae.columns))==0
    
    # The maximum forecast horizon of any of the models
    max_horizon = df_mae.index.max()
    
    # Find how delayed the most recent forecast of each model is
    df_latest_forecast_ids = get_latest_forecast_metadata(
        session=session, 
        model_names=all_regional_models, 
        t0=t0, 
        max_delay=max_horizon,
    )
    model_delays_dict = get_model_delays(df_latest_forecast_ids, t0)
    
    # If the model has not run recently it will not appear in the recent forecasts in the database. 
    # Therefore it will not appear in model_delays_dict. Add it with a delay of the maximum forecast 
    # horizon so it will be present, but disregarded in further steps
    model_delays_dict = {m: model_delays_dict.get(m, max_horizon) for m in all_regional_models}

    # Construct the expected horizon-MAE values for each model run with the current delays
    df_delayed_mae = delay_horizon_maes(df_mae, model_delays_dict)
    
    # Calculate the optimal blend weights between day-ahead models
    weights_df = calculate_optimal_blend_weights(
        df_delayed_mae[all_regional_models], 
        backup_model_name="pvnet_day_ahead", 
        kernel=BLEND_KERNEL, 
        score_func=make_avg_mae_func(8),
    )
    
    # Filter out any models which are not used in this blend
    weights_df = weights_df.loc[:, weights_df.sum(axis=0) > 0]

    # Shift the index to be relative to the forecast initialisation time
    weights_df.index = weights_df.index + t0

    return weights_df


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
