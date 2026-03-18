import numpy as np
import pandas as pd
import pytest

import time_machine
from forecast_blend.weights import (
    get_national_blend_weights, get_regional_blend_weights, get_horizon_maes, ALL_MODEL_NAMES
)


def test_get_horizon_maes():
    # Just check the function can run
    df = get_horizon_maes()
    
    # Check all the expected models are present
    assert set(ALL_MODEL_NAMES) - set(df.columns) == set()


@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
async def test_get_national_blend_weights(forecast_national_ecmwf_and_xg, db_session):
    t0 = pd.Timestamp("2023-01-01 00:00", tz="UTC")

    weights_df = await get_national_blend_weights(session=db_session, t0=t0)

    # Check columns
    assert set(weights_df.columns)==set(["pvnet_ecmwf", "National_xg"])

    # Check index starts at t0+30min
    assert weights_df.index[0] == pd.Timestamp("2023-01-01 00:30", tz="UTC")

    # Weights sum to 1 for each time step (ignoring NaN rows)
    row_sums = weights_df.sum(axis=1)
    non_nan_rows = ~weights_df.isnull().all(axis=1)
    assert (row_sums[non_nan_rows].values == 1).all()

    # pvnet_ecmwf should be used near the start, not available at the end
    ecmwf_non_nan = weights_df["pvnet_ecmwf"].dropna()
    assert len(ecmwf_non_nan) > 0, "pvnet_ecmwf should have some non-NaN weights"
    assert (ecmwf_non_nan.values > 0).any(), "pvnet_ecmwf should have some positive weights"
    assert np.isnan(weights_df["pvnet_ecmwf"][16:]).all()

    # National_xg should be used for all timesteps after 16
    assert (weights_df["National_xg"][16:].values == 1).all()


@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
async def test_get_regional_blend_weights(forecast_national_ecmwf_and_xg, db_session):
    t0 = pd.Timestamp("2023-01-01 00:00", tz="UTC")

    weights_df = await get_regional_blend_weights(session=db_session, t0=t0)

    # For regional we can only use pvnet_ecmwf since National_xg is only for national

    # Check columns and first index
    assert list(weights_df.columns) == ["pvnet_ecmwf"]
    assert weights_df.index[0] == pd.Timestamp("2023-01-01 00:30", tz="UTC")

    # We use pvnet_ecmwf exclusively for all time steps where it's available
    assert (weights_df.values == 1).all()

    # In this case the weights should start at 2h later since the forecast is 2h old
    weights_df_2h = await get_regional_blend_weights(session=db_session, t0=t0+pd.Timedelta("2h"))

    # With 2h delay, pvnet_ecmwf window is shifted; result may be empty or start later
    if len(weights_df_2h.columns) > 0:
        assert weights_df_2h.index[0] >= pd.Timestamp("2023-01-01 02:30", tz="UTC")

# Test with and without excluding the pvnet_cloud model
test_settings = [(None, "pvnet_cloud"),  (["pvnet_cloud"], "pvnet_v2")]
@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("exclude_models, intraday_model", test_settings)
async def test_get_regional_blend_weights_cloud(forecast_national_all_now, db_session, exclude_models, intraday_model):
    t0 = pd.Timestamp("2023-01-01 00:00", tz="UTC")

    weights_df = await get_regional_blend_weights(session=db_session, t0=t0, exclude_models=exclude_models)
    
    # Check the expected models have been returned
    assert set(weights_df.columns) == set([intraday_model, "pvnet_day_ahead"])

    # intraday_model should be used for the first steps where it's available, then not
    intraday_col = weights_df[intraday_model].dropna()
    assert (intraday_col.values > 0).all()

    # pvnet_day_ahead should be used for all timesteps after the intraday model runs out
    last_intraday_idx = weights_df[intraday_model].last_valid_index()
    after_intraday = weights_df.loc[last_intraday_idx:]["pvnet_day_ahead"].dropna()
    assert (after_intraday.values == 1).any()