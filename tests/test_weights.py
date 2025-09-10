import numpy as np
import pandas as pd
import pytest

import time_machine
from weights import (
    get_national_blend_weights, get_regional_blend_weights, get_horizon_maes, ALL_MODEL_NAMES
)


def test_get_horizon_maes():
    # Just check the function can run
    df = get_horizon_maes()
    
    # Check all the expected models are present
    assert set(ALL_MODEL_NAMES) - set(df.columns) == set()


@time_machine.travel("2023-01-01 00:00:01")
def test_get_national_blend_weights(forecast_national_ecmwf_and_xg, db_session):
    t0 = pd.Timestamp("2023-01-01 00:00", tz="UTC")

    weights_df = get_national_blend_weights(session=db_session, t0=t0)

    # Check columns and indices
    assert set(weights_df.columns)==set(["pvnet_ecmwf", "National_xg"])
    assert (
        pd.date_range("2023-01-01 00:30", "2023-01-02 12:00", freq="30min", tz="UTC")
        .equals(weights_df.index)
    )

    # Weights sum to 1 for each time step
    assert (weights_df.sum(axis=1).values==1).all()

    # pvnet_ecmwf should be used for the first 16 time steps then not available
    assert (weights_df["pvnet_ecmwf"][:16].values>0).all()
    assert np.isnan(weights_df["pvnet_ecmwf"][16:]).all()

    # National_xg should be used for all timesteps after 16
    assert (weights_df["National_xg"][16:].values==1).all()


@time_machine.travel("2023-01-01 00:00:01")
def test_get_regional_blend_weights(forecast_national_ecmwf_and_xg, db_session):
    t0 = pd.Timestamp("2023-01-01 00:00", tz="UTC")

    weights_df = get_regional_blend_weights(session=db_session, t0=t0)

    # For regional we can only use pvnet_ecmwf since National_xg is only for national
    
    # Check columns and indices
    assert weights_df.columns==["pvnet_ecmwf"]
    assert (
        pd.date_range("2023-01-01 00:30", "2023-01-01 08:00", freq="30min", tz="UTC")
        .equals(weights_df.index)
    )

    # We use pvnet_ecmwf exclusively for all time steps
    assert (weights_df.values==1).all()

    # In this case the weights should start at 2h later since the forecast is 2h old
    weights_df = get_regional_blend_weights(session=db_session, t0=t0+pd.Timedelta("2h"))
    
    # Check columns and indices
    assert weights_df.columns==["pvnet_ecmwf"]
    assert (
        pd.date_range("2023-01-01 02:30", "2023-01-01 08:00", freq="30min", tz="UTC")
        .equals(weights_df.index)
    )

    assert (weights_df.values==1).all()

# Test with and without excluding the pvnet_cloud model
test_settings = [(None, "pvnet_cloud"),  (["pvnet_cloud"], "pvnet_v2")]
@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.parametrize("exclude_models, intraday_model", test_settings)
def test_get_regional_blend_weights_cloud(forecast_national_all_now, db_session, exclude_models, intraday_model):
    t0 = pd.Timestamp("2023-01-01 00:00", tz="UTC")

    weights_df = get_regional_blend_weights(session=db_session, t0=t0, exclude_models=exclude_models)
    
    # Check the expected models have been returned
    assert set(weights_df.columns)==set([intraday_model, "pvnet_day_ahead"])

    # intraday_model should be used for the first 16 time steps then not available
    assert (weights_df[intraday_model][:16].values>0).all()
    assert np.isnan(weights_df[intraday_model][16:]).all()

    # pvnet_day_ahead should be used for all timesteps after 16
    assert (weights_df["pvnet_day_ahead"][16:].values==1).all()