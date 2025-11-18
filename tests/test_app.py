import pandas as pd
import time_machine
import os
import asyncio
import pytest

from nowcasting_datamodel.models import LocationSQL
from nowcasting_datamodel.models.forecast import (
    ForecastSQL,
    ForecastValueLatestSQL,
    ForecastValueSevenDaysSQL,
    ForecastValueSQL,
)
from nowcasting_datamodel.models.models import MLModelSQL

from forecast_blend.app import app, is_last_forecast_made_before_last_30_minutes_step


@time_machine.travel("2023-01-01 00:00:01")
def test_is_last_forecast_longer_30_minutes(db_session):

    assert is_last_forecast_made_before_last_30_minutes_step(db_session, blend_name=os.environ["BLEND_NAME"])


@time_machine.travel("2023-01-01 00:00:01")
def test_is_last_forecast_longer_30_minutes_dont_create(db_session, forecasts):

    # make sure model is the blend
    f = db_session.query(ForecastSQL).all()
    f[0].model.name = os.environ["BLEND_NAME"]

    assert not is_last_forecast_made_before_last_30_minutes_step(db_session, blend_name=os.environ["BLEND_NAME"])


@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
def test_app(db_session, forecasts):

    # Check the number forecasts have been made
    # (10 GSPs + 1 National) = 11 forecasts
    # This is for pvnet_v2, pvnet_day_ahead, pvnet_ecmwf, pvnet_cloud, and National-xg (which is only National)
    # 11 + 11 + 11 + 11 + 1 = 45
    N = 45
    # Doubled for historic and forecast
    assert len(db_session.query(ForecastSQL).all()) == N * 2
    assert len(db_session.query(LocationSQL).all()) == 11
    # 11 GSPs * 16 time steps in forecast
    assert len(db_session.query(ForecastValueSQL).all()) == N * 16
    assert len(db_session.query(ForecastValueLatestSQL).all()) == N * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == N * 16

    asyncio.run(app(gsps=list(range(0, 11))))

    assert len(db_session.query(ForecastValueSQL).all()) == (N + 11) * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == (N + 11) * 16
    assert len(db_session.query(ForecastSQL).all()) == 2*N + 11 * 2  # historic and not
    assert len(db_session.query(ForecastValueLatestSQL).all()) == (N + 11) * 16


@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
def test_app_twice(db_session, forecasts):

    # Check the number forecasts have been made
    # (10 GSPs + 1 National) = 11 forecasts
    # This is for pvnet_v2, pvnet_day_ahead, pvnet_ecmwf, pvnet_cloud, and National-xg (which is only National)
    # 11 + 11 + 11 + 11 + 1 = 45
    N = 45
    # Doubled for historic and forecast
    assert len(db_session.query(ForecastSQL).all()) == 2 * N
    assert len(db_session.query(LocationSQL).all()) == 11
    # 11 GSPs * 16 time steps in forecast
    assert len(db_session.query(ForecastValueSQL).all()) == N * 16
    assert len(db_session.query(ForecastValueLatestSQL).all()) == N * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == N * 16

    asyncio.run(app(gsps=list(range(0, 11))))

    assert len(db_session.query(ForecastValueSQL).all()) == (N + 11) * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == (N + 11) * 16
    assert len(db_session.query(ForecastSQL).all()) == (2*N) + 11 * 2  # historic and not
    assert len(db_session.query(ForecastValueLatestSQL).all()) == (N+11) * 16

    asyncio.run(app(gsps=list(range(0, 11))))

    # none should change, as only ForecastValueLatestSQL is being updated
    assert len(db_session.query(ForecastValueSQL).all()) == (N + 11) * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == (N + 11) * 16
    assert len(db_session.query(ForecastSQL).all()) == 2*N + 11 * 2  # historic and not
    assert len(db_session.query(ForecastValueLatestSQL).all()) == (N+11) * 16


@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
def test_app_only_national(db_session, forecast_national):

    # Check the number forecasts have been made
    # 1 National
    # This is for pvnet_v2, pvnet_day_ahead, pvnet_ecmwf, pvnet_cloud, and National-xg
    N = 5
    # Doubled for historic and forecast
    assert len(db_session.query(ForecastSQL).all()) == 2*N
    assert len(db_session.query(LocationSQL).all()) == 1
    #  16 time steps in forecast
    assert len(db_session.query(ForecastValueSQL).all()) == N * 16
    assert len(db_session.query(ForecastValueLatestSQL).all()) == N * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == N * 16

    asyncio.run(app(gsps=list(range(0, 2))))

    assert len(db_session.query(ForecastValueSQL).all()) == (N+1) * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == (N+1) * 16
    assert len(db_session.query(ForecastSQL).all()) == 2*(N+1)   # historic and not
    assert len(db_session.query(ForecastValueLatestSQL).all()) == (N+1) * 16


@time_machine.travel("2023-01-01 00:00:01")
@pytest.mark.asyncio(loop_scope="session")
def test_app_only_ecwmf_and_xg(db_session, forecast_national_ecmwf_and_xg):
    # Check the number forecasts have been made
    # This is for PVnet ecmwf and National_xg only is national
    # 4
    N = 2
    # Doubled for historic and forecast
    assert len(db_session.query(ForecastSQL).all()) == 2 * N
    assert len(db_session.query(LocationSQL).all()) == 1

    number_of_forecast_values = 120

    #  16 time steps in forecast
    assert len(db_session.query(ForecastValueSQL).all()) == N * number_of_forecast_values
    assert len(db_session.query(ForecastValueLatestSQL).all()) == N * number_of_forecast_values
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == N * number_of_forecast_values

    asyncio.run(app(gsps=[0]))

    # get all the blended forecast values latest
    models = db_session.query(MLModelSQL).where(MLModelSQL.name == os.environ["BLEND_NAME"]).all()
    assert len(models) == 1
    fvs = db_session.query(ForecastValueLatestSQL).where(ForecastValueLatestSQL.model_id == models[0].id).all()

    assert len(fvs) == 25

    expected_values = pd.Series(
        [0]*15+[0.25, 0.5, 0.75]+[1]*7,
        index=pd.date_range("2022-12-31 23:30", "2023-01-01 11:30", freq="30min", tz="UTC"),
    )

    for i, fv in enumerate(fvs):
        assert (
            (fv.expected_power_generation_megawatts, fv.target_time)
             == (expected_values.values[i], expected_values.index[i])
        )

    assert len(db_session.query(ForecastValueSQL).all()) == (N * number_of_forecast_values) + 25
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == (N * number_of_forecast_values) + 25
    assert len(db_session.query(ForecastSQL).all()) == 2 * (N + 1)  # historic and not
    assert len(db_session.query(ForecastValueLatestSQL).all()) == (N * number_of_forecast_values) + 25




