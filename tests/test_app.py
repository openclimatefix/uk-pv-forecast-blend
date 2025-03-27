import logging

from forecast_blend.app import app, is_last_forecast_made_before_last_30_minutes_step
from nowcasting_datamodel.models import LocationSQL
from nowcasting_datamodel.models.forecast import (
    ForecastSQL,
    ForecastValueLatestSQL,
    ForecastValueSevenDaysSQL,
    ForecastValueSQL,
)
from nowcasting_datamodel.models.models import MLModelSQL

logger = logging.getLogger(__name__)


def test_is_last_forecast_longer_30_minutes(db_session):

    assert is_last_forecast_made_before_last_30_minutes_step(db_session)


def test_is_last_forecast_longer_30_minutes_dont_create(db_session, forecasts):

    # make sure model is "blend"
    f = db_session.query(ForecastSQL).all()
    f[0].model.name = "blend"

    assert not is_last_forecast_made_before_last_30_minutes_step(db_session)


def test_app(db_session, forecasts):

    # Check the number forecasts have been made
    # (10 GSPs + 1 National) = 11 forecasts
    # This is for PVnet and PVnet DA, PVNet ECMWF, National-xg (which is only National)
    # 11 + 11 + 11 + 1 = 34
    N = 34
    # Doubled for historic and forecast
    assert len(db_session.query(ForecastSQL).all()) == N * 2
    assert len(db_session.query(LocationSQL).all()) == 11
    # 11 GSPs * 16 time steps in forecast
    assert len(db_session.query(ForecastValueSQL).all()) == N * 16
    assert len(db_session.query(ForecastValueLatestSQL).all()) == N * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == N * 16

    app(gsps=list(range(0, 11)))

    assert len(db_session.query(ForecastValueSQL).all()) == (N + 11) * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == (N + 11) * 16
    assert len(db_session.query(ForecastSQL).all()) == 2*N + 11 * 2  # historic and not
    assert len(db_session.query(ForecastValueLatestSQL).all()) == (N + 11) * 16


def test_app_twice(db_session, forecasts):

    # Check the number forecasts have been made
    # (10 GSPs + 1 National) = 11 forecasts
    # This is for PVnet and PVnet DA, PVNet ECMWF, National-xg (which is only National)
    # 11 + 11 + 11 +1 = 34
    N = 34
    # Doubled for historic and forecast
    assert len(db_session.query(ForecastSQL).all()) == 2 * N
    assert len(db_session.query(LocationSQL).all()) == 11
    # 11 GSPs * 16 time steps in forecast
    assert len(db_session.query(ForecastValueSQL).all()) == N * 16
    assert len(db_session.query(ForecastValueLatestSQL).all()) == N * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == N * 16

    app(gsps=list(range(0, 11)))

    assert len(db_session.query(ForecastValueSQL).all()) == (N + 11) * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == (N + 11) * 16
    assert len(db_session.query(ForecastSQL).all()) == (2*N) + 11 * 2  # historic and not
    assert len(db_session.query(ForecastValueLatestSQL).all()) == (N+11) * 16

    app(gsps=list(range(0, 11)))

    # none should change, as only ForecastValueLatestSQL is being updated
    assert len(db_session.query(ForecastValueSQL).all()) == (N + 11) * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == (N + 11) * 16
    assert len(db_session.query(ForecastSQL).all()) == 2*N + 11 * 2  # historic and not
    assert len(db_session.query(ForecastValueLatestSQL).all()) == (N+11) * 16


def test_app_only_national(db_session, forecast_national):

    # Check the number forecasts have been made
    # 1 National)
    # This is for PVnet and PVnet DA, PVNet ECMWF, National-xg (which is only National)
    # 4
    N = 4
    # Doubled for historic and forecast
    assert len(db_session.query(ForecastSQL).all()) == 2*N
    assert len(db_session.query(LocationSQL).all()) == 1
    #  16 time steps in forecast
    assert len(db_session.query(ForecastValueSQL).all()) == N * 16
    assert len(db_session.query(ForecastValueLatestSQL).all()) == N * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == N * 16

    app(gsps=list(range(0, 2)))

    assert len(db_session.query(ForecastValueSQL).all()) == (N+1) * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == (N+1) * 16
    assert len(db_session.query(ForecastSQL).all()) == 2*(N+1)   # historic and not
    assert len(db_session.query(ForecastValueLatestSQL).all()) == (N+1) * 16


def test_app_only_ecwmf_and_xg(db_session, forecast_national_ecmwf_and_xg):
    # Check the number forecasts have been made
    # This is for PVnet ecmwf and National_xg only is national
    # 4
    N = 2
    # Doubled for historic and forecast
    assert len(db_session.query(ForecastSQL).all()) == 2 * N
    assert len(db_session.query(LocationSQL).all()) == 1
    #  16 time steps in forecast
    assert len(db_session.query(ForecastValueSQL).all()) == N * 16
    assert len(db_session.query(ForecastValueLatestSQL).all()) == N * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == N * 16

    app(gsps=list(range(0, 1)))

    # get all the blended forecast values latest
    models = db_session.query(MLModelSQL).where(MLModelSQL.name == 'blend').all()
    assert len(models) == 1
    fvs = db_session.query(ForecastValueLatestSQL).where(ForecastValueLatestSQL.model_id == models[0].id).all()

    # make sure this is the ECMWF not xg, for xg, it would be 1
    assert len(fvs) == 16
    for fv in fvs:
        assert fv.expected_power_generation_megawatts == 0, f'{fv.expected_power_generation_megawatts} {fv.target_time}'

    assert len(db_session.query(ForecastValueSQL).all()) == (N + 1) * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == (N + 1) * 16
    assert len(db_session.query(ForecastSQL).all()) == 2 * (N + 1)  # historic and not
    assert len(db_session.query(ForecastValueLatestSQL).all()) == (N + 1) * 16




