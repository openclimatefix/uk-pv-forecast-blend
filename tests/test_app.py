import logging

from app import app, is_last_forecast_made_before_last_30_minutes_step
from nowcasting_datamodel.models import LocationSQL
from nowcasting_datamodel.models.forecast import (
    ForecastSQL,
    ForecastValueLatestSQL,
    ForecastValueSevenDaysSQL,
    ForecastValueSQL,
)

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
    # This is for PVnet and CNN, but National_xg only is national
    # 11 + 11 + 1 = 23
    # Doubled for historic and forecast
    assert len(db_session.query(ForecastSQL).all()) == 46
    assert len(db_session.query(LocationSQL).all()) == 11
    # 11 GSPs * 16 time steps in forecast
    assert len(db_session.query(ForecastValueSQL).all()) == 23 * 16
    assert len(db_session.query(ForecastValueLatestSQL).all()) == 23 * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == 23 * 16

    app(gsps=list(range(0, 11)))

    assert len(db_session.query(ForecastValueSQL).all()) == (23 + 11) * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == (23 + 11) * 16
    assert len(db_session.query(ForecastSQL).all()) == 46 + 11 * 2  # historic and not
    assert len(db_session.query(ForecastValueLatestSQL).all()) == 34 * 16


def test_app_twice(db_session, forecasts):

    # Check the number forecasts have been made
    # (10 GSPs + 1 National) = 11 forecasts
    # This is for PVnet and CNN, but National_xg only is national
    # 11 + 11 + 1 = 23
    # Doubled for historic and forecast
    assert len(db_session.query(ForecastSQL).all()) == 46
    assert len(db_session.query(LocationSQL).all()) == 11
    # 11 GSPs * 16 time steps in forecast
    assert len(db_session.query(ForecastValueSQL).all()) == 23 * 16
    assert len(db_session.query(ForecastValueLatestSQL).all()) == 23 * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == 23 * 16

    app(gsps=list(range(0, 11)))

    assert len(db_session.query(ForecastValueSQL).all()) == (23 + 11) * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == (23 + 11) * 16
    assert len(db_session.query(ForecastSQL).all()) == 46 + 11 * 2  # historic and not
    assert len(db_session.query(ForecastValueLatestSQL).all()) == 34 * 16

    app(gsps=list(range(0, 11)))

    # none should change, as only ForecastValueLatestSQL is being updated
    assert len(db_session.query(ForecastValueSQL).all()) == (23 + 11) * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == (23 + 11) * 16
    assert len(db_session.query(ForecastSQL).all()) == 46 + 11 * 2  # historic and not
    assert len(db_session.query(ForecastValueLatestSQL).all()) == 34 * 16
