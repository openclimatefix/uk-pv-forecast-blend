from app import app

from nowcasting_datamodel.models.forecast import (
    ForecastSQL,
    ForecastValueLatestSQL,
    ForecastValueSevenDaysSQL,
    ForecastValueSQL,
)


def test_app(db_session, forecasts):

    # Check the number forecasts have been made
    # (10 GSPs + 1 National) = 11 forecasts
    # This is for PVnet and CNN, but National_xg only is national
    # 11 + 11 + 1 = 23
    # Doubled for historic and forecast
    assert len(db_session.query(ForecastSQL).all()) == 46
    # 11 GSPs * 16 time steps in forecast
    assert len(db_session.query(ForecastValueSQL).all()) == 23 * 16
    assert len(db_session.query(ForecastValueLatestSQL).all()) == 23 * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == 23 * 16

    app(gsps=list(range(0,10)))

    # the same
    assert len(db_session.query(ForecastSQL).all()) == 46
    assert len(db_session.query(ForecastValueSQL).all()) == 23 * 16
    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == 23 * 16

    # extra 11 gsps for forecast value latest table
    assert len(db_session.query(ForecastValueLatestSQL).all()) == 34 * 16



