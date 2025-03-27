from datetime import datetime, timezone

from freezegun import freeze_time
from weights import get_national_blend_weights, get_regional_blend_weights


@freeze_time("2023-01-02 00:00:01")
def test_get_national_blend_weights(forecast_national_ecmwf_and_xg, db_session):
    start_datetime = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    weights = get_national_blend_weights(session=db_session, t0=start_datetime)

    # TODO add asserts


@freeze_time("2023-01-02 00:00:01")
def test_get_regional_blend_weights(forecast_national_ecmwf_and_xg, db_session):
    start_datetime = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    weights = get_regional_blend_weights(session=db_session, t0=start_datetime)

    # TODO add asserts

