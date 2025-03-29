from forecast_blend.utils import get_start_datetime
from freezegun import freeze_time
from datetime import datetime, timezone


def test_get_start_datetime():
    start_datetime = get_start_datetime()

    assert start_datetime.minute in [0, 30]


def test_get_start_datetime_midnight():

    with freeze_time("2021-01-01 00:00:01"):
        start_datetime = get_start_datetime()

        assert start_datetime == datetime(2020, 12, 31, 23, 30, 0, 0, timezone.utc)

    with freeze_time("2021-01-01 00:45:01"):
        start_datetime = get_start_datetime()

        assert start_datetime == datetime(2021, 1, 1, 0, 0, 0, 0, timezone.utc)
