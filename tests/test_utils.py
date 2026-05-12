"""
Tests:
1. test_get_start_datetime          - Start datetime is always on a 30-min boundary
2. test_get_start_datetime_midnight - Start datetime rolls back correctly across midnight
"""

import time_machine
from datetime import datetime, timezone
from forecast_blend.utils import get_start_datetime


def test_get_start_datetime():
    start_datetime = get_start_datetime()

    assert start_datetime.minute in [0, 30]


def test_get_start_datetime_midnight():

    with time_machine.travel(datetime(2021, 1, 1, 0, 0, 1, tzinfo=timezone.utc)):
        start_datetime = get_start_datetime()

        assert start_datetime == datetime(2020, 12, 31, 23, 30, 0, 0, timezone.utc)

    with time_machine.travel(datetime(2021, 1, 1, 0, 45, 1, tzinfo=timezone.utc)):
        start_datetime = get_start_datetime()

        assert start_datetime == datetime(2021, 1, 1, 0, 0, 0, 0, timezone.utc)
