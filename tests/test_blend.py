from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch
import pandas as pd
import pytest
from forecast_blend.blend import get_blend_forecast_values_latest


# Fake UUID map used in every test
FAKE_UUID_MAP = {
    0: {"location_uuid": "uuid-national", "effective_capacity_watts": 1_000_000_000},
    1: {"location_uuid": "uuid-gsp1", "effective_capacity_watts": 100_000_000},
}


def _make_df(model_name, rows):
    """Build a forecast DataFrame with the standard columns."""
    return pd.DataFrame(
        rows,
        columns=[
            "target_time",
            "expected_power_generation_megawatts",
            "adjust_mw",
            "created_utc",
            "properties",
            "model_name",
        ],
    ).assign(model_name=model_name)


def _now():
    return datetime.now(timezone.utc)


@pytest.mark.asyncio(loop_scope="session")
async def test_get_blend_forecast_values_latest_one_model():
    weights_df = pd.DataFrame(
        {
            "test_1": [1,1],
            "test_2": [0,0],
        }, 
        index=pd.to_datetime(
            [
                "2023-01-01 00:00", "2023-01-01 00:30",
            ], 
            utc=True,
        )
    )

    df1 = _make_df("test_1", [
        [datetime(2023, 1, 1, tzinfo=timezone.utc), 1, 0, _now(), {"10": 0.9, "90": 1.1}, "test_1"],
        [datetime(2023, 1, 1, 0, 30, tzinfo=timezone.utc), 1, 0, _now(), None, "test_1"],
    ])
    empty = _make_df("test_2", [])

    side_effects = [df1, empty]

    with patch("forecast_blend.blend.fetch_dp_latest_forecasts", new=AsyncMock(return_value=[])), \
         patch("forecast_blend.blend.get_forecast_values_from_data_platform",
               new=AsyncMock(side_effect=side_effects)):
        result = await get_blend_forecast_values_latest(
            gsp_id=0,
            weights_df=weights_df,
            start_datetime=datetime(2023, 1, 1, tzinfo=timezone.utc),
            gsp_uuid_map=FAKE_UUID_MAP,
            dp_client=AsyncMock(),
        )

    assert len(result) == 2
    assert result.iloc[0]["p50_mw"] == 1
    assert result.iloc[0]["target_datetime_utc"] == datetime(2023, 1, 1, tzinfo=timezone.utc)


@pytest.mark.asyncio(loop_scope="session")
async def test_get_blend_forecast_values_latest_two_model_read_one():
    weights_df = pd.DataFrame(
        {
            "test_1": [1,1],
            "test_2": [0,0],
        }, 
        index=pd.to_datetime(
            [
                "2023-01-01 00:00", "2023-01-01 00:30",
            ], 
            utc=True,
        )
    )

    df1 = _make_df("test_1", [
        [datetime(2023, 1, 1, tzinfo=timezone.utc), 1, 0, _now(), {"10": 0.9, "90": 1.1}, "test_1"],
        [datetime(2023, 1, 1, 0, 30, tzinfo=timezone.utc), 2, 0, _now(), {"10": 1.8, "90": 2.2}, "test_1"],
    ])
    df2 = _make_df("test_2", [
        [datetime(2023, 1, 1, tzinfo=timezone.utc), 1, 0, _now(), {"10": 0.9, "90": 1.1}, "test_2"],
        [datetime(2023, 1, 1, 0, 30, tzinfo=timezone.utc), 2, 0, _now(), {"10": 1.8, "90": 2.2}, "test_2"],
    ])

    with patch("forecast_blend.blend.fetch_dp_latest_forecasts", new=AsyncMock(return_value=[])), \
         patch("forecast_blend.blend.get_forecast_values_from_data_platform",
               new=AsyncMock(side_effect=[df1, df2])):
        result = await get_blend_forecast_values_latest(
            gsp_id=0,
            weights_df=weights_df,
            start_datetime=datetime(2023, 1, 1, tzinfo=timezone.utc),
            gsp_uuid_map=FAKE_UUID_MAP,
            dp_client=AsyncMock(),
        )

    assert len(result) == 2
    assert result.iloc[0]["p50_mw"] == 1
    assert result.iloc[1]["p50_mw"] == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_get_blend_forecast_values_latest_two_model_read_two():
    horizons_1 = [-60, -30, 0, 30, 8 * 30, 15 * 30]
    horizons_2 = [0, 30, 8 * 30, 15 * 30, 16 * 30]

    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)

    df1 = _make_df("test_1", [
        [t0 + timedelta(minutes=m), 1, 0, datetime(2023, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
         {"10": 0.9, "90": 1.1}, "test_1"]
        for m in horizons_1
    ])
    df2 = _make_df("test_2", [
        [t0 + timedelta(minutes=m), 3, 200, datetime(2023, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
         {"10": 2.7, "90": 3.3}, "test_2"]
        for m in horizons_2
    ])

    weights_df = pd.DataFrame(
        {
            "test_1": [1,1,1,1,.5,0,0],
            "test_2": [0,0,0,0,.5,1,1],
        }, 
        index=pd.to_datetime(
            [
                "2022-12-31 23:00", "2022-12-31 23:30", "2023-01-01 00:00", "2023-01-01 00:30",
                "2023-01-01 04:00", "2023-01-01 07:30", "2023-01-01 08:00",
            ], 
            utc=True,
        )
    )

    with patch("forecast_blend.blend.fetch_dp_latest_forecasts", new=AsyncMock(return_value=[])), \
         patch("forecast_blend.blend.get_forecast_values_from_data_platform",
               new=AsyncMock(side_effect=[df1, df2])):
        result = await get_blend_forecast_values_latest(
            gsp_id=1,
            weights_df=weights_df,
            start_datetime=datetime(2022, 12, 31, tzinfo=timezone.utc),
            gsp_uuid_map=FAKE_UUID_MAP,
            dp_client=AsyncMock(),
        )

    assert len(result) == 7
    assert result.iloc[0]["p50_mw"] == 1
    assert result.iloc[4]["p50_mw"] == 2   # blend at 50/50
    assert result.iloc[5]["p50_mw"] == 3


@pytest.mark.asyncio(loop_scope="session")
async def test_get_blend_forecast_values_latest_negative():
    """Blend with negative source values should run without error."""
    weights_df = pd.DataFrame(
        {
            "test_1": [1,1,.5,0,],
            "test_2": [0,0,.5,1,],
        }, 
        index=pd.to_datetime(
            [
                "2023-01-01 00:00", "2023-01-01 00:30",
                "2023-01-01 04:00", "2023-01-01 07:30",
            ], 
            utc=True,
        )
    )
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    horizons = [0, 30, 8 * 30, 15 * 30]

    df1 = _make_df("test_1", [
        [t0 + timedelta(minutes=m), -1, 1.0, _now(), {"10": 0.9, "90": 1.1}, "test_1"]
        for m in horizons
    ])
    df2 = _make_df("test_2", [
        [t0 + timedelta(minutes=m), -2, 2.0, _now(), {"10": 0.9, "90": 1.1}, "test_2"]
        for m in horizons
    ])

    with patch("forecast_blend.blend.fetch_dp_latest_forecasts", new=AsyncMock(return_value=[])), \
         patch("forecast_blend.blend.get_forecast_values_from_data_platform",
               new=AsyncMock(side_effect=[df1, df2])):
        result_df = await get_blend_forecast_values_latest(
            gsp_id=1,
            weights_df=weights_df,
            start_datetime=t0,
            gsp_uuid_map=FAKE_UUID_MAP,
            dp_client=AsyncMock(),
        )

    assert len(result_df) == 4


@pytest.mark.asyncio(loop_scope="session")
async def test_get_blend_forecast_values_latest_no_properties():
    """This test checks that when there are no _properties, an empty dictionary is returned"""


    weights_df = pd.DataFrame(
        {
            "cnn": [1,1,.5,0,],
            "National_xg": [0,0,.5,1,],
        }, 
        index=pd.to_datetime(
            [
                "2023-01-01 00:00", "2023-01-01 00:30",
                "2023-01-01 04:00", "2023-01-01 07:30",
            ], 
            utc=True,
        )
    )

    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    horizons = [0, 30, 8 * 30, 15 * 30]

    df1 = _make_df("cnn", [
        [t0 + timedelta(minutes=m), 1, 0, _now(), {}, "cnn"] for m in horizons
    ])
    df2 = _make_df("National_xg", [
        [t0 + timedelta(minutes=m), 1, 0, _now(), {}, "National_xg"] for m in horizons
    ])

    with patch("forecast_blend.blend.fetch_dp_latest_forecasts", new=AsyncMock(return_value=[])), \
         patch("forecast_blend.blend.get_forecast_values_from_data_platform",
               new=AsyncMock(side_effect=[df1, df2])):
        result_df = await get_blend_forecast_values_latest(
            gsp_id=0,
            weights_df=weights_df,
            start_datetime=t0,
            gsp_uuid_map=FAKE_UUID_MAP,
            dp_client=AsyncMock(),
        )

    assert len(result_df) == 4


@pytest.mark.asyncio(loop_scope="session")
async def test_get_blend_forecast_three_models():
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)

    weights_df = pd.DataFrame(
        {
            "test_1": [1,1,0,0, 0,],
            "test_2": [0,0,0,0,.5,],
            "test_3": [0,0,1,1,.5,],
        }, 
        index=pd.to_datetime(
            [
                "2023-01-01 00:00", "2023-01-01 00:30",
                "2023-01-01 02:00", "2023-01-01 07:00", "2023-01-01 08:00",
            ], 
            utc=True,
        )
    )
    

    df1 = _make_df("test_1", [
        [t0 + timedelta(minutes=m), 1, 0, _now(), {}, "test_1"]
        for m in [-60, -30, 0, 30, 2 * 60, 7 * 60]
    ])
    df2 = _make_df("test_2", [
        [t0 + timedelta(minutes=m), 2, 50, _now(), {}, "test_2"]
        for m in [0, 30, 2 * 60, 7 * 60, 8 * 60]
    ])
    df3 = _make_df("test_3", [
        [t0 + timedelta(minutes=m), 3, 100, _now(), {}, "test_3"]
        for m in [0, 30, 2 * 60, 7 * 60, 8 * 60]
    ])

    with patch("forecast_blend.blend.fetch_dp_latest_forecasts", new=AsyncMock(return_value=[])), \
         patch("forecast_blend.blend.get_forecast_values_from_data_platform",
               new=AsyncMock(side_effect=[df1, df2, df3])):
        result = await get_blend_forecast_values_latest(
            gsp_id=1,
            weights_df=weights_df,
            start_datetime=t0,
            gsp_uuid_map=FAKE_UUID_MAP,
            dp_client=AsyncMock(),
        )

    assert len(result) == 5
    assert result.iloc[0]["p50_mw"] == 1.0
    assert result.iloc[1]["p50_mw"] == 1.0
    assert result.iloc[2]["p50_mw"] == 3.0
    assert result.iloc[3]["p50_mw"] == 3.0
    assert result.iloc[4]["p50_mw"] == 2.5


@pytest.mark.asyncio(loop_scope="session")
async def test_get_blend_forecast_three_models_with_gap():
    """
    The idea of this test its to make a gap in the one of the forecast,
    In this gap we set the weights to load that model.

    The behaviour we hope is then to use future weights to fill the gap.
    :return:
    """
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)

    weights_df = pd.DataFrame(
        {
            "test_1": [1,1,0,0,0,],
            "test_2": [0,0,0,0,.5,],
            "test_3": [0,0,1,1,.5,],
        }, 
        index=pd.to_datetime(
            [
                "2023-01-01 00:00", "2023-01-01 00:30",
                "2023-01-01 02:00", "2023-01-01 07:00", "2023-01-01 08:00",
            ], 
            utc=True,
        )
    )

    # test_1 only has t=-60 (a gap at everything else)
    df1 = _make_df("test_1", [
        [t0 + timedelta(minutes=-60), 1, 0, _now(), {}, "test_1"],
    ])
    df2 = _make_df("test_2", [
        [t0 + timedelta(minutes=m), 2, 50, _now(), {}, "test_2"]
        for m in [0, 30, 2 * 60, 7 * 60, 8 * 60]
    ])
    df3 = _make_df("test_3", [
        [t0 + timedelta(minutes=m), 3, 100, _now(), {}, "test_3"]
        for m in [0, 30, 2 * 60, 7 * 60, 8 * 60]
    ])

    with patch("forecast_blend.blend.fetch_dp_latest_forecasts", new=AsyncMock(return_value=[])), \
         patch("forecast_blend.blend.get_forecast_values_from_data_platform",
               new=AsyncMock(side_effect=[df1, df2, df3])):
        result = await get_blend_forecast_values_latest(
            gsp_id=1,
            weights_df=weights_df,
            start_datetime=t0,
            gsp_uuid_map=FAKE_UUID_MAP,
            dp_client=AsyncMock(),
        )

    assert len(result) == 5
    assert result.iloc[0]["p50_mw"] == 3.0
    assert result.iloc[4]["p50_mw"] == 2.5
