from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from forecast_blend.legacy_gsps import add_legacy_gsp_results, load_gsp_merge_weights


def make_gsp_data(gsp_id):

    # create some fake data
    return pd.DataFrame({
            "target_datetime_utc": pd.date_range("2023-01-01", periods=5, freq="30T"),
            "p10_mw": [1, 2, 3, 4, 5],
            "p50_mw": [gsp_id, 3, 4, 5, 6],
            "p90_mw": [3, 4, 5, 6, 7],
            "adjust_mw": [0, 0, 0, 0, 0],
        })



def test_add_legacy_gsp_results_none_added():
    # Add your test implementation here

    # create some fake data
    forecast_values_by_gsp_id = {}
    forecast_values_by_gsp_id[0] = make_gsp_data(0)
    forecast_values_by_gsp_id[1] = make_gsp_data(1)

    # Call the function to test
    result = add_legacy_gsp_results(forecast_values_by_gsp_id)

    # Add your assertions here
    assert len(result) == 2

    # check first timestamp is 2023-01-01
    assert result[0]["target_datetime_utc"].iloc[0] == pd.Timestamp("2023-01-01")
    assert result[1]["target_datetime_utc"].iloc[1] == pd.Timestamp("2023-01-01 00:30")

def test_add_legacy_gsp_results():
    # Add your test implementation here

    # create some fake data
    forecast_values_by_gsp_id = {}
    forecast_values_by_gsp_id[0] = make_gsp_data(0)
    forecast_values_by_gsp_id[350] = make_gsp_data(350)

    # Call the function to test
    result = add_legacy_gsp_results(forecast_values_by_gsp_id)

    # Add your assertions here
    assert len(result) == 3
    # check the first value in p50_mw is gsp_id
    print(result[350])
    assert result[350]["p50_mw"].iloc[0] == 350
    assert result[4]["p50_mw"].iloc[0] == 350


def test_add_legacy_gsp_results_all():
    # Add your test implementation here

    # create some fake data
    forecast_values_by_gsp_id = {}
    for i in range(343,352):
        forecast_values_by_gsp_id[i] = make_gsp_data(i)
    assert len(forecast_values_by_gsp_id) == 9

    # Call the function to test
    result = add_legacy_gsp_results(forecast_values_by_gsp_id)

    # check that the extra 6 gsps have been added
    assert len(result) == 15
    assert result[350]["p50_mw"].iloc[0] == 350
    assert result[4]["p50_mw"].iloc[0] == 350

    assert result[351]["p50_mw"].iloc[0] == 351
    assert result[56]["p50_mw"].iloc[0] == 351*.27


class _FakeChannel:
    """Stand-in for grpclib's Channel so app() can be run without a Data Platform."""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_app_skips_legacy_gsps_in_blend_loop():
    """app() should never try to blend a legacy gsp, and should reconstruct it from source.

    This runs the whole of app() with the Data Platform mocked out, so it needs no containers.
    gsp 4 is legacy and is rebuilt from gsp 350 with weight 1.0.
    """
    from forecast_blend.app import app

    requested_gsps = [0, 4, 350]

    def make_forecast(gsp_id):
        return pd.DataFrame({
            "target_datetime_utc": pd.date_range("2023-01-01", periods=4, freq="30min"),
            "p50_mw": [float(gsp_id)] * 4,
            "adjust_mw": [0.0] * 4,
        })

    blended_gsp_ids = []

    async def fake_blend(gsp_id, **kwargs):
        blended_gsp_ids.append(gsp_id)
        return make_forecast(gsp_id)

    weights_df = pd.DataFrame(
        {"pvnet_v2": [1.0]}, index=pd.to_datetime(["2023-01-01"], utc=True)
    )
    uuid_map = {
        gsp_id: {"location_uuid": f"uuid-{gsp_id}", "effective_capacity_watts": 1_000_000}
        for gsp_id in requested_gsps
    }
    save_mock = AsyncMock()

    with (
        patch("forecast_blend.app.Channel", _FakeChannel),
        patch("forecast_blend.app.dp"),
        patch("forecast_blend.app.get_blend_weights", new=AsyncMock(return_value=weights_df)),
        patch("forecast_blend.app.backfill_weights", return_value=weights_df),
        patch("forecast_blend.app.fetch_dp_gsp_uuid_map", new=AsyncMock(return_value=uuid_map)),
        patch("forecast_blend.app.get_blend_forecast_values_latest", new=fake_blend),
        patch("forecast_blend.app.get_metadata", new=AsyncMock(return_value=None)),
        patch("forecast_blend.app.save_forecast_to_data_platform", new=save_mock),
    ):
        await app(gsps=requested_gsps)

    # the legacy gsp is never blended directly, even though a forecast was available for it
    assert blended_gsp_ids == [0, 350]

    saved = save_mock.call_args.kwargs["forecast_values_by_gsp_id"]

    # but it is still saved, reconstructed from its source gsp
    assert 4 in saved
    assert saved[4]["p50_mw"].iloc[0] == 350.0
    assert len(saved[4]) == 4


def test_legacy_gsp_ids_are_not_forecast_by_models():
    """The legacy ids we skip are exactly the target ids in the weights config."""
    legacy_gsp_ids = set(load_gsp_merge_weights().keys())

    assert legacy_gsp_ids == {4, 41, 56, 122, 158, 257}

    # a target gsp must never also be a source, or skipping it would break another target
    source_ids = {
        source.gsp_id
        for config in load_gsp_merge_weights().values()
        for source in config.pvlive_merge_weights
    }
    assert not (legacy_gsp_ids & source_ids)

