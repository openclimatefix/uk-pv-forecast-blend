from forecast_blend.legacy_gsps import add_legacy_gsp_results
import pandas as pd


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

