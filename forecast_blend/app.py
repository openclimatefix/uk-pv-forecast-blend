""" Main application

For each GSP
1. Load the various forecast
2. Blend them together
3. Save them to the database

"""

import os
from datetime import datetime, timedelta, timezone
from typing import List

import structlog
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models import (
    ForecastValue,
    ForecastSQL,
    MLModelSQL,
    LocationSQL,
    InputDataLastUpdatedSQL,
)
from nowcasting_datamodel.read.read import (
    get_latest_input_data_last_updated,
    get_location,
    get_model,
)
from nowcasting_datamodel.save.update import N_GSP, update_all_forecast_latest

from blend import get_blend_forecast_values_latest
from weights import weights

logger = structlog.stdlib.get_logger()

__version__ = "0.0.2"


def app(gsps: List[int] = None):
    """run main app"""

    if gsps is None:
        gsps = range(0, N_GSP)

    # make connection to database
    connection = DatabaseConnection(url=os.getenv("DB_URL", "not_set"))

    # get utc now minus 1 hour, for the start time of these blending
    start_datetime = datetime.now(tz=timezone.utc) - timedelta(hours=1)

    with connection.get_session() as session:

        model = get_model(name="blend", version=__version__, session=session)
        # This is not quite right as the model could have been made with a earlier version,
        # but I think its the best we can do
        input_data_last_updated = get_latest_input_data_last_updated(session=session)

        forecasts = []
        for gsp_id in gsps:
            logger.info(f"Blending forecasts for gsp id {gsp_id}")

            location = get_location(session=session, gsp_id=gsp_id)

            # 1. and 2. load and blend forecast values together
            forecast_values = get_blend_forecast_values_latest(
                session=session,
                gsp_id=gsp_id,
                start_datetime=start_datetime,
                properties_model="National_xg",
                weights=weights,
                model_names=["cnn", "National_xg", "pvnet_v2"],
            )

            # make Forecast SQL
            assert len(forecast_values) > 0, "No forecast values made"
            forecast = make_forecast(
                forecast_values=forecast_values,
                location=location,
                model=model,
                input_data_last_updated=input_data_last_updated,
            )
            forecasts.append(forecast)

        # 3. save to database
        # save to forecast_value_latest table, and not to the
        # - forecast_value_last_seven_days
        # - forecast_value
        # tables, as we will end up doubling the size of this table.
        assert len(forecasts) > 0, "No forecasts made"
        assert len(forecasts[0].forecast_values) > 0, "No forecast values sql made"
        logger.debug(
            f"Saving {len(forecasts[0].forecast_values)} forecast values to latest table for blended model"
        )
        update_all_forecast_latest(
            forecasts=forecasts,
            session=session,
            update_national=True,
            update_gsp=True,
        )

        # future save to forecast_value_last_seven_days, but remove anything made in the
        # last within the last 30 minute period. This is because when loading the X hour forecast, only the latest
        #  value in that settlement period is loaded.


def make_forecast(
    forecast_values: List[ForecastValue],
    model: MLModelSQL,
    location: LocationSQL,
    input_data_last_updated: InputDataLastUpdatedSQL,
) -> ForecastSQL:
    """
    Make a forecast object from a list of forecast values

    :param forecast_values: list of ForecastValue values
    :param model: the model object
    :param location:the location object
    :param input_data_last_updated: the input_data_last_updated object
    :return: a Forecast object.
    """

    forecast_values_sql = []
    for forecast_value in forecast_values:

        forecast_value_sql = forecast_value.to_orm()
        forecast_values_sql.append(forecast_value_sql)

    return ForecastSQL(
        model=model,
        forecast_creation_time=datetime.now(tz=timezone.utc),
        location=location,
        input_data_last_updated=input_data_last_updated,
        forecast_values=forecast_values_sql,
        historic=False,
    )


if __name__ == "__main__":
    app(list(range(0,10)))
