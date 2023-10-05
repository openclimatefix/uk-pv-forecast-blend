""" Main application

For each GSP
1. Load the various forecast
2. Blend them together
3. Save them to the database

"""

import os
import json
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
from nowcasting_datamodel.save.save import save
from nowcasting_datamodel.save.update import N_GSP, update_all_forecast_latest

from blend import get_blend_forecast_values_latest
from weights import weights

logger = structlog.stdlib.get_logger()

__version__ = "0.0.16"


def app(gsps: List[int] = None):
    """run main app"""

    if gsps is None:
        gsps = range(0, N_GSP + 1)

    # make connection to database
    connection = DatabaseConnection(url=os.getenv("DB_URL", "not_set"), echo=False)

    # get utc now minus 1 hour, for the start time of these blending
    start_datetime = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    # round up to nearest 30 minutes
    if start_datetime.minute < 30:
        start_datetime = start_datetime.replace(minute=30, second=0, microsecond=0)
    else:
        start_datetime = start_datetime.replace(
            hour=start_datetime.hour + 1, minute=0, second=0, microsecond=0
        )

    with connection.get_session() as session:

        model = get_blend_model(session)
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
        if is_last_forecast_made_before_last_30_minutes_step(session=session):
            logger.debug(f"Saving {len(forecasts)} forecasts")
            save(session=session, forecasts=forecasts, apply_adjuster=False)
        else:
            logger.debug(
                f"Saving {len(forecasts[0].forecast_values)} forecast values to latest table for blended model"
            )
            update_all_forecast_latest(
                forecasts=forecasts,
                session=session,
                update_national=True,
                update_gsp=True,
            )


def get_blend_model(session):
    """Get the blend model

    The version is made up of all the models version for example
    version = {"cnn": "0.0.1", "National_xg": "0.0.1", "pvnet_v2": "0.0.1", "blend": "0.0.1"}
    """
    # get all model versions
    models = {}
    for model_name in ["cnn", "National_xg", "pvnet_v2"]:
        model = get_model(name=model_name, session=session)
        models[model_name] = model.version

    # add blend version
    models['blend'] = __version__
    all_version = json.dumps(models)

    # get model object from database
    model = get_model(name="blend", version=all_version, session=session)
    return model


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

        if isinstance(forecast_value._properties, dict):
            forecast_value_sql.properties = forecast_value._properties

        forecast_values_sql.append(forecast_value_sql)

    return ForecastSQL(
        model=model,
        forecast_creation_time=datetime.now(tz=timezone.utc),
        location=location,
        input_data_last_updated=input_data_last_updated,
        forecast_values=forecast_values_sql,
        historic=False,
    )


def is_last_forecast_made_before_last_30_minutes_step(session):
    """
    Save the forecast to the database every 30 minutes

    This is beasue if we ran this evyer time, there would be double the about of
    Forecast, ForecastValues and ForecastValluesLastSevenDays
    """

    query = session.query(ForecastSQL)
    query = query.join(MLModelSQL)
    query = query.filter(MLModelSQL.name == "blend")
    query = query.filter(ForecastSQL.historic == False)
    query = query.order_by(ForecastSQL.forecast_creation_time.desc())
    last_forecast = query.limit(1).all()

    update_forecasts = False
    if len(last_forecast) == 0:
        logger.debug("Could not find any forecasts so will be saving full forecast")
        update_forecasts = True
    else:

        last_forecast = last_forecast[0]

        # round down to the nearest 30 minutes
        limit_creation_time = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
        limit_creation_time = limit_creation_time - timedelta(
            minutes=limit_creation_time.minute % 30
        )
        creation_time = last_forecast.forecast_creation_time
        if creation_time < limit_creation_time:
            logger.debug(
                f"Last forecast was made at {creation_time} so will be saving full forecast"
            )
            update_forecasts = True
        else:
            logger.info(
                f"Not updating forecasts as they have been updated recently ({creation_time})"
            )

    return update_forecasts


if __name__ == "__main__":
    app()
