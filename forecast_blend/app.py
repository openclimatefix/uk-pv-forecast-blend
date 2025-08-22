""" Main application

For each GSP
1. Load the various forecast
2. Blend them together
3. Save them to the database

"""

import os
import json
from datetime import datetime, timedelta, timezone
import sentry_sdk
from loguru import logger

from sqlalchemy.orm.session import Session

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
)
from nowcasting_datamodel.read.read_models import get_model
from nowcasting_datamodel.save.save import save
from nowcasting_datamodel.save.update import N_GSP, update_all_forecast_latest

from blend import get_blend_forecast_values_latest
from utils import get_start_datetime
from weights import (
    model_names,
    backfill_weights, 
    get_national_blend_weights, 
    get_regional_blend_weights,
)
import pandas as pd

__version__ = "1.1.5"

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("ENVIRONMENT", "local"),
    traces_sample_rate=1
)

sentry_sdk.set_tag("app_name", "uk_pv_forecast_blend")
sentry_sdk.set_tag("version", __version__)


def app(gsps: list[int] | None = None) -> None:
    """run main app"""

    if gsps is None:
        n_gsps = int(os.getenv("N_GSP", N_GSP))
        n_gsps = min([n_gsps, N_GSP])

        gsps = range(0, n_gsps + 1)

    # make connection to database
    connection = DatabaseConnection(url=os.getenv("DB_URL", "not_set"), echo=False)

    start_datetime = get_start_datetime()
    t0 = pd.Timestamp.utcnow().floor("30min")


    with connection.get_session() as session:

        model = get_blend_model(session)

        national_weights_df = get_national_blend_weights(session, t0)
        regional_weights_df = get_regional_blend_weights(session, t0)

        national_weights_df = backfill_weights(national_weights_df, start_datetime)
        regional_weights_df = backfill_weights(regional_weights_df, start_datetime)

        logger.info(f"Weights for national blend: {national_weights_df}")
        logger.info(f"Weights for regional blend: {regional_weights_df}")

        # Get the latest input data
        input_data_last_updated = get_latest_input_data_last_updated(session=session)
        # This is not quite right as the forecast could have been made with an earlier version,
        # but I think its the best we can do right now

        forecasts = []
        for gsp_id in gsps:
            logger.info(f"Blending forecasts for gsp id {gsp_id}")
            try:

                location = get_location(session=session, gsp_id=gsp_id)

                # 1. and 2. load and blend forecast values together
                forecast_values = get_blend_forecast_values_latest(
                    session=session,
                    gsp_id=gsp_id,
                    start_datetime=start_datetime,
                    weights_df=national_weights_df if gsp_id == 0 else regional_weights_df,
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
            except Exception as e:
                logger.exception(f"Failed to blend forecasts for gsp_id {gsp_id}")
                logger.debug(f"Exception: {e}")

        # 3. save to database
        # save to forecast_value_latest table, and not to the
        # - forecast_value_last_seven_days
        # - forecast_value
        # tables, as we will end up doubling the size of this table.
        assert len(forecasts) > 0, "No forecasts made"
        assert len(forecasts[0].forecast_values) > 0, "No forecast values sql made"
        if is_last_forecast_made_before_last_30_minutes_step(session=session):
            logger.debug(f"Saving {len(forecasts)} forecasts")
            save(
                session=session,
                forecasts=forecasts,
                apply_adjuster=False,
                remove_non_distinct_last_seven_days=True,
            )
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

    logger.info("Finished")


def get_blend_model(session: Session) -> MLModelSQL:
    """Get the blend model

    The version is made up of all the models version for example
    version = {"cnn": "0.0.1", "National_xg": "0.0.1", "pvnet_v2": "0.0.1", "blend": "0.0.1"}
    """
    # get all model versions
    models = {}
    for model_name in model_names:
        model = get_model(name=model_name, session=session)
        models[model_name] = model.version

    # add blend version
    models["blend"] = __version__
    all_version = json.dumps(models)

    # get model object from database
    return get_model(name="blend", version=all_version, session=session)


def make_forecast(
    forecast_values: list[ForecastValue],
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


def is_last_forecast_made_before_last_30_minutes_step(session: Session):
    """
    Save the forecast to the database every 30 minutes

    This is beasue if we ran this evyer time, there would be double the about of
    Forecast, ForecastValues and ForecastValluesLastSevenDays
    """

    one_week_ago = datetime.now(tz=timezone.utc) - timedelta(days=7)

    query = session.query(ForecastSQL)
    query = query.join(MLModelSQL)
    query = query.filter(MLModelSQL.name == "blend")
    query = query.filter(ForecastSQL.historic == False)
    query = query.filter(ForecastSQL.created_utc > one_week_ago)
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
