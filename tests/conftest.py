import os
from datetime import datetime, timedelta, timezone
import time_machine

import pytest
from testcontainers.postgres import PostgresContainer

from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.fake import make_fake_forecasts
from nowcasting_datamodel.save.save import save

from forecast_blend.weights import model_names


@pytest.fixture
@time_machine.travel("2023-01-01 00:00:01")
def forecasts(db_session):
    t0_datetime_utc = datetime.now(tz=timezone.utc) + timedelta(days=2)
    # time detal of 2 days is used as fake forecast are made 2 days in the past,
    # this makes them for now
    # create
    for model_name in model_names:

        if model_name == "National_xg":
            gsp_ids = [0]
        else:
            gsp_ids = list(range(0, 11))

        f = make_fake_forecasts(
            gsp_ids=gsp_ids,
            session=db_session,
            model_name=model_name,
            n_fake_forecasts=16,
            t0_datetime_utc=t0_datetime_utc,
        )
        

        save(forecasts=f, session=db_session, apply_adjuster=False)

    return None

@pytest.fixture
@time_machine.travel("2023-01-01 00:00:00")
def forecast_national(db_session):
    t0_datetime_utc = datetime.now(tz=timezone.utc) + timedelta(days=2)
    # time detal of 2 days is used as fake forecast are made 2 days in the past,
    # this makes them for now
    # create
    for model_name in model_names:

        gsp_ids = [0]

        f = make_fake_forecasts(
            gsp_ids=gsp_ids,
            session=db_session,
            model_name=model_name,
            n_fake_forecasts=16,
            t0_datetime_utc=t0_datetime_utc,
        )

        save(forecasts=f, session=db_session, apply_adjuster=False)

    return None

@pytest.fixture
@time_machine.travel("2023-01-01 00:00:00")
def forecast_national_ecmwf_and_xg(db_session):
    t0_datetime_utc = datetime.now(tz=timezone.utc)
    # time detal of 2 days is used as fake forecast are made 2 days in the past,
    # this makes them for now
    # create
    model_names_ecmwf_and_xg = ["pvnet_ecmwf", "National_xg"]
    for i, model_name in enumerate(model_names_ecmwf_and_xg):

        gsp_ids = [0]

        forecasts = make_fake_forecasts(
            gsp_ids=gsp_ids,
            session=db_session,
            model_name=model_name,
            n_fake_forecasts=120,
            t0_datetime_utc=t0_datetime_utc,
        )
        for f in forecasts:
            for fv in f.forecast_values:
                fv.expected_power_generation_megawatts = i

        save(forecasts=forecasts, session=db_session, apply_adjuster=False)

    return None


"""
This is a bit complicated and sensitive to change
https://gist.github.com/kissgyorgy/e2365f25a213de44b9a2 helped me get going
"""


@pytest.fixture(scope="session")
def engine_url():
    """Database engine, this includes the table creation."""
    with PostgresContainer("postgres:14.5") as postgres:
        url = postgres.get_connection_url()
        os.environ["DB_URL"] = url

        database_connection = DatabaseConnection(url, echo=False)

        engine = database_connection.engine

        # Would like to do this here but found the data
        # was not being deleted when using 'db_connection'
        # database_connection.create_all()
        # Base_PV.metadata.create_all(engine)

        yield url

        # Base_PV.metadata.drop_all(engine)
        # Base_Forecast.metadata.drop_all(engine)

        engine.dispose()


@pytest.fixture()
def db_connection(engine_url):
    database_connection = DatabaseConnection(engine_url, echo=False)

    engine = database_connection.engine
    # connection = engine.connect()
    # transaction = connection.begin()

    # There should be a way to only make the tables once
    # but make sure we remove the data
    database_connection.create_all()

    yield database_connection

    # transaction.rollback()
    # connection.close()

    database_connection.drop_all()


@pytest.fixture(scope="function", autouse=True)
def db_session(db_connection, engine_url):
    """Creates a new database session for a test."""

    # connection = db_connection.engine.connect()
    # t = connection.begin()

    with db_connection.Session() as s:
        s.begin()
        yield s
        s.rollback()

    # t.rollback()
    # connection.close()
