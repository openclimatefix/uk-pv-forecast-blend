import os
import time
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import time_machine

import pytest
from testcontainers.postgres import PostgresContainer
from testcontainers.core.container import DockerContainer

from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.fake import make_fake_forecasts
from nowcasting_datamodel.save.save import save

from forecast_blend.weights import ALL_MODEL_NAMES


@pytest.fixture(scope="session")
def dp_client():
    """Spin up a single shared Data Platform gRPC server for the entire test session.

    Yields (host, port) only. Callers must create their own Channel+stub within
    their own async event loop to avoid 'Future attached to a different loop' errors.
    Sets DATA_PLATFORM_HOST and DATA_PLATFORM_PORT env vars so app.py connects to it.
    """
    with PostgresContainer(
        f"ghcr.io/openclimatefix/data-platform-pgdb:{version('dp_sdk')}",
        username="postgres",
        password="postgres",  # noqa: S106
        dbname="postgres",
        env={"POSTGRES_HOST": "db"},
    ) as postgres:
        database_url = postgres.get_connection_url()
        database_url = database_url.replace("postgresql+psycopg2", "postgres")
        database_url = database_url.replace("localhost", "host.docker.internal")

        with DockerContainer(
            image=f"ghcr.io/openclimatefix/data-platform:{version('dp_sdk')}",
            env={"DATABASE_URL": database_url},
            ports=[50051],
            platform="linux/amd64",
        ) as data_platform_server:
            time.sleep(2)  # Give some time for the server to start

            port = data_platform_server.get_exposed_port(50051)
            host = data_platform_server.get_container_host_ip()

            # Set env vars so app.py connects to the test container
            os.environ["DATA_PLATFORM_HOST"] = host
            os.environ["DATA_PLATFORM_PORT"] = str(port)

            yield host, port
# Arbitrarily set the blend name so we can test it is properly set throughout tests
os.environ["BLEND_NAME"] = "test_blend_name"


@pytest.fixture
@time_machine.travel("2023-01-01 00:00:01")
def forecasts(db_session):
    t0_datetime_utc = datetime.now(tz=timezone.utc) + timedelta(days=2)
    # time delay of 2 days is used as fake forecast are made 2 days in the past,
    # this makes them for now
    # create
    for model_name in ALL_MODEL_NAMES:

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


@pytest.fixture
@time_machine.travel("2023-01-01 00:00:00")
def forecast_national(db_session):
    t0_datetime_utc = datetime.now(tz=timezone.utc) + timedelta(days=2)
    # time delay of 2 days is used as fake forecast are made 2 days in the past,
    # this makes them for now
    # create
    for model_name in ALL_MODEL_NAMES:

        gsp_ids = [0]

        f = make_fake_forecasts(
            gsp_ids=gsp_ids,
            session=db_session,
            model_name=model_name,
            n_fake_forecasts=16,
            t0_datetime_utc=t0_datetime_utc,
        )

        save(forecasts=f, session=db_session, apply_adjuster=False)


@pytest.fixture
@time_machine.travel("2023-01-01 00:00:00")
def forecast_national_ecmwf_and_xg(db_session):
    t0_datetime_utc = datetime.now(tz=timezone.utc)
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


@pytest.fixture
@time_machine.travel("2023-01-01 00:00:00")
def forecast_national_all_now(db_session):
    t0_datetime_utc = datetime.now(tz=timezone.utc)

    for model_name in ALL_MODEL_NAMES:

        gsp_ids = [0]

        f = make_fake_forecasts(
            gsp_ids=gsp_ids,
            session=db_session,
            model_name=model_name,
            n_fake_forecasts=16,
            t0_datetime_utc=t0_datetime_utc,
        )

        save(forecasts=f, session=db_session, apply_adjuster=False)


# This is a bit complicated and sensitive to change
# https://gist.github.com/kissgyorgy/e2365f25a213de44b9a2 helped me get going
@pytest.fixture(scope="session")
def engine_url():
    """Database engine, this includes the table creation."""
    with PostgresContainer("postgres:17.6") as postgres:
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

    # engine = database_connection.engine
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
