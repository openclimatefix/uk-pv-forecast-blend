import os
from typing import List

import pytest

from nowcasting_datamodel import N_GSP
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.fake import make_fake_forecasts, make_fake_me_latest
from nowcasting_datamodel.models import MetricValueSQL
from nowcasting_datamodel.models.forecast import ForecastSQL
from nowcasting_datamodel.models.pv import Base_PV
from nowcasting_datamodel.save import save




@pytest.fixture
def forecasts(db_session) -> List[ForecastSQL]:
    # create
    for model_name in ['cnn','National_xg','pvnet_v2']:

        if model_name == 'National_xg':
            gsp_ids = [0]
        else:
            gsp_ids = list(range(0, 10))

        f = make_fake_forecasts(gsp_ids=gsp_ids,
                                session=db_session,
                                model_name=model_name, # add
                                n_fake_forecasts=16) # add

        save(f)

    # add
    db_session.add_all(f)

    return f




"""
This is a bit complicated and sensitive to change
https://gist.github.com/kissgyorgy/e2365f25a213de44b9a2 helped me get going
"""


@pytest.fixture
def db_connection():
    url = os.getenv("DB_URL", "sqlite:///test.db")

    connection = DatabaseConnection(url=url, echo=True)
    connection.create_all()

    yield connection

    connection.drop_all()


@pytest.fixture(scope="function", autouse=True)
def db_session(db_connection):
    """Creates a new database session for a test."""

    connection = db_connection.engine.connect()
    t = connection.begin()

    with db_connection.Session(bind=connection) as s:
        s.begin()
        yield s
        s.rollback()

    t.rollback()
    connection.close()



