import os
import time
from importlib.metadata import version

import pytest
import pytest_asyncio
from grpclib.client import Channel
from ocf import dp
from testcontainers.postgres import PostgresContainer
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network


# Arbitrarily set the blend name so we can test it is properly set throughout tests
os.environ["BLEND_NAME"] = "test_blend_name"


@pytest.fixture(scope="session")
def dp_client():
    """Spin up a single shared Data Platform gRPC server for the entire test session.

    Yields (host, port) only. Callers must create their own Channel+stub within
    their own async event loop to avoid 'Future attached to a different loop' errors.
    Sets DATA_PLATFORM_HOST and DATA_PLATFORM_PORT env vars so app.py connects to it.
    """
    # Both containers share a network so the data-platform server can reach postgres
    # by its "db" alias. get_connection_url() would return the host-exposed port,
    # which isn't what postgres listens on from inside the network.
    with Network() as network:
        with PostgresContainer(
            f"ghcr.io/openclimatefix/data-platform-pgdb:{version('dp_sdk')}",
            username="postgres",
            password="postgres",  # noqa: S106
            dbname="postgres",
            env={"POSTGRES_HOST": "db"},
            network=network,
            network_aliases=["db"],
        ) as postgres:
            database_url = f"postgres://postgres:postgres@db:{postgres.port}/postgres"

            with DockerContainer(
                image=f"ghcr.io/openclimatefix/data-platform:{version('dp_sdk')}",
                env={"DATABASE_URL": database_url},
                ports=[50051],
                platform="linux/amd64",
                network=network,
            ) as data_platform_server:
                time.sleep(2)  # Give some time for the server to start

                port = data_platform_server.get_exposed_port(50051)
                host = data_platform_server.get_container_host_ip()

                # Set env vars so app.py connects to the test container
                os.environ["DATA_PLATFORM_HOST"] = host
                os.environ["DATA_PLATFORM_PORT"] = str(port)

                yield host, port


@pytest_asyncio.fixture(scope="session")
async def data_client(dp_client):
    host, port = dp_client
    channel = Channel(host=host, port=port)
    yield dp.DataPlatformDataServiceStub(channel)
    channel.close()
