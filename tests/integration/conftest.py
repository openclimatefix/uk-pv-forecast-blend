"""Integration test conftest.

All async fixtures create their own Channel within the correct async event loop
to avoid 'Future attached to a different loop' errors. The shared `dp_client`
fixture (tests/conftest.py) only provides (host, port); channels are never
created in sync context.
"""

import datetime

import pytest_asyncio
from betterproto.lib.google.protobuf import Struct, Value
from dp_sdk.ocf import dp
from grpclib.client import Channel


@pytest_asyncio.fixture(scope="session")
async def client(dp_client):
    """Session-scoped async client for test_save_to_data_platform.py.

    Creates Channel inside the session event loop so there is no loop mismatch.
    """
    host, port = dp_client
    channel = Channel(host=host, port=port)
    stub = dp.DataPlatformDataServiceStub(channel)
    yield stub
    channel.close()


@pytest_asyncio.fixture(scope="session")
async def setup_test_data(dp_client):
    """Set up test locations and forecasters in the shared Data Platform container.

    Creates its own Channel within the session event loop.
    Returns location UUIDs and forecaster info for use in integration tests.
    """
    host, port = dp_client
    channel = Channel(host=host, port=port)
    client = dp.DataPlatformDataServiceStub(channel)

    try:
        # Create test location for GSP 0 (national)
        metadata_gsp0 = Struct(fields={"gsp_id": Value(number_value=0)})
        response = await client.create_location(
            dp.CreateLocationRequest(
                location_name="test_national_gsp",
                energy_source=dp.EnergySource.SOLAR,
                geometry_wkt="POLYGON((-2 52, -1 52, -1 53, -2 53, -2 52))",
                location_type=dp.LocationType.GSP,
                effective_capacity_watts=1_000_000_000,  # 1 GW
                metadata=metadata_gsp0,
                valid_from_utc=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
            )
        )
        location_uuid_gsp0 = response.location_uuid

        # Create test location for GSP 1 (regional)
        metadata_gsp1 = Struct(fields={"gsp_id": Value(number_value=1)})
        response = await client.create_location(
            dp.CreateLocationRequest(
                location_name="test_regional_gsp_1",
                energy_source=dp.EnergySource.SOLAR,
                geometry_wkt="POLYGON((-3 51, -2 51, -2 52, -3 52, -3 51))",
                location_type=dp.LocationType.GSP,
                effective_capacity_watts=100_000_000,  # 100 MW
                metadata=metadata_gsp1,
                valid_from_utc=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
            )
        )
        location_uuid_gsp1 = response.location_uuid

        # Create forecasters for different models
        forecasters = {}
        for model_name in ["pvnet_day_ahead", "national_xg", "pvnet_v2"]:
            response = await client.create_forecaster(
                dp.CreateForecasterRequest(name=model_name, version="1.0.0")
            )
            forecasters[model_name] = response.forecaster

        yield {
            "host": host,
            "port": port,
            "locations": {
                0: {"uuid": location_uuid_gsp0, "capacity_watts": 1_000_000_000},
                1: {"uuid": location_uuid_gsp1, "capacity_watts": 100_000_000},
            },
            "forecasters": forecasters,
        }
    finally:
        channel.close()
