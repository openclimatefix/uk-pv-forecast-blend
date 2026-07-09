""" Main application

For each GSP
1. Load the various forecast
2. Blend them together
3. Save them to Data Platform

"""

import asyncio
import os
import sys
import sentry_sdk
from loguru import logger
from ocf import dp
from grpclib.client import Channel


from forecast_blend.blend import get_blend_forecast_values_latest
from forecast_blend.legacy_gsps import add_legacy_gsp_results
from forecast_blend.utils import get_start_datetime
from forecast_blend.weights import (
    backfill_weights, 
    get_national_blend_weights, 
    get_regional_blend_weights,
)
from forecast_blend.save import fetch_dp_gsp_uuid_map, save_forecast_to_data_platform, get_metadata
from forecast_blend.forecast.data_platform import get_data_platform_connection
import pandas as pd

__version__ = "1.1.9"

# GB has 342 GSPs
N_GSP = 342

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("ENVIRONMENT", "local"),
    traces_sample_rate=1
)

sentry_sdk.set_tag("app_name", "uk_pv_forecast_blend")
sentry_sdk.set_tag("version", __version__)

logger.remove(0)
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))


async def app(gsps: list[int] | None = None) -> None:
    """run main app"""

    blend_name = os.getenv("BLEND_NAME", "blend")
    allow_cloudcasting = os.getenv("ALLOW_CLOUDCASTING", "false").lower()=="true"

    exclude_models = None if allow_cloudcasting else ["pvnet_cloud"]

    if gsps is None:
        n_gsps = int(os.getenv("N_GSP", N_GSP))
        n_gsps = min([n_gsps, N_GSP])

        gsps = range(0, n_gsps + 1)

    start_datetime = get_start_datetime()
    t0 = pd.Timestamp.utcnow().floor("30min")

    host, port = get_data_platform_connection()
    forecast_values_by_gsp_id = {}
    async with Channel(host=host, port=port) as channel:
        client = dp.DataPlatformDataServiceStub(channel)

        national_weights_df = await get_national_blend_weights(client, t0, exclude_models)
        regional_weights_df = await get_regional_blend_weights(client, t0, exclude_models)

        national_weights_df = backfill_weights(national_weights_df, start_datetime)
        regional_weights_df = backfill_weights(regional_weights_df, start_datetime)

        logger.info(f"Weights for national blend: {national_weights_df}")
        logger.info(f"Weights for regional blend: {regional_weights_df}")
        gsp_uuid_map = await fetch_dp_gsp_uuid_map(client=client)

        for gsp_id in gsps:
            logger.info(f"Blending forecasts for gsp id {gsp_id}")
            try:
                forecast_values_df = await get_blend_forecast_values_latest(
                    gsp_id=gsp_id,
                    start_datetime=start_datetime,
                    weights_df=national_weights_df if gsp_id == 0 else regional_weights_df,
                    gsp_uuid_map=gsp_uuid_map,
                    dp_client=client,
                )
                forecast_values_by_gsp_id[gsp_id] = forecast_values_df
            except Exception as e:
                logger.exception(f"Failed to blend forecasts for gsp_id {gsp_id}")
                logger.debug(f"Exception: {e}")

            # add legacy gsps results
            forecast_values_by_gsp_id = add_legacy_gsp_results(forecast_values_by_gsp_id)

        # Save to Data Platform
        logger.info("Saving forecast to data platform")
        metadata = await get_metadata(client=client, location_uuid=gsp_uuid_map[0]["location_uuid"])
        _ = await save_forecast_to_data_platform(
            forecast_values_by_gsp_id=forecast_values_by_gsp_id,
            locations_uuid_and_capacity_by_gsp_id=gsp_uuid_map,
            model_tag=blend_name,
            init_time_utc=t0.to_pydatetime(),
            client=client,
            metadata=metadata,
        )
    logger.info("Finished")


if __name__ == "__main__":
    asyncio.run(app())
