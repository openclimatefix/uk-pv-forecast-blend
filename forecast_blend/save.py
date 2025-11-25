"""Functions to save forecasts to the database."""

import asyncio
import itertools
import logging
from datetime import UTC, datetime
from importlib.metadata import version

import betterproto
import pandas as pd
from betterproto.lib.google.protobuf import Struct
from dp_sdk.ocf import dp


logger = logging.getLogger(__name__)

async def save_forecast_to_data_platform(
    forecast_values_by_gsp_id: dict[int, pd.DataFrame],
    locations_uuid_and_capacity_by_gsp_id: dict[int, dict[str, str | int]],
    model_tag: str,
    init_time_utc: datetime,
    client: dp.DataPlatformDataServiceStub,
) -> None:
    """Save forecast DataArray to data platform.

    We do the following steps:
    1. get Forecaster
    2. loop over all gsps: get the location object
    3. Forecast the forecast values
    4. Save to the data platform
    5. If gsp_id=0 also save adjusted forecast

    Args:
        forecast_normed_da: DataArray of normalized forecasts for all GSPs
        locations_uuid: location uuids for the data platform
        model_tag: the name of the model to saved to the database
        init_time_utc: Forecast initialization time
        client: Data platform client. If None, a new client will be created.
    """

    # strip out timezone from init_time_utc, this works better with xarray datetime formats
    init_time_utc = init_time_utc.replace(tzinfo=None)

    # 1. get or update or create forecaster version ( this is similar to ml_model before)
    forecaster = await create_forecaster_if_not_exists(client=client, model_tag=model_tag)

    # 2. now loop over all gsps
    logger.debug("Processing forecasts for Data Platform")
    tasks = []
    for gsp_id in forecast_values_by_gsp_id.keys():
        # 3. Format the forecast values
        forecast_values = map_values_df_to_dp_requests(
            forecast_values_by_gsp_id[gsp_id],
            init_time_utc=init_time_utc,
            capacity_watts=locations_uuid_and_capacity_by_gsp_id[gsp_id]["effective_capacity_watts"],
        )
        # 4. Save to data platform
        forecast_request = dp.CreateForecastRequest(
            forecaster=forecaster,
            location_uuid=locations_uuid_and_capacity_by_gsp_id[gsp_id]["location_uuid"],
            energy_source=dp.EnergySource.SOLAR,
            init_time_utc=init_time_utc.replace(tzinfo=UTC),
            values=forecast_values,
        )
        tasks.append(asyncio.create_task(client.create_forecast(forecast_request)))

        # 5. save adjusted if gsp_id=0
        if gsp_id == 0:

            forecast_values = map_values_df_to_dp_requests(
            forecast_values_by_gsp_id[gsp_id],
            init_time_utc=init_time_utc,
            capacity_watts=locations_uuid_and_capacity_by_gsp_id[gsp_id]["effective_capacity_watts"],
            use_adjuster=True,
        )
            forecaster = await create_forecaster_if_not_exists(client=client, model_tag=model_tag+"_adjust")
            forecast_request = dp.CreateForecastRequest(
                forecaster=forecaster,
                location_uuid=locations_uuid_and_capacity_by_gsp_id[gsp_id]["location_uuid"],
                energy_source=dp.EnergySource.SOLAR,
                init_time_utc=init_time_utc.replace(tzinfo=UTC),
                values=forecast_values,
            )
            tasks.append(asyncio.create_task(client.create_forecast(forecast_request)))

    logger.info(f"Saving {len(tasks)} forecasts to Data Platform")
    list_results = await asyncio.gather(*tasks, return_exceptions=True)
    for exc in filter(lambda x: isinstance(x, Exception), list_results):
        raise exc

    logger.info("Saved forecast to Data Platform")


def map_values_df_to_dp_requests(
    forecast_values_df: pd.DataFrame,
    init_time_utc: datetime,
    capacity_watts: int,
    use_adjuster: bool = False
) -> list[dp.CreateForecastRequestForecastValue]:
    """Convert a Dataframe for a single GSP to a list of ForecastValue objects.

    Args:
        forecast_values_df: DataFrame for a single GSP. This has the following columns:
            - target_datetime_utc
            - p10_mw (optional)
            - p50_mw
            - p90_mw (optional)
            - adjuster_mw
        init_time_utc: Forecast initialization time
        capacity_watts: Capacity of the location in watts
        use_adjuster: Whether to apply the adjuster or not
    """

    # create horizon mins
    target_datetime_utc = pd.to_datetime(forecast_values_df['target_datetime_utc'].values)
    horizons_mins = (target_datetime_utc - init_time_utc).total_seconds() / 60
    horizons_mins = horizons_mins.astype(int)

    # get adjuster values
    if use_adjuster:
        forecast_values_df['p50_mw'] = forecast_values_df['p50_mw'] - forecast_values_df['adjust_mw']
        forecast_values_df['p50_mw'] = forecast_values_df['p50_mw'].clip(lower=0)
        for p_col in ['p10_mw', 'p90_mw']:
            if p_col in forecast_values_df.columns:
                forecast_values_df[p_col] = forecast_values_df[p_col] - forecast_values_df['adjust_mw']
                forecast_values_df[p_col] = forecast_values_df[p_col].clip(lower=0)

    # Reduce singular dimensions
    p50s = forecast_values_df['p50_mw'].values.astype(float)
    p50s = p50s * 1*10**6 / float(capacity_watts)
    
    # add p10s and p90s if they exist
    if 'p10_mw' in forecast_values_df.columns:
        p10s = forecast_values_df['p10_mw'].values.astype(float) 
        p10s = p10s * 1*10**6 / float(capacity_watts)
    else:
        p10s = [None]*len(p50s)
    if 'p90_mw' in forecast_values_df.columns:
        p90s = forecast_values_df['p90_mw'].values.astype(float)
        p90s = p90s * 1*10**6 / float(capacity_watts)
    else:
        p90s = [None]*len(p50s)

    forecast_values = []
    for h, p50, p10, p90 in zip(horizons_mins, p50s, p10s, p90s, strict=True):
        
        if h <0:
            # skip negative horizons
            continue

        other_statistics_fractions = {}
        if p10 is not None:
            other_statistics_fractions["p10"] = p10
        if p90 is not None:
            other_statistics_fractions["p90"] = p90

        forecast_values.append(
            dp.CreateForecastRequestForecastValue(
                horizon_mins=h,
                p50_fraction=p50,
                metadata=Struct().from_pydict({}),
                other_statistics_fractions=other_statistics_fractions
            ),
        )

    return forecast_values



async def fetch_dp_gsp_uuid_map(
    client: dp.DataPlatformDataServiceStub,
) -> dict[int, dict[str, int|str]]:
    """Fetch all GSP locations from data platform and map to their uuids."""
    tasks = [
        asyncio.create_task(client.list_locations(
            dp.ListLocationsRequest(
                location_type_filter=loc_type,
                energy_source_filter=dp.EnergySource.SOLAR,
            ),
        ))
        for loc_type in [dp.LocationType.GSP, dp.LocationType.NATION]
    ]
    list_results = await asyncio.gather(*tasks, return_exceptions=True)
    for exc in filter(lambda x: isinstance(x, Exception), list_results):
        raise exc

    locations_df = (
        # Convert and combine the location lists from the responses into a single DataFrame
        pd.DataFrame.from_dict(
            itertools.chain(*[
                r.to_dict(casing=betterproto.Casing.SNAKE, include_default_values=True)["locations"]
                for r in list_results],
            ),
        )
        # Filter the returned locations to those with a gsp_id in the metadata; extract it
        .loc[lambda df: df["metadata"].apply(lambda x: "gsp_id" in x)]
        .assign(gsp_id=lambda df: df["metadata"].apply(lambda x: int(x["gsp_id"]["number_value"])))
        .set_index("gsp_id", drop=False, inplace=False)
    )

    # reduce to the columns we need
    locations_df = locations_df[["location_uuid","effective_capacity_watts"]]

    # change to dict
    return locations_df.to_dict(orient="index")


async def create_forecaster_if_not_exists(
    client: dp.DataPlatformDataServiceStub,
    model_tag: str = "uk-pv-forecast-blend",
) -> dp.Forecaster:
    """Create the current forecaster if it does not exist."""
    name = model_tag.replace("-", "_")
    app_version = version("uk-pv-forecast-blend")

    list_forecasters_request = dp.ListForecastersRequest(
        forecaster_names_filter=[name],
    )
    list_forecasters_response = await client.list_forecasters(list_forecasters_request)

    if len(list_forecasters_response.forecasters) > 0:
        filtered_forecasters = [
            f for f in list_forecasters_response.forecasters if f.forecaster_version == app_version
        ]
        if len(filtered_forecasters) == 1:
            # Forecaster exists, return it
            return filtered_forecasters[0]
        else:
            # Forecaster version does not exist, update it
            update_forecaster_request = dp.UpdateForecasterRequest(
                name=name,
                new_version=app_version,
            )
            update_forecaster_response = await client.update_forecaster(update_forecaster_request)
            return update_forecaster_response.forecaster
    else:
        # Forecaster does not exist, create it
        create_forecaster_request = dp.CreateForecasterRequest(
            name=name,
            version=app_version,
        )
        create_forecaster_response = await client.create_forecaster(create_forecaster_request)
        return create_forecaster_response.forecaster

