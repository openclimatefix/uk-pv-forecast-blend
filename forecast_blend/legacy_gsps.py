
import logging
import yaml
import os
from pydantic import BaseModel, Field
import pandas as pd

logger = logging.getLogger(__name__)


class GSPMergeSource(BaseModel):
    """A single source GSP contributing to a merged/reconstructed target GSP."""

    gsp_id: int
    weight: float = Field(default=1.0, ge=0, description="Multiplier applied to this source's generation")


class GSPMergeConfig(BaseModel):
    """Merge configuration for a single target GSP ID.

    Holds the list of source GSPs (and their weights) whose generation is summed
    to reconstruct the target GSP's generation.
    """

    pvlive_merge_weights: list[GSPMergeSource] = Field(default_factory=list)


def load_gsp_merge_weights(config_path: str = None) -> dict[int, GSPMergeConfig]:
    """
    Load GSP merge weight config from YAML into validated Pydantic models.

    Returns a dict mapping each target GSP ID (int) to a :class:`GSPMergeConfig`.
    Weights default to 1.0 when omitted from the YAML.

    Missing or empty config files are handled gracefully — an empty dict is returned.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "data/gsp_legacy_weights.yaml")

    if not os.path.exists(config_path):
        logger.warning(f"No GSP merge weights config found at {config_path}")
        return {}

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if not raw:
        return {}

    result: dict[int, GSPMergeConfig] = {}
    for target_id, entry in raw.items():
        result[int(target_id)] = GSPMergeConfig.model_validate(entry or {})

    logger.info(f"Loaded GSP merge weights for {len(result)} target GSP IDs")
    return result


def add_legacy_gsp_results(forecast_values_by_gsp_id:dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    """
    Add legacy GSP results to the forecast_values_by_gsp_id dict.

    For each target GSP ID in gsp_merge_weights, sum the generation of the source GSPs
    (weighted by their specified weights) and add a new entry to forecast_values_by_gsp_id.
    """

    legacy_gsps = load_gsp_merge_weights()
    select_cols = ["p10_mw", "p50_mw", "p90_mw","adjust_mw"]

    # lets not let this be changed in place
    forecast_values_by_gsp_id = forecast_values_by_gsp_id.copy()

    for target_gsp_id, gsp_merge_config in legacy_gsps.items():

        if target_gsp_id in forecast_values_by_gsp_id:
            logger.warning(f"Target GSP ID {target_gsp_id} already exists in forecast values; skipping")
            continue


        source_dfs = []
        for gsp_merge_source in gsp_merge_config.pvlive_merge_weights:
            source_id = gsp_merge_source.gsp_id
            weight = gsp_merge_source.weight
            logger.info(f"Adding legacy GSP results from gsp {source_id} "
                        f"for target GSP ID {target_gsp_id}")
            source_df = forecast_values_by_gsp_id.get(source_id)
            if source_df is None:
                logger.warning(f"Source GSP ID {source_id} not found in forecast values; skipping")
                continue
            for col in select_cols:
                if col in source_df.columns:
                    source_df[col] *= weight
            source_dfs.append(source_df)

        if len(source_dfs) == 0:
            logger.warning(f"No source GSPs found for target GSP ID {target_gsp_id}; skipping")
            continue
        
        # Concatenate all weighted sources and sum every numeric column in one pass.
        # generation_mw reflects the applied weights; capacity columns are summed across sources.
        # datetime_gmt is the groupby key (not summed); the rest are numeric aggregates.
        all_cols = ["p10_mw", "p50_mw", "p90_mw", "adjust_mw"] + ["target_datetime_utc"]
        base = (
            pd.concat([df[all_cols] for df in source_dfs], ignore_index=True)
            .groupby("target_datetime_utc", as_index=False)
            .sum()
        )
        # change the index to a column with name target_datetime_utc
        base["target_datetime_utc"] = base.index
        base.reset_index(drop=True, inplace=True)

        logger.debug(
            f"GSP ID {target_gsp_id} reconstruction complete: "
            f"{len(source_dfs)} source(s) combined, "
        )

        forecast_values_by_gsp_id[target_gsp_id] = base

    return forecast_values_by_gsp_id



       