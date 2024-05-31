# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Combine Component 1 and Component 2 Fields
#
# This notebook combines Component 1 (Mass movement susceptibility) and Component 2 (Social Vulnerability Index) measuring the overall risk of an area
#
# ## Input
# - Component 1: Mass movement susceptibility
#   - Landslide susceptibility
#   - Flows susceptibility
# - Component 2: Social Vulnerability Index
#
# > Input files are produced in `03_susceptibility_model/` and `04_index_calculation/`, respectively.
#
# ## Process
# - To calculate the overall risk, we simply need to multiply the value of susceptibility and social vulnerability index
#
# ## Output
# - Combined dataset that includes the selected features and areas with the overall risk
#

# %%
# %load_ext autoreload
# %autoreload 2

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl
from loguru import logger

# %%
# Util imports
sys.path.append("../../")  # include parent directory
from src.settings import DATA_DIR

# %%
COMPONENT_1_DIR = DATA_DIR / "output/component_1"
TARGET_LANDSLIDE_FILE = COMPONENT_1_DIR / "landslide_susceptibility.gpkg"
TARGET_FLOWS_FILE = COMPONENT_1_DIR / "flows_susceptibility.gpkg"

GCS_ROLLOUT_DIR = "gs://immap-susceptibility-model"

COMPONENT_2_DIR = DATA_DIR / "output/component_2"
TARGET_COMPONENT_2_FILE = COMPONENT_2_DIR / "cdc_simplified_features_20240528.gpkg"

VERSION = datetime.now().strftime("%Y%m%d")
OUTPUT_DIR = DATA_DIR / "output/risk_index"
OUTPUT_FPATH = OUTPUT_DIR / f"risk_index_{VERSION}.gpkg"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_GCS_BUCKET = "gs://immap-risk-index"

# %% [markdown]
# ## Verify target files

# %%
for target_file in [TARGET_FLOWS_FILE, TARGET_LANDSLIDE_FILE, TARGET_COMPONENT_2_FILE]:
    if not target_file.is_file():
        logger.info(f"File {target_file} does not exist!")
    else:
        logger.info(f"File {target_file} exists!")

# %% [markdown]
# ## Load and inspect components

# %%
component_1_landslide_gdf = gpd.read_file(TARGET_LANDSLIDE_FILE)
component_1_landslide_gdf.info()

# %%
component_1_flows_gdf = gpd.read_file(TARGET_FLOWS_FILE)
component_1_flows_gdf.info()

# %%
# component_1_gdf = gpd.read_file(TARGET_COMPONENT_1_FILE)
# component_1_gdf.info()

# %%
component_2_gdf = gpd.read_file(TARGET_COMPONENT_2_FILE)
component_2_gdf.info()

# %% [markdown]
# ## Select fields for final output

# %%
component_1_landslide_gdf.columns

# %%
component_1_landslide_fields = [
    "quadkey",
    "MPIO_CCNCT",
    "MPIO_CNMBR",
    "MPIO_CNMBR_EN",
    "DPTO_CNMBR",
    "DPTO_CNMBR_EN",
    "LANDSLIDE_SUSC",
    "geometry",
]
process_component_1_landslide_gdf = component_1_landslide_gdf.loc[
    :, component_1_landslide_fields
]
process_component_1_landslide_gdf.head(3)

# %%
component_1_flows_fields = [
    "quadkey",
    "FLOWS_SUSC",
]
process_component_1_flows_gdf = component_1_flows_gdf.loc[:, component_1_flows_fields]
process_component_1_flows_gdf.head(3)

# %%
# ## Rename the probability fields based on the model type
# process_component_1_landslide_gdf = process_component_1_landslide_gdf.rename(
#     {
#         "pred_proba_1": "landslide_pred_probability",
#     },
#     axis=1,
# )
# process_component_1_flows_gdf = process_component_1_flows_gdf.rename(
#     {"pred_proba_2": "flows_pred_probability"}, axis=1
# )
# display(process_component_1_landslide_gdf.info())
# display(process_component_1_flows_gdf.info())

# %%
component_2_gdf.columns

# %%
component_2_fields = [
    "quadkey",
    "svi_mean_p_rank",
    # 'svi_extreme_count',
    "poverty_index_p_rank",
    "vul_pop_index_p_rank",
    "vul_asset_index_p_rank",
    "resource_deprivation_index_p_rank",
    # 'poverty_index',
    # 'vul_pop_index',
    # 'vul_asset_index',
    # 'resource_deprivation_index'
]

process_component_2_gdf = component_2_gdf.loc[:, component_2_fields]
process_component_2_gdf.head(10)

# %%
combined_gdf = pd.merge(
    process_component_1_landslide_gdf,
    process_component_1_flows_gdf,
    on="quadkey",
    how="left",
)
combined_gdf = pd.merge(combined_gdf, process_component_2_gdf, on="quadkey", how="left")
combined_gdf

# %% [markdown]
# ## Combine the two components
# Method: Multiply component 1 and component 2 to get the combined risk index. We calculate combined risk index for landslides and flows separately.

# %%
combined_gdf["risk_index_landslide"] = (
    combined_gdf["LANDSLIDE_SUSC"] * combined_gdf["svi_mean_p_rank"]
)
combined_gdf["risk_index_flows"] = (
    combined_gdf["FLOWS_SUSC"] * combined_gdf["svi_mean_p_rank"]
)
reordered_columns = [col for col in combined_gdf.columns if col != "geometry"] + [
    "geometry"
]
combined_gdf = combined_gdf.loc[:, reordered_columns]
combined_gdf.info()

# %% [markdown]
# ## Export to file

# %%
combined_gdf.to_file(OUTPUT_FPATH, index=False)
