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

# %%
# %load_ext autoreload
# %autoreload 2

import os
import re

# Standard imports
import sys
from datetime import datetime

import folium

# geospatial
import geopandas as gpd
import hvplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %%
# Util imports
sys.path.append("../../")  # include parent directory
from src.index_utils import calculate_svi, get_percentile_rank_pl
from src.settings import DATA_DIR

# %% [markdown]
# # Calculate Social Vulnerability Index
# This notebook will calculate SVI using the following method
# 1. Use PCA to compress 18 features into four main vulnerability indicators: vulnerable population, resource deprivation, poverty, and vulnerable assets
# 2. Combine the four indexes into one SVI (equally-weighted)
#
# For this implementation, we will also drop nulls and keep only grids with census data.
#
# > Input file for this notebook is produced in `04_index_calculation/01_create_index_base_date.ipynb`.
#

# %% [markdown]
# ## Set paths and input parameters

# %%
INDEX_OUTPUT_DIR = DATA_DIR / "output/component_2"
INDEX_BASE_DATA_PARQUET = (
    INDEX_OUTPUT_DIR / "index_base_data_dropnocensus_20240527.parquet"
)

AOI_GPKG = DATA_DIR / "admin_bounds/grids_target_muni_wadm_zoomlevel18_20240304.gpkg"

VERSION = pd.to_datetime("today").strftime("%Y%m%d")
OUTPUT_FILENAME = f"cdc_simplified_features_{VERSION}"
OUTPUT_PARQUET = INDEX_OUTPUT_DIR / f"{OUTPUT_FILENAME}.parquet"
OUTPUT_GPKG = INDEX_OUTPUT_DIR / f"{OUTPUT_FILENAME}.gpkg"
OUTPUT_GEOJSON = INDEX_OUTPUT_DIR / f"{OUTPUT_FILENAME}.geojson"
OUTPUT_GCS_BUCKET = "gs://immap-index/cdc/"

# %% [markdown]
# ## Specifiy features for each component

# %%
VULNERABLE_POPULATION_FEATURES = [
    "census_population_density_per_m2",
    "census_population_dependent_percent",
    "census_dwellings_ind_percent",
    "census_dwellings_eth_percent",
    "census_household_to_dwellings_ratio",
]

VULNERABLE_ASSET_FEATURES = [
    "building_area_fraction",
    "cropland_area_fraction",
    "builtup_area_fraction",
]

RESOURCE_DEPRIVATION_FEATURES = [
    "travel_time_to_cities_hr_median",
    "travel_time_to_nearest_healthcare_driving_min",
    "census_dwellings_no_water_service_percent",
    "census_dwellings_no_sewerage_service_percent",
    "census_dwellings_no_garbage_collection_service_percent",
    "census_dwellings_wo_elec_percent",
    "census_dwellings_no_internet_service_percent",
]

POVERTY_FEATURES = ["poverty_index"]

ALL_FEATURES = (
    RESOURCE_DEPRIVATION_FEATURES
    + VULNERABLE_ASSET_FEATURES
    + VULNERABLE_POPULATION_FEATURES
    + POVERTY_FEATURES
)

# %% [markdown]
# ## Load data

# %%
base_data_df = pl.read_parquet(INDEX_BASE_DATA_PARQUET)

# %%
base_data_df

# %%
base_data_df.schema

# %%
base_data_df.shape

# %%
# Check nulls
base_data_df.select(pl.all().is_null().sum())


# %% [markdown]
# ## Process capped features (healthcare)
# For healthcare features (`travel_time_to_nearest_healthcare_driving_min`), we will impute nulls with 60, which is the maximum value recorded.

# %%
def convert_gt60(value):
    if value == ">60":
        return float(60)
    else:
        return float(value)


base_data_df = base_data_df.with_columns(
    pl.col("travel_time_to_nearest_healthcare_driving_min")
    .map_elements(convert_gt60)
    .alias("travel_time_to_nearest_healthcare_driving_min")
)
base_data_df.select("travel_time_to_nearest_healthcare_driving_min")

# %% [markdown]
# ## Drop nulls

# %%
base_data_df = base_data_df.drop_nulls(ALL_FEATURES)
base_data_df.shape

# %%
base_data_df.select(ALL_FEATURES).to_pandas().info()

# %% [markdown]
# ## Create index features for each category
#
# The next code cell defines the `compress_features` function, which takes a dataframe and list of features and outputs the ff.
# 1. result: the resulting index that incorporates all of the input features, taken as the first component from PCA
# 2. pca_df: the output dataframe containing all the output components from PCA
# 3. pca: PCA object which contains metrics for assessing the performance of the PCA calculation
#
# The function also prints out the explained variance ratio, which gives us a measure of how much the output index represents the variation in the input features, from 0-1 (highest). Having a high explained variance ratio is ideal in this case as we want our index to capture as much of the input information as possible.

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compress_features(
    df, features_list, print_info=True, scaler=StandardScaler, n_components=10
):
    """Use PCA to compress list of features into one column

    Args
        df (pl.DataFrame): dataframe that contains features
        features_list (list): list of features to include in calculation
        print_info (bool): If True, print validation info about PCA result

    Returns
        result (pl.Series): compressed feature
        pca_df (pl.DataFrame): resulting components
        pca (sklearn.decomposition.PCA): resulting PCA object
    """
    # Set up scaler
    scaler = scaler()
    scaler.set_output(transform="polars")

    # Get features
    features_df = df.select(features_list).clone()
    features_df = features_df.drop_nulls()

    # Scale features
    scaled_features_df = scaler.fit_transform(features_df)

    # Get n_components as n_features - 1
    n_components = len(features_list) - 1

    # Perform PCA
    pca = PCA(n_components=n_components).set_output(transform="polars")
    pca_df = pca.fit_transform(scaled_features_df)

    result = pca_df["pca1"].alias("result")

    if print_info:
        print(
            f"Explained variance ratio of first component / result feature: {pca.explained_variance_ratio_[0]}"
        )
        print(
            f"Total explained variance across {n_components} components {sum(pca.explained_variance_ratio_)}"
        )
        loadings = pl.DataFrame(
            pca.components_.T * np.sqrt(pca.explained_variance_),
            schema=[f"pca{x}" for x in range(1, pca.n_components_ + 1)],
        )
        loadings = loadings.select(
            [pl.Series(name="feature", values=features_df.columns), pl.all()]
        )
        with pl.Config(fmt_str_lengths=1000, tbl_rows=100):
            display(loadings.sort(pl.col("pca1").abs(), descending=True))

    return result, pca_df, pca


# %% [markdown]
# In the following steps, we calculate indices for vulnerable population, vulnerable assets, and resource deprivation.

# %%
vul_pop_index, vul_pop_pca_df, vul_pop_pca = compress_features(
    base_data_df, VULNERABLE_POPULATION_FEATURES
)

# %%
vul_asset_index, vul_asset_pca_df, vul_asset_pca = compress_features(
    base_data_df, VULNERABLE_ASSET_FEATURES
)

# %%
(
    resource_deprivation_index,
    resource_deprivation_pca_df,
    resource_deprivation_pca,
) = compress_features(base_data_df, RESOURCE_DEPRIVATION_FEATURES)

# %% [markdown]
# ## Compile indices into one dataframe

# %%
index_df = base_data_df.select(
    ["quadkey", "poverty_index", "MPIO_CCNCT", "MPIO_CNMBR_EN"]
)
index_df = index_df.with_columns(
    vul_pop_index.alias("vul_pop_index"),
    vul_asset_index.alias("vul_asset_index"),
    resource_deprivation_index.alias("resource_deprivation_index"),
)
index_df

# %% [markdown]
# ## Rank and calculate SVI by municipality
#
# In this last step, we calculate the social vulnerability index by municipality using the function `calculate_svi`. This does the following steps:
# 1. For each feature in `POSITIVE_FEATURES`, rank each grid against other grids from lowest to highest, then based on its rank, assign it a value of 0-1 using percentile ranking.
# 2. Conversely, for each feature in `NEGATIVE_FEATURES`, rank each grid against other grids from highest to lowest, then based on its rank, assign it a value of 0-1 using percentile ranking
# 3. Take the average of all percentile rank scores to get the final SVI

# %%
municipality_names = list(base_data_df["MPIO_CNMBR_EN"].unique())
municipality_names

# %%
POSITIVE_FEATURES = [
    "poverty_index",
    "vul_pop_index",
    "vul_asset_index",
    "resource_deprivation_index",
]

NEGATIVE_FEATURES = []

# %%
muni_outputs = []
for name in municipality_names:
    muni_df = index_df.filter(pl.col("MPIO_CNMBR_EN") == name)

    muni_svi = calculate_svi(
        muni_df, POSITIVE_FEATURES, NEGATIVE_FEATURES, return_all_features=True
    )
    muni_svi = muni_svi.join(muni_df, on="quadkey")

    # Reorder columns
    first_cols = ["quadkey", "MPIO_CCNCT", "MPIO_CNMBR_EN"]
    reordered_cols = first_cols + [
        col for col in muni_svi.columns if col not in first_cols
    ]
    muni_svi = muni_svi.select(reordered_cols)

    muni_outputs.append(muni_svi)

# Concatenate DataFrames in the list
output_df = pl.concat(muni_outputs)
output_df

# %%
# Check mulls
output_df.select(pl.all().is_null().sum())

# %%
output_df.select(pl.col("svi_mean_p_rank")).null_count()

# %% [markdown]
# ## Plot in map
# In this next section, we plot an interactive chloropleth map of the output SVI by municipality using `geopandas`.

# %%
aoi_gdf = gpd.read_file(AOI_GPKG)
aoi_gdf = aoi_gdf[["quadkey", "geometry"]]

# %%
output_df_pd = output_df.to_pandas()

# %%
output_df_pd.info()

# %%
aoi_with_svi = pd.merge(aoi_gdf, output_df_pd, on="quadkey")
aoi_with_svi = aoi_with_svi.dropna()

# %%
aoi_with_svi.info()

# %%
muni = "SIBUNDOY"
plot_col = "svi_mean_p_rank"

# Get the bounding box of the GeoDataFrame
bbox = aoi_with_svi[
    aoi_with_svi["MPIO_CNMBR_EN"] == muni
].total_bounds  # Returns (minx, miny, maxx, maxy)

# Calculate the center coordinates
center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]

m = folium.Map(
    location=center,
    zoom_start=12,
    control_scale=True,
    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    attr="Google",
    name="Google Satellite",
    overlay=True,
    control=True,
)

aoi_with_svi.explore(plot_col, cmap="turbo", style_kwds={"opacity": 0.4}, m=m)

m

# %% [markdown]
# ## Export Output File

# %%
aoi_with_svi.to_file(OUTPUT_GPKG, index=False)
aoi_with_svi.to_file(OUTPUT_GEOJSON, index=False)

# %%
output_df.write_parquet(OUTPUT_PARQUET)
