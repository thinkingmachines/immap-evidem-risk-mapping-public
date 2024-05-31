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
#     name: python3
# ---

# %% [markdown] id="ENQwVZWL0D8D"
# # Criteria Development Interactive Notebook

# %% [markdown] id="wMQhS5bu0Iur"
# # Set Up

# %% [markdown] id="2UjOVwajjObu"
# ## Installation and Dependencies

# %% id="ILyBIxCUh2Hr"
# %load_ext autoreload
# %autoreload 2

import json

# Standard imports
import sys
from pathlib import Path
from uuid import uuid4

import folium
import geopandas as gpd
import numpy as np
import pandas as pd

# Geospatial processing packages
from matplotlib import pyplot as plt
from shapely import wkt

# %%
# Util imports
sys.path.append("../../../")  # include parent directory
from src.criteria_scoring import CriteriaScoring
from src.settings import CONFIG_DIR, DATA_DIR, PROJ_CRS

# %% [markdown]
# ## Set up parameters

# %% [markdown] id="2s6SylmH0LWd"
# ## Load Data

# %% colab={"base_uri": "https://localhost:8080/"} id="hfyN6kMY0cxC" outputId="240b1bd2-7991-470b-bcb9-fa233a52e064"
filepath = DATA_DIR / "csv/features.csv"
tile_features = pd.read_csv(filepath)

# %% colab={"base_uri": "https://localhost:8080/"} id="zX9PmTxdiXWX" outputId="e8cec15a-d49b-49e7-f43c-595c39dc4139"
tile_features.shape

# %%
tile_features.columns

# %%
tile_features.drop(["Unnamed: 0"], axis=1, inplace=True)

# %% id="LvaFO9Dj8ob_"
# Check for duplicates
tile_features.drop_duplicates(subset=["geometry"], inplace=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="3GRhy2bp9CD3" outputId="410be4b1-8778-4d1d-c92a-0c59e2c96a2c"
tile_features.shape

# %%
tile_features.describe()

# %%
tile_features = gpd.GeoDataFrame(
    tile_features, geometry=tile_features.geometry.apply(wkt.loads)
)

# %% [markdown] id="RzDBAbcoHiBT"
# # Criteria Testing

# %% [markdown] id="3rHS9N1u-izL"
# ## Define Criteria

# %% id="wUyC2gHVHkxR"
# Adjust scoring implementation here. Update criteria date and version to track changes, do not overwrite config files
criteria_ver = "20230412_v1"
criteria_config = {
    "criteria_name": criteria_ver,
    "date": "2023-02-21",
    "is_weighted": True,
    "scoring_thresholds": {
        # column_name:(low_threshold, high_threshold, direction)
        "dist_nearest_road": (0, 1_000, "low_val_high_score", 10),
        "ntl_mean_rad": (5, 50, "low_val_high_score", 10),
        "otherpois_1km": (0, 20, "low_val_high_score", 10),
        "fw4a_1km": (0, 5, "low_val_high_score", 10),
        "broadband_downspeed": (0, 150_000, "low_val_high_score", 10),
        "mobile_downspeed": (0, 150_000, "low_val_high_score", 10),
        "hazard_mean": (0, 0.12, "high_val_high_score", 10),
        "poverty_incidence": (0, 78.5, "high_val_high_score", 10),
        "pop_total_1km": (1_000, 5_000, "high_val_high_score", 10),
        "insurgency_count": (0, 3, "low_val_high_score", 10),
    },
    "binary_filter": {},
    "direct_score_mapping": {},
    "threshold_filter": {
        # column_name:(threshold, filter direction)
    },
    "tile_score_threshold": 0.5,
    "land_area_ha_threshold": 30,
    "group_by_tile_class": True,
    "threshold_m_coast_tile_counting": 500,
    "min_tiles_near_coast": 10,
    "crs_proj": PROJ_CRS,
}

# %% id="_68_25T3ImN9"
# export parameters as config files; this does not allow overwritting existing files
export_config_json = CONFIG_DIR / (criteria_config["criteria_name"] + ".json")
if export_config_json.is_file():
    print(f"{export_config_json} already exists! Please do not rewrite criteria.")
else:
    with open(export_config_json, "w") as write_file:
        json.dump(criteria_config, write_file, indent=4)

# %% colab={"base_uri": "https://localhost:8080/"} id="mvS6a3SyTtAT" outputId="24c25762-7bfa-45b5-f509-397d99fa81f1"
# load and check criteria config
CONFIG_JSON = CONFIG_DIR / f"{criteria_ver}.json"
with open(CONFIG_JSON, "r") as read_file:
    criteria_config = json.load(read_file)
criteria_config

# %% [markdown] id="tKs1C3jn-aaJ"
# ## Run Criteria on Data

# %% colab={"base_uri": "https://localhost:8080/", "height": 346} id="7flNe6TYTsSd" outputId="d795bdbd-de50-4f1a-826b-290c56c41da4"
# %%time
criteria_scoring = CriteriaScoring(criteria_config)
tile_scores = criteria_scoring.apply_tile_criteria(tile_features)
tile_scores.head(2)

# %% colab={"base_uri": "https://localhost:8080/"} id="BUhGA07qklxx" outputId="7dd98154-2c74-4d12-acf3-91d8ec672b78"
perc_pass = (
    tile_scores[tile_scores.tile_score > 70].shape[0] / tile_scores.shape[0]
) * 100
print(f"{round(perc_pass, 2)}% scored above 70%.")

# %% colab={"base_uri": "https://localhost:8080/"} id="tkZwpSaZwVch" outputId="419b2278-bc5c-443e-aa69-468dbe0a0365"
tile_scores["tile_score"].describe()

# %% colab={"base_uri": "https://localhost:8080/", "height": 283} id="Wmv2al4rwO9l" outputId="dee64609-f727-4152-f7f0-fcb7f04a0b28"
tile_scores["tile_score"].hist()

# %% [markdown] id="fHiHLmaEmi15"
# # Visualization by Province

# %% [markdown] id="HeJkSUd5eqBN"
# ## Map set-up

# %% id="iDdtyhgRlakR"
tile_scores_subset = tile_scores[tile_scores.ADM2_EN == "Aurora"]

# %% id="x8cDX979ImBl"
tile_scores_subset = tile_scores_subset[
    [
        "name",
        "category",
        "class",
        "ADM1_EN",
        "ADM2_EN",
        "ADM3_EN",
        "ADM4_EN",
        "dist_nearest_road",
        "road_type",
        "pop_total_1km",
        "otherpois_1km",
        "fw4a_1km",
        "broadband_downspeed",
        "mobile_downspeed",
        "ntl_mean_rad",
        "insurgency_count",
        "hazard_mean",
        "poverty_incidence",
        "tile_score",
        "geometry",
    ]
]

# %% colab={"base_uri": "https://localhost:8080/"} id="j85czOyh4IwY" outputId="b0ff78db-3440-4c49-be05-712e16fb6cf5"
tile_scores_subset["geometry"] = gpd.GeoSeries.from_wkt(tile_scores_subset["geometry"])

# %% id="WGYMdbfG485h"
tile_scores_subset = gpd.GeoDataFrame(tile_scores_subset, geometry="geometry")

# %% id="pMl2Gm5H_BIn"
gdf_centroid_lat, gdf_centroid_lon = (
    tile_scores_subset.geometry.y.mean(),
    tile_scores_subset.geometry.x.mean(),
)

# %% id="7y0OpFfjepmT"
# replace the basemap on the display
map = folium.Map(
    location=[gdf_centroid_lat, gdf_centroid_lon],
    zoom_start=10,
    control_scale=True,
    tiles="cartodb positron",
)  # this let's us choose a basemap

# %% [markdown] id="aw9R254V-qtI"
# ## Choropleth Map based on Score

# %% colab={"base_uri": "https://localhost:8080/", "height": 711} id="U7KLoGo-mlRh" outputId="04b4dfb6-4d9b-4338-f635-f40d475b20a8"
# %%time
tile_scores_subset.explore(m=map, column="tile_score", cmap="RdYlGn", popup=True)

# %% id="qHL5wS0UpLeD"
# Please use uuid4 for naming the html to gurantee you are not overwriting anything
# id = uuid4()
# map.save(f"{id}.html")

# %% id="xkrFxmvFWmsd"
map.save(f"{DATA_DIR}/html/{criteria_ver}.html")

# %% colab={"base_uri": "https://localhost:8080/"} id="eFTK4e62pisL" outputId="8744071d-cf28-4a10-bae0-a4e130da2d68"
# !~/google-cloud-sdk/bin/gsutil -m cp $DATA_DIR'/html/'$criteria_ver'.html' gs://public-tm8-map-dev

# %% colab={"base_uri": "https://localhost:8080/"} id="u3_Aw7c-pvMh" outputId="fa929a4d-4da4-429b-a866-31ac128b4ff7"
url = f"https://storage.googleapis.com/public-tm8-map-dev/{criteria_ver}.html"
print(f"URL: {url}")
