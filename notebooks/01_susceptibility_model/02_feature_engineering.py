# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Feature Engineering
#
# This notebook runs neighborhood feature generation (lattice averages). Make sure to download the necssary input files and match them to the directories listed below.
#
# ### Input
# - Training data from `01_create_training_data.ipynb`
# - Features table
# - Bingtile x and y coordinates
#
# ### Output
# - training data with the neighborhood aggregates of features

# %% [markdown]
# # Imports and Set up
#
# *DO NOT SKIP THIS SECTION.* This section imports the packages needed to run this notebook and initializes the data file paths.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys

import geopandas as gpd
import pandas as pd
import polars as pl
from loguru import logger
from polars import selectors as cs

# %%
sys.path.append("../../")  # include parent directory
from src import bing_tile_utils
from src.polars_utils import log_condition, log_duplicates

# %%
from src.settings import DATA_DIR

MODEL_DIR = DATA_DIR / "models"
ADMIN_DIR = DATA_DIR / "admin_bounds"
OUTPUT_DIR = DATA_DIR / "output/component_1"

# %%
TRAIN_VERSION = "20240503"
TRAIN_TABLE_FPATH = MODEL_DIR / f"training_data/training_data_{TRAIN_VERSION}.parquet"

BINGTILE_FPATH = ADMIN_DIR / f"grids_landslide_w_xyz_zoomlevel18_20240320.parquet"

# %% [markdown]
# Set the function parameters/arguments that are necessary for the rest of the steps. For the `BING_TILE_ZOOM_LEVEL` make sure that is is the same as the zoom level for our base grids. For the `LATTICE_RADIUS`, this can be adjusted based on the neighborhood size that you want to consider in studying the area.

# %%
# lattice variables
BING_TILE_ZOOM_LEVEL = 18
USE_WEIGHTED_LATTICE = False
LATTICE_RADIUS = 3
CHEBYSHEV_DIST_COL = "chebyshev_dist_col"

OUTPUT_VERSION = pd.to_datetime("today").strftime("%Y%m%d")
OUTPUT_FPATH = (
    MODEL_DIR
    / f"training_data/training_data_w_lattice{LATTICE_RADIUS}_{OUTPUT_VERSION}.parquet"
)

# %% [markdown]
# ## Load Datasets

# %%
bingtiles = pl.read_parquet(BINGTILE_FPATH)
bingtiles.head(3)

# %%
train_table = pl.read_parquet(TRAIN_TABLE_FPATH)

# %%
if "__index_level_0__" in train_table.columns:
    train_table = train_table.drop("__index_level_0__")

# %%
train_table.head()

# %%
train_table = train_table.fill_null("non_landslide")

# %%
train_table["MOV_TYPE"].value_counts()

# %%
features_df = pl.read_parquet(TRAIN_TABLE_FPATH)

# %%
features_df.head()

# %%
features_df.describe()

# %% [markdown]
# ## Remove nulls and impute values

# %%
# rainfall median
features_df = features_df.with_columns(
    pl.when(pl.col("rainfall_mm_median") < 0)
    .then(0)
    .otherwise(pl.col("rainfall_mm_median"))
    .alias("rainfall_mm_median")
)

# %% [markdown]
# # Get lattices

# %% [markdown]
# ## Get weight expressions

# %%
exprs_dict = bing_tile_utils.get_lattice_weight_exprs(
    use_weighted_lattice=USE_WEIGHTED_LATTICE,
    radius=LATTICE_RADIUS,
    group_by_cols="center_quadkey",
    chebyshev_dist_col=CHEBYSHEV_DIST_COL,
)

chebyshev_count_exprs = exprs_dict["chebyshev_count_exprs"]
lattice_weight_exprs = exprs_dict["lattice_weight_exprs"]
lattice_weight_multiplier = exprs_dict["lattice_weight_multiplier"]

# %% [markdown]
# ## Add X and Y components to quadkeys

# %%
landslide_df = train_table.join(bingtiles, on="quadkey", how="inner")

# %%
landslide_df.head(2)

# %%
landslide_df = landslide_df.with_columns(
    pl.col("x").cast(pl.Int64),
    pl.col("y").cast(pl.Int64),
    pl.col("z").cast(pl.Int64),
)

# %%
landslide_df["MOV_TYPE"].value_counts()

# %%
landslide_df.select("*").null_count()

# %%
columns = ["quadkey", "x", "y"]
landslide_quadkeys = (
    landslide_df.pipe(log_duplicates, columns)
    .unique()
    .pipe(log_condition, pl.any_horizontal([pl.col("*").is_null()]))
    .drop_nulls()
)
landslide_quadkeys.head()

# %%
bing_tile_utils.get_bing_cluster_tile_length_m(BING_TILE_ZOOM_LEVEL, LATTICE_RADIUS)

# %%
lattice_df = bing_tile_utils.generate_lattice(
    landslide_quadkeys.select("x", "y"),
    LATTICE_RADIUS,
    zoom_level=BING_TILE_ZOOM_LEVEL,
    include_chebyshev_dist=USE_WEIGHTED_LATTICE,
)
assert not lattice_df.is_duplicated().any()

print(len(lattice_df))
lattice_df.head()

# %%
rename_dict = {"x": "center_x", "y": "center_y", "quadkey": "center_quadkey"}
columns = ["center_quadkey", "lattice_quadkey"]
if USE_WEIGHTED_LATTICE:
    columns += bing_tile_utils.CHEBYSHEV_DIST_COLS

lattice_df = lattice_df.join(
    landslide_quadkeys.rename(rename_dict), on=["center_x", "center_y"], how="left"
).select(columns)
print(len(lattice_df))
lattice_df.head()

# %% [markdown]
# # Calculate lattice aggregates

# %%
print(features_df.columns)

# %% [markdown]
# ## Features that need the average

# %%
aggregated_cols = [
    "elevation_median",
    "slope_median",
    "aspect_median",
    "hillshade_median",
    "rainfall_mm_median",
    "sand_5-15cm_mean",
    "sand_100-200cm_mean",
    "silt_5-15cm_mean",
    "silt_100-200cm_mean",
    "clay_5-15cm_mean",
    "clay_100-200cm_mean",
    "distance_m_roads",
    "distance_m_rivers",
]

# %%
agg_expr = [(pl.col(col) * lattice_weight_multiplier).mean() for col in aggregated_cols]

aggregated_metrics = (
    features_df.select(["quadkey"] + aggregated_cols)
    .rename({"quadkey": "lattice_quadkey"})
    .join(
        lattice_df,
        on="lattice_quadkey",
        how="inner",
        validate="1:m",
    )
    .drop("lattice_quadkey")
    .with_columns(chebyshev_count_exprs)
    .with_columns(lattice_weight_exprs)
    .group_by("center_quadkey")
    .agg(agg_expr)
    .rename({"center_quadkey": "quadkey"})
    .sort(by="quadkey")
)
print(len(aggregated_metrics))
aggregated_metrics.head()

# %%
aggregated_metrics = aggregated_metrics.with_columns(
    pl.all().name.suffix(f"_lattice_{LATTICE_RADIUS}")
)

# %%
aggregated_metrics = aggregated_metrics.select(f"^.*_lattice_{LATTICE_RADIUS}$").rename(
    {f"quadkey_lattice_{LATTICE_RADIUS}": "quadkey"}
)

# %% [markdown]
# ## Aggregation for lithology type

# %%
lithology_metrics = (
    features_df.select(["quadkey"] + ["lithology_type"])
    .rename({"quadkey": "lattice_quadkey"})
    .join(
        lattice_df,
        on="lattice_quadkey",
        how="inner",
        validate="1:m",
    )
    .drop("lattice_quadkey")
    .with_columns(chebyshev_count_exprs)
    .with_columns(lattice_weight_exprs)
    .group_by("center_quadkey")
    .agg(pl.col("lithology_type").mode())
    .rename(
        {
            "center_quadkey": "quadkey",
            "lithology_type": f"lithology_type_lattice_{LATTICE_RADIUS}",
        }
    )
    .sort(by="quadkey")
)
print(len(lithology_metrics))
lithology_metrics.head()

# %%
# get first element from mode list
lithology_metrics = (
    lithology_metrics.with_row_index()
    .with_columns(
        pl.col(f"lithology_type_lattice_{LATTICE_RADIUS}")
        .explode()
        .gather(0)
        .over(pl.col("index"))
    )
    .drop("index")
)
lithology_metrics.head()

# %% [markdown]
# ## Aggregation for soilclass

# %%
soilclass_metrics = (
    features_df.select(["quadkey"] + ["soil_class"])
    .rename({"quadkey": "lattice_quadkey"})
    .join(
        lattice_df,
        on="lattice_quadkey",
        how="inner",
        validate="1:m",
    )
    .drop("lattice_quadkey")
    .with_columns(chebyshev_count_exprs)
    .with_columns(lattice_weight_exprs)
    .group_by("center_quadkey")
    .agg(pl.col("soil_class").mode())
    .rename(
        {
            "center_quadkey": "quadkey",
            "soil_class": f"soil_class_lattice_{LATTICE_RADIUS}",
        }
    )
    .sort(by="quadkey")
)
print(len(soilclass_metrics))
soilclass_metrics.head()

# %%
# get first element from mode list
soilclass_metrics = (
    soilclass_metrics.with_row_index()
    .with_columns(
        pl.col(f"soil_class_lattice_{LATTICE_RADIUS}")
        .explode()
        .gather(0)
        .over(pl.col("index"))
    )
    .drop("index")
)
soilclass_metrics.head()

# %% [markdown]
# # Add lattice features to train df

# %%
lattice_features = (
    features_df.join(lithology_metrics, on="quadkey")
    .join(soilclass_metrics, on="quadkey")
    .join(aggregated_metrics, on="quadkey")
)

# %%
lattice_features.head()

# %%
lattice_features.write_parquet(OUTPUT_FPATH)

# %%
lattice_features["MOV_TYPE"].value_counts()
