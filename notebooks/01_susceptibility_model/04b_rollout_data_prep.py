# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: immap-evidem
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Roll-out data preparation
#
# This notebook runs feature engineering on the target municipalites needed for model roll-out
#
# ### Input
# - Features for the priority municipalities
#
# ### Output
# - Feature table for roll-out

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Imports and Set-up

# %%
import sys

import geopandas as gpd
import pandas as pd
import polars as pl
from loguru import logger

# %%
sys.path.append("../../")  # include parent directory
from src import bing_tile_utils
from src.polars_utils import log_condition, log_duplicates
from src.settings import DATA_DIR

# %%
MODEL_DIR = DATA_DIR / "models"
VECTOR_DIR = DATA_DIR / "vectors"
OUTPUT_DIR = DATA_DIR / "output/component_1"

# %%
FEATURES_FPATH = (
    DATA_DIR / "aligned/parquets/aligned_dataset_consolidated_20240503.parquet"
)
BINGTILE_FPATH = (
    DATA_DIR / "admin_bounds/grids_landslide_w_xyz_zoomlevel18_20240320.parquet"
)

ADM_GRIDS = DATA_DIR / "admin_bounds/grids_target_muni_wadm_zoomlevel18_20240304.gpkg"

# %%
# lattice variables
BING_TILE_ZOOM_LEVEL = 18
USE_WEIGHTED_LATTICE = False
LATTICE_RADIUS = 3
CHEBYSHEV_DIST_COL = "chebyshev_dist_col"

OUTPUT_VERSION = pd.to_datetime("today").strftime("%Y%m%d")
OUTPUT_FPATH = (
    MODEL_DIR
    / f"rollout_data/rollout_data_w_lattice{LATTICE_RADIUS}_{OUTPUT_VERSION}.parquet"
)

# %% [markdown]
# ## Import Data
#

# %%
bingtiles = pl.read_parquet(BINGTILE_FPATH)

# %%
bingtiles.head(), bingtiles.shape

# %%
adm_grids = gpd.read_file(ADM_GRIDS)
adm_grids.drop(columns=["geometry"], inplace=True)
adm_grids = pl.from_pandas(adm_grids)

# %%
adm_grids.head(), adm_grids.shape

# %%
features_df = pl.read_parquet(FEATURES_FPATH)

# %%
features_df.head(), features_df.shape

# %%
priority_cities_quadkeys = list(adm_grids["quadkey"])

# %%
# subset features to priority municipalities
priority_city_feat = features_df.filter(
    pl.col("quadkey").is_in(priority_cities_quadkeys)
)

# %%
priority_city_feat.head(), priority_city_feat.shape

# %%
# Get bingtile x,y for target municipalities
priority_city_feat = priority_city_feat.join(bingtiles, on="quadkey")

# %%
priority_city_feat.select(pl.all().is_null().sum())

# %% [markdown]
# ## Impute missing values

# %%
priority_city_feat = priority_city_feat.drop(
    ["ndvi2023_min", "ndvi2023_max", "ndvi2023_count", "ndvi2023_median"]
)

# %%
priority_city_feat = priority_city_feat.with_columns(
    pl.col(
        [
            "rainfall_mm_min",
            "rainfall_mm_max",
            "rainfall_mm_count",
            "rainfall_mm_median",
        ]
    ).fill_null(strategy="zero")
)

# %%
priority_city_feat = priority_city_feat.with_columns(
    pl.col(["distance_m_roads", "distance_m_rivers"]).fill_null(-99)
)

# %%
# rainfall median
priority_city_feat = priority_city_feat.with_columns(
    pl.when(pl.col("rainfall_mm_median") < 0)
    .then(pl.lit(0))
    .otherwise(pl.col("rainfall_mm_median"))
    .alias("rainfall_mm_median")
)

# %%
priority_city_feat.with_columns(pl.col("rainfall_mm_median") < 0)

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

# %%
columns = ["quadkey"]
landslide_quadkeys = (
    priority_city_feat.pipe(log_duplicates, columns)
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
# # Calculate lattice features

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

# %%
lattice_features = priority_city_feat.join(lithology_metrics, on="quadkey").join(
    aggregated_metrics, on="quadkey"
)

# %%
lattice_features.head(), lattice_features.shape

# %%
lattice_features = lattice_features.drop("__index_level_0__")

# %% [markdown]
# ## Impute missing values for lattice features

# %%
lattice_features.select(pl.all().is_null().sum())

# %%
lattice_features.write_parquet(OUTPUT_FPATH)
