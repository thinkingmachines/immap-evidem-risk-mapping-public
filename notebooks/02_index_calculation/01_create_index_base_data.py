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
# # Create index base data
#
# This notebook creates the base table from which the index will be calculated
# - Selecting area (if needed)
# - Filtering grids with nulls
# - Feature selection
#
# ## Input
# - Aligned dataset with all processed features
#
# > Input file for this notebook is produced in `02_dataset_alignment/01_align_index_datasets.ipynb`. The file is provided in the project Google Drive folder.
#
# ## Output
# - Aligned dataset with selected features and areas, to be used in index calculation
#

# %% [markdown]
# ## Imports and Set-up

# %%
# %load_ext autoreload
# %autoreload 2

import re

# Standard imports
import sys

import pandas as pd

# geospatial
import polars as pl

# %%
# Util imports
sys.path.append("../../")  # include parent directory
from src.index_utils import add_flag_to_filepath
from src.settings import DATA_DIR

# %% [markdown]
# ## Set paths and input paramters

# %%
ALIGNED_DIR = DATA_DIR / "aligned"
ALIGNED_DATA_PARQUET = (
    ALIGNED_DIR / "parquets/aligned_index_datasets_consolidated_20240423.parquet"
)

VERSION = pd.to_datetime("today").strftime("%Y%m%d")
INDEX_OUTPUT_DIR = DATA_DIR / "output/component_2"

# if True, drop data with nulls in census data
DROP_NO_CENSUS = True
CHECK_CENSUS_COLS = ["census_population_total_count"]

# %% [markdown]
# ## Create output filepath based on options
#

# %%
BASE_FPATH = INDEX_OUTPUT_DIR / f"index_base_data.parquet"
OUT_FPATH = BASE_FPATH

if DROP_NO_CENSUS:
    OUT_FPATH = add_flag_to_filepath(OUT_FPATH, "dropnocensus")

# Add version
OUT_FPATH = add_flag_to_filepath(OUT_FPATH, VERSION)

OUT_FPATH

# %% [markdown]
# ## Load data

# %%
aligned_df = pl.read_parquet(ALIGNED_DATA_PARQUET)
aligned_df

# %%
aligned_df.schema

# %%
aligned_df.shape

# %%
process_df = aligned_df.clone()

# %% [markdown]
# ## Drop null columns

# %%
if DROP_NO_CENSUS:
    print(f"Shape before dropping nulls in {CHECK_CENSUS_COLS}: {process_df.shape}")
    process_df = process_df.drop_nulls(CHECK_CENSUS_COLS)
    print(f"Shape after dropping nulls in {CHECK_CENSUS_COLS}: {process_df.shape}")
else:
    print("Nulls in data retained")

# %% [markdown]
# ## Inspect correlation between features
#
# This part will help us identify features that are highly correlated, which can indicate redundant quantities

# %%
# Get the data types of all columns
dtypes = process_df.schema

# Filter out columns with data type String
non_string_columns = [
    col for col, dtype in dtypes.items() if dtype != pl.datatypes.String
]

# Select only non-string columns
features_df = process_df.select(non_string_columns)

# Drop the municipality data
drop_muni_cols = [
    "MPIO_NAREA",
    "MPIO_NANO",
    "SHAPE_AREA",
    "SHAPE_LEN",
]  # List of columns to drop
features_df = features_df.drop(drop_muni_cols)

features_df

# %%
## Get correlation table
## TODO: add as util
long = features_df.select(
    [pl.corr(pl.all(), pl.col(c)).name.suffix("|" + c) for c in features_df.columns]
).melt()
table = long.select(
    pl.col("variable").map_elements(lambda x: x.split("|")[0]).alias("var1"),
    pl.col("variable").map_elements(lambda x: x.split("|")[1]).alias("var2"),
    pl.col("value").alias("corr"),
).sort(by="corr")

with pl.Config(fmt_str_lengths=2000):
    display(table)
    print(table.glimpse())

# %% [markdown]
# ## Select features
# - Get id cols (quadkey, MPIO_CNMBR_EN, MPIO_CCNCT)
# - Get the median values for raster zonal stats
# - Get percent incidence for census data
# - Get area fraction columns for buildings and landcover
# - Get specific columns from census data: population density, total population, and total number of dwellings and households
# - Drop population_density and population_density UNadj (from WorldPop)

# %%
id_cols = ["quadkey", "MPIO_CNMBR_EN", "MPIO_CCNCT"]
manual_cols = [
    "census_dwellings_count",
    "census_household_count",
    "census_population_total_count",
    "census_population_density_per_m2",
    "census_household_to_dwellings_ratio",
    "poverty_index",
    "rwi_scaled",
    "distance_to_nearest_healthcare_m",
    "travel_time_to_nearest_healthcare_driving_min",
    "travel_time_to_nearest_healthcare_walking_min",
]

# drop_cols = ["population_density_UNadj_median", "population_density_median"]
drop_cols = []

# %%
median_pattern = re.compile(".*_median$")
median_cols = [col for col in process_df.columns if median_pattern.match(col)]
median_cols

# %%
census_percent_pattern = re.compile("^census_[^_]+(?:_[^_]+)*_percent$")
census_percent_cols = [
    col for col in process_df.columns if census_percent_pattern.match(col)
]
census_percent_cols

# %%
area_fraction_pattern = re.compile(".*_area_fraction$")
area_fraction_cols = [
    col for col in process_df.columns if area_fraction_pattern.match(col)
]
area_fraction_cols

# %%
all_features = manual_cols
all_features += median_cols + census_percent_cols + area_fraction_cols
all_features = [x for x in all_features if x not in drop_cols]
all_features = list(dict.fromkeys(all_features))  # dedup while keeping order
all_features

# %%
selected_features_df = process_df.select(
    [col for col in process_df.columns if col in id_cols + all_features]
)
selected_features_df

# %%
selected_features_df.schema

# %%
selected_features_df.shape

# %%
# check for nulls in final output
nulls_check_df = selected_features_df.select(pl.all().is_null().sum())
nulls_check_df

# %% [markdown]
# ## Output table

# %%
selected_features_df.write_parquet(OUT_FPATH)
